#!/usr/bin/env python3
"""Read-only H7 mirror: 48-channel CRC-protected JustFloat -> closed-chain pose.

Only kinematics are evaluated; this program never sends serial commands, runs a
policy, integrates physics or estimates world translation. H7's public feedback
already includes driver calibration and wheel reduction. The additional model
mapping is read from the delivered H7 machine_config.c, never inferred here.
"""
from __future__ import annotations

import argparse
import binascii
from contextlib import nullcontext
import json
from pathlib import Path
import re
import threading
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "resources/robots/chuanliantui_new_1/multimode_model3023/chuanliantui.xml"
MACHINE_CONFIG = ROOT / "deployment/H7_RL/imcalib/config/machine_config.c"
TAIL, FRAME_BYTES = bytes.fromhex("0000807f"), 196
STALE_S, CLOSURE_TOL_M = .2, .001
DRIVES = (("lf0", "lf00"), ("rf0", "rf00"))
PASSIVE = (("lf1", "lf01", "lf02", "lf03"), ("rf1", "rf01", "rf02", "rf03"))
WHEELS = ("lfwheel", "rfwheel")
PIN_PAIRS = tuple(tuple((f"{side}_rear_pin{i}", f"{side}_front_pin{i}")
                        for i in (1, 2)) for side in ("left", "right"))


def read_u32(values: np.ndarray) -> int:
    if not np.isfinite(values).all() or any(v != int(v) or not 0 <= v <= 65535 for v in values):
        raise ValueError("invalid uint32 halves")
    return int(values[0]) | (int(values[1]) << 16)


def decode_frame(raw: bytes) -> np.ndarray:
    if len(raw) != FRAME_BYTES or raw[-4:] != TAIL:
        raise ValueError("frame length/tail")
    values = np.frombuffer(raw[:192], dtype="<f4").copy()
    if values[3] != 7101. or values[4] != 1.:
        raise ValueError("not mirror v1")
    if values[47] != binascii.crc_hqx(raw[:188], 0xffff):
        raise ValueError("CRC mismatch")
    if values[5] != 0.:
        raise ValueError("unsupported machine (only big wheelleg)")
    read_u32(values[6:8])
    read_u32(values[8:10])
    mask = values[29]
    if not np.isfinite(mask) or mask != int(mask) or not 0 <= mask <= 127:
        raise ValueError("invalid source mask")
    return values


class FrameParser:
    """Scan every candidate tail; a tail inside payload must not truncate a frame."""

    def __init__(self):
        self.buffer = bytearray()
        self.accepted = self.rejected = 0
        self.last_error = ""

    def feed(self, chunk: bytes) -> list[np.ndarray]:
        self.buffer.extend(chunk)
        frames, search = [], 0
        while True:
            end = self.buffer.find(TAIL, search)
            if end < 0:
                break
            stop = end + 4
            if stop >= FRAME_BYTES:
                try:
                    frame = decode_frame(bytes(self.buffer[stop - FRAME_BYTES:stop]))
                except ValueError as exc:
                    self.rejected += 1
                    self.last_error = str(exc)
                else:
                    frames.append(frame)
                    self.accepted += 1
                    del self.buffer[:stop]
                    search = 0
                    continue
            search = end + 1
        # Retain a complete possible prefix of the next frame, even after noise.
        if len(self.buffer) >= FRAME_BYTES:
            del self.buffer[:-(FRAME_BYTES - 1)]
        return frames


class FrameClock:
    """Submitted-frame sequence diagnostics, independent of Jetson's wall clock."""

    def __init__(self):
        self.sequence = self.tick = None
        self.missing = self.duplicates = self.restarts = self.out_of_order = 0

    def accept(self, frame: np.ndarray) -> bool:
        sequence, tick = read_u32(frame[6:8]), read_u32(frame[8:10])
        if self.sequence is not None:
            delta = (sequence - self.sequence) & 0xffffffff
            elapsed = (tick - self.tick) & 0xffffffff
            # A reboot after >24.8 days can look like a forward uint32 time
            # interval. Both counters rolling back still identifies a reset;
            # natural wraps have a small positive sequence delta instead.
            counters_back = tick < self.tick and sequence <= self.sequence and (delta == 0 or delta >= 0x80000000)
            if elapsed >= 0x80000000 or counters_back:
                self.restarts += 1
            elif delta == 0:
                self.duplicates += 1
                return False
            elif delta >= 0x80000000:
                self.out_of_order += 1
                return False
            else:
                self.missing += delta - 1
        self.sequence, self.tick = sequence, tick
        return True


def load_mapping(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read only the existing big-machine RL mapping; do not repeat driver signs."""
    text = path.read_text()
    block = text.split("[MACHINE_ID_BIG_WHEELLEG] = {", 1)[1]
    block = block.split("[MACHINE_ID_SMALL_WHEELLEG]", 1)[0].split(".rl = {", 1)[1]
    arrays = []
    for name, count in (("motor_sign", 6), ("motor_zero", 4)):
        body = re.search(r"\." + name + r"\s*=\s*\{([^}]+)\}", block).group(1)
        values = np.asarray([float(v.strip().rstrip("fF")) for v in body.split(",") if v.strip()])
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f"cannot read {name} from delivered machine_config.c")
        arrays.append(values)
    return arrays[0], arrays[1]


class MirrorPose:
    def __init__(self, model_path: Path = MODEL, machine_config: Path = MACHINE_CONFIG):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.sign, self.zero = load_mapping(machine_config)
        self.drive_q = [np.array([self.model.joint(n).qposadr[0] for n in names]) for names in DRIVES]
        self.passive_q = [np.array([self.model.joint(n).qposadr[0] for n in names]) for names in PASSIVE]
        self.wheel_q = [self.model.joint(n).qposadr[0] for n in WHEELS]
        self.spring_q = [np.array([self.model.joint(f"{side}_{part}_joint").qposadr[0]
                                  for part in ("spring1", "spring2")]) for side in ("left", "right")]
        self.status = ["NO DATA"] * 5  # IMU, left/right leg, left/right wheel
        self.residual_m = [None, None]
        self.last_received = None
        self.seen = [False] * 5
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self.original_rgba = self.model.geom_rgba.copy()

    def _pins(self, side: int) -> np.ndarray:
        return np.concatenate([self.data.site(a).xpos - self.data.site(b).xpos
                               for a, b in PIN_PAIRS[side]])

    def _solve_leg(self, side: int, angles: np.ndarray) -> bool:
        drive, passive, spring = self.drive_q[side], self.passive_q[side], self.spring_q[side]
        indices = np.concatenate((drive, passive, spring))
        previous = self.data.qpos[indices].copy()
        initial = self.data.qpos[passive].copy()
        self.data.qpos[drive] = angles

        def residual(values):
            self.data.qpos[passive] = values
            mujoco.mj_kinematics(self.model, self.data)
            return self._pins(side)

        try:
            solved = least_squares(residual, initial, max_nfev=60,
                                   xtol=1e-10, ftol=1e-10, gtol=1e-10)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            self.data.qpos[indices] = previous
            mujoco.mj_kinematics(self.model, self.data)
            self.residual_m[side] = None
            self.status[side + 1] = "INVALID CLOSURE"
            return False
        # Equivalent angles nearest to the last pose keep accumulated turns continuous.
        result = initial + (solved.x - initial + np.pi) % (2 * np.pi) - np.pi
        residual(result)
        name = ("left", "right")[side]
        parent = DRIVES[side][0]
        delta = self.data.body(parent).xmat.reshape(3, 3).T @ (
            self.data.site(f"{name}_gas_spring_lower").xpos
            - self.data.site(f"{name}_gas_spring_upper").xpos)
        quat = self.model.body(f"{name}_spring1").quat
        self.data.qpos[spring[0]] = np.arctan2(delta[0], delta[2]) - 2 * np.arctan2(quat[2], quat[0])
        self.data.qpos[spring[1]] = np.hypot(delta[0], delta[2]) - self.model.body(f"{name}_spring2").pos[2]
        mujoco.mj_kinematics(self.model, self.data)
        pin_error = max(np.linalg.norm(v) for v in self._pins(side).reshape(2, 3))
        spring_error = np.linalg.norm(self.data.site(f"{name}_spring_tip").xpos
                                      - self.data.site(f"{name}_gas_spring_lower").xpos)
        error = float(max(pin_error, spring_error))
        self.residual_m[side] = error
        # A different assembly branch may close the pins but exceed a hard stop.
        within_range = True
        for name, angle in zip(PASSIVE[side], result):
            joint = self.model.joint(name)
            if joint.limited[0] and not joint.range[0] - 1e-8 <= angle <= joint.range[1] + 1e-8:
                within_range = False
                break
        if not np.isfinite(error) or error > CLOSURE_TOL_M or not within_range:
            self.data.qpos[indices] = previous
            mujoco.mj_kinematics(self.model, self.data)
            self.status[side + 1] = "INVALID CLOSURE" if within_range else "INVALID JOINT RANGE"
            return False
        self.status[side + 1] = "OK"
        self.seen[side + 1] = True
        return True

    def apply(self, frame: np.ndarray, received_at: float):
        self.last_received = received_at
        mask = int(frame[29])
        # Fresh UART packets can still contain an old software publication.
        # These ages are computed on H7; no cross-host clock subtraction is used.
        imu_recent = bool(np.isfinite(frame[30]) and 0 <= frame[30] < STALE_S * 1000)
        motor_recent = bool(np.isfinite(frame[31]) and 0 <= frame[31] < STALE_S * 1000)
        quat = frame[22:26].astype(np.float64)
        if (imu_recent and mask & 1 and np.isfinite(quat).all()
                and abs(np.linalg.norm(quat) - 1.) <= 1e-3):
            self.data.qpos[3:7] = quat / np.linalg.norm(quat)
            self.status[0], self.seen[0] = "OK", True
        else:
            self.status[0] = "INVALID IMU" if imu_recent else "STALE IMU SOURCE"
        for side in (0, 1):
            segment = slice(2 * side, 2 * side + 2)
            angles = frame[10:14][segment]
            required = 0x6 << (2 * side)
            if motor_recent and mask & required == required and np.isfinite(angles).all():
                model_angles = self.sign[:4][segment] * (angles - self.zero[segment])
                self._solve_leg(side, model_angles)
            else:
                self.status[side + 1] = "INVALID FEEDBACK" if motor_recent else "STALE MOTOR SOURCE"
            angle = frame[18 + side]
            if motor_recent and mask & (1 << (5 + side)) and np.isfinite(angle):
                self.data.qpos[self.wheel_q[side]] = self.sign[4 + side] * float(angle)
                self.status[3 + side], self.seen[3 + side] = "OK", True
            else:
                self.status[3 + side] = "INVALID WHEEL" if motor_recent else "STALE MOTOR SOURCE"
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def stale(self, now: float) -> bool:
        return self.last_received is None or now - self.last_received >= STALE_S

    def labels(self, now: float) -> list[str]:
        return ["STALE" if seen else "NO DATA" for seen in self.seen] if self.stale(now) else self.status.copy()

    def paint_status(self, now: float):
        labels = self.labels(now)
        self.model.geom_rgba[:] = self.original_rgba
        for geom in range(self.model.ngeom):
            name = self.model.body(self.model.geom_bodyid[geom]).name
            source = (0 if name == "base_link" else
                      3 if name == "lfwheel" else 4 if name == "rfwheel" else
                      1 if name.startswith(("lf", "left_")) else
                      2 if name.startswith(("rf", "right_")) else None)
            if source is not None and labels[source] != "OK":
                self.model.geom_rgba[geom] = (.85, .20, .16, 1.)


class SerialReceiver(threading.Thread):
    """Own the serial reader; the viewer consumes only the most recent valid frame."""

    def __init__(self, port: str, baud: int):
        super().__init__(daemon=True)
        self.port, self.baud = port, baud
        self.parser, self.clock = FrameParser(), FrameClock()
        self.stop_event, self.lock = threading.Event(), threading.Lock()
        self.latest = None
        self.state = "CONNECTING"

    def run(self):
        import serial
        while not self.stop_event.is_set():
            try:
                # No write(), no command protocol. DTR/RTS are not used for board reset.
                with serial.Serial(port=None, baudrate=self.baud, timeout=.03) as connection:
                    connection.dtr = connection.rts = False
                    connection.port = self.port
                    connection.open()
                    self.parser.buffer.clear()
                    self.state = "CONNECTED"
                    while not self.stop_event.is_set():
                        chunk = connection.read(min(connection.in_waiting or 1, 4096))
                        for frame in self.parser.feed(chunk):
                            if self.clock.accept(frame):
                                with self.lock:
                                    self.latest = (frame, time.monotonic())
            except (serial.SerialException, OSError) as exc:
                self.state = f"DISCONNECTED: {exc}"
                self.stop_event.wait(.5)

    def snapshot(self):
        with self.lock:
            return self.latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="H7 USB-to-UART device, preferably /dev/serial/by-id/...")
    parser.add_argument("--baud", type=int, default=1152000)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--machine-config", type=Path, default=MACHINE_CONFIG)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0., help="0 keeps the display open until closed")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.seconds < 0 or not np.isfinite(args.seconds):
        parser.error("--seconds must be finite and nonnegative")
    if args.report and args.report.exists():
        raise FileExistsError(args.report)
    pose = MirrorPose(args.model, args.machine_config)
    receiver = SerialReceiver(args.port, args.baud)
    viewer, applied, last_status = None, None, None
    started = time.monotonic()
    try:
        if not args.headless:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(pose.model, pose.data, show_left_ui=False, show_right_ui=False)
            with viewer.lock():
                viewer.cam.lookat[:] = pose.data.qpos[:3]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.7, 135., -18.
                viewer.opt.geomgroup[3] = 0
        receiver.start()
        while (not args.seconds or time.monotonic() - started < args.seconds) and (not viewer or viewer.is_running()):
            now = time.monotonic()
            latest = receiver.snapshot()
            with viewer.lock() if viewer else nullcontext():
                if latest is not None and latest[1] != applied:
                    pose.apply(*latest)
                    applied = latest[1]
                pose.paint_status(now)
            if viewer:
                labels = pose.labels(now)
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100.value,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                    f"H7 READ-ONLY MIRROR | {'STALE / NO DATA' if pose.stale(now) else 'LIVE PACKETS'}\n"
                    f"IMU {labels[0]} | legs {labels[1]} / {labels[2]} | wheels {labels[3]} / {labels[4]}\n"
                    f"closure m: {pose.residual_m}\n"
                    f"POSITION FIXED z={pose.data.qpos[2]:.3f} m (NOT MEASURED)\n"
                    f"{receiver.state}\nrejected candidates {receiver.parser.rejected} | missing submitted frames {receiver.clock.missing}\n"
                    f"last rejected candidate: {receiver.parser.last_error or '-'}", "")])
                viewer.sync()
            current_status = (receiver.state, tuple(pose.labels(now)))
            if current_status != last_status:
                print(json.dumps({"state": receiver.state, "sources": pose.labels(now),
                                  "sequence": receiver.clock.sequence,
                                  "closure_m": pose.residual_m}), flush=True)
                last_status = current_status
            time.sleep(.02)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop_event.set()
        if receiver.is_alive():
            receiver.join(timeout=1.)
        if viewer:
            viewer.close()
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({
                "scope": "read-only kinematic mirror; world translation fixed; mapping unverified on bench",
                "accepted_frames": receiver.parser.accepted, "rejected_candidates": receiver.parser.rejected,
                "missing_submitted_frames": receiver.clock.missing,
                "restarts": receiver.clock.restarts, "duplicates": receiver.clock.duplicates,
                "serial_state": receiver.state, "sources": pose.labels(time.monotonic()),
                "closure_m": pose.residual_m, "qpos": pose.data.qpos.tolist(),
            }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
