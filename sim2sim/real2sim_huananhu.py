#!/usr/bin/env python3
"""Read-only H7 mirror: CRC-protected JustFloat -> pose and control telemetry.

Only kinematics are evaluated; this program never sends serial commands, runs a
policy, integrates physics or estimates world translation. H7's public feedback
already includes driver calibration and wheel reduction. The additional model
mapping is read from the delivered H7 machine_config.c, never inferred here.
"""
from __future__ import annotations

import argparse
import binascii
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import re
import sys
import threading
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml"
MACHINE_CONFIG = ROOT / "deployment/H7_RL/imcalib/config/machine_config.c"
TAIL, FRAME_BYTES = bytes.fromhex("0000807f"), 196
FRAME_BYTES_BY_VERSION = {1: FRAME_BYTES, 2: 644}
MAX_FRAME_BYTES = max(FRAME_BYTES_BY_VERSION.values())
STALE_S, CLOSURE_TOL_M = .2, .001
DRIVES = (("lf0", "lf00"), ("rf0", "rf00"))
PASSIVE = (("lf1", "lf01", "lf02", "lf03"), ("rf1", "rf01", "rf02", "rf03"))
WHEELS = ("lfwheel", "rfwheel")
PIN_PAIRS = tuple(tuple((f"{side}_rear_pin{i}", f"{side}_front_pin{i}")
                        for i in (1, 2)) for side in ("left", "right"))
MOTORS = ("lf0", "lf00", "rf0", "rf00", "lfwheel", "rfwheel")
ACTION_MOTORS = ("lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel")
# Channel 158 identifies the immutable controller profile, not an active-output flag.
WHEEL_RESPONSE_CATALOG = {
    "0": {"name": "original wheel P", "friction_nm": [0., 0.], "smoothing_rad_s": .5},
    "1": {"name": "experimental smooth friction", "friction_nm":
          [float(np.float32(.11604)), float(np.float32(.06147))], "smoothing_rad_s": .5},
}
KNOWN_DIFFERENCES = [
    "Previously recorded prone pose: left/right knee exceeded model limits by about "
    "7/4 degrees. User accepts the current mirror display; cause and physical travel "
    "remain unverified on bench. This is not a measured calibration error."
]


def read_u32(values: np.ndarray) -> int:
    if not np.isfinite(values).all() or any(v != int(v) or not 0 <= v <= 65535 for v in values):
        raise ValueError("invalid uint32 halves")
    return int(values[0]) | (int(values[1]) << 16)


def decode_frame(raw: bytes) -> np.ndarray:
    if len(raw) not in FRAME_BYTES_BY_VERSION.values() or raw[-4:] != TAIL:
        raise ValueError("frame length/tail")
    values = np.frombuffer(raw[:-4], dtype="<f4").copy()
    if values[3] != 7101. or values[4] not in FRAME_BYTES_BY_VERSION:
        raise ValueError("not mirror v1/v2")
    if len(raw) != FRAME_BYTES_BY_VERSION[int(values[4])]:
        raise ValueError("version/frame length mismatch")
    if values[-1] != binascii.crc_hqx(raw[:-8], 0xffff):
        raise ValueError("CRC mismatch")
    if values[5] != 0.:
        raise ValueError("unsupported machine (only big wheelleg)")
    read_u32(values[6:8])
    read_u32(values[8:10])
    mask = values[29]
    if not np.isfinite(mask) or mask != int(mask) or not 0 <= mask <= 127:
        raise ValueError("invalid source mask")
    if values[4] == 2:
        for index, limit in ((47, 3), (109, 63)):
            value = values[index]
            if not np.isfinite(value) or value != int(value) or not 0 <= value <= limit:
                raise ValueError("invalid telemetry mask/flags")
        for start in (48, 50, 52, 103, 105, 107, 156):
            read_u32(values[start:start + 2])
        if not np.isfinite(values[158]) or values[158] != int(values[158]) or not 0 <= values[158] <= 65535:
            raise ValueError("invalid wheel response profile")
    return values


class FrameParser:
    """Scan every candidate tail; a tail inside payload must not truncate a frame."""

    def __init__(self):
        self.buffer = bytearray()
        self.accepted = self.rejected = 0
        self.versions = {1: 0, 2: 0}
        self.last_version = None
        self.last_error = ""

    def feed(self, chunk: bytes) -> list[np.ndarray]:
        # Revisit only a possible tail split across reads, not old rejected tails.
        search = max(0, len(self.buffer) - len(TAIL) + 1)
        self.buffer.extend(chunk)
        frames = []
        while True:
            end = self.buffer.find(TAIL, search)
            if end < 0:
                break
            stop = end + 4
            if stop >= FRAME_BYTES:
                error = "frame length/tail"
                for size in FRAME_BYTES_BY_VERSION.values():
                    if stop < size:
                        continue
                    try:
                        frame = decode_frame(bytes(self.buffer[stop - size:stop]))
                    except ValueError as exc:
                        error = str(exc)
                    else:
                        frames.append(frame)
                        self.accepted += 1
                        self.versions[int(frame[4])] += 1
                        self.last_version = int(frame[4])
                        del self.buffer[:stop]
                        search = 0
                        break
                else:
                    self.rejected += 1
                    self.last_error = error
                    search = end + 1
                    continue
                continue
            search = end + 1
        # Retain a complete possible prefix of the next frame, even after noise.
        if len(self.buffer) >= MAX_FRAME_BYTES:
            del self.buffer[:-(MAX_FRAME_BYTES - 1)]
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
        from wheel_response import prepare_wheel_response
        prepare_wheel_response(self.model)
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
        # Read-only mirroring shows closed poses beyond training limits with a warning.
        # This does not validate the physical mapping or alter the model's limits.
        within_range = True
        for name, angle in zip(PASSIVE[side], result):
            joint = self.model.joint(name)
            if joint.limited[0] and not joint.range[0] - 1e-8 <= angle <= joint.range[1] + 1e-8:
                within_range = False
                break
        if not np.isfinite(error) or error > CLOSURE_TOL_M:
            self.data.qpos[indices] = previous
            mujoco.mj_kinematics(self.model, self.data)
            self.status[side + 1] = "INVALID CLOSURE"
            return False
        self.status[side + 1] = "OK" if within_range else "MODEL RANGE WARNING"
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
                self.model.geom_rgba[geom] = ((1., .65, .10, 1.) if labels[source] == "MODEL RANGE WARNING"
                                             else (.85, .20, .16, 1.))


def file_identity(path: Path) -> dict:
    path = path.resolve()
    try:
        relative = str(path.relative_to(ROOT))
    except ValueError:
        relative = None
    return {"path": str(path), "project_path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class SessionRecorder:
    """A bounded disk queue keeps slow storage out of the USB read loop."""

    def __init__(self, directory: Path, model_path: Path = MODEL,
                 machine_config: Path = MACHINE_CONFIG, firmware_manifest: Path | None = None):
        firmware_info = None
        if firmware_manifest is not None:
            firmware_info = {"file": file_identity(firmware_manifest), "expected_build": json.loads(firmware_manifest.read_text()),
                             "scope": "operator-supplied expected build; not hardware boot attestation"}
        self.directory = Path(directory)
        # A new directory prevents accidentally mixing two physical experiments.
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started_ns = time.monotonic_ns()
        self.pending = queue.Queue(maxsize=1024)
        self.error = None
        self.closed = False
        self.last_ns = self.event_count = self.written_bytes = 0
        self.metadata = {
            "format": "h7-mirror-session", "version": 1,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "timebase": "Jetson monotonic elapsed ns at read return; not sensor sample time",
            "model": file_identity(model_path), "machine_config": file_identity(machine_config),
            "program": file_identity(Path(__file__)),
            "wheel_response_catalog": WHEEL_RESPONSE_CATALOG,
            "firmware_manifest": firmware_info,
            "known_differences": KNOWN_DIFFERENCES,
            "scope": "read-only kinematics/control telemetry; no hardware commands or measured world translation",
            "complete": False, "error": None,
        }
        self._save_metadata()
        self.stream = (self.directory / "stream.jsonl").open("xb")
        self.digest = hashlib.sha256()
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()

    def _save_metadata(self):
        temporary = self.directory / "meta.json.tmp"
        temporary.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.directory / "meta.json")

    def _write_event(self, event: dict):
        raw = (json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        self.stream.write(raw)
        self.digest.update(raw)
        self.event_count += 1
        self.written_bytes += len(raw)

    def _write(self):
        try:
            while True:
                event = self.pending.get()
                if event is None:
                    break
                self._write_event(event)
            self.stream.flush()
        except (OSError, ValueError) as exc:
            self.error = f"recording write failed: {exc}"
        finally:
            try:
                self.stream.close()
            except OSError as exc:
                self.error = f"recording close failed: {exc}"

    def record(self, kind: str, elapsed_ns: int | None = None, **payload) -> bool:
        if self.closed or self.error:
            return False
        elapsed_ns = time.monotonic_ns() - self.started_ns if elapsed_ns is None else elapsed_ns
        self.last_ns = max(self.last_ns, elapsed_ns)
        try:
            self.pending.put_nowait({"type": kind, "t_ns": elapsed_ns, **payload})
        except queue.Full:
            self.error = "recording queue full; session incomplete (USB mirror continues)"
            return False
        return True

    def finish(self, elapsed_ns: int | None = None):
        if self.closed:
            return
        ended = max(self.last_ns, time.monotonic_ns() - self.started_ns
                    if elapsed_ns is None else elapsed_ns)
        self.record("stop", elapsed_ns=ended)
        self.closed = True
        # Shutdown may wait for storage; the USB reader has already stopped.
        while self.writer.is_alive():
            try:
                self.pending.put(None, timeout=.05)
                break
            except queue.Full:
                continue
        self.writer.join()
        self.metadata.update(complete=self.error is None, error=self.error,
                             duration_ns=ended, event_count=self.event_count,
                             stream_sha256=self.digest.hexdigest())
        try:
            self._save_metadata()
        except OSError as exc:
            self.error = f"cannot finalize recording metadata: {exc}"
            self.metadata.update(complete=False, error=self.error)


def recording_info(recorder: SessionRecorder, frames: int, elapsed_s: float, seconds: float) -> dict:
    return {"status": "ERROR" if recorder.error else "RECORDING",
            "elapsed_s": round(elapsed_s, 1),
            "remaining_s": round(max(0., seconds - elapsed_s), 1) if seconds else None,
            "received_frames": frames, "written_bytes": recorder.written_bytes,
            "pending_events": recorder.pending.qsize(),
            "directory": str(recorder.directory), "error": recorder.error}


def wire_info(parser: FrameParser) -> dict:
    notice = ("v2 last-successful policy pairs/latest output; repeated sequence is not new inference"
              if parser.last_version == 2 else
              "v1: command/obs/action/target/torque/send telemetry UNSUPPORTED"
              if parser.last_version == 1 else "no validated wire frames received")
    return {"wire_version": parser.last_version, "wire_version_counts": parser.versions.copy(),
            "control_telemetry_supported": parser.last_version == 2, "control_telemetry_notice": notice}


def load_session(directory: Path, model_path: Path | None = None,
                 machine_config: Path | None = None) -> tuple[dict, Path, Path]:
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    if meta.get("format") != "h7-mirror-session" or meta.get("version") != 1:
        raise ValueError("unsupported recording format/version")
    if not meta.get("complete"):
        raise ValueError(f"incomplete recording: {meta.get('error') or 'not finalized'}")
    if (type(meta.get("duration_ns")) is not int or meta["duration_ns"] < 0
            or type(meta.get("event_count")) is not int):
        raise ValueError("invalid recording duration/event count")
    digest = hashlib.sha256()
    with (directory / "stream.jsonl").open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    if digest.hexdigest() != meta.get("stream_sha256"):
        raise ValueError("recording stream hash mismatch (truncated or changed)")
    count, last = 0, None
    for event in iter_session_events(directory):
        count, last = count + 1, event
    if count != meta["event_count"] or last["t_ns"] != meta["duration_ns"]:
        raise ValueError("recording event count/duration mismatch")
    paths = []
    for key, explicit in (("model", model_path), ("machine_config", machine_config)):
        identity = meta[key]
        candidates = ([Path(explicit)] if explicit is not None else
                      ([ROOT / identity["project_path"]] if identity.get("project_path") else [])
                      + [Path(identity["path"])])
        if key == "model" and explicit is None:
            # Promotion changes the active XML; old recordings still require
            # their exact frozen model bytes, never a relaxed hash check.
            candidates.append(ROOT / 'artifacts/model-versions' / identity['sha256']
                              / 'source/model.xml')
        chosen = next((path for path in candidates if path.is_file()
                       and file_identity(path)["sha256"] == identity["sha256"]), None)
        if chosen is None:
            raise ValueError(f"{key} hash mismatch or file missing; supply the recorded version")
        paths.append(chosen)
    return meta, *paths


def iter_session_events(directory: Path):
    previous, stopped, connected = -1, False, False
    with (Path(directory) / "stream.jsonl").open("rb") as source:
        for number, raw in enumerate(source, 1):
            if not raw.endswith(b"\n"):
                raise ValueError(f"truncated recording event at line {number}")
            try:
                event = json.loads(raw)
                stamp, kind = event["t_ns"], event["type"]
                if type(stamp) is not int or stamp < 0 or stamp < previous or stopped:
                    raise ValueError("nonmonotonic time or events after stop")
                if kind == "data":
                    if not connected:
                        raise ValueError("data outside a connection")
                    chunk = bytes.fromhex(event["data_hex"])
                    if not chunk or len(chunk) > 4096:
                        raise ValueError("invalid read chunk size")
                elif kind == "connected":
                    connected = True
                elif kind == "disconnected":
                    connected = False
                elif kind == "stop":
                    stopped = True
                else:
                    raise ValueError("unknown event type")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid recording event at line {number}: {exc}") from exc
            previous = stamp
            yield event
    if not stopped:
        raise ValueError("incomplete recording: missing stop event")


class SerialReceiver(threading.Thread):
    """Own the serial reader; the viewer consumes only the most recent valid frame."""

    def __init__(self, port: str, baud: int, recorder: SessionRecorder | None = None):
        super().__init__(daemon=True)
        self.port, self.baud = port, baud
        self.parser, self.clock = FrameParser(), FrameClock()
        self.stop_event, self.lock = threading.Event(), threading.Lock()
        self.latest = None
        self.state = "CONNECTING"
        self.recorder = recorder

    def now(self) -> float:
        return time.monotonic()

    def consume(self, chunk: bytes, received_at: float):
        for frame in self.parser.feed(chunk):
            if self.clock.accept(frame):
                with self.lock:
                    self.latest = (frame, received_at)

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
                    if self.recorder:
                        self.recorder.record("connected", state=self.state, port=self.port, baud=self.baud)
                    while not self.stop_event.is_set():
                        chunk = connection.read(min(connection.in_waiting or 1, 4096))
                        received_ns = time.monotonic_ns()
                        if chunk and self.recorder:
                            self.recorder.record("data", elapsed_ns=received_ns - self.recorder.started_ns,
                                                 data_hex=chunk.hex())
                        self.consume(chunk, received_ns / 1e9)
            except (serial.SerialException, OSError) as exc:
                self.state = f"DISCONNECTED: {exc}"
                if self.recorder:
                    self.recorder.record("disconnected", state=self.state)
                self.stop_event.wait(.5)

    def snapshot(self):
        with self.lock:
            return self.latest


class ReplayReceiver(SerialReceiver):
    """Replay USB chunks on their recorded clock, without opening a serial port."""

    def __init__(self, directory: Path):
        super().__init__("", 0)
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "meta.json").read_text())
        self.duration_s = self.metadata["duration_ns"] / 1e9
        self.finished_event = threading.Event()
        self.error = None
        self.position = 0.
        self.wall_started = None
        self.state = "REPLAY READY"

    def now(self) -> float:
        if self.wall_started is None or self.finished_event.is_set():
            return self.position
        return min(self.duration_s, time.monotonic() - self.wall_started)

    def run(self):
        self.wall_started = time.monotonic()
        try:
            for event in iter_session_events(self.directory):
                at = event["t_ns"] / 1e9
                if self.stop_event.wait(max(0., at - self.now())):
                    break
                self.position = at
                if event["type"] == "data":
                    self.consume(bytes.fromhex(event["data_hex"]), at)
                elif event["type"] == "connected":
                    self.parser.buffer.clear()
                    self.state = "REPLAY CONNECTED"
                elif event["type"] == "disconnected":
                    self.state = "REPLAY DISCONNECTED"
                else:
                    self.state = "REPLAY COMPLETE"
        except (OSError, ValueError) as exc:
            self.error = str(exc)
            self.state = f"REPLAY ERROR: {exc}"
        finally:
            self.position = self.now()
            self.finished_event.set()


def telemetry_row(frame: np.ndarray) -> dict:
    """Keep v1 and unavailable v2 groups empty rather than inventing control data."""
    columns = ["telemetry_valid_mask", "policy_pair_valid", "output_snapshot_valid", "rl_pd_valid",
               "policy_sequence", "policy_sample_tick_ms", "action_publish_tick_ms", "policy_run_us",
               "command_vx_m_s", "command_yaw_rad_s", "command_leg_length_m",
               "output_sequence", "output_tick_ms", "output_used_policy_sequence", "output_flags",
               "output_zero", "output_dm_hal_success", "output_dji_hal_success", "output_enabled", "drive_permitted",
               "imu_sample_age_ms", "rc_sample_age_ms", "rc_strategy", "rc_flags",
               "recovery_phase", "policy_ready", "action_reset_count", "wheel_response_mode",
               "wheel_compensation_l_nm", "wheel_compensation_r_nm", "wheel_compensation_smoothing_rad_s"]
    columns += [f"obs_{i:02d}" for i in range(39)]
    columns += [f"{prefix}_{name}" for prefix in ("action", "pd_used_action") for name in ACTION_MOTORS]
    columns += [f"{prefix}_{name}_{'rad' if i < 4 else 'rad_s'}"
                for prefix in ("pd_target", "pd_error") for i, name in enumerate(MOTORS)]
    columns += [f"pd_pos_{name}_rad" for name in MOTORS[:4]]
    columns += [f"pd_vel_{name}_rad_s" for name in MOTORS]
    columns += [f"gated_torque_{name}_firmware_nm" for name in MOTORS]
    columns += [f"motor_age_{name}_ms" for name in MOTORS]
    columns += [f"feedback_torque_{name}_firmware_nm" for name in MOTORS[:4]]
    columns += [f"wheel_current_{name}_raw" for name in MOTORS[4:]]
    row = dict.fromkeys(columns)
    if len(frame) != 160 or frame[4] != 2:
        return row

    def number(index, available=True):
        return float(frame[index]) if available and np.isfinite(frame[index]) else None

    valid, output_flags = int(frame[47]), int(frame[109])
    policy_ok, output_ok = bool(valid & 1), bool(valid & 2)
    pd_ok = output_ok and bool(output_flags & 1)
    row.update(telemetry_valid_mask=valid, policy_pair_valid=int(policy_ok),
               output_snapshot_valid=int(output_ok), rl_pd_valid=int(pd_ok))
    # Sequence/ticks identify the paired snapshots even when their data group is invalid.
    for key, start in (("policy_sequence", 48), ("policy_sample_tick_ms", 50),
                       ("action_publish_tick_ms", 52), ("output_sequence", 103),
                       ("output_tick_ms", 105), ("output_used_policy_sequence", 107),
                       ("action_reset_count", 156)):
        row[key] = read_u32(frame[start:start + 2])
    row["output_flags"] = output_flags if output_ok else None
    for key, bit in (("output_zero", 1), ("output_dm_hal_success", 2), ("output_dji_hal_success", 3),
                     ("output_enabled", 4), ("drive_permitted", 5)):
        row[key] = int(bool(output_flags & (1 << bit))) if output_ok else None
    for i, key in enumerate(("command_vx_m_s", "command_yaw_rad_s", "command_leg_length_m")):
        row[key] = number(54 + i, policy_ok)
    for i in range(39):
        row[f"obs_{i:02d}"] = number(57 + i, policy_ok)
    for i, name in enumerate(ACTION_MOTORS):
        row[f"action_{name}"] = number(96 + i, policy_ok)
        row[f"pd_used_action_{name}"] = number(116 + i, pd_ok)
    row["policy_run_us"] = number(102, policy_ok)
    for i, name in enumerate(MOTORS):
        unit = "rad" if i < 4 else "rad_s"
        target = row[f"pd_target_{name}_{unit}"] = number(110 + i, pd_ok)
        velocity = row[f"pd_vel_{name}_rad_s"] = number(126 + i, pd_ok)
        position = number(122 + i, pd_ok) if i < 4 else None
        if i < 4:
            row[f"pd_pos_{name}_rad"] = position
        actual = position if i < 4 else velocity
        row[f"pd_error_{name}_{unit}"] = target - actual if target is not None and actual is not None else None
        row[f"gated_torque_{name}_firmware_nm"] = number(132 + i, output_ok)
        age = number(138 + i)
        row[f"motor_age_{name}_ms"] = age if age is not None and age >= 0 else None
        source_ok = (bool(int(frame[29]) & (1 << (i + 1)))
                     and np.isfinite(frame[31]) and 0 <= frame[31] < STALE_S * 1000
                     and age is not None and 0 <= age < STALE_S * 1000)
        if i < 4:
            row[f"feedback_torque_{name}_firmware_nm"] = number(146 + i, source_ok)
        else:
            row[f"wheel_current_{name}_raw"] = number(150 + i - 4, source_ok)
    for key, index in (("imu_sample_age_ms", 144), ("rc_sample_age_ms", 145)):
        age = number(index)
        row[key] = age if age is not None and age >= 0 else None
    for key, index in (("rc_strategy", 152), ("rc_flags", 153),
                       ("recovery_phase", 154), ("policy_ready", 155)):
        row[key] = number(index)
    row['wheel_response_mode'] = int(frame[158])
    profile = WHEEL_RESPONSE_CATALOG.get(str(int(frame[158])))
    if profile is not None:
        row['wheel_compensation_l_nm'], row['wheel_compensation_r_nm'] = profile['friction_nm']
        row['wheel_compensation_smoothing_rad_s'] = profile['smoothing_rad_s']
    return row


def trajectory_row(pose: MirrorPose, frame: np.ndarray, at: float, segment: int) -> dict:
    """Export current measurements; never export held/invalid values as live data."""
    row = dict(t_s=at, h7_tick_ms=read_u32(frame[8:10]), sequence=read_u32(frame[6:8]),
               segment=segment, online_mask=float(frame[0]), state_bits=float(frame[1]),
               fault=float(frame[2]), valid_mask=int(frame[29]),
               imu_age_ms=float(frame[30]), motor_age_ms=float(frame[31]))
    motor_recent = np.isfinite(frame[31]) and 0 <= frame[31] < STALE_S * 1000
    mask = int(frame[29])
    raw_q = np.concatenate((frame[10:14], frame[18:20]))
    raw_qd = np.concatenate((frame[14:18], frame[20:22]))
    for i, name in enumerate(MOTORS):
        available = motor_recent and bool(mask & (1 << (i + 1)))
        q = float(raw_q[i]) if available and np.isfinite(raw_q[i]) else None
        qd = float(raw_qd[i]) if available and np.isfinite(raw_qd[i]) else None
        row[f"{name}_raw_q_rad"] = q
        row[f"{name}_q_rad"] = (float(pose.sign[i] * (q - (pose.zero[i] if i < 4 else 0.)))
                                 if q is not None else None)
        row[f"{name}_qd_rad_s"] = float(pose.sign[i] * qd) if qd is not None else None
    imu_ok = pose.status[0] == "OK"
    quat = pose.data.qpos[3:7] if imu_ok else [None] * 4
    row.update({f"quat_{n}": float(v) if v is not None else None for n, v in zip("wxyz", quat)})
    rpy = [None] * 3
    if imu_ok:
        w, x, y, z = quat
        rpy = [np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
               np.arcsin(np.clip(2 * (w*y - z*x), -1., 1.)),
               np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))]
    row.update({f"{name}_rad": float(v) if v is not None else None
                for name, v in zip(("roll", "pitch", "yaw"), rpy)})
    row.update({f"gyro_{name}_rad_s": float(v) if imu_ok and np.isfinite(v) else None
                for name, v in zip("xyz", frame[26:29])})
    base_rotation = pose.data.body("base_link").xmat.reshape(3, 3)
    for side, label in enumerate(("left", "right")):
        closed = pose.status[side + 1] in ("OK", "MODEL RANGE WARNING")
        length = angle = knee = excess = None
        if closed:
            # Same mechanism-plane definition as H7; lateral CAD offsets are excluded.
            delta = base_rotation.T @ (pose.data.body(WHEELS[side]).xpos
                                      - pose.data.body(DRIVES[side][0]).xpos)
            length, angle = float(np.hypot(delta[0], delta[2])), float(np.arctan2(delta[0], -delta[2]))
            knee = float(pose.data.qpos[pose.passive_q[side][0]])
            excess = 0.
            for name, address in zip(PASSIVE[side], pose.passive_q[side]):
                joint = pose.model.joint(name)
                if joint.limited[0]:
                    q = pose.data.qpos[address]
                    excess = max(excess, float(joint.range[0] - q), float(q - joint.range[1]))
        residual = (pose.residual_m[side] if closed or pose.status[side + 1] == "INVALID CLOSURE" else None)
        row.update({f"{label}_leg_length_m": length, f"{label}_leg_angle_rad": angle,
                    f"{label}_knee_rad": knee, f"{label}_closure_m": residual,
                    f"{label}_range_excess_rad": excess})
    row.update(dict(zip(("imu_status", "left_leg_status", "right_leg_status",
                         "left_wheel_status", "right_wheel_status"), pose.labels(at))))
    row.update(telemetry_row(frame))
    return row


def interval_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p95": None}
    return {"count": len(values), "min": float(np.min(values)), "max": float(np.max(values)),
            "mean": float(np.mean(values)), "p95": float(np.percentile(values, 95))}


def analyze_session(directory: Path, model_path: Path | None = None,
                    machine_config: Path | None = None) -> dict:
    metadata, model_path, machine_config = load_session(directory, model_path, machine_config)
    # Matplotlib is only needed for offline analysis on the existing mjlab machine.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(directory) / "analysis"
    output.mkdir(exist_ok=False)
    pose = MirrorPose(model_path, machine_config)
    parser, clock = FrameParser(), FrameClock()
    rows, host_intervals, board_intervals, stale_intervals = [], [], [], []
    connections = disconnections = 0
    last_at = last_tick = first_at = None
    status_counts, channels = {}, {}
    with (output / "trajectory.csv").open("x", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(trajectory_row(pose, np.zeros(48), 0., 0)))
        writer.writeheader()
        for event in iter_session_events(directory):
            at = event["t_ns"] / 1e9
            if event["type"] == "connected":
                connections += 1
                parser.buffer.clear()
                continue
            if event["type"] == "disconnected":
                disconnections += 1
                continue
            if event["type"] != "data":
                continue
            for frame in parser.feed(bytes.fromhex(event["data_hex"])):
                restarts = clock.restarts
                if not clock.accept(frame):
                    continue
                tick = clock.tick
                if last_at is not None:
                    host_intervals.append((at - last_at) * 1000)
                    if clock.restarts == restarts:
                        board_intervals.append(float((tick - last_tick) & 0xffffffff))
                    if at - last_at >= STALE_S:
                        stale_intervals.append({"start_s": last_at + STALE_S, "end_s": at})
                else:
                    first_at = at
                last_at, last_tick = at, tick
                pose.apply(frame, at)
                row = trajectory_row(pose, frame, at, clock.restarts)
                # JSON and CSV keep missing measurements distinct from real zero.
                row = {key: (None if isinstance(value, (float, np.floating))
                             and not np.isfinite(value) else value) for key, value in row.items()}
                writer.writerow(row)
                rows.append(row)
                for key, value in row.items():
                    if key.endswith("_status"):
                        counts = status_counts.setdefault(key, {})
                        counts[value] = counts.get(value, 0) + 1
                    elif value is not None:
                        item = channels.setdefault(key, {"count": 0, "min": value, "max": value, "sum": 0.})
                        item["count"] += 1
                        item["min"], item["max"] = min(item["min"], value), max(item["max"], value)
                        item["sum"] += value
    duration = metadata["duration_ns"] / 1e9
    if last_at is not None and duration - last_at >= STALE_S:
        stale_intervals.append({"start_s": last_at + STALE_S, "end_s": duration})
    for item in channels.values():
        item["mean"] = item.pop("sum") / item["count"]
    report = {
        "scope": "read-only kinematic/control telemetry; derived leg geometry is not independent measurement; "
                 "100 Hz USB samples do not contain every 500 Hz output; arrival/publication is not sensor timing or dynamics gap",
        "recording_complete": metadata["complete"], "duration_s": duration,
        "accepted_frames": parser.accepted, "applied_frames": len(rows),
        "rejected_candidates": parser.rejected, "last_rejected_candidate": parser.last_error,
        "missing_submitted_frames": clock.missing, "restarts": clock.restarts,
        "duplicates": clock.duplicates, "out_of_order": clock.out_of_order,
        "connections": connections, "disconnections": disconnections,
        "host_frame_interval_ms": interval_summary(host_intervals),
        "h7_frame_interval_ms": interval_summary(board_intervals),
        "no_data_before_first_frame_s": first_at if first_at is not None else duration,
        "stale_intervals": stale_intervals, "status_counts": status_counts, "channels": channels,
        "model": metadata["model"], "machine_config": metadata["machine_config"],
        "known_differences": metadata.get("known_differences", []),
        "world_translation_measured": False, "physical_mapping_verified": False,
    }
    telemetry_rows = [row for row in rows if row["telemetry_valid_mask"] is not None]
    output_rows = [row for row in telemetry_rows if row["output_snapshot_valid"]]
    report["control_telemetry"] = {
        "supported": bool(telemetry_rows), "accepted_wire_versions": parser.versions,
        "v1_unsupported_frames": len(rows) - len(telemetry_rows),
        "unsupported": ([] if telemetry_rows else
                        ["policy command/obs/action, PD target/feedback, gated torque and output send flags require wire v2"]),
        "policy_pair_frames": sum(row["policy_pair_valid"] for row in telemetry_rows),
        "output_snapshot_frames": len(output_rows),
        "unique_policy_sequences": len({(row["segment"], row["policy_sequence"])
                                        for row in telemetry_rows if row["policy_pair_valid"]}),
        "unique_output_sequences": len({(row["segment"], row["output_sequence"]) for row in output_rows}),
        "rl_pd_frames": sum(row["rl_pd_valid"] for row in telemetry_rows),
        "policy_array_order": list(ACTION_MOTORS), "motor_array_order": list(MOTORS),
        "observation": "actual 39 clipped/scaled values from the last successful policy pair; retained historical pairs may repeat after action clearing, not a reconstruction or proof of current RL engagement",
        "command": "raw RC vel/yaw/leg-length of the policy pair, before network clipping/scaling; leg length is not measured world height",
        "sequence_scope": "unique sequences are counted within detected board-restart segments; repeated 100 Hz telemetry frames are not new inferences or a full output history",
        "output_scope": "latest 500 Hz snapshot sampled by 100 Hz telemetry; firmware-coordinate gated Nm are drive-entry requests, not measured applied torque",
        "send_scope": "DM/DJI flags report group HAL submission success only; false may mean no submission, and success is not motor acknowledgement",
        "timing_scope": "policy_sample_tick_ms is the 500 Hz cache-use tick, not the original sensor timestamp; source ages are reported separately",
        "wheel_feedback": "current_raw, not calibrated Nm",
        "wheel_response_modes": sorted({row['wheel_response_mode'] for row in telemetry_rows}),
        "wheel_response_catalog_archived": 'wheel_response_catalog' in metadata,
        "wheel_response_scope": "channel 158 is the compiled controller profile; coefficients are versioned control settings, not measured friction or proof of active output",
        "output_sample_counts": {
            name: sum(bool(int(row["output_flags"]) & (1 << bit)) for row in output_rows)
            for name, bit in (("zero_output", 1), ("dm_hal_success", 2), ("dji_hal_success", 3),
                              ("output_enabled", 4), ("drive_permitted", 5))},
    }
    report["pd_tracking_errors"] = {"scope": "target minus the same PD snapshot feedback; not an independent simulation/dynamics comparison", "motors": {}}
    for i, name in enumerate(MOTORS):
        key = f"pd_error_{name}_{'rad' if i < 4 else 'rad_s'}"
        values = [row[key] for row in rows if row[key] is not None]
        report["pd_tracking_errors"]["motors"][name] = {
            "unit": "rad" if i < 4 else "rad/s", "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "mae": float(np.mean(np.abs(values))) if values else None,
            "rmse": float(np.sqrt(np.mean(np.square(values)))) if values else None,
            "p95_absolute": float(np.percentile(np.abs(values), 95)) if values else None,
        }
    report["range_warnings"] = {}
    for side in ("left", "right"):
        counts = status_counts.get(f"{side}_leg_status", {})
        warnings = counts.get("MODEL RANGE WARNING", 0)
        closed = warnings + counts.get("OK", 0)
        report["range_warnings"][side] = {
            "closed_frames": closed, "warning_frames": warnings,
            "warning_fraction": warnings / closed if closed else None,
            "max_excess_rad": channels.get(f"{side}_range_excess_rad", {}).get("max"),
        }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(6, 1, figsize=(13, 17), sharex=True, constrained_layout=True)

    def series(name):
        x, y, previous = [], [], None
        for row in rows:
            at = row["t_s"]
            if previous is not None and at - previous >= STALE_S:
                x.append(previous + STALE_S)
                y.append(np.nan)
            x.append(at)
            y.append(row.get(name) if row.get(name) is not None else np.nan)
            previous = at
        return x, y

    for name in MOTORS:
        axes[0].plot(*series(f"{name}_q_rad"), label=name, linewidth=.9)
        axes[1].plot(*series(f"{name}_qd_rad_s"), label=name, linewidth=.9)
    for name in ("roll", "pitch", "yaw"):
        axes[2].plot(*series(f"{name}_rad"), label=name)
    for name in "xyz":
        axes[3].plot(*series(f"gyro_{name}_rad_s"), label=name)
    for name in ("left", "right"):
        axes[4].plot(*series(f"{name}_leg_length_m"), label=name)
        axes[5].plot(*series(f"{name}_knee_rad"), label=name)
        bounds = pose.model.joint(PASSIVE[(0 if name == "left" else 1)][0]).range
        for bound in bounds:
            axes[5].axhline(bound, linestyle=":", color="grey", linewidth=.7)
    labels = ("Motor q [rad]", "Motor qd [rad/s]", "IMU attitude [rad]", "IMU gyro [rad/s]",
              "Model planar leg length [m]", "Model knee [rad]; dotted XML limits")
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.grid(alpha=.3)
        if rows:
            axis.legend(loc="upper right", ncol=3, fontsize="small")
        for gap in stale_intervals:
            axis.axvspan(gap["start_s"], gap["end_s"], color="red", alpha=.1)
    axes[-1].set_xlabel("Host receive time since recording start [s] (not control latency)")
    fig.suptitle("Read-only real2sim | fixed world position NOT MEASURED | no dynamics comparison")
    fig.savefig(output / "signals.png", dpi=140)
    plt.close(fig)
    # Group by units and controller stage so command, feedback and drive requests are not conflated.
    fig, axes = plt.subplots(10, 1, figsize=(13, 25), sharex=True, constrained_layout=True)
    for key, label in (("command_vx_m_s", "vx [m/s]"), ("command_yaw_rad_s", "yaw [rad/s]"),
                       ("command_leg_length_m", "raw leg length [m]")):
        axes[0].plot(*series(key), label=label)
    for name in ACTION_MOTORS:
        axes[1].plot(*series(f"action_{name}"), label=name, linewidth=.9)
    for name in MOTORS[:4]:
        axes[2].plot(*series(f"pd_target_{name}_rad"), label=name, linewidth=.9)
        axes[4].plot(*series(f"pd_error_{name}_rad"), label=name, linewidth=.9)
        axes[6].plot(*series(f"feedback_torque_{name}_firmware_nm"), label=name, linewidth=.9)
    for name in MOTORS[4:]:
        axes[3].plot(*series(f"pd_target_{name}_rad_s"), label=f"{name} target")
        axes[3].plot(*series(f"pd_vel_{name}_rad_s"), label=f"{name} feedback", linestyle="--")
        axes[7].plot(*series(f"wheel_current_{name}_raw"), label=name)
    for name in MOTORS:
        axes[5].plot(*series(f"gated_torque_{name}_firmware_nm"), label=name, linewidth=.9)
        axes[8].plot(*series(f"motor_age_{name}_ms"), label=name, linewidth=.9)
    for key, label in (("imu_sample_age_ms", "IMU"), ("rc_sample_age_ms", "RC")):
        axes[8].plot(*series(key), label=label, linestyle="--")
    for bit, label in ((0, "RL PD valid"), (1, "zero output"), (2, "DM HAL success"),
                       (3, "DJI HAL success"), (4, "output enabled"), (5, "drive permitted")):
        x, flags = series("output_flags")
        axes[9].step(x, [float(bool(int(value) & (1 << bit))) + bit * 1.2
                         if np.isfinite(value) else np.nan for value in flags], where="post", label=label)
    control_labels = ("Raw paired RC commands", "Raw network actions", "PD leg targets [rad]",
                      "PD wheel speed [rad/s]", "Same-snapshot leg error [rad]",
                      "Gated drive entry [firmware Nm]", "DM feedback [firmware Nm]",
                      "Wheel current [raw, NOT Nm]", "Original source age [ms]", "Output flags (offset per bit)")
    for axis, label in zip(axes, control_labels):
        axis.set_ylabel(label)
        axis.grid(alpha=.3)
        if telemetry_rows:
            axis.legend(loc="upper right", ncol=3, fontsize="small")
        else:
            axis.text(.5, .5, "UNSUPPORTED: this recording has no wire v2 telemetry", ha="center", transform=axis.transAxes)
        for gap in stale_intervals:
            axis.axvspan(gap["start_s"], gap["end_s"], color="red", alpha=.1)
    axes[-1].set_xlabel("Host receive time [s]; 100 Hz samples of 500 Hz output, not a complete output history")
    fig.suptitle("Read-only control telemetry | drive-entry requests and HAL flags do not measure physical actuation")
    fig.savefig(output / "telemetry.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(8, 1, figsize=(13, 20), sharex=True, constrained_layout=True)
    groups = ((0, 4, "Command slots"), (4, 7, "Scaled gyro"), (7, 10, "Projected gravity"),
              (10, 16, "Position slots"), (16, 22, "Scaled velocity"), (22, 28, "Previous actions"),
              (28, 35, "Mode flags"), (35, 39, "State flags"))
    for axis, (start, stop, label) in zip(axes, groups):
        for i in range(start, stop):
            axis.plot(*series(f"obs_{i:02d}"), label=f"obs[{i}]", linewidth=.9)
        axis.set_ylabel(label)
        axis.grid(alpha=.3)
        if telemetry_rows:
            axis.legend(loc="upper right", ncol=4, fontsize="small")
        else:
            axis.text(.5, .5, "UNSUPPORTED: actual network input was not recorded", ha="center", transform=axis.transAxes)
        for gap in stale_intervals:
            axis.axvspan(gap["start_s"], gap["end_s"], color="red", alpha=.1)
    axes[-1].set_xlabel("Host receive time [s]; repeated policy snapshots do not mean new inferences")
    fig.suptitle("Last successful H7 network input, 39 values | retained pairs may be historical; clipping/scaling already applied")
    fig.savefig(output / "observations.png", dpi=140)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--port", help="H7 onboard USB CDC or USB-to-UART device; prefer /dev/serial/by-id/...")
    source.add_argument("--replay", type=Path, help="replay a recorded session at original speed; no USB access")
    source.add_argument("--analyze", type=Path, help="offline CSV/JSON/PNG analysis with the existing mjlab Python")
    parser.add_argument("--record", type=Path, help="new session directory; only with --port")
    parser.add_argument("--firmware-manifest", type=Path, help="archive the expected build manifest with --record; not boot attestation")
    parser.add_argument("--baud", type=int, default=1152000, help="UART baud rate; CDC accepts line coding but uses USB link speed")
    parser.add_argument("--model", type=Path, help="live model or recorded-model override (matching SHA256)")
    parser.add_argument("--machine-config", type=Path, help="live mapping or recorded-config override (matching SHA256)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0., help="0 keeps the display open until closed")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.seconds < 0 or not np.isfinite(args.seconds):
        parser.error("--seconds must be finite and nonnegative")
    if args.record and not args.port:
        parser.error("--record requires --port")
    if args.firmware_manifest and not args.record:
        parser.error("--firmware-manifest requires --record")
    if args.report and args.report.exists():
        raise FileExistsError(args.report)
    if args.analyze:
        report = analyze_session(args.analyze, args.model, args.machine_config)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"analysis": str(args.analyze / "analysis"),
                          "accepted_frames": report["accepted_frames"],
                          "applied_frames": report["applied_frames"]}), flush=True)
        return 0
    if args.replay:
        _, model_path, config_path = load_session(args.replay, args.model, args.machine_config)
    else:
        model_path, config_path = args.model or MODEL, args.machine_config or MACHINE_CONFIG
    pose = MirrorPose(model_path, config_path)
    recorder = SessionRecorder(args.record, model_path, config_path, args.firmware_manifest) if args.record else None
    receiver = ReplayReceiver(args.replay) if args.replay else SerialReceiver(args.port, args.baud, recorder)
    viewer, applied, last_status, last_record_error = None, None, None, None
    viewer_threads, started, last_progress = [], None, None
    try:
        if not args.headless:
            import mujoco.viewer
            before_launch = set(threading.enumerate())
            viewer = mujoco.viewer.launch_passive(pose.model, pose.data, show_left_ui=False, show_right_ui=False)
            # MuJoCo 3.11 close() can return before daemon window teardown finishes.
            viewer_threads = [thread for thread in threading.enumerate()
                              if thread not in before_launch and thread.daemon]
            with viewer.lock():
                viewer.cam.lookat[:] = pose.data.qpos[:3]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.7, 135., -18.
                viewer.opt.geomgroup[3] = 0
        receiver.start()
        started = time.monotonic()
        while (not args.seconds or time.monotonic() - started < args.seconds) and (not viewer or viewer.is_running()):
            # Replay timestamps and source ages use the recorded virtual clock.
            now = receiver.now()
            elapsed = time.monotonic() - started
            info = recording_info(recorder, receiver.parser.accepted, elapsed, args.seconds) if recorder else None
            wire = wire_info(receiver.parser)
            if info:
                info.update(wire)
            latest = receiver.snapshot()
            with viewer.lock() if viewer else nullcontext():
                if latest is not None and latest[1] != applied:
                    pose.apply(*latest)
                    applied = latest[1]
                pose.paint_status(now)
            if viewer and viewer.is_running():
                labels = pose.labels(now)
                texts = [(mujoco.mjtFont.mjFONT_NORMAL.value,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                    f"H7 READ-ONLY {'REPLAY' if args.replay else 'MIRROR'} | {'STALE / NO DATA' if pose.stale(now) else 'FRESH PACKETS'}\n"
                    f"IMU {labels[0]} | legs {labels[1]} / {labels[2]} | wheels {labels[3]} / {labels[4]}\n"
                    f"closure m: {pose.residual_m}\n"
                    f"POSITION FIXED z={pose.data.qpos[2]:.3f} m (NOT MEASURED)\n"
                    f"WIRE {wire['control_telemetry_notice']}\n"
                    f"{receiver.state}\nrejected candidates {receiver.parser.rejected} | missing submitted frames {receiver.clock.missing}\n"
                    f"last rejected candidate: {receiver.parser.last_error or '-'}", "")]
                if info:
                    remaining = (f" | left {info['remaining_s']:.1f}s" if args.seconds else " | manual stop")
                    texts.append((mujoco.mjtFont.mjFONT_NORMAL.value,
                        mujoco.mjtGridPos.mjGRID_TOPRIGHT.value,
                        f"REC {info['status']} | {info['elapsed_s']:.1f}s{remaining}\n"
                        f"RX frames {info['received_frames']} | stream {info['written_bytes'] / 1048576:.2f} MiB\n"
                        f"Pending writes {info['pending_events']}\nSAVE {info['directory']}\n"
                        f"{info['error'] or 'Ctrl+C or close window to finish saving'}", ""))
                viewer.set_texts(texts)
                viewer.sync()
            if info and (last_progress is None or elapsed - last_progress >= 1.):
                print(json.dumps({"recording": info}), flush=True)
                last_progress = elapsed
            current_status = (receiver.state, tuple(pose.labels(now)), receiver.parser.last_version)
            if current_status != last_status:
                print(json.dumps({"state": receiver.state, "sources": pose.labels(now),
                                  "sequence": receiver.clock.sequence,
                                  "closure_m": pose.residual_m, **wire}), flush=True)
                last_status = current_status
            if recorder and recorder.error != last_record_error:
                print(f"RECORD ERROR: {recorder.error}", file=sys.stderr, flush=True)
                last_record_error = recorder.error
            if isinstance(receiver, ReplayReceiver) and receiver.finished_event.is_set():
                break
            time.sleep(.02)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop_event.set()
        if receiver.is_alive():
            receiver.join(timeout=1.)
        latest = receiver.snapshot()
        with viewer.lock() if viewer else nullcontext():
            if latest is not None and latest[1] != applied:
                pose.apply(*latest)
        finished_at = receiver.now()
        elapsed = time.monotonic() - started if started is not None else 0.
        final_sources = pose.labels(finished_at)
        if recorder:
            recorder.finish()
            if recorder.error:
                print(f"RECORD ERROR: {recorder.error}", file=sys.stderr, flush=True)
        if viewer:
            viewer.close()
            # Join outside viewer.lock(): renderer destruction also acquires it.
            for thread in viewer_threads:
                thread.join()
        if recorder:
            info = recording_info(recorder, receiver.parser.accepted, elapsed, args.seconds)
            info.update(status="ERROR" if recorder.error else "SAVED",
                        directory=str(recorder.directory.resolve()), **wire_info(receiver.parser))
            print(json.dumps({"recording": info}), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps({
                "scope": "read-only pose/control telemetry; world translation fixed; mapping unverified on bench",
                **wire_info(receiver.parser),
                "accepted_frames": receiver.parser.accepted, "rejected_candidates": receiver.parser.rejected,
                "missing_submitted_frames": receiver.clock.missing,
                "restarts": receiver.clock.restarts, "duplicates": receiver.clock.duplicates,
                "out_of_order": receiver.clock.out_of_order,
                "serial_state": receiver.state, "sources": final_sources,
                "closure_m": pose.residual_m, "qpos": pose.data.qpos.tolist(),
                "recording": str(args.record) if recorder else None,
                "recording_complete": recorder.metadata["complete"] if recorder else None,
                "recording_error": recorder.error if recorder else None,
                "replay": str(args.replay) if args.replay else None,
            }, ensure_ascii=False, indent=2) + "\n")
    return int(bool((recorder and recorder.error) or getattr(receiver, "error", None)))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
