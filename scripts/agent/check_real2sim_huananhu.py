#!/usr/bin/env python3
"""Check actual H7 mirror packing, damaged streams and measured-motor kinematics.

Uses a native C harness for the actual vofa_debug.c (only hardware/RTOS are stubs),
known XML/pose-bank poses, and a pseudo-terminal for the real serial reader. Optional
--build-h7 compiles/links both machine variants from the current eIDE source list,
writing all outputs outside deployment/H7_RL. No board is flashed or controlled.
"""
from __future__ import annotations

import argparse
import binascii
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import pty
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
H7 = ROOT / "deployment/H7_RL"
sys.path.insert(0, str(ROOT / "sim2sim"))
import real2sim_huananhu as mirror

STUB = r'''
#ifndef MIRROR_STUB
#define MIRROR_STUB
#include <stdint.h>
typedef struct { uint8_t seen; uint64_t published_ns; } feedback_meta_t;
typedef struct { float quat[4], gyro_rad_s[3]; uint8_t online; } imu_t;
typedef struct { float pos_continuous_rad[4], vel_rad_s[4]; uint8_t online[4]; } dm_t;
typedef struct { float angle_total_rad[2], vel_rad_s[2]; uint8_t online[2]; } dji_t;
typedef struct { struct { uint8_t valid; } output; } leg_t;
typedef struct {
    imu_t imu; struct { dm_t dm; dji_t dji; } motor;
    struct { uint8_t online; } rc; leg_t left,right;
    feedback_meta_t imu_meta,motor_meta; uint64_t captured_ns;
} robot_feedback_t;
extern struct robot_t { uint8_t motor_enabled,fallen; } robot_state;
extern volatile uint32_t ctrl_fault;
uint32_t HAL_GetTick(void);
uint8_t Robot_Feedback_Read(robot_feedback_t *out);
uint8_t Dm_Is_Enabled(uint8_t i);
uint8_t output_task_lqr_engaged(void);
uint8_t output_task_rl_engaged(void);
uint8_t Vofa_Send(const float *data,uint8_t n);
#define DM_MOTOR_NUM 4u
#define DJI_MOTOR_NUM 2u
#define DM_MOTOR_LEG_F_LFT 0u
#define DM_MOTOR_LEG_B_LFT 1u
#define DM_MOTOR_LEG_F_RGT 2u
#define DM_MOTOR_LEG_B_RGT 3u
#define MACHINE_DEFAULT 0u
#define VOFA_MAX_CH 48u
#define taskENTER_CRITICAL() ((void)0)
#define taskEXIT_CRITICAL() ((void)0)
#endif
'''


def c_packets(directory: Path):
    # Check the actual sender too: hardware stubs must not hide format drift.
    sender = (H7 / "imcalib/bsp/Vofa_send.c").read_text()
    header = (H7 / "imcalib/bsp/Vofa_send.h").read_text()
    channels = int(re.search(r"#define\s+VOFA_MAX_CH\s+(\d+)", header).group(1))
    tail_text = re.search(r"tail\[4\]\s*=\s*\{([^}]+)\}", sender).group(1)
    tail = bytes(int(value.strip(), 0) for value in tail_text.split(","))
    assert channels == 48 and channels * 4 + len(tail) == mirror.FRAME_BYTES
    assert tail == mirror.TAIL, "actual Vofa_Send tail differs from mirror protocol"
    stub = directory / "stubs"
    stub.mkdir()
    (stub / "robot_state.h").write_text(STUB)
    for name in ("vofa_debug.h", "dm.h", "Vofa_send.h", "dr16.h", "FreeRTOS.h", "task.h"):
        (stub / name).write_text('#include "robot_state.h"\n')
    probe = directory / "probe.c"
    probe.write_text(r'''
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "robot_state.h"
#include "''' + str(H7 / "imcalib/task/vofa_debug.c") + r'''"
struct robot_t robot_state = {0,0};
volatile uint32_t ctrl_fault = 0;
static robot_feedback_t source;
static uint32_t tick;
static unsigned attempts;
static uint8_t ready = 1;
static FILE *output;
uint32_t HAL_GetTick(void) { return tick; }
uint8_t Dm_Is_Enabled(uint8_t i) { return 0; }
uint8_t output_task_lqr_engaged(void) { return 0; }
uint8_t output_task_rl_engaged(void) { return 0; }
uint8_t Robot_Feedback_Read(robot_feedback_t *out) {
    *out = source;
    out->captured_ns = (uint64_t)tick * 1000000u;
    out->imu_meta.published_ns = out->captured_ns - 2000000u;
    out->motor_meta.published_ns = out->captured_ns - 3000000u;
    return 1;
}
uint8_t Vofa_Send(const float *data,uint8_t n) {
    const uint8_t tail[4] = {0,0,0x80,0x7f};
    assert(n == 48); attempts++;
    if (!ready) { return 0; }
    assert(fwrite(data,4,n,output)==n);
    assert(fwrite(tail,1,4,output)==4);
    return 1;
}
int main(int argc,char **argv) {
    float halves[2];
    assert(Mirror_Crc16((const uint8_t *)"123456789",9)==0x29b1);
    Mirror_U32(halves,0xffffffffu);
    assert(halves[0]==65535 && halves[1]==65535);
    output=fopen(argv[1],"wb"); assert(output);
    source.imu.online=source.rc.online=1;
    source.imu.quat[0]=1;
    source.imu_meta.seen=source.motor_meta.seen=1;
    for (int i=0;i<4;i++) {
        source.motor.dm.online[i]=1;
        source.motor.dm.pos_continuous_rad[i]=.1f*(i+1);
        source.motor.dm.vel_rad_s[i]=i+2;
    }
    for (int i=0;i<2;i++) {
        source.motor.dji.online[i]=1;
        source.motor.dji.angle_total_rad[i]=3.f+i;
        source.motor.dji.vel_rad_s[i]=5.f+i;
    }
    tick=1; Vofa_Debug_Send(); tick=9; Vofa_Debug_Send(); assert(attempts==0);
    tick=10; Vofa_Debug_Send(); tick=15; Vofa_Debug_Send(); assert(attempts==1);
    ready=0; tick=20; Vofa_Debug_Send(); assert(attempts==2);
    ready=1; tick=30; Vofa_Debug_Send();
    source.motor.dji.online[0]=0; tick=40; Vofa_Debug_Send();
    source.motor.dm.pos_continuous_rad[0]=INFINITY; tick=50; Vofa_Debug_Send();
    source.imu.online=0; tick=60; Vofa_Debug_Send();
    tick=0xfffffff0u; Vofa_Debug_Send(); tick=4; Vofa_Debug_Send();
    assert(attempts==8 && robot_state.motor_enabled==0 && ctrl_fault==0);
    fclose(output); return 0;
}
''')
    executable, data = directory / "probe", directory / "h7-frames.bin"
    subprocess.run(["cc", "-std=c11", "-Wall", "-I", str(stub), str(probe), "-lm", "-o", str(executable)], check=True)
    subprocess.run([str(executable), str(data)], check=True)
    raw = data.read_bytes()
    frames = mirror.FrameParser().feed(raw)
    assert len(frames) == 7
    assert [mirror.read_u32(f[6:8]) for f in frames] == list(range(1, 8))
    assert [mirror.read_u32(f[8:10]) for f in frames] == [10, 30, 40, 50, 60, 0xfffffff0, 4]
    assert frames[0][29] == 127 and np.array_equal(frames[0][30:32], [2, 3])
    assert np.allclose(frames[0][10:14], [.1, .2, .3, .4])
    assert np.array_equal(frames[0][14:22], [2, 3, 4, 5, 3, 4, 5, 6])
    assert np.array_equal(frames[0][22:29], [1, 0, 0, 0, 0, 0, 0])
    assert frames[2][29] == 95 and frames[2][18] == 0
    assert frames[3][29] == 93 and frames[3][10] == 0
    assert frames[4][29] == 92 and np.all(frames[4][22:29] == 0)
    assert np.all(frames[0][32:47] == 0)
    return raw[:196], {"actual_c_frames": len(frames), "cadence_busy_wrap_invalid_sources": True,
                      "actual_sender_channel_count_and_tail": True}


def encode(frame):
    payload = np.asarray(frame[:47], dtype="<f4").tobytes()
    return payload + struct.pack("<f", binascii.crc_hqx(payload, 0xffff)) + mirror.TAIL


def check_stream(raw):
    frame = mirror.decode_frame(raw)
    damaged = bytearray(raw)
    damaged[40:44] = struct.pack("<f", .11)  # finite/plausible, but a bad CRC
    false_tail = frame.copy()
    false_tail[10] = np.inf
    false_tail[29] = int(false_tail[29]) & ~2
    old = frame.copy()
    old[3] = 0
    wrong_machine = frame.copy()
    wrong_machine[5] = 1
    stream = b"noise" + bytes(damaged) + encode(old) + encode(false_tail) + encode(wrong_machine) + raw
    parser = mirror.FrameParser()
    received = []
    for i in range(0, len(stream), 7):
        received += parser.feed(stream[i:i+7])
    assert len(received) == 2 and np.isinf(received[0][10])
    assert np.array_equal(received[1], frame)
    parser.feed(b"x" * 10000)
    assert len(parser.buffer) <= 195
    clock = mirror.FrameClock()
    def f(seq, tick):
        v = frame.copy()
        v[6:10] = (seq & 65535, seq >> 16, tick & 65535, tick >> 16)
        return v
    assert clock.accept(f(0xfffffffe, 0xfffffff0))
    assert clock.accept(f(0xffffffff, 0xfffffff9))
    assert clock.accept(f(1, 4)) and clock.missing == 1
    assert not clock.accept(f(1, 4)) and clock.duplicates == 1
    assert not clock.accept(f(0, 5)) and clock.out_of_order == 1
    assert clock.accept(f(1, 0)) and clock.restarts == 1
    clock = mirror.FrameClock()
    assert clock.accept(f(100000, 0x90000000))
    assert clock.accept(f(1, 10)) and clock.restarts == 1
    return {"damaged_and_legacy_rejected": True, "false_tail_resynchronized": True,
            "fragmentation_sequence_wrap_restart": True}


def frame_for_pose(pose, qpos):
    frame = np.zeros(48, dtype=np.float32)
    frame[22:26] = qpos[3:7]
    native = np.concatenate([qpos[q] for q in pose.drive_q])
    frame[10:14] = native / pose.sign[:4] + pose.zero
    frame[18:20] = qpos[pose.wheel_q] / pose.sign[4:]
    frame[29] = 127
    return frame


def check_geometry():
    pose = mirror.MirrorPose()
    assert (np.linalg.norm(pose.data.light_xdir, axis=1) > .9).all()
    fixed_translation = pose.data.qpos[:3].copy()
    poses = np.load(mirror.MODEL.with_name("huananhu_pose_bank.npz"), allow_pickle=False)
    joints = [("lf0", "lf1", "lfwheel", "left_spring1_joint", "left_spring2_joint", "lf00", "lf01", "lf02", "lf03"),
              ("rf0", "rf1", "rfwheel", "right_spring1_joint", "right_spring2_joint", "rf00", "rf01", "rf02", "rf03")]
    count, maximum, wheel_error, times, cold_error = 0, 0., 0., [], 0.
    for mode in ("normal", "positive", "negative"):
        for side, name in enumerate(("left", "right")):
            for index in np.linspace(0, 1023, 16, dtype=int):
                pose.data.qpos[:] = pose.model.qpos0
                addresses = [pose.model.joint(n).qposadr[0] for n in joints[side]]
                pose.data.qpos[addresses] = poses[f"{mode}_{name}_pos"][index]
                mujoco.mj_kinematics(pose.model, pose.data)
                expected_wheel = pose.data.body(mirror.WHEELS[side]).xpos.copy()
                measured = pose.data.qpos[pose.drive_q[side]].copy()
                expected_qpos = pose.data.qpos.copy()
                # First telemetry may arrive while folded: test starting from XML,
                # without supplying the hidden passive angles to the solver.
                pose.data.qpos[:] = pose.model.qpos0
                assert pose._solve_leg(side, measured), ("cold", mode, side, index)
                cold_error = max(cold_error, float(np.linalg.norm(pose.data.body(mirror.WHEELS[side]).xpos - expected_wheel)))
                pose.data.qpos[:] = expected_qpos
                pose.data.qpos[pose.passive_q[side]] += .01
                started = time.perf_counter()
                assert pose._solve_leg(side, measured), (mode, side, index, pose.residual_m)
                times.append(time.perf_counter() - started)
                assert np.array_equal(pose.data.qpos[pose.drive_q[side]], measured)
                maximum = max(maximum, pose.residual_m[side])
                wheel_error = max(wheel_error, float(np.linalg.norm(pose.data.body(mirror.WHEELS[side]).xpos - expected_wheel)))
                count += 1
    assert wheel_error < .001 and cold_error < .001, (wheel_error, cold_error)
    pose.data.qpos[:] = pose.model.qpos0
    frame = frame_for_pose(pose, pose.data.qpos)
    frame[22:26] = [np.cos(.2), np.sin(.2), 0, 0]
    frame[18:20] = [1., -2.]
    pose.apply(frame, 1.)
    assert pose.labels(1.) == ["OK"] * 5
    assert np.array_equal(pose.data.qpos[:3], fixed_translation)
    assert np.allclose(pose.data.qpos[3:7], frame[22:26], atol=1e-7)
    assert np.allclose(pose.data.qpos[pose.wheel_q], pose.sign[4:] * frame[18:20])
    previous = pose.data.qpos.copy()
    frame[29] = 127 & ~(1 << 5)
    frame[18:20] = [9., -3.]
    pose.apply(frame, 1.01)
    assert pose.status[3] == "INVALID WHEEL" and pose.status[4] == "OK"
    assert pose.data.qpos[pose.wheel_q[0]] == previous[pose.wheel_q[0]]
    assert pose.data.qpos[pose.wheel_q[1]] != previous[pose.wheel_q[1]]
    frame[22:26] = 0
    pose.apply(frame, 1.02)
    assert pose.status[0] == "INVALID IMU" and np.array_equal(pose.data.qpos[3:7], previous[3:7])
    before = pose.data.qpos.copy()
    frame[10:12] = [np.nan, np.inf]
    pose.apply(frame, 1.03)
    assert pose.status[1] == "INVALID FEEDBACK"
    assert np.array_equal(pose.data.qpos[pose.drive_q[0]], before[pose.drive_q[0]])
    frame = frame_for_pose(pose, pose.data.qpos.copy())
    frame[31] = 201
    frame[10:14] += .5
    frame[18:20] += 3
    saved_motor = pose.data.qpos[7:].copy()
    pose.apply(frame, 1.04)
    assert pose.status[1:] == ["STALE MOTOR SOURCE"] * 4
    assert np.array_equal(pose.data.qpos[7:], saved_motor)
    frame = frame_for_pose(pose, pose.data.qpos.copy())
    frame[30] = 201
    frame[22:26] = [1, 0, 0, 0]
    saved_quat = pose.data.qpos[3:7].copy()
    pose.apply(frame, 1.05)
    assert pose.status[0] == "STALE IMU SOURCE" and np.array_equal(pose.data.qpos[3:7], saved_quat)
    assert pose.stale(1.251) and pose.labels(1.251) == ["STALE"] * 5
    before = pose.data.qpos.copy()
    original_solver = mirror.least_squares
    try:
        mirror.least_squares = lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad initial pose"))
        assert not pose._solve_leg(0, [1., 1.])
        assert np.array_equal(pose.data.qpos[pose.drive_q[0]], before[pose.drive_q[0]])
        from types import SimpleNamespace
        mirror.least_squares = lambda residual, initial, **kw: SimpleNamespace(x=initial)
        side_indices = np.concatenate((pose.drive_q[0], pose.passive_q[0], pose.spring_q[0]))
        saved = pose.data.qpos[side_indices].copy()
        assert not pose._solve_leg(0, saved[:2] + [.5, -.3])
        assert np.array_equal(pose.data.qpos[side_indices], saved)
    finally:
        mirror.least_squares = original_solver
    # Reproduce a closed but mechanically impossible branch from a bad initial guess.
    pose.data.qpos[:] = pose.model.qpos0
    pose.data.qpos[pose.passive_q[0][0]] = -3.
    saved = pose.data.qpos.copy()
    assert not pose._solve_leg(0, pose.model.qpos0[pose.drive_q[0]])
    assert pose.status[1] == "INVALID JOINT RANGE" and pose.residual_m[0] < .001
    assert np.array_equal(pose.data.qpos, saved)
    return {"known_model_leg_poses": count, "max_closure_m": maximum,
            "max_reconstructed_wheel_error_m": wheel_error,
            "cold_start_poses": count, "max_cold_start_wheel_error_m": cold_error,
            "solve_p95_ms": float(np.percentile(times, 95) * 1000),
            "individual_source_failure_and_200ms_stale": True,
            "publication_stop_with_live_uart": True,
            "closed_but_out_of_range_branch_rejected": True,
            "scope": "model kinematics; assembly branch, physical signs and pose require bench confirmation"}


def check_serial(raw):
    master, slave = pty.openpty()
    receiver = mirror.SerialReceiver(os.ttyname(slave), 1152000)
    receiver.start()
    try:
        deadline = time.monotonic() + 3
        while receiver.state != "CONNECTED" and time.monotonic() < deadline:
            time.sleep(.01)
        assert receiver.state == "CONNECTED", receiver.state
        os.write(master, raw[:37])
        time.sleep(.04)
        os.write(master, raw[37:])
        while receiver.snapshot() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert receiver.snapshot() is not None
        assert mirror.read_u32(receiver.snapshot()[0][6:8]) == 1
        os.set_blocking(master, False)
        try:
            assert not os.read(master, 1024), "mirror wrote to H7"
        except BlockingIOError:
            pass
        os.close(master)
        master = None
        deadline = time.monotonic() + 1
        while not receiver.state.startswith("DISCONNECTED") and time.monotonic() < deadline:
            time.sleep(.01)
        assert receiver.state.startswith("DISCONNECTED"), receiver.state
        return {"actual_serial_reader_fragmentation_disconnect": True, "no_outbound_bytes": True}
    finally:
        receiver.stop_event.set()
        receiver.join(timeout=1.)
        os.close(slave)
        if master is not None:
            os.close(master)


def build_h7(directory):
    import yaml
    config = yaml.safe_load((H7 / ".eide/eide.yml").read_text())
    sources = set()
    libraries = set()
    missing_references = []
    def visit(folder):
        for item in folder["files"]:
            path = H7 / item["path"]
            if not path.is_file():
                if path.suffix.lower() in (".c", ".s", ".a"):
                    missing_references.append(item["path"])
                continue
            if path.suffix.lower() in (".c", ".s"):
                sources.add(path)
            elif path.suffix == ".a":
                libraries.add(path)
        for child in folder["folders"]:
            visit(child)
    visit(config["virtualFolder"])
    for name in config["srcDirs"]:
        sources.update((H7 / name).rglob("*.c"))
    target = config["targets"]["CtrBoard-H7_ALL"]
    attrs = target["cppPreprocessAttrs"]
    gcc = os.environ.get("ARM_CC") or shutil.which("arm-none-eabi-gcc")
    assert gcc, "ARM GCC missing"
    common = [gcc, "-mcpu=cortex-m7", "-mthumb", "-mfpu=fpv5-d16", "-mfloat-abi=hard",
              "--specs=nano.specs", "--specs=nosys.specs", "-g3", "-O0", "-Wall",
              "-ffunction-sections", "-fdata-sections"]
    common += ["-D" + d for d in attrs["defineList"]]
    common += ["-I" + str(H7 / inc) for inc in attrs["incList"]]
    summaries = []
    for machine in (0, 1):
        output = directory / f"h7-machine-{machine}"
        output.mkdir(parents=True, exist_ok=True)
        commands = []
        for i, source in enumerate(sorted(sources)):
            obj = output / f"{i:03}-{source.stem}.o"
            language = ["-x", "assembler-with-cpp"] if source.suffix.lower() == ".s" else ["-std=gnu11"]
            commands.append(common + [f"-DMACHINE_DEFAULT={machine}"] + language + ["-c", str(source), "-o", str(obj)])
        def compile_one(command):
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            return result.returncode, result.stdout
        with ThreadPoolExecutor(max_workers=8) as pool:
            compiled = list(pool.map(compile_one, commands))
        (output / "compile.log").write_text("\n".join(r[1] for r in compiled))
        (output / "commands.json").write_text(json.dumps(commands, indent=2))
        assert all(code == 0 for code, _ in compiled), f"compile failed: {output / 'compile.log'}"
        elf = output / "firmware.elf"
        link = common[:5] + ["--specs=nano.specs", "--specs=nosys.specs", "-nostartfiles",
                "-Wl,--gc-sections", "-T" + str(H7 / target["toolchainConfigMap"]["GCC"]["scatterFilePath"]),
                "-Wl,-Map," + str(output / "firmware.map")]
        link += [command[-1] for command in commands]
        link += ["-Wl,--start-group"] + [str(p) for p in sorted(libraries)] + ["-lc", "-lm", "-Wl,--end-group", "-o", str(elf)]
        result = subprocess.run(link, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / "link.log").write_text(result.stdout)
        (output / "link-command.json").write_text(json.dumps(link, indent=2))
        assert result.returncode == 0, f"link failed: {output / 'link.log'}"
        summaries.append({"machine": machine, "translation_units": len(commands), "linked": True,
                          "absent_eide_virtual_references": missing_references,
                          "elf": str(elf), "warning_log": str(output / "compile.log")})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-h7", action="store_true")
    parser.add_argument("--build-only", action="store_true", help="ARM compile/link only; no native C compiler needed")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if not args.build_only:
        with tempfile.TemporaryDirectory(prefix="real2sim-check-") as temporary:
            raw, c_result = c_packets(Path(temporary))
            results = {"actual_h7_packer": c_result, "stream": check_stream(raw),
                       "kinematics": check_geometry(), "serial": check_serial(raw)}
    if args.build_h7 or args.build_only:
        results["h7_builds"] = build_h7(args.out.parent)
    report = {"ok": True, "scope": "software and model checks; no H7 hardware/axis/branch acceptance", "results": results}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
