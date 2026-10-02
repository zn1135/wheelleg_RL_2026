#!/usr/bin/env python3
"""通过 HPI1 USB CDC 在上位机运行 chuanliantui 完整策略。"""

from __future__ import annotations

import argparse
from collections import deque
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import zlib

import numpy as np


HEADER = struct.Struct("<4sBBHI")
HELLO, ARM, ACTION, STOP = 1, 2, 3, 4
STATUS, INPUT, ACK = 0x81, 0x82, 0x83
STATUS_PAYLOAD = struct.Struct("<IBBHHHI64s")
INPUT_PAYLOAD = struct.Struct("<IIQ150f")
ACTION_PAYLOAD = struct.Struct("<II6f")
ACK_PAYLOAD = struct.Struct("<BII")
MAX_PAYLOAD = INPUT_PAYLOAD.size


def encode(kind, seq, payload=b""):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("HPI1 payload 超长")
    body = HEADER.pack(b"HPI1", 1, kind, len(payload), seq) + payload
    return body + struct.pack("<I", zlib.crc32(body))


class Decoder:
    def __init__(self):
        self.buffer = bytearray()
        self.crc_errors = 0
        self.pending = deque()

    def feed(self, chunk):
        self.buffer.extend(chunk)
        frames = []
        while len(self.buffer) >= 4:
            offset = self.buffer.find(b"HPI1")
            if offset < 0:
                del self.buffer[:max(0, len(self.buffer) - 3)]
                break
            if offset:
                del self.buffer[:offset]
            if len(self.buffer) < HEADER.size:
                break
            magic, version, kind, length, seq = HEADER.unpack_from(self.buffer)
            if magic != b"HPI1" or version != 1 or length > MAX_PAYLOAD:
                del self.buffer[0]
                continue
            total = HEADER.size + length + 4
            if len(self.buffer) < total:
                break
            expected = struct.unpack_from("<I", self.buffer, total - 4)[0]
            if zlib.crc32(self.buffer[:total - 4]) != expected:
                self.crc_errors += 1
                del self.buffer[0]
                continue
            frames.append((kind, seq, bytes(self.buffer[HEADER.size:total - 4])))
            del self.buffer[:total]
        return frames


def receive(port, decoder, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not decoder.pending:
            decoder.pending.extend(decoder.feed(port.read(4096)))
        while decoder.pending:
            yield decoder.pending.popleft()


def run(args):
    import isaacgym  # noqa: F401
    import torch
    import serial
    from mj_sim2sim_ct import load_policy

    if args.out.exists():
        raise ValueError("--out 必须为不存在的新目录")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    policy = load_policy(str(args.checkpoint), device="cpu")
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    decoder = Decoder()
    seq = 1
    session = 0
    armed = False
    samples = 0
    last_input_seq = 0
    args.out.mkdir(parents=True)
    log_file = args.out / "host_policy.csv"
    port = serial.Serial(port=None, baudrate=1152000, timeout=0.002,
                         write_timeout=0.02, rtscts=False, dsrdtr=False)
    port.dtr = False
    port.rts = False
    port.port = args.port
    port.open()
    report = {"checkpoint_sha256": checkpoint_hash, "port": args.port,
              "enabled_output": args.enable_output, "samples": 0}
    try:
        with log_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["input_seq", "mcu_sample_us", "host_recv_ns",
                             "inference_us", "host_send_ns",
                             *(f"action_{i}" for i in range(6))])
            port.write(encode(HELLO, seq))
            status = None
            for kind, frame_seq, payload in receive(port, decoder, 2.0):
                if kind == STATUS and frame_seq == seq and len(payload) == STATUS_PAYLOAD.size:
                    status = STATUS_PAYLOAD.unpack(payload)
                    break
                if kind == ACK and frame_seq == seq:
                    raise RuntimeError("H7 拒绝 HELLO；需左拨杆下位、电机失能、USB CDC 空闲")
            if status is None:
                raise TimeoutError("H7 未返回 HPI1 STATUS")
            session, locked, is_armed, obs_size, hist_size, action_size, fault, source_hash = status
            if not locked or is_armed or (obs_size, hist_size, action_size) != (25, 125, 6):
                raise RuntimeError("H7 接口状态或张量维度不匹配")
            report["session_id"] = session
            report["firmware_source_sha256"] = source_hash.decode("ascii", errors="replace")
            report["initial_fault"] = fault
            if fault:
                raise RuntimeError("H7 报告故障，拒绝启动")
            if not args.enable_output:
                report["result"] = "probe_only"
                return report
            print("HPI1 会话已建立。将遥控左拨杆拨上、右拨杆拨中；等待板端 ARM 条件满足。", flush=True)
            arm_deadline = time.monotonic() + args.arm_wait_s
            while time.monotonic() < arm_deadline:
                seq += 1
                port.write(encode(ARM, seq, struct.pack("<I", session)))
                for kind, frame_seq, payload in receive(port, decoder, 0.12):
                    if kind == ACK and frame_seq == seq and len(payload) == ACK_PAYLOAD.size:
                        code, ack_session, _ = ACK_PAYLOAD.unpack(payload)
                        if code == 0 and ack_session == session:
                            armed = True
                            break
                if armed:
                    break
            if not armed:
                raise TimeoutError("H7 未满足 ARM 条件；检查遥控、传感器、故障和 USB 状态")
            print("H7 已 ARM；开始接收策略输入并回送动作。", flush=True)
            started = time.monotonic()
            last_input_time = started
            while time.monotonic() - started < args.duration_s:
                chunk = port.read(4096)
                decoder.pending.extend(decoder.feed(chunk))
                while decoder.pending:
                    kind, frame_seq, payload = decoder.pending.popleft()
                    if kind == ACK and len(payload) == ACK_PAYLOAD.size:
                        code, ack_session, _ = ACK_PAYLOAD.unpack(payload)
                        if code or ack_session != session:
                            raise RuntimeError("H7 拒绝动作或会话改变：code={}".format(code))
                    if kind != INPUT:
                        continue
                    if len(payload) != INPUT_PAYLOAD.size:
                        raise ValueError("INPUT 长度错误")
                    received_ns = time.monotonic_ns()
                    unpacked = INPUT_PAYLOAD.unpack(payload)
                    input_session, input_seq, sample_us = unpacked[:3]
                    if input_session != session or input_seq <= last_input_seq:
                        raise RuntimeError("INPUT 会话或序号错误")
                    values = np.asarray(unpacked[3:], dtype=np.float32)
                    obs = values[:25]
                    history = values[25:]
                    if not np.isfinite(values).all() or not np.allclose(history[-25:], obs, atol=1e-5):
                        raise RuntimeError("INPUT 含无效值或历史末帧与当前观测不符")
                    with torch.inference_mode():
                        action, _ = policy.act_inference(
                            torch.from_numpy(obs.copy())[None],
                            torch.from_numpy(history.copy())[None])
                    action = action[0].numpy()
                    inference_us = (time.monotonic_ns() - received_ns) / 1000.0
                    if not np.isfinite(action).all():
                        raise RuntimeError("策略输出非有限数")
                    action = np.clip(action, -100.0, 100.0)
                    seq += 1
                    command = ACTION_PAYLOAD.pack(session, input_seq,
                                                  *(float(x) for x in action))
                    port.write(encode(ACTION, seq, command))
                    sent_ns = time.monotonic_ns()
                    writer.writerow([input_seq, sample_us, received_ns,
                                     inference_us, sent_ns, *map(float, action)])
                    samples += 1
                    last_input_seq = input_seq
                    last_input_time = time.monotonic()
                if time.monotonic() - last_input_time > 0.04:
                    raise TimeoutError("超过 40 ms 未收到 H7 新观测；板端应已超时锁止")
            report["result"] = "duration_complete"
    finally:
        if port.is_open:
            try:
                seq += 1
                port.write(encode(STOP, seq))
            except (OSError, serial.SerialException):
                pass
            port.close()
        report["samples"] = samples
        report["crc_errors"] = decoder.crc_errors
        (args.out / "host_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="H7 板载 USB CDC 串口")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="包含 encoder 的 chuanliantui model_*.pt")
    parser.add_argument("--out", type=Path, required=True, help="尚不存在的结果目录")
    parser.add_argument("--enable-output", action="store_true",
                        help="显式允许发送 ARM；默认仅做无出力接口探测")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--arm-wait-s", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("--duration-s 必须为正数")
    if not math.isfinite(args.arm_wait_s) or args.arm_wait_s <= 0:
        parser.error("--arm-wait-s 必须为正数")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
