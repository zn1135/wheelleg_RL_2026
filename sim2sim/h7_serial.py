#!/usr/bin/env python3
"""Check H7/Jetson UART in both directions without policy or motor commands.

Requires only pyserial. Receive the existing 196-byte mirror, send a CRC-protected
probe at at most 10 Hz, and match its sequence/token in the mirror acknowledgement.
Run this OR the read-only MuJoCo mirror; both programs cannot own the same port.
"""
from __future__ import annotations

import argparse
import binascii
import json
import math
from pathlib import Path
import secrets
import struct
import time

TAIL = bytes.fromhex("0000807f")
FRAME_BYTES = 196
PROBE_PERIOD_S = .1
ACK_TIMEOUT_S = .25


def read_u32(values) -> int:
    if any(not math.isfinite(v) or v != int(v) or not 0 <= v <= 65535 for v in values):
        raise ValueError("invalid uint32 halves")
    return int(values[0]) | (int(values[1]) << 16)


def encode_probe(sequence: int, token: int) -> bytes:
    payload = struct.pack("<4sII", b"J7P1", sequence, token)
    return payload + struct.pack("<H", binascii.crc_hqx(payload, 0xffff))


def decode_mirror(raw: bytes) -> tuple[float, ...]:
    if len(raw) != FRAME_BYTES or raw[-4:] != TAIL:
        raise ValueError("mirror length/tail")
    values = struct.unpack("<48f", raw[:192])
    if values[3:5] != (7101., 1.) or values[5] not in (0., 1.):
        raise ValueError("not mirror v1 / unsupported machine")
    if values[47] != binascii.crc_hqx(raw[:188], 0xffff):
        raise ValueError("mirror CRC mismatch")
    read_u32(values[6:8])
    read_u32(values[8:10])
    return values


class FrameParser:
    """Scan fixed-length candidates, including tails embedded inside the payload."""

    def __init__(self):
        self.buffer = bytearray()
        self.accepted = self.rejected = 0

    def feed(self, chunk: bytes) -> list[tuple[float, ...]]:
        self.buffer.extend(chunk)
        frames, search = [], 0
        while True:
            end = self.buffer.find(TAIL, search)
            if end < 0:
                break
            stop = end + len(TAIL)
            if stop >= FRAME_BYTES:
                try:
                    frame = decode_mirror(bytes(self.buffer[stop - FRAME_BYTES:stop]))
                except ValueError:
                    self.rejected += 1
                else:
                    frames.append(frame)
                    self.accepted += 1
                    del self.buffer[:stop]
                    search = 0
                    continue
            search = end + 1
        if len(self.buffer) >= FRAME_BYTES:
            del self.buffer[:-(FRAME_BYTES - 1)]
        return frames


def check_link(connection, seconds: float) -> dict:
    parser = FrameParser()
    started = time.monotonic()
    next_probe, next_status = started, started
    sequence = sent = acknowledged = timeouts = 0
    pending = None
    rtt_ms = []
    sequence_changes = 0
    previous_sequence = None
    protocol_seen = False
    last_received = None
    while time.monotonic() - started < seconds:
        chunk = connection.read(min(connection.in_waiting or 1, 4096))
        now = time.monotonic()
        # Expire before matching so a late ACK cannot clear an overdue request.
        if pending is not None and now - pending[2] >= ACK_TIMEOUT_S:
            timeouts += 1
            pending = None
        for frame in parser.feed(chunk):
            last_received = now
            mirror_sequence = read_u32(frame[6:8])
            if mirror_sequence != previous_sequence:
                sequence_changes += 1
                previous_sequence = mirror_sequence
            if frame[32] != 1.:
                continue
            protocol_seen = True
            try:
                ack_sequence = read_u32(frame[33:35])
                ack_token = read_u32(frame[35:37])
                read_u32(frame[37:39])
            except ValueError:
                continue
            # Match both fields so queued telemetry or another session cannot
            # acknowledge the current request. All RTT times use the host clock.
            if pending is not None and (ack_sequence, ack_token) == pending[:2]:
                rtt_ms.append((now - pending[2]) * 1000.)
                acknowledged += 1
                pending = None
        # Only send after seeing firmware's probe extension; old mirrors remain
        # receive-only. One outstanding request bounds H7's input traffic.
        if protocol_seen and pending is None and now >= next_probe and seconds - (now - started) >= ACK_TIMEOUT_S:
            sequence = (sequence + 1) & 0xffffffff
            token = secrets.randbits(32)
            packet = encode_probe(sequence, token)
            before_write = time.monotonic()
            if connection.write(packet) != len(packet):
                raise OSError("incomplete probe write")
            pending = (sequence, token, before_write)
            sent += 1
            next_probe = before_write + PROBE_PERIOD_S
        if now >= next_status:
            print(json.dumps({"mirror_frames": parser.accepted, "probe_sent": sent,
                              "probe_ack": acknowledged, "timeouts": timeouts,
                              "probe_firmware": protocol_seen}), flush=True)
            next_status = now + 1.
    # A pending request at shutdown has no verified acknowledgement.
    if pending is not None:
        timeouts += 1
    return {"ok": bool(acknowledged and not timeouts and last_received is not None
                       and time.monotonic() - last_received < ACK_TIMEOUT_S),
            "scope": "UART probes only; no inference or motor commands",
            "seconds": time.monotonic() - started,
            "mirror_frames": parser.accepted, "mirror_sequence_changes": sequence_changes,
            "rejected_candidates": parser.rejected, "probe_firmware": protocol_seen,
            "probe_sent": sent, "probe_ack": acknowledged, "timeouts": timeouts,
            "rtt_ms_mean": sum(rtt_ms) / len(rtt_ms) if rtt_ms else None,
            "rtt_ms_max": max(rtt_ms) if rtt_ms else None}


def main() -> int:
    import serial
    from serial.tools import list_ports

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Jetson FUNC_UART or USB-to-UART device")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--baud", type=int, default=1152000)
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.list_ports:
        for port in list_ports.comports():
            print(f"{port.device}: {port.description}")
        return 0
    if not args.port:
        parser.error("--port is required (or use --list-ports)")
    if not math.isfinite(args.seconds) or args.seconds < 1.:
        parser.error("--seconds must be finite and at least 1")
    if args.report and args.report.exists():
        raise FileExistsError(args.report)
    try:
        with serial.Serial(port=None, baudrate=args.baud, timeout=.01, write_timeout=.1, exclusive=True) as connection:
            connection.dtr = connection.rts = False
            connection.port = args.port
            connection.open()
            # Discard telemetry queued before this process owned the link.
            connection.reset_input_buffer()
            report = check_link(connection, args.seconds)
    except (serial.SerialException, OSError) as exc:
        report = {"ok": False, "scope": "UART probes only; no motor commands", "error": str(exc)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        with args.report.open("x") as output:
            output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
