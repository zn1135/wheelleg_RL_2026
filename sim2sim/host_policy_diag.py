#!/usr/bin/env python3
"""2701 USB disabled policy diagnostic (A5 5A, never ARM or torque).

Run with the project's Isaac Gym Python 3.8 environment. --verify rechecks an
existing records.jsonl without opening any device. Records are kept in memory
until STOP has completed, keeping filesystem writes out of the 10 ms loop.
This verifies only disabled data flow, not actuator execution or stability.
"""
import argparse
from collections import deque
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import select
import struct
import subprocess
import sys
import termios
import time
import tty
import zlib

MAGIC = b"\xa5\x5a"
WIRE_HEADER = struct.Struct("<BBHI")
HEADER = struct.Struct("<16I")
FLOATS = struct.Struct("<150f")
ACTION = struct.Struct("<II6f")
MAX_PAYLOAD = 768
STATUS, REMOTE_QUERY = 1, 5
DIAG_START, DIAG_ACTION, DIAG_STOP, DIAG_QUERY = 0x10, 0x11, 0x12, 0x13
LEGACY_STATUS, REMOTE_STATUS, DIAG_SAMPLE, DIAG_STATUS = 0x81, 0x82, 0x90, 0x91
OFF, RUNNING, FAULT, NO_SEQ = 0, 1, 2, 0xffffffff
EXPECTED_SHA256 = "8070977e19114cb9bbf358a090bd78cdbd1be568a76e02d417c369e828bc8f36"
HEADER_NAMES = ("layout", "session", "sample_seq", "sample_ms", "tx_ms",
                "applied_action_source_seq", "applied_action_rx_ms",
                "applied_action_latency_ms", "state", "fault", "online_mask",
                "dm_enabled_mask", "period_ms", "reset_count", "accepted_actions",
                "rejected_actions")
FAULT_NAMES = {0: "NONE", 1: "SOURCE", 2: "DEADLINE", 3: "SESSION", 4: "SEQUENCE",
               5: "NONFINITE", 6: "TX_BACKPRESSURE", 7: "USB", 8: "REMOTE", 9: "START_REJECTED"}


def encode(kind, seq, payload=b""):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("payload exceeds 768 bytes")
    body = WIRE_HEADER.pack(1, kind, len(payload), seq) + payload
    return MAGIC + body + struct.pack("<I", zlib.crc32(body) & NO_SEQ)


class Decoder:
    """Retain split magic/frames; recover after CRC or header corruption."""
    def __init__(self):
        self.buffer = bytearray()
        self.crc_errors = 0
        self.header_errors = 0
        self.discarded_bytes = 0

    def feed(self, chunk):
        self.buffer.extend(chunk)
        result = []
        while self.buffer:
            offset = self.buffer.find(MAGIC)
            if offset < 0:
                keep = 1 if self.buffer[-1:] == MAGIC[:1] else 0
                self.discarded_bytes += len(self.buffer) - keep
                self.buffer[:] = self.buffer[-1:] if keep else b""
                break
            if offset:
                self.discarded_bytes += offset
                del self.buffer[:offset]
            if len(self.buffer) < 10:
                break
            version, kind, length, seq = WIRE_HEADER.unpack_from(self.buffer, 2)
            if version != 1 or length > MAX_PAYLOAD:
                self.header_errors += 1
                del self.buffer[0]
                continue
            total = 14 + length
            if len(self.buffer) < total:
                break
            expected = struct.unpack_from("<I", self.buffer, total - 4)[0]
            if zlib.crc32(self.buffer[2:total - 4]) & NO_SEQ != expected:
                self.crc_errors += 1
                del self.buffer[0]
                continue
            result.append((kind, seq, bytes(self.buffer[10:total - 4])))
            del self.buffer[:total]
        return result


def unpack_header(payload):
    if len(payload) != 64:
        raise ValueError("DIAG_STATUS header must be 64 bytes")
    return dict(zip(HEADER_NAMES, HEADER.unpack(payload)))


def unpack_sample(seq, payload):
    if len(payload) != 664:
        raise ValueError("DIAG_SAMPLE must be 664 bytes")
    values = FLOATS.unpack_from(payload, 64)
    return {"event": "sample", "outer_seq": seq,
            "header": unpack_header(payload[:64]), "obs": list(values[:25]),
            "history": list(values[25:])}


def unpack_legacy(payload):
    if len(payload) != 132:
        raise ValueError("legacy STATUS must be 132 bytes")
    uptime, flags, online, enabled, machine, reason, last_seq = struct.unpack_from("<IIBBBBI", payload)
    return dict(uptime=uptime, flags=flags, online=online, enabled=enabled,
                machine=machine, reason=reason, last_seq=last_seq,
                target=list(struct.unpack_from("<6f", payload, 108)))


def unpack_remote(payload):
    if len(payload) != 64:
        raise ValueError("REMOTE_STATUS must be 64 bytes")
    names = ("layout", "uptime", "last_rx_ms", "valid_frames", "invalid_frames",
             "transport_errors", "fault_generation", "disarm_generation")
    value = dict(zip(names, struct.unpack_from("<8I", payload)))
    value.update(zip(("online", "left_switch", "right_switch", "permit", "reset_seen",
                      "armed", "has_command", "enabled_mask"), struct.unpack_from("<8B", payload, 32)))
    value["usb_connected"] = payload[51]
    value["motors_online"] = struct.unpack_from("<I", payload, 52)[0]
    return value


def require_disabled(status):
    if status["flags"] & 3 or status["enabled"] or any(x != 0.0 for x in status["target"]):
        raise RuntimeError("legacy STATUS is not disabled with six zero targets")


def require_fresh_start(started, before):
    # The first observation tick can occur before START's status is transmitted.
    # Both an empty epoch and sample zero are valid; SAMPLE itself must still
    # start at zero and prove zero action plus five identical initial frames.
    if (started["reset_count"] != ((before["reset_count"] + 1) & NO_SEQ) or
            started["sample_seq"] not in (NO_SEQ, 0) or
            started["accepted_actions"] or started["rejected_actions"] or
            started["dm_enabled_mask"] or started["applied_action_source_seq"] != NO_SEQ or
            started["applied_action_rx_ms"] or started["applied_action_latency_ms"]):
        raise RuntimeError("DIAG_START did not reset sample/action state")


def bits(values):
    return struct.pack("<%df" % len(values), *values)


def finite(values, length):
    return len(values) == length and all(math.isfinite(x) for x in values)


class RecordVerifier:
    """Online and offline verification use the same independent FIFO model."""
    def __init__(self, session, reset_count):
        self.session = session
        self.reset_count = reset_count
        self.previous = None
        self.action = None
        self.samples = 0
        self.errors = []
        self.intervals = []
        self.latencies = []
        self.consecutive = 0
        self.max_consecutive = 0
        self.fault = None

    def check_sample(self, record):
        h, obs, history = record["header"], record["obs"], record["history"]
        checks = {}
        checks["dimensions_finite"] = finite(obs, 25) and finite(history, 125)
        checks["running_disabled"] = h["state"] == RUNNING and h["fault"] == 0 and h["dm_enabled_mask"] == 0
        checks["layout_session_sources"] = (h["layout"] == 2701 and h["session"] == self.session and
            h["reset_count"] == self.reset_count and h["period_ms"] == 10 and h["online_mask"] == 63)
        checks["outer_seq"] = record["outer_seq"] == h["sample_seq"]
        checks["not_after_fault"] = self.fault is None
        checks["no_rejected_actions"] = h["rejected_actions"] == 0
        checks["tx_before_deadline"] = ((h["tx_ms"] - h["sample_ms"]) & NO_SEQ) < 10
        if self.previous is None:
            checks["sequence"] = h["sample_seq"] == 0
            checks["initial_action"] = bits(obs[19:25]) == bits([0.0] * 6)
            checks["initial_ack"] = (h["applied_action_source_seq"] == NO_SEQ and
                h["applied_action_rx_ms"] == 0 and h["applied_action_latency_ms"] == 0 and h["accepted_actions"] == 0)
            expected_history = obs * 5
        else:
            prev = self.previous["header"]
            delta = (h["sample_ms"] - prev["sample_ms"]) & NO_SEQ
            self.intervals.append(delta)
            checks["sequence"] = h["sample_seq"] == ((prev["sample_seq"] + 1) & NO_SEQ)
            checks["cadence_10ms"] = delta == 10
            checks["last_action_exact"] = self.action is not None and bits(obs[19:25]) == bits(self.action)
            checks["action_source"] = h["applied_action_source_seq"] == prev["sample_seq"]
            latency = (h["applied_action_rx_ms"] - prev["sample_ms"]) & NO_SEQ
            self.latencies.append(latency)
            checks["action_deadline"] = latency < 10 and latency == h["applied_action_latency_ms"]
            checks["accepted_count"] = h["accepted_actions"] == h["sample_seq"]
            expected_history = self.previous["history"][25:] + obs
        checks["history_exact"] = checks["dimensions_finite"] and bits(history) == bits(expected_history)
        bad = [name for name, ok in checks.items() if not ok]
        self.errors.extend("sample %s: %s" % (h["sample_seq"], name) for name in bad)
        self.consecutive = self.consecutive + 1 if not bad else 0
        self.max_consecutive = max(self.max_consecutive, self.consecutive)
        self.previous = record
        self.action = None
        self.samples += 1
        return checks

    def record_action(self, record):
        raw, published = record.get("raw_actor", []), record.get("published_action", [])
        if not finite(raw, 6) or not finite(published, 6) or not finite(record.get("latent", []), 3):
            self.errors.append("sample %s: nonfinite/dimension actor, published action or latent" % record["header"]["sample_seq"])
            return
        expected = [max(-100.0, min(100.0, x)) for x in raw]
        if bits(expected) != bits(published):
            self.errors.append("published action differs from clipped float32 actor")
        if not record.get("action_dropped", False):
            self.action = published

    def record_fault(self, header):
        if header["session"] != self.session or header["state"] != FAULT or header["dm_enabled_mask"]:
            self.errors.append("invalid FAULT session/state/disabled mask")
        if self.previous is not None and header["sample_seq"] != self.previous["header"]["sample_seq"]:
            self.errors.append("FAULT did not preserve latest sample identity")
        self.fault = header
        self.previous = None
        self.action = None
        self.consecutive = 0

    def summary(self):
        return {"samples": self.samples, "max_consecutive_valid_frames": self.max_consecutive,
                "cadence_100hz_verified": self.max_consecutive >= 100 and not self.errors,
                "board_effective_hz": (1000.0 * len(self.intervals) / sum(self.intervals)) if self.intervals and sum(self.intervals) else None,
                "board_interval_ms_min": min(self.intervals) if self.intervals else None,
                "board_interval_ms_max": max(self.intervals) if self.intervals else None,
                "board_action_latency_ms_max": max(self.latencies) if self.latencies else None,
                "errors": self.errors, "fault": self.fault}


def verify_records(records, expected_fault_seq=None):
    """Recompute evidence; recorded 'checks' and summary verdicts are not trusted."""
    verifier = None
    errors, epochs = [], []
    seen_sessions = set()
    stopped = False
    disabled = False
    dropped = []
    last_reset = None
    for record in records:
        event = record["event"]
        if event == "start":
            h = record["header"]
            if verifier is not None:
                epochs.append(verifier.summary())
                if not stopped:
                    errors.append("restart without confirmed STOP")
            if (not h["session"] or h["session"] in seen_sessions or h["state"] != RUNNING or
                    h["fault"] or h["dm_enabled_mask"] or h["layout"] != 2701):
                errors.append("invalid/new-session START")
            if last_reset is not None and h["reset_count"] != ((last_reset + 1) & NO_SEQ):
                errors.append("reset_count did not advance on restart")
            seen_sessions.add(h["session"])
            last_reset = h["reset_count"]
            verifier = RecordVerifier(h["session"], h["reset_count"])
            stopped = disabled = False
        elif event == "sample":
            if verifier is None or stopped:
                errors.append("sample outside active epoch")
                continue
            verifier.check_sample(record)
            verifier.record_action(record)
            if record.get("action_dropped"):
                dropped.append(record["header"]["sample_seq"])
        elif event == "fault":
            if verifier is None:
                errors.append("FAULT without START")
            else:
                verifier.record_fault(record["header"])
        elif event == "stop":
            h = record["header"]
            stopped = (verifier is not None and h["session"] == verifier.session and h["state"] == OFF and
                       h["fault"] == 0 and h["dm_enabled_mask"] == 0)
            if not stopped:
                errors.append("STOP not confirmed OFF for current session")
            elif verifier.fault is None and h["accepted_actions"] != verifier.samples:
                errors.append("final action was not acknowledged before STOP")
        elif event == "final_legacy":
            try:
                require_disabled(record["status"])
                disabled = True
            except RuntimeError as exc:
                errors.append(str(exc))
        elif event in ("unprocessed_sample", "error"):
            errors.append(record.get("message", "unprocessed sample before STOP"))
    if verifier:
        epochs.append(verifier.summary())
    else:
        errors.append("no accepted START")
    for epoch in epochs:
        errors.extend(epoch["errors"])
    if not stopped or not disabled:
        errors.append("missing confirmed STOP/disabled final STATUS")
    if expected_fault_seq is None:
        if dropped or any(e["fault"] for e in epochs):
            errors.append("unexpected dropped action or board FAULT")
        if not epochs or any(e["max_consecutive_valid_frames"] < 100 for e in epochs):
            errors.append("fewer than 100 consecutive verified frames")
        result = "pass" if not errors else "fail"
    else:
        faults = [epoch["fault"] for epoch in epochs if epoch["fault"]]
        fault = faults[0] if len(faults) == 1 else None
        if dropped != [expected_fault_seq] or not fault or fault["fault"] != 2 or fault["sample_seq"] != expected_fault_seq:
            errors.append("expected omitted action / DEADLINE FAULT identity not observed")
        result = "success(expected_fault)" if not errors else "fail"
    return {"result": result, "errors": errors, "epochs": epochs,
            "stop_confirmed": stopped, "final_disabled_zero_targets": disabled}


class Transport:
    def __init__(self, path, allowed_kinds=None):
        self.fd = None
        self.attrs = None
        self.decoder = Decoder()
        self.pending = deque()
        self.seq = secrets.randbelow(0x7fffffff) + 1
        self.allowed_kinds = (STATUS, REMOTE_QUERY, DIAG_START, DIAG_ACTION,
                              DIAG_STOP, DIAG_QUERY) if allowed_kinds is None else tuple(allowed_kinds)
        # fuser before opening detects existing readers; TIOCEXCL prevents later opens.
        path = os.path.realpath(path)
        probe = subprocess.run(["fuser", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if probe.returncode == 0 or probe.stdout.strip():
            raise RuntimeError("serial port is already in use: " + probe.stdout.decode(errors="replace"))
        if probe.returncode != 1:
            raise RuntimeError("fuser could not check serial exclusivity")
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            fcntl.ioctl(self.fd, termios.TIOCEXCL)
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.attrs = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.fd is not None:
            try:
                if self.attrs is not None:
                    termios.tcsetattr(self.fd, termios.TCSANOW, self.attrs)
            finally:
                try:
                    fcntl.ioctl(self.fd, termios.TIOCNXCL)
                finally:
                    os.close(self.fd)
                    self.fd = None

    def send(self, kind, payload=b""):
        if kind not in self.allowed_kinds:
            raise ValueError("command is not in this transport's allowlist")
        self.seq = (self.seq + 1) & NO_SEQ
        packet = encode(kind, self.seq, payload)
        deadline = time.monotonic() + 0.008
        sent = 0
        while sent < len(packet):
            try:
                sent += os.write(self.fd, packet[sent:])
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [self.fd], [], remaining)[1]:
                    raise TimeoutError("USB write exceeded 8 ms")
        return self.seq

    def receive(self, timeout):
        if self.pending:
            return self.pending.popleft()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ready = select.select([self.fd], [], [], max(0.0, deadline - time.monotonic()))[0]
            if not ready:
                break
            try:
                chunk = os.read(self.fd, 4096)
            except BlockingIOError:
                continue
            if not chunk:
                raise OSError("USB disconnected")
            received_ns = time.monotonic_ns()
            self.pending.extend((kind, seq, payload, received_ns) for kind, seq, payload in self.decoder.feed(chunk))
            if self.pending:
                return self.pending.popleft()
        return None

    def request(self, kind, response_type, payload=b"", timeout=1.0):
        seq = self.send(kind, payload)
        deadline = time.monotonic() + timeout
        deferred = []
        try:
            while time.monotonic() < deadline:
                packet = self.receive(max(0.0, deadline - time.monotonic()))
                if packet is None:
                    break
                if packet[0] == response_type and packet[1] == seq:
                    return packet
                deferred.append(packet)
            raise TimeoutError("no matching response to command 0x%02x" % kind)
        finally:
            self.pending.extendleft(reversed(deferred))


def load_inference(checkpoint):
    # Import order required by Isaac Gym even though no simulation is created.
    import isaacgym  # noqa: F401
    import torch
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mj_sim2sim_ct import load_policy
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    policy = load_policy(str(checkpoint), device="cpu")

    def inference(obs, history):
        started = time.monotonic_ns()
        with torch.inference_mode():
            action, latent = policy.act_inference(
                torch.from_numpy(np.asarray(obs, dtype=np.float32)).reshape(1, 25),
                torch.from_numpy(np.asarray(history, dtype=np.float32)).reshape(1, 125))
        raw = action.reshape(6).numpy().astype(np.float32, copy=False)
        hidden = latent.reshape(3).numpy().astype(np.float32, copy=False)
        if not np.isfinite(raw).all() or not np.isfinite(hidden).all():
            raise ValueError("nonfinite encoder/actor output")
        published = np.clip(raw, -100.0, 100.0).astype(np.float32, copy=False)
        return raw.tolist(), published.tolist(), hidden.tolist(), started, time.monotonic_ns()

    for _ in range(100):
        inference([0.0] * 25, [0.0] * 125)
    return inference


def run(args):
    output = args.output.resolve()
    root_logs = Path(__file__).resolve().parents[1] / "logs"
    if output == root_logs or root_logs in output.parents:
        raise ValueError("--output must not be inside historical root logs/")
    output.mkdir(parents=True, exist_ok=False)
    records, failures = [], []
    link = None
    session = None
    start_sent = False
    owned = False
    verifier = None
    metadata = {"protocol": 2701, "checkpoint": str(args.checkpoint.resolve()),
                "host_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "port": args.port, "seconds": args.seconds, "drop_action_at": args.drop_action_at,
                "created_unix_ns": time.time_ns(), "python": sys.executable,
                "scope": "disabled data path only; not motor execution or closed-loop stability"}
    try:
        digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        metadata["checkpoint_sha256"] = digest
        if digest != EXPECTED_SHA256:
            raise ValueError("checkpoint SHA256 differs from approved full model_6000.pt")
        infer = load_inference(args.checkpoint)
        metadata["warmup_complete_ns"] = time.monotonic_ns()
        link = Transport(args.port)
        legacy = unpack_legacy(link.request(STATUS, LEGACY_STATUS)[2])
        records.append({"event": "preflight_legacy", "status": legacy})
        require_disabled(legacy)
        remote = unpack_remote(link.request(REMOTE_QUERY, REMOTE_STATUS)[2])
        records.append({"event": "preflight_remote", "status": remote})
        if (remote["layout"] != 2602 or remote["online"] != 1 or remote["left_switch"] != 2 or
                remote["right_switch"] != 2 or remote["armed"] or remote["enabled_mask"] or
                remote["permit"] or remote["has_command"] or remote["usb_connected"] != 1 or
                remote["motors_online"] != 63):
            raise RuntimeError("REMOTE_STATUS requires online, both switches DOWN, disabled and USB connected")
        before = unpack_header(link.request(DIAG_QUERY, DIAG_STATUS)[2])
        records.append({"event": "preflight_diag", "header": before})
        if before["layout"] != 2701 or before["state"] != OFF or before["dm_enabled_mask"] or before["online_mask"] != 63:
            raise RuntimeError("DIAG_QUERY must report compatible OFF, online and disabled")
        session = secrets.randbelow(NO_SEQ) + 1
        while session == before["session"]:
            session = secrets.randbelow(NO_SEQ) + 1
        metadata["session"] = session
        start_sent = True
        started = unpack_header(link.request(DIAG_START, DIAG_STATUS, struct.pack("<II", 2701, session))[2])
        owned = started["session"] == session and started["state"] in (RUNNING, FAULT)
        if not owned or started["state"] != RUNNING or started["fault"]:
            raise RuntimeError("DIAG_START rejected: " + str(started))
        records.append({"event": "start", "header": started, "host_ns": time.monotonic_ns()})
        verifier = RecordVerifier(session, started["reset_count"])
        require_fresh_start(started, before)
        deadline = time.monotonic() + args.seconds + 2.0
        first_ms = None
        dropped = False
        while time.monotonic() < deadline:
            packet = link.receive(0.15)
            if packet is None:
                # Query only after absent samples; normal 100 Hz path has no extra traffic.
                packet = link.request(DIAG_QUERY, DIAG_STATUS)
            kind, seq, payload, recv_ns = packet
            if kind == DIAG_STATUS:
                h = unpack_header(payload)
                if h["session"] != session:
                    raise RuntimeError("DIAG_STATUS changed session")
                if h["state"] == FAULT:
                    records.append({"event": "fault", "header": h, "host_ns": recv_ns})
                    verifier.record_fault(h)
                    if not (dropped and h["fault"] == 2 and h["sample_seq"] == args.drop_action_at):
                        raise RuntimeError("board FAULT: " + FAULT_NAMES.get(h["fault"], str(h["fault"])))
                    break
                if h["state"] != RUNNING:
                    raise RuntimeError("diagnostic unexpectedly left RUNNING")
                continue
            if kind != DIAG_SAMPLE:
                records.append({"event": "other_packet", "kind": kind, "outer_seq": seq, "payload_hex": payload.hex()})
                continue
            record = unpack_sample(seq, payload)
            record["host_recv_ns"] = recv_ns
            checks = verifier.check_sample(record)
            record["checks"] = checks
            if not all(checks.values()):
                records.append(record)
                raise RuntimeError("sample validation failed: " + ", ".join(k for k, v in checks.items() if not v))
            try:
                raw, published, latent, infer_start, infer_end = infer(record["obs"], record["history"])
            except BaseException:
                records.append(record)
                raise
            do_drop = record["header"]["sample_seq"] == args.drop_action_at
            send_start = time.monotonic_ns()
            if not do_drop:
                link.send(DIAG_ACTION, ACTION.pack(session, record["header"]["sample_seq"], *published))
            sent_ns = time.monotonic_ns()
            # ACTION write precedes log conversion or disk activity.
            record.update(raw_actor=raw, published_action=published, latent=latent,
                          inference_start_ns=infer_start, inference_end_ns=infer_end,
                          inference_ns=infer_end - infer_start, host_send_start_ns=send_start,
                          host_send_ns=None if do_drop else sent_ns, action_dropped=do_drop)
            records.append(record)
            verifier.record_action(record)
            if do_drop:
                dropped = True
            sample_ms = record["header"]["sample_ms"]
            if first_ms is None:
                first_ms = sample_ms
            elapsed_ms = (sample_ms - first_ms) & NO_SEQ
            if not dropped and elapsed_ms >= args.seconds * 1000:
                break
        else:
            raise TimeoutError("requested board-clock duration or expected FAULT was not reached")
    except BaseException as exc:
        failures.append(type(exc).__name__ + ": " + str(exc))
    finally:
        if link is not None:
            try:
                # A lost START reply is ambiguous: query first, never stop another epoch.
                if start_sent and not owned:
                    state = unpack_header(link.request(DIAG_QUERY, DIAG_STATUS)[2])
                    owned = state["session"] == session and state["state"] in (RUNNING, FAULT)
                if owned:
                    stopped = unpack_header(link.request(DIAG_STOP, DIAG_STATUS, struct.pack("<I", session))[2])
                    # request() retains preceding packets. A queued sample/fault
                    # cannot be hidden by STOP clearing the board's fault state.
                    for kind, seq, payload, received_ns in list(link.pending):
                        if kind == DIAG_SAMPLE:
                            failures.append("unprocessed SAMPLE arrived before STOP confirmation")
                            records.append({"event": "unprocessed_sample", "outer_seq": seq,
                                            "payload_hex": payload.hex(), "host_recv_ns": received_ns})
                        elif kind == DIAG_STATUS:
                            pending_h = unpack_header(payload)
                            if pending_h["session"] == session and pending_h["state"] == FAULT:
                                failures.append("board FAULT pending at STOP: " + str(pending_h["fault"]))
                                records.append({"event": "fault", "header": pending_h, "host_ns": received_ns})
                    records.append({"event": "stop", "header": stopped, "host_ns": time.monotonic_ns()})
                    if stopped["state"] != OFF or stopped["session"] != session or stopped["dm_enabled_mask"]:
                        raise RuntimeError("matching DIAG_STOP did not confirm OFF and disabled")
                final = unpack_legacy(link.request(STATUS, LEGACY_STATUS)[2])
                records.append({"event": "final_legacy", "status": final})
                require_disabled(final)
            except BaseException as exc:
                failures.append("cleanup: " + type(exc).__name__ + ": " + str(exc))
            finally:
                metadata["decoder"] = {"crc_errors": link.decoder.crc_errors,
                    "header_errors": link.decoder.header_errors, "discarded_bytes": link.decoder.discarded_bytes,
                    "unparsed_bytes": len(link.decoder.buffer)}
                if link.decoder.crc_errors or link.decoder.header_errors or link.decoder.discarded_bytes:
                    failures.append("USB parser observed corruption or discarded bytes")
                try:
                    link.close()
                except BaseException as exc:
                    failures.append("serial close: " + str(exc))
        records.extend({"event": "error", "message": failure} for failure in failures)
        result = verify_records(records, args.drop_action_at)
        if failures:
            result["result"] = "fail"
        result["metadata"] = metadata
        samples = [r for r in records if r["event"] == "sample" and "inference_ns" in r]
        times = sorted(r["inference_ns"] for r in samples)
        result["inference_ns_median"] = times[len(times) // 2] if times else None
        result["inference_ns_max"] = max(times) if times else None
        with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=True, separators=(",", ":")) + "\n")
        (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--drop-action-at", type=int, help="omit one zero-based sample's ACTION; expect DEADLINE FAULT")
    parser.add_argument("--verify", type=Path, help="offline records.jsonl verification; no serial access")
    args = parser.parse_args()
    if args.verify:
        with args.verify.open(encoding="utf-8") as stream:
            result = verify_records([json.loads(line) for line in stream if line.strip()], args.drop_action_at)
    else:
        if not args.port or not args.checkpoint or not args.output:
            parser.error("live diagnostic requires --port --checkpoint --output")
        if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 3600:
            parser.error("--seconds must be finite and between 1 and 3600")
        if args.drop_action_at is not None and not 0 <= args.drop_action_at < args.seconds * 100:
            parser.error("--drop-action-at must be in the requested run's sample range")
        try:
            result = run(args)
        except (OSError, ValueError) as exc:
            parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"] in ("pass", "success(expected_fault)") else 2


if __name__ == "__main__":
    sys.exit(main())
