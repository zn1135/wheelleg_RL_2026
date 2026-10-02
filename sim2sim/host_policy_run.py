#!/usr/bin/env python3
"""电脑 CPU 推理与 H7 2901 执行会话；默认只显示预检信息。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import select
import struct
import sys
import termios
import time
import tty

if __package__:
    from .host_policy_diag import (Transport, encode, load_inference, require_disabled,
                                   unpack_legacy, unpack_remote, STATUS, REMOTE_QUERY,
                                   LEGACY_STATUS, REMOTE_STATUS)
else:
    from host_policy_diag import (Transport, encode, load_inference, require_disabled,
                                  unpack_legacy, unpack_remote, STATUS, REMOTE_QUERY,
                                  LEGACY_STATUS, REMOTE_STATUS)

LAYOUT = 2901
CHECKPOINT_SHA256 = "8070977e19114cb9bbf358a090bd78cdbd1be568a76e02d417c369e828bc8f36"
POLICY_START, POLICY_ACTION, POLICY_CONTACT, POLICY_STOP, POLICY_QUERY = 0x30, 0x31, 0x32, 0x33, 0x34
POLICY_SAMPLE, POLICY_STATUS = 0xB0, 0xB1
GLOBAL_STOP = 4
OFF, PRECONTACT, CONTACT_EDGE, ACTIVE, FAULT = range(5)
NO_SEQ = 0xffffffff
TRIAL_EXPIRED_FLAG = 1 << 16
HEADER_NAMES = ("layout", "session", "sample_seq", "sample_ms", "tx_ms",
                "applied_action_source_seq", "applied_action_rx_ms", "phase", "fault",
                "online_mask", "dm_enabled_mask", "period_ms", "contact_source_seq",
                "tick_seq", "guard_flags", "saturation_count")
HEADER = struct.Struct("<16I")
FLOATS = struct.Struct("<150f")


def start_payload(session, checkpoint_sha256):
    if not 0 < session <= NO_SEQ or checkpoint_sha256 != CHECKPOINT_SHA256:
        raise ValueError("session or checkpoint SHA-256 does not match 2901 contract")
    return struct.pack("<II", LAYOUT, session) + bytes.fromhex(checkpoint_sha256)


def contact_payload(session, sample_seq):
    if not 0 < session <= NO_SEQ or not 0 <= sample_seq < NO_SEQ:
        raise ValueError("invalid contact session or sample")
    return struct.pack("<II", session, sample_seq)


def unpack_header(payload, session):
    if len(payload) != HEADER.size:
        raise ValueError("2901 STATUS header length mismatch")
    result = dict(zip(HEADER_NAMES, HEADER.unpack(payload)))
    if result["layout"] != LAYOUT or result["session"] != session:
        raise ValueError("2901 layout/session mismatch")
    if result["phase"] not in (OFF, PRECONTACT, CONTACT_EDGE, ACTIVE, FAULT):
        raise ValueError("2901 phase invalid")
    return result


def unpack_sample(seq, payload, session):
    if len(payload) != HEADER.size + FLOATS.size:
        raise ValueError("2901 SAMPLE length mismatch")
    header = unpack_header(payload[:HEADER.size], session)
    if header["sample_seq"] != seq or header["period_ms"] != 10:
        raise ValueError("2901 sample identity or period mismatch")
    values = FLOATS.unpack_from(payload, HEADER.size)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("2901 SAMPLE contains nonfinite observation/history")
    return {"header": header, "obs": list(values[:25]), "history": list(values[25:])}


def action_for_phase(phase, actor):
    if len(actor) != 6 or not all(math.isfinite(value) for value in actor):
        raise ValueError("actor must return six finite actions")
    if phase in (PRECONTACT, CONTACT_EDGE):
        return [0.0] * 6
    if phase != ACTIVE:
        raise ValueError("non-running policy phase")
    return [max(-100.0, min(100.0, float(value))) for value in actor]


class PolicySession:
    """Validate the FIFO/phase boundary before forming a board ACTION."""
    def __init__(self, session, expected_enabled_mask=15, expected_dry=False):
        if not 0 < session <= NO_SEQ:
            raise ValueError("invalid session")
        if expected_enabled_mask not in (0, 15) or expected_dry != (expected_enabled_mask == 0):
            raise ValueError("invalid 2901 enable/dry contract")
        self.session = session
        self.expected_enabled_mask = expected_enabled_mask
        self.expected_dry = expected_dry
        self.previous = None
        self.previous_action = None
        self.expected_phase = PRECONTACT
        self.contact_source_seq = NO_SEQ

    def accept(self, sample, raw_actor, contact_requested=False):
        header, obs, history = sample["header"], sample["obs"], sample["history"]
        seq = header["sample_seq"]
        if header["layout"] != LAYOUT or header["session"] != self.session:
            raise ValueError("session/layout changed")
        if header["fault"] or header["phase"] != self.expected_phase:
            raise ValueError("board fault or unexpected contact phase")
        if header["online_mask"] != 63 or header["dm_enabled_mask"] != self.expected_enabled_mask:
            raise ValueError("board motor feedback or enable gate changed")
        if bool(header["guard_flags"] & (1 << 14)) != self.expected_dry:
            raise ValueError("board dry-run ownership changed")
        if (header["tx_ms"] - header["sample_ms"]) & NO_SEQ >= 10:
            raise ValueError("board transmitted SAMPLE after action deadline")
        if len(obs) != 25 or len(history) != 125:
            raise ValueError("observation/history dimension mismatch")
        if self.previous is None:
            if seq != 0 or header["applied_action_source_seq"] != NO_SEQ:
                raise ValueError("first sample identity mismatch")
            if struct.pack("<125f", *history) != struct.pack("<125f", *(obs * 5)):
                raise ValueError("first history not repeated five times")
        else:
            prev = self.previous["header"]
            if seq != (prev["sample_seq"] + 1) & NO_SEQ:
                raise ValueError("sample duplicate or gap")
            if (header["sample_ms"] - prev["sample_ms"]) & NO_SEQ != 10:
                raise ValueError("board sample cadence mismatch")
            if header["applied_action_source_seq"] != prev["sample_seq"]:
                raise ValueError("board action acknowledgement mismatch")
            if (header["applied_action_rx_ms"] - prev["sample_ms"]) & NO_SEQ >= 10:
                raise ValueError("board acknowledged an action after deadline")
            if struct.pack("<6f", *obs[19:25]) != struct.pack("<6f", *self.previous_action):
                raise ValueError("last-action feedback mismatch")
            if struct.pack("<125f", *history) != struct.pack(
                    "<125f", *(self.previous["history"][25:] + obs)):
                raise ValueError("history FIFO mismatch")
        if header["phase"] in (CONTACT_EDGE, ACTIVE):
            if header["contact_source_seq"] != self.contact_source_seq:
                raise ValueError("contact source changed")
        elif header["contact_source_seq"] != NO_SEQ:
            raise ValueError("contact appeared before confirmation")
        contact = None
        if contact_requested:
            if header["phase"] != PRECONTACT or self.contact_source_seq != NO_SEQ:
                raise ValueError("contact can only be confirmed once in PRECONTACT")
            contact = contact_payload(self.session, seq)
            self.contact_source_seq = seq
            self.expected_phase = CONTACT_EDGE
        elif header["phase"] == CONTACT_EDGE:
            self.expected_phase = ACTIVE
        action = action_for_phase(header["phase"], raw_actor)
        packet = struct.pack("<II6f", self.session, seq, *action)
        self.previous = sample
        self.previous_action = action
        return contact, packet, action


def status_header(payload):
    if len(payload) != HEADER.size:
        raise ValueError("2901 STATUS header length mismatch")
    result = dict(zip(HEADER_NAMES, HEADER.unpack(payload)))
    if result["layout"] != LAYOUT or result["phase"] not in (OFF, PRECONTACT,
                                                       CONTACT_EDGE, ACTIVE, FAULT):
        raise ValueError("2901 STATUS layout/phase mismatch")
    return result


def read_key():
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    key = os.read(sys.stdin.fileno(), 1)
    return key.decode("ascii", errors="ignore").lower() if key else None


def disabled_status(status):
    try:
        require_disabled(status)
        return True
    except RuntimeError:
        return False


def unpack_motor_feedback(payload):
    """Legacy STATUS feedback: driver estimates, not external torque sensors."""
    if len(payload) != 132:
        raise ValueError("motor feedback STATUS length mismatch")
    status = unpack_legacy(payload)
    for name, offset, count in (("dm_pos_rad", 16, 4), ("dm_vel_rad_s", 32, 4),
                               ("dm_torque_nm", 48, 4), ("wheel_angle_rad", 64, 2),
                               ("wheel_vel_rad_s", 72, 2), ("euler_rad", 84, 3),
                               ("gyro_rad_s", 96, 3)):
        status[name] = list(struct.unpack_from("<%df" % count, payload, offset))
    status["wheel_current_raw"] = list(struct.unpack_from("<2h", payload, 80))
    return status


class MotorFeedbackCapture:
    """One optional STATUS in flight; actions never wait for its reply."""
    def __init__(self):
        self.pending = None
        self.requested = 0
        self.received = 0
        self.missing = 0

    def after_action(self, link, sample_seq, records):
        if self.pending is not None and (sample_seq - self.pending["source_sample_seq"]) & NO_SEQ >= 10:
            records.append(dict(self.pending, event="motor_feedback_missing",
                                reason="no matching reply within 10 policy samples"))
            self.missing += 1
            self.pending = None
        if (sample_seq + 1) % 5 or self.pending is not None:
            return
        outer_seq = link.send(STATUS)
        self.pending = {"outer_seq": outer_seq, "source_sample_seq": sample_seq,
                        "host_send_ns": time.monotonic_ns()}
        self.requested += 1
        records.append(dict(self.pending, event="motor_feedback_request"))

    def accept(self, packet, records):
        kind, seq, payload, received_ns = packet
        if self.pending is None or kind != LEGACY_STATUS or seq != self.pending["outer_seq"]:
            return False
        event = dict(self.pending, host_recv_ns=received_ns,
                     raw_frame_hex=encode(kind, seq, payload).hex())
        try:
            event.update(event="motor_feedback", status=unpack_motor_feedback(payload))
            self.received += 1
        except ValueError as exc:
            event.update(event="motor_feedback_missing", reason=str(exc))
            self.missing += 1
        records.append(event)
        self.pending = None
        return True

    def summary(self):
        return {"enabled": True, "requested": self.requested,
                "received": self.received, "missing": self.missing,
                "unfinished": int(self.pending is not None), "pending": self.pending,
                "scope": "driver torque/current estimates; not external torque measurements"}


def wait_for_stop_disarm(link, records):
    """Allow the asynchronous DM disable feedback to settle after an OFF ACK."""
    deadline = time.monotonic() + 0.050
    while True:
        remaining = deadline - time.monotonic()
        # Transport.request() starts its reply timer after send(), which has
        # its own 8 ms write bound. Reserve that time inside this 50 ms budget.
        if remaining <= 0.008:
            raise TimeoutError("DM disabled feedback was not confirmed within 50 ms after STOP")
        try:
            packet = link.request(STATUS, LEGACY_STATUS, timeout=remaining - 0.008)
        except TimeoutError as exc:
            raise TimeoutError("DM disabled feedback was not confirmed within 50 ms after STOP") from exc
        status = unpack_legacy(packet[2])
        records.append({"event": "stop_disarm_poll", "status": status,
                        "host_recv_ns": packet[3],
                        "raw_frame_hex": encode(packet[0], packet[1], packet[2]).hex()})
        if time.monotonic() >= deadline:
            raise TimeoutError("DM disabled feedback was not confirmed within 50 ms after STOP")
        if disabled_status(status):
            return
        time.sleep(min(0.002, max(0.0, deadline - time.monotonic())))


def wait_remote_arm(link, records):
    print("保持右拨杆下档，将左拨杆从下档拨到上档；等待板端 ARM 和四台 DM 使能。", flush=True)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        key = read_key()
        if key in ("q", "\x03"):
            raise KeyboardInterrupt("operator cancelled before START")
        packet = link.request(REMOTE_QUERY, REMOTE_STATUS, timeout=0.5)
        remote = unpack_remote(packet[2])
        records.append({"event": "remote_arm_poll", "status": remote,
                        "host_ns": time.monotonic_ns()})
        if (remote["online"] == 1 and remote["left_switch"] == 1
                and remote["right_switch"] == 2 and remote["armed"] == 1
                and remote["enabled_mask"] == 15 and remote["permit"] == 1
                and remote["motors_online"] == 63):
            return
        time.sleep(0.05)
    raise TimeoutError("remote ARM/DM enable was not observed within 30 s")


def run(args):
    if not sys.stdin.isatty():
        raise ValueError("live 2901 execution requires an interactive terminal")
    if not str(args.port).startswith("/dev/serial/by-id/"):
        raise ValueError("--port must be a stable /dev/serial/by-id/ path")
    root_logs = Path(__file__).resolve().parents[1] / "logs"
    output = args.out.resolve()
    if output == root_logs or root_logs in output.parents:
        raise ValueError("--out must not be inside historical root logs/")
    if output.exists():
        raise FileExistsError("--out must be a new directory")
    checkpoint = args.checkpoint.resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 differs from approved full model_6000.pt")
    infer = load_inference(checkpoint)
    records = []
    failures = []
    link = None
    session = secrets.randbelow(NO_SEQ - 1) + 1
    owned = False
    start_sent = False
    contact_sent = False
    board_auto_stopped = False
    feedback = MotorFeedbackCapture() if getattr(args, "motor_feedback", False) else None
    original_tty = termios.tcgetattr(sys.stdin.fileno())
    metadata = {"protocol": LAYOUT, "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest, "port": str(args.port),
                "host_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "session": session, "created_unix_ns": time.time_ns(),
                "contact_source": "onsite_confirmed_before_start"
                    if getattr(args, "contact_confirmed_at_start", False) else "operator_key_c",
                "python": sys.executable,
                "firmware_elf": str(args.firmware_elf.resolve()),
                "firmware_elf_sha256": hashlib.sha256(args.firmware_elf.read_bytes()).hexdigest()}
    try:
        tty.setcbreak(sys.stdin.fileno())
        link = Transport(str(args.port), allowed_kinds=(STATUS, REMOTE_QUERY,
                        POLICY_START, POLICY_ACTION, POLICY_CONTACT,
                        POLICY_STOP, POLICY_QUERY, GLOBAL_STOP))
        legacy = unpack_legacy(link.request(STATUS, LEGACY_STATUS)[2])
        require_disabled(legacy)
        records.append({"event": "preflight_legacy", "status": legacy})
        remote = unpack_remote(link.request(REMOTE_QUERY, REMOTE_STATUS)[2])
        records.append({"event": "preflight_remote", "status": remote})
        if (remote["layout"] != 2602 or remote["online"] != 1
                or remote["left_switch"] != 2 or remote["right_switch"] != 2
                or remote["armed"] or remote["enabled_mask"]
                or remote["permit"] or remote["motors_online"] != 63):
            raise RuntimeError("preflight requires online, both switches DOWN and disabled")
        before = status_header(link.request(POLICY_QUERY, POLICY_STATUS)[2])
        records.append({"event": "preflight_2901", "header": before})
        if before["phase"] != OFF or before["dm_enabled_mask"]:
            raise RuntimeError("2901 mode is not OFF before START")
        while session == before["session"]:
            session = secrets.randbelow(NO_SEQ - 1) + 1
        metadata["session"] = session
        wait_remote_arm(link, records)
        start_sent = True
        started = unpack_header(link.request(POLICY_START, POLICY_STATUS,
                                  start_payload(session, digest))[2], session)
        owned = started["phase"] in (PRECONTACT, CONTACT_EDGE, ACTIVE, FAULT)
        records.append({"event": "start", "header": started,
                        "host_ns": time.monotonic_ns()})
        if not owned or started["phase"] != PRECONTACT or started["fault"]:
            raise RuntimeError("2901 START rejected: " + str(started))
        session_check = PolicySession(session)
        deadline = time.monotonic() + args.seconds
        contact_requested = bool(getattr(args, "contact_confirmed_at_start", False))
        print("运行中：" + ("按已收到的现场轮接地确认接管；"
              if contact_requested else "轮接地后按 c 接管；") + "按 q 停止。", flush=True)
        contact_sent = False
        while time.monotonic() < deadline:
            key = read_key()
            if key in ("q", "\x03"):
                break
            if key == "c" and not contact_sent:
                contact_requested = True
            packet = link.receive(0.03)
            if packet is None:
                raise TimeoutError("no 2901 SAMPLE within 30 ms")
            kind, seq, payload, recv_ns = packet
            if feedback is not None and feedback.accept(packet, records):
                continue
            if kind == POLICY_STATUS:
                h = unpack_header(payload, session)
                records.append({"event": "status", "header": h,
                                "host_recv_ns": recv_ns,
                                "raw_frame_hex": encode(kind, seq, payload).hex()})
                if h["phase"] == FAULT or h["fault"]:
                    raise RuntimeError("board 2901 FAULT: " + str(h))
                if h["phase"] == OFF and h["guard_flags"] & TRIAL_EXPIRED_FLAG:
                    board_auto_stopped = True
                    break
                continue
            if kind != POLICY_SAMPLE:
                records.append({"event": "other_packet", "kind": kind, "seq": seq,
                                "payload_hex": payload.hex(), "host_recv_ns": recv_ns})
                continue
            sample = unpack_sample(seq, payload, session)
            infer_start = time.monotonic_ns()
            raw, clipped, latent, _, infer_end = infer(sample["obs"], sample["history"])
            contact, action_packet, action = session_check.accept(
                sample, raw, contact_requested=contact_requested)
            if contact is not None:
                contact_seq = link.send(POLICY_CONTACT, contact)
                records.append({"event": "contact_tx", "source_sample_seq": seq,
                                "host_send_ns": time.monotonic_ns(),
                                "outer_seq": contact_seq,
                                "raw_frame_hex": encode(POLICY_CONTACT, contact_seq, contact).hex()})
                contact_sent = True
                contact_requested = False
            sent_seq = link.send(POLICY_ACTION, action_packet)
            records.append({"event": "sample", "header": sample["header"],
                            "obs": sample["obs"], "history": sample["history"],
                            "raw_actor": raw, "clipped_actor": clipped,
                            "published_action": action, "latent": latent,
                            "contact_sent": contact is not None,
                            "host_recv_ns": recv_ns, "infer_start_ns": infer_start,
                            "infer_end_ns": infer_end, "host_send_ns": time.monotonic_ns(),
                            "host_action_outer_seq": sent_seq,
                            "raw_action_frame_hex": encode(POLICY_ACTION, sent_seq,
                                                           action_packet).hex(),
                            "raw_frame_hex": encode(kind, seq, payload).hex()})
            if feedback is not None:
                feedback.after_action(link, seq, records)
        if not records or not any(r["event"] == "sample" for r in records):
            raise RuntimeError("2901 START returned no valid SAMPLE")
    except BaseException as exc:
        failures.append(type(exc).__name__ + ": " + str(exc))
    finally:
        if link is not None:
            stop_failed = False
            try:
                if start_sent and not owned:
                    queried = status_header(link.request(POLICY_QUERY, POLICY_STATUS)[2])
                    owned = queried["session"] == session and queried["phase"] != OFF
                if owned:
                    stopped = unpack_header(link.request(POLICY_STOP, POLICY_STATUS,
                                            struct.pack("<I", session))[2], session)
                    records.append({"event": "stop", "header": stopped,
                                    "host_ns": time.monotonic_ns()})
                    if stopped["phase"] != OFF:
                        raise RuntimeError("2901 STOP did not confirm OFF")
                    if stopped["dm_enabled_mask"]:
                        wait_for_stop_disarm(link, records)
            except BaseException as exc:
                failures.append("cleanup: " + type(exc).__name__ + ": " + str(exc))
                stop_failed = True
            if stop_failed:
                try:
                    link.send(GLOBAL_STOP)
                    records.append({"event": "global_stop_fallback",
                                    "host_ns": time.monotonic_ns()})
                except BaseException as exc:
                    failures.append("global STOP fallback: " + type(exc).__name__ + ": " + str(exc))
            try:
                final = unpack_legacy(link.request(STATUS, LEGACY_STATUS)[2])
                records.append({"event": "final_legacy", "status": final})
                require_disabled(final)
            except BaseException as exc:
                failures.append("final disarm: " + type(exc).__name__ + ": " + str(exc))
                try:
                    link.send(GLOBAL_STOP)
                    records.append({"event": "global_stop_after_final_status",
                                    "host_ns": time.monotonic_ns()})
                    final = unpack_legacy(link.request(STATUS, LEGACY_STATUS)[2])
                    records.append({"event": "final_legacy_after_global_stop",
                                    "status": final})
                    require_disabled(final)
                except BaseException as fallback_exc:
                    failures.append("final global STOP: " + type(fallback_exc).__name__
                                    + ": " + str(fallback_exc))
            try:
                # request() preserves unmatched packets. Retain their evidence
                # before closing, including faults preceding a STOP/OFF reply.
                for kind, seq, payload, received_ns in list(link.pending):
                    if feedback is not None and feedback.accept((kind, seq, payload, received_ns), records):
                        continue
                    records.append({"event": "cleanup_pending_packet", "kind": kind,
                                    "outer_seq": seq, "host_recv_ns": received_ns,
                                    "raw_frame_hex": encode(kind, seq, payload).hex()})
                    if kind == POLICY_STATUS:
                        try:
                            pending = status_header(payload)
                        except ValueError as exc:
                            failures.append("invalid policy status pending during cleanup: " + str(exc))
                            continue
                        if pending["session"] == session and (pending["phase"] == FAULT
                                                              or pending["fault"]):
                            failures.append("board 2901 FAULT pending during cleanup: " + str(pending))
                metadata["decoder"] = {"crc_errors": link.decoder.crc_errors,
                    "header_errors": link.decoder.header_errors,
                    "discarded_bytes": link.decoder.discarded_bytes,
                    "unparsed_bytes": len(link.decoder.buffer)}
                if any(metadata["decoder"].values()):
                    failures.append("2901 USB parser observed corruption or incomplete frames")
                link.close()
            except BaseException as exc:
                failures.append("serial close: " + type(exc).__name__ + ": " + str(exc))
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, original_tty)
        except BaseException as exc:
            failures.append("terminal restore: " + type(exc).__name__ + ": " + str(exc))
        output.mkdir(parents=True, exist_ok=False)
        with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        active_seen = any(r["event"] == "sample"
                          and r["header"]["phase"] == ACTIVE for r in records)
        transport_ok = (not failures and contact_sent and active_seen
                        and any(r["event"] == "stop" and r["header"]["phase"] == OFF
                                for r in records)
                        and any(r["event"] == "final_legacy" for r in records))
        summary = {"result": "protocol_pass" if transport_ok else "fail",
                   "failures": failures, "samples": sum(r["event"] == "sample" for r in records),
                   "contact_sent": contact_sent,
                   "active_sample_seen": active_seen,
                   "board_auto_stopped": board_auto_stopped,
                   "motor_feedback": feedback.summary() if feedback is not None else {"enabled": False},
                   "scope": "2901 timing/transport record only; physical standup requires independent height and posture evidence",
                   "stop_confirmed": any(r["event"] == "stop" and r["header"]["phase"] == OFF
                                         for r in records),
                   "final_disabled": any(r["event"] in ("final_legacy",
                                                         "final_legacy_after_global_stop")
                                         and disabled_status(r["status"]) for r in records),
                   "metadata": metadata}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
                                         indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="start supervised hardware execution")
    parser.add_argument("--port", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--firmware-elf", type=Path)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--contact-confirmed-at-start", action="store_true",
                        help="仅当现场已明确确认双轮接地时，在首个样本发送一次人工接管事件")
    parser.add_argument("--motor-feedback", action="store_true",
                        help="动作发送后以20Hz异步采集驱动反馈；不会等待反馈或改变策略动作")
    args = parser.parse_args()
    if args.run:
        if not args.port or not args.checkpoint or not args.out or not args.firmware_elf:
            parser.error("--run requires --port --checkpoint --out --firmware-elf")
        if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 120:
            parser.error("--seconds must be 1..120")
        result = run(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["result"] == "protocol_pass" else 2
    print("2901 preflight only: checkpoint SHA-256", CHECKPOINT_SHA256)
    print("--run requires an interactive operator and a physically qualified H7 build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
