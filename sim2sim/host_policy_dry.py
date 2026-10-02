#!/usr/bin/env python3
"""Run an explicitly disabled H7 2901 transport rehearsal; never ARM motors."""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import struct
import sys
import time

if __package__:
    from . import host_policy_run as policy
    from .host_policy_diag import (Transport, encode, load_inference,
                                   require_disabled, unpack_legacy, unpack_remote,
                                   STATUS, REMOTE_QUERY, LEGACY_STATUS, REMOTE_STATUS)
else:
    import host_policy_run as policy
    from host_policy_diag import (Transport, encode, load_inference,
                                  require_disabled, unpack_legacy, unpack_remote,
                                  STATUS, REMOTE_QUERY, LEGACY_STATUS, REMOTE_STATUS)

DRY_START = 0x35
DRY_FLAG = 1 << 14


def run(args):
    output = args.out.resolve()
    root_logs = Path(__file__).resolve().parents[1] / "logs"
    if output == root_logs or root_logs in output.parents or output.exists():
        raise ValueError("--out must be a new directory outside historical logs/")
    checkpoint = args.checkpoint.resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != policy.CHECKPOINT_SHA256:
        raise ValueError("checkpoint hash does not match 2901 contract")
    infer = load_inference(checkpoint)
    session = secrets.randbelow(policy.NO_SEQ - 1) + 1
    records, errors = [], []
    link = None
    owned = False
    samples = 0
    contact_sent = False
    fault_seen = False
    stop_confirmed = False
    final_disabled = False
    metadata = {"session": session, "checkpoint_sha256": digest,
                "firmware_elf_sha256": hashlib.sha256(args.firmware_elf.read_bytes()).hexdigest(),
                "host_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "port": args.port, "sample_goal": args.samples,
                "drop_action_at": args.drop_action_at, "scope": "disabled 2901 rehearsal only"}
    try:
        link = Transport(args.port, allowed_kinds=(STATUS, REMOTE_QUERY, DRY_START,
                    policy.POLICY_ACTION, policy.POLICY_CONTACT, policy.POLICY_STOP,
                    policy.POLICY_QUERY, policy.GLOBAL_STOP))
        legacy = unpack_legacy(link.request(STATUS, LEGACY_STATUS)[2])
        require_disabled(legacy)
        records.append({"event": "preflight_legacy", "status": legacy})
        remote = unpack_remote(link.request(REMOTE_QUERY, REMOTE_STATUS)[2])
        records.append({"event": "preflight_remote", "status": remote})
        if (remote["online"] != 1 or remote["left_switch"] != 2
                or remote["right_switch"] != 2 or remote["armed"]
                or remote["enabled_mask"] or remote["motors_online"] != 63):
            raise RuntimeError("dry rehearsal requires remote 2/2 and all motors disabled/online")
        before = policy.status_header(link.request(policy.POLICY_QUERY,
                                                   policy.POLICY_STATUS)[2])
        if before["phase"] != policy.OFF or before["dm_enabled_mask"]:
            raise RuntimeError("2901 is not OFF/disabled before dry start")
        while session == before["session"]:
            session = secrets.randbelow(policy.NO_SEQ - 1) + 1
        metadata["session"] = session
        started = policy.unpack_header(link.request(DRY_START, policy.POLICY_STATUS,
                                      policy.start_payload(session, digest))[2], session)
        owned = started["session"] == session and started["phase"] != policy.OFF
        records.append({"event": "dry_start", "header": started})
        if (started["phase"] != policy.PRECONTACT or started["fault"]
                or started["dm_enabled_mask"] or not started["guard_flags"] & DRY_FLAG):
            raise RuntimeError("dry START rejected or output gate unexpectedly enabled")
        check = policy.PolicySession(session, expected_enabled_mask=0, expected_dry=True)
        deadline = time.monotonic() + args.samples * 0.01 + 3.0
        while time.monotonic() < deadline:
            packet = link.receive(0.05)
            if packet is None:
                raise TimeoutError("2901 dry SAMPLE/FAULT absent for 50 ms")
            kind, outer_seq, payload, recv_ns = packet
            if kind == policy.POLICY_STATUS:
                h = policy.unpack_header(payload, session)
                records.append({"event": "status", "header": h,
                                "host_recv_ns": recv_ns})
                if h["phase"] == policy.FAULT:
                    if args.drop_action_at is None or h["fault"] != 2:
                        raise RuntimeError("unexpected dry 2901 fault: " + str(h))
                    fault_seen = True
                    break
                continue
            if kind != policy.POLICY_SAMPLE:
                continue
            sample = policy.unpack_sample(outer_seq, payload, session)
            raw, clipped, latent, infer_start, infer_end = infer(
                sample["obs"], sample["history"])
            contact, action_packet, action = check.accept(
                sample, raw, contact_requested=(outer_seq == 5))
            if contact is not None:
                link.send(policy.POLICY_CONTACT, contact)
                contact_sent = True
            dropped = outer_seq == args.drop_action_at
            if not dropped:
                link.send(policy.POLICY_ACTION, action_packet)
            records.append({"event": "sample", "header": sample["header"],
                            "obs": sample["obs"], "history": sample["history"],
                            "raw_actor": raw, "published_action": action,
                            "latent": latent, "dropped": dropped,
                            "host_recv_ns": recv_ns, "infer_start_ns": infer_start,
                            "infer_end_ns": infer_end,
                            "raw_frame_hex": encode(kind, outer_seq, payload).hex()})
            samples += 1
            if args.drop_action_at is None and samples >= args.samples:
                break
        if args.drop_action_at is not None and not fault_seen:
            raise RuntimeError("omitted action did not produce DEADLINE fault")
        if not contact_sent or not any(r["event"] == "sample"
                 and r["header"]["phase"] == policy.ACTIVE for r in records):
            raise RuntimeError("synthetic contact transition/ACTIVE not observed")
        if args.drop_action_at is None and samples < args.samples:
            raise RuntimeError("too few dry samples")
    except BaseException as exc:
        errors.append(type(exc).__name__ + ": " + str(exc))
    finally:
        if link is not None:
            try:
                if owned:
                    stopped = policy.unpack_header(link.request(policy.POLICY_STOP,
                        policy.POLICY_STATUS, struct.pack("<I", session))[2], session)
                    records.append({"event": "stop", "header": stopped})
                    stop_confirmed = stopped["phase"] == policy.OFF and stopped["dm_enabled_mask"] == 0
                    if not stop_confirmed:
                        raise RuntimeError("POLICY_STOP failed to confirm OFF/disabled")
            except BaseException as exc:
                errors.append("cleanup: " + type(exc).__name__ + ": " + str(exc))
            if not stop_confirmed:
                try:
                    link.send(policy.GLOBAL_STOP)
                    records.append({"event": "global_stop_fallback"})
                except BaseException as exc:
                    errors.append("global STOP: " + str(exc))
            try:
                final = unpack_legacy(link.request(STATUS, LEGACY_STATUS)[2])
                require_disabled(final)
                final_disabled = True
                records.append({"event": "final_legacy", "status": final})
            except BaseException as exc:
                errors.append("final disabled: " + str(exc))
            metadata["decoder"] = {"crc_errors": link.decoder.crc_errors,
                "header_errors": link.decoder.header_errors,
                "discarded_bytes": link.decoder.discarded_bytes,
                "unparsed_bytes": len(link.decoder.buffer)}
            link.close()
        output.mkdir(parents=True, exist_ok=False)
        with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        if any(metadata.get("decoder", {}).values()):
            errors.append("USB decoder observed corruption or incomplete frame")
        summary = {"result": "pass" if not errors and stop_confirmed and final_disabled else "fail",
                   "errors": errors, "samples": samples, "contact_sent": contact_sent,
                   "fault_seen": fault_seen, "stop_confirmed": stop_confirmed,
                   "final_disabled": final_disabled, "metadata": metadata}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
                                            indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--firmware-elf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--drop-action-at", type=int)
    args = parser.parse_args()
    if not 30 <= args.samples <= 1000:
        parser.error("--samples must be 30..1000")
    if args.drop_action_at is not None and not 10 <= args.drop_action_at < args.samples:
        parser.error("--drop-action-at must be 10..samples-1")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"] == "pass" else 2


if __name__ == "__main__":
    sys.exit(main())
