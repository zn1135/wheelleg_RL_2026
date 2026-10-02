#!/usr/bin/env python3
"""Offline 2701 framing/history/action diagnostic checks; no device or simulation.

This is a protocol check, not the project's training/replay/sim2sim validation.
Optionally --checkpoint loads the approved complete real policy on CPU and
checks float32 inference; importing the protocol alone needs no torch/Gym.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("host_policy_diag", ROOT / "sim2sim/host_policy_diag.py")
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


def header(session=19, reset=1):
    h = dict.fromkeys(D.HEADER_NAMES, 0)
    h.update(layout=2701, session=session, sample_seq=D.NO_SEQ,
             applied_action_source_seq=D.NO_SEQ, state=D.RUNNING,
             online_mask=63, period_ms=10, reset_count=reset)
    return h


def fixture(count=110, session=19, reset=1, drop=None):
    records = [{"event": "start", "header": header(session, reset)}]
    previous_action = [0.0] * 6
    history = None
    for seq in range(count):
        obs = list(struct.unpack("<25f", struct.pack("<25f", *([seq / 100.0] * 19 + previous_action))))
        history = obs * 5 if history is None else history[25:] + obs
        h = header(session, reset)
        h.update(sample_seq=seq, sample_ms=(D.NO_SEQ - 49 + seq * 10) & D.NO_SEQ,
                 accepted_actions=seq)
        h["tx_ms"] = (h["sample_ms"] + 1) & D.NO_SEQ
        if seq:
            h.update(applied_action_source_seq=seq - 1,
                     applied_action_rx_ms=(h["sample_ms"] - 8) & D.NO_SEQ,
                     applied_action_latency_ms=2)
        raw = list(struct.unpack("<6f", struct.pack("<6f", seq / 7, -seq / 9, 101, -101, 0.1, -0.0)))
        action = [max(-100.0, min(100.0, x)) for x in raw]
        records.append({"event": "sample", "outer_seq": seq, "header": h, "obs": obs,
                        "history": history, "raw_actor": raw, "published_action": action,
                        "latent": [0.2, 0.3, 0.4], "action_dropped": seq == drop})
        previous_action = action
        if seq == drop:
            fault = dict(h, state=D.FAULT, fault=2)
            records.append({"event": "fault", "header": fault})
            break
    stopped = header(session, reset)
    stopped.update(state=D.OFF, accepted_actions=seq if drop is not None else seq + 1)
    records.append({"event": "stop", "header": stopped})
    records.append({"event": "final_legacy", "status": {"flags": 0, "enabled": 0, "target": [0.] * 6}})
    return records


def expect_bad(records, label):
    report = D.verify_records(records)
    assert report["result"] == "fail", label + " was incorrectly accepted"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    passed = []
    before = header(reset=8)
    for seq in (D.NO_SEQ, 0):
        first = header(reset=9)
        first["sample_seq"] = seq
        D.require_fresh_start(first, before)
    for key, value in (("sample_seq", 1), ("accepted_actions", 1),
                       ("applied_action_source_seq", 0), ("reset_count", 8)):
        bad = header(reset=9)
        bad[key] = value
        try:
            D.require_fresh_start(bad, before)
        except RuntimeError:
            pass
        else:
            raise AssertionError("START incorrectly accepted stale " + key)
    passed.append("START accepts prepared sample zero, rejects stale action/session state")
    valid = fixture()
    assert D.verify_records(valid)["result"] == "pass"
    passed.append("110-frame FIFO/action/float32 clipping and uint32 clock wrap")
    for field, index, label in (("history", 0, "bad_history"), ("obs", 19, "wrong_previous_action")):
        bad = copy.deepcopy(valid)
        bad[6][field][index] += 0.25
        expect_bad(bad, label)
        passed.append(label)
    mutations = (("sample_seq", 77, "wrong_sequence"), ("sample_ms", 500, "bad_cadence"),
                 ("dm_enabled_mask", 1, "enabled_motor"), ("applied_action_latency_ms", 10, "late_action"),
                 ("session", 20, "wrong_session"))
    for key, value, label in mutations:
        bad = copy.deepcopy(valid)
        bad[6]["header"][key] = value
        expect_bad(bad, label)
        passed.append(label)
    bad = copy.deepcopy(valid)
    bad[1]["obs"][19] = 1.0
    expect_bad(bad, "uncleared_initial_action")
    bad = copy.deepcopy(valid)
    bad[6]["published_action"][2] = 0.0
    expect_bad(bad, "incorrect_clipping")
    bad = copy.deepcopy(valid)
    bad[6]["raw_actor"][0] = float("nan")
    expect_bad(bad, "nonfinite_actor")
    expect_bad(fixture(count=99), "too_short")
    passed.extend(["uncleared_initial_action", "incorrect_clipping", "nonfinite_actor", "minimum_100_frames"])
    dropped = fixture(count=110, drop=100)
    assert D.verify_records(dropped, 100)["result"] == "success(expected_fault)"
    assert D.verify_records(dropped, 99)["result"] == "fail"
    assert D.verify_records(dropped)["result"] == "fail"
    wrong_fault = copy.deepcopy(dropped)
    wrong_fault[-3]["header"]["fault"] = 6
    assert D.verify_records(wrong_fault, 100)["result"] == "fail"
    restarted = fixture(session=20, reset=2)
    # Independently recompute the restarted epoch's zero action and fill history.
    assert D.verify_records(restarted)["result"] == "pass"
    assert D.verify_records(valid + restarted)["result"] == "pass"
    assert D.verify_records(dropped + restarted, 100)["result"] == "success(expected_fault)"
    assert D.verify_records(valid + fixture())["result"] == "fail"
    bad_restart = copy.deepcopy(restarted)
    bad_restart[1]["history"] = valid[-3]["history"]
    expect_bad(bad_restart, "stale_history_after_restart")
    passed.extend(["expected_DEADLINE_fault", "wrong_fault_rejected", "new_session_restart_zero_fill", "session_reuse_rejected"])
    bad = copy.deepcopy(valid)
    bad[-2]["header"]["accepted_actions"] -= 1
    expect_bad(bad, "unacknowledged_final_action")
    expect_bad(valid + [{"event": "unprocessed_sample"}], "unprocessed_sample_at_stop")
    passed.extend(["unacknowledged_final_action", "unprocessed_sample_at_stop", "fault_then_restart_initialization"])
    sample = valid[1]
    payload = D.HEADER.pack(*(sample["header"][n] for n in D.HEADER_NAMES)) + D.FLOATS.pack(*(sample["obs"] + sample["history"]))
    wire = D.encode(D.DIAG_SAMPLE, 0, payload)
    assert len(payload) == 664 and len(wire) == 678 and D.ACTION.size == 32
    # Ground truth CRC excludes magic; legacy STATUS framing is shared.
    assert wire[:2] == b"\xa5\x5a" and struct.unpack("<I", wire[-4:])[0] == D.zlib.crc32(wire[2:-4])
    for split in range(len(wire) + 1):
        decoder = D.Decoder()
        frames = decoder.feed(wire[:split]) + decoder.feed(wire[split:])
        assert frames == [(D.DIAG_SAMPLE, 0, payload)], split
    decoder = D.Decoder()
    frames = []
    for byte in wire + wire:
        frames += decoder.feed(bytes([byte]))
    assert len(frames) == 2 and not decoder.buffer
    corrupt = bytearray(wire)
    corrupt[100] ^= 1
    decoder = D.Decoder()
    frames = decoder.feed(b"garbage\xa5" + bytes(corrupt) + wire + wire)
    assert len(frames) == 2 and decoder.crc_errors == 1
    decoder = D.Decoder()
    assert decoder.feed(b"\xa5\x5a" + D.WIRE_HEADER.pack(1, 0x90, 769, 0) + wire) == [(D.DIAG_SAMPLE, 0, payload)]
    assert decoder.header_errors == 1
    parsed = D.unpack_sample(0, payload)
    assert parsed["obs"] == sample["obs"] and parsed["history"] == sample["history"]
    passed.extend(["664-byte_payload_32-byte_action", "CRC_excludes_magic", "every_split_boundary", "bytewise_concatenated_frames", "bad_CRC_recovery", "oversize_header_recovery"])
    if args.checkpoint:
        assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == D.EXPECTED_SHA256
        infer = D.load_inference(args.checkpoint)
        raw, action, latent, start, end = infer(sample["obs"], sample["history"])
        assert D.finite(raw, 6) and D.finite(action, 6) and D.finite(latent, 3) and end >= start
        assert D.bits(action) == D.bits([max(-100., min(100., x)) for x in raw])
        passed.append("approved_real_model_encoder_actor_CPU_single_thread")
    print(json.dumps({"result": "pass", "checks": passed, "count": len(passed),
                      "scope": "offline protocol/inference checks; no serial, hardware or simulation"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
