#!/usr/bin/env python3
"""合成 2901 帧检查主机协议与人工接管边界；不打开串口。"""
import struct
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from collections import deque
from unittest.mock import patch

from sim2sim import host_policy_run as run
from sim2sim.host_policy_diag import Decoder, Transport, encode


class Fake2901Transport:
    interrupt_after = None
    stop_fail = False
    start_reject = False
    board_auto_stop = False
    stop_masks = (0,)
    board_remote_fault = False
    capture_feedback = False
    feedback_tail = False
    latest = None

    def __init__(self, path, allowed_kinds=None):
        Fake2901Transport.latest = self
        self.path = path
        self.allowed_kinds = allowed_kinds
        self.session = 0
        self.remote_queries = 0
        self.sample_index = 0
        self.contact_seen = False
        self.stop_seen = False
        self.global_stop_seen = False
        self.disarm_polls = 0
        self.pending = deque()
        self.feedback_reply = None
        self.commands = []
        self.decoder = type("DecoderState", (), dict(crc_errors=0, header_errors=0,
                                discarded_bytes=0, buffer=b""))()

    def _header(self, phase, seq=run.NO_SEQ, enabled=0, contact=run.NO_SEQ):
        values = (run.LAYOUT, self.session, seq, 100 + (seq if seq != run.NO_SEQ else 0) * 10,
                  101 + (seq if seq != run.NO_SEQ else 0) * 10,
                  run.NO_SEQ if seq in (run.NO_SEQ, 0) else seq - 1,
                  0 if seq in (run.NO_SEQ, 0) else 102 + (seq - 1) * 10,
                  phase, 0, 63, enabled, 10, contact, seq * 5 if seq != run.NO_SEQ else 0,
                  0, 0)
        return struct.pack("<16I", *values)

    def request(self, kind, response_type, payload=b"", timeout=1.0):
        if kind == 1:
            status = bytearray(132)
            status[8] = 63
            if self.stop_seen and not self.global_stop_seen:
                self.disarm_polls += 1
                status[9] = self.stop_masks[min(self.disarm_polls,
                                               len(self.stop_masks) - 1)]
            return response_type, 1, bytes(status), 0
        if kind == 5:
            self.remote_queries += 1
            remote = bytearray(64)
            struct.pack_into("<I", remote, 0, 2602)
            remote[32] = 1
            remote[33] = 2 if self.remote_queries == 1 else 1
            remote[34] = 2
            remote[35] = 0 if self.remote_queries == 1 else 1
            remote[37] = 0 if self.remote_queries == 1 else 1
            remote[39] = 0 if self.remote_queries == 1 else 15
            remote[51] = 1
            struct.pack_into("<I", remote, 52, 63)
            return response_type, 1, bytes(remote), 0
        if kind == run.POLICY_QUERY:
            return response_type, 2, self._header(run.OFF), 0
        if kind == run.POLICY_START:
            self.session = struct.unpack_from("<I", payload, 4)[0]
            if self.start_reject:
                return response_type, 3, self._header(run.OFF), 0
            return response_type, 3, self._header(run.PRECONTACT, enabled=15), 0
        if kind == run.POLICY_STOP:
            if self.stop_fail:
                raise TimeoutError("synthetic POLICY_STOP reply loss")
            assert struct.unpack("<I", payload)[0] == self.session
            self.stop_seen = True
            return response_type, 4, self._header(run.OFF, enabled=self.stop_masks[0]), 0
        raise AssertionError("unexpected request {}".format(kind))

    def send(self, kind, payload=b""):
        assert kind in self.allowed_kinds
        self.commands.append(kind)
        if self.stop_seen:
            assert kind == run.GLOBAL_STOP, "cleanup must never resume motion commands"
        if kind == run.POLICY_CONTACT:
            assert self.sample_index == 1
            self.contact_seen = True
        elif kind == run.POLICY_ACTION:
            assert struct.unpack_from("<I", payload, 0)[0] == self.session
        elif kind == run.GLOBAL_STOP:
            self.stop_seen = True
            self.global_stop_seen = True
        elif kind == run.STATUS:
            assert self.capture_feedback and self.commands[-2] == run.POLICY_ACTION
            assert self.feedback_reply is None
            self.feedback_reply = (run.LEGACY_STATUS, 9 + self.sample_index,
                                   motor_feedback_payload(), 1000010)
        else:
            raise AssertionError("unexpected command")
        return 9 + self.sample_index

    def receive(self, timeout):
        if self.feedback_reply is not None:
            packet = self.feedback_reply
            self.feedback_reply = None
            if self.feedback_tail:
                self.pending.append(packet)
            else:
                return packet
        if self.interrupt_after is not None and self.sample_index >= self.interrupt_after:
            raise KeyboardInterrupt("synthetic host interruption")
        seq = self.sample_index
        if self.capture_feedback and seq == 6:
            return run.LEGACY_STATUS, 999, motor_feedback_payload(), 1000020
        if self.board_remote_fault and seq == 3:
            payload = bytearray(self._header(run.FAULT, enabled=15))
            struct.pack_into("<I", payload, 32, 6)  # REMOTE
            return run.POLICY_STATUS, 30, bytes(payload), 1000003
        if self.board_auto_stop and seq == 3:
            payload = bytearray(self._header(run.OFF))
            struct.pack_into("<I", payload, 56, 1 << 16)
            return run.POLICY_STATUS, 30, bytes(payload), 1000003
        assert seq < (6 if self.capture_feedback else 3)
        if seq > 0:
            assert self.contact_seen
        phase = (run.PRECONTACT, run.CONTACT_EDGE, run.ACTIVE)[min(seq, 2)]
        contact = run.NO_SEQ if seq == 0 else 0
        payload = self._header(phase, seq, 15, contact) + struct.pack("<150f", *([0.0] * 150))
        self.sample_index += 1
        return run.POLICY_SAMPLE, seq, payload, 1000000 + seq

    def close(self):
        pass


def check_full_host_cleanup(interrupt_after=None, stop_fail=False, start_reject=False,
                            board_auto_stop=False, preconfirmed=False,
                            stop_masks=(0,), board_remote_fault=False,
                            motor_feedback=False, feedback_tail=False):
    with tempfile.TemporaryDirectory(prefix="h7-2901-offline-") as temp:
        root = Path(temp)
        checkpoint = root / "model_6000.pt"
        checkpoint.write_bytes(b"synthetic full checkpoint for transport test")
        firmware = root / "firmware.elf"
        firmware.write_bytes(b"synthetic firmware identity")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        args = argparse.Namespace(out=root / "capture", checkpoint=checkpoint,
                                  firmware_elf=firmware,
                                  port=Path("/dev/serial/by-id/fake2901"), seconds=1.0,
                                  contact_confirmed_at_start=preconfirmed,
                                  motor_feedback=motor_feedback)
        Fake2901Transport.interrupt_after = interrupt_after
        Fake2901Transport.stop_fail = stop_fail
        Fake2901Transport.start_reject = start_reject
        Fake2901Transport.board_auto_stop = board_auto_stop
        Fake2901Transport.stop_masks = stop_masks
        Fake2901Transport.board_remote_fault = board_remote_fault
        Fake2901Transport.capture_feedback = motor_feedback
        Fake2901Transport.feedback_tail = feedback_tail
        keys = iter((None, None if preconfirmed else "c", None, None,
                     None if board_auto_stop or board_remote_fault else "q"))
        if motor_feedback:
            keys = iter((None, "c", None, None, None, None, None, None, "q"))
        clock = iter(i * 0.001 for i in range(10000))
        with patch.object(run, "CHECKPOINT_SHA256", digest), \
             patch.object(run, "Transport", Fake2901Transport), \
             patch.object(run, "load_inference", lambda _: (
                 lambda obs, history: ([0.0] * 6, [0.0] * 6, [0.0] * 3, 1, 2))), \
             patch.object(run, "read_key", lambda: next(keys, None)), \
             patch.object(run.sys.stdin, "isatty", lambda: True), \
             patch.object(run.termios, "tcgetattr", lambda _: []), \
             patch.object(run.termios, "tcsetattr", lambda *args: None), \
             patch.object(run.tty, "setcbreak", lambda _: None), \
             patch.object(run.time, "monotonic", lambda: next(clock)), \
             patch.object(run.time, "sleep", lambda _: None):
            result = run.run(args)
        assert result["stop_confirmed"] == (not stop_fail and not start_reject)
        assert result["final_disabled"]
        assert result.get("board_auto_stopped", False) == board_auto_stop
        assert result["contact_sent"] == (not start_reject)
        disarm_timeout = stop_masks[-1] != 0
        assert result["result"] == ("fail" if interrupt_after is not None or stop_fail or start_reject
                                    or disarm_timeout or board_remote_fault
                                    else "protocol_pass")
        if stop_fail:
            assert "global_stop_fallback" in (json.loads(line)["event"] for line in
                       (args.out / "records.jsonl").read_text().splitlines())
        assert (args.out / "records.jsonl").is_file()
        records = [json.loads(line) for line in (args.out / "records.jsonl").read_text().splitlines()]
        feedback = [row for row in records if row["event"] == "motor_feedback"]
        if motor_feedback:
            assert len(feedback) == 1
            assert feedback[0]["source_sample_seq"] == 4
            assert feedback[0]["status"]["dm_torque_nm"] == [9., -10., 11., -12.]
            assert feedback[0]["status"]["wheel_current_raw"] == [-1234, 2345]
            assert feedback[0]["status"]["target"] == [13., 14., 15., 16., 17., 18.]
            assert result["samples"] == 6
            assert result["motor_feedback"]["received"] == 1
            assert result["motor_feedback"]["unfinished"] == 0
            assert [row["header"]["sample_seq"] for row in records
                    if row["event"] == "sample"] == [0, 1, 2, 3, 4, 5]
        else:
            assert not feedback
            assert run.STATUS not in Fake2901Transport.latest.commands
        if stop_masks[0]:
            polls = [row for row in records if row["event"] == "stop_disarm_poll"]
            assert polls, "nonzero STOP feedback must be checked with bounded read-only polls"
            if disarm_timeout:
                assert len(polls) > 1
                assert any("50 ms" in failure for failure in result["failures"])
                assert any(row["event"] == "global_stop_fallback" for row in records)
            else:
                assert polls[-1]["status"]["enabled"] == 0
                assert not any("cleanup:" in failure for failure in result["failures"])
                assert not any(row["event"] == "global_stop_fallback" for row in records)
        if board_remote_fault:
            faults = [row for row in records if row["event"] == "status"
                      and row["header"]["phase"] == run.FAULT]
            assert len(faults) == 1 and faults[0]["header"]["fault"] == 6
            assert any("board 2901 FAULT" in failure for failure in result["failures"])
        if not start_reject:
            contacts = [row for row in records if row["event"] == "contact_tx"]
            assert len(contacts) == 1
            contact_frame = Decoder().feed(bytes.fromhex(contacts[0]["raw_frame_hex"]))
            assert contact_frame == [(run.POLICY_CONTACT, contacts[0]["outer_seq"],
                                      struct.pack("<II", result["metadata"]["session"], 0))]
            samples = [row for row in records if row["event"] == "sample"]
            assert samples
            for row in samples:
                action_frame = Decoder().feed(bytes.fromhex(row["raw_action_frame_hex"]))
                assert len(action_frame) == 1
                kind, outer_seq, payload = action_frame[0]
                assert kind == run.POLICY_ACTION and outer_seq == row["host_action_outer_seq"]
                assert struct.unpack_from("<II", payload) == (result["metadata"]["session"],
                                                               row["header"]["sample_seq"])
                assert struct.unpack_from("<6f", payload, 8) == tuple(row["published_action"])
        assert (args.out / "summary.json").is_file()


def expect_error(fn, error=ValueError):
    try:
        fn()
    except error:
        return
    raise AssertionError("expected {}".format(error.__name__))


def motor_feedback_payload():
    payload = bytearray(132)
    struct.pack_into("<II", payload, 0, 5678, 3)
    payload[8:12] = bytes((63, 15, 0, 0))
    struct.pack_into("<4f", payload, 16, 1., 2., 3., 4.)
    struct.pack_into("<4f", payload, 32, 5., 6., 7., 8.)
    struct.pack_into("<4f", payload, 48, 9., -10., 11., -12.)
    struct.pack_into("<2f", payload, 64, 21., 22.)
    struct.pack_into("<2f", payload, 72, 23., 24.)
    struct.pack_into("<2h", payload, 80, -1234, 2345)
    struct.pack_into("<3f", payload, 84, 25., 26., 27.)
    struct.pack_into("<3f", payload, 96, 28., 29., 30.)
    struct.pack_into("<6f", payload, 108, 13., 14., 15., 16., 17., 18.)
    return bytes(payload)


def check_feedback_missing_and_fields():
    class SendOnly:
        def __init__(self):
            self.sent = []

        def send(self, kind):
            assert kind == run.STATUS
            self.sent.append(kind)
            return 100 + len(self.sent)

    link, records = SendOnly(), []
    capture = run.MotorFeedbackCapture()
    for seq in range(20):
        capture.after_action(link, seq, records)
    assert len(link.sent) == 2  # sample 4 times out at sample 14; never two in flight
    assert capture.summary()["missing"] == 1
    assert capture.summary()["unfinished"] == 1
    assert not capture.accept((run.LEGACY_STATUS, 101, motor_feedback_payload(), 500), records)
    assert not capture.accept((run.POLICY_SAMPLE, 102, motor_feedback_payload(), 500), records)
    assert capture.accept((run.LEGACY_STATUS, 102, motor_feedback_payload(), 501), records)
    assert capture.summary()["received"] == 1 and capture.summary()["unfinished"] == 0
    row = records[-1]
    assert row["source_sample_seq"] == 14 and row["host_recv_ns"] == 501
    assert Decoder().feed(bytes.fromhex(row["raw_frame_hex"])) == [
        (run.LEGACY_STATUS, 102, motor_feedback_payload())]
    got = row["status"]
    for name, expected in (("dm_pos_rad", [1., 2., 3., 4.]),
                           ("dm_vel_rad_s", [5., 6., 7., 8.]),
                           ("dm_torque_nm", [9., -10., 11., -12.]),
                           ("wheel_angle_rad", [21., 22.]),
                           ("wheel_vel_rad_s", [23., 24.]),
                           ("wheel_current_raw", [-1234, 2345]),
                           ("euler_rad", [25., 26., 27.]),
                           ("gyro_rad_s", [28., 29., 30.]),
                           ("target", [13., 14., 15., 16., 17., 18.])):
        assert got[name] == expected
    assert (got["uptime"], got["flags"], got["enabled"], got["online"]) == (5678, 3, 15, 63)


def check_disarm_poll_preserves_transport_queue():
    # Exercise the real request() FIFO behavior without opening a serial port.
    link = Transport.__new__(Transport)
    first = (run.POLICY_STATUS, 41, b"queued current-session fault", 9001)
    other = (run.POLICY_STATUS, 42, b"queued other-session status", 9002)
    link.pending = deque((first, other))
    masks = iter((8, 0))

    def send_status(kind, payload=b""):
        assert kind == run.STATUS and payload == b""
        status = bytearray(132)
        status[8] = 63
        status[9] = next(masks)
        link.pending.append((run.LEGACY_STATUS, 43, bytes(status), 9003))
        return 43

    link.send = send_status
    records = []
    run.wait_for_stop_disarm(link, records)
    assert list(link.pending) == [first, other]
    assert [row["status"]["enabled"] for row in records] == [8, 0]


def sample(seq, phase, session=17, layout=2901):
    meta = (layout, session, seq, 100 + seq * 10, 101 + seq * 10,
            0xffffffff if seq == 0 else seq - 1, 0 if seq == 0 else 102 + (seq - 1) * 10,
            phase, 0, 63, 15, 10,
            0 if phase in (run.CONTACT_EDGE, run.ACTIVE) else 0xffffffff,
            seq * 5, 0, 0)
    payload = struct.pack("<16I150f", *(meta + tuple([0.0] * 150)))
    return run.unpack_sample(seq, payload, 17)


def main():
    digest = run.CHECKPOINT_SHA256
    assert run.start_payload(17, digest) == struct.pack("<II", 2901, 17) + bytes.fromhex(digest)
    expect_error(lambda: run.start_payload(17, "0" * 64))
    assert run.contact_payload(17, 2) == struct.pack("<II", 17, 2)
    first = sample(0, run.PRECONTACT)
    assert first["header"]["phase"] == run.PRECONTACT
    assert first["obs"] == [0.0] * 25 and first["history"] == [0.0] * 125
    expect_error(lambda: sample(0, run.PRECONTACT, layout=2701))
    expect_error(lambda: sample(0, run.PRECONTACT, session=18))
    actor = [1, -2, 3, -4, 5, -6]
    assert run.action_for_phase(run.PRECONTACT, actor) == [0.0] * 6
    assert run.action_for_phase(run.CONTACT_EDGE, actor) == [0.0] * 6
    assert run.action_for_phase(run.ACTIVE, actor) == actor
    expect_error(lambda: run.action_for_phase(run.FAULT, actor))
    expect_error(lambda: run.action_for_phase(run.ACTIVE, [0.0] * 5))
    expect_error(lambda: run.action_for_phase(run.ACTIVE, [float("nan")] + [0.0] * 5))
    assert run.action_for_phase(run.ACTIVE, [1000, -1000, 0, 0, 0, 0])[:2] == [100.0, -100.0]
    wire = encode(run.POLICY_SAMPLE, 0, struct.pack("<16I150f", *(
        tuple(first["header"][name] for name in run.HEADER_NAMES) + tuple([0.0] * 150))))
    decoder = Decoder()
    assert decoder.feed(wire[:3]) == []
    assert decoder.feed(wire[3:]) == [(run.POLICY_SAMPLE, 0, wire[10:-4])]
    corrupt = bytearray(wire)
    corrupt[-1] ^= 1
    assert Decoder().feed(corrupt + wire)[0][0] == run.POLICY_SAMPLE
    session = run.PolicySession(17)
    contact, packet, action = session.accept(first, actor, contact_requested=True)
    assert contact == struct.pack("<II", 17, 0)
    assert packet == struct.pack("<II6f", 17, 0, *([0.0] * 6))
    assert action == [0.0] * 6
    edge = sample(1, run.CONTACT_EDGE)
    contact, packet, action = session.accept(edge, actor)
    assert contact is None and action == [0.0] * 6
    active = sample(2, run.ACTIVE)
    contact, packet, action = session.accept(active, actor)
    assert contact is None and action == actor
    expect_error(lambda: session.accept(active, actor))
    expect_error(lambda: run.PolicySession(17).accept(sample(1, run.ACTIVE), actor))
    late_tx = sample(0, run.PRECONTACT)
    late_tx["header"]["tx_ms"] = late_tx["header"]["sample_ms"] + 10
    expect_error(lambda: run.PolicySession(17).accept(late_tx, actor))
    disabled = sample(0, run.PRECONTACT)
    disabled["header"]["dm_enabled_mask"] = 0
    expect_error(lambda: run.PolicySession(17).accept(disabled, actor))
    dry = run.PolicySession(17, expected_enabled_mask=0, expected_dry=True)
    disabled["header"]["guard_flags"] = 1 << 14
    assert dry.accept(disabled, actor)[2] == [0.0] * 6
    missing_dry_flag = sample(0, run.PRECONTACT)
    missing_dry_flag["header"]["dm_enabled_mask"] = 0
    expect_error(lambda: run.PolicySession(17, expected_enabled_mask=0,
                                           expected_dry=True).accept(missing_dry_flag, actor))
    stale_action = sample(1, run.PRECONTACT)
    stale_action["header"]["applied_action_rx_ms"] = stale_action["header"]["sample_ms"]
    previous = run.PolicySession(17)
    previous.accept(first, actor)
    expect_error(lambda: previous.accept(stale_action, actor))
    check_full_host_cleanup()
    check_full_host_cleanup(interrupt_after=2)
    check_full_host_cleanup(stop_fail=True)
    check_full_host_cleanup(start_reject=True)
    check_full_host_cleanup(board_auto_stop=True)
    check_full_host_cleanup(preconfirmed=True)
    check_full_host_cleanup(stop_masks=(8, 8, 0))
    check_full_host_cleanup(stop_masks=(8, 8))
    check_full_host_cleanup(stop_masks=(8, 0), board_remote_fault=True)
    check_disarm_poll_preserves_transport_queue()
    check_full_host_cleanup(motor_feedback=True)
    check_full_host_cleanup(motor_feedback=True, feedback_tail=True)
    check_feedback_missing_and_fields()
    print("2901 offline framing, layout, hash, phase and action checks passed")


if __name__ == "__main__":
    main()
