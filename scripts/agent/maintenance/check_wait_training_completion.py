#!/usr/bin/env python3
"""Finite real-process checks; no training and no external model invocation."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

SOURCE = Path(__file__).with_name("wait_training_completion.py")
spec = importlib.util.spec_from_file_location("completion_waiter", SOURCE)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def run_checks():
    checks = []
    with tempfile.TemporaryDirectory(prefix="training-exit-check-") as temporary:
        root = Path(temporary)
        # Disposable fixture only; none of the real training PIDs are used.
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.3); raise SystemExit(7)"])
        try:
            binding = {"pid": child.pid, "starttime": monitor.process(child.pid)["starttime"],
                       "boot_id": monitor.BOOT_ID.read_text().strip()}
            assert monitor.exit_event(binding) is None
            assert monitor.exit_event(binding | {"starttime": binding["starttime"] + 1}) == "pid_reused"
            assert child.poll() is None
            checks.append("PID reuse detected without signalling the watched process")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                assert monitor.watch(child.pid, root / "wait", .01) == 0
            assert child.wait() == 7
            event = json.loads(output.getvalue())
            assert event["event"] == "process_exited" and event["status"] == "finished"
            assert event["training_exit_code"] is None
            checks.append("real exit returns one tool event; nonzero child exit is not claimed as success")
            assert json.loads((root / "wait/status.json").read_text()) == event
            checks.append("completion event persisted for the calling agent")
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait()
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            with patch.object(monitor, "wait_for_exit", side_effect=KeyboardInterrupt):
                with contextlib.redirect_stdout(io.StringIO()):
                    assert monitor.watch(other.pid, root / "interrupted", .01) == 130
            assert other.poll() is None
            checks.append("interrupted waiter leaves the watched process running")
            try:
                monitor.watch(other.pid, root / "wait", .01)
            except FileExistsError:
                checks.append("existing output cannot overwrite or duplicate a recorded event")
            else:
                raise AssertionError("Existing wait output was overwritten")
        finally:
            other.terminate()
            other.wait()
    return {"passed": True, "checks": checks,
            "scope": "real disposable processes and interrupted waiting; no training, peer CLI or external model"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    result = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
