#!/usr/bin/env python3
"""Wait for a Linux training process and return its exit event to this agent.

Run as a live tool call: its completion resumes the calling agent directly.
There is no model polling, peer CLI, periodic task or training control action.
"""
from datetime import datetime
import argparse
import json
from pathlib import Path
import sys
import time

BOOT_ID = Path("/proc/sys/kernel/random/boot_id")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def process(pid):
    try:
        # comm may contain spaces or ')'; starttime is field 22 after final ')'.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"starttime": int(fields[19]), "state": fields[0]}


def exit_event(binding):
    if BOOT_ID.read_text().strip() != binding["boot_id"]:
        return "boot_changed"
    current = process(binding["pid"])
    if current is None or current["state"] in ("Z", "X"):
        return "process_exited"
    if current["starttime"] != binding["starttime"]:
        return "pid_reused"
    return None


def wait_for_exit(binding, poll_seconds=15.):
    while (event := exit_event(binding)) is None:
        time.sleep(poll_seconds)  # Only the waiting tool wakes up; no model call.
    return event


def watch(pid, output, poll_seconds=15.):
    identity = process(pid)
    if identity is None or identity["state"] in ("Z", "X"):
        raise ValueError("Training PID is not a live process")
    binding = {"pid": pid, "starttime": identity["starttime"],
               "boot_id": BOOT_ID.read_text().strip()}
    # Unique output prevents a second wait from overwriting the first event.
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "waiting", "started_at": now(), "binding": binding,
              "event": None, "training_exit_code": None}
    save(output / "status.json", result)
    try:
        result.update(status="finished", event=wait_for_exit(binding, poll_seconds))
    except KeyboardInterrupt:
        result.update(status="interrupted")
    result["ended_at"] = now()
    result["note"] = "Return this event to the calling agent; verify logs and checkpoint before claiming success."
    save(output / "status.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "finished" else 130


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True, help="Existing training Python process ID")
    parser.add_argument("--output", type=Path, required=True, help="New wait artifact directory")
    args = parser.parse_args()
    if args.pid <= 0:
        parser.error("PID must be positive")
    try:
        return watch(args.pid, args.output.resolve())
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
