#!/usr/bin/env python3
"""Offline 2701 action -> nominal virtual PD preview; never opens a device.

Uses recorded observations and published actions, not reconstructed motor angles.
The current sample and its action are paired for a hypothetical first PD step;
these torques were NOT sent to hardware. Later 2 ms PD steps are not recorded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import isaacgym  # Must precede torch imported by mj_sim2sim_ct.
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2sim import mj_sim2sim_ct as sim
from sim2sim.host_policy_diag import verify_records

URDF = ROOT / "resources/robots/chuanliantui_new_1/urdf/chuanliantui_train.urdf"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_versions():
    paths = (Path(__file__).resolve(), ROOT / "sim2sim/mj_sim2sim_ct.py",
             ROOT / "sim2sim/host_policy_diag.py", URDF,
             ROOT / "wheel_legged_gym/envs/base/legged_robot.py",
             ROOT / "wheel_legged_gym/envs/chuanliantui/chuanliantui_config.py",
             ROOT / "wheel_legged_gym/envs/chuanliantui_standup/chuanliantui_standup_config.py")
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def joint_limits():
    joints = {j.attrib["name"]: j for j in ET.parse(URDF).getroot().findall("joint")}
    return {name: {k: float(v) for k, v in joints[name].find("limit").attrib.items()}
            for name in sim.JOINT_NAMES}


def preview_sample(record):
    obs = np.asarray(record["obs"], dtype=np.float64)
    action = np.asarray(record["published_action"], dtype=np.float64)
    if obs.shape != (25,) or action.shape != (6,) or not np.all(np.isfinite(obs)) or not np.all(np.isfinite(action)):
        raise ValueError("invalid observation/action")
    # At the clipping boundary original q/dq cannot be recovered uniquely.
    if np.any(np.abs(obs[9:19]) >= sim.CLIP_OBS):
        raise ValueError("clipped joint observation cannot be inverted")
    if np.any(np.abs(action) > sim.CLIP_ACTIONS):
        raise ValueError("published action exceeds training clip")
    q = sim.DEFAULT_DOF_POS.copy()
    q[sim.LEG_POSITION_INDICES] += obs[9:13] / sim.OBS_SCALE_DOF_POS
    dq = obs[13:19] / sim.OBS_SCALE_DOF_VEL
    target_pos = sim.DEFAULT_DOF_POS.copy()
    target_pos[sim.LEG_POSITION_INDICES] += action[sim.LEG_POSITION_INDICES] * sim.POS_ACTION_SCALE
    target_vel = np.zeros(6)
    target_vel[[2, 5]] = action[[2, 5]] * sim.VEL_ACTION_SCALE
    before_clip = sim.P_GAINS * (target_pos - q) + sim.D_GAINS * (target_vel - dq)
    torque = sim.compute_torques(action, q, dq)
    # Independent reduced formula: q-default is already present in obs.
    expected = np.zeros(6)
    for pos_index, dof in enumerate((0, 1, 3, 4)):
        expected[dof] = 10.0 * (0.5 * action[dof] - obs[9 + pos_index]) - 20.0 * obs[13 + dof]
    for dof in (2, 5):
        expected[dof] = action[dof] - 2.0 * obs[13 + dof]
    expected = np.clip(expected, -sim.TORQUE_LIMITS, sim.TORQUE_LIMITS)
    if not np.allclose(torque, expected, rtol=0, atol=1e-10):
        raise ValueError("nominal training/preview PD mismatch; re-audit contract")
    h = record["header"]
    return {"session": h["session"], "source_sample_seq": h["sample_seq"],
            "source_sample_ms": h["sample_ms"], "published_action": action.tolist(),
            "leg_q_rad": q[sim.LEG_POSITION_INDICES].tolist(),
            "joint_dq_rad_s": dq.tolist(),
            "leg_target_rad": target_pos[sim.LEG_POSITION_INDICES].tolist(),
            "wheel_target_rad_s": target_vel[[2, 5]].tolist(),
            "virtual_torque_before_clip_nm": before_clip.tolist(),
            "virtual_torque_nm": torque.tolist(),
            "torque_clipped": (np.abs(before_clip) > sim.TORQUE_LIMITS).tolist()}


def ranges(values):
    a = np.asarray(values, dtype=np.float64)
    return {"min": a.min(axis=0).tolist(), "max": a.max(axis=0).tolist()}


def analyze(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    verification = verify_records(records)
    if verification["result"] != "pass":
        raise ValueError("{}: normal disabled run verification failed: {}".format(path, verification["errors"]))
    samples = [r for r in records if r["event"] == "sample"]
    rows = [preview_sample(r) for r in samples]
    limits = joint_limits()
    report = {"records": str(path), "records_sha256": digest(path),
              "input_verification": verification, "samples": len(rows),
              "torque_clipped_frames": np.asarray([r["torque_clipped"] for r in rows]).sum(axis=0).tolist()}
    for field in ("published_action", "leg_q_rad", "joint_dq_rad_s", "leg_target_rad",
                  "wheel_target_rad_s", "virtual_torque_before_clip_nm", "virtual_torque_nm"):
        report[field] = ranges([r[field] for r in rows])
    report["leg_target_outside_urdf_frames"] = {}
    report["leg_observation_outside_urdf_frames"] = {}
    for i, name in enumerate(("lf0", "lf1", "rf0", "rf1")):
        lo, hi = limits[name]["lower"], limits[name]["upper"]
        for field, key in (("leg_target_rad", "leg_target_outside_urdf_frames"),
                           ("leg_q_rad", "leg_observation_outside_urdf_frames")):
            report[key][name] = sum(not lo <= r[field][i] <= hi for r in rows)
    report["wheel_target_above_urdf_velocity_frames"] = {
        name: sum(abs(r["wheel_target_rad_s"][i]) > limits[name]["velocity"] for r in rows)
        for i, name in enumerate(("lfwheel", "rfwheel"))}
    metadata_path = path.with_name("summary.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")).get("metadata", {})
        report["input_summary_sha256"] = digest(metadata_path)
        report["recorded_checkpoint_sha256"] = metadata.get("checkpoint_sha256")
        report["recorded_host_code_sha256"] = metadata.get("host_code_sha256")
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new output directory, outside logs")
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT / "logs") == output or (ROOT / "logs") in output.parents or (ROOT / "wheel_legged_gym/logs") in output.parents:
        parser.error("historical logs are read-only")
    if output.exists():
        parser.error("output must be a new directory")
    results = [analyze(path.resolve()) for path in args.records]
    report = {"scope": "offline nominal virtual PD preview; no motor commands or motion prediction",
              "dof_order": sim.JOINT_NAMES, "leg_order": ["lf0", "lf1", "rf0", "rf1"],
              "wheel_order": ["lfwheel (physical right)", "rfwheel (physical left)"],
              "kp": sim.P_GAINS.tolist(), "kd": sim.D_GAINS.tolist(),
              "torque_limit_nm": sim.TORQUE_LIMITS.tolist(), "urdf_limits": joint_limits(),
              "source_sha256": source_versions(), "runs": [r for r, _ in results],
              "limitations": [
                  "Recorded published action paired with its source observation; not an actually executed torque.",
                  "Nominal PD only: training gain/offset/strength randomization is not reconstructed.",
                  "Only 100 Hz snapshots exist; this does not validate 500 Hz PD or actual control latency.",
                  "No same-sample raw DM angles; physical motor Jacobian torque is intentionally not fabricated.",
                  "Training does not clamp knee position targets or wheel velocity targets before PD by default.",
                  "Static disabled policy output and clipping do not establish closed-loop stability or bad observations.",
                  "Mapping, zeros and wheel ratio acceptance remain unchanged."]}
    output.mkdir(parents=True, exist_ok=False)
    for index, (_, rows) in enumerate(results):
        with (output / "preview-{}.jsonl".format(index)).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
