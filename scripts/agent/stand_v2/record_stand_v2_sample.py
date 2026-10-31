#!/usr/bin/env python3
"""Record one original stand_v2 diagnostic initial state with the same policy."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from ct_mjlab.envs import task_registry
from scripts.agent.stand_v2.diagnose_chuanliantui_stand_v2 import diagnose


def main(args):
    original = json.loads(args.report.read_text())
    assert 0 <= args.env < original["num_envs"]
    assert 0 < args.seconds <= original["duration_s"]
    source_make_env = task_registry.make_env
    frames = []

    def record_env(*positional, **keywords):
        wrapped, cfg = source_make_env(*positional, **keywords)
        raw = wrapped.unwrapped
        mujoco.mj_saveModel(raw.sim.mj_model, str(args.output/"scene.mjb"), None)
        frames.append(raw.sim.data.qpos[args.env].clone())
        step = wrapped.step

        def recorded_step(actions):
            result = step(actions)
            frames.append(raw.sim.data.qpos[args.env].clone())
            return result

        wrapped.step = recorded_step
        return wrapped, cfg

    task_registry.make_env = record_env
    try:
        run_args = argparse.Namespace(checkpoint=Path(original["checkpoint"]), output=args.output,
            num_envs=original["num_envs"], seconds=args.seconds, seed=original["seed"],
            device=args.device, allow_recovery_change=False)
        diagnose(run_args)
        assert len(frames) == round(args.seconds / original["policy_dt_s"])+1
        qpos = torch.stack(frames).cpu().numpy()
        report = json.loads((args.output/"result.json").read_text())
        assert report["checkpoint_sha256"] == original["checkpoint_sha256"]
        with np.load(args.report.parent/"measurements.npz") as old, np.load(args.output/"measurements.npz") as new:
            for name in ("initial_qpos", "initial_qvel", "weight_N", "fields"):
                assert np.array_equal(old[name], new[name]), name
            start = round((args.seconds-5.) / original["policy_dt_s"])
            original_speed = float(np.median(old["values"][start:len(frames)-1, args.env, 5]))
            recorded_speed = float(np.median(new["values"][start:, args.env, 5]))
            assert abs(recorded_speed-original_speed) < .1, "Original sample's motion did not reproduce"
        np.savez_compressed(args.output/"sample-qpos.npz", qpos=qpos,
            time_s=np.arange(len(frames))*original["policy_dt_s"])
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        result = dict(env=args.env, num_envs=original["num_envs"], seed=original["seed"],
            seconds=args.seconds, checkpoint_sha256=report["checkpoint_sha256"],
            original_report_sha256=sha(args.report), initial_states_match=True,
            original_same_window_median_speed_m_s=original_speed, recorded_median_speed_m_s=recorded_speed,
            actor_updates=0, frames=len(frames), qpos_sha256=sha(args.output/"sample-qpos.npz"),
            scope="Same 64 initial states, new first-episode rollout; not bitwise-identical replay")
        (args.output/"recording.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
    finally:
        task_registry.make_env = source_make_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", type=int, default=44)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
