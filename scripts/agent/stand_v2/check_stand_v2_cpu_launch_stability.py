#!/usr/bin/env python3
"""Screen all recorded starts for the reported early native CPU launch; no learning."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"sim2sim"))
from mj_sim2sim_chuanliantui_stand_v2 import StandV2Playback, OnnxPolicy
from mj_sim2sim_huananhu_multimode import sha256


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    env = StandV2Playback(args.initial_states)
    policy = OnnxPolicy(args.checkpoint, args.onnx)
    rows = []
    for sample in range(len(env.initial_qpos)):
        env.sample = sample
        env.reset()
        heights, speeds = [], []
        for _ in range(round(args.seconds/.01)):
            env.step(policy)
            heights.append(float(env.data.qpos[2]))
            speeds.append(abs(float(env.data.qvel[2])))
            if env.done:
                break
        rows.append(dict(sample=sample, max_height_m=max(heights), max_abs_vz_m_s=max(speeds),
            has_stood=bool(env.has_stood), final_height_m=float(env.data.qpos[2]),
            finite=bool(np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all())))
        if (sample+1) % 16 == 0:
            print(f"Screened {sample+1}/{len(env.initial_qpos)} starts; peak height {max(r['max_height_m'] for r in rows):.3f}m", flush=True)
    failed = [r["sample"] for r in rows if not r["finite"] or r["max_height_m"] > 1. or r["max_abs_vz_m_s"] > 5.]
    result = dict(passed=not failed, cases=len(rows), seconds_per_case=args.seconds,
        integration_dt_s=env.model.opt.timestep, pd_dt_s=.002, policy_dt_s=.01,
        parameter_updates=0, anomalous_sample_ids=failed,
        screen="Observed finite state, height<=1m, abs(world vz)<=5m/s; no state clipping or physics modification during measurement",
        scope="Early numeric launch screen only; not get-up success or full 40s balance acceptance",
        checkpoint_sha256=sha256(args.checkpoint), source_sha256=sha256(Path(__file__)), rows=rows)
    (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"},indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--onnx",type=Path,required=True)
    p.add_argument("--initial-states",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=5.)
    main(p.parse_args())
