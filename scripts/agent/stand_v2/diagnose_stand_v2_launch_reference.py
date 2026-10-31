#!/usr/bin/env python3
"""Run the reported saved state in the actual training engine without learning."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_stand_v2.observations import get_state


def capture(raw):
    samples = getattr(raw, "_launch_trace", None)
    if samples is not None:
        d = raw.sim.data
        samples.append(torch.stack((d.qpos[0, 2], d.qvel[0, 2], d.qfrc_constraint[0].abs().max(),
            d.qfrc_actuator[0].abs().max(), *(-raw.scene["wheel_contact"].data.force[0, :, 2]).unbind())))
    return torch.zeros(raw.num_envs, device=raw.device)


def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, agent = task_registry.get_cfgs("chuanliantui_stand_v2", play=True)
    cfg.metrics["launch_trace"] = MetricsTermCfg(func=capture, per_substep=True, reduce="last")
    env, _ = task_registry.make_env("chuanliantui_stand_v2", env_cfg=cfg, train_cfg=agent,
        num_envs=1, seed=42, device="cuda:0")
    try:
        runner, _ = task_registry.make_alg_runner(env, "chuanliantui_stand_v2", train_cfg=agent, device="cuda:0")
        runner.load(str(args.checkpoint), load_cfg={"actor":True}, map_location="cuda:0")
        raw = env.unwrapped
        raw.reset()
        with np.load(args.initial_states) as saved:
            q, v = saved["qpos"][args.sample].copy(), saved["qvel"][args.sample].copy()
        q[:2] = 0.
        raw.sim.data.qpos[0] = torch.tensor(q, device=raw.device)
        raw.sim.data.qvel[0] = torch.tensor(v, device=raw.device)
        raw.sim.forward(); raw.sim.sense()
        get_state(raw).reset(torch.tensor([0], device=raw.device))
        raw.observation_manager.reset()
        raw.obs_buf = raw.observation_manager.compute(update_history=True)
        policy = runner.get_inference_policy(device=raw.device)
        obs = env.get_observations()
        raw._launch_trace = []
        actions = []
        with torch.inference_mode():
            for _ in range(round(args.seconds/.01)):
                action = policy(obs)
                actions.append(action[0].clone())
                obs, _, done, _ = env.step(action)
                if done.any():
                    break
        trace = torch.stack(raw._launch_trace).cpu().numpy()
        np.savez_compressed(args.output/"trace.npz", trace=trace, actions=torch.stack(actions).cpu().numpy(),
            fields=["height", "vz", "constraint_force", "actuator_force", "left_support", "right_support"])
        result = dict(sample=args.sample, seconds=len(trace)*.002, peak_height_m=float(trace[:, 0].max()),
            max_upward_speed_m_s=float(trace[:, 1].max()), max_constraint_force=float(trace[:, 2].max()),
            ever_stood=bool(raw.command_manager.get_term("standup").has_stood[0]), parameter_updates=0)
        (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))
        # Preserve compiled reference fields for name-based CPU model comparisons.
        import mujoco
        mujoco.mj_saveModel(raw.sim.mj_model, str(args.output/"reference.mjb"), None)
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--initial-states", type=Path, required=True)
    p.add_argument("--sample", type=int, default=17)
    p.add_argument("--seconds", type=float, default=3.)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
