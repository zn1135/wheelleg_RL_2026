#!/usr/bin/env python3
"""Record a full quiet rollout for diagnosis; never issue an acceptance verdict."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "mjlab_training/src"))
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from mjlab.managers.metrics_manager import MetricsTermCfg
from scripts.agent.stand import assess_chuanliantui_stand as assess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("nominal", "randomized"), default="nominal")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.num_envs < 1 or not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("Environment count and duration must be positive")
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, agent = task_registry.get_cfgs("chuanliantui_stand", play=args.profile == "nominal")
    from ct_mjlab.envs.chuanliantui_stand.env_cfg import configure_evaluation
    configure_evaluation(cfg)
    for key in ("stand_disturbance", "huananhu_velocity_push", "huananhu_local_wrench"):
        cfg.events.pop(key, None)
    cfg.commands["standup"].disturbances = False
    cfg.auto_reset, cfg.seed = False, args.seed
    cfg.metrics["diagnostic_physics"] = MetricsTermCfg(
        func=assess.sample_physics, per_substep=True, reduce="last")
    wrapped, cfg = task_registry.make_env("chuanliantui_stand", env_cfg=cfg, train_cfg=agent,
        num_envs=args.num_envs, seed=args.seed, device=args.device)
    env = wrapped.unwrapped
    frames, actions, flags, resets = [], [], [], []
    error = None
    try:
        assert abs(env.physics_dt - .002) < 1e-10 and abs(env.step_dt - .01) < 1e-10
        steps = max(1, round(args.seconds / env.step_dt))
        recorder = assess.PhysicsRecorder(env, steps * cfg.decimation)
        env._stand_acceptance_recorder = recorder
        runner, _ = task_registry.make_alg_runner(wrapped, "chuanliantui_stand",
            train_cfg=agent, device=args.device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        obs, _ = env.reset(seed=args.seed)
        frames.append(env.sim.data.qpos.cpu().numpy().copy())
        state = env.command_manager.get_term("standup")
        mujoco.mj_saveModel(env.sim.mj_model, str(args.output / "scene.mjb"), None)
        with torch.inference_mode():
            for tick in range(steps):
                try:
                    action = policy(obs)
                    obs, _, terminated, timeout, _ = env.step(action)
                    actions.append(action.cpu().numpy().copy())
                    frames.append(env.sim.data.qpos.cpu().numpy().copy())
                    flags.append(torch.stack((terminated, timeout, state.recover,
                        state.fallen, state.airborne), -1).cpu().numpy().copy())
                    ids = (terminated | timeout).nonzero(as_tuple=False).flatten()
                    if len(ids):
                        resets.append({"step": tick + 1, "env_ids": ids.cpu().tolist()})
                        env.reset(env_ids=ids)
                        obs = env.get_observations()
                except (RuntimeError, AssertionError, FloatingPointError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    break
        # Preserve every 2 ms sample even after an acceptance-style early failure.
        raw = recorder.trace[:recorder.count].cpu().numpy()
        np.savez_compressed(args.output / "trajectory.npz", physics=raw,
            physics_time_s=(np.arange(len(raw)) + 1) * env.physics_dt,
            physics_channels=np.asarray(assess.CHANNELS), qpos=np.asarray(frames),
            policy_time_s=np.arange(len(frames)) * env.step_dt, action=np.asarray(actions),
            flags=np.asarray(flags), flag_names=np.asarray(
                ("terminated", "timeout", "recover", "fallen", "airborne")))
        metadata = {"kind": "diagnostic_only; no acceptance verdict", "profile": args.profile,
            "seed": args.seed, "physics_dt_s": env.physics_dt, "policy_dt_s": env.step_dt,
            "checkpoint": assess.identity(args.checkpoint), "xml": assess.identity(MODEL_XML),
            "fixed_command": [0., 0., .27], "steps_captured": len(frames) - 1,
            "steps_planned": steps, "resets": resets, "error": error,
            "configuration": asdict(cfg)}
        (args.output / "metadata.json").write_text(json.dumps(metadata,
            default=assess.json_default, indent=2) + "\n")
        print(json.dumps({key: metadata[key] for key in
            ("kind", "steps_captured", "steps_planned", "resets", "error")}))
    finally:
        env.close()
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
