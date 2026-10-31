#!/usr/bin/env python3
"""Measure first-episode stand_v2 motion at 100 Hz; no learning or behavior acceptance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.envs.chuanliantui_stand_v2.diagnostics import BASE_CONTACT_THRESHOLD_N
from ct_mjlab.envs.chuanliantui_stand_v2.mdp import RECOVERY_CONTRACT, LEGACY_RECOVERY_CONTRACT


FIELDS = ("x", "y", "height", "roll", "pitch", "planar_speed", "max_leg_speed",
          "left_wheel_support", "right_wheel_support", "base_force", "roll_pitch_speed",
          "isaac_has_stood", "recover_mode", "velocity_error")


from scripts.agent.stand_v2._contact_diagnostics import contact_summary, reprocess_contacts as _reprocess_contacts


def reprocess_contacts(args):
    # 来源仍记录调用入口，同时共享统计实现。
    return _reprocess_contacts(args, Path(__file__))


def diagnose(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, agent = task_registry.get_cfgs("chuanliantui_stand_v2", play=True)
    cfg.auto_reset = False
    env, _ = task_registry.make_env("chuanliantui_stand_v2", env_cfg=cfg, train_cfg=agent,
        play=True, num_envs=args.num_envs, seed=args.seed, device=args.device)
    try:
        runner, _ = task_registry.make_alg_runner(env, "chuanliantui_stand_v2",
            train_cfg=agent, device=args.device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device,
                    allow_recovery_change=args.allow_recovery_change)
        policy = runner.get_inference_policy(device=args.device)
        raw, n, dt = env.unwrapped, env.num_envs, env.unwrapped.step_dt
        state, robot = raw.command_manager.get_term("standup"), raw.scene["robot"]
        leg_ids, _ = robot.find_joints(("lf0", "lf00", "rf0", "rf00"), preserve_order=True)
        weight = raw.sim.model.body_mass.sum(-1) * np.linalg.norm(raw.sim.mj_model.opt.gravity)
        steps = round(args.seconds / dt)
        samples = torch.zeros(steps, n, len(FIELDS), device=args.device)
        active = torch.zeros(steps, n, dtype=torch.bool, device=args.device)
        live = torch.ones(n, dtype=torch.bool, device=args.device)
        ever_stood = torch.zeros_like(live)
        balance_run = torch.zeros(n, device=args.device)
        quiet_run = torch.zeros_like(balance_run)
        termination_time = torch.full_like(balance_run, float("nan"))
        diagnostic_resets = []
        initial_qpos, initial_qvel = raw.sim.data.qpos.clone(), raw.sim.data.qvel.clone()
        target = state.command[:, 2].clone()
        obs = env.get_observations()
        with torch.inference_mode():
            for step in range(steps):
                active[step] = live
                velocity_error = (policy.encoder(obs["history"]) / 2.
                    - robot.data.root_com_lin_vel_b).square().mean(-1).sqrt()
                obs, _, done, _ = env.step(policy(obs))
                data = robot.data
                gravity = data.projected_gravity_b
                roll = torch.atan2(-gravity[:, 1], -gravity[:, 2]).abs()
                pitch = gravity[:, 0].clamp(-1., 1.).asin().abs()
                support = -raw.scene["wheel_contact"].data.force[:, :, 2]
                base = raw.scene["base_contact"].data.force.norm(dim=-1).amax(-1)
                speed = data.root_link_lin_vel_b[:, :2].norm(dim=-1)
                leg_speed = data.joint_vel[:, leg_ids].abs().amax(-1)
                angular_speed = data.root_link_ang_vel_b[:, :2].norm(dim=-1)
                height = state.height_above_ground
                balanced = ((support >= torch.maximum(weight[:, None] * .1,
                    weight.new_tensor(5.))).all(-1) & (support.sum(-1) >= weight * .8)
                    & (support.sum(-1) <= weight * 1.2) & (base <= 1.)
                    & (torch.maximum(roll, pitch) <= np.deg2rad(35.)) & (angular_speed <= .5))
                quiet = balanced & (speed <= .05) & (leg_speed <= .5) & ((height-target).abs() <= .03)
                balance_run = torch.where(live, torch.where(balanced, balance_run+dt, 0.), balance_run)
                quiet_run = torch.where(live, torch.where(quiet, quiet_run+dt, 0.), quiet_run)
                ever_stood |= live & state.has_stood
                samples[step] = torch.stack((data.root_link_pos_w[:, 0], data.root_link_pos_w[:, 1],
                    height, roll, pitch, speed, leg_speed, support[:, 0], support[:, 1], base,
                    angular_speed, state.has_stood.float(), state.recover.float(), velocity_error), -1)
                first_done = live & done.bool()
                termination_time[first_done] = (step+1) * dt
                # A normal timeout at the planned endpoint still has a complete window.
                failed = first_done & (state._isaac_failure | (step+1 < steps))
                balance_run[failed] = quiet_run[failed] = 0.
                live &= ~failed
                if not live.any():
                    break
                if step+1 < steps:
                    reset_ids = done.nonzero(as_tuple=False).flatten()
                    if len(reset_ids):
                        # Freeze the failed first episode; later resets are excluded by live/active.
                        diagnostic_resets.append(dict(step=step+1, env_ids=reset_ids.cpu().tolist()))
                        raw.reset(env_ids=reset_ids)
                        obs = env.get_observations()
                if (step+1) % 1000 == 0:
                    print(f"Measured {(step+1)*dt:.0f}s; {int(live.sum())}/{n} first episodes active", flush=True)
        used = step+1
        values, valid = samples[:used].cpu().numpy(), active[:used].cpu().numpy()
        late = valid.copy()
        late[:max(0, steps-round(5./dt))] = False
        all_late = values[late]
        scalar = lambda col: None if not len(all_late) else float(np.median(all_late[:, col]))
        rows = []
        for index in range(n):
            mask = late[:, index]
            row = dict(env=index, ever_isaac_stood=bool(ever_stood[index]),
                survived_to_end=bool(live[index]), final_balance_hold_s=float(balance_run[index]),
                final_quiet_hold_s=float(quiet_run[index]),
                termination_s=None if torch.isnan(termination_time[index]) else float(termination_time[index]))
            if mask.any():
                v = values[mask, index]
                row.update(last5_median_speed_m_s=float(np.median(v[:, 5])),
                    last5_median_leg_speed_rad_s=float(np.median(v[:, 6])),
                    last5_xy_extent_m=float(np.linalg.norm(np.ptp(v[:, :2], axis=0))))
            rows.append(row)
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        report = dict(scope="64/128-style first-episode diagnostic at 100 Hz; not 500 Hz/20s formal acceptance",
            recovery_contract=RECOVERY_CONTRACT,
            diagnostic_recovery_change_allowed=args.allow_recovery_change,
            source_recovery_contract=torch.load(args.checkpoint, map_location="cpu", weights_only=False)
                .get("infos", {}).get("recovery_contract", LEGACY_RECOVERY_CONTRACT),
            checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
            xml_sha256=sha(MODEL_XML), seed=args.seed, num_envs=n, duration_s=args.seconds,
            policy_dt_s=dt, play_noise_and_delays=False, auto_reset=False, parameter_updates=0,
            diagnostic_resets_after_first_result_frozen=diagnostic_resets,
            target_height_m=target.cpu().tolist(), ever_isaac_stood=int(ever_stood.sum()),
            survived_to_end=int(live.sum()), final_balance_1s=int((balance_run >= 1.-1e-5).sum()),
            final_quiet_1s=int((quiet_run >= 1.-1e-5).sum()),
            last5_median_height_m=scalar(2), last5_median_planar_speed_m_s=scalar(5),
            last5_median_max_leg_speed_rad_s=scalar(6), last5_median_velocity_rmse_m_s=scalar(13),
            finite_observed_samples=bool(np.isfinite(values[valid]).all()), episodes=rows)
        contacts = contact_summary(values, valid, late, dt)
        report["contact_diagnostics"] = {k:v for k,v in contacts.items() if k != "episodes"}
        for row, contact in zip(report["episodes"], contacts["episodes"], strict=True):
            row["contact_diagnostics"] = contact
        sources = [Path(__file__), *sorted(Path("mjlab_training/src/ct_mjlab").rglob("*.py"))]
        report["sources"] = {str(p): sha(p) for p in sources}
        np.savez_compressed(args.output/"measurements.npz", values=values, active=valid,
            fields=np.asarray(FIELDS), initial_qpos=initial_qpos.cpu().numpy(),
            initial_qvel=initial_qvel.cpu().numpy(), weight_N=weight.cpu().numpy())
        (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k not in ("episodes", "sources", "target_height_m")}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--from-report", type=Path, help="Recompute contact diagnostics from recorded measurements")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=40.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-recovery-change", action="store_true",
        help="Diagnostic only: evaluate legacy weights under the changed NORMAL rule")
    args = parser.parse_args()
    if args.from_report:
        reprocess_contacts(args)
    else:
        diagnose(args)
