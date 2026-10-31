#!/usr/bin/env python3
"""Frozen stand_v2 velocity-command probe after ground get-up; no training."""

from scripts.agent.motion.assess import STANDARD, THRESHOLDS, tracking_rows

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.managers.metrics_manager import MetricsTermCfg

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_stand_v2.mdp import TRACKING_COMMAND_CONTRACT
from scripts.agent.stand_v2.assess_chuanliantui_stand_v2 import Recorder, FIELDS, sample_physics, digest


SPEEDS = (0., .1, -.1, .3, -.3, .5, -.5)
WARMUP_S, DRIVE_END_S, END_S = 8., 18., 22.
STEADY_START_S = 10.


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, agent = task_registry.get_cfgs("chuanliantui_stand_v2", play=True)
    cfg.auto_reset = False
    cfg.metrics["tracking_physics"] = MetricsTermCfg(func=sample_physics, per_substep=True, reduce="last")
    env, _ = task_registry.make_env("chuanliantui_stand_v2", env_cfg=cfg, train_cfg=agent,
        play=True, num_envs=64, seed=args.seed, device=args.device)
    try:
        runner, _ = task_registry.make_alg_runner(env, "chuanliantui_stand_v2", train_cfg=agent, device=args.device)
        tracking_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get(
            "infos", {}).get("command_contract") == TRACKING_COMMAND_CONTRACT
        state = env.unwrapped.command_manager.get_term("standup")
        state.cfg.post_stand_tracking = tracking_checkpoint
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device)
        # 此工具按固定8秒预热/速度分组探测；覆盖训练命令调度，不更改策略参数。
        state.cfg.post_stand_tracking = False
        policy = runner.get_inference_policy(device=args.device)
        frozen = {k:v.clone() for k,v in runner.alg.actor.state_dict().items()}
        raw, dt = env.unwrapped, env.unwrapped.step_dt
        state, robot = raw.command_manager.get_term("standup"), raw.scene["robot"]
        assert raw.physics_dt == .002 and dt == .01 and raw.max_episode_length*dt == 40.
        r = Recorder(raw)
        r.trace = torch.empty((round(END_S/raw.physics_dt), 64, len(FIELDS)), device=args.device)
        raw._stand_v2_acceptance = r
        initial_qpos, initial_qvel = raw.sim.data.qpos.clone(), raw.sim.data.qvel.clone()
        weight = raw.sim.model.body_mass.sum(-1)*np.linalg.norm(raw.sim.mj_model.opt.gravity)
        desired = torch.tensor([SPEEDS[i % len(SPEEDS)] for i in range(64)], device=args.device)
        ready = torch.zeros(64, dtype=torch.bool, device=args.device)
        launched = False
        launch_info = {}
        schedule_tick = 0
        update_command = state._update_command

        def pin_command(env_ids):
            nonlocal launched
            update_command(env_ids)
            t = schedule_tick*dt
            if not launched and t >= WARMUP_S:
                # Ground get-up stays unchanged; only standing, still-live cases receive a command.
                ready.copy_(r.live & state.has_stood & ~state.recover)
                launched = True
                launch_info.update(schedule_tick=schedule_tick, common_step_counter=int(raw.common_step_counter),
                    stood=int(state.has_stood.sum()), recover=int(state.recover.sum()), ready=int(ready.sum()))
                print(f'Command launch: {launch_info}', flush=True)
                assert ready.any(), "No standing sample received a command"
            ids = slice(None) if env_ids is None else env_ids
            active = ready & (t >= WARMUP_S) & (t < DRIVE_END_S)
            state._command[ids, 0] = torch.where(active, desired, 0.)[ids]
            state._command[ids, 1] = 0.

        state._update_command = pin_command
        obs = env.get_observations()
        rows = []
        qpos = [initial_qpos.clone()]
        active_steps = []
        termination_s = torch.full((64,), float("nan"), device=args.device)
        for tick in range(round(END_S/dt)):
            schedule_tick = tick+1
            current = state.command.clone()
            expected_vx = torch.where(ready & (tick*dt >= WARMUP_S)
                                     & (tick*dt < DRIVE_END_S), desired, 0.)
            assert torch.equal(current[:, 0], expected_vx)
            expected = current*current.new_tensor((2., .25, 5.))
            if tick >= round(WARMUP_S/dt):
                assert launched and ready.any(), "Command update hook did not execute"
            # Confirm the actor and newest history frame both see the actual requested command.
            assert torch.allclose(obs["actor"][:, 6:9], expected, atol=1e-6)
            assert torch.equal(obs["history"][:, -32:], obs["actor"])
            before_live = r.live.clone()
            obs, _, done, _ = env.step(policy(obs))
            d = robot.data
            rows.append(torch.stack((current[:, 0], d.root_link_lin_vel_b[:, 0],
                d.root_link_lin_vel_b[:, 1], d.root_link_ang_vel_b[:, 2],
                state.height_above_ground, state.has_stood.float(), state.recover.float(),
                (policy.encoder(obs["history"])/2.)[:, 0]), -1))
            active_steps.append(before_live)
            qpos.append(raw.sim.data.qpos.clone())
            first_done = before_live & done.bool()
            termination_s[first_done] = (tick+1)*dt
            r.live &= ~done.bool()
            if done.any() and tick+1 < round(END_S/dt):
                ids = done.nonzero(as_tuple=False).flatten()
                raw.reset(env_ids=ids)
                obs = env.get_observations()
            if (tick+1) % 500 == 0:
                print(f't={(tick+1)*dt:.1f}s live={int(r.live.sum())}/64 ready={int(ready.sum())}/64', flush=True)
        assert r.count == 11000
        assert all(torch.equal(frozen[k], v) for k,v in runner.alg.actor.state_dict().items())
        values = torch.stack(rows).cpu().numpy()
        live = torch.stack(active_steps).cpu().numpy()
        physical = r.trace.cpu().numpy()
        weights = weight.cpu().numpy()
        balance = ((physical[:, :, :2] >= np.maximum(weights*.1, 5.)[None, :, None]).all(-1)
            & (physical[:, :, :2].sum(-1) >= weights*.8)
            & (physical[:, :, :2].sum(-1) <= weights*1.2)
            & (physical[:, :, 2] <= 1.) & (physical[:, :, 3:5].max(-1) <= np.deg2rad(35.))
            & (physical[:, :, 5] <= .5) & (physical[:, :, 12] > .5) & (physical[:, :, 13] > .5))
        # Statistics exclude the first two seconds of speed-command response.
        steady = slice(round(STEADY_START_S/dt), round(DRIVE_END_S/dt))
        steady_physics = slice(round(STEADY_START_S/.002), round(DRIVE_END_S/.002))
        result_rows = []
        for i in range(64):
            mask = live[steady, i]
            v = values[steady, i][mask]
            eligible = bool(ready[i])
            measured = bool(eligible and len(v))
            full = bool(eligible and mask.all() and (torch.isnan(termination_s[i])
                        or termination_s[i] > DRIVE_END_S))
            row = dict(env=i, requested_vx_m_s=float(desired[i]), ready_at_8s=eligible,
                complete_tracking_window=full, survived_22s=bool(r.live[i]),
                termination_s=None if torch.isnan(termination_s[i]) else float(termination_s[i]),
                measured_s=float(len(v)*dt) if eligible else 0.,
                actual_mean_vx_m_s=float(v[:, 1].mean()) if measured else None,
                vx_mean_abs_error_m_s=float(np.abs(v[:, 1]-v[:, 0]).mean()) if measured else None,
                vx_rmse_m_s=float(np.sqrt(np.square(v[:, 1]-v[:, 0]).mean())) if measured else None,
                vy_mean_abs_m_s=float(np.abs(v[:, 2]).mean()) if measured else None,
                yaw_mean_abs_rad_s=float(np.abs(v[:, 3]).mean()) if measured else None,
                height_mean_m=float(v[:, 4].mean()) if measured else None,
                steady_balance_fraction=float(balance[steady_physics, i].mean()) if eligible else None,
                steady_base_contact_s=float((physical[steady_physics, i, 2] > 1.).sum()*.002) if eligible else None,
                final_drive_balance_1s=bool(full and balance[8500:9000, i].all()),
                stopped_last1_mean_speed_m_s=(float(np.linalg.norm(values[-100:, i, 1:3],axis=-1).mean())
                    if bool(r.live[i]) and eligible else None))
            result_rows.append(row)
        groups = []
        for speed in SPEEDS:
            group = [row for i,row in enumerate(result_rows) if i % len(SPEEDS) == SPEEDS.index(speed)]
            complete = [row for row in group if row["complete_tracking_window"]]
            groups.append(dict(command_vx_m_s=speed, total=len(group), ready=sum(row["ready_at_8s"] for row in group),
                completed=len(complete), final_drive_balanced=sum(row["final_drive_balance_1s"] for row in group),
                actual_mean_vx_m_s=float(np.mean([row["actual_mean_vx_m_s"] for row in complete])) if complete else None,
                mean_abs_error_m_s=float(np.mean([row["vx_mean_abs_error_m_s"] for row in complete])) if complete else None,
                mean_balance_fraction=float(np.mean([row["steady_balance_fraction"] for row in complete])) if complete else None))
        np.savez_compressed(args.output/"trace.npz", values=values, active=live, physical=physical,
            physical_fields=np.asarray(FIELDS), physical_time_s=(np.arange(11000)+1)*.002,
            fields=np.asarray(("command_vx", "actual_vx", "actual_vy", "yaw_rate", "height", "has_stood", "recover", "estimated_vx")),
            initial_qpos=initial_qpos.cpu().numpy(), initial_qvel=initial_qvel.cpu().numpy(),
            weight_N=weights, ready=ready.cpu().numpy(), time_s=(np.arange(2200)+1)*dt)
        np.savez_compressed(args.output/"qpos.npz", qpos=torch.stack(qpos).cpu().numpy(), time_s=np.arange(2201)*dt)
        mujoco.mj_saveModel(raw.sim.mj_model, str(args.output/"scene.mjb"), None)
        sources = [Path("scripts/agent/motion/assess.py"), Path(__file__), Path("scripts/agent/stand_v2/assess_chuanliantui_stand_v2.py"),
                   *sorted(Path("mjlab_training/src/ct_mjlab").rglob("*.py"))]
        report = dict(task="chuanliantui_stand_v2", checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=digest(args.checkpoint), num_envs=64, seed=args.seed, physics_dt_s=.002,
            policy_dt_s=.01, parameter_updates=0, actor_and_encoder_unchanged=True,
            warmup_s=WARMUP_S, command_window_s=[WARMUP_S, DRIVE_END_S],
            measured_window_s=[STEADY_START_S, DRIVE_END_S], stop_window_s=[DRIVE_END_S, END_S],
            height_m=float(state.target_height), yaw_command=0., groups=groups, rows=result_rows,
            checkpoint_has_tracking_recipe=tracking_checkpoint,
            command_launch=launch_info,
            sources={str(p):digest(p) for p in sources}, trace_sha256=digest(args.output/"trace.npz"),
            qpos_sha256=digest(args.output/"qpos.npz"),
            scope=("Nominal play with frozen policy; fixed8s warmup and grouped speed probe. "
                  + ("Training covers magnitudes0.1to0.3m/s;0.5m/s is outside that range. " if tracking_checkpoint
                     else "Zero-command-trained policy; all nonzero commands are outside training distribution. ")
                  + "Different initial samples per speed, one seed; only ready cases receive commands, no hardware claim."))
        motion_rows, motion_windows = tracking_rows(report, values, physical, live, (np.arange(2200)+1)*dt)
        report['motion_standard'] = dict(standard=STANDARD, thresholds=THRESHOLDS.copy(),
            rows=motion_rows, windows=motion_windows, passed=sum(row['passed'] for row in motion_rows), total=64)
        (args.output/"result.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(groups,indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
