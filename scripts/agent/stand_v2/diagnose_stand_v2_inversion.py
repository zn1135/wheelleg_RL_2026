#!/usr/bin/env python3
"""Trace an existing stand_v2 diagnostic's get-up actions; never update a policy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS, MODEL_XML
from scripts.agent.stand_v2.diagnose_chuanliantui_stand_v2 import FIELDS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    original = json.loads(args.report.read_text())
    saved = np.load(args.report.parent / "measurements.npz")
    assert tuple(saved["fields"]) == FIELDS
    assert sha(original["checkpoint"]) == original["checkpoint_sha256"]
    assert sha(MODEL_XML) == original["xml_sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg, agent = task_registry.get_cfgs("chuanliantui_stand_v2", play=True)
    cfg.auto_reset = False
    env, _ = task_registry.make_env("chuanliantui_stand_v2", env_cfg=cfg,
        train_cfg=agent, play=True, num_envs=original["num_envs"], seed=original["seed"])
    try:
        runner, _ = task_registry.make_alg_runner(env, "chuanliantui_stand_v2", train_cfg=agent)
        runner.load(original["checkpoint"], load_cfg={"actor": True}, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        raw = env.unwrapped
        state = raw.command_manager.get_term("standup")
        robot, action = raw.scene["robot"], raw.action_manager.get_term("pd")
        initial_qpos, initial_qvel = raw.sim.data.qpos.clone(), raw.sim.data.qvel.clone()
        assert np.array_equal(initial_qpos.cpu().numpy(), saved["initial_qpos"])
        assert np.array_equal(initial_qvel.cpu().numpy(), saved["initial_qvel"])
        if args.motors_off:
            # Simulation-only control: retain the same springs and initial state.
            action.motor_strength.zero_()
        before = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
        leg_ids, _ = robot.find_joints(("lf0", "lf00", "rf0", "rf00"), preserve_order=True)
        hip_ids, _ = robot.find_bodies(("lf0", "rf0"), preserve_order=True)
        wheel_ids, _ = robot.find_bodies(("lfwheel", "rfwheel"), preserve_order=True)
        apply_actions = action.apply_actions
        physics_qpos, physics_qvel, physics_torque = [], [], []

        def record_applied_actions():
            # Observe the real 500 Hz controller without changing its output.
            apply_actions()
            physics_qpos.append(raw.sim.data.qpos.clone())
            physics_qvel.append(raw.sim.data.qvel.clone())
            physics_torque.append(action.torques.clone())

        action.apply_actions = record_applied_actions
        observations = env.get_observations()
        measurements, actions, rates, velocities, gravities, leg_geometry = [], [], [], [], [], []
        reward_names = list(raw.reward_manager.active_terms)
        dt = raw.step_dt
        steps = round(args.seconds / dt)
        with torch.inference_mode():
            for step in range(steps):
                error = (policy.encoder(observations["history"]) / 2.
                    - robot.data.root_com_lin_vel_b).square().mean(-1).sqrt()
                chosen = policy(observations)
                if args.zero_actions:
                    chosen = torch.zeros_like(chosen)
                actions.append(chosen.clone())
                observations, _, done, _ = env.step(chosen)
                assert not done.any(), "Get-up trace ended before requested duration"
                data = robot.data
                gravity = data.projected_gravity_b
                roll = torch.atan2(-gravity[:, 1], -gravity[:, 2]).abs()
                pitch = gravity[:, 0].clamp(-1., 1.).asin().abs()
                support = -raw.scene["wheel_contact"].data.force[:, :, 2]
                base = raw.scene["base_contact"].data.force.norm(dim=-1).amax(-1)
                speed = data.root_link_lin_vel_b[:, :2].norm(dim=-1)
                leg_speed = data.joint_vel[:, leg_ids].abs().amax(-1)
                angular = data.root_link_ang_vel_b[:, :2].norm(dim=-1)
                measurements.append(torch.stack((data.root_link_pos_w[:, 0], data.root_link_pos_w[:, 1],
                    state.height_above_ground, roll, pitch, speed, leg_speed, support[:, 0],
                    support[:, 1], base, angular, state.has_stood.float(), state.recover.float(), error), -1))
                rates.append(raw.reward_manager._step_reward.clone())
                velocities.append(torch.cat((data.root_link_lin_vel_b, data.root_link_ang_vel_b), -1))
                gravities.append(gravity.clone())
                leg_geometry.append((data.body_link_pos_w[:, wheel_ids]
                    - data.body_link_pos_w[:, hip_ids]).clone())
        assert len(physics_qpos) == steps * round(dt / raw.physics_dt)
        assert all(torch.equal(before[k], v) for k, v in runner.alg.actor.state_dict().items())
        arrays = dict(values=torch.stack(measurements).cpu().numpy(), fields=np.asarray(FIELDS),
            actions=torch.stack(actions).cpu().numpy(), reward_rates=torch.stack(rates).cpu().numpy(),
            reward_names=np.asarray(reward_names), body_velocities=torch.stack(velocities).cpu().numpy(),
            projected_gravity=torch.stack(gravities).cpu().numpy(),
            hip_to_wheel_world=torch.stack(leg_geometry).cpu().numpy(),
            physics_qpos=torch.stack(physics_qpos).cpu().numpy(),
            physics_qvel=torch.stack(physics_qvel).cpu().numpy(),
            physics_torque=torch.stack(physics_torque).cpu().numpy(),
            initial_qpos=initial_qpos.cpu().numpy(), initial_qvel=initial_qvel.cpu().numpy(),
            motor_q_adr=action._motor_q_adr.cpu().numpy(), motor_v_adr=action._motor_v_adr.cpu().numpy(),
            motor_names=np.asarray(ACTUATED_JOINTS), weight_N=saved["weight_N"])
        np.savez_compressed(args.output / "trace.npz", **arrays)
        # Existing 40 s records retain recovery outcomes; this rerun adds the launch mechanics.
        old = saved["values"]
        tilt = np.rad2deg(np.arccos(np.clip(np.cos(old[:, :, 3]) * np.cos(old[:, :, 4]), -1., 1.)))
        valid = saved["active"].copy()
        valid[:round(.2 / dt)] = False
        ids = np.where(((tilt > 90.) & valid).any(0))[0]
        rows = []
        for i in ids:
            inv = np.where((tilt[:, i] > 90.) & valid[:, i])[0]
            stood = np.where(old[:, i, 11] > .5)[0]
            early = arrays["values"][:min(100, steps), i]
            torque = arrays["physics_torque"][:, i]
            # Substep qpos/qvel are sampled before integration; policy records are after it.
            trace_tilt = np.rad2deg(np.arccos(np.clip(-arrays["projected_gravity"][:, i, 2], -1., 1.)))
            trace_inv = np.where(trace_tilt > 90.)[0]
            rows.append(dict(env=int(i), initial_quaternion=saved["initial_qpos"][i, 3:7].tolist(),
                original_episode=dict(first_inverted_s=float((inv[0]+1)*dt),
                    last_inverted_s=float((inv[-1]+1)*dt), inverted_duration_s=float(len(inv)*dt),
                    maximum_tilt_deg=float(tilt[:, i].max()), final_tilt_deg=float(tilt[-1, i]),
                    first_stood_s=None if not len(stood) else float((stood[0]+1)*dt),
                    final_balance_hold_s=original["episodes"][i]["final_balance_hold_s"]),
                trace_first_inverted_s=None if not len(trace_inv) else float((trace_inv[0]+1)*dt),
                trace_maximum_tilt_deg=float(trace_tilt.max()),
                maximum_first1s_height_m=float(early[:, 2].max()),
                maximum_first1s_leg_speed_rad_s=float(early[:, 6].max()),
                maximum_first1s_roll_pitch_speed_rad_s=float(early[:, 10].max()),
                maximum_abs_motor_torque_Nm=np.abs(torque).max(0).tolist(),
                reward_rates_at_first_active_step=dict(zip(reward_names, arrays["reward_rates"][round(.2/dt), i].tolist()))))
        difference = np.abs(arrays["values"] - old[:steps])
        report = dict(scope=__doc__, source_report=str(args.report), source_report_sha256=sha(args.report),
            checkpoint_sha256=original["checkpoint_sha256"], xml_sha256=sha(MODEL_XML),
            seed=original["seed"], num_envs=env.num_envs, trace_duration_s=args.seconds,
            original_duration_s=original["duration_s"], physics_dt_s=raw.physics_dt, policy_dt_s=dt,
            physics_trace_phase="before integration, after unmodified apply_actions; motor torque is bounded PD demand",
            policy_trace_phase="after env.step, weighted reward rates exclude dt",
            exact_initial_state_match=True, original_first_seconds_max_field_error=dict(zip(FIELDS, difference.max((0, 1)).tolist())),
            actor_and_encoder_unchanged=True, parameter_updates=0,
            nominal_play=True, action_mode="zero" if args.zero_actions else "policy",
            motors_disabled=args.motors_off,
            inversion_envs=rows, source_sha256=sha(__file__))
        (args.output / "result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k != "inversion_envs"}, indent=2))
        for row in rows:
            print(json.dumps(row), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=5.)
    parser.add_argument("--zero-actions", action="store_true",
        help="Diagnostic counterfactual: keep PD and springs, replace policy outputs with zero")
    parser.add_argument("--motors-off", action="store_true",
        help="Diagnostic counterfactual: disable six motor torques, retain physical springs")
    main(parser.parse_args())
