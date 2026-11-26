"""Verify manual walk/step handoff and bounded native playback; no training."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from ct_mjlab.fudan_ppo_walk.policy import load_policy
from sim2sim.rmuc2027_arena import ArenaPlayback, REGIONAL_CHECKPOINT as WALK_CHECKPOINT
from sim2sim.step_playground import CHECKPOINT, ManualStepPolicies, build_model, layout


def snapshot(env):
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, kind))
    mujoco.mj_getState(env.model, env.data, state, kind)
    values = dict(integration=state, history=env.history.copy(), action=env.last_action.copy(),
        sampled_velocity=env.velocity.copy(), torque=env.torque.copy(), last_sample_q=env.last_sample_q.copy(),
        xy_command=env.command[:2].copy(), pd_tick=env.pd_tick, steps=env.steps,
        data_id=id(env.data), model_id=id(env.model),
        action_head=env.delays.action_head, sensor_head=env.delays.sensor_head)
    for key, value in vars(env.delays).items():
        if isinstance(value, torch.Tensor):
            values['delay_' + key] = value.cpu().numpy().copy()
    return values


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--walk-checkpoint', type=Path, default=WALK_CHECKPOINT)
    parser.add_argument('--step-checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--parking-checkpoint', type=Path)
    parser.add_argument('--expect-step-speed', type=float, default=1.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    step_policy, saved = load_policy(args.step_checkpoint)
    switch = ManualStepPolicies(args.walk_checkpoint, args.step_checkpoint, step_policy, saved,
                                parking_checkpoint=args.parking_checkpoint)
    assert switch.mode == 'walk' and not switch.metadata['walk']['partial_update']
    assert switch.checkpoints['walk']['checkpoint'] == str(args.walk_checkpoint.resolve())
    assert switch.checkpoints['step']['checkpoint'] == str(args.step_checkpoint.resolve())
    assert switch.keyboard_commands == (3., 2.)
    model = build_model(saved)
    env = ArenaPlayback(model, saved)
    spot = layout(saved)[0]
    env.command[:] = (0., 0., .28)
    env.reset((*spot['spawn'], 0.), yaw=0.)
    digests = {role: {key: value.clone() for key, value in policy.state_dict().items()}
               for role, policy in switch.policies.items()}
    checks, rows, modes, maximum_torque = [], [], [], np.zeros(6)

    def select(mode):
        before = snapshot(env)
        changed = switch.select(mode, env)
        after = snapshot(env)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        if changed:
            expected = .35 if mode == 'step' else .28
            assert abs(float(env.command[2]) - expected) < 1e-7
        keyboard = (args.expect_step_speed, 0. if args.parking_checkpoint else .5) if mode == 'step' else (3., 2.)
        assert switch.keyboard_commands == keyboard
        checks.append(dict(mode=mode, changed=changed, physical_fifo_history_exact=True, keyboard=keyboard))

    with torch.inference_mode():
        for tick in range(600):
            if tick == 100:
                select('step')
                select('step')  # 重复进入同模式不能重置任何状态。
            if tick == 300:
                select('walk')
            if tick == 400:
                for i in range(20):
                    select('step' if i % 2 == 0 else 'walk')
            if args.parking_checkpoint:
                moving = 110 <= tick < 170 or 230 <= tick < 280
            else:
                moving = 100 <= tick < 300
            env.command[0] = .5 if moving else 0.
            switch.apply_step_command(env.command)
            obs = env.observation()
            env.history[-19:-16] = obs[6:9]
            inputs = torch.from_numpy(obs[None]), torch.from_numpy(env.history[None])
            before = snapshot(env) if args.parking_checkpoint and tick in (100, 110, 170, 230, 280, 300) else None
            actual, _ = switch.act_inference(*inputs)
            if before is not None:
                after = snapshot(env)
                for key in before:
                    np.testing.assert_array_equal(before[key], after[key], err_msg=key)
            parking = args.parking_checkpoint is not None and switch.mode == 'step' and 170 <= tick < 300 and not moving
            role = 'parking' if parking else switch.mode
            assert switch.active_role == role
            if args.parking_checkpoint is not None and switch.mode == 'step':
                np.testing.assert_array_equal(env.command[:2], [args.expect_step_speed if moving else 0., 0.])
            expected, _ = switch.policies[role].act_inference(*inputs)
            assert torch.equal(actual, expected)
            env.step(actual[0].numpy())
            maximum_torque = np.maximum(maximum_torque, abs(env.torque))
            rows.append([env.data.time, *env.command, *env.data.qpos[:3], *env.data.qvel[:6]])
            modes.append(role)
    # 原控制器以float32保存力矩，3.9的实际存储值为3.900000095。
    caps = np.array([40., 40., 3.9, 40., 40., 3.9], dtype=np.float32).astype(np.float64)
    assert np.all(maximum_torque <= caps), maximum_torque
    for role, policy in switch.policies.items():
        assert all(torch.equal(value, digests[role][key]) for key, value in policy.state_dict().items())
    if args.parking_checkpoint:
        select('step')
        env.command[:2] = (2., 0.)
        switch.act_inference(torch.from_numpy(env.observation()[None]), torch.from_numpy(env.history[None]))
        env.command[:2] = 0.
        switch.reset_parking()
        switch.act_inference(torch.from_numpy(env.observation()[None]), torch.from_numpy(env.history[None]))
        assert switch.active_role == 'step'
        select('walk')
    bad = dict(saved, control=dict(saved['control'], leg_stiffness=21.))
    try:
        ManualStepPolicies(args.walk_checkpoint, args.step_checkpoint, step_policy, bad)
    except ValueError as error:
        assert 'control' in str(error)
    else:
        raise AssertionError('Incompatible physical control was accepted')
    np.savez_compressed(args.output / 'trace.npz', state=np.asarray(rows), mode=np.asarray(modes))
    result = dict(passed=True, mode_switch=switch.record(), checks=checks,
        physical_state_fifo_pd_history_preserved=True, active_actor_output_exact=True,
        saved_walk_height_restored=True, incompatible_control_rejected=True,
        all_networks_unchanged=True, sampled_motor_torque_max_Nm=maximum_torque.tolist(),
        native_physics_steps=6010, policy_steps=600, optimizer_updates=0, formal_acceptance=False,
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path('sim2sim/step_playground.py'), Path('sim2sim/rmuc2027_arena.py')]},
        scope='Manual switch continuity and finite native playback; not stair traversal or hardware acceptance')
    (args.output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(passed=True, transitions=len(switch.transitions), policy_steps=600, optimizer_updates=0)))


if __name__ == '__main__':
    main()
