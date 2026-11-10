"""Verify persistent six-axis episode loads, pulse composition and actual GPU steps."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ct_mjlab.envs.fudan_ppo_walk.disturbances import DisturbanceConfig, DisturbanceState, from_record, from_task_config
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.fudan_ppo_walk.measurement import configure_samples
from ct_mjlab.fudan_ppo_walk.protocol import samples
from .check_disturbances import control_fixture


def rotate(quat, vectors):
    # 独立旋转矩阵复算，不调用运行时的quat_apply。
    w, x, y, z = quat.T
    matrix = np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                       2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                       2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)
    return np.einsum('nij,nj->ni', matrix, vectors)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    checks, evidence = {}, {}

    def verify(name, value):
        checks[name] = bool(value)
        assert checks[name], name

    default = DisturbanceConfig()
    legacy = default.record()
    legacy.pop('external_force_range')
    legacy.pop('external_torque_range')
    verify('old_record_without_wrench_stays_disabled', from_record(legacy) == default)
    rng = torch.get_rng_state().clone()
    disabled = DisturbanceState(default, 3, 'cpu')
    disabled.reset(slice(None), torch.zeros(3, 16))
    verify('disabled_wrench_does_not_consume_rng', torch.equal(rng, torch.get_rng_state()))
    cfg = DisturbanceConfig(external_force_range=(-10., 10.), external_torque_range=(-10., 10.))
    verify('new_record_roundtrip', from_record(cfg.record()) == cfg)
    for device in ('cpu', 'cuda:0'):
        state = DisturbanceState(cfg, 64, device)
        sensors = torch.zeros(64, 16, device=device)
        state.reset(slice(None), sensors)
        value = state.episode_wrench.clone()
        verify(device+'_six_axes_within10', value.abs().le(10).all())
        verify(device+'_each_axis_samples_both_signs', (value.min(0).values<0).all() and (value.max(0).values>0).all())
        identity = torch.zeros(64, 4, device=device)
        identity[:, 0] = 1.
        verify(device+'_constant_through_episode', torch.equal(state.wrench(identity), value)
               and torch.equal(state.wrench(identity), value) and torch.equal(state.episode_wrench, value))
        quarter = identity.clone()
        quarter[:, 0] = quarter[:, 3] = 2**-.5
        expected = torch.stack((-value[:, 1], value[:, 0], value[:, 2], -value[:, 4], value[:, 3], value[:, 5]), -1)
        verify(device+'_body_to_world_force_and_moment', torch.allclose(state.wrench(quarter), expected, atol=3e-6))
        ids = torch.tensor([1, 7, 31], device=device)
        other = torch.tensor([i for i in range(64) if i not in (1, 7, 31)], device=device)
        state.reset(ids, sensors)
        verify(device+'_partial_reset_resamples_only_selected', not torch.equal(state.episode_wrench[ids], value[ids])
               and torch.equal(state.episode_wrench[other], value[other]))

    pulse_cfg = DisturbanceConfig(pushes=True, push_interval_s=.006, max_push_vel=.1,
        external_force_range=(-10., 10.), external_torque_range=(-10., 10.))
    fixture = control_fixture(pulse_cfg)
    action, state = fixture.action, fixture.action.disturbances
    state.episode_wrench[:] = torch.tensor([1., 2., 3., 4., 5., 6.])
    fixture.env.sim.data.qpos[:, 3:7] = torch.tensor([1., 0., 0., 0.])
    applied = []
    for _ in range(4):
        action.apply_actions()
        applied.append(fixture.env.sim.data.xfrc_applied[:, action._base_id].clone())
    verify('persistent_load_present_without_pulse', torch.equal(applied[0], state.episode_wrench)
           and torch.equal(applied[1], state.episode_wrench) and torch.equal(applied[3], state.episode_wrench))
    verify('pulse_adds_force_without_erasing_torque', not torch.equal(applied[2][:, :3], state.episode_wrench[:, :3])
           and torch.equal(applied[2][:, 3:], state.episode_wrench[:, 3:]))

    items = [dict(sample, id=f'wrench_{i}', seed=31001+i) for i, sample in enumerate(samples()[:3])]
    configure_samples(fixture, items, args.output/'fixture-a')
    first = state.episode_wrench.clone()
    configure_samples(fixture, items[::-1], args.output/'fixture-b')
    verify('evaluation_wrench_independent_of_batch_order', torch.equal(first.flip(0), state.episode_wrench))
    with np.load(args.output/'fixture-a/wrench_0_initial.npz') as saved:
        verify('evaluation_fixture_saves_body_wrench', np.array_equal(saved['episode_wrench_body'], first[0].numpy()))

    wrapper = FudanEnv(20, seed=30001, randomized=True)
    env, action = wrapper.env, wrapper.action
    state = action.disturbances
    traces, errors = [], []
    original_step, original_force = env.sim.step, state.force
    last_push = np.zeros((20, 3))

    def observed_force(*a, **kw):
        nonlocal last_push
        result = original_force(*a, **kw)
        last_push = result.cpu().numpy().copy()
        return result

    def observed_step(*a, **kw):
        quat = env.sim.data.qpos[:, 3:7].cpu().numpy().astype(float)
        body = state.episode_wrench.cpu().numpy().astype(float)
        expected = np.concatenate((rotate(quat, body[:, :3])+last_push, rotate(quat, body[:, 3:])), -1)
        actual = env.sim.data.xfrc_applied[:, action._base_id].cpu().numpy().copy()
        errors.append(float(np.max(abs(actual-expected))))
        traces.append(actual)
        return original_step(*a, **kw)

    try:
        verify('actual_training_wrench_ranges', from_task_config().external_force_range == (-10., 10.)
               and from_task_config().external_torque_range == (-10., 10.))
        env.sim.step, state.force = observed_step, observed_force
        wrapper.reset()
        state.ticks.fill_(state.push_interval-1)
        for _ in range(10):
            wrapper.step(torch.zeros(20, 6, device=wrapper.device), auto_reset=False, resample=False)
        verify('actual_gpu_each_step_force_and_torque', max(errors)<.0001 and len(errors)>=55)
        verify('actual_gpu_pulse_occurred', state.push_count.eq(1).all())
        before = state.episode_wrench.clone()
        ids = torch.tensor([0, 3, 7], device=wrapper.device)
        other = torch.tensor([i for i in range(20) if i not in (0, 3, 7)], device=wrapper.device)
        old_applied = env.sim.data.xfrc_applied[other, action._base_id].clone()
        wrapper.reset_idx(ids)
        verify('actual_gpu_partial_reset_selective', not torch.equal(before[ids], state.episode_wrench[ids])
               and torch.equal(before[other], state.episode_wrench[other]))
        verify('actual_gpu_reset_clears_stale_load_only_selected', env.sim.data.xfrc_applied[ids, action._base_id].eq(0).all()
               and torch.equal(old_applied, env.sim.data.xfrc_applied[other, action._base_id]))
        wrapper.step(torch.zeros(20, 6, device=wrapper.device), auto_reset=False, resample=False)
        verify('actual_gpu_new_episode_load_applied', max(errors)<.0001 and state.episode_wrench.abs().le(10).all())
        verify('actual_gpu_state_finite', torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all())
        np.savez_compressed(args.output/'gpu-applied.npz', xfrc=np.asarray(traces),
            final_body_wrench=state.episode_wrench.cpu().numpy())
        evidence.update(gpu_step_calls=len(traces), maximum_applied_error=float(max(errors)),
            physics_backend='mujoco_warp_gpu', num_envs=20)
    finally:
        env.sim.step, state.force = original_step, original_force
        env.close()
    result = dict(passed=all(checks.values()), checks=checks, evidence=evidence,
        disturbances=from_task_config().record(), model_forwards=0, optimizer_updates=0,
        scope='Actual GPU applied load and reset checks; zero-action numerical check, not balance or behavior acceptance')
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
