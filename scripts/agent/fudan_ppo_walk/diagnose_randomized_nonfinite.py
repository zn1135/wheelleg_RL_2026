"""Bounded fixed-policy GPU replay that saves the first non-finite substep; no updates."""
import argparse
from collections import deque
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk import fudan_ppo_walk as task
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, section_dict
from ct_mjlab.fudan_ppo_walk.policy import load_policy
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import initialize_rollout
from ct_mjlab.envs.chuanliantui.gas_spring import gas_spring_force


def install_diagnostic_substeps(wrapper, subdivisions, refresh_spring=False):
    sim, action = wrapper.env.sim, wrapper.action
    control_dt = wrapper.env.physics_dt
    for field in ('eq_solref', 'jnt_solref', 'dof_solref'):
        if hasattr(sim.model, field):
            value = getattr(sim.model, field)
            if value.ndim == 3 and value.stride(0) == 0:
                value = value[0]
            first = value[..., 0]
            first.copy_(torch.where(first > 0, first.clamp(min=2*control_dt), first))
        if hasattr(sim.mj_model, field):
            first = getattr(sim.mj_model, field)[..., 0]
            first[first > 0] = np.maximum(first[first > 0], 2*control_dt)
    sim.mj_model.opt.timestep = control_dt/subdivisions
    sim.wp_model.opt.timestep.fill_(control_dt/subdivisions)
    original_step = sim.step
    def integrate():
        state = action.disturbances
        if state.has_episode_wrench:
            pulse = sim.data.xfrc_applied[:, action._base_id, :3] - state.wrench(sim.data.qpos[:, 3:7])[:, :3]
        for i in range(subdivisions):
            if i and refresh_spring:
                sim.data.ctrl[:, action._spring_ids] = gas_spring_force(
                    sim.data.qpos[:, action._spring_q_adr], sim.data.qvel[:, action._spring_v_adr], action._spring_force_bias)
            if i and state.has_episode_wrench:
                world = state.wrench(sim.data.qpos[:, 3:7])
                world[:, :3] += pulse
                sim.data.xfrc_applied[:, action._base_id] = world
            original_step()
    sim.step = integrate


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy-steps', type=int, default=600)
    parser.add_argument('--solver', choices=('newton', 'cg'), default='newton')
    parser.add_argument('--integration-substeps', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--refresh-spring', action='store_true')
    parser.add_argument('--force-scale', type=float, choices=(0., 1.), default=1.)
    parser.add_argument('--torque-scale', type=float, choices=(0., 1.), default=1.)
    parser.add_argument('--fall-confirmation-ticks', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    original_cfg = task.make_env_cfg
    def diagnostic_cfg(*a, **kw):
        cfg = original_cfg(*a, **kw)
        cfg.sim.mujoco.solver = args.solver
        return cfg
    diagnostic_task = FudanWalkCfg()
    if args.fall_confirmation_ticks is not None:
        diagnostic_task.env = SimpleNamespace(**section_dict(diagnostic_task.env))
        diagnostic_task.env.fail_ticks = args.fall_confirmation_ticks
    with patch.object(task, 'make_env_cfg', diagnostic_cfg):
        wrapper = FudanEnv(4096, seed=1, randomized=True,
            reset_contract='fudan-physical-pd-centred-warmup-clean4-noisy1-unclipped-history-v2', cfg=diagnostic_task)
    env = wrapper.env
    original_wrench = wrapper.action.disturbances.wrench
    def ablated_wrench(quat):
        result = original_wrench(quat)
        result[:, :3] *= args.force_scale
        result[:, 3:] *= args.torque_scale
        return result
    wrapper.action.disturbances.wrench = ablated_wrench
    if args.integration_substeps > 1:
        install_diagnostic_substeps(wrapper, args.integration_substeps, args.refresh_spring)
    policy, _ = load_policy(args.checkpoint, 'cuda:0')
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    wrapper.speed_limit = source['speed_limit']
    wrapper.total_steps = source['policy_steps']
    env.common_step_counter = wrapper.total_steps
    torch.set_rng_state(source['torch_rng'])
    torch.cuda.set_rng_state_all(source['cuda_rng'])
    calls, failure = 0, None
    state_history = deque(maxlen=64)
    original_step = env.sim.step

    def observed_step(*a, **kw):
        nonlocal calls, failure
        before = {name: getattr(env.sim.data, name).clone() for name in ('qpos', 'qvel', 'ctrl', 'xfrc_applied')}
        state_history.append(before)
        result = original_step(*a, **kw)
        calls += 1
        bad = ~(torch.isfinite(env.sim.data.qpos).all(-1) & torch.isfinite(env.sim.data.qvel).all(-1))
        if bad.any():
            ids = bad.nonzero().flatten()
            saved = {name+'_before': value[ids].cpu().numpy() for name, value in before.items()}
            saved.update({name+'_history': np.stack([row[name][ids].cpu().numpy() for row in state_history]) for name in before})
            saved.update(qpos_after=env.sim.data.qpos[ids].cpu().numpy(),
                qvel_after=env.sim.data.qvel[ids].cpu().numpy(), env_ids=ids.cpu().numpy(),
                origins=env.scene.env_origins[ids].cpu().numpy(),
                episode_wrench_body=wrapper.action.disturbances.episode_wrench[ids].cpu().numpy(),
                kp=wrapper.action.kp_scale[ids].cpu().numpy(), kd=wrapper.action.kd_scale[ids].cpu().numpy(),
                motor_strength=wrapper.action.motor_strength[ids].cpu().numpy())
            for field in env.sim.expanded_fields:
                value = getattr(env.sim.model, field)
                saved['model__'+field] = value[ids].cpu().numpy()
            np.savez_compressed(args.output/'first-bad-substep.npz', **saved)
            mujoco.mj_saveModel(env.sim.mj_model, str(args.output/'model.mjb'), None)
            failure = dict(env_ids=ids.cpu().tolist(), physics_call=calls,
                before_finite=all(bool(torch.isfinite(value[ids]).all()) for value in before.values()))
            raise FloatingPointError('Captured first non-finite physics substep')
        return result

    completed, terminations = 0, 0
    env.sim.step = observed_step
    try:
        obs, _, history = initialize_rollout(wrapper)
        with torch.inference_mode():
            for completed in range(args.policy_steps):
                actions = policy.act(obs, history)
                obs, _, _, done, _, history = wrapper.step(actions)
                terminations += int(done.sum())
                if completed % 100 == 0:
                    print(json.dumps(dict(policy_step=completed, physics_calls=calls)), flush=True)
    except FloatingPointError as exc:
        if failure is None:
            raise
        failure['error'] = repr(exc)
    finally:
        env.sim.step = original_step
        env.close()
    report = dict(solver=args.solver, integration_substeps=args.integration_substeps,
        fall_confirmation_ticks=wrapper.cfg.env.fail_ticks, terminations=terminations,
        external_force_scale=args.force_scale, external_torque_scale=args.torque_scale,
        refresh_spring_each_substep=args.refresh_spring,
        controller_dt_s=wrapper.cfg.sim.dt, integrator_dt_s=wrapper.cfg.sim.dt/args.integration_substeps,
        nonfinite_reproduced=failure is not None, failure=failure,
        last_policy_step=completed, physics_calls=calls, num_envs=4096,
        model_forwards=completed+1, optimizer_updates=0,
        scope='Fixed source policy with stochastic actions; omits PPO updates and their RNG consumption, not exact training replay')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
