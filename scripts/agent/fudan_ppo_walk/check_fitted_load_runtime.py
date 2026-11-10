"""GPU evidence for the explicitly enabled fitted load; no policy updates or behavior verdict."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.fitted_leg_load import parameters, parameter_record
from ct_mjlab.envs.chuanliantui.gas_spring import gas_spring_force
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, section_dict
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config, from_sim2sim_config
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import snapshot


def reference_load(q, bundle):
    # 独立按JSON展开保守负载，左右同时翻转角度和力矩，前后次序不换。
    p, b = bundle['candidate'], bundle['baseline']
    sign = np.array([-1., 1.])[:, None]
    delta = q * sign - np.array(b['reference_q_model_rad'][2:])
    u, w = np.array([-1., 1.]) / np.sqrt(2.), np.ones(2) / np.sqrt(2.)
    x, y = delta @ u, delta @ w
    force = -(np.array(b['bias_nm'][2:]) + delta @ np.array(b['stiffness_nm_rad'])[2:, 2:].T
              + np.array(p['bias_delta_nm']))
    force -= (p['length_stiffness_delta_nm_rad'] * x
              + p['length_quadratic_delta_nm_rad2'] * x * x)[..., None] * u
    force -= p['cross_stiffness_delta_nm_rad'] * (x[..., None] * w + y[..., None] * u)
    return force * sign


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    bundle = parameters()
    p = bundle['candidate']
    checks, evidence = {}, {}

    def verify(name, value):
        checks[name] = bool(value)
        assert checks[name], name

    verify('nominal_json', np.allclose(p['static_nm'], [.5, .5])
           and np.allclose(p['damping_nms_rad'], [.1436058254, .9722270164])
           and np.isclose(p['seal_damping_ns_m'], 2000.)
           and p['seal_static_n'] == 110. and p['seal_kinetic_n'] == 55.5
           and p['seal_velocity_scale_m_s'] == .0005
           and np.isclose(p['seal_bristle_stiffness_n_m'], 300000.))
    try:
        for randomized in (False, True):
            label = 'training' if randomized else 'nominal_gpu'
            task=FudanWalkCfg();task.sim=SimpleNamespace(**section_dict(task.sim));task.sim.fitted_leg_load=True
            wrapper = FudanEnv(20, seed=30001, randomized=randomized,
                               disturbances=from_task_config() if randomized else from_sim2sim_config(),cfg=task)
            env, action = wrapper.env, wrapper.action
            load = action.leg_load
            try:
                verify(label + '_active_profile', load is not None and load.record == parameter_record())
                names = [[env.sim.mj_model.joint(int(env.sim.mj_model.dof_jntid[v])).name
                          for v in row] for row in load.vi.cpu().tolist()]
                verify(label + '_motor_order', names == [['robot/lf0', 'robot/lf00'], ['robot/rf0', 'robot/rf00']])
                verify(label + '_actual_friction', torch.allclose(env.sim.model.dof_frictionloss[:, load.vi],
                       torch.tensor(p['static_nm'], device=wrapper.device).expand(20, 2, 2)))
                verify(label + '_actual_damping', torch.allclose(env.sim.model.dof_damping[:, load.vi],
                       torch.tensor(p['damping_nms_rad'], device=wrapper.device).expand(20, 2, 2)))
                expected_damping = env.sim.get_default_field('dof_damping')[load.sv] + p['seal_damping_ns_m']
                verify(label + '_slider_extra_damping', torch.allclose(env.sim.model.dof_damping[:, load.sv],
                       expected_damping.expand(20, 2)))
                verify(label + '_initial_memory_zero', (load.force == 0).all() and load.pending.all())
                # 直接跟踪实际施力和积分入口；不以配置开关代替运行证据。
                calls = dict(apply=0, physics=0)
                errors = dict(conservative_nm=0., memory_n=0., spring_n=0.)
                non_target = torch.ones(env.sim.mj_model.nv, dtype=torch.bool, device=wrapper.device)
                non_target[load.vi.flatten()] = False
                non_target[load.sv] = False
                original_apply, original_step = load.apply, env.sim.step

                def observed_apply():
                    data = env.sim.data
                    q = data.qpos[:, load.qi].cpu().numpy().copy()
                    position = data.qpos[:, load.sq].cpu().numpy().copy()
                    velocity = data.qvel[:, load.sv].cpu().numpy().copy()
                    previous = load.previous.cpu().numpy().copy()
                    previous[load.pending.cpu().numpy()] = position[load.pending.cpu().numpy()]
                    before = data.qfrc_applied[:, non_target].clone()
                    cap = p['seal_kinetic_n'] + (p['seal_static_n'] - p['seal_kinetic_n']) * np.exp(
                        -(velocity / p['seal_velocity_scale_m_s']) ** 2)
                    delta = position - previous
                    target = np.sign(delta) * cap
                    expected_memory = target + (load.force.cpu().numpy() - target) * np.exp(
                        -p['seal_bristle_stiffness_n_m'] * abs(delta) / cap)
                    original_apply()
                    errors['conservative_nm'] = max(errors['conservative_nm'], float(np.max(abs(
                        data.qfrc_applied[:, load.vi].cpu().numpy() - reference_load(q, bundle)))))
                    errors['memory_n'] = max(errors['memory_n'], float(np.max(abs(
                        data.qfrc_applied[:, load.sv].cpu().numpy() - (p['seal_force_bias_n'] - expected_memory)))))
                    expected_spring = gas_spring_force(data.qpos[:, load.sq], data.qvel[:, load.sv], action._spring_force_bias)
                    errors['spring_n'] = max(errors['spring_n'], float((data.ctrl[:, action._spring_ids] - expected_spring).abs().max()))
                    assert torch.equal(before, data.qfrc_applied[:, non_target])
                    assert calls['apply'] == calls['physics']
                    calls['apply'] += 1

                def observed_step(*a, **kw):
                    assert calls['apply'] == calls['physics'] + 1
                    result = original_step(*a, **kw)
                    calls['physics'] += 1
                    return result

                load.apply, env.sim.step = observed_apply, observed_step
                wrapper.reset()
                for _ in range(30):
                    wrapper.step(torch.zeros(20, 6, device=wrapper.device), auto_reset=False, resample=False)
                verify(label + '_one_load_per_physics_tick', calls == dict(apply=155, physics=155))
                verify(label + '_actual_qfrc_matches_full_json', errors['conservative_nm'] < 2e-5 and errors['memory_n'] < .002)
                verify(label + '_original_spring_still_present', errors['spring_n'] < .0001)
                verify(label + '_memory_dynamically_nonzero', load.force.abs().max() > 1.)
                verify(label + '_finite_gpu_state', torch.isfinite(env.sim.data.qpos).all()
                       and torch.isfinite(env.sim.data.qvel).all() and torch.isfinite(load.force).all())
                load.apply, env.sim.step = original_apply, original_step
                np.savez_compressed(args.output / (label + '-forces.npz'),
                    qpos=env.sim.data.qpos.cpu().numpy(), qvel=env.sim.data.qvel.cpu().numpy(),
                    qfrc_applied=env.sim.data.qfrc_applied.cpu().numpy(),
                    memory_force=load.force.cpu().numpy(), ctrl=env.sim.data.ctrl.cpu().numpy())
                ids = torch.tensor([0, 3, 7], device=wrapper.device)
                other = torch.tensor([i for i in range(20) if i not in [0, 3, 7]], device=wrapper.device)
                saved = [v[other].clone() for v in (load.force, load.previous, load.pending,
                         env.sim.data.qpos, env.sim.data.qvel, env.sim.data.qfrc_applied)]
                wrapper.reset_idx(ids)
                verify(label + '_partial_reset_clears_memory', (load.force[ids] == 0).all() and load.pending[ids].all())
                verify(label + '_partial_reset_clears_applied_force',
                       (env.sim.data.qfrc_applied[ids][:, load.vi] == 0).all()
                       and (env.sim.data.qfrc_applied[ids][:, load.sv] == 0).all())
                verify(label + '_other_environments_unchanged', all(torch.equal(v[other], old) for v, old in zip(
                    (load.force, load.previous, load.pending, env.sim.data.qpos, env.sim.data.qvel,
                     env.sim.data.qfrc_applied), saved)))
                original_apply()
                verify(label + '_partial_reset_latches_pose_without_impulse', (load.force[ids] == 0).all())
                wrapper.reset_idx(torch.arange(20, device=wrapper.device))
                verify(label + '_all_reset_clears_memory', (load.force == 0).all() and load.pending.all())
                original_apply()
                verify(label + '_all_reset_latches_pose_without_impulse', (load.force == 0).all())
                # 固定状态再次施力不得累加；零位移也不得制造记忆力。
                force_before = env.sim.data.qfrc_applied.clone()
                original_apply()
                verify(label + '_no_force_accumulation', torch.equal(force_before, env.sim.data.qfrc_applied))
                evidence[label] = dict(profile=load.record, motor_names=names, calls=calls, maximum_errors=errors,
                    dof_frictionloss=env.sim.model.dof_frictionloss[0, load.vi].cpu().tolist(),
                    dof_damping=env.sim.model.dof_damping[0, load.vi].cpu().tolist(),
                    slider_damping=env.sim.model.dof_damping[0, load.sv].cpu().tolist(),
                    randomization_events=list(env.cfg.events),
                    note='Fudan randomizes mass/contact/PD gains; joint friction/damping retain the fitted nominal values.')
            finally:
                env.close()
        # 左右完全镜像输入须给相反力矩；非对称前后输入用于排除轴交换。
        q = np.array([[[-.71, .23], [.71, -.23]], [[-.55, .39], [.55, -.39]]])
        expected = reference_load(q, bundle)
        verify('mirror_direction_front_rear_not_swapped', np.allclose(expected[:, 0], -expected[:, 1])
               and not np.allclose(expected[:, 0, 0], expected[:, 0, 1]))
        snapshot(args.output / 'snapshot')
        frozen = args.output / 'snapshot/source/resources/robots/chuanliantui_new_1/mjlab/fitted_leg_load.json'
        verify('snapshot_identity', hashlib.sha256(frozen.read_bytes()).hexdigest() == parameter_record()['sha256'])
    finally:
        result = dict(passed=bool(checks) and all(checks.values()) and 'snapshot_identity' in checks,
                      checks=checks, evidence=evidence, physical_leg_load=parameter_record(),
                      physics_backend='mujoco_warp_gpu', policy_training_updates=0,
                      scope='GPU actual-force/integration/reset/randomization/source checks; not motion acceptance')
        (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
