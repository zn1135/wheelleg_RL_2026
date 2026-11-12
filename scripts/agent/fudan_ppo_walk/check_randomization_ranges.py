"""Read actual startup randomization in20 GPU worlds without policy or integration."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import torch

from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config, from_sim2sim_config


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    checks, values = {}, {}

    def verify(name, value):
        checks[name] = bool(value)
        assert checks[name], name

    def forbidden(*a, **kw):
        raise AssertionError('No physics integration, network forward or optimizer update')

    with ExitStack() as stack:
        for name in ('mjlab.envs.ManagerBasedRlEnv.step', 'mujoco.mj_step',
                     'torch.nn.Linear.forward', 'torch.optim.Adam.step'):
            stack.enter_context(patch(name, side_effect=forbidden))
        wrapper = FudanEnv(20, seed=30001, randomized=True)
        try:
            action = wrapper.action
            verify('legacy_physics', action.leg_load is None)
            legs=action._leg_action_ids;control=wrapper.cfg.control
            verify('nominal_leg_pd_matches_config',
                action._base_position_gains[legs].eq(control.leg_stiffness).all()
                and action._base_velocity_gains[legs].eq(control.leg_damping).all())
            verify('randomized_leg_pd_matches_config',
                torch.equal(action.position_gains[:,legs],control.leg_stiffness*action.kp_scale[:,legs])
                and torch.equal(action.velocity_gains[:,legs],control.leg_damping*action.kd_scale[:,legs]))
            values['nominal_leg_pd']=[control.leg_stiffness,control.leg_damping]
            values['actual_leg_p_minmax']=[float(action.position_gains[:,legs].min()),float(action.position_gains[:,legs].max())]
            values['actual_leg_d_minmax']=[float(action.velocity_gains[:,legs].min()),float(action.velocity_gains[:,legs].max())]
            for name in ('kp_scale', 'kd_scale', 'motor_strength'):
                value = getattr(action, name)
                verify(name + '_within_85to115_percent', value.ge(.85).all() and value.le(1.15).all())
                verify(name + '_actually_wider_than_before', value.min() < .95 and value.max() > 1.05)
                values[name] = value.cpu().tolist()
            state = action.disturbances
            verify('action_delay_3to7_physics_ticks', state.action_steps.ge(3).all() and state.action_steps.le(7).all())
            verify('action_delay_spans_both_sides_of10ms', state.action_steps.min() < 5 and state.action_steps.max() > 5)
            verify('training_sensor_still2ms', state.sensor_steps.eq(1).all())
            sim = from_sim2sim_config()
            verify('sim2sim_still_action10ms_sensor4ms', sim.action_delay_ms == (10., 10.) and sim.sensor_delay_ms == (4., 4.))
            env = wrapper.env
            ids = wrapper.robot.indexing.body_ids
            base_local = wrapper.robot.body_names.index('base_link')
            parameters = env._fudan_parameters
            default_mass = env.sim.get_default_field('body_mass')[ids]
            denominator = default_mass.expand(20, -1).clone()
            denominator[:, base_local] = parameters['base_mass']
            positive = default_mass > 0
            scales = env.sim.model.body_mass[:, ids][:, positive] / denominator[:, positive]
            verify('mass_multiplier_85to115_percent', scales.ge(.85-1e-6).all() and scales.le(1.15+1e-6).all())
            verify('mass_multiplier_actually_wider', scales.min() < .9 and scales.max() > 1.1)
            added = parameters['base_mass'] - default_mass[base_local]
            verify('base_added_mass_minus1point5_to3kg', added.ge(-1.5-1e-6).all() and added.le(3.+1e-6).all())
            verify('all_robot_mass_nonnegative', env.sim.model.body_mass[:, ids].ge(0).all())
            inertia = env.sim.model.body_inertia[:, ids][:, positive]
            expected_inertia = env.sim.get_default_field('body_inertia')[ids][positive][None] * scales[..., None]
            verify('inertia_scaled_with_mass', torch.allclose(inertia, expected_inertia, atol=1e-6))
            base = int(ids[base_local])
            com = env.sim.model.body_ipos[:, base] - env.sim.get_default_field('body_ipos')[base]
            verify('actual_com_within3cm', com.abs().le(.03+1e-6).all() and torch.allclose(com, parameters['com'], atol=1e-6))
            verify('actual_com_wider_than2cm', com.abs().max() > .02)
            body_ids = set(ids.cpu().tolist())
            geoms = [i for i, body in enumerate(env.sim.mj_model.geom_bodyid) if int(body) in body_ids]
            friction = parameters['friction']
            restitution = parameters['restitution']
            verify('friction_parameter_point4_to1point6', friction.ge(.4).all() and friction.le(1.6).all())
            verify('actual_contact_friction_matches_combination', torch.allclose(
                env.sim.model.geom_friction[:, geoms, 0], ((friction+.5)/2)[:, None].expand(-1, len(geoms))))
            verify('restitution_parameter_point4_to1', restitution.ge(.4).all() and restitution.le(1.).all())
            log_e = ((restitution+.5)/2).log()
            expected_damping = -200*(-log_e/(torch.pi**2+log_e.square()).sqrt())
            verify('actual_contact_restitution_matches_combination', torch.allclose(
                env.sim.model.geom_solref[:, geoms, 1], expected_damping[:, None].expand(-1, len(geoms)), atol=1e-5))
            verify('nominal_offsets_within_point05rad', action.nominal_offset.abs().le(.05).all())
            verify('nominal_offsets_actually_wider', action.nominal_offset.abs().max() > .03)
            # 同一状态开关噪声，直接检查送入Actor的变化，不推进物理。
            wrapper.randomized = False
            clean, clean_critic = wrapper.observations()
            wrapper.randomized = True
            noisy, noisy_critic = wrapper.observations()
            noise = noisy-clean
            limits = noisy.new_tensor(FudanWalkCfg.normalization.noise_scales)
            old_limits = noisy.new_tensor([.05]*6+[0.]*3+[.02]*4+[.075]*6+[0.]*6)
            verify('actor_noise_within_new_limits', noise.abs().le(limits+1e-6).all())
            verify('actor_noise_actually_wider', (noise.abs()>old_limits+1e-6).any())
            verify('critic_not_noised', torch.equal(clean_critic, noisy_critic))
            verify('commands_and_previous_action_not_noised', noise[:, 6:9].eq(0).all() and noise[:, 19:].eq(0).all())
            verify('small_push_config', state.cfg.pushes and sim.pushes and state.cfg.max_push_vel == sim.max_push_vel == .3
                   and state.cfg.push_interval_s == sim.push_interval_s == 7.)
            expected_scale = env.sim.model.body_mass[..., action._base_id].mean()*.3/wrapper.cfg.sim.dt
            verify('actual_push_force_scale', torch.allclose(action.push_force_scale, expected_scale.expand_as(action.push_force_scale)))
            state.ticks.fill_(state.push_interval-1)
            quat = torch.zeros(20, 4, device=wrapper.device)
            quat[:, 0] = 1.
            force = state.force(quat, action.push_force_scale)
            verify('push_occurs_at7s_boundary', state.push_count.eq(1).all() and force.abs().max()>0)
            verify('push_amplitude_including_half_vertical', force[:, :2].abs().le(action.push_force_scale[:, None]).all()
                   and force[:, 2].abs().le(action.push_force_scale*.5).all())
            verify('push_is_one_physics_tick', state.force(quat, action.push_force_scale).eq(0).all())
            values.update(mass_scale_minmax=[float(scales.min()), float(scales.max())],
                added_mass_kg=added.cpu().tolist(), com_m=com.cpu().tolist(),
                friction_parameter=friction.cpu().tolist(), restitution_parameter=restitution.cpu().tolist(),
                nominal_offsets=action.nominal_offset.cpu().tolist(), actor_noise=noise.cpu().tolist(),
                push_force_scale_n=action.push_force_scale.cpu().tolist(), synthetic_push_n=force.cpu().tolist())
            values.update(action_delay_steps=state.action_steps.cpu().tolist(), sensor_delay_steps=state.sensor_steps.cpu().tolist())
            result = dict(passed=all(checks.values()), checks=checks, actual=values,
                disturbances=from_task_config().record(), physics_backend='mujoco_warp_gpu',
                num_envs=20, physics_steps=0, model_forwards=0, optimizer_updates=0,
                scope='Actual GPU startup fields; not behavior validation or training precheck')
            (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({k: v for k, v in result.items() if k != 'actual'}, indent=2))
        finally:
            wrapper.env.close()


if __name__ == '__main__':
    main()
