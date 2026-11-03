"""Motion-only imagination; deliberately differs from the real task MDP.

Upstream mdp.py estimates reward torques with PD and fills omitted wheel q
with zero for kinematics.  This adapter follows that construction using this
robot's unchanged PD configuration.  It never reconstructs contact forces.
"""

import math

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from .analytic_reward import WEIGHTS, differentiable_reward, _upright
from .state import (KD, KP, MEMORY, NOMINAL, SENSORS, NOISE, PHYSICAL,
                    STRENGTH, StateAdapter, unit_quat)


MOTION_AUXILIARY_RECIPE = dict(
    version='control-endpoint-pd-soft-termination-v3',
    torque='analytic_pd_from_predicted_endpoint_q_v_current_action_not_learned',
    action_order=['lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel'],
    leg_position_scale=.5, wheel_velocity_scale=10., max_wheel_velocity=150.,
    wheel_stiffness=.1, wheel_damping=0., motor_response_time_constant_s=0.,
    ground_start_hold_s=.2, leg_target_reference_clip_rad=3.14,
    action_lag='endpoint_reward_uses_current_action_no_imagined_actuator_queue',
    observation='endpoint_sensors_plus_noise_no_2ms_delay_queue',
    reward_time='control_endpoint_10ms',
    torque_limits_before_xml_gain=[40., 40., 3.9, 40., 40., 3.9],
    wheel_response='xml_speed_attenuation_then_xml_fixed_gain_once',
    wheel_response_backward='same_forward_safe_fraction_floor_1e-6',
    wheel_geometry_q='zero', height_failure_m=.18, tilt_failure_degrees=40.,
    root_linear_speed_failure_m_s=5., root_angular_speed_failure_rad_s=30.,
    root_speed_norm='all_three_components', nonfinite_retained_motion='hard_failure',
    invalid_quaternion_norm_min=1e-8,
    soft_height_width_m=.01, soft_tilt_cos_width=.02,
    soft_linear_speed_width_m_s=.5, soft_angular_speed_width_rad_s=3.,
    memory='age_only_other_nine_zero',
    disabled_model_reward_terms=[],
    model_reward_differences=['estimated_not_actual_motor_torque',
        'base_height_without_contact_attenuation',
        'motion_only_soft_termination_not_real_task_failure',
        'endpoint_geometry_not_pre_last_physics_step',
        'current_action_torque_not_delayed_action'],
    height_reward='exp(-(height-target)^2/.0025)*upright_gate; no_contact_factor',
    reward_not_equivalent_to_real_task=True,
)


IMAGINATION_CLAMP_RECIPE = dict(
    version='control-endpoint-imagination-clamp-v1',
    height_range_m=[.02, 2.], body_linear_component_limit_m_s=20.,
    body_angular_component_limit_rad_s=40.,
    joint_position_limits='compiled_XML_jnt_limited_only',
    wheel_speed_limit_rad_s=MOTION_AUXILIARY_RECIPE['max_wheel_velocity'],
    wheel_speed_limit_source='existing_PD_target_bound_used_as_numerical_guard_not_measured_speed_limit',
    other_joint_speed_limits='unbounded_no_verified_local_limits',
    nonfinite='preserved_for_existing_failure_checks',
    application='after_prediction_before_observation_reward_survival_and_next_imagination_step',
    world_fit_and_real_labels='not_clamped',
)


def validate_motion_pd_config(action_cfg):
    """Reject a task PD change instead of silently estimating another actuator."""
    for name in ('leg_position_scale', 'wheel_velocity_scale', 'max_wheel_velocity',
                 'wheel_stiffness', 'wheel_damping', 'motor_response_time_constant_s',
                 'ground_start_hold_s'):
        if getattr(action_cfg, name) != MOTION_AUXILIARY_RECIPE[name]:
            raise ValueError('Motion-only analytic PD configuration mismatch: '+name)


def clean_physical(physical):
    """The 64-wide compatibility container carries no removed quantities."""
    result = physical.clone()
    result[..., 9] = 0.
    result[..., 14] = 0.
    result[..., 49:] = 0.
    return result


class MotionStateAdapter(StateAdapter):
    def __init__(self):
        super().__init__()
        from ct_mjlab.envs.chuanliantui_split.env_cfg import make_walk_env_cfg
        from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS, load_model
        validate_motion_pd_config(make_walk_env_cfg().actions['pd'])
        model = load_model()
        motors = [model.actuator(name+'_motor').id for name in ACTUATED_JOINTS]
        self.register_buffer('pd_reference', torch.tensor(
            model.qpos0[self.motor_q.cpu().numpy()], dtype=torch.float32))
        self.register_buffer('pd_gain', torch.tensor(model.actuator_gainprm[motors, 0], dtype=torch.float32))
        self.register_buffer('pd_response', torch.tensor(model.actuator_gainprm[motors, :3], dtype=torch.float32))
        self.register_buffer('pd_limits', torch.tensor(model.actuator_ctrlrange[motors], dtype=torch.float32))
        self.legs, self.wheel_motor_ids = [0, 1, 3, 4], [2, 5]
        lower = torch.zeros(18)
        upper = torch.zeros(18)
        limited = torch.zeros(18, dtype=torch.bool)
        for index in range(1, model.njnt):
            if model.jnt_limited[index]:
                column = int(model.jnt_qposadr[index])-7
                lower[column], upper[column] = map(float, model.jnt_range[index])
                limited[column] = True
        velocity = torch.zeros(18)
        velocity[self.motor_v[self.wheel_motor_ids]-31] = IMAGINATION_CLAMP_RECIPE['wheel_speed_limit_rad_s']
        self.register_buffer('imagination_q_lower', lower)
        self.register_buffer('imagination_q_upper', upper)
        self.register_buffer('imagination_q_limited', limited)
        self.register_buffer('imagination_joint_speed_limit', velocity)

    def sanitize_endpoint(self, physical):
        """Clamp imagined physical endpoints only; preserve unknown limits and nonfinites."""
        from mjlab.utils.lab_api.math import quat_apply
        recipe = IMAGINATION_CLAMP_RECIPE
        q = unit_quat(physical[..., 3:7])
        body_velocity = quat_apply_inverse(q, physical[..., 25:28])
        bound = recipe['body_linear_component_limit_m_s']
        clipped_body = body_velocity.clamp(-bound, bound)
        # Keep already-valid world velocities exactly, avoiding rotation roundoff.
        linear = torch.where((body_velocity.abs()>bound).any(-1, keepdim=True),
                             quat_apply(q, clipped_body), physical[..., 25:28])
        angular_bound = recipe['body_angular_component_limit_rad_s']
        joint_q = torch.where(self.imagination_q_limited,
            torch.maximum(torch.minimum(physical[..., 7:25], self.imagination_q_upper), self.imagination_q_lower), physical[..., 7:25])
        joint_v = torch.where(self.imagination_joint_speed_limit>0,
            torch.maximum(torch.minimum(physical[..., 31:49], self.imagination_joint_speed_limit), -self.imagination_joint_speed_limit), physical[..., 31:49])
        bounded = torch.cat((physical[..., :2], physical[..., 2:3].clamp(*recipe['height_range_m']),
            physical[..., 3:7], joint_q, linear, physical[..., 28:31].clamp(-angular_bound, angular_bound),
            joint_v, physical[..., 49:]), -1)
        return torch.where(torch.isfinite(physical), bounded, physical)

    def estimated_torque(self, physical, parameters, action):
        """Reward-only PD estimate at the predicted endpoint using current action."""
        legs, wheels = self.legs, self.wheel_motor_ids
        q, qd = physical[..., self.motor_q], physical[..., self.motor_v]
        kp, kd = parameters[..., KP], parameters[..., KD]
        target = (parameters[..., NOMINAL][..., legs]+.5*action[..., legs]).clamp(
            self.pd_reference[legs]-3.14, self.pd_reference[legs]+3.14)
        leg_torque = kp[..., legs]*(target-q[..., legs])-kd[..., legs]*qd[..., legs]
        velocity = (10.*action[..., wheels]).clamp(-150., 150.)
        # Captured KD contains kd_scale*(wheel_stiffness+wheel_damping).
        wheel_torque = kd[..., wheels]*(velocity-qd[..., wheels])
        result = torch.stack((leg_torque[..., 0], leg_torque[..., 1], wheel_torque[..., 0],
                              leg_torque[..., 2], leg_torque[..., 3], wheel_torque[..., 1]), -1)
        result = (result*parameters[..., STRENGTH]).clamp(self.pd_limits[:, 0], self.pd_limits[:, 1])
        command = result[..., wheels]
        along = (command.sign()*qd[..., wheels]).clamp_min(0.)
        response = self.pd_response[wheels]
        fraction = (1.-along/response[:, 1]).clamp_min(0.)
        attenuation = fraction.pow(response[:, 2])
        safe_attenuation = fraction.clamp_min(1e-6).pow(response[:, 2])
        # Preserve XML forward attenuation; avoid infinite derivative at zero.
        attenuation = attenuation.detach()+(safe_attenuation-safe_attenuation.detach())
        wheel_torque = command*attenuation
        result = torch.stack((result[..., 0], result[..., 1], wheel_torque[..., 0],
                              result[..., 3], result[..., 4], wheel_torque[..., 1]), -1)
        return result*self.pd_gain

    def reward_features(self, path, parameters=None, action=None, previous=None, state=None):
        path = clean_physical(path)
        current = path[..., -1, :]
        q = unit_quat(current[..., 3:7])
        gravity = torch.zeros_like(current[..., :3]); gravity[..., 2] = -1.
        position, _, linear, angular = self.kin(current[..., :25], current[..., 25:49])
        relative = position[..., self.wheels, :]-position[..., self.hips, :]
        relative = quat_apply_inverse(q[..., None, :].expand(*relative.shape[:-1], 4), relative)
        swing = torch.atan2(-relative[..., 0], -relative[..., 2])
        torque = torch.zeros_like(current[..., :6])
        if parameters is not None:
            torque = self.estimated_torque(current, parameters, action)
            if state is not None:
                hold = (parameters[..., 63] > .5) & (state[..., MEMORY][..., 0] < 20.)
                torque = torque.masked_fill(hold[..., None], 0.)
        # Last three zero slots mean unavailable, not predicted force/support.
        return torch.cat((quat_apply_inverse(q, linear[..., 1, :]),
            quat_apply_inverse(q, angular[..., 1, :]), quat_apply_inverse(q, gravity),
            current[..., 2:3], swing, current[..., self.leg_q], current[..., self.knee_q],
            current[..., self.leg_v], torque, torch.zeros_like(current[..., :3])), -1)

    def _failure_coordinates(self, physical):
        physical = clean_physical(physical[..., :64])
        finite = (torch.isfinite(physical[..., :49]).all(-1)
                  & (physical[..., 3:7].norm(dim=-1) > 1e-8))
        safe = torch.nan_to_num(physical, nan=0., posinf=0., neginf=0.)
        q = unit_quat(safe[..., 3:7])
        upright_cos = 1.-2.*(q[..., 1].square()+q[..., 2].square())
        return (safe[..., 2], upright_cos, safe[..., 25:28].norm(dim=-1),
                safe[..., 28:31].norm(dim=-1), finite)

    def hard_failure(self, physical):
        height, upright, speed, angular_speed, finite = self._failure_coordinates(physical)
        return (~finite | (height <= .18) | (upright < math.cos(math.radians(40.)))
                | (speed > 5.) | (angular_speed > 30.))

    def soft_survival(self, physical):
        """Product of sigmoid survival factors, as upstream's soft-term union."""
        height, upright, speed, angular_speed, finite = self._failure_coordinates(physical)
        result = (torch.sigmoid((height-.18)/.01)
            *torch.sigmoid((upright-math.cos(math.radians(40.)))/.02)
            *torch.sigmoid((5.-speed)/.5)*torch.sigmoid((30.-angular_speed)/3.))
        return result.masked_fill(~finite, 0.)

    def continuation(self, path):
        return self.soft_survival(path[..., -1, :])

    def advance_memory(self, state, parameters, path):
        physical = clean_physical(path[..., -1, :])
        age = state[..., MEMORY][..., :1]+1.
        memory = torch.cat((age, torch.zeros_like(state[..., MEMORY][..., 1:])), -1)
        return memory, self.hard_failure(physical), ~self._failure_coordinates(physical)[-1]

    def advance(self, state, parameters, path, command, action, previous, previous_previous, context, noise):
        path = clean_physical(path)
        memory, failure, numerical = self.advance_memory(state, parameters, path)
        # Keep the recorded-state container, but do not fabricate a2ms sensor queue.
        next_state = torch.cat((path[..., -1, :], torch.zeros_like(state[..., SENSORS]), noise, memory), -1)
        frame = self.actor_frame(next_state, parameters, command, action)
        features = self.reward_features(path, parameters, action, previous, state)
        active = ~((parameters[..., 63] > .5) & (state[..., MEMORY][..., 0] < 20.))
        reward_context = torch.cat((context[..., :5], numerical[..., None].float(),
                                    active[..., None].float(), context[..., 7:]), -1)
        values, _ = differentiable_reward(features, action, previous, previous_previous, reward_context)
        # The model's height reward has no contact term, even as a placeholder.
        height_reward = (torch.exp(-(features[...,9]-reward_context[...,2]).square()/.0025)
                         *_upright(features[...,8].clamp(-1.,1.)))
        masked = (reward_context[...,5] > .5) | ~(reward_context[...,6] > .5)
        values['base_height'] = height_reward.masked_fill(masked,0.)
        reward = torch.zeros_like(values['tracking_lin_vel'])
        for name, weight in WEIGHTS.items():
            reward = reward+torch.nan_to_num(values[name]*weight*reward_context[..., 20],
                                           nan=0., posinf=0., neginf=0.)
        return next_state, frame, reward, failure, features

    def actor_frame(self, state, parameters, command, action):
        sensor = self.raw_sensors(state[..., PHYSICAL], parameters)+state[..., NOISE]
        return torch.cat((sensor[..., :3]*.25, sensor[..., 3:6], command,
            sensor[..., 6:10], sensor[..., 12:18][..., (0,1,4,2,3,5)]*.05,
            action), -1).clamp(-100., 100.)
