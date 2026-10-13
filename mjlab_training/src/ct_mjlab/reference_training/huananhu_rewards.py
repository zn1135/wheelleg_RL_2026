"""Published V14 flat reward weights on native chuanliantui motors.

RewardManager supplies the sole control-dt multiplication. The v7 experiment
uses the published 2026-07-19_09-14-50 weights; v9 tightens its velocity-tracking
height gate from 5–10 cm to 3–6 cm. v11 adds a soft knee joint-limit penalty so a
squat resting on the passive knee stops is no longer free. Local r11 restores
the reference weight after r9/r10 disabled it; local r12 disables it for resumed training. No landing/standing gate
or per-term cap is applied.
v12 replaces only linear-velocity exponential tracking with a reciprocal quadratic
so errors beyond 1 m/s remain distinguishable. Peak reward and local curvature
are unchanged; this is not the published source formula.
Local r13 restores the 5–10 cm velocity-reward height gate. Historical
checkpoints keep their recorded 3–6 cm gate during replay.
The source repository is not needed at runtime.
"""

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .model import ACTUATED_JOINTS


REWARD_WEIGHTS = {
    'termination': -200., 'leg_joint_acc': -5e-7, 'leg_joint_vel': -.005,
    'joint_torque': -1e-4, 'wheel_acc': -1e-8, 'wheel_vel': -1e-5,
    'wheel_power': -1e-4, 'lin_vel_z': -.5, 'ang_vel_xy': -.05,
    'action_smoothness_leg': -.05, 'action_rate': -.01,
    'action_smoothness_wheel': -.01, 'flat_orientation_y_v': -2.,
    'flat_orientation_y_exp': 1., 'flat_orientation_x_v': -2.,
    'flat_orientation_x_exp': 1., 'track_lin_vel_xy': 1.,
    'track_lin_vel_xy_square': -.1, 'track_ang_vel_z': 1.,
    'track_ang_vel_z_square': -.1, 'stand_still_lin_vel': -2.,
    'track_height_exp_tight': 1., 'track_height_square': -1.,
    'no_fork': -1., 'no_fork_square': -1., 'undesired_contact': -2.,
    # Keep the formula for historical replay; r12 disables its contribution.
    'knee_limit': 0.,
}
# Soft zone inside each passive knee limit. Commanded 0.27–0.34 m stances keep the
# knees well over 0.15 rad from both stops; the v7/v10 squat sits at the stop.
KNEE_LIMIT_MARGIN = .15
KNEE_JOINTS = ('lf1', 'rf1')
HEIGHT_GATE = (.05, .10)
LEGS = (0, 1, 3, 4)
WHEELS = (2, 5)


def reward_values(*, command, linear_velocity, angular_velocity, gravity, pitch,
                  height, wheel_x, joint_velocity, joint_acceleration, torque,
                  action, previous_action, previous_previous_action,
                  undesired_contact, terminated, knee_limit_distance,
                  height_gate=HEIGHT_GATE):
    """Height-gated source formulas; command=[vx,yaw,height], motors=L,L,W,R,R,W.

    knee_limit_distance is each knee's distance to its nearer XML limit (negative
    when past it), order (lf1, rf1).
    """
    still = command[:, 0].abs() < .1
    vx_command = torch.where(still, 0., command[:, 0])
    yaw_command = torch.where(command[:, 1].abs() < .1, 0., command[:, 1])
    velocity_error = vx_command - linear_velocity[:, 0] * pitch.cos()
    yaw_error = yaw_command - angular_velocity[:, 2]
    height_error = height - command[:, 2]
    full, off = height_gate
    if not 0. < full < off:
        raise ValueError('Height gate requires 0 < full < off')
    velocity_height_gate = ((off - height_error.abs()) / (off - full)).clamp(0., 1.)
    orientation_gain = 2. * torch.exp(-command[:, 0].square() / 3.) + 2.
    second_difference = action - 2. * previous_action + previous_previous_action
    wheel_dx = wheel_x[:, 0] - wheel_x[:, 1]
    return {
        'termination': terminated.float(),
        'leg_joint_acc': joint_acceleration[:, LEGS].square().sum(-1),
        'leg_joint_vel': joint_velocity[:, LEGS].square().sum(-1),
        'joint_torque': torque.square().sum(-1),
        'wheel_acc': joint_acceleration[:, WHEELS].square().sum(-1),
        'wheel_vel': joint_velocity[:, WHEELS].square().sum(-1),
        'wheel_power': (torque[:, WHEELS] * joint_velocity[:, WHEELS]).clamp_min(0.).sum(-1),
        'lin_vel_z': linear_velocity[:, 2].square(),
        'ang_vel_xy': angular_velocity[:, :2].square().sum(-1),
        'action_smoothness_leg': second_difference[:, LEGS].square().sum(-1),
        'action_rate': (action - previous_action).square().sum(-1),
        'action_smoothness_wheel': second_difference[:, WHEELS].square().sum(-1),
        'flat_orientation_y_v': (orientation_gain * gravity[:, 0]).square(),
        'flat_orientation_y_exp': torch.exp(-gravity[:, 0].square() / .02),
        'flat_orientation_x_v': (orientation_gain * gravity[:, 1]).square(),
        'flat_orientation_x_exp': torch.exp(-gravity[:, 1].square() / .01),
        'track_lin_vel_xy': (1. + 2. * velocity_error.square()).reciprocal() * velocity_height_gate,
        'track_lin_vel_xy_square': (.25 * velocity_error).square() * velocity_height_gate,
        'track_ang_vel_z': torch.exp(-yaw_error.clamp(-.8, .8).square() / .25) * velocity_height_gate,
        'track_ang_vel_z_square': (.5 * yaw_error).square() * velocity_height_gate,
        'stand_still_lin_vel': still.float() * linear_velocity[:, :2].abs().sum(-1),
        'track_height_exp_tight': torch.exp(-height_error.clamp(-.15, .15).square() / .001),
        'track_height_square': (10. * height_error).square(),
        'no_fork': (wheel_dx.abs() > .05).float(),
        'no_fork_square': (5. * wheel_dx).square(),
        'undesired_contact': undesired_contact.float(),
        # 0 outside the margin, 1 at the stop; capped so soft-constraint overshoot stays bounded.
        'knee_limit': ((KNEE_LIMIT_MARGIN - knee_limit_distance).clamp_min(0.) / KNEE_LIMIT_MARGIN
                       ).clamp_max(1.5).square().sum(-1),
    }


class RewardState:
    def __init__(self, env, height_gate=HEIGHT_GATE):
        self.height_gate = tuple(height_gate)
        self.robot = env.scene['robot']
        self.joint_ids, names = self.robot.find_joints(ACTUATED_JOINTS, preserve_order=True)
        if tuple(names) != ACTUATED_JOINTS:
            raise ValueError(f'Unexpected reward motor order: {names}')
        self.v_adr = self.robot.indexing.joint_v_adr[self.joint_ids]
        self.wheel_ids, _ = self.robot.find_bodies(('lfwheel', 'rfwheel'), preserve_order=True)
        knee_ids, knee_names = self.robot.find_joints(KNEE_JOINTS, preserve_order=True)
        if tuple(knee_names) != KNEE_JOINTS:
            raise ValueError(f'Unexpected knee joints: {knee_names}')
        self.knee_q_adr = self.robot.indexing.joint_q_adr[torch.tensor(knee_ids, device=env.device)]
        # Static XML limits; mjlab prefixes scene joints with the entity name.
        limits = self.robot.data.default_joint_pos_limits[0, knee_ids]
        self.knee_lower, self.knee_upper = limits[:, 0].clone(), limits[:, 1].clone()
        self.step = -1
        self.values = {}

    def compute(self, env):
        if self.step == env.common_step_counter:
            return self.values
        data = self.robot.data
        relative = data.body_link_pos_w[:, self.wheel_ids] - data.root_link_pos_w[:, None, :]
        quat = data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
        wheel_pos = quat_apply_inverse(quat.reshape(-1, 4), relative.reshape(-1, 3)).reshape(-1, 2, 3)
        force = env.scene['undesired_contact'].data.force_history
        contact = force.norm(dim=-1).amax(dim=(1, 2)) > 3.
        gravity = data.projected_gravity_b
        knee = env.sim.data.qpos[:, self.knee_q_adr]
        manager = env.action_manager
        # The reward kernel remains in physical motor order. Convert all three
        # requested-action histories together; observations retain policy order.
        action_order = manager.get_term('pd').policy_to_physical
        command_state = env.command_manager.get_term('standup')
        height = getattr(command_state, 'height_above_ground', data.root_link_pos_w[:, 2])
        self.values = reward_values(
            command=command_state.command,
            linear_velocity=data.root_link_lin_vel_b, angular_velocity=data.root_link_ang_vel_b,
            gravity=gravity, pitch=torch.asin(gravity[:, 0].clamp(-1., 1.)),
            height=height, wheel_x=wheel_pos[:, :, 0],
            joint_velocity=env.sim.data.qvel[:, self.v_adr],
            joint_acceleration=env.sim.data.qacc[:, self.v_adr],
            torque=env.sim.data.qfrc_actuator[:, self.v_adr],
            action=manager.action[:, action_order], previous_action=manager.prev_action[:, action_order],
            previous_previous_action=manager.prev_prev_action[:, action_order],
            undesired_contact=contact, terminated=env.reset_terminated,
            knee_limit_distance=torch.minimum(knee - self.knee_lower, self.knee_upper - knee),
            height_gate=self.height_gate,
        )
        state = env.command_manager.get_term('standup')
        if hasattr(state, 'jump_phase'):
            jumping = state.jump_phase > 0
            # The multimode task uses a dedicated airborne target during a
            # requested jump. Grounded height/velocity targets must not punish
            # the same takeoff that the jump reward asks for.
            for name in ('track_height_exp_tight', 'track_height_square',
                         'track_lin_vel_xy', 'track_lin_vel_xy_square',
                         'track_ang_vel_z', 'track_ang_vel_z_square'):
                self.values[name] = torch.where(jumping, 0., self.values[name])
        self.values = {name: torch.nan_to_num(value, nan=0., posinf=0., neginf=0.).masked_fill(
            state.numerical_failure, 0.) for name, value in self.values.items()}
        self.step = env.common_step_counter
        return self.values


def reward(env, *, name, height_gate=HEIGHT_GATE):
    if not hasattr(env, '_huananhu_rewards'):
        env._huananhu_rewards = RewardState(env, height_gate)
    if env._huananhu_rewards.height_gate != tuple(height_gate):
        raise ValueError('All cached reward terms must use the same height gate')
    return env._huananhu_rewards.compute(env)[name]
