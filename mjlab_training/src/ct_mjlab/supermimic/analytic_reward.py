"""Differentiable copy of the current chuanliantui walk reward, without physics edits.

Source: ct_mjlab.envs.chuanliantui_split.rewards.WalkRewardState.compute,
recipe split-walk-relative-velocity-parking-noboost-tilt5-symmetry3cap10-no-vz-contact-rate-smooth-v7.
This module does not instantiate a simulator or change the source reward.

Features (31, in reward-time physical units): LINK linear velocity [0:3],
LINK angular velocity [3:6], projected gravity [6:9], terrain-relative LINK
height [9], left/right hip-to-wheel swing [10:12], driven-leg q [12:16],
passive-knee q [16:18], driven-leg dq [18:22], six actual motor torques
[22:28], base contact force norm [28], two wheel-support flags [29:31].
Driven-leg order is lf0, lf00, rf0, rf00; motor order is lf0, lf00, lfwheel,
rf0, rf00, rfwheel. Knees use reward_state.knees, in its preserved order.

Context (21): command vx/yaw/height [0:3], tracking_active [3],
position_locked [4], numerical_failure [5], reward_active [6], four driven
leg soft limits, flattened joint-major lower/upper [7:15], two passive-knee
limits in the same layout [15:19], tight kernel denominator [19], dt [20].
Booleans are encoded as 0/1. Limits, masks, commands and dt are exogenous
context; the world model must not invent them. Unlimited soft limits remain
infinite, as in the XML and source reward.

capture() must run during the ORIGINAL reward compute, after termination
state updates and before auto-reset, sim.forward() and command updates.
Some geometry deliberately lags qpos by one 2 ms substep in this mjlab
version. Capturing after env.step() silently changes the reward target.
action/previous/prev_previous are unscaled, unclipped raw six-motor actions,
at that same reward time; do not substitute the observation's clipped last
action. Imagined rollouts must shift these two histories on every step.

The forward formula, masks, 13 weights, per-term caps and dt match the source.
The upright factor uses a bounded equivalent backward path to avoid the
undefined acos derivative at an exactly upright or inverted gravity vector.
Its forward values remain those of the original source expression. Contact
support, masks, max/clamp boundaries are discrete/nondifferentiable; no soft
contact or extra reward is introduced. Gradients through physical features
only help when a learned differentiable dynamics model predicts them.
"""

import math

import torch


FEATURE_DIM = 31
CONTEXT_DIM = 21
FEATURE_SLICES = dict(linear_velocity=slice(0, 3), angular_velocity=slice(3, 6),
    gravity=slice(6, 9), height=slice(9, 10), leg_swing=slice(10, 12),
    leg_q=slice(12, 16), knee_q=slice(16, 18), leg_dq=slice(18, 22),
    torque=slice(22, 28), base_force=slice(28, 29), wheel_supported=slice(29, 31))
CONTEXT_SLICES = dict(command=slice(0, 3), tracking_active=slice(3, 4),
    position_locked=slice(4, 5), numerical_failure=slice(5, 6),
    reward_active=slice(6, 7), soft_limits=slice(7, 15),
    knee_limits=slice(15, 19), tight_denominator=slice(19, 20), dt=slice(20, 21))

# Preserve source configuration order, including the tight term appended last.
WEIGHTS = dict(tracking_lin_vel=.2, tracking_lin_vel_enhance=.2,
    tracking_ang_vel=1., tracking_ang_vel_enhance=.2, base_height=1.,
    orientation=-100., nominal_state=-3., dof_pos_limits=-1.,
    knee_limit_margin=-.3,
    ang_vel_xy=-.2, dof_vel=-5e-5, torques=-1e-4,
    tracking_lin_vel_tight=.5)
WEIGHTED_LIMITS = dict(orientation=5., nominal_state=10.)
ORIENTATION_FREE_RAD = math.radians(5.)


def return_bounds(context, gamma):
    """Conservative discounted return bounds from the reward's rate caps.

    Positive rates: boosted vx<=1, yaw<=1, height<=1, tight vx<=.5.
    Negative rates: orientation>=-5, nominal>=-10, four other capped costs
    total >=-4, knee>=-.3 and enhancements>=-.4, giving >=-19.7/s.
    """
    scale = context[...,20]/(1.-gamma)
    return -20.*scale, 3.5*scale


def _upright(gravity_z):
    """Keep the original forward values and finite gradients in saturated regions."""
    source = ((math.radians(40)-torch.acos(-gravity_z))/math.radians(25)).clamp(0., 1.)
    # The true derivative is zero outside the unsaturated 15..40 degree range.
    safe_z = gravity_z.clamp(-math.cos(math.radians(15)), -math.cos(math.radians(40)))
    safe = ((math.radians(40)-torch.acos(-safe_z))/math.radians(25)).clamp(0., 1.)
    return source.detach() + (safe-safe.detach())


def differentiable_reward(features, action, previous, previous_previous, context):
    """Return (unweighted source term dictionary, total weighted per-step reward).

    Inputs can have any shared leading dimensions (..., 31/6/21). Context
    may broadcast over those dimensions. Values include the source's term
    caps and numerical/reward-active mask. The total also reproduces mjlab
    RewardManager's per-term weight, dt and nan_to_num, in source order.
    """
    linear, angular = features[..., 0:3], features[..., 3:6]
    gravity_z = features[..., 8].clamp(-1., 1.)
    height, swing = features[..., 9], features[..., 10:12]
    leg_q, knee_q = features[..., 12:16], features[..., 16:18]
    leg_dq, torque = features[..., 18:22], features[..., 22:28]
    contact = (features[..., 28]/5.).clamp(0., 1.)
    command = context[..., 0:3]
    tracking_active, locked = context[..., 3] > .5, context[..., 4] > .5
    masked = (context[..., 5] > .5) | ~(context[..., 6] > .5)
    soft_limits = context[..., 7:15].reshape(*context.shape[:-1], 4, 2)
    knee_limits = context[..., 15:19].reshape(*context.shape[:-1], 2, 2)
    vx_error = (command[..., 0]-linear[..., 0]).square()
    yaw_error = (command[..., 1]-angular[..., 2]).square()
    boost = torch.where(tracking_active & ~locked, 5., 1.)
    vx_kernel = command[..., 0].square().clamp_min(.25)
    penetration = torch.maximum(knee_limits[..., 0]+.05-knee_q,
                                knee_q-knee_limits[..., 1]+.05).clamp_min(0.)
    values = dict(
        tracking_lin_vel=boost*torch.exp(-vx_error/vx_kernel),
        tracking_lin_vel_enhance=torch.exp(-vx_error/2.5)-1.,
        tracking_ang_vel=torch.exp(-yaw_error/.25),
        tracking_ang_vel_enhance=torch.exp(-yaw_error/2.5)-1.,
        tracking_lin_vel_tight=torch.exp(-vx_error/context[..., 19])
            .masked_fill(~tracking_active, 0.),
        base_height=torch.exp(-(height-command[..., 2]).square()/.0025)
            *_upright(gravity_z)*(1.-.8*contact),
        nominal_state=(swing[..., 0]-swing[..., 1]).square(),
        orientation=(1.+gravity_z-(1.-math.cos(ORIENTATION_FREE_RAD))).clamp_min(0.),
        dof_pos_limits=((soft_limits[..., 0]-leg_q).clamp_min(0.)+
                        (leg_q-soft_limits[..., 1]).clamp_min(0.)).sum(-1),
        knee_limit_margin=(penetration/.05).square().mean(-1),
        ang_vel_xy=angular[..., :2].square().sum(-1),
        dof_vel=leg_dq.square().sum(-1),
        torques=torque.square().sum(-1))
    values = {name:(value if name == 'tracking_lin_vel_tight' else
        value.clamp(-WEIGHTED_LIMITS.get(name, 1.)/abs(WEIGHTS[name]),
                    WEIGHTED_LIMITS.get(name, 1.)/abs(WEIGHTS[name])))
        .masked_fill(masked, 0.) for name,value in values.items()}
    total = torch.zeros_like(values['tracking_lin_vel'])
    for name,weight in WEIGHTS.items():
        # Match scalar-dt multiplication even for source's float32 support term.
        weighted = values[name]*weight*context[..., 20].to(values[name].dtype)
        total = total+torch.nan_to_num(weighted, nan=0., posinf=0., neginf=0.)
    return values, total


@torch.no_grad()
def capture(env, reward_state):
    """Snapshot reward-time inputs using the original reward's exact physical reads.

    reward_state is the original WalkRewardState object (it provides joint,
    body and motor-velocity indices and the precomputed limits). This function
    never calls compute(), sim.forward() or any stateful command update. The
    returned detached snapshots survive action-history shifts and auto-resets.
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse
    from ct_mjlab.envs.chuanliantui.standup_mdp import base_contact_force

    state = env.command_manager.get_term('standup')
    data = reward_state.robot.data
    relative = data.body_link_pos_w[:, reward_state.wheels]-data.body_link_pos_w[:, reward_state.hips]
    quat = data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
    relative = quat_apply_inverse(quat.reshape(-1, 4), relative.reshape(-1, 3)).reshape(-1, 2, 3)
    swing = torch.atan2(-relative[..., 0], -relative[..., 2])
    support = -env.scene['wheel_contact'].data.force[:, :, 2] > 1.
    features = torch.cat((data.root_link_lin_vel_b, data.root_link_ang_vel_b,
        data.projected_gravity_b, state.height_above_ground[:, None], swing,
        data.joint_pos[:, reward_state.legs], data.joint_pos[:, reward_state.knees],
        data.joint_vel[:, reward_state.legs], env.sim.data.qfrc_actuator[:, reward_state.motor_vadr],
        base_contact_force(env)[:, None], support.to(data.root_link_lin_vel_b.dtype)), -1)
    batch = features.shape[0]
    soft_limits = reward_state.soft_limits.expand(batch, -1, -1).reshape(batch, 8)
    knee_limits = reward_state.knee_limits.expand(batch, -1, -1).reshape(batch, 4)
    flags = torch.stack((state.tracking_active, state.position_locked,
        state.numerical_failure, state.reward_active), -1).to(features.dtype)
    constants = torch.empty((batch, 2), device=features.device, dtype=features.dtype)
    constants[:, 0] = state.cfg.tracking_tight_kernel_denominator
    constants[:, 1] = env.step_dt
    context = torch.cat((state.command[:, :3], flags, soft_limits, knee_limits, constants), -1)
    manager = env.action_manager
    return dict(features=features.detach().clone(), context=context.detach().clone(),
        action=manager.action.detach().clone(), previous=manager.prev_action.detach().clone(),
        previous_previous=manager.prev_prev_action.detach().clone())
