"""Dense zero-speed and physical leg-quiet rewards, with no control filtering."""

from functools import lru_cache

import mujoco
import numpy as np
from scipy.optimize import least_squares
import torch

from ..chuanliantui import huananhu_rewards
from ..chuanliantui.huananhu_poses import _mesh_vertices
from ..chuanliantui.model import load_model, solve_standing_joint_positions, closed_chain_residual
from .mdp import HEIGHT_COMMAND_M, LEG_MOTORS, standing_mask, supported_pose_mask, standing_pose


RECIPE_VERSION = "stand-r27-wheel-placeholders-39"
HOLD_SPEED_SCALE_M_S = .01
WHEEL_SPEED_SCALE_RAD_S = .05
BODY_STILL_SPEED_SCALE_M_S = .005
LEG_SPEED_LIMIT_RAD_S = .5
BODY_SPEED_MARGIN_M_S = .03
POSTURE_SCALE_RAD = .05
RECOVERY_HEIGHT_FLOOR_M = .16


@lru_cache(maxsize=1)
def balanced_standing_motor_targets():
    """Reward-only pose: .27 m and nominal COM over the wheel axle, with closed links."""
    model = load_model()
    data = mujoco.MjData(model)
    base = model.body("base_link").id
    wheels = [model.body(name).id for name in ("lfwheel", "rfwheel")]
    geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] in wheels
             and (model.geom_contype[i] or model.geom_conaffinity[i])]
    initial = np.asarray(standing_pose()["front_joint_targets"]["left"])

    def residual(values):
        positions = solve_standing_joint_positions(model, front_joint_targets={
            "left": tuple(values), "right": tuple(-values)})
        data.qpos[:] = model.qpos0
        data.qpos[:7] = (0., 0., 0., 1., 0., 0., 0.)
        for name, value in positions.items():
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        height = -min(_mesh_vertices(model, data, i)[:, 2].min() for i in geoms)
        offset = data.subtree_com[base, 0] - data.xpos[wheels, 0].mean()
        return np.asarray((height - .27, offset))

    lower, upper = model.joint("lf1").range
    fit = least_squares(residual, initial,
        bounds=([initial[0] - .5, lower + 1e-4], [initial[0] + .5, upper - 1e-4]),
        diff_step=1e-4, ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=50)
    if not fit.success or np.abs(residual(fit.x)).max() > 1e-7 or closed_chain_residual(model, data) > 1e-8:
        raise ValueError("Could not solve a legal balanced standing reward target")
    for index in range(1, model.njnt):
        if model.jnt_limited[index]:
            value = data.qpos[model.jnt_qposadr[index]]
            if not model.jnt_range[index, 0] <= value <= model.jnt_range[index, 1]:
                raise ValueError("Standing reward target exceeds a joint limit")
    return tuple(float(data.qpos[model.joint(name).qposadr[0]]) for name in LEG_MOTORS)


def sample_leg_velocity(env):
    """Accumulate all five physical samples before the 100 Hz reward callback."""
    state = env.command_manager.get_term("standup")
    if not hasattr(state, "leg_v_adr"):
        robot = env.scene["robot"]
        ids, _ = robot.find_joints(LEG_MOTORS, preserve_order=True)
        state.leg_v_adr = robot.indexing.joint_v_adr[ids]
    if not hasattr(state, "wheel_v_adr"):
        robot = env.scene["robot"]
        ids, _ = robot.find_joints(("lfwheel", "rfwheel"), preserve_order=True)
        state.wheel_v_adr = robot.indexing.joint_v_adr[ids]
        state.wheel_velocity_sum = torch.zeros_like(state.leg_velocity_sum)
    if not hasattr(state, "body_xy_v_adr"):
        state.body_xy_v_adr = env.scene["robot"].indexing.free_joint_v_adr[:2]
        state.body_speed_peak = torch.zeros_like(state.leg_velocity_sum)
    if state.leg_velocity_step != env.common_step_counter:
        state.leg_velocity_sum.zero_()
        state.leg_absolute_velocity_sum.zero_()
        state.leg_speed_peak.zero_()
        state.wheel_velocity_sum.zero_()
        state.body_speed_peak.zero_()
        state.leg_velocity_count = 0
        state.leg_velocity_step = env.common_step_counter
    value = env.sim.data.qvel[:, state.leg_v_adr].square().mean(-1)
    state.leg_velocity_sum += value
    state.leg_absolute_velocity_sum += env.sim.data.qvel[:, state.leg_v_adr].abs().mean(-1)
    state.leg_speed_peak = torch.maximum(state.leg_speed_peak,
        env.sim.data.qvel[:, state.leg_v_adr].abs().amax(-1))
    state.wheel_velocity_sum += env.sim.data.qvel[:, state.wheel_v_adr].square().mean(-1)
    state.body_speed_peak = torch.maximum(state.body_speed_peak,
        env.sim.data.qvel[:, state.body_xy_v_adr].norm(dim=-1))
    state.leg_velocity_count += 1
    return value


def low_velocity_hold(env):
    state = env.command_manager.get_term("standup")
    speed_squared = env.scene["robot"].data.root_link_lin_vel_b[:, :2].square().sum(-1)
    active = ((state.stable_time >= .5) & supported_pose_mask(env)
              & ~state.recover & ~state.airborne & ~state.settle_required)
    return active.float() / (1. + speed_squared / HOLD_SPEED_SCALE_M_S**2)


def body_stillness(env):
    speed_squared = env.scene["robot"].data.root_link_lin_vel_b[:, :2].square().sum(-1)
    # Quiet joints alone must not earn full hold bonuses while the body rolls away.
    return 1. / (1. + speed_squared / BODY_STILL_SPEED_SCALE_M_S**2)


def standing_posture(env):
    state = env.command_manager.get_term("standup")
    robot = env.scene["robot"]
    if not hasattr(state, "posture_q_adr"):
        ids, _ = robot.find_joints(LEG_MOTORS, preserve_order=True)
        state.posture_q_adr = robot.indexing.joint_q_adr[ids]
        state.posture_target = torch.tensor(balanced_standing_motor_targets(), device=env.device,
                                           dtype=env.sim.data.qpos.dtype)
    error = (env.sim.data.qpos[:, state.posture_q_adr] - state.posture_target).square().mean(-1)
    # This is a soft reward on real coordinates, never a PD target or a joint lock.
    active = ((state.stable_time >= .5) & supported_pose_mask(env)
              & ~state.recover & ~state.airborne & ~state.fallen & ~state.settle_required
              & (~state.push_applied | state.quiet_ready) & ~state.numerical_failure)
    return torch.where(active, torch.exp(-error / POSTURE_SCALE_RAD**2), 0.)


def quiet_legs(env):
    state = env.command_manager.get_term("standup")
    mean_squared_velocity = state.leg_velocity_sum / max(state.leg_velocity_count, 1)
    # Pure hold episodes need current low body speed, not 50 consecutive low-speed ticks.
    # An actual push still requires continuous restabilization before this reward returns.
    active = ((state.stable_time >= .5) & standing_mask(env)
              & ~state.recover & ~state.airborne & ~state.fallen & ~state.settle_required
              & (~state.push_applied | state.quiet_ready))
    return active.float() / (1. + mean_squared_velocity / .2**2)


def quiet_wheels(env):
    state = env.command_manager.get_term("standup")
    speed_squared = state.wheel_velocity_sum / max(state.leg_velocity_count, 1)
    # Restore r07's bounded rest bonus; do not penalize wheel motion needed for balance.
    active = ((state.stable_time >= .5) & supported_pose_mask(env)
              & ~state.recover & ~state.airborne & ~state.fallen & ~state.settle_required
              & (~state.push_applied | state.quiet_ready))
    return active.float() / (1. + speed_squared / WHEEL_SPEED_SCALE_RAD_S**2)


def quiet_leg_speed_cost(env):
    state = env.command_manager.get_term("standup")
    excess = (state.leg_speed_peak - LEG_SPEED_LIMIT_RAD_S).clamp_min(0.)
    active = ((state.stable_time >= .5) & supported_pose_mask(env)
              & ~state.recover & ~state.airborne & ~state.fallen & ~state.settle_required
              & (~state.push_applied | state.quiet_ready) & ~state.numerical_failure)
    # Penalize the worst true substep overspeed; allow balancing below the new limit.
    return torch.where(active, excess, 0.)


def body_speed_margin_cost(env):
    state = env.command_manager.get_term("standup")
    excess = (state.body_speed_peak - BODY_SPEED_MARGIN_M_S).clamp(0., .1)
    # Leave margin below the .05 m/s acceptance bound; use every true physics sample.
    active = ((state.stable_time >= .5) & supported_pose_mask(env)
              & ~state.recover & ~state.airborne & ~state.fallen & ~state.settle_required
              & (~state.push_applied | state.quiet_ready) & ~state.numerical_failure)
    # A partial reset clears stable_time; its old peak cannot affect the next reward.
    return torch.where(active, excess, 0.)


def hold_position(env):
    state = env.command_manager.get_term("standup")
    position = env.scene["robot"].data.root_link_pos_w[:, :2]
    state.hold_anchor_valid &= ~(state.recover | state.fallen)
    anchor = (state.quiet_ready & ~state.hold_anchor_valid & ~state.settle_required
              & ~state.recover & ~state.fallen & ~state.airborne)
    # Lock once after body settling; only a reset, actual push or fall unlocks it.
    state.hold_anchor_xy[anchor] = position[anchor]
    state.hold_anchor_valid[anchor] = True
    distance = (position - state.hold_anchor_xy).norm(dim=-1).clamp(max=.1)
    active = (state.hold_anchor_valid & supported_pose_mask(env) & ~state.settle_required
              & ~state.recover & ~state.fallen & ~state.airborne)
    return torch.where(active, distance, 0.)


def body_speed_cost(env):
    state = env.command_manager.get_term("standup")
    speed = env.scene["robot"].data.root_link_lin_vel_w[:, :2].norm(dim=-1).clamp(max=.1)
    active = ((state.stable_time >= .5) & supported_pose_mask(env)
              & ~state.recover & ~state.airborne & ~state.fallen & ~state.settle_required
              & (~state.push_applied | state.quiet_ready))
    # Current velocity is observable to the critic; a hidden position anchor is not.
    return torch.where(active, speed, 0.)


def orientation_outside_recovery(env, *, name):
    state = env.command_manager.get_term("standup")
    recovering = state.settle_required | state.recover | state.fallen
    value = huananhu_rewards.reward(env, name=name)
    return torch.where(recovering, 0., value)


def recovery_upright(env):
    state = env.command_manager.get_term("standup")
    active = (state.settle_required | state.recover | state.fallen) & ~state.numerical_failure
    data = env.scene["robot"].data
    height = data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    progress = ((height - RECOVERY_HEIGHT_FLOOR_M)
                / (HEIGHT_COMMAND_M - RECOVERY_HEIGHT_FLOOR_M)).clamp(0., 1.)
    # Lying low must not earn the full upright bonus; lifting still gives a dense signal.
    value = -data.projected_gravity_b[:, 2] - (1. - progress)
    return torch.where(active, value, 0.)


def recovery_wheel_height(env):
    state = env.command_manager.get_term("standup")
    robot = env.scene["robot"]
    height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    progress = ((height - RECOVERY_HEIGHT_FLOOR_M)
                / (HEIGHT_COMMAND_M - RECOVERY_HEIGHT_FLOOR_M)).clamp(0., 1.)
    weight = (env.sim.model.body_mass[:, robot.indexing.body_ids].sum(-1)
              * float(np.linalg.norm(env.sim.mj_model.opt.gravity)))
    support = -env.scene["wheel_contact"].data.force[:, :, 2]
    # Cap each wheel separately: a single loaded wheel cannot earn full support.
    load = (support / (.5 * weight[:, None])).clamp(0., 1.).mean(-1)
    upright = (-robot.data.projected_gravity_b[:, 2]).clamp(0., 1.)
    value = upright * progress * (.25 + .75 * load)
    active = (state.settle_required | state.recover | state.fallen) & ~state.numerical_failure
    return torch.where(active, value, 0.)


def hold_cost_outside_recovery(env, *, name):
    state = env.command_manager.get_term("standup")
    recovering = state.settle_required | state.recover | state.fallen
    value = huananhu_rewards.reward(env, name=name)
    return torch.where(recovering, 0., value)
