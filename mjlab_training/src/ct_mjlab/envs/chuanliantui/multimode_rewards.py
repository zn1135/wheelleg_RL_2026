"""Additional active-mode rewards; the 27 flat terms remain v12-compatible."""

import torch


WEIGHTS = {
    "zero_command_hold": 3.0,
    "jump_push": 1.0,
    "jump_target": 2.0,
    "airborne_balance": -0.1,
    "step_height": 1.0,
    "stair_height": 1.0,
}


def zero_command_hold(*, command, linear_velocity, position_error, anchor_valid,
                      recover, airborne, jump, gravity, stable):
    # 速度低还不够：站稳后的位置偏离2cm时，保持奖励同样明显降低。
    active = ((command[:, :2].abs() < .05).all(-1) & ~recover & ~airborne & ~jump
              & (gravity[:, 2] < -.9) & stable & anchor_valid)
    velocity_score = torch.exp(-linear_velocity[:, :2].square().sum(-1) / .005**2)
    position_score = torch.exp(-position_error.square() / .02**2)
    return active.float() * velocity_score * position_score


def reward(env, *, name):
    state = env.command_manager.get_term("standup")
    robot = env.scene["robot"].data
    if name == "zero_command_hold":
        position_error = (robot.root_link_pos_w[:, :2] - state.zero_hold_anchor_xy).norm(dim=-1)
        return zero_command_hold(command=state.command,
            linear_velocity=robot.root_link_lin_vel_b, position_error=position_error,
            anchor_valid=state.zero_hold_anchor_valid, recover=state.recover,
            airborne=state.airborne, jump=state.jump_phase > 0,
            gravity=robot.projected_gravity_b, stable=state.stable_time >= .5)
    if name == "jump_push":
        return (state.jump_phase == 1).float() * robot.root_link_lin_vel_w[:, 2].clamp(0., 3.)
    if name == "jump_target":
        error = (state.height_above_ground - state.jump_target).clamp(-1., 1.)
        return (state.jump_phase == 2).float() * torch.exp(-error.square() / .01)
    if name == "airborne_balance":
        return state.airborne.float() * robot.root_link_ang_vel_b[:, :2].square().sum(-1)
    if name in ("step_height", "stair_height"):
        active = state.step_active if name == "step_height" else state.stair_active
        error = state.height_above_ground - state.command[:, 2]
        return active.float() * torch.exp(-error.square() / .0025)
    raise KeyError(name)
