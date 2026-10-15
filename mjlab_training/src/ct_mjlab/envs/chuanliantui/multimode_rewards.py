"""Additional active-mode rewards; the 27 flat terms remain v12-compatible."""

import torch


WEIGHTS = {
    "jump_push": 1.0,
    "jump_target": 2.0,
    "airborne_balance": -0.1,
    "step_height": 1.0,
    "stair_height": 1.0,
}


def reward(env, *, name):
    state = env.command_manager.get_term("standup")
    robot = env.scene["robot"].data
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
