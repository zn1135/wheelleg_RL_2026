"""Effective Huananhu V14 domain randomization, adapted to MuJoCo.

Mass/inertia and joint coordinates are based on the retained physical robot.
MuJoCo has one sliding coefficient and one dry joint friction parameter: dynamic
contact friction and sampled static joint friction are the respective mappings.
PhysX's separate static/dynamic transitions cannot be reproduced by these fields.
Source floor restitution is zero with multiplication, so contacts remain damped.
Robot contact geoms must have priority=1 (floor priority=0), to retain low friction.
"""

from __future__ import annotations

import torch

from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields

from .model import ACTUATED_JOINTS, GAS_SPRING_JOINTS


class HuananhuRandomization:
    def __init__(self, env):
        self.env = env
        robot, model = env.scene["robot"], env.sim.mj_model
        self.body_ids = robot.indexing.body_ids
        self.base_local_id = robot.body_names.index("base_link")
        self.base_id = int(self.body_ids[self.base_local_id])
        self.local_wrench = torch.zeros(env.num_envs, 6, device=env.device)
        self.mass_scales = torch.empty(env.num_envs, len(self.body_ids), device=env.device).uniform_(.9, 1.1)
        self.mass_scales[:, self.base_local_id].uniform_(.9, 1.3)
        self.com_offset = (2 * torch.rand(env.num_envs, 3, device=env.device) - 1) * torch.tensor(
            [.04, .02, .02], device=env.device)
        collision = lambda i: bool(model.geom_contype[i] or model.geom_conaffinity[i])
        self.base_geom_ids = torch.tensor([
            i for i in range(model.ngeom) if model.geom_bodyid[i] == self.base_id and collision(i)
        ], dtype=torch.long, device=env.device)
        wheel_bodies = {int(self.body_ids[robot.body_names.index(name)]) for name in ("lfwheel", "rfwheel")}
        self.wheel_geom_ids = torch.tensor([
            i for i in range(model.ngeom) if model.geom_bodyid[i] in wheel_bodies and collision(i)
        ], dtype=torch.long, device=env.device)
        self.materials: dict[str, torch.Tensor] = {}

    def apply_local_wrench(self) -> None:
        """Rotate a persistent COM wrench from body to world each physical tick."""
        quat = self.env.sim.data.xquat[:, self.base_id]
        vector = self.local_wrench.reshape(-1, 2, 3)
        xyz = quat[:, None, 1:].expand(-1, 2, -1)
        rotated = vector + 2 * torch.linalg.cross(
            xyz, torch.linalg.cross(xyz, vector) + quat[:, None, :1] * vector)
        self.env.sim.data.xfrc_applied[:, self.base_id] = rotated.flatten(1)


def get_state(env) -> HuananhuRandomization:
    return env._huananhu_randomization


def _material(env, state, name, geom_ids, static_range, dynamic_range):
    # Like the source, material properties use a fixed set of 64 buckets.
    buckets = torch.empty(64, 3, device=env.device)
    buckets[:, 0].uniform_(*static_range)
    buckets[:, 1].uniform_(*dynamic_range)
    buckets[:, 1] = torch.minimum(buckets[:, 0], buckets[:, 1])
    buckets[:, 2].uniform_(.02, .2)
    values = buckets[torch.randint(64, (env.num_envs, len(geom_ids)), device=env.device)]
    state.materials[name] = values
    env.sim.model.geom_friction[:, geom_ids, 0] = values[..., 1]
    # The source combines restitution by multiplication with a zero-bounce floor.
    env.sim.model.geom_solref[:, geom_ids, 0] = 2 * env.physics_dt
    env.sim.model.geom_solref[:, geom_ids, 1] = 1.


@requires_model_fields(
    "body_mass", "body_inertia", "body_ipos", "geom_friction", "geom_solref",
    "dof_frictionloss", "dof_damping", recompute=RecomputeLevel.set_const)
def randomize_at_startup(env, env_ids):
    state = HuananhuRandomization(env)
    env._huananhu_randomization = state
    ids = state.body_ids
    env.sim.model.body_mass[:, ids] = env.sim.get_default_field("body_mass")[ids] * state.mass_scales
    # Scaling mass recomputes inertia at fixed density shape in the source. Its
    # additional independent base/wheel inertia events are explicitly disabled.
    env.sim.model.body_inertia[:, ids] = (
        env.sim.get_default_field("body_inertia")[ids] * state.mass_scales[..., None])
    env.sim.model.body_ipos[:, state.base_id] = (
        env.sim.get_default_field("body_ipos")[state.base_id] + state.com_offset)
    _material(env, state, "base", state.base_geom_ids, (.01, .1), (.01, .1))
    _material(env, state, "wheel", state.wheel_geom_ids, (.5, 1.2), (.4, 1.))

    robot = env.scene["robot"]
    for index, name in enumerate(robot.joint_names):
        if name in GAS_SPRING_JOINTS:
            # The spring slider damping is applied once by HuananhuPdAction.
            continue
        if name in ("lfwheel", "rfwheel"):
            dry, viscous = (.05, .25), (.0, .01)
        elif name in ACTUATED_JOINTS:
            dry, viscous = (.25, 1.), (.05, .2)
        else:
            dry, viscous = (.05, .1), (.01, .025)
        dof = int(robot.indexing.joint_v_adr[index])
        env.sim.model.dof_frictionloss[:, dof] = (
            env.sim.get_default_field("dof_frictionloss")[dof]
            + torch.empty(env.num_envs, device=env.device).uniform_(*dry))
        env.sim.model.dof_damping[:, dof] = (
            env.sim.get_default_field("dof_damping")[dof]
            + torch.empty(env.num_envs, device=env.device).uniform_(*viscous))


def reset_disturbances(env, env_ids):
    state = get_state(env)
    ids = slice(None) if env_ids is None else env_ids
    state.local_wrench[ids] = 0.
    env.sim.data.xfrc_applied[ids, state.base_id] = 0.


def push_root_velocity(env, env_ids):
    robot = env.scene["robot"]
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    velocity = robot.data.root_link_vel_w[ids].clone()
    velocity[:, :2].uniform_(-.25, .25)
    robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)


def sample_local_wrench(env, env_ids):
    state = get_state(env)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    wrench = torch.empty(len(ids), 6, device=env.device)
    wrench[:, :3].uniform_(-10., 10.)
    wrench[:, 3:].uniform_(-1., 1.)
    state.local_wrench[ids] = wrench
    state.apply_local_wrench()


def make_huananhu_events(play: bool = False) -> dict[str, EventTermCfg]:
    """Startup physics, per-reset clearing, and independent 5–10 s disturbances.

    Root/joint pose resets belong to the task command module. Action parameter
    randomization is performed by HuananhuPdAction.reset with its 720-step gate.
    """
    if play:
        return {}
    return {
        "huananhu_domain_randomization": EventTermCfg(func=randomize_at_startup, mode="startup"),
        "huananhu_reset_disturbances": EventTermCfg(func=reset_disturbances, mode="reset"),
        "huananhu_velocity_push": EventTermCfg(func=push_root_velocity, mode="interval",
                                               interval_range_s=(5., 10.)),
        "huananhu_local_wrench": EventTermCfg(func=sample_local_wrench, mode="interval",
                                              interval_range_s=(5., 10.)),
    }
