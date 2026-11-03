"""Wheel-Legged-Gym standup domain parameters, adapted to MuJoCo contacts.

Parameters are sampled once per environment. Push schedules alone reset each
episode. See docs/ai/standup_randomization.md for engine-specific boundaries.
"""

import math

import torch

from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from .standup_mdp import STABLE_DURATION_S


def restitution_to_solref(restitution: torch.Tensor) -> torch.Tensor:
    """Damped-oscillator approximation, not a PhysX restitution parameter.

    The source ground has e=0.5 and uses average material combination. A fixed
    contact stiffness keeps the period resolved by the 2 ms physics step.
    MuJoCo's unilateral soft contact differs from the ideal oscillator; the
    bounce regression records this approximation's numerical accuracy.
    """
    effective = (restitution + .5) * .5
    log_e = effective.log()
    damping_ratio = -log_e / torch.sqrt(math.pi**2 + log_e.square())
    stiffness = torch.full_like(restitution, 10_000.)
    damping = 2 * damping_ratio * stiffness.sqrt()
    return torch.stack((-stiffness, -damping), dim=-1)


class StandupRandomization:
    def __init__(self, env):
        self.env = env
        robot = env.scene["robot"]
        model = env.sim.mj_model
        n, device = env.num_envs, env.device
        self.body_ids = robot.indexing.body_ids
        self.base_local_id = robot.body_names.index("base_link")
        self.base_id = int(self.body_ids[self.base_local_id])
        body_set = set(self.body_ids.tolist())
        self.collision_geom_ids = torch.tensor([
            i for i in range(model.ngeom)
            if int(model.geom_bodyid[i]) in body_set
            and (model.geom_contype[i] or model.geom_conaffinity[i])
        ], device=device, dtype=torch.long)
        high_band = torch.randint(0, 2, (n,), device=device)
        self.robot_friction = .5 * high_band + torch.rand(n, device=device) * .2
        self.restitution = torch.empty(n, device=device).uniform_(.6, 1.)
        self.added_base_mass = torch.empty(n, device=device).uniform_(-1., 2.)
        self.com_offset = torch.empty(n, 3, device=device).uniform_(-.02, .02)
        self.density_scales = torch.ones(n, len(self.body_ids), device=device)
        # The source collapses each fixed rear linkage into its hip body.
        # Preserve that correlation on the now-articulated physical linkage.
        groups = [("base_link",)]
        for prefix in ("lf", "rf"):
            groups.extend([(prefix+"0", prefix+"00", prefix+"01", prefix+"02", prefix+"03"),
                           (prefix+"1",), (prefix+"wheel",)])
        for group in groups:
            ids = [robot.body_names.index(name) for name in group]
            self.density_scales[:, ids] = torch.empty(n, 1, device=device).uniform_(.9, 1.1)
        # Physical spring barrel/rod masses have no source counterpart; keep them.
        self.push_force = torch.zeros(n, 3, device=device)
        self.push_steps_left = torch.zeros(n, dtype=torch.long, device=device)
        self.push_wait_steps = torch.zeros_like(self.push_steps_left)
        self.push_count = torch.zeros_like(self.push_steps_left)
        self.push_duration_steps = max(1, round(.10 / env.step_dt))
        self.push_interval_steps = (math.ceil(3. / env.step_dt), math.ceil(5. / env.step_dt))
        self.reset(torch.arange(n, device=device))

    def sample_wait(self, count):
        lo, hi = self.push_interval_steps
        return torch.randint(lo, hi + 1, (count,), device=self.env.device)

    def reset(self, env_ids):
        self.push_force[env_ids] = 0.
        self.push_steps_left[env_ids] = 0
        self.push_count[env_ids] = 0
        self.push_wait_steps[env_ids] = self.sample_wait(len(env_ids))
        self.env.sim.data.xfrc_applied[env_ids, self.base_id] = 0.

    def update_push(self):
        """100 Hz pre-physics scheduling; xfrc_applied persists across substeps."""
        command = self.env.command_manager.get_term("standup")
        idle = self.push_steps_left == 0
        self.push_force[idle] = 0.
        ready = idle & command.has_landed
        start = (ready & (self.push_wait_steps <= 0)).nonzero().flatten()
        self.push_wait_steps[ready] -= 1
        self.push_wait_steps.clamp_(min=0)
        if len(start):
            stable = command.has_stood[start] & (command.standing_time[start] >= STABLE_DURATION_S)
            lo, hi = torch.where(stable, 5., 1.), torch.where(stable, 15., 5.)
            magnitude = lo + torch.rand(len(start), device=self.env.device) * (hi - lo)
            angle = torch.rand(len(start), device=self.env.device) * (2 * math.pi)
            self.push_force[start, 0] = magnitude * angle.cos()
            self.push_force[start, 1] = magnitude * angle.sin()
            self.push_steps_left[start] = self.push_duration_steps
            self.push_wait_steps[start] = self.sample_wait(len(start))
            self.push_count[start] += 1
        self.push_steps_left.sub_(1).clamp_(min=0)
        # World-frame wrench at the body's COM; a force here adds no extra torque.
        self.env.sim.data.xfrc_applied[:, self.base_id, :3] = self.push_force
        self.env.sim.data.xfrc_applied[:, self.base_id, 3:] = 0.


def get_state(env) -> StandupRandomization:
    return env._standup_randomization


@requires_model_fields("body_mass", "body_inertia", "body_ipos", "geom_friction",
                       "geom_solref", recompute=RecomputeLevel.set_const)
def randomize_at_startup(env, env_ids):
    state = StandupRandomization(env)
    env._standup_randomization = state
    ids = state.body_ids
    mass = env.sim.get_default_field("body_mass")[ids].expand(env.num_envs, -1).clone()
    mass[:, state.base_local_id] += state.added_base_mass
    env.sim.model.body_mass[:, ids] = mass * state.density_scales
    env.sim.model.body_inertia[:, ids] = (
        env.sim.get_default_field("body_inertia")[ids] * state.density_scales[..., None]
    )
    env.sim.model.body_ipos[:, state.base_id] = (
        env.sim.get_default_field("body_ipos")[state.base_id] + state.com_offset
    )
    geoms = state.collision_geom_ids
    # Priority=1 is set by the standup EntityCfg. It prevents MuJoCo's max rule
    # from suppressing the low-friction branch against the 0.4 floor.
    env.sim.model.geom_friction[:, geoms, 0] = ((state.robot_friction + .4) / 2)[:, None]
    env.sim.model.geom_solref[:, geoms] = restitution_to_solref(state.restitution)[:, None, :]


def reset_pushes(env, env_ids):
    if hasattr(env, "_standup_randomization"):
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        get_state(env).reset(env_ids)
