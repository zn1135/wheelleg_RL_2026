"""External handover with hysteresis; no policy receives the selected role."""

import math
import torch
from ..chuanliantui_stand_v2.mdp import wheel_load

SWITCH_CONTRACT = 'chuanliantui-two-policy-support-height22-tilt10-hold500ms-fall100ms-v1'


def physical_conditions(env):
    robot = env.scene['robot']
    q = env.sim.data.qpos[:, robot.indexing.free_joint_q_adr]
    v = env.sim.data.qvel[:, robot.indexing.free_joint_v_adr]
    w, x, y, z = q[:, 3:].unbind(-1)
    gz = (-1+2*(x*x+y*y)).clamp(-1., 1.)
    height = q[:, 2] - env.scene.env_origins[:, 2]
    support, weight = wheel_load(env)
    base = env.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
    finite = (torch.isfinite(env.sim.data.qpos).all(-1)
              & torch.isfinite(env.sim.data.qvel).all(-1) & torch.isfinite(base)
              & torch.isfinite(support).all(-1))
    return conditions(gz, height, v[:,3:5].norm(dim=-1), base, support, weight, finite)


def conditions(gz, height, omega_xy, base, support, weight, finite):
    loaded = ((support >= (.1*weight).clamp_min(5.)[:, None]).all(-1)
              & (support.sum(-1) >= .8*weight))
    ready = (finite & loaded & (support.sum(-1) <= 1.2*weight) & (base <= 1.) & ((height-.22).abs() <= .03)
             & (gz <= -math.cos(math.radians(10))) & (omega_xy <= .5))
    down = (~finite | ~loaded | (base > 10.) | (height <= .18)
            | (gz > -math.cos(math.radians(40))))
    emergency = ~finite | (gz > 0.)
    return ready, down, emergency, finite


class PolicySwitch:
    def __init__(self, num_envs, device, dt=.01):
        self.walk = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ready_ticks = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.down_ticks = torch.zeros_like(self.ready_ticks)
        self.enter_ticks, self.exit_ticks = math.ceil(.5/dt), math.ceil(.1/dt)

    def reset(self, ids):
        self.walk[ids] = False
        self.ready_ticks[ids] = self.down_ticks[ids] = 0

    def update(self, ready, down, emergency, advance=None):
        before = self.walk.clone()
        if advance is None:
            advance = torch.ones_like(self.walk)
        self.ready_ticks = torch.where(advance, torch.where(ready & ~self.walk, self.ready_ticks+1, 0), self.ready_ticks)
        self.down_ticks = torch.where(advance, torch.where(down & self.walk, self.down_ticks+1, 0), self.down_ticks)
        next_walk = (self.walk | (self.ready_ticks >= self.enter_ticks)) & ~((self.down_ticks >= self.exit_ticks) | emergency)
        self.walk = torch.where(advance, next_walk, self.walk)
        changed = before != self.walk
        self.ready_ticks[changed] = self.down_ticks[changed] = 0
        return changed
