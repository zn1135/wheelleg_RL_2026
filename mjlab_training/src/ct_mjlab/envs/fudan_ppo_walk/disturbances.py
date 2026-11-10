"""Optional simulation delays and reference-shaped force pulses for this task only."""
from dataclasses import asdict, dataclass
import math
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply
from ct_mjlab.fudan_ppo_walk.contract import PHYSICS_DT
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
VERSION = 'fudan-optional-disturbances-v1'
SENSOR_SLOTS = (*range(6), *range(9, 19))

@dataclass(frozen=True)
class DisturbanceConfig:
    action_delay: bool = False
    sensor_delay: bool = False
    pushes: bool = False
    action_delay_ms: tuple[float, float] = (0.0, 10.0)
    sensor_delay_ms: tuple[float, float] | None = None
    push_interval_s: float = 7.0
    max_push_vel: float = 2.0

    def __post_init__(self):
        for name in ('action_delay', 'sensor_delay', 'pushes'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be boolean')
        for name in ('action_delay_ms', 'sensor_delay_ms'):
            value = getattr(self, name)
            if value is None:
                if name == 'action_delay_ms' or self.sensor_delay:
                    raise ValueError(f'{name} must be specified; sensor delay has no measured default')
                continue
            if len(value) != 2 or not all((math.isfinite(x) for x in value)) or (not 0 <= value[0] <= value[1]):
                raise ValueError(f'{name} requires finite nonnegative MIN MAX')
            object.__setattr__(self, name, tuple(value))
        if not math.isfinite(self.push_interval_s) or self.push_interval_s < PHYSICS_DT or (not math.isfinite(self.max_push_vel)) or (self.max_push_vel <= 0):
            raise ValueError('Push interval must be >=2ms and max_push_vel must be positive')

    def record(self):
        result = asdict(self)
        for name in ('action_delay_ms', 'sensor_delay_ms'):
            if result[name] is not None:
                result[name] = list(result[name])
        return dict(version=VERSION, **result)

def from_record(record=None):
    if record is None:
        return DisturbanceConfig()
    record = dict(record)
    if record.pop('version', None) != VERSION:
        raise ValueError('Unsupported disturbance contract')
    return DisturbanceConfig(**record)

def from_task_config(task_cfg=None):
    cfg = FudanWalkCfg.domain_rand if task_cfg is None else task_cfg.domain_rand
    return DisturbanceConfig(
        action_delay=cfg.randomize_action_delay,
        sensor_delay=cfg.randomize_sensor_delay,
        pushes=cfg.push_robots,
        action_delay_ms=cfg.delay_ms_range,
        sensor_delay_ms=cfg.sensor_delay_ms_range,
        push_interval_s=cfg.push_interval_s,
        max_push_vel=cfg.max_push_vel_xy
    )

def from_sim2sim_config(task_cfg=None):
    task = FudanWalkCfg() if task_cfg is None else task_cfg
    cfg, domain = task.sim2sim, task.domain_rand
    return DisturbanceConfig(
        action_delay=cfg.randomize_action_delay,
        sensor_delay=cfg.randomize_sensor_delay,
        pushes=domain.push_robots,
        action_delay_ms=cfg.delay_ms_range,
        sensor_delay_ms=cfg.sensor_delay_ms_range,
        push_interval_s=domain.push_interval_s,
        max_push_vel=domain.max_push_vel_xy
    )

class DisturbanceState:

    def __init__(self, cfg, n, device, seed=1, physics_dt=None):
        self.cfg = cfg
        self.physics_dt = FudanWalkCfg.sim.dt if physics_dt is None else physics_dt
        self.ids = torch.arange(n, device=device)

        def delays(enabled, bounds):
            if not enabled:
                return (torch.zeros(n, dtype=torch.long, device=device), 0)
            low, high = bounds
            steps = torch.round(torch.empty(n, device=device).uniform_(low, high) / (self.physics_dt * 1000)).long()
            return (steps, math.ceil(high / (self.physics_dt * 1000)))
        self.action_steps, action_max = delays(cfg.action_delay, cfg.action_delay_ms)
        self.sensor_steps, sensor_max = delays(cfg.sensor_delay, cfg.sensor_delay_ms)
        self.actions = torch.zeros(action_max + 1, n, 6, device=device)
        self.sensors = torch.zeros(sensor_max + 1, n, 16, device=device)
        self.action_head = 0
        self.sensor_head = 0
        self.ticks = torch.zeros(n, dtype=torch.long, device=device)
        self.push_count = torch.zeros_like(self.ticks)
        self.push_seeds = np.arange(n, dtype=np.int64) + int(seed)
        self.push_interval = math.ceil(cfg.push_interval_s / self.physics_dt - 1e-09)

    def reset(self, ids, clean_sensors):
        self.actions[:, ids] = 0.0
        self.sensors[:, ids] = clean_sensors[ids]
        self.ticks[ids] = 0
        self.push_count[ids] = 0
        if self.cfg.pushes:
            indices = self.ids[ids].cpu().numpy()
            self.push_seeds[indices] += len(self.ids)

    def action(self, value):
        if not self.cfg.action_delay:
            return value
        self.action_head = (self.action_head + 1) % len(self.actions)
        self.actions[self.action_head].copy_(value)
        return self.actions[(self.action_head - self.action_steps) % len(self.actions), self.ids]

    def sample(self, value):
        if not self.cfg.sensor_delay:
            return
        self.sensor_head = (self.sensor_head + 1) % len(self.sensors)
        self.sensors[self.sensor_head].copy_(value)

    def observed(self):
        return self.sensors[(self.sensor_head - self.sensor_steps) % len(self.sensors), self.ids]

    def force(self, quat, scale):
        self.ticks += 1
        result = torch.zeros(len(self.ids), 3, device=self.ids.device)
        if not self.cfg.pushes:
            return result
        ids = (self.ticks.remainder(self.push_interval) == 0).nonzero().flatten()
        indices = ids.cpu().tolist()
        counts = self.push_count[ids].cpu().tolist()
        if indices:
            local = np.array([np.random.default_rng([int(self.push_seeds[i]), count]).uniform(-1.0, 1.0, 3) for i, count in zip(indices, counts)])
            local = torch.as_tensor(local, dtype=quat.dtype, device=quat.device) * scale[ids, None]
            result[ids] = quat_apply(quat[ids], local)
            result[ids, 2] *= 0.5
            self.push_count[ids] += 1
        return result
