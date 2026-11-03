"""Task-specific parking command sampling, separate from policy/physical control."""
import math
import torch
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from .fudan_ppo_walk_config import policy_dt
LEGACY_COMMAND_SAMPLING = dict(version='fudan-uniform-commands-v1', standstill_env_ratio=0.0, stop_env_ratio=0.0)

def command_config(task_cfg=None):
    task = FudanWalkCfg() if task_cfg is None else task_cfg
    cfg = task.commands
    a, b = (cfg.standstill_env_ratio, cfg.stop_env_ratio)
    if not all((math.isfinite(x) and 0 <= x <= 1 for x in (a, b))) or a + b > 1:
        raise ValueError('Parking episode ratios must be in[0,1] and sum to at most1')
    for name in ('stop_drive_time_s', 'stop_speed_range'):
        value = getattr(cfg, name)
        if len(value) != 2 or not all((math.isfinite(x) for x in value)) or (not 0 < value[0] <= value[1]):
            raise ValueError(f'Invalid {name}')
    if cfg.stop_drive_time_s[1] >= task.env.episode_length_s:
        raise ValueError('Driving interval must leave time to park in a20s episode')
    if cfg.stop_speed_range[1] > task.commands.velocity_initial:
        raise ValueError('Stopping speed must fit the initial2m/s curriculum')
    if not cfg.parking_heights or not all((math.isfinite(x) and task.commands.height_range[0] <= x <= task.commands.height_range[1] for x in cfg.parking_heights)):
        raise ValueError('Parking heights must stay in the existing0.22-0.30m range')
    return dict(
        version='fudan-parking-commands-v1',
        standstill_env_ratio=a,
        stop_env_ratio=b,
        stop_drive_time_s=list(cfg.stop_drive_time_s),
        stop_speed_range=list(cfg.stop_speed_range),
        parking_heights=list(cfg.parking_heights)
    )

def initialize(env, cfg=None):
    env.command_sampling = command_config(getattr(env, 'cfg', None)) if cfg is None else cfg
    env.command_mode = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env.command_elapsed = torch.zeros_like(env.command_mode)
    env.stop_command_step = torch.zeros_like(env.command_mode)

def reset_modes(env, ids):
    env.command_mode[ids] = 0
    env.command_elapsed[ids] = 0
    env.stop_command_step[ids] = 0
    cfg = env.command_sampling
    a, b = (cfg['standstill_env_ratio'], cfg['stop_env_ratio'])
    if a + b == 0:
        return
    draw = torch.rand(len(ids), device=env.device)
    env.command_mode[ids[draw < a]] = 1
    stop = ids[(draw >= a) & (draw < a + b)]
    env.command_mode[stop] = 2
    if len(stop):
        seconds = torch.empty(len(stop), device=env.device).uniform_(*cfg['stop_drive_time_s'])
        env.stop_command_step[stop] = torch.ceil(seconds / policy_dt(getattr(env, 'cfg', None))).long()

def apply_parking_commands(env, ids):
    parking = ids[env.command_mode[ids] != 0]
    if len(parking):
        heights = env.commands.new_tensor(env.command_sampling['parking_heights'])
        env.commands[parking, 2] = heights[torch.randint(len(heights), (len(parking),), device=env.device)]
    hold = ids[env.command_mode[ids] == 1]
    env.commands[hold, :2] = 0.0
    stop = ids[env.command_mode[ids] == 2]
    if len(stop):
        speed = torch.empty(len(stop), device=env.device).uniform_(*env.command_sampling['stop_speed_range'])
        direction = torch.where(torch.rand(len(stop), device=env.device) < 0.5, -1.0, 1.0)
        env.commands[stop, 0] = speed * direction
        env.commands[stop, 1] = 0.0

def update_stops(env):
    stop = (env.command_mode == 2) & (env.command_elapsed >= env.stop_command_step)
    env.commands[stop, :2] = 0.0
