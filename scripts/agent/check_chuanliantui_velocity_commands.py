#!/usr/bin/env python3
"""Check r14 command sampling, heading feedback and historical replay; no learning."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from ct_mjlab.policy import checkpoint_command_config, load_actor
from ct_mjlab.reference_training.entry import manifest
from ct_mjlab.reference_training.huananhu_env_cfg import make_huananhu_env_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-checkpoint', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.manual_seed(42)
    n = 8192
    env = SimpleNamespace(num_envs=n, device='cpu', step_dt=.02, common_step_counter=0,
        episode_length_buf=torch.full((n,), 400),
        scene={'robot': SimpleNamespace(data=SimpleNamespace(heading_w=torch.zeros(n)))})
    cfg = make_huananhu_env_cfg().commands['standup']
    assert not cfg.zero_velocity_commands and not cfg.special_modes_enabled
    assert cfg.linear_velocity_range == (-2.7, 2.7)
    assert cfg.yaw_velocity_range == (-2*math.pi, 2*math.pi)
    assert cfg.dynamic_modes and cfg.heading_probability == .5
    state = cfg.build(env)
    ids = torch.arange(n)
    state.reset(ids)
    state.compute(0.)
    assert state.sampled[:,0].min() < -2.6 and state.sampled[:,0].max() > 2.6
    assert state.sampled[:,1].min() < -6.1 and state.sampled[:,1].max() > 6.1
    heading_fraction = float(state.heading.float().mean())
    assert .48 < heading_fraction < .52
    assert torch.equal(state.command[state.standing,:2], torch.zeros_like(state.command[state.standing,:2]))
    # Controlled wrap, saturation, partial updates and preservation of recovery flags.
    state.standing[:4] = False
    state.heading[:4] = True
    state.heading_target[:4] = torch.tensor([.2, 2., math.pi-.1, 0.])
    env.scene['robot'].data.heading_w[:4] = torch.tensor([0., 0., -math.pi+.1, 0.])
    before = state.command[4:].clone()
    state._update_command(torch.arange(4))
    assert torch.allclose(state.command[:4,1], torch.tensor([1., 2*math.pi, -1., 0.]), atol=2e-6)
    assert torch.equal(before, state.command[4:])
    assert state.control_mode is not None
    # Ground starts retain the source's first 1.5 s override (vx/yaw in [-1,1]).
    state.ground_start[:4] = True
    env.episode_length_buf[:4] = 0
    state._update_command(torch.arange(4))
    assert torch.equal(state.command[:4,:2], state.ground_command[:4])
    assert state.command[:4,:2].abs().max() <= 1.
    env.episode_length_buf[:] = 400
    state.ground_start[:] = False
    for iteration in (2000, 3000, 4000, 20000):
        state.training_iteration = iteration
        state._resample_command(ids)
        state._update_command(None)
        assert (state.special_mode == -1).all()
        assert state.command[:,0].abs().max() <= 2.7
        assert state.command[:,1].abs().max() <= 2*math.pi
    historical = []
    for path in args.historical_checkpoint:
        actor = load_actor(path)
        command_cfg = checkpoint_command_config(path)
        old_cfg = make_huananhu_env_cfg(command_config=command_cfg).commands['standup']
        old_state = old_cfg.build(env)
        old_state.reset(ids)
        old_state.compute(0.)
        assert old_cfg.zero_velocity_commands and (old_state.command[:,:2] == 0.).all()
        historical.append(dict(checkpoint=str(path.resolve()), zero_commands_restored=True,
                               actor_loaded=actor is not None))
    result = dict(interface=manifest()['interface'], sampling_envs=n,
        vx_range=list(cfg.linear_velocity_range), yaw_range=list(cfg.yaw_velocity_range),
        heading_fraction=heading_fraction, heading_gain=cfg.heading_control_stiffness,
        heading_wrap_saturation_partial_update=True, ground_override=True,
        dynamic_modes=True, special_modes_disabled=True, historical=historical,
        scope='command/interface checks only; no learning or policy behavior acceptance')
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
