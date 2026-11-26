"""Preserve the 5200 learner state and verify the new stop/start schedule."""
import argparse
import copy
import json
from pathlib import Path

import torch

from ct_mjlab.envs.chuanliantui.fudan_ppo_step.config import FudanStepCfg, recipe_record
from .launch_training import ROOT, sha, write
from .remote_training import same_state

SOURCE = ROOT/'logs/mjlab/fudan_ppo_step/20261118_fixed_speed2_L40_resume4847_to5200_01/model_final.pt'
SOURCE_SHA = '27bab820183c11556d3f5643cdab8e026e996a54c706b401282f49398d707dd2'


def prepare(output):
    assert sha(SOURCE) == SOURCE_SHA
    saved = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert saved['task'] == 'fudan_ppo_step' and not saved['partial_update']
    assert (saved['iterations'], saved['transitions'], saved['num_envs']) == (5200, 1022361600, 4096)
    assert saved['primary_updates'] == saved['encoder_updates'] == 104000
    before = saved['training_recipe']
    assert before['step']['version'] == 'fixed200-straight-2mps-v10'
    after = recipe_record(FudanStepCfg())
    expected = copy.deepcopy(before)
    for key in ('version', 'restart_flat_probability', 'delayed_stair_probability', 'restart_initial_wait_s',
                'restart_stable_s', 'restart_first_drive_s', 'restart_second_drive_s'):
        expected['step'][key] = after['step'][key]
    assert expected == after
    migration = dict(source=str(SOURCE), source_sha256=SOURCE_SHA, before_recipe=before, after_recipe=after,
        previous_migration=saved.get('recipe_migration'), learning_state_reset=False,
        terrain_assignment_preserved=True, physical_episodes_reinitialized=True)
    derived = dict(saved, training_recipe=after, recipe_migration=migration)
    output.mkdir(parents=True, exist_ok=False)
    path = output/'resume5200_restart.pt'
    torch.save(derived, path)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    assert same_state(derived, restored)
    assert all(same_state(value, restored[key]) for key, value in saved.items()
               if key not in ('training_recipe', 'recipe_migration'))
    write(output/'migration.json', dict(migration, passed=True, derived_sha256=sha(path), optimizer_updates=0))


def check_restart(env):
    import math
    ticks = math.ceil(env.cfg.step.restart_stable_s/env.dt)
    flat = env.modes == 0
    up = env.modes == 1
    down = env.modes == 2
    slots = torch.arange(env.num_envs, device=env.device)
    found = torch.zeros(4, dtype=torch.bool, device=env.device)
    for _ in range(64):
        env.resample(slots)
        waiting = env.command_mode == env.WAIT_TO_START
        assert env.commands[:, 1].eq(0).all() and env.commands[waiting, 0].eq(0).all()
        assert env.commands[down, 0].eq(2).all()
        found |= torch.stack(((waiting&flat).any(), (waiting&up).any(),
                             ((~waiting)&up).any(), (flat&(env.command_mode == 2)).any()))
    assert found.all() and env.terrain.terrain_levels.eq(1).all()
    data = env.robot.data
    actual_vel = data.root_link_lin_vel_w
    with torch.no_grad():
        # Isolate the command gate from the frozen policy's balance performance.
        from unittest.mock import PropertyMock, patch
        speed = torch.zeros_like(actual_vel)
        speed[:, 0] = .1
        active = flat.nonzero().flatten()[:1]
        upper = up.nonzero().flatten()[:1]
        selected = torch.cat((active, upper))
        env.command_mode[selected] = env.WAIT_TO_START
        env.commands[selected, :2] = 0.
        env.command_elapsed[selected] = env.stop_command_step[selected]
        env.stable_command_ticks.zero_()
        env.gravity[selected] = env.gravity.new_tensor((0., 0., -1.))
        env.normal_support[selected] = True
        env.contacts.step_contact[selected] = False
        with patch.object(type(data), 'root_link_lin_vel_w', new_callable=PropertyMock, return_value=speed):
            env._advance_restart_commands()
            assert env.commands[selected, 0].eq(0).all()
            speed.zero_()
            for _ in range(ticks-1):
                env._advance_restart_commands()
            assert env.commands[selected, 0].eq(0).all()
            env._advance_restart_commands()
            assert env.commands[selected, 0].eq(2).all()
            assert env.command_mode[active].eq(env.FIRST_DRIVE).all() and env.command_mode[upper].eq(0).all()
            env.command_elapsed[active] = round(env.cfg.step.restart_first_drive_s/env.dt)
            env._advance_restart_commands()
            assert env.command_mode[active].eq(env.WAIT_TO_RESTART).all() and env.commands[active, 0].eq(0).all()
            speed[:, 0] = .1
            env._advance_restart_commands()
            assert env.commands[active, 0].eq(0).all()
            speed.zero_()
            for _ in range(ticks):
                env._advance_restart_commands()
            assert env.command_mode[active].eq(env.SECOND_DRIVE).all() and env.commands[active, 0].eq(2).all()
            env.command_elapsed[active] = round(env.cfg.step.restart_second_drive_s/env.dt)
            env._advance_restart_commands()
            assert env.command_mode[active].eq(env.FINAL_STOP).all() and env.commands[active, 0].eq(0).all()
    env.reset()
    return dict(flat_and_up_start_checked=True, flat_stop_restart_checked=True, moving_blocks_start=True,
        stable_ticks=ticks, preserves_immediate_up_and_down=True, yaw=0., motion_speed=2., platform_height=.2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
