"""准备从4200恢复固定2m/s直行训练，检查命令采样与停车切换。"""
import argparse
import copy
import json
from pathlib import Path

import torch

from ct_mjlab.envs.chuanliantui.fudan_ppo_step.config import FudanStepCfg, recipe_record
from .launch_training import ROOT, sha
from .remote_training import same_state

SOURCE = ROOT/'logs/mjlab/fudan_ppo_step/20261117_fixed200_turn_v8_4096_resume3200_to4200_01/model_final.pt'
SOURCE_SHA = 'becd82418d671c9ea7113b84102fb3fa387055d57a6fe2edfec835dd853ad1f8'


def prepare(output):
    assert sha(SOURCE) == SOURCE_SHA
    saved = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert saved['task'] == 'fudan_ppo_step' and not saved['partial_update']
    assert (saved['iterations'], saved['num_envs'], saved['transitions']) == (4200, 4096, 825753600)
    assert saved['primary_updates'] == saved['encoder_updates'] == 84000
    before = saved['training_recipe']
    assert before['step']['version'] == 'fixed200-turn-rehearsal-v8'
    after = recipe_record(FudanStepCfg())
    expected = copy.deepcopy(before)
    for key in ('flat_turn_probability', 'turn_yaw_magnitude'):
        del expected['step'][key]
    for key in ('version', 'patch_size', 'command_curriculum', 'approach_speed', 'flat_cruise_s', 'flat_forward_limit_m'):
        expected['step'][key] = after['step'][key]
    expected['commands'].update(yaw_range=[0., 0.], stop_speed_range=[2., 2.])
    assert expected == after
    migration = dict(source=str(SOURCE), source_sha256=SOURCE_SHA, before_recipe=before, after_recipe=after,
        previous_migration=saved.get('recipe_migration'), learning_state_reset=False,
        terrain_assignment_preserved=True, physical_episodes_reinitialized=True)
    derived = dict(saved, training_recipe=after, recipe_migration=migration)
    output.mkdir(parents=True, exist_ok=False)
    path = output/'resume4200_fixed_speed.pt'
    torch.save(derived, path)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    assert same_state(derived, restored)
    assert all(same_state(value, restored[key]) for key, value in saved.items()
               if key not in ('training_recipe', 'recipe_migration'))
    (output/'migration.json').write_text(json.dumps(dict(migration, passed=True,
        all_other_checkpoint_fields_exact=True, derived_checkpoint=str(path.resolve()),
        derived_sha256=sha(path), optimizer_updates=0), indent=2)+'\n')
    print(path)


def check_commands(env):
    cfg = env.cfg.step
    ids = torch.arange(env.num_envs, device=env.device)
    flat = env.modes == 0
    assert cfg.heights == (0., .20) and env.terrain.terrain_levels.eq(1).all()
    assert cfg.flat_forward_limit_m > 2.*cfg.flat_cruise_s + 1.
    cruise_ticks = round(cfg.flat_cruise_s/env.dt)
    found = set()
    for _ in range(64):
        env.resample(ids)
        assert env.commands[:, 1].eq(0).all()
        assert env.commands[~flat, 0].eq(2).all()
        assert env.commands[flat&(env.command_mode == 1), 0].eq(0).all()
        assert env.commands[flat&(env.command_mode != 1), 0].eq(2).all()
        ticks = env.stop_command_step[flat&(env.command_mode == 2)]
        assert ((ticks == cruise_ticks) | ((ticks >= 300)&(ticks <= 500))).all()
        if (ticks == cruise_ticks).any():
            found.add('cruise')
        if (ticks <= 500).any():
            found.add('early_stop')
        if (flat&(env.command_mode == 1)).any():
            found.add('stand')
    assert found == {'cruise', 'early_stop', 'stand'}
    env.reset()
    moving = flat&(env.command_mode == 2)
    while not moving.any():
        env.resample(ids[flat])
        moving = flat&(env.command_mode == 2)
    env.command_elapsed[moving] = env.stop_command_step[moving]-1
    with torch.no_grad():
        env.step(torch.zeros(env.num_envs, 6, device=env.device), auto_reset=False)
    assert env.commands[moving, :2].eq(0).all()
    env.reset()
    return dict(fixed_moving_vx=2., yaw=0., stair_height_m=.20, cruise_s=cfg.flat_cruise_s,
        flat_forward_limit_m=cfg.flat_forward_limit_m, behaviors=sorted(found), timed_stop_checked=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)
