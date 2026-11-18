"""显式迁移3200轮学习状态到固定200mm/高站姿转向配方，不执行优化更新。"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg, recipe_record
from scripts.agent.fudan_ppo_step.remote_training import same_state

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'artifacts/fudan_ppo_step/20261117-v7-final3200-01/training/model_final.pt'
SOURCE_SHA = '536cf8db91b71e711f1cc74ad6e1d6e059a47e04c27b7b9beb405b032633c12a'


def prepare(output):
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    saved = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert saved['iterations'] == 3200 and saved['num_envs'] == 4096 and not saved['partial_update']
    assert saved['transitions'] == 629145600
    assert saved['primary_updates'] == saved['encoder_updates'] == 64000
    before, after = saved['training_recipe'], recipe_record(FudanStepCfg())
    assert before['step']['version'] == 'contact-assisted-platform-exploration-v7'
    expected = copy.deepcopy(before)
    changes = {key: val for key, val in after['step'].items() if before['step'].get(key) != val}
    assert set(changes) == {'version','heights','curriculum','side_obstacle_level',
        'flat_regular_height_range','turn_height','turn_height_probability',
        'flat_turn_probability','turn_yaw_magnitude','flat_teacher_height_max'}
    expected['step'].update(changes)
    expected['commands']['height_range'] = [.22,.35]
    assert expected == after, 'Unrelated physical/reward/PPO changes are forbidden'
    migration = dict(source=str(SOURCE), source_sha256=SOURCE_SHA,
        before_recipe=before, after_recipe=after,
        reason='用户指定只保留200mm台阶，并回练STEP 0.35m高站姿原地及行驶转向；新增1000轮，本机4096环境',
        learning_state_reset=False, physical_episodes_reinitialized=True,
        terrain_levels_remapped_to=1, additional_iterations=1000, target_iterations=4200)
    original = copy.deepcopy(saved)
    saved['training_recipe'] = after
    saved['terrain_levels'] = torch.ones_like(saved['terrain_levels'])
    saved['recipe_migration'] = migration
    # 新目录保存派生输入，原模型、优化器、RNG及全部学习计数保持。
    output.mkdir(parents=True, exist_ok=False)
    path = output/'resume3200_fixed200_turn.pt'
    torch.save(saved, path)
    restored = torch.load(path, map_location='cpu', weights_only=False)
    unchanged = {key: same_state(value, restored[key]) for key, value in original.items()
                 if key not in ('training_recipe','terrain_levels','recipe_migration')}
    assert all(unchanged.values()), unchanged
    assert restored['terrain_levels'].eq(1).all()
    receipt = dict(passed=True, **migration, derived_checkpoint=str(path.resolve()),
        derived_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        unchanged_checkpoint_fields=unchanged, std=restored['model_state_dict']['std'].tolist(),
        optimizer_updates=0, physics_steps=0)
    (output/'migration.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(passed=True, derived_checkpoint=str(path), learning_state_exact=True)))


def check_commands(env, runner):
    """检查真实GPU环境的采样、固定地形等级和教师作用范围。"""
    cfg = env.cfg
    assert cfg.step.heights == (0.,.2) and not cfg.step.curriculum
    assert env.terrain.terrain_levels.eq(1).all()
    ids = torch.arange(env.num_envs, device=env.device)
    flat = env.modes == 0
    samples=[]
    for _ in range(512):
        env.resample(ids)
        assert env.commands[~flat,1].eq(0).all()
        assert torch.allclose(env.commands[~flat,2], torch.full_like(env.commands[~flat,2],.35))
        samples.append(env.commands[flat].clone())
    command = torch.cat(samples)
    tall = torch.isclose(command[:,2], command.new_tensor(.35))
    pure = command[:,0].eq(0)&command[:,1].abs().ge(.5)
    moving = command[:,0].abs().ge(.2)&command[:,1].abs().ge(.5)
    assert .4 < float(tall.float().mean()) < .6
    for mask in (pure & tall, moving & tall):
        assert (command[mask,1]>.5).any() and (command[mask,1]<-.5).any()
    assert command[:,0].abs().max() > 2 and command[:,1].abs().max() > 1.9
    assert (command[:,0].eq(0)&command[:,1].eq(0)&tall).any()
    obs,_,history = env.reset()
    obs = obs.clone()
    obs[:,8] = .35*cfg.normalization.command[2]
    assert runner.flat_preservation_loss(obs,history,flat) is None
    obs[:,8] = .26*cfg.normalization.command[2]
    assert runner.flat_preservation_loss(obs,history,flat) is not None
    # 复位/失败均不能将200mm退成其他高度。
    env.finished.fill_(True);env.crossed.fill_(False);env.reset_idx(ids)
    assert env.terrain.terrain_levels.eq(1).all()
    return dict(fixed_stair_height_m=.2, stair_yaw_zero=True,
        flat_samples=len(command), tall_fraction=float(tall.float().mean()),
        tall_pure_turn_samples=int((pure&tall).sum()), tall_moving_turn_samples=int((moving&tall).sum()),
        both_turn_directions=True, tall_stand_present=True,
        teacher_excludes_tall_commands=True, teacher_retains_regular_commands=True,
        terrain_level_fixed_across_failure_reset=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
