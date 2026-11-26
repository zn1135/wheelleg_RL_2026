"""从固定4200轮派生小转向专项续训输入，并检查实际命令和奖励。"""
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
    assert sha(SOURCE)==SOURCE_SHA
    saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
    assert saved['task']=='fudan_ppo_step' and not saved['partial_update']
    assert (saved['iterations'],saved['num_envs'],saved['transitions'])==(4200,4096,825753600)
    assert saved['primary_updates']==saved['encoder_updates']==84000
    assert saved['terrain_levels'].eq(1).all()
    before=saved['training_recipe']
    assert before['step']['version']=='fixed200-turn-rehearsal-v8'
    after=recipe_record(FudanStepCfg())
    expected=copy.deepcopy(before)
    for key in ('version','focus_yaw_commands','focus_spin_probability',
                'focus_moving_turn_probability','focus_yaw_reward_multiplier'):
        expected['step'][key]=after['step'][key]
    assert expected==after
    migration=dict(source=str(SOURCE),source_sha256=SOURCE_SHA,before_recipe=before,after_recipe=after,
        previous_migration=saved.get('recipe_migration'),source_iterations=4200,
        learning_state_reset=False,terrain_assignment_preserved=True,physical_episodes_reinitialized=True)
    derived=dict(saved,training_recipe=after,recipe_migration=migration)
    output.mkdir(parents=True,exist_ok=False)
    path=output/'resume4200_small_yaw.pt'
    torch.save(derived,path)
    restored=torch.load(path,map_location='cpu',weights_only=False)
    assert same_state(derived,restored)
    assert all(same_state(value,restored[key]) for key,value in saved.items()
               if key not in ('training_recipe','recipe_migration'))
    receipt=dict(migration,passed=True,all_other_checkpoint_fields_exact=True,
        derived_checkpoint=str(path.resolve()),derived_sha256=sha(path),optimizer_updates=0)
    (output/'migration.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(path)


def check_focus(env):
    cfg=env.cfg.step
    ids=torch.arange(env.num_envs,device=env.device);flat=env.modes==0
    samples=[]
    for _ in range(384):
        env.resample(ids)
        assert env.commands[~flat,1].eq(0).all() and env.terrain.terrain_levels.eq(1).all()
        samples.append(torch.cat((env.commands[flat],env.command_mode[flat,None]),-1))
    values=torch.cat(samples);tall=values[:,2]==cfg.turn_height;ordinary=values[:,3]==0
    matched=(values[:,1,None]==values.new_tensor(cfg.focus_yaw_commands)).any(-1)
    spin=tall&ordinary&(values[:,0]==0)
    moving=tall&ordinary&(values[:,0]!=0)
    fractions=[float(matched[group].float().mean()) for group in (spin,moving)]
    assert abs(fractions[0]-cfg.focus_spin_probability)<.12 and abs(fractions[1]-cfg.focus_moving_turn_probability)<.12
    for yaw in cfg.focus_yaw_commands:
        assert (values[spin,1]==yaw).any() and (values[moving,1]==yaw).any()
    assert not (matched&(~tall|~ordinary)).any()
    assert (values[:,2]<=.30).any() and (values[tall,3]==1).any() and (values[tall,3]==2).any()
    env.reset()
    env.commands[flat]=env.commands.new_tensor((0.,.5,.35))
    first=flat.nonzero().flatten()[0];env.commands[first,2]=.30
    focus=flat&(env.commands[:,2]==cfg.turn_height)
    env.angular[:,2]=0.
    saved=env.sums.clone()
    _,boosted=env.reward()
    multiplier=cfg.focus_yaw_reward_multiplier
    try:
        cfg.focus_yaw_reward_multiplier=1.
        _,original=env.reward()
    finally:
        cfg.focus_yaw_reward_multiplier=multiplier
        env.sums.copy_(saved)
    for name in boosted:
        if name=='tracking_ang_vel':
            assert torch.equal(boosted[name][~focus],original[name][~focus])
            assert torch.equal(boosted[name][focus],original[name][focus]*multiplier)
            assert (boosted[name][focus]>original[name][focus]).all()
        else:
            assert torch.equal(boosted[name],original[name]),name
    env.reset()
    return dict(flat_samples=len(values),exact_targets=list(cfg.focus_yaw_commands),
        exact_spin_fraction=fractions[0],exact_moving_turn_fraction=fractions[1],
        other_reward_terms_exact=True,normal_height_and_stair_yaw_reward_exact=True,
        targeted_reward_multiplier=multiplier)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
