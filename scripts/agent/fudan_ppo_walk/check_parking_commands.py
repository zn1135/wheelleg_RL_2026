"""Actual command/reset/step paths with a nonphysical backend; no learning."""
import argparse
import ast
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch
from ct_mjlab.fudan_ppo_walk import commands as sampling
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import initialize_rollout,FudanRunner
from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT
from .reset_v2_fixture import fixture


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);checks=[]
    def verify(name,value):assert bool(value),name;checks.append(name)
    def forbidden(*a,**kw):raise AssertionError('Real physics, network forward and updates forbidden')
    with ExitStack() as stack:
        for name in ('mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2',
                'mjlab.envs.ManagerBasedRlEnv.step','torch.nn.Linear.forward','torch.optim.Adam.step'):
            stack.enter_context(patch(name,side_effect=forbidden))
        cfg=sampling.command_config()
        verify('default10percent_stand10percent_stop',cfg['standstill_env_ratio']==cfg['stop_env_ratio']==.1)
        for field,value in [('standstill_env_ratio',-.1),('stop_env_ratio',1.),('stop_drive_time_s',[0.,3.]),
                ('stop_drive_time_s',[3.,20.]),('stop_speed_range',[.25,3.])]:
            with patch.object(FudanWalkCfg.commands,field,value):
                try:sampling.command_config()
                except ValueError:rejected=True
                else:rejected=False
                verify('reject_'+field+str(value),rejected)
        env=fixture(4,RESET_CONTRACT);sampling.initialize(env,dict(cfg,stop_drive_time_s=[3.,3.]))
        ids=torch.arange(4)
        with patch('torch.rand',return_value=torch.tensor([.05,.15,.4,.9])):sampling.reset_modes(env,ids)
        env.resample(ids)
        verify('three_modes_assigned',env.command_mode.tolist()==[1,2,0,0])
        verify('standing_command_exact_zero',env.commands[0,:2].eq(0).all())
        verify('stopping_starts_with_nonzero_linear_only',.25<=abs(env.commands[1,0])<=1. and env.commands[1,1]==0.)
        verify('height_range_preserved',((env.commands[:,2]>=.22)&(env.commands[:,2]<=.30)).all())
        verify('parking_uses_three_height_choices',torch.isin(env.commands[:2,2],torch.tensor([.22,.26,.30])).all())
        height=env.commands[:2,2].clone();driving=env.commands[1,0].clone()
        env.command_elapsed[:]=298;env.episode_steps[:]=498
        env.step(torch.zeros(4,6),auto_reset=False)
        verify('drive_until_tick299',env.command_elapsed[1]==299 and env.commands[1,0]==driving)
        obs,_,_,_,_,_=env.step(torch.zeros(4,6),auto_reset=False)
        verify('stop_at_tick300_even_on_resample_boundary',env.command_elapsed[1]==300 and env.commands[1,:2].eq(0).all())
        verify('next_actor_frame_sees_zero_command',obs[1,6:8].eq(0).all())
        verify('parking_height_not_resampled',torch.equal(env.commands[:2,2],height))
        for boundary in (999,1499,1998):
            env.episode_steps[:]=boundary;env.command_elapsed[:]=boundary
            env.step(torch.zeros(4,6),auto_reset=False)
            verify('hold_zero_through_'+str(boundary+1),env.commands[:2,:2].eq(0).all())
        env.command_mode[:]=2;env.command_elapsed[:]=10000;env.commands[:,:2]=.7
        env.step(torch.zeros(4,6),auto_reset=False,resample=False)
        verify('frozen_evaluation_commands_not_overridden',env.commands[:,:2].eq(.7).all())

        before_modes=env.command_mode[[0,1,3]].clone();before_commands=env.commands[[0,1,3]].clone()
        before_elapsed=env.command_elapsed[[0,1,3]].clone()
        with patch('torch.rand',side_effect=[torch.tensor([.15]),torch.tensor([.8])]):env.reset_idx(torch.tensor([2]))
        verify('selected_reset_gets_fresh_schedule',env.command_elapsed[2]==0 and env.command_mode[2]==2 and env.stop_command_step[2]==300)
        verify('other_environments_untouched',torch.equal(before_modes,env.command_mode[[0,1,3]])
            and torch.equal(before_commands,env.commands[[0,1,3]]) and torch.equal(before_elapsed,env.command_elapsed[[0,1,3]]))

        env=fixture(4,RESET_CONTRACT);sampling.initialize(env,dict(sampling.LEGACY_COMMAND_SAMPLING))
        torch.manual_seed(621);expected=torch.zeros(4,3)
        expected[:,0]=torch.empty(4).uniform_(-2.,2.);expected[:,1]=torch.empty(4).uniform_(-2.,2.)
        expected[:,2]=torch.empty(4).uniform_(.22,.30);rng=torch.get_rng_state().clone()
        torch.manual_seed(621);sampling.reset_modes(env,ids);env.resample(ids)
        verify('disabled_sampling_matches_original_commands_and_rng',torch.equal(env.commands,expected) and torch.equal(torch.get_rng_state(),rng))

        env=fixture(32,RESET_CONTRACT);sampling.initialize(env,dict(cfg,standstill_env_ratio=0.,stop_env_ratio=1.,stop_drive_time_s=[3.,3.]))
        initialize_rollout(env)
        verify('warmup_counts_one_physical_policy_step',env.command_elapsed.eq(1).all())
        verify('random_episode_phase_does_not_move_stop_timer',env.episode_steps.unique().numel()>1 and env.stop_command_step.eq(300).all())
        bulk=SimpleNamespace(num_envs=10000,device='cpu',commands=torch.zeros(10000,3))
        sampling.initialize(bulk,cfg);torch.manual_seed(632);sampling.reset_modes(bulk,torch.arange(10000))
        sampling.apply_parking_commands(bulk,torch.arange(10000))
        fractions=torch.bincount(bulk.command_mode,minlength=3)/10000.
        verify('seeded_episode_mix_matches10_10_80',abs(fractions[0]-.8)<.02 and abs(fractions[1]-.1)<.015 and abs(fractions[2]-.1)<.015)
        forward=float((bulk.commands[bulk.command_mode==2,0]>0).float().mean())
        verify('stopping_directions_balanced',.45<forward<.55)

        with tempfile.TemporaryDirectory() as temporary:
            learner=object.__new__(FudanRunner);learner.device='cpu';learner.log_dir=Path(temporary)
            learner.env=SimpleNamespace(command_sampling=cfg,reset_contract=RESET_CONTRACT,num_envs=4,total_steps=49,
                speed_limit=2.,env=SimpleNamespace(common_step_counter=49))
            learner.policy=torch.nn.Linear(1,1);encoder=torch.nn.Linear(1,1)
            learner.alg=SimpleNamespace(optimizer=torch.optim.Adam(learner.policy.parameters(),lr=.001),
                extra_optimizer=torch.optim.Adam(encoder.parameters(),lr=.001),learning_rate=.001)
            learner.iteration=1;learner.transitions=192;learner.primary_updates=learner.encoder_updates=20
            learner.partial_update=False;learner.initialization='random_from_scratch_seed_1';learner.resume_from=None
            if not hasattr(learner, 'train_cfg'):
                from scripts.agent.fudan_ppo_walk.reset_v2_fixture import runner_configuration
                runner_configuration(learner)
            saved=learner.save('synthetic-checkpoint.pt');record=torch.load(saved,weights_only=False)
            verify('checkpoint_records_sampling_recipe',record['command_sampling']==cfg)
            if not hasattr(learner, 'train_cfg'):
                from scripts.agent.fudan_ppo_walk.reset_v2_fixture import runner_configuration
                runner_configuration(learner)
            restored=learner.load(saved)
            verify('restore_records_source_and_target_recipe',restored['source_command_sampling']==restored['command_sampling']==cfg)
            record.pop('command_sampling');torch.save(record,Path(temporary)/'legacy.pt')
            if not hasattr(learner, 'train_cfg'):
                from scripts.agent.fudan_ppo_walk.reset_v2_fixture import runner_configuration
                runner_configuration(learner)
            restored=learner.load(Path(temporary)/'legacy.pt')
            verify('old_checkpoint_condition_stays_uniform',restored['source_command_sampling']==sampling.LEGACY_COMMAND_SAMPLING
                and restored['command_sampling']==cfg)

        from ct_mjlab.fudan_ppo_walk.source_migration import legacy_source_directory
        sealed = legacy_source_directory()
        path='mjlab_training/src/ct_mjlab/fudan_ppo_walk/env.py'
        old=ast.parse(subprocess.check_output(['git','show','HEAD:'+path],text=True));new=ast.parse((sealed/'env.py').read_text())
        for name in ('reward','frame','_update_state','PhysicalAction','reset_pose'):
            node=lambda tree:next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
            verify(name+'_historical_source_unchanged',ast.dump(node(old),include_attributes=False)==ast.dump(node(new),include_attributes=False))
        for name in ('ppo.py','network.py','storage.py','contract.py','reset_geometry.py'):
            path='mjlab_training/src/ct_mjlab/fudan_ppo_walk/'+name
            verify(name+'_historical_source_unchanged',(sealed/name).read_bytes()==subprocess.check_output(['git','show','HEAD:'+path]))
    result=dict(passed=True,count=len(checks),checks=checks,command_sampling=cfg,
        sampled_episode_fractions=fractions.tolist(),stopping_forward_fraction=forward,
        physics_steps=0,model_forwards=0,optimizer_updates=0,scope='Command scheduling and actual reset/step with nonphysical backend; not trained behavior')
    with args.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
