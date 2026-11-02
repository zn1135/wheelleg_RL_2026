"""Check v3 history/control flow with a nonphysical backend; never run a policy."""
import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch

from ct_mjlab.fudan_ppo_walk import __main__ as cli
from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT,RESET_V2_CONTRACT,LEGACY_RESET_CONTRACT
from ct_mjlab.fudan_ppo_walk.export import history_metadata
from ct_mjlab.fudan_ppo_walk.measurement import configure_samples,update_evaluation_commands
from ct_mjlab.fudan_ppo_walk.runner import initialize_rollout
from .reset_v2_fixture import fixture


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.manual_seed(719);checks=[]
    def verify(name,value):
        assert value,name
        checks.append(name)
    def forbidden(*a,**kw):raise AssertionError('Real physics/policy/optimizer operation forbidden')
    with ExitStack() as stack,tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        for name in ('mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2','torch.optim.Adam.step',
                'mjlab.envs.ManagerBasedRlEnv.step','torch.nn.Linear.forward'):
            stack.enter_context(patch(name,side_effect=forbidden))
        env=fixture(32,reset_contract=RESET_CONTRACT)
        with patch('torch.rand_like',side_effect=lambda v:torch.full_like(v,.75)) as noise:
            obs,critic,history=initialize_rollout(env)
        verify('startup_same_noisy_frame5',torch.equal(history,obs.repeat(1,5)))
        verify('startup_single_noise_draw',noise.call_count==1 and not torch.equal(obs,env.frame()))
        verify('startup_one_mock_zero_step',env.mock_physics_calls==1 and env.total_steps==1)
        verify('startup_phase_after_warmup',env.events==['physical_reset','command_resample','mock_physics_step']
            and env.episode_steps.min()>=0 and env.episode_steps.max()<2000
            and len(env.episode_steps.unique())>1 and torch.equal(env.episode_steps,env.env.episode_length_buf))
        verify('fresh_zero_action',torch.equal(env.action.raw_action,torch.zeros(32,6)))
        before=history.clone()
        env.env.sim.data.qvel[:,3]=1000.;env.env.sim.data.qvel[:,4]=-1000.
        with patch('torch.rand_like',side_effect=lambda v:torch.full_like(v,.75)) as noise:
            obs,_,_,_,_,history=env.step(torch.full((32,6),.3),auto_reset=False,resample=False)
        verify('normal_fifo_shift_once',torch.equal(history[:,:100],before[:,25:]))
        verify('normal_same_current_and_history',torch.equal(obs,history[:,-25:]) and noise.call_count==1)
        verify('clip_before_append',history.abs().max()<=100 and obs[0,0]==100 and obs[0,1]==-100)

        env=fixture(4,reset_contract=RESET_CONTRACT);initialize_rollout(env)
        env.episode_steps[:]=torch.tensor([100,2000,200,300]);env.env.episode_length_buf.copy_(env.episode_steps)
        env.history=torch.linspace(-.9,.9,500).reshape(4,125);before=env.history.clone()
        env.env.sim.data.qvel[:,3]=torch.tensor([0.,0.,1000.,0.])
        with patch('torch.rand_like',side_effect=lambda v:torch.full_like(v,.75)) as noise:
            obs,_,_,done,_,history=env.step(torch.full((4,6),.3))
        verify('selected_timeout_reset_only',done.tolist()==[False,True,False,False] and env.episode_steps.tolist()==[101,0,201,301])
        verify('selected_history_refilled',torch.equal(history[1],obs[1].repeat(5)))
        verify('unselected_histories_preserved_then_shifted',torch.equal(history[[0,2,3],:100],before[[0,2,3],25:]))
        verify('selected_fresh_action_and_state',torch.equal(env.action.raw_action[1],torch.zeros(6)) and env.angular[1,0]!=0. and env.angular[2,0]==1000.)
        verify('local_reset_single_noise_draw',noise.call_count==1 and torch.equal(history[:,-25:],obs))
        before=history.clone();env.reset_idx(torch.tensor([1]))
        verify('reset_idx_does_not_touch_other_history',torch.equal(env.history[[0,2,3]],before[[0,2,3]]) and env._history_needs_fill.tolist()==[False,True,False,False])

        env=fixture(3,reset_contract=RESET_CONTRACT)
        items=[dict(id=f'fixture_{i}',seed=431+i,randomized=i>0,case=dict(id=f'case_{i}',segments=[dict(start=0.,command=[.5,0.,.26])])) for i in range(3)]
        saved=directory/'configuration';saved.mkdir()
        with patch('mujoco.mj_saveModel',side_effect=lambda model,path:Path(path).write_bytes(b'nonphysical fixture')),patch('torch.rand_like',side_effect=lambda v:torch.full_like(v,.75)) as noise:
            obs,_=configure_samples(env,items,saved)
        verify('evaluation_same_frame_fill',torch.equal(env.history,obs.repeat(1,5)) and noise.call_count==1 and env.mock_physics_calls==0)
        mapping=json.loads((saved/'mapping.json').read_text())
        verify('evaluation_version_and_pd_centred_geometry',mapping['reset_contract']==RESET_CONTRACT
            and torch.equal(env.env.sim.data.qpos[:,env.action._motor_q_adr],env.action.nominal_positions))
        first=env.history[:,:100].clone();old=obs.clone()
        with patch('torch.rand_like',side_effect=forbidden):update_evaluation_commands(env,[[100.,1000.,100.]]*3,obs)
        verify('evaluation_command_does_not_resample_or_refill',torch.equal(first,env.history[:,:100]) and torch.equal(obs[:,0:6],old[:,0:6]))
        verify('evaluation_command_clip_and_latest_match',obs[:,6:9].eq(100.).all() and torch.equal(obs,env.history[:,-25:]))

        verify('export_v3_metadata',history_metadata(RESET_CONTRACT)['initial_fill']=='repeat_same_noisy_clipped_frame5' and not history_metadata(RESET_CONTRACT)['history_before_clip'])
        verify('export_v2_metadata_preserved',history_metadata(RESET_V2_CONTRACT)['history_before_clip'])

        from . import evaluate
        class ReachedVersionSelection(Exception):pass
        for version in (RESET_V2_CONTRACT,RESET_CONTRACT):
            def selected(*a,**kw):
                verify('gpu_selects_checkpoint_'+version,kw['reset_contract']==version)
                raise ReachedVersionSelection()
            target=directory/('gpu_'+version[-2:])
            with patch.object(evaluate,'load_policy',return_value=(None,{'reset_contract':version})),patch('ct_mjlab.fudan_ppo_walk.env.FudanEnv',side_effect=selected):
                try:evaluate.gpu(SimpleNamespace(checkpoint=Path('unused'),diagnostic=True,diagnostic_height=None,output=target,batch_size=20,deadline=time.time()+60))
                except ReachedVersionSelection:pass
                else:raise AssertionError('GPU contract routing did not run')
        for version in (LEGACY_RESET_CONTRACT,'unknown'):
            with patch.object(evaluate,'load_policy',return_value=(None,{'reset_contract':version})),patch('ct_mjlab.fudan_ppo_walk.env.FudanEnv',side_effect=forbidden):
                try:evaluate.gpu(SimpleNamespace(checkpoint=Path('unused')))
                except ValueError:rejected=True
                else:rejected=False
            verify('gpu_rejects_'+version,rejected)

        # proof仅存在于临时夹具，env/runner均替代，不能用作真实物理预检证据。
        proof=directory/'fixture-proof.json'
        proof.write_text(json.dumps(dict(passed=True,reset_contract=RESET_V2_CONTRACT,kind='physical-visible-precheck',num_envs=20,physics_steps=1000)))
        with patch.object(cli,'FudanEnv',side_effect=forbidden):
            try:cli.main(['train','--num-envs','4096','--iterations','1','--precheck',str(proof),'--log-dir',str(directory/'reject')])
            except ValueError:rejected=True
            else:rejected=False
        verify('old_precheck_cannot_approve_v3',rejected)
        proof.write_text(json.dumps(dict(passed=True,reset_contract=RESET_CONTRACT,kind='physical-visible-precheck',num_envs=20,physics_steps=1000)))
        with patch.object(cli,'FudanEnv',side_effect=forbidden):
            try:cli.main(['train','--num-envs','4096','--iterations','1','--migrate-history-v2','--precheck',str(proof),'--log-dir',str(directory/'missing_resume')])
            except ValueError:rejected=True
            else:rejected=False
        verify('migration_requires_parent_checkpoint',rejected)
        entered={}
        class EntryRunner:
            def __init__(self,*a,**kw):self.iteration=0
            def load(self,path,*,migrate_history_v2=False):
                entered['migration']=migrate_history_v2;self.iteration=7000;return {}
            def learn_until(self,deadline,max_iterations=None):
                entered.update(deadline=deadline,target=max_iterations);return {}
        with patch.object(cli,'FudanEnv',return_value=SimpleNamespace(close=lambda:None)),patch.object(cli,'FudanRunner',EntryRunner),patch.object(cli,'snapshot'),patch.object(cli.torch,'load',return_value={}),redirect_stdout(io.StringIO()):
            cli.main(['train','--num-envs','4096','--iterations','1','--resume','fixture.pt','--migrate-history-v2','--precheck',str(proof),'--log-dir',str(directory/'mock_entry')])
        verify('cli_passes_explicit_migration_and_additional_budget',entered==dict(migration=True,deadline=float('inf'),target=7001))
    result=dict(passed=True,reset_contract=RESET_CONTRACT,checks=checks,count=len(checks),
        physics_steps=0,model_forwards=0,optimizer_updates=0,physical_precheck='not_run',training='not_started',
        scope='Real preprocessing/reset/CLI control flow with nonphysical backend and synthetic sensor values')
    with args.output.open('x') as output:output.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
