"""Zero-physics/zero-update checks for real reset, rollout startup and evaluator preprocessing."""
import argparse,ast,io,json,tempfile,time
from contextlib import redirect_stdout,ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import initialize_rollout,FudanRunner
from ct_mjlab.fudan_ppo_walk.contract import RESET_V2_CONTRACT as RESET_CONTRACT,RESET_CONTRACT as CURRENT_RESET_CONTRACT,LEGACY_RESET_CONTRACT
from ct_mjlab.fudan_ppo_walk.measurement import configure_samples,update_evaluation_commands
from .reset_v2_fixture import fixture


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[3];torch.manual_seed(73)
    with ExitStack() as stack:
        for name in ('mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2','torch.optim.Adam.step','mjlab.envs.ManagerBasedRlEnv.step'):
            stack.enter_context(patch(name,side_effect=AssertionError('Real physics/optimizer operation forbidden')))
        env=fixture(32);env.events.clear()
        original_rand=torch.randint_like
        def phase_rand(*args,**kwargs):
            assert env.mock_physics_calls==1 and env.total_steps==1
            env.events.append('random_phase');return original_rand(*args,**kwargs)
        with patch('torch.randint_like',side_effect=phase_rand):obs,critic,history=initialize_rollout(env)
        assert env.events==['physical_reset','command_resample','mock_physics_step','random_phase']
        assert torch.equal(env.episode_steps,env.env.episode_length_buf)
        assert len(env.episode_steps.unique())>1 and env.episode_steps.min()>=0 and env.episode_steps.max()<2000
        assert env.total_steps==env.env.common_step_counter==1
        assert torch.equal(obs,history[:,-25:])

        env=fixture();env.reset_idx(torch.arange(4));env.resample_calls.clear()
        env.episode_steps[:]=torch.tensor([499,500,1999,2000]);env.env.episode_length_buf.copy_(env.episode_steps)
        result=env.step(torch.zeros(4,6),auto_reset=False)
        assert env.episode_steps.tolist()==[500,501,2000,2001]
        assert [x.tolist() for x in env.resample_calls]==[[0,2]]
        assert result[3].tolist()==[False,False,False,True]

        env=fixture();env.reset_idx(torch.arange(4));env.episode_steps[:]=torch.tensor([100,2000,200,300]);env.env.episode_length_buf.copy_(env.episode_steps)
        env.history=torch.arange(500,dtype=torch.float32).reshape(4,125)
        before=env.history.clone();env.env.sim.data.qvel[:,3]=torch.tensor([0.,0.,1000.,0.])
        with patch('torch.rand_like',side_effect=lambda value:torch.full_like(value,.75)):
            obs,_,_,done,_,history=env.step(torch.full((4,6),.3))
        assert done.tolist()==[False,True,False,False]
        assert env.episode_steps.tolist()==[101,0,201,301]
        assert torch.equal(history[0,:100],before[0,25:])
        clean=env.frame()[1]
        assert torch.equal(history[1,:100],clean.repeat(4))
        assert not torch.equal(history[1,-25:],clean)
        assert torch.equal(env.action.raw_action[1],torch.zeros(6))
        assert env.angular[1,0]!=0. and env.angular[2,0]==1000.  # Fresh selected reset, other env preserved.
        assert history[2,-25]>100. and obs[2,0]==100.  # Raw history precedes current clipping.
        assert torch.equal(env.previous[1],torch.zeros(6,2))

        env=fixture();env.reset_idx(torch.arange(4));env.total_steps=1999;env.env.common_step_counter=1999
        env.episode_steps[:]=torch.tensor([2000,17,20,99]);env.sums[:]=20.
        env.step(torch.zeros(4,6),resample=False)
        assert env.total_steps==2000 and abs(env.speed_limit-2.1)<1e-6
        assert env.episode_steps.tolist()==[0,18,21,100]

        # Real configure_samples must re-solve after the new per-sample PD centre is chosen.
        env=fixture(3)
        items=[dict(id=f'fixture_{i}',seed=430+i,randomized=i>0,case=dict(id=f'case_{i}',segments=[dict(start=0.,command=[.5,0.,.26])])) for i in range(3)]
        with tempfile.TemporaryDirectory() as directory,patch('mujoco.mj_saveModel',side_effect=lambda model,path:Path(path).write_bytes(b'fixture only')):
            obs,_=configure_samples(env,items,Path(directory))
            q=env.env.sim.data.qpos[:,env.action._motor_q_adr]
            assert torch.equal(q,env.action.nominal_positions)
            clean=env.frame();expected=torch.cat((clean.repeat(1,4),env._history_frame),-1)
            assert torch.equal(env.history,expected) and env.mock_physics_calls==0
            assert not torch.equal(env.history[1],obs[1].repeat(5))
            initial=np.load(Path(directory)/'fixture_1_initial.npz')
            mapping=json.loads((Path(directory)/'mapping.json').read_text())
            assert mapping['reset_contract']==RESET_CONTRACT
            assert np.array_equal(initial['qpos'][mapping['motor_q']],initial['nominal'])
            first100=env.history[:,:100].clone();raw=env._history_frame.clone()
            update_evaluation_commands(env,[[1.,.1,.22]]*3,obs)
            assert torch.equal(env.history[:,:100],first100)
            # Updating requested commands never replaces sensor noise/history with noisy.repeat(5).
            sensor_slots=[0,1,2,3,4,5,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
            assert torch.equal(env._history_frame[:,sensor_slots],raw[:,sensor_slots])

        # Default train entry reaches the same real startup flow with a non-physical backend.
        from ct_mjlab.scripts import train as cli
        entered={}
        class EntryRunner:
            def __init__(self,env,**kwargs):self.env=env
            def learn_until(self,deadline):
                initialize_rollout(self.env);entered['env']=self.env;return {'fixture':True}
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);proof=directory/'proof.json';proof.write_text(json.dumps(dict(passed=True,reset_contract=CURRENT_RESET_CONTRACT,kind='physical-visible-precheck',num_envs=20,physics_steps=1000,command_sampling=__import__('ct_mjlab.envs.fudan_ppo_walk.commands',fromlist=['command_config']).command_config())))
            with patch('ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk.FudanEnv',side_effect=lambda *args,**kwargs:fixture(32,reset_contract=CURRENT_RESET_CONTRACT)),patch('ct_mjlab.rsl_rl.runners.fudan_on_policy_runner.FudanRunner',EntryRunner),patch('ct_mjlab.rsl_rl.runners.fudan_on_policy_runner.snapshot'),redirect_stdout(io.StringIO()):
                cli.main(['--task','fudan_ppo_walk','--num-envs','32','--log-dir',str(directory/'run'),'--precheck',str(proof),'--deadline',str(time.time()+60)])
            assert entered['env'].mock_physics_calls==1
            proof.write_text(json.dumps(dict(passed=True,reset_contract=RESET_CONTRACT,physics_steps=0)))
            with patch('ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk.FudanEnv',side_effect=AssertionError('Legacy proof must fail before environment construction')):
                try:cli.main(['--task','fudan_ppo_walk','--num-envs','32','--log-dir',str(directory/'legacy'),
                    '--precheck',str(proof),'--deadline',str(time.time()+60)])
                except ValueError:pass
                else:raise AssertionError('Legacy precheck silently approved reset v2')
        from . import evaluate
        with patch.object(evaluate,'load_policy',return_value=(None,{})):
            try:evaluate.gpu(SimpleNamespace(checkpoint=Path('unused')))
            except ValueError:pass
            else:raise AssertionError('New GPU evaluation silently accepted a legacy model')

    # Save metadata is versioned, but this is not a full training restore test.
    with tempfile.TemporaryDirectory() as directory,patch('torch.cuda.get_rng_state_all',return_value=[]):
        saved=object.__new__(FudanRunner);saved.log_dir=Path(directory)
        saved.policy=SimpleNamespace(state_dict=lambda:{'fixture':torch.zeros(1)})
        empty=SimpleNamespace(state_dict=lambda:{'state':{},'param_groups':[]})
        saved.alg=SimpleNamespace(optimizer=empty,extra_optimizer=empty);saved.env=SimpleNamespace(speed_limit=2.,num_envs=32,total_steps=0,reset_contract=RESET_CONTRACT)
        saved.initialization='random_from_scratch_seed_1';saved.resume_from=None
        saved.iteration=saved.transitions=saved.primary_updates=saved.encoder_updates=0;saved.partial_update=False
        if not hasattr(saved, 'train_cfg'):
            from scripts.agent.fudan_ppo_walk.reset_v2_fixture import runner_configuration
            runner_configuration(saved)
        data=torch.load(saved.save('fixture.pt'),map_location='cpu',weights_only=True)
        assert data['reset_contract']==RESET_CONTRACT and data['primary_updates']==data['encoder_updates']==0
        assert 'physics_state' not in data
    # Reward/domain-randomization code and all recipe coefficients remain exactly unchanged.
    before=root/'artifacts/fudan_ppo_walk/fudan-reset-v2-20261101/before/fudan_ppo_walk'
    def function(text,name):
        return next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name)
    from ct_mjlab.fudan_ppo_walk.source_migration import legacy_source_directory
    current=legacy_source_directory()
    for name in ('reward','randomize','process_actions','_ground_hold'):
        assert ast.dump(function((before/'env.py').read_text(),name),include_attributes=False)==ast.dump(function((current/'env.py').read_text(),name),include_attributes=False)
    for name in ('WEIGHTS','PPO_CONFIG'):
        def value(path):
            return next(ast.dump(n.value,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        assert value(before/'contract.py')==value(current/'contract.py')
    # Check the evaluator's actual default call sites, rather than an unused helper alone.
    tree=ast.parse((root/'scripts/agent/fudan_ppo_walk/evaluate.py').read_text())
    gpu=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gpu')
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='update_evaluation_commands' for n in ast.walk(gpu))
    assert not any(isinstance(n,ast.Attribute) and n.attr=='repeat' for n in ast.walk(gpu))
    report=dict(passed=True,reset_contract=RESET_CONTRACT,planned_startup_zero_step_calls=1,
        real_physics_steps=0,optimizer_updates=0,model_forward_calls=0,random_phase_after_warmup=True,
        local_resets_independent=True,fresh_reset_state_and_cleared_action=True,history_clean4_noisy1=True,
        history_before_current_clip=True,configure_rebuilds_pd_centred_geometry=True,
        gpu_command_updates_preserve_history=True,
        strict_timeout_2000_and_resampling_500=True,global_curriculum_clock=True,default_cli_verified=True,
        checkpoint_reset_metadata_verified=True,legacy_precheck_and_gpu_model_rejected_before_physics=True,static_proof_cannot_approve_training=True,historical_reward_ast_and_known_migration_verified=True,
        dynamic_precheck='not_run',training='not_run',full_training_resume='not_covered_by_this_check')
    if a.output:a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
