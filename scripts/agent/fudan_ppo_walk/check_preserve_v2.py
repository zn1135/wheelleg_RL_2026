"""Check explicit v2 continuation, matching precheck and exact7000->8000 stop; no dynamics/NN/Adam."""
import argparse
from contextlib import ExitStack,redirect_stdout,redirect_stderr
import gc
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ct_mjlab.scripts import train as cli
from ct_mjlab.rsl_rl.runners import fudan_on_policy_runner as module
from ct_mjlab.fudan_ppo_walk.contract import RESET_CONTRACT,RESET_V2_CONTRACT
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.envs.fudan_ppo_walk.commands import command_config,LEGACY_COMMAND_SAMPLING
from ct_mjlab.envs.fudan_ppo_walk.disturbances import DisturbanceConfig
from . import precheck
from .check_resume import equal


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(1)
    parent=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert parent['iterations']==7000 and parent['num_envs']==4096 and parent['reset_contract']==RESET_V2_CONTRACT
    checks=[]
    def verify(name,value):
        assert value,name
        checks.append(name)
    def forbidden(*a,**kw):raise AssertionError('Real dynamics, NN forward or optimizer update forbidden')
    with ExitStack() as stack,tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        for name in ['mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2','mjlab.envs.ManagerBasedRlEnv.step',
                     'torch.nn.Linear.forward','torch.optim.Adam.step']:
            stack.enter_context(patch(name,side_effect=forbidden))
        proof=directory/'proof.json'
        def write_proof(version=RESET_V2_CONTRACT,**changes):
            row=dict(passed=True,reset_contract=version,kind='physical-visible-precheck',num_envs=20,
                     physics_steps=1000,command_sampling=command_config())
            row.update(changes);proof.write_text(json.dumps(row))
        write_proof();record={}
        class FakeRunner:
            def __init__(self,env,**kw):self.iteration=0
            def load(self,path,**kw):record['load_options']=kw;self.iteration=7000;return {}
            def learn_until(self,deadline,max_iterations=None):record.update(deadline=deadline,target=max_iterations);return {}
        def fake_env(*a,**kw):record.update(reset_contract=kw['reset_contract'],num_envs=kw['num_envs']);return SimpleNamespace(close=lambda:None)
        argv=['--task','fudan_ppo_walk','--num-envs','4096','--iterations','1000','--resume',str(args.checkpoint),
              '--preserve-history-v2','--precheck',str(proof)]
        with patch('ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk.FudanEnv',side_effect=fake_env),patch('ct_mjlab.rsl_rl.runners.fudan_on_policy_runner.FudanRunner',FakeRunner),patch('ct_mjlab.rsl_rl.runners.fudan_on_policy_runner.snapshot'),redirect_stdout(io.StringIO()):
            cli.main(argv+['--log-dir',str(directory/'cli')])
        verify('explicit_v2_to_v2_4096_exact1000',record==dict(load_options={},reset_contract=RESET_V2_CONTRACT,
            num_envs=4096,deadline=float('inf'),target=8000))
        cases=[('wrong_history',dict(reset_contract=RESET_CONTRACT)),('old_sampling',dict(command_sampling=LEGACY_COMMAND_SAMPLING)),
               ('zero_physics',dict(physics_steps=0)),('failed_precheck',dict(passed=False)),
               ('wrong_disturbance',dict(disturbances=DisturbanceConfig(action_delay=True).record()))]
        for name,changes in cases:
            write_proof(**changes)
            with patch('ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk.FudanEnv',side_effect=forbidden):
                try:cli.main(argv+['--log-dir',str(directory/name)])
                except ValueError:rejected=True
                else:rejected=False
            verify('reject_'+name,rejected)
        write_proof()
        for source in [dict(parent,reset_contract=RESET_CONTRACT),dict(parent,reset_contract='unknown')]:
            with patch('torch.load',return_value=source),patch('ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk.FudanEnv',side_effect=forbidden):
                try:cli.main(argv+['--log-dir',str(directory/'source')])
                except ValueError:rejected=True
                else:rejected=False
            verify('reject_non_v2_source_'+source['reset_contract'],rejected)
        with redirect_stderr(io.StringIO()):
            try:cli.main(argv+['--migrate-history-v2','--log-dir',str(directory/'conflict')])
            except SystemExit as exc:rejected=exc.code==2
            else:rejected=False
        verify('preserve_and_migrate_mutually_exclusive',rejected)
        class Reached(Exception):pass
        for version,contract in [('v2',RESET_V2_CONTRACT),('v3',RESET_CONTRACT)]:
            def selected(*a,**kw):
                verify('precheck_selects_'+version,kw['reset_contract']==contract)
                raise Reached()
            destination=directory/('precheck_'+version)
            command=['precheck','--output',str(destination),'--deadline',str(time.time()+60),'--viewer','viser','--history-version',version]
            with patch.object(sys,'argv',command),patch.object(precheck,'FudanEnv',side_effect=selected):
                try:precheck.main()
                except Reached:pass
            verify('precheck_records_commands_'+version,json.loads((destination/'report.json').read_text())['command_sampling']==command_config())

        # 真实训练先创建CUDA物理场景；先初始化上下文，避免旧的延迟manual_seed覆盖恢复状态。
        torch.cuda.get_rng_state_all()
        for device in ['cpu','cuda:0']:
            env=object.__new__(FudanEnv);env.reset_contract=RESET_V2_CONTRACT;env.num_envs=4096
            env.speed_limit=2.;env.total_steps=0;env.env=SimpleNamespace(common_step_counter=0)
            env.disturbance_config=DisturbanceConfig();env.command_sampling=command_config()
            with redirect_stdout(io.StringIO()):r=module.FudanRunner(env,log_dir=directory/device.replace(':','_'),device=device)
            if not hasattr(r, 'train_cfg'):
                from scripts.agent.fudan_ppo_walk.reset_v2_fixture import runner_configuration
                runner_configuration(r)
            restored=r.load(args.checkpoint)
            equal(r.policy.state_dict(),parent['model_state_dict'])
            equal(r.alg.optimizer.state_dict(),parent['optimizer_state_dict'])
            equal(r.alg.extra_optimizer.state_dict(),parent['encoder_optimizer_state_dict'])
            equal(torch.get_rng_state(),parent['torch_rng']);equal(torch.cuda.get_rng_state_all(),parent['cuda_rng'])
            verify('complete_model_twoAdam_RNG_restore_'+device,restored['source_reset_contract']==restored['target_reset_contract']==RESET_V2_CONTRACT)
            verify('no_history_migration_'+device,restored['history_migration'] is None)
            verify('command_sampling_only_change_'+device,restored['source_command_sampling']==LEGACY_COMMAND_SAMPLING
                   and restored['command_sampling']==command_config() and restored['disturbances']==DisturbanceConfig().record())
            verify('counters_and_rates_'+device,(r.iteration,r.transitions,r.primary_updates,r.encoder_updates)==(7000,860160000,140000,140000)
                   and r.alg.learning_rate==parent['optimizer_state_dict']['param_groups'][0]['lr']
                   and r.alg.extra_optimizer.param_groups[0]['lr']==parent['encoder_optimizer_state_dict']['param_groups'][0]['lr'])
            r.writer.close();del r,env;gc.collect();torch.cuda.empty_cache()

        r=object.__new__(module.FudanRunner);r.iteration=7000;r.transitions=860160000
        r.primary_updates=r.encoder_updates=140000;r.partial_update=False;r.resume_from={}
        r.log_dir=directory/'loop';r.log_dir.mkdir();calls=dict(act=0,update=0)
        obs=torch.zeros(4,25);critic=torch.zeros(4,141);history=torch.zeros(4,125);reward=torch.zeros(4);done=torch.zeros(4,dtype=torch.bool)
        info={'terms':{k:reward for k in module.WEIGHTS}}
        r.env=SimpleNamespace(num_envs=4096,speed_limit=2.5,reset_contract=RESET_V2_CONTRACT,
            command_sampling=command_config(),step=lambda a:(obs,critic,reward,done,info,history))
        def act(*a):calls['act']+=1;return torch.zeros(4,6)
        def update():calls['update']+=1;r.primary_updates+=20;r.encoder_updates+=20;return [0.]*4
        def save(name):path=r.log_dir/name;path.write_text(str(r.iteration));return path
        r.save=save;r.policy=SimpleNamespace(encode=lambda h:torch.zeros(4,3))
        r.alg=SimpleNamespace(act=act,process_env_step=lambda *a:None,compute_returns=lambda *a:None,update=update,learning_rate=.001)
        r.writer=SimpleNamespace(add_scalar=lambda *a:None,flush=lambda:None,close=lambda:None)
        with patch.object(module,'initialize_rollout',return_value=(obs,critic,history)),redirect_stdout(io.StringIO()):
            if not hasattr(r, 'train_cfg'):
                from scripts.agent.fudan_ppo_walk.reset_v2_fixture import runner_configuration
                runner_configuration(r)
            result=r.learn_until(float('inf'),max_iterations=8000)
        verify('real_loop_stops8000_not8001',calls==dict(act=48000,update=1000) and r.iteration==8000)
        verify('exact_transitions_and_both_updates',r.transitions==1056768000 and r.primary_updates==r.encoder_updates==160000)
        verify('final_saved_and_no_partial',(r.log_dir/'model_final.pt').read_text()=='8000' and not result['partial_update'])
    report=dict(passed=True,checks=checks,count=len(checks),physics_steps=0,model_forwards=0,optimizer_updates=0,
                scope='Actual model/Adam/RNG restoration and real runner boundary with fake collection/update; not behavior validation')
    with args.output.open('x') as out:out.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))


if __name__=='__main__':main()
