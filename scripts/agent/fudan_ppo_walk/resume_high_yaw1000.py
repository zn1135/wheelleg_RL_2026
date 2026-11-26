"""Authorized1000-iteration high-yaw continuation; retain the chosen source recipe."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

from scripts.agent.fudan_ppo_walk.resume_pd200_random_from3000 import ROOT,PYTHON,XML,sha,write,same
from scripts.agent.fudan_ppo_walk.run_torque_ablation import run_command

MODULE='scripts.agent.fudan_ppo_walk.resume_high_yaw1000'
SOURCES={
    3000:('20261114_pd200_p20d1_half_random_delay_wide_resume1000_to3000_01',
          '20261114_model3000_pd200_p20d1_half_random_delay10to20.pt',
          'dfb221c09648d278d5c5e5c7b78e4d729b7a7a5a56458762b7313dc50831c4f8'),
    4000:('20261115_pd200_full_random_force3_push_resume3018_to4000_01',
          '20261115_model4000_pd200_p20d1_full_random_ground04to12_force3_push.pt',
          '1513279bcc519814af3f1aea266967a937f845244002c7c03e83037fa16878b6')}


def prepare(out,start):
    import torch
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg,section_dict,timing_record
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.commands import command_config
    run_name,model_name,digest=SOURCES[start]
    source_run=ROOT/'logs/mjlab/fudan_ppo_walk'/run_name
    model=ROOT/'artifacts/fudan_ppo_walk/models'/model_name;assert sha(model)==digest
    source=torch.load(model,map_location='cpu',weights_only=False)
    assert source['iterations']==start and not source['partial_update']
    assert source['primary_updates']==source['encoder_updates']==start*20
    for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in source[key]['state'].values()}=={start*20}
    baseline=json.loads((source_run/'effective-training-config.json').read_text())
    config=json.loads(json.dumps(baseline));config['commands']['yaw_range']=[-10.,10.]
    config['domain_rand'].setdefault('ground_friction_range',None)  # 旧3000缺此字段，None保持原摩擦混合公式。
    assert baseline['commands']['yaw_range']==[-2.,2.]
    log=ROOT/f'logs/mjlab/fudan_ppo_walk/20261116_high_yaw10_resume{start}_to{start+1000}_01'
    assert not log.exists()
    with ExitStack() as stack:
        for section,values in config.items():
            for key,value in values.items():stack.enter_context(patch.object(getattr(FudanWalkCfg,section),key,value,create=True))
        cfg=FudanWalkCfg();actual=json.loads(json.dumps({name:section_dict(getattr(cfg,name)) for name in config}))
        assert actual==config and timing_record(cfg)==source['control_timing']
        sampling=command_config(cfg);assert sampling['yaw_range']==[-10.,10.] and sampling['zero_vx_above_yaw']==2.
    manifest=json.loads((source_run/'source_sha256.json').read_text())
    for name,value in manifest.items():
        if name.startswith('resources/'):assert sha(ROOT/name)==value,name
    abandoned=source['transitions']-start*4096*48
    assert abandoned=={3000:0,4000:184320}[start]
    out.mkdir(parents=True,exist_ok=False)
    write(out/'budget.json',dict(authorization='User authorized1000 new training iterations after setting yaw+/-10 and |yaw|>2 =>vx0; source recipe selected by user',
        checkpoint=str(model),checkpoint_sha256=digest,source_run=str(source_run),start_iteration=start,target_iteration=start+1000,
        additional_iterations=1000,num_envs=4096,rollout_steps=48,seed=1,log_dir=str(log),actual_config=config,
        command_sampling=sampling,control_timing=source['control_timing'],disturbances=source['disturbances'],
        height_reward=source['height_reward'],physical_wheel_model=source['physical_wheel_model'],xml_sha256=sha(XML),
        inherited_abandoned_rollout_transitions=abandoned,only_config_change={'commands.yaw_range':[[-2.,2.],[-10.,10.]]},
        command_rule='For sampled training commands only, |yaw|>2 =>vx0; low-yaw mixed motion and parking retained',created_unix=time.time()))
    shutil.copy2(__file__,out/'launcher.py');shutil.copy2(source_run/'effective-training-config.json',out/'source-effective-config.json')
    write(out/'source-identity.json',dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=True))
    write(out/'status.json',dict(status='prepared',training_started=False))


def worker(out,stage):
    import torch
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg,section_dict,timing_record
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.commands import command_config
    from scripts.agent.fudan_ppo_walk.check_ground_friction import material_record
    budget=json.loads((out/'budget.json').read_text());model=Path(budget['checkpoint']);log=Path(budget['log_dir'])
    assert sha(model)==budget['checkpoint_sha256'] and sha(XML)==budget['xml_sha256'];torch.set_num_threads(1)
    source=torch.load(model,map_location='cpu',weights_only=False);original=FudanEnv.__init__;proof=[]
    def checked_init(env,*args,**kwargs):
        original(env,*args,**kwargs);a=env.action;sim=env.env.sim;state=a.disturbances
        assert (a.cfg.leg_stiffness,a.cfg.leg_damping,a.cfg.wheel_stiffness,a.cfg.wheel_damping)==(20.,1.,.1,0.)
        assert timing_record(env.cfg)==budget['control_timing']==dict(physics_dt=.001,pd_dt=.005,policy_dt=.01)
        assert env.disturbance_config.record()==budget['disturbances']
        assert env.wheel_physics_profile==budget['physical_wheel_model'] and not a.wheel_motor.enabled
        assert command_config(env.cfg)==env.command_sampling==budget['command_sampling']
        for attr,key in [('kp_scale','kp_range'),('kd_scale','kd_range'),('motor_strength','motor_strength_range'),('nominal_offset','nominal_offset_range')]:
            value=getattr(a,attr);low,high=budget['actual_config']['domain_rand'][key]
            assert value.min()>=low-1e-7 and value.max()<=high+1e-7,attr
        resample=env.resample
        def checked_resample(ids):
            resample(ids)
            high=env.commands[ids,1].abs()>2.
            assert env.commands[ids[high],0].eq(0).all() and env.commands[ids,1].abs().le(10.).all()
            if len(ids) and not (out/(stage+'-command-rule.json')).exists():
                write(out/(stage+'-command-rule.json'),dict(passed=True,num_envs=len(ids),high_yaw_count=int(high.sum()),
                    high_yaw_vx_all_zero=True,yaw_minmax=[float(env.commands[ids,1].min()),float(env.commands[ids,1].max())]))
        env.resample=checked_resample
        write(out/(stage+'-actual-control.json'),dict(num_envs=env.num_envs,control=source['control'],
            control_timing=timing_record(env.cfg),material=material_record(env),disturbances=env.disturbance_config.record(),
            command_sampling=env.command_sampling,noise_scales=env.cfg.normalization.noise_scales))
        if stage=='precheck':
            cadence=dict(physics_ticks=0,pd_updates=0,held_ticks=0,held_torque_exact=True)
            maximum=[0.];error=[0.];pulses=[0];pulse=[torch.zeros(env.num_envs,3,device=env.device)]
            force=state.force
            def observed_force(quat,scale):
                pulse[0]=force(quat,scale);pulses[0]+=int(pulse[0].ne(0).any(-1).sum());return pulse[0]
            state.force=observed_force
            apply=a.apply_actions;step=sim.step
            def observed_apply():
                if not cadence['physics_ticks'] and env.disturbance_config.pushes:state.ticks.fill_(state.push_interval-1)
                tick=a._pd_tick;before=a.torques.clone();apply();cadence['physics_ticks']+=1
                cadence['pd_updates']+=int(tick==0);cadence['held_ticks']+=int(tick!=0)
                if tick!=0:cadence['held_torque_exact'] &= bool(torch.equal(before,a.torques))
            def observed_step(*args,**kwargs):
                expected=state.wrench(sim.data.qpos[:,3:7]);expected[:,:3]+=pulse[0]
                error[0]=max(error[0],float((sim.data.xfrc_applied[:,a._base_id]-expected).abs().max()))
                maximum[0]=max(maximum[0],float(sim.data.xfrc_applied.abs().max()))
                return step(*args,**kwargs)
            a.apply_actions=observed_apply;sim.step=observed_step
            proof.append(lambda:dict(passed=error[0]<1e-6 and cadence['held_torque_exact'],PD_cadence=cadence,
                external_formula_max_error=error[0],world_wrench_peak=maximum[0],push_observations=pulses[0],optimizer_updates=0))
    with ExitStack() as stack:
        for section,values in budget['actual_config'].items():
            for key,value in values.items():stack.enter_context(patch.object(getattr(FudanWalkCfg,section),key,value,create=True))
        stack.enter_context(patch.object(FudanEnv,'__init__',checked_init))
        if stage=='precheck':
            from scripts.agent.fudan_ppo_walk import precheck
            with patch.object(sys,'argv',['precheck','--output',str(out/'precheck'),'--deadline',str(time.time()+240),
                '--viewer','viser','--port','8805','--history-version','v3','--training-resets','--training-parameters']):precheck.main()
            result=proof[0]();assert result['passed']
            assert result['PD_cadence']['held_ticks']==4*result['PD_cadence']['pd_updates']
            report=json.loads((out/'precheck/report.json').read_text())
            assert report['passed'] and report['command_sampling']==budget['command_sampling']
            assert report['training_randomization']==budget['actual_config']['domain_rand']
            if budget['disturbances']['pushes']:assert result['push_observations']>=20
            else:assert result['world_wrench_peak']==0.
            write(out/'precheck-runtime-audit.json',result)
        else:
            from ct_mjlab.scripts import train
            from ct_mjlab.rsl_rl.runners import fudan_on_policy_runner as runner
            old_save=runner.FudanRunner.save;old_snapshot=runner.snapshot
            def save(learner,name):
                path=old_save(learner,name)
                if name=='model_initial.pt':
                    initial=torch.load(path,map_location='cpu',weights_only=False)
                    for key in ('model_state_dict','optimizer_state_dict','encoder_optimizer_state_dict','torch_rng','cuda_rng',
                        'iterations','transitions','primary_updates','encoder_updates','speed_limit','policy_steps'):
                        assert same(initial[key],source[key]),key
                    assert initial['command_sampling']==budget['command_sampling']
                    write(out/'initialization-audit.json',dict(passed=True,source_iteration=source['iterations'],
                        source_checkpoint_sha256=sha(model),learner_parameters_exact=True,both_optimizers_exact=True,
                        random_states_exact=True,counters_and_curriculum_exact=True,physical_environment_reinitialized=True))
                return path
            def snapshot(path):
                old_snapshot(path);path=Path(path);shutil.copy2(__file__,path/'continuation-launcher.py')
                write(path/'effective-training-config.json',budget['actual_config']);shutil.copy2(out/'budget.json',path/'authorized-budget.json')
                target=path/'source/resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml';shutil.copy2(XML,target)
            with patch.object(runner,'snapshot',snapshot),patch.object(runner.FudanRunner,'save',save):
                train.main(['--task','fudan_ppo_walk','--num-envs','4096','--iterations','1000','--seed','1',
                    '--resume',str(model),'--precheck',str(out/'precheck/report.json'),'--log-dir',str(log)])


def audit(out):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    b=json.loads((out/'budget.json').read_text());log=Path(b['log_dir'])
    source=torch.load(b['checkpoint'],map_location='cpu',weights_only=False)
    final=torch.load(log/'model_final.pt',map_location='cpu',weights_only=False);report=json.loads((log/'training_report.json').read_text())
    assert json.loads((out/'initialization-audit.json').read_text())['passed']
    assert report['stopped_by']=='iteration_budget' and report['iterations_added']==1000
    assert final['iterations']==b['target_iteration'] and not final['partial_update']
    assert final['transitions']-source['transitions']==1000*4096*48
    assert final['primary_updates']==final['encoder_updates']==b['target_iteration']*20
    assert final['command_sampling']==b['command_sampling']
    for key in ('control','control_timing','height_reward','disturbances','physical_wheel_model'):assert final[key]==source[key],key
    assert all(torch.isfinite(value).all() for value in final['model_state_dict'].values())
    events=EventAccumulator(str(log),size_guidance={'scalars':0}).Reload()
    for tag in ['Train/reward',*['Rewards/'+name for name in final['reward_names']]]:
        assert [v.step for v in events.Scalars(tag)]==list(range(b['start_iteration']+1,b['target_iteration']+1)),tag
    manifest=json.loads((log/'source_sha256.json').read_text());drift=[]
    for name,digest in manifest.items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            assert sha(log/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/'))==digest,name
            if sha(ROOT/name)!=digest:drift.append(name)
    write(out/'training-audit.json',dict(budget_and_learning_state_passed=True,saved_source_snapshot_passed=True,
        current_workspace_source_unchanged=not drift,current_workspace_drift=drift,final_iteration=final['iterations'],
        additional_iterations=1000,checkpoint_sha256=sha(log/'model_final.pt'),behavior_accepted=False))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','precheck','train','run'))
    p.add_argument('--source-iteration',type=int,choices=(3000,4000));p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if args.mode=='prepare':
        if args.source_iteration is None:p.error('Explicit source iteration required')
        prepare(out,args.source_iteration);return
    if args.mode in ('precheck','train'):worker(out,args.mode);return
    try:
        assert not Path(json.loads((out/'budget.json').read_text())['log_dir']).exists()
        if not (out/'precheck-runtime-audit.json').exists():run_command(out,'precheck',[str(PYTHON),'-u','-m',MODULE,'precheck','--output',str(out)])
        assert json.loads((out/'precheck-runtime-audit.json').read_text())['passed']
        run_command(out,'train',[str(PYTHON),'-u','-m',MODULE,'train','--output',str(out)],out/'process-result.json')
        audit(out);b=json.loads((out/'budget.json').read_text())
        write(out/'status.json',dict(status='training_complete',final_iteration=b['target_iteration'],iterations_added=1000,behavior_accepted=False))
    except BaseException as exc:
        write(out/'status.json',dict(status='failed',error=repr(exc),automatic_retry=False));raise


if __name__=='__main__':main()
