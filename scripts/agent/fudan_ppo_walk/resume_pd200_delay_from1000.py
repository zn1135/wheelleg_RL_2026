"""Authorized wider-delay continuation: saved1000 -> total3000, all other recipe fields fixed."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

from scripts.agent.fudan_ppo_walk.run_torque_ablation import run_command

ROOT=Path(__file__).resolve().parents[3]
PYTHON=ROOT/'mjlab_training/.venv/bin/python'
XML=ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
SOURCE_RUN=ROOT/'logs/mjlab/fudan_ppo_walk/20261114_pd200_p20d1_half_random_height1_scratch3000_01'
MODEL=SOURCE_RUN/'model_1000.pt'
LOG=ROOT/'logs/mjlab/fudan_ppo_walk/20261114_pd200_p20d1_half_random_delay_wide_resume1000_to3000_01'
MODULE='scripts.agent.fudan_ppo_walk.resume_pd200_delay_from1000'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def same(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def prepare(output,action_delay,sensor_delay):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg,section_dict,timing_record
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config
    from ct_mjlab.envs.fudan_ppo_walk.commands import command_config
    from ct_mjlab.envs.fudan_ppo_walk import posture_reward,wheel_physics,height_reward
    source=torch.load(MODEL,map_location='cpu',weights_only=False)
    assert source['iterations']==1000 and source['transitions']==1000*4096*48 and not source['partial_update']
    assert source['primary_updates']==source['encoder_updates']==20000
    assert not LOG.exists()
    baseline=json.loads((SOURCE_RUN/'effective-training-config.json').read_text())
    config=json.loads(json.dumps(baseline));config['domain_rand']['delay_ms_range']=action_delay
    config['domain_rand']['sensor_delay_ms_range']=sensor_delay
    assert source['control']==config['control']
    hashes=json.loads((SOURCE_RUN/'source_sha256.json').read_text());assert sha(XML)==hashes[str(XML.relative_to(ROOT))]
    with ExitStack() as stack:
        for section,values in config.items():
            for key,value in values.items():stack.enter_context(patch.object(getattr(FudanWalkCfg,section),key,value,create=True))
        cfg=FudanWalkCfg();effective=json.loads(json.dumps({name:section_dict(getattr(cfg,name)) for name in config}))
        assert effective==config
        assert timing_record(cfg)==source['control_timing']==dict(physics_dt=.001,pd_dt=.005,policy_dt=.01)
        disturbances=from_task_config(cfg)
        assert not disturbances.pushes and disturbances.external_force_range==disturbances.external_torque_range==(0.,0.)
        assert height_reward.from_task_config(cfg)==source['height_reward']==dict(version='fudan-height-exp-scaled-v1',scale=1.,sigma=.001)
        output.mkdir(parents=True,exist_ok=False)
        write(output/'budget.json',dict(authorization='User requested wider action and sensor delays and continuation from saved1000; keep existing total3000 target, add2000 iterations',
            checkpoint=str(MODEL),checkpoint_sha256=sha(MODEL),start_iteration=1000,target_iteration=3000,iteration_budget=2000,
            additional_iterations=2000,num_envs=4096,rollout_steps=48,seed=1,log_dir=str(LOG),history_version='v3',
            disturbances=disturbances.record(),command_sampling=command_config(cfg),posture_reward=posture_reward.from_task_config(cfg),
            actual_config=effective,physical_wheel_model=wheel_physics.record('mechanical'),height_reward=height_reward.from_task_config(cfg),
            control_timing=timing_record(cfg),xml_sha256=sha(XML),source_config_sha256=sha(SOURCE_RUN/'effective-training-config.json'),
            only_active_config_changes={'domain_rand.delay_ms_range':[baseline['domain_rand']['delay_ms_range'],action_delay],
                'domain_rand.sensor_delay_ms_range':[baseline['domain_rand']['sensor_delay_ms_range'],sensor_delay]},
            initialization='full_learning_state_resume_fresh_physics_history',created_unix=time.time()))
    shutil.copy2(__file__,output/'launcher.py');shutil.copy2(XML,output/'training-model.xml')
    shutil.copy2(SOURCE_RUN/'effective-training-config.json',output/'source-effective-config.json')
    shutil.copy2(wheel_physics.PARAMETERS,output/'wheel_mechanical.json')
    write(output/'source-identity.json',dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=True))
    (output/'training-workspace.patch').write_bytes(subprocess.check_output(['git','diff','--binary','--','mjlab_training/','scripts/agent/fudan_ppo_walk/'],cwd=ROOT))
    write(output/'status.json',dict(status='prepared',training_started=False))


def worker(output,stage):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg,section_dict,timing_record
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.envs.fudan_ppo_walk import wheel_physics,height_reward
    budget=json.loads((output/'budget.json').read_text());torch.set_num_threads(1)
    assert sha(MODEL)==budget['checkpoint_sha256']
    source=torch.load(MODEL,map_location='cpu',weights_only=False)
    assert sha(XML)==budget['xml_sha256']
    assert wheel_physics.record('mechanical')==budget['physical_wheel_model']
    original_init=FudanEnv.__init__
    force_records=[]
    def checked_init(env,*args,**kwargs):
        original_init(env,*args,**kwargs)
        a=env.action;sim=env.env.sim;ids=a._wheel_action_ids;dofs=a._motor_v_adr[ids]
        legs=sim.model.dof_armature[:,a._motor_v_adr[[0,1,3,4]]]
        assert torch.allclose(legs,torch.full_like(legs,.03159),atol=1e-8,rtol=0)
        assert (a.cfg.leg_stiffness,a.cfg.leg_damping,a.cfg.wheel_stiffness,a.cfg.wheel_damping)==(20.,1.,.1,0.)
        assert not a._wheel_speed_shaping and not a.wheel_motor.enabled and a.wheel_motor.gain.eq(1.).all()
        assert env.wheel_physics_profile==budget['physical_wheel_model']
        assert timing_record(env.cfg)==budget['control_timing']
        assert sim.mj_model.opt.timestep==.001 and a.cfg.pd_decimation==5
        for name,key in [('action_steps','delay_ms_range'),('sensor_steps','sensor_delay_ms_range')]:
            low,high=budget['actual_config']['domain_rand'][key]
            values=getattr(a.disturbances,name)
            assert values.min()>=round(low) and values.max()<=round(high),name
        assert height_reward.from_task_config(env.cfg)==budget['height_reward']==dict(version='fudan-height-exp-scaled-v1',scale=1.,sigma=.001)
        parameter_ranges={}
        for name,key in [('kp_scale','kp_range'),('kd_scale','kd_range'),('motor_strength','motor_strength_range'),('nominal_offset','nominal_offset_range')]:
            values=getattr(a,name);low,high=budget['actual_config']['domain_rand'][key]
            assert values.min()>=low-1e-7 and values.max()<=high+1e-7,name
            parameter_ranges[name]=[float(values.min()),float(values.max())]
        for name,key in [('com','com_range'),('friction','friction_range'),('restitution','restitution_range')]:
            values=env.env._fudan_parameters[name];low,high=budget['actual_config']['domain_rand'][key]
            assert values.min()>=low-1e-7 and values.max()<=high+1e-7,name
            parameter_ranges[name]=[float(values.min()),float(values.max())]
        state=a.disturbances
        assert not env.disturbance_config.pushes and not state.has_episode_wrench
        assert state.episode_wrench.eq(0).all() and state.push_count.eq(0).all()
        assert env.disturbance_config.external_force_range==env.disturbance_config.external_torque_range==(0.,0.)
        if stage=='precheck':
            # 核对关闭推扰后，即使到达原触发边界也不会产生推力。
            ticks=state.ticks.clone();state.ticks.fill_(state.push_interval-1)
            pulse=state.force(sim.data.qpos[:,3:7],a.push_force_scale)
            assert pulse.eq(0).all() and state.push_count.eq(0).all();state.ticks.copy_(ticks)
            original_step=sim.step;maximum=torch.zeros(2,device=env.device);count=[0]
            cadence=dict(physics_ticks=0,pd_updates=0,held_ticks=0,held_torque_exact=True)
            apply_actions=a.apply_actions
            def observed_actions():
                tick=a._pd_tick;before=a.torques.clone()
                apply_actions()
                cadence['physics_ticks']+=1
                cadence['pd_updates']+=int(tick==0)
                cadence['held_ticks']+=int(tick!=0)
                if tick!=0:cadence['held_torque_exact'] &= bool(torch.equal(before,a.torques))
            a.apply_actions=observed_actions
            def observed_step(*args,**kwargs):
                nonlocal maximum
                external=sim.data.xfrc_applied
                maximum=torch.maximum(maximum,torch.stack((external[:,:,:3].abs().max(),external[:,:,3:].abs().max())))
                count[0]+=1
                return original_step(*args,**kwargs)
            sim.step=observed_step
            force_records.append(lambda:(maximum.cpu().tolist(),count[0],cadence))
        for name in ('armature','damping','frictionloss'):
            values=getattr(sim.model,'dof_'+name)
            if values.ndim==1:values=values.unsqueeze(0)
            expected=torch.tensor(budget['physical_wheel_model'][name],device=env.device)
            assert torch.allclose(values[:,dofs],expected.expand_as(values[:,dofs]),atol=1e-7,rtol=0),name
        write(output/(stage+'-actual-control.json'),dict(num_envs=env.num_envs,leg_p=20.,leg_d=1.,wheel_p=.1,wheel_d=0.,
            parameter_ranges=parameter_ranges,noise_scales=env.cfg.normalization.noise_scales,
            leg_armature_unique=torch.unique(legs).cpu().tolist(),physical_wheel_model=env.wheel_physics_profile,
            speed_curve=False,extra_output_layer=False,control_timing=timing_record(env.cfg),
            action_delay_steps_minmax=[int(a.disturbances.action_steps.min()),int(a.disturbances.action_steps.max())],
            sensor_delay_steps_minmax=[int(a.disturbances.sensor_steps.min()),int(a.disturbances.sensor_steps.max())],disturbances=env.disturbance_config.record(),height_reward=height_reward.from_task_config(env.cfg)))
    with ExitStack() as stack:
        for section,values in budget['actual_config'].items():
            for key,value in values.items():stack.enter_context(patch.object(getattr(FudanWalkCfg,section),key,value,create=True))
        stack.enter_context(patch.object(FudanEnv,'__init__',checked_init))
        actual=json.loads(json.dumps({name:section_dict(getattr(FudanWalkCfg(),name)) for name in budget['actual_config']}))
        assert actual==budget['actual_config']
        if stage=='precheck':
            from scripts.agent.fudan_ppo_walk import precheck
            with patch.object(sys,'argv',['precheck','--output',str(output/'precheck'),'--deadline',str(time.time()+180),
                '--viewer','viser','--port','8798','--history-version','v3','--training-resets','--training-parameters']):precheck.main()
            maximum,count,cadence=force_records[0]()
            proof=json.loads((output/'precheck/report.json').read_text())
            assert proof['control_timing']==budget['control_timing']
            assert proof['training_randomization']==budget['actual_config']['domain_rand']
            assert proof['noise_scales']==budget['actual_config']['normalization']['noise_scales']
            assert cadence['held_torque_exact'] and cadence['held_ticks']==4*cadence['pd_updates']
            assert cadence['physics_ticks']==count
            assert maximum==[0.,0.] and count>=2000 and not any(proof['push_count'])
            write(output/'precheck-zero-external.json',dict(passed=True,observed_GPU_integration_steps=count,
                maximum_external_force_N=maximum[0],maximum_external_torque_Nm=maximum[1],
                push_count=proof['push_count'],PD_cadence=cadence,control_timing=proof['control_timing'],disabled_push_boundary_probe_passed=True,optimizer_updates=0))
        else:
            from ct_mjlab.scripts import train
            from ct_mjlab.rsl_rl.runners import fudan_on_policy_runner as runner
            old_snapshot=runner.snapshot
            old_save=runner.FudanRunner.save
            def checked_save(learner,name):
                path=old_save(learner,name)
                if name=='model_initial.pt':
                    initial=torch.load(path,map_location='cpu',weights_only=False)
                    assert initial['iterations']==source['iterations']==1000
                    for key in ('transitions','primary_updates','encoder_updates','speed_limit','torch_rng','cuda_rng',
                                'model_state_dict','optimizer_state_dict','encoder_optimizer_state_dict'):
                        assert same(initial[key],source[key]),key
                    assert initial['initialization']=='resumed_learning_state_fresh_environment'
                    assert initial['resume_from']['sha256']==sha(MODEL)
                    assert initial['control_timing']==source['control_timing']==budget['control_timing']
                    assert initial['control']==source['control']==budget['actual_config']['control']
                    assert initial['height_reward']==source['height_reward']==budget['height_reward']
                    assert initial['disturbances']==budget['disturbances']
                    write(output/'initialization-audit.json',dict(passed=True,source_iteration=1000,
                        learner_parameters_exact=True,both_optimizers_exact=True,random_states_exact=True,
                        source_checkpoint_sha256=sha(MODEL),control_timing=initial['control_timing'],
                        physical_environment_reinitialized=True,checkpoint_sha256=sha(path)))
                return path
            def snapshot(path):
                old_snapshot(path);path=Path(path)
                shutil.copy2(__file__,path/'continuation-launcher.py')
                write(path/'effective-training-config.json',actual)
                shutil.copy2(output/'budget.json',path/'authorized-budget.json')
                target=path/'source/resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
                shutil.copy2(XML,target)
            with patch.object(runner,'snapshot',snapshot),patch.object(runner.FudanRunner,'save',checked_save):
                train.main(['--task','fudan_ppo_walk','--num-envs','4096','--iterations','2000','--seed','1','--resume',str(MODEL),
                    '--precheck',str(output/'precheck/report.json'),'--log-dir',str(LOG)])


def audit(output):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    budget=json.loads((output/'budget.json').read_text());source=torch.load(MODEL,map_location='cpu',weights_only=False)
    initial=torch.load(LOG/'model_initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(LOG/'model_final.pt',map_location='cpu',weights_only=False);report=json.loads((LOG/'training_report.json').read_text())
    assert json.loads((output/'process-result.json').read_text())['returncode']==0
    assert json.loads((output/'initialization-audit.json').read_text())['passed']
    assert initial['iterations']==1000 and final['iterations']==3000 and not final['partial_update']
    assert report['stopped_by']=='iteration_budget' and report['iterations_added']==report['iteration_budget']==2000
    assert final['transitions']-source['transitions']==2000*4096*48
    assert final['primary_updates']-source['primary_updates']==final['encoder_updates']-source['encoder_updates']==40000
    for key in ('model_state_dict','optimizer_state_dict','encoder_optimizer_state_dict','torch_rng','cuda_rng'):
        assert same(initial[key],source[key]),key
    assert initial['control']==final['control']==source['control']
    assert initial['control_timing']==final['control_timing']==budget['control_timing']
    assert initial['height_reward']==final['height_reward']==source['height_reward']
    assert initial['disturbances']==final['disturbances']==budget['disturbances']
    assert final['physical_wheel_model']==source['physical_wheel_model']==budget['physical_wheel_model']
    assert final['resume_from']['sha256']==sha(MODEL)
    events=EventAccumulator(str(LOG),size_guidance={'scalars':0}).Reload()
    for tag in ['Train/reward',*['Rewards/'+name for name in final['reward_names']]]:
        assert [v.step for v in events.Scalars(tag)]==list(range(1001,3001)),tag
    assert all(torch.isfinite(value).all() for value in final['model_state_dict'].values())
    for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in final[key]['state'].values()}=={60000}
    manifest=json.loads((LOG/'source_sha256.json').read_text())
    for name,digest in manifest.items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            assert sha(LOG/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/'))==digest,name
    write(output/'training-audit.json',dict(passed=True,source_iteration=1000,final_iteration=3000,additional_iterations=2000,
        transitions_added=2000*4096*48,optimizer_updates_added_each=40000,checkpoint_sha256=sha(LOG/'model_final.pt'),behavior_accepted=False))


def run(output):
    try:
        proof=output/'precheck-zero-external.json'
        if not proof.exists():run_command(output,'precheck',[str(PYTHON),'-u','-m',MODULE,'precheck','--output',str(output)])
        assert json.loads(proof.read_text())['passed']
        run_command(output,'train',[str(PYTHON),'-u','-m',MODULE,'train','--output',str(output)],output/'process-result.json')
        audit(output)
        write(output/'status.json',dict(status='training_complete',finished_unix=time.time(),source_iteration=1000,
            final_iteration=3000,iterations_added=2000,checkpoint=str(LOG/'model_final.pt'),checkpoint_sha256=sha(LOG/'model_final.pt'),
            behavior_accepted=False,next_step='Matched clean-delay and wider-delay GPU parking/leg motion comparison to source1000; no extra learning'))
    except BaseException as exc:
        write(output/'status.json',dict(status='failed',error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','run','precheck','train'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--action-delay-ms',type=float,nargs=2)
    p.add_argument('--sensor-delay-ms',type=float,nargs=2);args=p.parse_args();output=args.output.resolve()
    if args.mode=='prepare':
        if args.action_delay_ms is None or args.sensor_delay_ms is None:p.error('Explicit action and sensor delay bounds required')
        prepare(output,args.action_delay_ms,args.sensor_delay_ms)
    elif args.mode=='run':run(output)
    else:worker(output,args.mode)


if __name__=='__main__':main()
