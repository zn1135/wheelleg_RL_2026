"""Authorized full-randomization continuation: saved3000 -> total4000, direct ground friction."""
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
SOURCE_RUN=ROOT/'logs/mjlab/fudan_ppo_walk/20261114_pd200_p20d1_half_random_delay_wide_resume1000_to3000_01'
MODEL=SOURCE_RUN/'model_final.pt'
LOG=ROOT/'logs/mjlab/fudan_ppo_walk/20261115_pd200_full_random_ground04to12_force3_push_resume3000_to4000_01'
MODULE='scripts.agent.fudan_ppo_walk.resume_pd200_random_from3000'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def same(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def prepare(output):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg,section_dict,timing_record
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config
    from ct_mjlab.envs.fudan_ppo_walk.commands import command_config
    from ct_mjlab.envs.fudan_ppo_walk import posture_reward,wheel_physics,height_reward
    source=torch.load(MODEL,map_location='cpu',weights_only=False)
    assert source['iterations']==3000 and source['transitions']==3000*4096*48 and not source['partial_update']
    assert source['primary_updates']==source['encoder_updates']==60000
    assert not LOG.exists()
    baseline=json.loads((SOURCE_RUN/'effective-training-config.json').read_text())
    config=json.loads(json.dumps(baseline))
    # 恢复原全幅并开启3N外力/短推扰；双延迟和其余配方继承3000基线。
    ranges=dict(kp_range=[.85,1.15],kd_range=[.85,1.15],motor_strength_range=[.85,1.15],
        nominal_offset_range=[-.05,.05],added_mass_range=[-1.5,3.],mass_scale_range=[.85,1.15],
        com_range=[-.03,.03],restitution_range=[.4,1.],ground_friction_range=[.4,1.2],external_force_range=[-3.,3.],push_robots=True)
    config['domain_rand'].update(ranges)
    config['normalization']['noise_scales']=[value*2 for value in baseline['normalization']['noise_scales']]
    changes={f'domain_rand.{key}':[baseline['domain_rand'].get(key),value] for key,value in ranges.items()}
    changes['normalization.noise_scales']=[baseline['normalization']['noise_scales'],config['normalization']['noise_scales']]
    assert config['domain_rand']['delay_ms_range']==config['domain_rand']['sensor_delay_ms_range']==[10.,20.]
    assert sha(MODEL)=='dfb221c09648d278d5c5e5c7b78e4d729b7a7a5a56458762b7313dc50831c4f8'
    assert source['control']==config['control']
    hashes=json.loads((SOURCE_RUN/'source_sha256.json').read_text());assert sha(XML)==hashes[str(XML.relative_to(ROOT))]
    with ExitStack() as stack:
        for section,values in config.items():
            for key,value in values.items():stack.enter_context(patch.object(getattr(FudanWalkCfg,section),key,value,create=True))
        cfg=FudanWalkCfg();effective=json.loads(json.dumps({name:section_dict(getattr(cfg,name)) for name in config}))
        assert effective==config
        assert timing_record(cfg)==source['control_timing']==dict(physics_dt=.001,pd_dt=.005,policy_dt=.01)
        disturbances=from_task_config(cfg)
        assert disturbances.pushes and disturbances.external_force_range==(-3.,3.) and disturbances.external_torque_range==(0.,0.)
        assert height_reward.from_task_config(cfg)==source['height_reward']==dict(version='fudan-height-exp-scaled-v1',scale=1.,sigma=.001)
        output.mkdir(parents=True,exist_ok=False)
        write(output/'budget.json',dict(authorization='User authorized saved3000 plus1000 iterations, full parameter/noise randomization, actual contact friction0.4to1.2, body-axis episode force+/-3N and7s/0.3mps single-tick pushes; torque0',
            checkpoint=str(MODEL),checkpoint_sha256=sha(MODEL),start_iteration=3000,target_iteration=4000,iteration_budget=1000,
            additional_iterations=1000,num_envs=4096,rollout_steps=48,seed=1,log_dir=str(LOG),history_version='v3',
            disturbances=disturbances.record(),command_sampling=command_config(cfg),posture_reward=posture_reward.from_task_config(cfg),
            actual_config=effective,physical_wheel_model=wheel_physics.record('mechanical'),height_reward=height_reward.from_task_config(cfg),
            control_timing=timing_record(cfg),xml_sha256=sha(XML),source_config_sha256=sha(SOURCE_RUN/'effective-training-config.json'),
            only_active_config_changes=changes,
            initialization='full_learning_state_resume_fresh_physics_history',created_unix=time.time()))
    write(output/'recipe-diff.json',changes)
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
        for name,key in [('com','com_range'),('friction','ground_friction_range'),('restitution','restitution_range')]:
            values=env.env._fudan_parameters[name];low,high=budget['actual_config']['domain_rand'][key]
            assert values.min()>=low-1e-7 and values.max()<=high+1e-7,name
            parameter_ranges[name]=[float(values.min()),float(values.max())]
        body_ids=env.robot.indexing.body_ids;local=env.robot.body_names.index('base_link')
        default_mass=sim.get_default_field('body_mass')[body_ids]
        added=env.env._fudan_parameters['base_mass']-default_mass[local]
        low,high=env.cfg.domain_rand.added_mass_range
        assert added.min()>=low-1e-6 and added.max()<=high+1e-6
        denominator=default_mass.expand(env.num_envs,-1).clone()
        denominator[:,local]=env.env._fudan_parameters['base_mass']
        positive=default_mass>0
        scale=sim.model.body_mass[:,body_ids][:,positive]/denominator[:,positive]
        low,high=env.cfg.domain_rand.mass_scale_range
        assert scale.min()>=low-1e-6 and scale.max()<=high+1e-6
        assert sim.model.body_mass[:,body_ids].ge(0).all()
        assert torch.allclose(sim.model.body_inertia[:,body_ids][:,positive],
            sim.get_default_field('body_inertia')[body_ids][positive][None]*scale[...,None],atol=1e-6)
        parameter_ranges['added_mass_kg']=[float(added.min()),float(added.max())]
        parameter_ranges['mass_scale']=[float(scale.min()),float(scale.max())]
        from scripts.agent.fudan_ppo_walk.check_ground_friction import material_record
        parameter_ranges['contact_material']=material_record(env)
        state=a.disturbances
        assert env.disturbance_config.pushes and state.has_episode_wrench
        assert env.disturbance_config.push_interval_s==7. and env.disturbance_config.max_push_vel==.3
        assert state.episode_wrench[:,:3].abs().le(3.).all() and state.episode_wrench[:,3:].eq(0).all()
        assert state.push_count.eq(0).all()
        assert env.disturbance_config.external_force_range==(-3.,3.) and env.disturbance_config.external_torque_range==(0.,0.)
        if stage=='precheck':
            original_step=sim.step;maximum=torch.zeros(2,device=env.device);count=[0]
            cadence=dict(physics_ticks=0,pd_updates=0,held_ticks=0,held_torque_exact=True)
            contacts=[];wrench_error=[0.];body_force_max=[0.]
            push_output=[torch.zeros(env.num_envs,3,device=env.device)]
            push_events=[0];previous_push=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
            original_force=state.force
            def observed_force(quat,scale):
                pulse=original_force(quat,scale);active=pulse.ne(0).any(-1)
                assert not (active & previous_push).any()  # 短推扰只持续一个积分步。
                assert state.ticks[active].remainder(state.push_interval).eq(0).all()
                previous_push.copy_(active);push_events[0]+=int(active.sum())
                push_output[0]=pulse
                return pulse
            state.force=observed_force
            apply_actions=a.apply_actions
            def observed_actions():
                # 推到真实7秒触发边界，确保倒地复位也不会漏掉推扰施力检查。
                if cadence['physics_ticks']==0:state.ticks.fill_(state.push_interval-1)
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
                expected=state.wrench(sim.data.qpos[:,3:7])
                expected[:,:3]+=push_output[0]
                wrench_error[0]=max(wrench_error[0],float((external[:,a._base_id]-expected).abs().max()))
                body_force_max[0]=max(body_force_max[0],float(state.episode_wrench[:,:3].abs().max()))
                assert state.episode_wrench[:,:3].abs().le(3.).all() and state.episode_wrench[:,3:].eq(0).all()
                count[0]+=1
                result=original_step(*args,**kwargs)
                if count[0]%100==0:
                    from scripts.agent.fudan_ppo_walk.check_ground_friction import contact_record
                    contacts.append(contact_record(env))
                return result
            sim.step=observed_step
            force_records.append(lambda:(maximum.cpu().tolist(),count[0],cadence,contacts,wrench_error[0],body_force_max[0],push_events[0]))
        for name in ('armature','damping','frictionloss'):
            values=getattr(sim.model,'dof_'+name)
            if values.ndim==1:values=values.unsqueeze(0)
            expected=torch.tensor(budget['physical_wheel_model'][name],device=env.device)
            assert torch.allclose(values[:,dofs],expected.expand_as(values[:,dofs]),atol=1e-7,rtol=0),name
        write(output/(stage+'-force3-actual-control.json'),dict(num_envs=env.num_envs,leg_p=20.,leg_d=1.,wheel_p=.1,wheel_d=0.,
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
            with patch.object(sys,'argv',['precheck','--output',str(output/'precheck-force3-push'),'--deadline',str(time.time()+180),
                '--viewer','viser','--port','8799','--history-version','v3','--training-resets','--training-parameters']):precheck.main()
            maximum,count,cadence,contacts,wrench_error,body_force_max,push_events=force_records[0]()
            proof=json.loads((output/'precheck-force3-push/report.json').read_text())
            assert proof['control_timing']==budget['control_timing']
            assert proof['training_randomization']==budget['actual_config']['domain_rand']
            assert proof['noise_scales']==budget['actual_config']['normalization']['noise_scales']
            assert cadence['held_torque_exact'] and cadence['held_ticks']==4*cadence['pd_updates']
            assert cadence['physics_ticks']==count
            assert maximum[0]>0 and maximum[1]==0. and count>=7020 and push_events>=20
            assert wrench_error<1e-6 and 0<body_force_max<=3.
            assert sum(row['wheel_ground_contacts'] for row in contacts)>0
            write(output/'precheck-force3-push-contact-friction.json',dict(passed=True,observations=contacts,optimizer_updates=0))
            write(output/'precheck-force3-push-external.json',dict(passed=True,observed_GPU_integration_steps=count,
                body_axis_force_bound_N=3.,observed_body_axis_force_max_N=body_force_max,world_wrench_formula_max_error=wrench_error,
                maximum_external_force_N=maximum[0],maximum_external_torque_Nm=maximum[1],
                push_count=proof['push_count'],push_event_observations=push_events,forced_first_boundary=True,
                single_physics_tick_push=True,PD_cadence=cadence,control_timing=proof['control_timing'],optimizer_updates=0))
        else:
            from ct_mjlab.scripts import train
            from ct_mjlab.rsl_rl.runners import fudan_on_policy_runner as runner
            old_snapshot=runner.snapshot
            old_save=runner.FudanRunner.save
            def checked_save(learner,name):
                path=old_save(learner,name)
                if name=='model_initial.pt':
                    initial=torch.load(path,map_location='cpu',weights_only=False)
                    assert initial['iterations']==source['iterations']==budget['start_iteration']
                    for key in ('transitions','primary_updates','encoder_updates','speed_limit','torch_rng','cuda_rng',
                                'model_state_dict','optimizer_state_dict','encoder_optimizer_state_dict'):
                        assert same(initial[key],source[key]),key
                    assert initial['initialization']=='resumed_learning_state_fresh_environment'
                    assert initial['resume_from']['sha256']==sha(MODEL)
                    assert initial['control_timing']==source['control_timing']==budget['control_timing']
                    assert initial['control']==source['control']==budget['actual_config']['control']
                    assert initial['height_reward']==source['height_reward']==budget['height_reward']
                    assert initial['disturbances']==budget['disturbances']
                    write(output/'initialization-audit.json',dict(passed=True,source_iteration=source['iterations'],
                        learner_parameters_exact=True,both_optimizers_exact=True,random_states_exact=True,
                        source_checkpoint_sha256=sha(MODEL),control_timing=initial['control_timing'],
                        physical_environment_reinitialized=True,checkpoint_sha256=sha(path)))
                return path
            def snapshot(path):
                old_snapshot(path);path=Path(path)
                shutil.copy2(__file__,path/'continuation-launcher.py')
                helper=ROOT/'scripts/agent/fudan_ppo_walk/check_ground_friction.py'
                shutil.copy2(helper,path/'check_ground_friction.py')
                write(path/'launcher-sha256.json',{str(file.relative_to(ROOT)):sha(file) for file in (Path(__file__),helper)})
                write(path/'effective-training-config.json',actual)
                shutil.copy2(output/'budget.json',path/'authorized-budget.json')
                target=path/'source/resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
                shutil.copy2(XML,target)
            with patch.object(runner,'snapshot',snapshot),patch.object(runner.FudanRunner,'save',checked_save):
                train.main(['--task','fudan_ppo_walk','--num-envs','4096','--iterations',str(budget['additional_iterations']),'--seed','1','--resume',str(MODEL),
                    '--precheck',str(output/'precheck-force3-push/report.json'),'--log-dir',str(LOG)])


def audit(output):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    budget=json.loads((output/'budget.json').read_text());source=torch.load(MODEL,map_location='cpu',weights_only=False)
    initial=torch.load(LOG/'model_initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(LOG/'model_final.pt',map_location='cpu',weights_only=False);report=json.loads((LOG/'training_report.json').read_text())
    assert json.loads((output/'process-result.json').read_text())['returncode']==0
    assert json.loads((output/'initialization-audit.json').read_text())['passed']
    assert initial['iterations']==3000 and final['iterations']==4000 and not final['partial_update']
    assert report['stopped_by']=='iteration_budget' and report['iterations_added']==report['iteration_budget']==1000
    assert final['transitions']-source['transitions']==1000*4096*48
    assert final['primary_updates']-source['primary_updates']==final['encoder_updates']-source['encoder_updates']==20000
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
        assert [v.step for v in events.Scalars(tag)]==list(range(3001,4001)),tag
    assert all(torch.isfinite(value).all() for value in final['model_state_dict'].values())
    for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in final[key]['state'].values()}=={80000}
    manifest=json.loads((LOG/'source_sha256.json').read_text())
    for name,digest in manifest.items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            assert sha(LOG/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/'))==digest,name
            assert sha(ROOT/name)==digest,name
    for name,digest in json.loads((LOG/'launcher-sha256.json').read_text()).items():assert sha(ROOT/name)==digest,name
    write(output/'training-audit.json',dict(passed=True,source_iteration=3000,final_iteration=4000,additional_iterations=1000,
        transitions_added=1000*4096*48,optimizer_updates_added_each=20000,checkpoint_sha256=sha(LOG/'model_final.pt'),behavior_accepted=False))


def run(output):
    try:
        assert not LOG.exists(), 'This run already exists; no automatic restart or extra learning'
        assert json.loads((output/'friction-check/report.json').read_text())['passed']
        proof=output/'precheck-force3-push-external.json'
        if not proof.exists():run_command(output,'precheck',[str(PYTHON),'-u','-m',MODULE,'precheck','--output',str(output)])
        assert json.loads(proof.read_text())['passed']
        run_command(output,'train',[str(PYTHON),'-u','-m',MODULE,'train','--output',str(output)],output/'process-result.json')
        audit(output)
        write(output/'status.json',dict(status='training_complete',finished_unix=time.time(),source_iteration=3000,
            final_iteration=4000,iterations_added=1000,checkpoint=str(LOG/'model_final.pt'),checkpoint_sha256=sha(LOG/'model_final.pt'),
            behavior_accepted=False,next_step='Matched baseline47 and randomized-friction GPU comparison to source3000; inspect wheel support, leg motion and drift; no extra learning'))
    except BaseException as exc:
        write(output/'status.json',dict(status='failed',error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','run','precheck','train'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();output=args.output.resolve()
    if args.mode=='prepare':
        prepare(output)
    elif args.mode=='run':run(output)
    else:worker(output,args.mode)


if __name__=='__main__':main()
