"""Authorized fresh1500 with height scale5/sigma0.01 and no external loads."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

from scripts.agent.fudan_ppo_walk.run_torque_ablation import run_command

ROOT=Path(__file__).resolve().parents[3]
PYTHON=ROOT/'mjlab_training/.venv/bin/python'
XML=ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
LOG=ROOT/'logs/mjlab/fudan_ppo_walk/20261113_p60d2_wheelP01_mechanical_height5_sigma01_scratch1500_01'
MODULE='scripts.agent.fudan_ppo_walk.train_mechanical_height_wide_1500'
OVERRIDES=(('control','leg_stiffness',60.),('control','leg_damping',2.),
    ('control','wheel_stiffness',.1),('control','wheel_damping',0.),
    ('sim','wheel_model','mechanical'),('sim','wheel_motor_gain',1.))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def prepare(output):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg,section_dict
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config
    from ct_mjlab.envs.fudan_ppo_walk.commands import command_config
    from ct_mjlab.envs.fudan_ppo_walk import posture_reward,wheel_physics,height_reward
    assert not LOG.exists()
    with ExitStack() as stack:
        for section,key,value in OVERRIDES:stack.enter_context(patch.object(getattr(FudanWalkCfg,section),key,value))
        cfg=FudanWalkCfg()
        names=('commands','domain_rand','env','sim','control','normalization','rewards')
        effective=json.loads(json.dumps({name:section_dict(getattr(cfg,name)) for name in names}))
        baseline=ROOT/'logs/mjlab/fudan_ppo_walk/20261113_p60d2_wheelP01_mechanical_noexternal_scratch1500_01/effective-training-config.json'
        expected=json.loads(baseline.read_text())
        expected['rewards']['scales']['base_height']=5.
        expected['rewards']['height_sigma']=.01
        assert effective==expected, 'Only height reward scale/sigma may change from previous no-external recipe'
        output.mkdir(parents=True,exist_ok=False)
        write(output/'budget.json',dict(authorization='User confirmed from-scratch1500 iterations with approved height scale5/sigma0.01, no external loads',
            resume_checkpoint=None,start_iteration=0,target_iteration=1500,iteration_budget=1500,
            additional_iterations=1500,num_envs=4096,rollout_steps=48,seed=1,log_dir=str(LOG),history_version='v3',
            disturbances=from_task_config(cfg).record(),command_sampling=command_config(cfg),
            posture_reward=posture_reward.from_task_config(cfg),actual_config=effective,
            physical_wheel_model=wheel_physics.record('mechanical'),height_reward=height_reward.from_task_config(cfg),xml_sha256=sha(XML),
            initialization='random_from_scratch_seed_1',created_unix=time.time()))
    shutil.copy2(__file__,output/'launcher.py');shutil.copy2(XML,output/'training-model.xml')
    shutil.copy2(wheel_physics.PARAMETERS,output/'wheel_mechanical.json')


def worker(output,stage):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg,section_dict
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.envs.fudan_ppo_walk import wheel_physics,height_reward
    budget=json.loads((output/'budget.json').read_text());torch.set_num_threads(1)
    assert sha(XML)==budget['xml_sha256']
    assert wheel_physics.record('mechanical')==budget['physical_wheel_model']
    original_init=FudanEnv.__init__
    force_records=[]
    def checked_init(env,*args,**kwargs):
        original_init(env,*args,**kwargs)
        a=env.action;sim=env.env.sim;ids=a._wheel_action_ids;dofs=a._motor_v_adr[ids]
        legs=sim.model.dof_armature[:,a._motor_v_adr[[0,1,3,4]]]
        assert torch.allclose(legs,torch.full_like(legs,.03159),atol=1e-8,rtol=0)
        assert (a.cfg.leg_stiffness,a.cfg.leg_damping,a.cfg.wheel_stiffness,a.cfg.wheel_damping)==(60.,2.,.1,0.)
        assert not a._wheel_speed_shaping and not a.wheel_motor.enabled and a.wheel_motor.gain.eq(1.).all()
        assert env.wheel_physics_profile==budget['physical_wheel_model']
        assert height_reward.from_task_config(env.cfg)==budget['height_reward']==dict(version='fudan-height-exp-scaled-v1',scale=5.,sigma=.01)
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
            def observed_step(*args,**kwargs):
                nonlocal maximum
                external=sim.data.xfrc_applied
                maximum=torch.maximum(maximum,torch.stack((external[:,:,:3].abs().max(),external[:,:,3:].abs().max())))
                count[0]+=1
                return original_step(*args,**kwargs)
            sim.step=observed_step
            force_records.append(lambda:(maximum.cpu().tolist(),count[0]))
        for name in ('armature','damping','frictionloss'):
            values=getattr(sim.model,'dof_'+name)
            if values.ndim==1:values=values.unsqueeze(0)
            expected=torch.tensor(budget['physical_wheel_model'][name],device=env.device)
            assert torch.allclose(values[:,dofs],expected.expand_as(values[:,dofs]),atol=1e-7,rtol=0),name
        write(output/(stage+'-actual-control.json'),dict(num_envs=env.num_envs,leg_p=60.,leg_d=2.,wheel_p=.1,wheel_d=0.,
            leg_armature_unique=torch.unique(legs).cpu().tolist(),physical_wheel_model=env.wheel_physics_profile,
            speed_curve=False,extra_output_layer=False,disturbances=env.disturbance_config.record(),height_reward=height_reward.from_task_config(env.cfg)))
    with ExitStack() as stack:
        for section,key,value in OVERRIDES:stack.enter_context(patch.object(getattr(FudanWalkCfg,section),key,value))
        stack.enter_context(patch.object(FudanEnv,'__init__',checked_init))
        actual=json.loads(json.dumps({name:section_dict(getattr(FudanWalkCfg(),name)) for name in budget['actual_config']}))
        assert actual==budget['actual_config']
        if stage=='precheck':
            from scripts.agent.fudan_ppo_walk import precheck
            with patch.object(sys,'argv',['precheck','--output',str(output/'precheck'),'--deadline',str(time.time()+180),
                '--viewer','viser','--port','8798','--history-version','v3','--training-resets']):precheck.main()
            maximum,count=force_records[0]()
            proof=json.loads((output/'precheck/report.json').read_text())
            assert maximum==[0.,0.] and count>=1000 and not any(proof['push_count'])
            write(output/'precheck-zero-external.json',dict(passed=True,observed_GPU_integration_steps=count,
                maximum_external_force_N=maximum[0],maximum_external_torque_Nm=maximum[1],
                push_count=proof['push_count'],disabled_push_boundary_probe_passed=True,optimizer_updates=0))
        else:
            from ct_mjlab.scripts import train
            from ct_mjlab.rsl_rl.runners import fudan_on_policy_runner as runner
            old_snapshot=runner.snapshot
            def snapshot(path):
                old_snapshot(path);path=Path(path)
                shutil.copy2(__file__,path/'continuation-launcher.py')
                write(path/'effective-training-config.json',actual)
                shutil.copy2(output/'budget.json',path/'authorized-budget.json')
                target=path/'source/resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
                shutil.copy2(XML,target)
            with patch.object(runner,'snapshot',snapshot):
                train.main(['--task','fudan_ppo_walk','--num-envs','4096','--iterations','1500','--seed','1',
                    '--precheck',str(output/'precheck/report.json'),'--log-dir',str(LOG)])


def run(output):
    try:
        for stage in ('precheck','train'):
            run_command(output,stage,[str(PYTHON),'-u','-m',MODULE,stage,'--output',str(output)],
                output/'process-result.json' if stage=='train' else None)
        audit(output)
        write(output/'status.json',dict(status='training_complete',finished_unix=time.time(),iterations_added=1500,
            final_iteration=1500,checkpoint=str(LOG/'model_final.pt'),checkpoint_sha256=sha(LOG/'model_final.pt'),
            behavior_accepted=False,next_step='Three-height GPU actual height, wheel support and body contact verification; compare previous no-external checkpoint'))
    except BaseException as exc:
        write(output/'status.json',dict(status='failed',error=repr(exc),finished_unix=time.time(),automatic_retry=False))
        raise


def audit(output):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    budget=json.loads((output/'budget.json').read_text())
    assert json.loads((output/'process-result.json').read_text())['returncode']==0
    initial=torch.load(LOG/'model_initial.pt',map_location='cpu',weights_only=False)
    final=torch.load(LOG/'model_final.pt',map_location='cpu',weights_only=False)
    report=json.loads((LOG/'training_report.json').read_text())
    assert initial['iterations']==initial['transitions']==initial['primary_updates']==initial['encoder_updates']==0
    assert initial['resume_from'] is None and final['resume_from'] is None
    assert initial['initialization']==final['initialization']=='random_from_scratch_seed_1'
    assert not initial['optimizer_state_dict']['state'] and not initial['encoder_optimizer_state_dict']['state']
    assert report['stopped_by']=='iteration_budget' and report['iterations']==report['iteration_budget']==1500
    assert final['iterations']==1500 and final['transitions']==report['transitions']==1500*4096*48
    assert final['primary_updates']==final['encoder_updates']==30000 and not final['partial_update']
    assert initial['physical_wheel_model']==final['physical_wheel_model']==budget['physical_wheel_model']
    assert initial['height_reward']==final['height_reward']==report['height_reward']==budget['height_reward']
    assert not initial['disturbances']['pushes'] and initial['disturbances']==final['disturbances']==budget['disturbances']
    assert final['disturbances']['external_force_range']==final['disturbances']['external_torque_range']==[0.,0.]
    assert json.loads((output/'precheck-zero-external.json').read_text())['passed']
    assert all(torch.isfinite(v).all() for v in final['model_state_dict'].values())
    for name in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in final[name]['state'].values()}=={30000}
    events=EventAccumulator(str(LOG),size_guidance={'scalars':0}).Reload()
    for tag in ['Train/reward',*['Rewards/'+name for name in final['reward_names']]]:
        assert [v.step for v in events.Scalars(tag)]==list(range(1,budget['iteration_budget']+1)),tag
    manifest=json.loads((LOG/'source_sha256.json').read_text())
    for name,digest in manifest.items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            assert sha(LOG/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/'))==digest,name
    write(output/'training-audit.json',dict(passed=True,iterations=1500,from_scratch=True,
        transitions=final['transitions'],updates_each=30000,checkpoint_sha256=sha(LOG/'model_final.pt'),behavior_accepted=False))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','run','precheck','train'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();output=a.output.resolve()
    if a.mode=='prepare':prepare(output)
    elif a.mode=='run':run(output)
    else:worker(output,a.mode)


if __name__=='__main__':main()
