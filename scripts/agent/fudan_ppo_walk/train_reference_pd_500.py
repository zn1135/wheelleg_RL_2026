"""Authorized model3000 continuation: only nominal leg PD changes60/2 to20/1."""
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
MODEL=ROOT/'artifacts/fudan_ppo_walk/models/20261113_model3000_p60d2_armature2_wheelP01_mechanical.pt'
SOURCE_RUN=ROOT/'logs/mjlab/fudan_ppo_walk/20261113_p60d2_wheelP01_mechanical_scratch3000_01'
LOG=ROOT/'logs/mjlab/fudan_ppo_walk/20261113_p20d1_wheelP01_mechanical_resume3000_to3500_01'
REFERENCE=ROOT/'artifacts/fudan_ppo_walk/fudan-ppo-walk-20261101/reference/plane/wheel_legged_gym/envs/wheel_legged/wheel_legged_config.py'
XML=ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
MODULE='scripts.agent.fudan_ppo_walk.train_reference_pd_500'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def prepare(out):
    import torch
    checkpoint=torch.load(MODEL,map_location='cpu',weights_only=False)
    assert checkpoint['iterations']==3000 and not checkpoint['partial_update']
    assert sha(MODEL)=='9d4d3123d42451e8942581b7cc943bc35fa8e35f7947b5c5747abeb89de215ee'
    assert not LOG.exists()
    hashes=json.loads((SOURCE_RUN/'source_sha256.json').read_text())
    assert sha(XML)==hashes[str(XML.relative_to(ROOT))]
    baseline=json.loads((SOURCE_RUN/'effective-training-config.json').read_text())
    assert baseline['control']['leg_stiffness']==60. and baseline['control']['leg_damping']==2.
    cfg=json.loads(json.dumps(baseline));cfg['control']['leg_stiffness']=20.;cfg['control']['leg_damping']=1.
    cfg['sim']['wheel_motor_time_constant_range_ms']=None
    out.mkdir(parents=True,exist_ok=False)
    budget=dict(authorization='User requested Fudan reference joint PD and500 additional iterations; source3000 from preceding task context',
        checkpoint=str(MODEL),checkpoint_sha256=sha(MODEL),xml_sha256=sha(XML),start_iteration=3000,target_iteration=3500,additional_iterations=500,num_envs=4096,
        log_dir=str(LOG),source_config_sha256=sha(SOURCE_RUN/'effective-training-config.json'),reference_config_sha256=sha(REFERENCE),
        only_active_config_changes={'control.leg_stiffness':[60.,20.],'control.leg_damping':[2.,1.]},
        passive_new_schema_field='sim.wheel_motor_time_constant_range_ms=None (disabled)',actual_config=cfg,
        height_reward='Inherited scale1/sigma0.001',external_loads='Inherited source3000 pushes and episode +/-10N / +/-10Nm, not current scratch1500 defaults',
        reset_contract=checkpoint['reset_contract'],created_unix=time.time())
    write(out/'budget.json',budget);shutil.copy2(__file__,out/'launcher.py');shutil.copy2(REFERENCE,out/'reference-control.py')
    shutil.copy2(SOURCE_RUN/'effective-training-config.json',out/'source-effective-config.json')
    write(out/'status.json',dict(status='prepared',training_started=False))


def worker(out,stage):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg,section_dict
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.envs.fudan_ppo_walk import height_reward,wheel_physics
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config,from_record
    budget=json.loads((out/'budget.json').read_text());torch.set_num_threads(1);assert sha(MODEL)==budget['checkpoint_sha256']
    assert sha(XML)==budget['xml_sha256']
    source=torch.load(MODEL,map_location='cpu',weights_only=False)
    original_init=FudanEnv.__init__
    def checked_init(env,*args,**kwargs):
        original_init(env,*args,**kwargs);a=env.action;sim=env.env.sim
        assert (a.cfg.leg_stiffness,a.cfg.leg_damping,a.cfg.wheel_stiffness,a.cfg.wheel_damping)==(20.,1.,.1,0.)
        assert not a.wheel_motor.enabled and not a._wheel_speed_shaping and a.wheel_motor.gain.eq(1).all()
        assert env.wheel_physics_profile==source['physical_wheel_model']==wheel_physics.record('mechanical')
        assert height_reward.from_task_config(env.cfg)==height_reward.from_record(source.get('height_reward'))
        assert env.disturbance_config==from_record(source['disturbances'])
        leg_armature=sim.model.dof_armature[:,a._motor_v_adr[[0,1,3,4]]]
        assert torch.allclose(leg_armature,torch.full_like(leg_armature,.03159),atol=1e-8,rtol=0)
        assert a.kp_scale.min()>=.85 and a.kp_scale.max()<=1.15 and a.kd_scale.min()>=.85 and a.kd_scale.max()<=1.15
        write(out/(stage+'-actual-control.json'),dict(num_envs=env.num_envs,control=section_dict(env.cfg.control),
            kp_scale_minmax=[float(a.kp_scale.min()),float(a.kp_scale.max())],kd_scale_minmax=[float(a.kd_scale.min()),float(a.kd_scale.max())],
            wheel_response_enabled=a.wheel_motor.enabled,physical_wheel_model=env.wheel_physics_profile,height_reward=height_reward.from_task_config(env.cfg),
            disturbances=env.disturbance_config.record(),leg_armature_unique=torch.unique(leg_armature).cpu().tolist()))
    with ExitStack() as stack:
        # 仅本次进程沿用源配方；不把其他实验的奖励/外力开关混入本次PD对照。
        for section,values in budget['actual_config'].items():
            for key,value in values.items():stack.enter_context(patch.object(getattr(FudanWalkCfg,section),key,value,create=True))
        stack.enter_context(patch.object(FudanEnv,'__init__',checked_init))
        actual=json.loads(json.dumps({name:section_dict(getattr(FudanWalkCfg(),name)) for name in budget['actual_config']}))
        assert actual==budget['actual_config']
        write(out/(stage+'-effective-config.json'),actual)
        if stage=='precheck':
            from scripts.agent.fudan_ppo_walk import precheck
            with patch.object(sys,'argv',['precheck','--output',str(out/'precheck'),'--deadline',str(time.time()+180),
                '--viewer','viser','--port','8798','--history-version','v3','--training-resets']):precheck.main()
        else:
            from ct_mjlab.scripts import train
            from ct_mjlab.rsl_rl.runners import fudan_on_policy_runner as runner
            original_snapshot=runner.snapshot
            def snapshot(path):
                original_snapshot(path);path=Path(path)
                shutil.copy2(__file__,path/'continuation-launcher.py');shutil.copy2(out/'budget.json',path/'authorized-budget.json')
                write(path/'effective-training-config.json',actual)
                shutil.copy2(XML,path/'source/resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')
            with patch.object(runner,'snapshot',snapshot):
                train.main(['--task','fudan_ppo_walk','--num-envs','4096','--iterations','500','--seed','1',
                    '--resume',str(MODEL),'--precheck',str(out/'precheck/report.json'),'--log-dir',str(LOG)])


def audit(out):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    source=torch.load(MODEL,map_location='cpu',weights_only=False);initial=torch.load(LOG/'model_initial.pt',map_location='cpu',weights_only=False);final=torch.load(LOG/'model_final.pt',map_location='cpu',weights_only=False)
    report=json.loads((LOG/'training_report.json').read_text())
    assert initial['iterations']==3000 and final['iterations']==3500 and not final['partial_update']
    assert report['iterations_added']==report['iteration_budget']==500 and report['stopped_by']=='iteration_budget'
    assert final['transitions']-source['transitions']==500*4096*48
    assert final['primary_updates']-source['primary_updates']==final['encoder_updates']-source['encoder_updates']==10000
    assert initial['resume_from']['sha256']==sha(MODEL)
    for key in source['model_state_dict']:assert torch.equal(initial['model_state_dict'][key],source['model_state_dict'][key])
    for checkpoint in (initial,final):
        assert checkpoint['control']['leg_stiffness']==20. and checkpoint['control']['leg_damping']==1. and checkpoint['control']['wheel_stiffness']==.1
        assert checkpoint['wheel_response']['time_constant_ms']==0 and checkpoint['wheel_response']['time_constant_range_ms'] is None
    events=EventAccumulator(str(LOG),size_guidance={'scalars':0});events.Reload()
    for tag in events.Tags()['scalars']:
        if tag.startswith('Rewards/'):
            assert [v.step for v in events.Scalars(tag)]==list(range(3001,3501)),tag
    write(out/'completion-audit.json',dict(passed=True,initial_iteration=3000,final_iteration=3500,iterations_added=500,
        checkpoint=str(LOG/'model_final.pt'),checkpoint_sha256=sha(LOG/'model_final.pt'),behavior_accepted=False))


def run(out):
    try:
        for stage in ('precheck','train'):run_command(out,stage,[str(PYTHON),'-u','-m',MODULE,stage,'--output',str(out)])
        audit(out);write(out/'status.json',dict(status='training_complete',final_iteration=3500,iterations_added=500,finished_unix=time.time(),behavior_accepted=False))
    except BaseException as exc:
        write(out/'status.json',dict(status='failed',error=repr(exc),finished_unix=time.time(),automatic_retry=False));raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','run','precheck','train'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='run':run(out)
    else:worker(out,a.mode)


if __name__=='__main__':main()
