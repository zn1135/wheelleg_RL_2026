"""Authorized 14179+500 continuation with unchanged PD and post-limit wheel gain0.45."""
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
MODEL=ROOT/'artifacts/fudan_ppo_walk/models/20261112_model14179_p60d2_armature2_wheel045.pt'
MODEL_SHA='4cad085dfac886a3a3bc07f01c5ff52900495f311842ab44aa89bc8c86d6d85c'
XML=ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
XML_SHA='ed817cf8b0d41bc3625021c97deef33549f0c8aef7ec4e98a49d6d6a9ac2bb11'
LOG=ROOT/'logs/mjlab/fudan_ppo_walk/20261112_p60d2_wheelP045_output045_resume14179_to14679_01'
MODULE='scripts.agent.fudan_ppo_walk.resume_output_gain_14179'
OVERRIDES=(('control','leg_stiffness',60.),('control','leg_damping',2.),
    ('control','wheel_stiffness',.045),('sim','wheel_motor_gain',.45))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def prepare(output):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg,section_dict
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config
    from ct_mjlab.envs.fudan_ppo_walk.commands import command_config
    from ct_mjlab.envs.fudan_ppo_walk import posture_reward
    assert sha(MODEL)==MODEL_SHA and sha(XML)==XML_SHA and not LOG.exists()
    parent=torch.load(MODEL,map_location='cpu',weights_only=False)
    assert parent['iterations']==14179 and parent['primary_updates']==parent['encoder_updates']==283580
    saved=ROOT/'logs/mjlab/fudan_ppo_walk/20261111_p60d2_armature2_wheel045_resume13679_to14179_01/effective-training-config.json'
    original=json.loads(saved.read_text())
    with ExitStack() as stack:
        for section,key,value in OVERRIDES:stack.enter_context(patch.object(getattr(FudanWalkCfg,section),key,value))
        cfg=FudanWalkCfg();names=('commands','domain_rand','env','sim','control','normalization','rewards')
        effective=json.loads(json.dumps({name:section_dict(getattr(cfg,name)) for name in names}))
        expected=json.loads(json.dumps(original))
        # 父模型的PD、奖励和随机化逐项保留，仅增加独立轮端输出层。
        expected['sim'].update(wheel_motor_gain=.45,wheel_motor_delay_ms=0.,
            wheel_motor_time_constant_ms=0.,wheel_motor_torque_limit_nm=None)
        assert effective==expected
        output.mkdir(parents=True,exist_ok=False)
        write(output/'budget.json',dict(authorization='User requested500 training iterations then MuJoCo validation',
            resume_checkpoint=str(MODEL),resume_checkpoint_sha256=MODEL_SHA,start_iteration=14179,target_iteration=14679,
            additional_iterations=500,num_envs=4096,source_num_envs=parent['num_envs'],log_dir=str(LOG),
            expected_reset_contract=parent['reset_contract'],disturbances=from_task_config(cfg).record(),
            command_sampling=command_config(cfg),posture_reward=posture_reward.from_task_config(cfg),actual_config=effective,
            xml_sha256=XML_SHA,leg_armature_kg_m2=.03159,source_physics_xml_sha256=XML_SHA,
            only_change_from_parent_recipe='Post-limit wheel motor gain1->0.45; wheelP0.045/legP60D2 unchanged',
            created_unix=time.time()))
    shutil.copy2(__file__,output/'launcher.py');shutil.copy2(XML,output/'training-model.xml')


def worker(output,stage):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg,section_dict
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    assert sha(MODEL)==MODEL_SHA and sha(XML)==XML_SHA
    budget=json.loads((output/'budget.json').read_text());torch.set_num_threads(1)
    original_init=FudanEnv.__init__
    def checked_init(env,*args,**kwargs):
        original_init(env,*args,**kwargs)
        a=env.action;motor=a.wheel_motor
        values=env.env.sim.model.dof_armature[:,a._motor_v_adr[[0,1,3,4]]]
        assert torch.allclose(values,torch.full_like(values,.03159),atol=1e-8,rtol=0)
        assert (a.cfg.leg_stiffness,a.cfg.leg_damping,a.cfg.wheel_stiffness,a.cfg.wheel_damping)==(60.,2.,.045,0.)
        assert motor.enabled and motor.gain.eq(.45).all() and motor.delay_steps.eq(0).all() and motor.tau_s.eq(0).all()
        assert torch.isinf(motor.limit).all()
        write(output/(stage+'-actual-control.json'),dict(num_envs=env.num_envs,leg_p=60.,leg_d=2.,wheel_p=.045,wheel_d=0.,
            armature_unique=torch.unique(values).cpu().tolist(),wheel_gain_unique=torch.unique(motor.gain).cpu().tolist(),
            motor_strength_minmax=[float(a.motor_strength.min()),float(a.motor_strength.max())],
            disturbances=env.disturbance_config.record(),posture_reward=env.posture_reward_profile,physics_xml_sha256=sha(XML),
            wheel_extra_delay_ms=0.,wheel_time_constant_ms=0.,wheel_extra_torque_limit_nm=None))
    with ExitStack() as stack:
        for section,key,value in OVERRIDES:stack.enter_context(patch.object(getattr(FudanWalkCfg,section),key,value))
        stack.enter_context(patch.object(FudanEnv,'__init__',checked_init))
        cfg=FudanWalkCfg();actual={name:section_dict(getattr(cfg,name)) for name in budget['actual_config']}
        assert json.loads(json.dumps(actual))==budget['actual_config']
        if stage=='precheck':
            from scripts.agent.fudan_ppo_walk import precheck
            with patch.object(sys,'argv',['precheck','--output',str(output/'precheck'),'--deadline',str(time.time()+120),
                '--viewer','viser','--port','8798','--history-version','v2','--training-resets']):precheck.main()
        else:
            from ct_mjlab.scripts import train
            from ct_mjlab.rsl_rl.runners import fudan_on_policy_runner as runner
            old_snapshot=runner.snapshot
            def snapshot(path):
                old_snapshot(path);path=Path(path)
                shutil.copy2(__file__,path/'continuation-launcher.py')
                write(path/'effective-training-config.json',actual)
                model_path=path/'source/resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
                model_path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(XML,model_path)
            with patch.object(runner,'snapshot',snapshot):
                train.main(['--task','fudan_ppo_walk','--num-envs','4096','--iterations','500','--seed','1',
                    '--resume',str(MODEL),'--preserve-history-v2','--precheck',str(output/'precheck/report.json'),'--log-dir',str(LOG)])


def run(output):
    py=str(PYTHON)
    try:
        run_command(output,'responsecheck',[py,'-m','scripts.agent.fudan_ppo_walk.check_wheel_motor_response','--output',str(output/'response-check')])
        for stage in ('precheck','train'):
            result=output/'process-result.json' if stage=='train' else None
            run_command(output,stage,[py,'-u','-m',MODULE,stage,'--output',str(output)],result)
        run_command(output,'audit',[py,'-m','scripts.agent.fudan_ppo_walk.audit_resume_run','--experiment',str(output),'--output',str(output/'resume-audit.json')])
        audit=json.loads((output/'resume-audit.json').read_text());assert audit['passed'] and audit['final_iteration']==14679
        write(output/'status.json',dict(status='training_complete',finished_unix=time.time(),iterations_added=500,final_iteration=14679,
            checkpoint=str(LOG/'model_final.pt'),checkpoint_sha256=sha(LOG/'model_final.pt'),behavior_accepted=False))
    except BaseException as exc:
        previous=json.loads((output/'status.json').read_text())
        write(output/'status.json',{**previous,'status':'failed','error':repr(exc),'finished_unix':time.time()})
        raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','run','precheck','train'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();output=a.output.resolve()
    if a.mode=='prepare':prepare(output)
    elif a.mode=='run':run(output)
    else:worker(output,a.mode)


if __name__=='__main__':main()
