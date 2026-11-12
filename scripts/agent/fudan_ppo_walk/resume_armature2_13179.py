"""Authorized P60D2/x2-armature continuation:13179+500=13679, then read-only audit."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
import time
from unittest.mock import patch

from scripts.agent.fudan_ppo_walk.run_torque_ablation import run_command

ROOT=Path(__file__).resolve().parents[3]
PYTHON=ROOT/'mjlab_training/.venv/bin/python'
MODEL=ROOT/'artifacts/fudan_ppo_walk/models/20261111_model13179_p60d2_roll_pitch.pt'
MODEL_SHA='8ef6f785dab2bbb47bb0145fd0a59844d7051b323e73498cc4ca6450a559b77c'
XML=ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
XML_SHA='ed817cf8b0d41bc3625021c97deef33549f0c8aef7ec4e98a49d6d6a9ac2bb11'
LOG=ROOT/'logs/mjlab/fudan_ppo_walk/20261111_p60d2_armature2_resume13179_to13679_01'
MODULE='scripts.agent.fudan_ppo_walk.resume_armature2_13179'


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
    assert parent['iterations']==13179 and parent['primary_updates']==parent['encoder_updates']==263580
    saved=ROOT/'logs/mjlab/fudan_ppo_walk/20261111_p60d2_resume11179_to13179_01/source/ct_mjlab/envs/fudan_ppo_walk/fudan_ppo_walk_config.py'
    original=runpy.run_path(str(saved))['FudanWalkCfg']()
    with patch.object(FudanWalkCfg.control,'leg_stiffness',60.),patch.object(FudanWalkCfg.control,'leg_damping',2.):
        cfg=FudanWalkCfg();names=('commands','domain_rand','env','sim','control','normalization','rewards')
        effective={name:section_dict(getattr(cfg,name)) for name in names}
        assert all(effective[name]==section_dict(getattr(original,name)) for name in names)
        output.mkdir(parents=True,exist_ok=False)
        write(output/'budget.json',dict(authorization='User requested continuation from13179/P60D2 for500 additional iterations with adoptedx2 armature',
            resume_checkpoint=str(MODEL),resume_checkpoint_sha256=MODEL_SHA,start_iteration=13179,target_iteration=13679,
            additional_iterations=500,num_envs=4096,source_num_envs=parent['num_envs'],log_dir=str(LOG),
            expected_reset_contract=parent['reset_contract'],disturbances=from_task_config(cfg).record(),
            command_sampling=command_config(cfg),posture_reward=posture_reward.from_task_config(cfg),
            actual_config=effective,xml_sha256=XML_SHA,leg_armature_kg_m2=.03159,source_physics_xml_sha256='505c9b9bb2b7a20e15cf9d2cd4ca664332db402687e34185d8a57de173dedbe8',
            only_change_from_parent_recipe='four active leg armatures x2; full parent training recipe retained',created_unix=time.time()))
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
        indices=env.action._motor_v_adr[[0,1,3,4]];values=env.env.sim.model.dof_armature[:,indices]
        assert torch.allclose(values,torch.full_like(values,.03159),atol=1e-8,rtol=0)
        assert (env.action.cfg.leg_stiffness,env.action.cfg.leg_damping)==(60.,2.)
        write(output/(stage+'-actual-control.json'),dict(num_envs=env.num_envs,leg_p=60.,leg_d=2.,
            armature_unique=torch.unique(values).cpu().tolist(),disturbances=env.disturbance_config.record(),
            posture_reward=env.posture_reward_profile,physics_xml_sha256=sha(XML),
            pd_velocity_noise_added=False,pd_velocity_delay_added=False))
    with patch.object(FudanWalkCfg.control,'leg_stiffness',60.),patch.object(FudanWalkCfg.control,'leg_damping',2.),patch.object(FudanEnv,'__init__',checked_init):
        cfg=FudanWalkCfg();actual={name:section_dict(getattr(cfg,name)) for name in budget['actual_config']}
        assert json.loads(json.dumps(actual))==budget['actual_config']
        if stage=='gaincheck':
            from scripts.agent.fudan_ppo_walk import check_randomization_ranges
            with patch.object(sys,'argv',['check','--output',str(output/'randomization-check')]):check_randomization_ranges.main()
        elif stage=='precheck':
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
        for stage in ('gaincheck','precheck','train'):
            result=output/'process-result.json' if stage=='train' else None
            run_command(output,stage,[py,'-u','-m',MODULE,stage,'--output',str(output)],result)
        run_command(output,'audit',[py,'-m','scripts.agent.fudan_ppo_walk.audit_resume_run','--experiment',str(output),'--output',str(output/'resume-audit.json')])
        audit=json.loads((output/'resume-audit.json').read_text());assert audit['passed'] and audit['final_iteration']==13679
        report=dict(status='complete',finished_unix=time.time(),iterations_added=500,final_iteration=13679,
            checkpoint=str(LOG/'model_final.pt'),checkpoint_sha256=sha(LOG/'model_final.pt'),behavior_accepted=False)
        write(output/'status.json',report)
        (output/'COMPLETED.md').write_text('# 两倍惯量续训500轮完成\n\n从13179/P60D2到13679，原奖励/随机化/外载保持，四腿惯量0.03159。恢复、轮数、日志和学习状态审计通过；未追加训练或行为评估。\n\n模型：'+report['checkpoint']+'\n\nSHA256：'+report['checkpoint_sha256']+'\n\n下一步在新惯量下与13179做同条件摆腿及停车评估；训练完成不等于行为通过。\n')
    except BaseException as exc:
        previous=json.loads((output/'status.json').read_text())
        write(output/'status.json',{**previous,'status':'failed','error':repr(exc),'finished_unix':time.time()})
        raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','run','gaincheck','precheck','train'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();output=a.output.resolve()
    if a.mode=='prepare':prepare(output)
    elif a.mode=='run':run(output)
    else:worker(output,a.mode)


if __name__=='__main__':main()
