"""Run the authorized 10000 -> 10500 paired torque experiment in separate processes."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
PYTHON=ROOT/'mjlab_training/.venv/bin/python'
MODEL=ROOT/'artifacts/fudan_ppo_walk/models/20261110_model10000_legacy_action10ms.pt'
MODEL_SHA='2d3ff255b1996a39a3dc1d0d36e51cf68c241117c6563dc9acb852309de30434'
BASE=ROOT/'artifacts/fudan_ppo_walk/20261111-model10000-parking-posture-01'
MODULE='scripts.agent.fudan_ppo_walk.run_torque_ablation'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def prepare(output):
    assert sha(MODEL)==MODEL_SHA
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'docs/model-deliveries/data/20261110-model10000-manifest.json').read_text())
    relative='mjlab_training/src/ct_mjlab/envs/fudan_ppo_walk/fudan_ppo_walk_config.py'
    source=ROOT/'logs/mjlab/fudan_ppo_walk/20261110_legacy_action10ms_resume9000_to10000_01/source/ct_mjlab/envs/fudan_ppo_walk/fudan_ppo_walk_config.py'
    assert sha(source)==manifest['runtime_source_sha256'][relative]
    shutil.copy2(source,output/'baseline-config.py')
    saved=runpy.run_path(str(source))['FudanWalkCfg']
    names=('commands','domain_rand','env','sim','control','normalization','rewards')
    common={s:{k:v for k,v in vars(getattr(saved,s)).items() if not k.startswith('_')} for s in names}
    common['env']['fail_ticks']=0  # 两组共同保留现有倒地立即终止修复。
    common['domain_rand']['external_force_range']=[0.,0.]
    specs={}
    for arm,torque in (('no_torque',0.),('torque10',10.)):
        cfg=json.loads(json.dumps(common));cfg['domain_rand']['external_torque_range']=[-torque,torque]
        folder=output/arm;folder.mkdir()
        specs[arm]=dict(arm=arm,config=cfg,iterations=500,start_iteration=10000,target_iteration=10500,
            num_envs=4096,seed=1,checkpoint=str(MODEL),checkpoint_sha256=MODEL_SHA,
            log_dir=str(ROOT/f'logs/mjlab/fudan_ppo_walk/20261111_torque_ablation_10000_{arm}_500_01'))
        write(folder/'spec.json',specs[arm])
        assert not Path(specs[arm]['log_dir']).exists()
    a=json.loads(json.dumps(specs['no_torque']['config']));b=json.loads(json.dumps(specs['torque10']['config']))
    a['domain_rand'].pop('external_torque_range');b['domain_rand'].pop('external_torque_range');assert a==b
    write(output/'plan.json',dict(authorization='User confirmed starting before torque addition, each arm500 iterations,1000 total',
        only_between_arm_config_difference='domain_rand.external_torque_range',arms=specs,
        shared_change_from_historical10000='fail_ticks100->0, current immediate-fall termination retained',
        ordering=['no_torque','torque10'],evaluation='Same recorded9 parking cases in clean and configured external conditions; joint motion plus unified motion scoring',
        inference='One seed, exploratory continuation; no automatic extra training or retry',baseline_recording=str(BASE),
        source_config_sha256=sha(source),prepared_unix=time.time()))
    shutil.copy2(__file__,output/'run_torque_ablation.py')


def worker(output,arm,stage):
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg,section_dict
    folder=output/arm;spec=json.loads((folder/'spec.json').read_text())
    assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
    with ExitStack() as stack:
        # 只覆盖当前子进程，训练仍走注册器和原runner；不改共享默认配置文件。
        for section,values in spec['config'].items():
            for key,value in values.items():
                stack.enter_context(patch.object(getattr(FudanWalkCfg,section),key,value,create=True))
        effective={name:section_dict(getattr(FudanWalkCfg(),name)) for name in spec['config']}
        # 旧奖励没有新soft-band字段；其计算由scales中的orientation决定。
        for name,values in spec['config'].items():
            for key,value in values.items():assert effective[name][key]==value,(name,key)
        write(folder/f'{stage}-effective-config.json',effective)
        if stage=='precheck':
            from scripts.agent.fudan_ppo_walk import precheck
            port=8796 if arm=='no_torque' else 8797
            with patch.object(sys,'argv',['precheck','--output',str(folder/'precheck'),'--deadline',str(time.time()+120),
                '--viewer','viser','--port',str(port),'--history-version','v2','--training-resets']):precheck.main()
            return
        from ct_mjlab.scripts import train
        from ct_mjlab.rsl_rl.runners import fudan_on_policy_runner as runner
        from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config
        from ct_mjlab.envs.fudan_ppo_walk.commands import command_config
        from ct_mjlab.envs.fudan_ppo_walk import posture_reward
        from ct_mjlab.fudan_ppo_walk.contract import RESET_V2_CONTRACT
        cfg=FudanWalkCfg()
        write(folder/'budget.json',dict(resume_checkpoint=spec['checkpoint'],resume_checkpoint_sha256=spec['checkpoint_sha256'],
            start_iteration=10000,target_iteration=10500,additional_iterations=500,num_envs=4096,source_num_envs=4096,
            log_dir=spec['log_dir'],expected_reset_contract=RESET_V2_CONTRACT,disturbances=from_task_config(cfg).record(),
            command_sampling=command_config(cfg),posture_reward=posture_reward.from_task_config(cfg),
            authorized_total_iterations=1000,arm=arm,actual_config=str(folder/'train-effective-config.json')))
        original_snapshot=runner.snapshot
        def snapshot(path):
            original_snapshot(path)
            shutil.copy2(__file__,Path(path)/'ablation-launcher.py')
            shutil.copy2(folder/'spec.json',Path(path)/'ablation-spec.json')
            shutil.copy2(folder/'train-effective-config.json',Path(path)/'effective-config.json')
        original_make=train.task_registry.make_env
        def make_env(*args,**kwargs):
            env,c=original_make(*args,**kwargs)
            control=env.action.cfg;dist=env.disturbance_config
            assert control.leg_stiffness==60. and control.leg_damping==2.
            assert dist==from_task_config(cfg)
            write(folder/'actual-training-control.json',dict(leg_p=control.leg_stiffness,leg_d=control.leg_damping,
                disturbances=dist.record(),posture_reward=env.posture_reward_profile,num_envs=env.num_envs,
                kp_minmax=[float(env.action.kp_scale.min()),float(env.action.kp_scale.max())],
                kd_minmax=[float(env.action.kd_scale.min()),float(env.action.kd_scale.max())],
                episode_wrench_min=env.action.disturbances.episode_wrench.amin(0).cpu().tolist(),
                episode_wrench_max=env.action.disturbances.episode_wrench.amax(0).cpu().tolist()))
            return env,c
        with patch.object(train,'snapshot',snapshot,create=True),patch.object(runner,'snapshot',snapshot),patch.object(train.task_registry,'make_env',make_env):
            train.main(['--task','fudan_ppo_walk','--num-envs','4096','--iterations','500','--seed','1',
                '--resume',spec['checkpoint'],'--preserve-history-v2','--precheck',str(folder/'precheck/report.json'),
                '--log-dir',spec['log_dir']])


def run_command(output,name,cmd,result_path=None):
    started=time.time();write(output/'status.json',dict(stage=name,status='running',started_unix=started,command=cmd))
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'mjlab_training/src');env['MUJOCO_GL']='egl'
    with (output/(name+'.log')).open('w') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        write(output/'status.json',dict(stage=name,status='running',started_unix=started,pid=process.pid,command=cmd))
        code=process.wait()
    result=dict(returncode=code,started_unix=started,finished_unix=time.time(),command=cmd)
    write(result_path or output/(name+'-process.json'),result)
    if code:raise RuntimeError(f'{name} failed with exit{code}; no automatic retry')


def run(output):
    plan=json.loads((output/'plan.json').read_text())
    assert plan['arms']['no_torque']['iterations']==plan['arms']['torque10']['iterations']==500
    py=str(PYTHON)
    try:
        for arm in ('no_torque','torque10'):
            run_command(output,arm+'-precheck',[py,'-u','-m',MODULE,'worker','--output',str(output),'--arm',arm,'--stage','precheck'])
        for arm in ('no_torque','torque10'):
            folder=output/arm
            run_command(output,arm+'-train',[py,'-u','-m',MODULE,'worker','--output',str(output),'--arm',arm,'--stage','train'],folder/'process-result.json')
            run_command(output,arm+'-audit',[py,'-m','scripts.agent.fudan_ppo_walk.audit_resume_run','--experiment',str(folder),'--output',str(folder/'resume-audit.json')])
        models=output/'models';models.mkdir()
        for arm in ('no_torque','torque10'):
            original=Path(plan['arms'][arm]['log_dir'])/'model_final.pt';model=models/f'model10500_{arm}.pt'
            shutil.copy2(original,model);assert sha(original)==sha(model)
            recording=output/(arm+'-evaluation')
            run_command(output,arm+'-evaluation',[py,'-u','-m','scripts.agent.fudan_ppo_walk.diagnose_parking_posture',
                '--checkpoint',str(model),'--output',str(recording),'--leg-p','60','--leg-d','2','--seconds','300'])
            for group in ('no_external','configured_external'):
                run_command(output,arm+'-'+group+'-accept',[str(ROOT/'scripts/mjlab.sh'),'accept','--task','fudan_ppo_walk',
                    '--recording',str(recording/group),'--output',str(recording/(group+'-acceptance'))])
            run_command(output,arm+'-video',[py,'-m','scripts.agent.fudan_ppo_walk.diagnose_parking_posture',
                '--checkpoint',str(model),'--output',str(recording),'--render-only'])
        run_command(output,'leg-motion',[py,'-m','scripts.agent.fudan_ppo_walk.report_parking_leg_motion','--recordings',
            str(BASE),str(output/'no_torque-evaluation'),str(output/'torque10-evaluation'),'--output',str(output/'leg-motion')])
        write(output/'status.json',dict(status='complete',finished_unix=time.time(),iterations_per_arm=500,total_iterations=1000,
            note='Training and recorded evaluations complete; inspect paired behavior before recommendation'))
    except BaseException as exc:
        current=json.loads((output/'status.json').read_text())
        write(output/'status.json',dict(**{k:v for k,v in current.items() if k!='status'},status='failed',error=repr(exc),finished_unix=time.time()))
        raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('prepare','run','worker'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=('no_torque','torque10'))
    p.add_argument('--stage',choices=('precheck','train'));args=p.parse_args();output=args.output.resolve()
    if args.mode=='prepare':prepare(output)
    elif args.mode=='run':run(output)
    else:worker(output,args.arm,args.stage)


if __name__=='__main__':main()
