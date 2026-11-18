"""Start one explicitly budgeted step run, then audit its stop and saved learning state."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
PYTHON=ROOT/'mjlab_training/.venv/bin/python'


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def start(args):
    from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg,recipe_record,SOURCE,SOURCE_SHA
    from scripts.agent.fudan_ppo_step.check import source_fingerprint
    cfg=FudanStepCfg();proof=json.loads(args.precheck.read_text())
    assert args.iterations>0 and args.num_envs>=4 and args.num_envs%4==0
    assert not args.log_dir.exists() and not (args.output/'budget.json').exists()
    assert proof['passed'] and proof['kind']=='step_gpu_precheck' and proof['num_envs']==20
    assert proof['optimizer_updates']==0 and proof['physics_steps']>=2000
    assert proof['recipe']==recipe_record(cfg) and proof['source_fingerprint']==source_fingerprint()
    assert sha(ROOT/SOURCE)==SOURCE_SHA
    args.output.mkdir(parents=True,exist_ok=True)
    argv=['--task','fudan_ppo_step','--num-envs',str(args.num_envs),'--iterations',str(args.iterations),
        '--precheck',str(args.precheck.resolve()),'--log-dir',str(args.log_dir.resolve())]
    initial_iterations=initial_transitions=initial_updates=0
    if args.resume:
        import torch
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert saved['task']=='fudan_ppo_step' and saved['training_recipe']==recipe_record(cfg)
        assert saved['num_envs']==args.num_envs and saved['primary_updates']==saved['encoder_updates']
        initial_iterations=saved['iterations'];initial_transitions=saved['transitions'];initial_updates=saved['primary_updates']
        argv+=['--resume',str(args.resume.resolve())]
    target=initial_iterations+args.iterations
    budget=dict(authorization=f'用户明确授权本次台阶训练{args.iterations}轮、目标总{target}轮；配方见recipe，到预算停止，不自动追加。',
        iterations=target,additional_iterations=args.iterations,initial_iterations=initial_iterations,
        initial_transitions=initial_transitions,initial_each_adam_updates=initial_updates,
        resume_checkpoint=str(args.resume.resolve()) if args.resume else None,
        resume_sha256=sha(args.resume) if args.resume else None,
        num_envs=args.num_envs,rollout_steps=48,seed=1,
        expected_transitions=initial_transitions+args.iterations*args.num_envs*48,
        expected_each_adam_updates=initial_updates+args.iterations*20,
        log_dir=str(args.log_dir.resolve()),precheck=str(args.precheck.resolve()),
        source_checkpoint=str(ROOT/SOURCE),source_sha256=SOURCE_SHA,recipe=recipe_record(cfg),
        source_fingerprint=source_fingerprint(),created_unix=time.time(),training_argv=argv,
        training_command=['scripts/mjlab.sh','train',*argv],
        tensorboard_command=[str(PYTHON),'-m','tensorboard.main','--logdir',str(args.log_dir.resolve()),'--port','6022'],
        git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(args.output/'budget.json',budget)
    frozen=args.output/'launch-source.py';frozen.write_bytes(Path(__file__).read_bytes())
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'mjlab_training/src')+':'+str(ROOT)
    with (args.output/'train.log').open('x') as stream:
        proc=subprocess.Popen([str(PYTHON),str(frozen.resolve()),'worker','--output',str(args.output.resolve())],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(args.output/'process.json',dict(pid=proc.pid,launched_unix=time.time(),worker_sha256=sha(frozen)))
    print(json.dumps(dict(pid=proc.pid,target=target,additional_iterations=args.iterations,log_dir=budget['log_dir']),ensure_ascii=False))


def audit(output,budget):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run=Path(budget['log_dir']);final=torch.load(run/'model_final.pt',map_location='cpu',weights_only=False)
    initial=torch.load(run/'model_initial.pt',map_location='cpu',weights_only=False)
    source=torch.load(budget.get('resume_checkpoint') or budget['source_checkpoint'],map_location='cpu',weights_only=False)
    assert initial['iterations']==budget.get('initial_iterations',0)
    assert initial['primary_updates']==initial['encoder_updates']==budget.get('initial_each_adam_updates',0)
    if budget.get('resume_checkpoint'):
        for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
            assert initial[key]['param_groups']==source[key]['param_groups']
            for index,state in source[key]['state'].items():
                for name,value in state.items():assert torch.equal(initial[key]['state'][index][name],value)
        assert torch.equal(initial['torch_rng'],source['torch_rng'])
        assert all(torch.equal(x,y) for x,y in zip(initial['cuda_rng'],source['cuda_rng']))
    else:
        assert not initial['optimizer_state_dict']['state'] and not initial['encoder_optimizer_state_dict']['state']
    assert all(torch.equal(v,source['model_state_dict'][k]) for k,v in initial['model_state_dict'].items())
    assert final['iterations']==budget['iterations'] and not final['partial_update']
    assert final['primary_updates']==final['encoder_updates']==budget['expected_each_adam_updates']
    assert final['transitions']==budget['expected_transitions'] and final['num_envs']==budget['num_envs']
    assert final['training_recipe']==budget['recipe'] and final['transfer']['sha256']==budget['source_sha256']
    assert all(torch.isfinite(v).all() for v in final['model_state_dict'].values())
    for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in final[key]['state'].values()}=={budget['expected_each_adam_updates']}
    events=EventAccumulator(str(run),size_guidance={'scalars':0}).Reload()
    for tag in ['Train/reward',*['Rewards/'+key for key in final['reward_names']]]:
        steps=[r.step for r in events.Scalars(tag)]
        assert steps in (list(range(1,budget['iterations']+1)),
                         list(range(budget.get('initial_iterations',0)+1,budget['iterations']+1))),tag
    for name,digest in json.loads((run/'source_sha256.json').read_text()).items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            assert sha(run/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/'))==digest,name
    result=dict(passed=True,iterations=final['iterations'],transitions=final['transitions'],
        each_adam_updates=final['primary_updates'],checkpoint=str(run/'model_final.pt'),
        checkpoint_sha256=sha(run/'model_final.pt'),all_networks_inherited_at_initialization=True,
        fresh_optimizers_at_initialization=not bool(budget.get('resume_checkpoint')),
        resumed_learning_state=bool(budget.get('resume_checkpoint')),training_snapshot_verified=True,behavior_accepted=False)
    write(output/'training-audit.json',result)
    return result


def worker(args):
    budget=json.loads((args.output/'budget.json').read_text())
    write(args.output/'status.json',dict(status='starting',pid=os.getpid(),budget=budget['iterations']))
    try:
        import torch
        from ct_mjlab.envs.fudan_ppo_step.env import StepEnv
        from ct_mjlab.scripts.train import main as train
        original=StepEnv.__init__
        def checked_init(env,*a,**kw):
            original(env,*a,**kw)
            assert env.num_envs==budget['num_envs'] and env.dt==.01
            assert env.cfg.sim.dt==.001 and env.cfg.control.pd_decimation==5
            actual={}
            for field,key in [('kp_scale','kp_range'),('kd_scale','kd_range'),
                ('motor_strength','motor_strength_range'),('nominal_offset','nominal_offset_range')]:
                values=getattr(env.action,field);low,high=getattr(env.cfg.domain_rand,key)
                assert values.min()>=low-1e-7 and values.max()<=high+1e-7
                actual[field]=[float(values.min()),float(values.max())]
            for field in ('action_steps','sensor_steps'):
                values=getattr(env.action.disturbances,field)
                assert values.min()>=10 and values.max()<=20
                actual[field]=[int(values.min()),int(values.max())]
            friction=env.env._fudan_parameters['friction']
            assert friction.min()>=.4-1e-6 and friction.max()<=1.2+1e-6
            actual['friction']=[float(friction.min()),float(friction.max())]
            assert not env.disturbance_config.pushes and not env.action.disturbances.has_episode_wrench
            assert env.env.sim.wp_data.njmax==env.cfg.step.constraint_capacity
            assert env.env.sim.wp_data.naconmax==env.num_envs*env.cfg.step.contact_capacity
            write(args.output/'runtime-config.json',dict(passed=True,num_envs=env.num_envs,
                physics_dt=.001,pd_dt=.005,policy_dt=.01,actual_randomization=actual,no_external_wrench=True,
                recipe=env.cfg.step.version,approach_speed_range=list(env.cfg.step.approach_speed),
                step_height_range=list(env.cfg.step.height_range),reverse_probability=env.cfg.step.reverse_probability,
                max_approach_speed=env.cfg.step.max_approach_speed,
                constraint_capacity=env.env.sim.wp_data.njmax,global_contact_capacity=env.env.sim.wp_data.naconmax))
            write(args.output/'status.json',dict(status='training',pid=os.getpid(),budget=budget['iterations'],log_dir=budget['log_dir']))
        StepEnv.__init__=checked_init
        train(budget['training_argv'])
        result=audit(args.output,budget)
        write(args.output/'status.json',dict(status='completed',pid=os.getpid(),**result))
    except BaseException as exc:
        write(args.output/'status.json',dict(status='failed',pid=os.getpid(),error=repr(exc),automatic_restart=False))
        traceback.print_exc();raise


def repair(args):
    import copy
    import torch
    from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg,recipe_record
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    before=saved['training_recipe'];after=recipe_record(FudanStepCfg())
    expected=copy.deepcopy(before);expected['step']['version']=after['step']['version']
    expected['step']['border_width']=2.
    assert expected==after, 'Only the recorded terrain border repair may migrate this checkpoint'
    assert saved['iterations']==8 and saved['primary_updates']==saved['encoder_updates']==160
    assert saved['transitions']==54528
    args.output.mkdir(parents=True,exist_ok=False)
    migration=dict(source=str(args.checkpoint.resolve()),source_sha256=sha(args.checkpoint),
        reason='Add real outer ground border for scan footprint; preserve network/optimizers/RNG/counters',
        before_recipe=before,after_recipe=after,completed_iterations=8,abandoned_rollout_transitions=5376,
        remaining_of_authorized_500=492)
    saved['training_recipe']=after;saved['recipe_migration']=migration
    target=args.output/'model_8_recipe_repaired.pt';torch.save(saved,target)
    write(args.output/'migration.json',dict(**migration,derived_checkpoint=str(target),derived_sha256=sha(target)))


def resize(args):
    import copy
    import torch
    from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg,recipe_record
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert saved['primary_updates']==saved['encoder_updates']==saved['iterations']*20
    assert saved['num_envs']==128 and args.num_envs==4096
    before=saved['training_recipe'];after=recipe_record(FudanStepCfg())
    expected=copy.deepcopy(before)
    expected['step']['version']=after['step']['version'];expected['env']['num_envs']=4096
    expected['step']['contact_capacity']=64;expected['step']['constraint_capacity']=256
    assert expected==after, 'Environment count resize must not change physics, rewards or command recipe'
    args.output.mkdir(parents=True,exist_ok=False)
    ids=torch.arange(args.num_envs).remainder(saved['num_envs'])
    migration=dict(source=str(args.checkpoint.resolve()),source_sha256=sha(args.checkpoint),
        reason='用户要求128环境切换4096；重复课程分布，保留网络/两优化器/RNG/学习计数，物理和历史重新初始化',
        old_num_envs=128,new_num_envs=4096,source_iterations=saved['iterations'],
        source_transitions=saved['transitions'],before_recipe=before,after_recipe=after,
        source_world_index=ids.tolist(),remaining_of_authorized_500=500-saved['iterations'])
    saved['training_recipe']=after;saved['num_envs']=args.num_envs
    saved['terrain_levels']=saved['terrain_levels'][ids].clone()
    saved['terrain_types']=saved['terrain_types'][ids].clone()
    saved['environment_resize']=migration
    target=args.output/'model_resized_4096.pt';torch.save(saved,target)
    restored=torch.load(target,map_location='cpu',weights_only=False)
    original=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert all(torch.equal(v,restored['model_state_dict'][k]) for k,v in original['model_state_dict'].items())
    for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        assert original[key]['param_groups']==restored[key]['param_groups']
        for i,state in original[key]['state'].items():
            for name,value in state.items():assert torch.equal(value,restored[key]['state'][i][name])
    assert torch.equal(original['torch_rng'],restored['torch_rng'])
    assert all(torch.equal(a,b) for a,b in zip(original['cuda_rng'],restored['cuda_rng']))
    write(args.output/'migration.json',dict(migration,source_world_index='index modulo128; exact map in derived PT',
        derived_checkpoint=str(target),derived_sha256=sha(target),learning_state_exact=True))


def repair_rays(args):
    import copy
    import torch
    from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg,recipe_record
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    before=saved['training_recipe'];after=recipe_record(FudanStepCfg())
    expected=copy.deepcopy(before);expected['step']['version']=after['step']['version']
    assert expected==after and saved['num_envs']==4096
    assert saved['iterations']==258 and saved['primary_updates']==saved['encoder_updates']==5160
    args.output.mkdir(parents=True,exist_ok=False)
    migration=dict(source=str(args.checkpoint.resolve()),source_sha256=sha(args.checkpoint),
        reason='Ray-missing episode fails with zero terminal reward/no bootstrap; never label cached terminal heights valid',
        before_recipe=before,after_recipe=after,completed_iterations=258,remaining_of_authorized_500=242,
        inherited_abandoned_transitions=56320)
    saved['training_recipe']=after;saved['ray_termination_migration']=migration
    target=args.output/'model_258_ray_termination.pt';torch.save(saved,target)
    write(args.output/'migration.json',dict(migration,derived_checkpoint=str(target),derived_sha256=sha(target)))


def main():
    p=argparse.ArgumentParser(__doc__);s=p.add_subparsers(dest='mode',required=True)
    a=s.add_parser('start');a.add_argument('--output',type=Path,required=True)
    a.add_argument('--precheck',type=Path,required=True);a.add_argument('--log-dir',type=Path,required=True)
    a.add_argument('--iterations',type=int,required=True);a.add_argument('--num-envs',type=int,default=4096)
    a.add_argument('--resume',type=Path)
    w=s.add_parser('worker');w.add_argument('--output',type=Path,required=True)
    r=s.add_parser('repair');r.add_argument('--checkpoint',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    n=s.add_parser('resize');n.add_argument('--checkpoint',type=Path,required=True);n.add_argument('--output',type=Path,required=True)
    n.add_argument('--num-envs',type=int,required=True)
    f=s.add_parser('repair-rays');f.add_argument('--checkpoint',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    args=p.parse_args();{'start':start,'worker':worker,'repair':repair,'resize':resize,'repair-rays':repair_rays}[args.mode](args)


if __name__=='__main__':main()
