"""冻结远程台阶运行的输入、零更新容量检查与定额启动；不覆盖已有目录。"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def same_state(left,right):
    import torch
    if torch.is_tensor(left):return torch.equal(left.cpu(),right.cpu())
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(same_state(value,right[key]) for key,value in left.items())
    if isinstance(left,(list,tuple)):
        return len(left)==len(right) and all(same_state(a,b) for a,b in zip(left,right))
    return left==right


def pack(args):
    args.output.mkdir(parents=True,exist_ok=False)
    paths=set()
    for directory in ('mjlab_training/src','scripts','sim2sim'):
        paths.update(p for p in (ROOT/directory).rglob('*') if p.suffix in ('.py','.sh') and '__pycache__' not in p.parts)
    paths.update((ROOT/'resources/robots/chuanliantui_new_1').rglob('*'))
    paths.update((ROOT/'docs').rglob('*.md'))
    for relative in ('AGENTS.md','CONTRIBUTING.md','README.md','COMMANDS.md','LICENSE',
        'mjlab_training/pyproject.toml','mjlab_training/uv.lock',
        'logs/mjlab/fudan_ppo_walk/20261114_pd200_p20d1_half_random_delay_wide_resume1000_to3000_01/model_final.pt',
        'artifacts/fudan_ppo_step/20261116-contact-exploration-v7-01/reference/motion.npz',
        'artifacts/fudan_ppo_step/20261116-contact-exploration-v7-01/reference/reference.json'):
        path=ROOT/relative
        if path.is_file():paths.add(path)
    manifest=dict(files={},git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),created_unix=time.time(),
        scope='Frozen dirty working-copy inputs; no Git metadata, credentials, environments or unrelated logs')
    archive=args.output/'inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(paths):
            if not path.is_file():continue
            data=path.read_bytes();name=str(path.relative_to(ROOT))
            manifest['files'][name]=hashlib.sha256(data).hexdigest()
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=path.stat().st_mode&0o777
            tar.addfile(info,io.BytesIO(data))
        data=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
        info=tarfile.TarInfo('remote-input-manifest.json');info.size=len(data);info.mode=0o644
        tar.addfile(info,io.BytesIO(data))
    write(args.output/'input-manifest.json',manifest)
    write(args.output/'archive.json',dict(archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(manifest['files'])))
    print(json.dumps(dict(archive=str(archive),files=len(manifest['files']),bytes=archive.stat().st_size)))


def verify(args):
    manifest=json.loads((ROOT/'remote-input-manifest.json').read_text())
    mismatches=[name for name,digest in manifest['files'].items() if not (ROOT/name).is_file() or sha(ROOT/name)!=digest]
    if mismatches:raise ValueError('Remote input mismatch: '+str(mismatches[:10]))
    import sys,importlib.metadata as metadata,torch
    report=dict(passed=True,files=len(manifest['files']),python=sys.executable,version=sys.version,
        packages={name:metadata.version(name) for name in ('mjlab','torch','mujoco','mujoco-warp','warp-lang','rsl-rl-lib','tensorboard')},
        gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,git_branch=manifest['git_branch'],git_head=manifest['git_head'],
        input_manifest_sha256=sha(ROOT/'remote-input-manifest.json'))
    args.output.parent.mkdir(parents=True,exist_ok=True);write(args.output,report);print(json.dumps(report))


def capacity(args):
    import copy
    import torch
    from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg,FudanStepCfgPPO,SOURCE
    from ct_mjlab.envs.fudan_ppo_step.env import StepEnv
    from ct_mjlab.envs.fudan_ppo_step.runner import StepRunner
    from scripts.agent.fudan_ppo_step.check import source_fingerprint
    args.output.mkdir(parents=True,exist_ok=False);env=runner=None
    result=dict(passed=False,optimizer_updates=0,num_envs=args.num_envs)
    torch.set_num_threads(1);torch.manual_seed(1);torch.cuda.reset_peak_memory_stats()
    began=time.monotonic()
    try:
        with patch('torch.optim.Adam.step',side_effect=AssertionError('Remote capacity must not update')):
            env=StepEnv(args.num_envs,cfg=FudanStepCfg(),randomized=True,seed=1)
            runner=StepRunner(env,train_cfg=FudanStepCfgPPO(),log_dir=args.output/'runner',device='cuda:0')
            if args.resume:
                saved=torch.load(args.resume,map_location='cpu',weights_only=False)
                runner.load(args.resume)
                # 续训须同时恢复两套Adam、课程和随机数，不能只加载网络。
                receipt=dict(networks=same_state(saved['model_state_dict'],runner.policy.state_dict()),
                    primary_optimizer=same_state(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict()),
                    encoder_optimizer=same_state(saved['encoder_optimizer_state_dict'],runner.alg.extra_optimizer.state_dict()),
                    curriculum_levels=same_state(saved['terrain_levels'],env.terrain.terrain_levels),
                    curriculum_modes=same_state(saved['terrain_types'],env.terrain.terrain_types),
                    torch_rng=same_state(saved['torch_rng'],torch.get_rng_state()),
                    cuda_rng=same_state(saved['cuda_rng'],torch.cuda.get_rng_state_all()),
                    counters=(runner.iteration==saved['iterations'] and runner.transitions==saved['transitions']
                              and runner.primary_updates==runner.encoder_updates==saved['primary_updates']))
                assert all(receipt.values()),receipt
                result.update(resume_sha256=sha(args.resume),resume_restoration=receipt,
                              initial_iterations=runner.iteration)
            else:
                runner.initialize_from_ppo(ROOT/SOURCE)
            optimizer_before=[copy.deepcopy(opt.state_dict()) for opt in (runner.alg.optimizer,runner.alg.extra_optimizer)]
            initial_update_counts=(runner.primary_updates,runner.encoder_updates)
            frozen={name:value.clone() for name,value in runner.policy.state_dict().items()}
            obs,critic,history=env.reset();sample_start=time.monotonic()
            with torch.inference_mode():
                for _ in range(runner.num_steps_per_env):
                    if time.time()>=args.deadline:raise TimeoutError('Remote capacity deadline')
                    action=runner.alg.act(obs,history,critic)
                    obs,critic,reward,done,info,history=env.step(action)
                    runner.alg.process_env_step(reward,done,info,obs)
                runner.alg.compute_returns(torch.cat((critic,runner.policy.encode(history)),-1))
            sample_seconds=time.monotonic()-sample_start
            try:runner.alg.update()
            except AssertionError as exc:assert str(exc)=='Remote capacity must not update'
            else:raise AssertionError('Expected Adam boundary interception')
            assert all(torch.equal(value,frozen[name]) for name,value in runner.policy.state_dict().items())
            learning_rates=[]
            for before,opt in zip(optimizer_before,(runner.alg.optimizer,runner.alg.extra_optimizer)):
                after=opt.state_dict()
                assert same_state(before['state'],after['state'])
                # PPO在Adam边界前调整lr；此处核动量/步数不变，不阻止原算法的调度。
                assert same_state([{k:v for k,v in group.items() if k!='lr'} for group in before['param_groups']],
                                  [{k:v for k,v in group.items() if k!='lr'} for group in after['param_groups']])
                learning_rates.append(dict(before=[g['lr'] for g in before['param_groups']],
                                           after_backward=[g['lr'] for g in after['param_groups']]))
            assert (runner.primary_updates,runner.encoder_updates)==initial_update_counts
            result.update(passed=True,source_fingerprint=source_fingerprint(),rollout_steps=runner.num_steps_per_env,
                physics_ticks=int(env.contacts.total_ticks),constraint_peak=int(env.contacts.constraint_peak),
                capacity_overflow=bool(env.contacts.capacity_overflow.any()),sample_seconds=sample_seconds,
                policy_transitions_per_second=args.num_envs*runner.num_steps_per_env/sample_seconds,
                peak_torch_bytes=torch.cuda.max_memory_allocated(),gpu_free_total_bytes=list(torch.cuda.mem_get_info()),
                primary_updates=runner.primary_updates,encoder_updates=runner.encoder_updates,
                all_weights_unchanged=True,ppo_and_flat_anchor_backward_checked=True,elapsed_seconds=time.monotonic()-began)
            result.update(optimizer_moments_and_steps_unchanged=True,learning_rate_schedule_probe=learning_rates)
            assert not result['capacity_overflow']
    except BaseException as exc:result['error']=repr(exc);raise
    finally:
        if runner is not None:runner.writer.close()
        if env is not None:env.close()
        write(args.output/'report.json',result)
    print(json.dumps(result))


def start(args):
    from ct_mjlab.envs.fudan_ppo_step.config import FudanStepCfg,recipe_record,SOURCE,SOURCE_SHA
    from scripts.agent.fudan_ppo_step.check import source_fingerprint
    from scripts.agent.fudan_ppo_step import launch_training
    cfg=FudanStepCfg();proof=json.loads(args.precheck.read_text());sizing=json.loads(args.capacity.read_text())
    manifest=json.loads((ROOT/'remote-input-manifest.json').read_text())
    assert args.iterations>0 and args.num_envs>0 and args.num_envs%4==0
    assert proof['passed'] and proof['kind']=='step_gpu_precheck' and proof['num_envs']==20
    assert proof['optimizer_updates']==0 and proof['physics_steps']>=2000 and proof['recipe']==recipe_record(cfg)
    assert sizing['passed'] and sizing['num_envs']==args.num_envs and sizing['optimizer_updates']==0
    assert proof['source_fingerprint']==sizing['source_fingerprint']==source_fingerprint()
    assert sha(ROOT/SOURCE)==SOURCE_SHA and not args.log_dir.exists()
    argv=['--task','fudan_ppo_step','--num-envs',str(args.num_envs),'--iterations',str(args.iterations),
          '--precheck',str(args.precheck.resolve()),'--log-dir',str(args.log_dir.resolve())]
    initial_iterations=initial_transitions=initial_updates=0
    resume_sha256=None
    if args.resume:
        import torch
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert saved['task']=='fudan_ppo_step' and saved['training_recipe']==recipe_record(cfg)
        assert not saved['partial_update'] and saved['num_envs']==args.num_envs
        assert saved['primary_updates']==saved['encoder_updates']==saved['iterations']*20
        assert all(torch.isfinite(value).all() for value in saved['model_state_dict'].values())
        initial_iterations=saved['iterations'];initial_transitions=saved['transitions'];initial_updates=saved['primary_updates']
        resume_sha256=sha(args.resume)
        assert sizing['resume_sha256']==resume_sha256 and all(sizing['resume_restoration'].values())
        argv+=['--resume',str(args.resume.resolve())]
    else:
        assert not sizing.get('resume_sha256')
    target=initial_iterations+args.iterations
    args.output.mkdir(parents=True,exist_ok=False)
    python=ROOT/'mjlab_training/.venv/bin/python'
    budget=dict(authorization=f'用户明确要求在指定SSH远程续训{args.iterations}轮；从{initial_iterations}到总{target}轮，到预算自动停止，不追加。',
        iterations=target,additional_iterations=args.iterations,initial_iterations=initial_iterations,initial_transitions=initial_transitions,
        initial_each_adam_updates=initial_updates,resume_checkpoint=str(args.resume.resolve()) if args.resume else None,
        resume_sha256=resume_sha256,num_envs=args.num_envs,
        rollout_steps=48,seed=1,expected_transitions=initial_transitions+args.iterations*args.num_envs*48,
        expected_each_adam_updates=initial_updates+args.iterations*20,log_dir=str(args.log_dir.resolve()),
        precheck=str(args.precheck.resolve()),capacity=str(args.capacity.resolve()),
        source_checkpoint=str(ROOT/SOURCE),source_sha256=SOURCE_SHA,recipe=recipe_record(cfg),
        source_fingerprint=source_fingerprint(),created_unix=time.time(),training_argv=argv,
        training_command=['scripts/mjlab.sh','train',*argv],
        tensorboard_command=[str(python),'-m','tensorboard.main','--logdir',str(args.log_dir.resolve()),'--port','6030'],
        git_branch=manifest['git_branch'],git_head=manifest['git_head'],
        deployment_manifest_sha256=sha(ROOT/'remote-input-manifest.json'),execution='Remote frozen archive; no remote Git checkout')
    write(args.output/'budget.json',budget)
    frozen=args.output/'launch-source.py';shutil.copy2(Path(launch_training.__file__),frozen)
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'mjlab_training/src')+':'+str(ROOT)
    with (args.output/'train.log').open('x') as log:
        process=subprocess.Popen([str(python),str(frozen.resolve()),'worker','--output',str(args.output.resolve())],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(args.output/'process.json',dict(pid=process.pid,launched_unix=time.time(),worker_sha256=sha(frozen)))
    print(json.dumps(dict(pid=process.pid,budget=target,additional_iterations=args.iterations,num_envs=args.num_envs,log_dir=budget['log_dir'])))


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('pack');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('capacity');p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--resume',type=Path)
    p=sub.add_parser('start');p.add_argument('--output',type=Path,required=True);p.add_argument('--precheck',type=Path,required=True)
    p.add_argument('--capacity',type=Path,required=True);p.add_argument('--log-dir',type=Path,required=True)
    p.add_argument('--iterations',type=int,required=True);p.add_argument('--num-envs',type=int,default=4096)
    p.add_argument('--resume',type=Path)
    args=parser.parse_args();{'pack':pack,'verify':verify,'capacity':capacity,'start':start}[args.mode](args)


if __name__=='__main__':main()
