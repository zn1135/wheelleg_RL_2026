"""Entrypoints for the independent mjlab chuanliantui task."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import torch
from .config import ROOT, manifest, runner_cfg

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('train','play','eval','export','check'):
        p=sub.add_parser(name)
        p.add_argument('--task',choices=['chuanliantui'],default='chuanliantui')
        p.add_argument('--seed',type=int,default=42)
        if name in ('train','play','check'):
            p.add_argument('--num-envs',type=int,default=4096 if name=='train' else 1)
            p.add_argument('--device',default='cuda:0')
        if name in ('train','check'):
            p.add_argument('--training-profile', choices=('huananhu-v14', 'huananhu-v14-height', 'huananhu-v14-stand', 'huananhu-v14-recovery', 'huananhu-v14-wheel-support', 'getup-r15', 'getup-r15-orientation', 'recover-r14'),
                default='huananhu-v14', help='Original SCUT V14 Flat-v0 (default), r15 curriculum or r14 recover')
        if name in ('play','eval','export'):p.add_argument('--checkpoint',type=Path,required=name=='export')
        if name in ('play','eval'):
            p.add_argument('--seconds',type=float,default=30)
            p.add_argument('--render',action='store_true');p.add_argument('--report',type=Path)
        if name=='play':
            p.add_argument('--backend',choices=['mjlab','cpu'],default='mjlab')
            p.add_argument('--random-reset',action='store_true',help='Preview the training pose distribution')
        if name=='export':p.add_argument('--output',type=Path,required=True)
        if name=='train':
            p.add_argument('--iterations',type=int,default=20000);p.add_argument('--run-name',default='')
            p.add_argument('--resume',type=Path)
            p.add_argument('--resume-v14-wheel-support', action='store_true',
                help='Explicit known recovery model2999 migration to wheel-to-ground load-bearing goals')
            p.add_argument('--resume-v14-recovery-fix', action='store_true',
                help='Explicit evaluated 2000-update height model migration to recovery and height curriculum')
            p.add_argument('--resume-v14-stand-fix', action='store_true',
                help='Explicit evaluated V14 height model_1999 migration to the standing phase')
            p.add_argument('--resume-v14-height-fix', action='store_true',
                help='Explicit evaluated V14 model_999 migration: height range and velocity height gate')
            p.add_argument('--resume-contact-force-fix', action='store_true',
                help='Historical v1 to v2 migration; use the archived v2 release')
            p.add_argument('--resume-pose-bias-fix', action='store_true',
                help='Explicit known assisted-r15 v2 to v3 removal of pose_bias in all stages')
            p.add_argument('--resume-orientation-weight', action='store_true',
                help='Explicit r15-v3 migration with stronger orientation weights and unchanged formula')
            p.add_argument('--resume-wide-height-gate',action='store_true',
                help='Archived r11 to r13 gate-only migration; unavailable in current r14')
            p.add_argument('--resume-disable-knee-limit',action='store_true',
                help='Historical r11 to r12 knee-only migration; unavailable in current r14')
            p.add_argument('--render',action='store_true',help='Show the actual PPO environments in a native grid')
            p.add_argument('--viewer-fps',type=float,default=30)
        if name=='check':p.add_argument('--steps',type=int,default=32)
    args=parser.parse_args();torch.manual_seed(args.seed)
    support_migration = getattr(args, 'resume_v14_wheel_support', False)
    if support_migration and (args.training_profile != 'huananhu-v14-wheel-support' or not args.resume):
        parser.error('--resume-v14-wheel-support requires huananhu-v14-wheel-support and --resume')
    wheel_support = args.command in ('train', 'check') and args.training_profile == 'huananhu-v14-wheel-support'
    if args.command in ('play', 'eval', 'export') and args.checkpoint:
        metadata = torch.load(args.checkpoint, map_location='cpu', weights_only=False).get('infos') or {}
        wheel_support = (metadata.get('interface') or {}).get('interface') == 'chuanliantui-mjlab-v14-wheel-support-v1'
        if wheel_support and (args.command != 'play' or args.backend != 'mjlab' or args.report):
            parser.error('Wheel support profile currently supports native mjlab play')
    if wheel_support:
        if any(getattr(args, flag, False) for flag in ('resume_v14_recovery_fix', 'resume_v14_height_fix',
                'resume_v14_stand_fix', 'resume_contact_force_fix', 'resume_pose_bias_fix',
                'resume_orientation_weight', 'resume_wide_height_gate', 'resume_disable_knee_limit')):
            parser.error('Wheel support profile accepts only --resume-v14-wheel-support or its own checkpoint')
        from .experiments.huananhu_v14_wheel_support import run
        run(args)
        return
    recovery_migration = getattr(args, 'resume_v14_recovery_fix', False)
    if recovery_migration and (args.training_profile != 'huananhu-v14-recovery' or not args.resume):
        parser.error('--resume-v14-recovery-fix requires huananhu-v14-recovery and --resume')
    recovery_v14 = args.command in ('train', 'check') and args.training_profile == 'huananhu-v14-recovery'
    if args.command in ('play', 'eval', 'export') and args.checkpoint:
        metadata = torch.load(args.checkpoint, map_location='cpu', weights_only=False).get('infos') or {}
        recovery_v14 = (metadata.get('interface') or {}).get('interface') == 'chuanliantui-mjlab-v14-recovery-height-v1'
        if recovery_v14 and (args.command != 'play' or args.backend != 'mjlab' or args.report):
            parser.error('Recovery height profile currently supports native mjlab play; use its independent evaluator for reports')
    if recovery_v14:
        if any(getattr(args, flag, False) for flag in ('resume_v14_height_fix', 'resume_v14_stand_fix',
                'resume_contact_force_fix', 'resume_pose_bias_fix', 'resume_orientation_weight',
                'resume_wide_height_gate', 'resume_disable_knee_limit')):
            parser.error('Recovery profile accepts only --resume-v14-recovery-fix or its own checkpoint')
        from .experiments.huananhu_v14_recovery import run
        run(args)
        return
    if getattr(args, 'resume_v14_stand_fix', False) and (
            args.training_profile != 'huananhu-v14-stand' or not args.resume):
        parser.error('--resume-v14-stand-fix requires huananhu-v14-stand and --resume')
    stand_v14 = args.command in ('train', 'check') and args.training_profile == 'huananhu-v14-stand'
    if args.command == 'play' and args.checkpoint:
        metadata = torch.load(args.checkpoint, map_location='cpu', weights_only=False).get('infos') or {}
        stand_v14 = (metadata.get('interface') or {}).get('interface') == 'chuanliantui-mjlab-scut-v14-stand-v1'
        if stand_v14 and (args.backend != 'mjlab' or args.report):
            parser.error('V14 standing playback requires mjlab')
    if stand_v14:
        if any(getattr(args, flag, False) for flag in ('resume_v14_height_fix', 'resume_contact_force_fix',
                'resume_pose_bias_fix', 'resume_orientation_weight', 'resume_wide_height_gate',
                'resume_disable_knee_limit')):
            parser.error('V14 standing accepts only --resume-v14-stand-fix or its own checkpoint')
        from .experiments.huananhu_v14_stand import run
        run(args)
        return
    if getattr(args, 'resume_v14_height_fix', False) and (
            args.training_profile != 'huananhu-v14-height' or not args.resume):
        parser.error('--resume-v14-height-fix requires huananhu-v14-height and --resume')
    height_v14 = args.command in ('train', 'check') and args.training_profile == 'huananhu-v14-height'
    if args.command == 'play' and args.checkpoint:
        metadata = torch.load(args.checkpoint, map_location='cpu', weights_only=False).get('infos') or {}
        height_v14 = (metadata.get('interface') or {}).get('interface') == 'chuanliantui-mjlab-scut-v14-height-v1'
        if height_v14 and (args.backend != 'mjlab' or args.report):
            parser.error('V14 height playback requires mjlab')
    if height_v14:
        if any(getattr(args, flag, False) for flag in ('resume_contact_force_fix', 'resume_pose_bias_fix',
                'resume_orientation_weight', 'resume_wide_height_gate', 'resume_disable_knee_limit')):
            parser.error('V14 height profile accepts only --resume-v14-height-fix or its own checkpoint')
        from .experiments.huananhu_v14_height import run
        run(args)
        return
    original_v14 = args.command in ('train', 'check') and args.training_profile == 'huananhu-v14'
    if args.command == 'play' and args.checkpoint:
        metadata = torch.load(args.checkpoint, map_location='cpu', weights_only=False).get('infos') or {}
        original_v14 = (metadata.get('interface') or {}).get('interface') == 'chuanliantui-mjlab-scut-v14-flat-v1'
        if original_v14 and (args.backend != 'mjlab' or args.report):
            parser.error('Original V14 playback requires mjlab; CPU evaluation is not implemented')
    if original_v14:
        if any(getattr(args, flag, False) for flag in ('resume_contact_force_fix', 'resume_pose_bias_fix',
                'resume_orientation_weight', 'resume_wide_height_gate', 'resume_disable_knee_limit')):
            parser.error('Original V14 accepts only its own complete checkpoint, without migration flags')
        from .experiments.huananhu_v14 import run
        run(args)
        return
    # Fresh r15 runs use 50% assistance; historical resume keeps its own schedule.
    half_assistance = False
    if args.command in ('train', 'check') and args.training_profile in ('getup-r15', 'getup-r15-orientation'):
        if not getattr(args, 'resume', None):
            half_assistance = True
        else:
            metadata = torch.load(args.resume, map_location='cpu', weights_only=False).get('infos') or {}
            from .experiments.getup_half_assistance import INTERFACES
            half_assistance = (metadata.get('interface') or {}).get('interface') in INTERFACES
    if args.command == 'play' and args.checkpoint:
        metadata = torch.load(args.checkpoint, map_location='cpu', weights_only=False).get('infos') or {}
        half_assistance = (metadata.get('interface') or {}).get('interface') in (
            'chuanliantui-mjlab-getup-curriculum-r15-half-assistance-v1',
            'chuanliantui-mjlab-getup-curriculum-r15-orientation-half-assistance-v1')
        if half_assistance and (args.backend != 'mjlab' or args.report):
            parser.error('Half-assistance playback requires mjlab; CPU evaluation is not implemented')
    if half_assistance:
        from .experiments.getup_half_assistance import run
        run(args)
        return
    if getattr(args, 'resume_orientation_weight', False) and (
            args.training_profile != 'getup-r15-orientation' or not args.resume):
        parser.error('--resume-orientation-weight requires getup-r15-orientation and --resume')
    if args.command in ('train','check') and args.training_profile == 'getup-r15-orientation':
        if any(getattr(args, flag, False) for flag in ('resume_contact_force_fix',
                'resume_pose_bias_fix', 'resume_wide_height_gate', 'resume_disable_knee_limit')):
            parser.error('Orientation profile accepts only its own resume or explicit r15-v3 migration')
        from .experiments.getup_orientation import run
        run(args)
        return
    if getattr(args, 'resume_contact_force_fix', False) and (
            args.training_profile != 'getup-r15' or not args.resume):
        parser.error('--resume-contact-force-fix requires getup-r15 and --resume')
    if getattr(args, 'resume_pose_bias_fix', False) and (
            args.training_profile != 'getup-r15' or not args.resume):
        parser.error('--resume-pose-bias-fix requires getup-r15 and --resume')
    if args.command in ('train','check') and args.training_profile == 'getup-r15':
        if getattr(args, 'resume_wide_height_gate', False) or getattr(args, 'resume_disable_knee_limit', False):
            parser.error('r15 requires its own checkpoint; historical migration flags cannot be used')
        from .experiments.getup_curriculum import run
        run(args)
        return
    if args.command == 'play' and args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if ((checkpoint.get('infos') or {}).get('interface') or {}).get('interface') == (
                'chuanliantui-mjlab-getup-curriculum-r15-orientation-weight-v1'):
            if args.backend != 'mjlab' or args.report:
                parser.error('Orientation playback uses --backend mjlab; CPU evaluation is not implemented')
            from .experiments.getup_orientation import run
            run(args)
            return
        if ((checkpoint.get('infos') or {}).get('interface') or {}).get('interface') in (
                'chuanliantui-mjlab-getup-curriculum-r15',
                'chuanliantui-mjlab-getup-curriculum-r15-assisted-v1',
                'chuanliantui-mjlab-getup-curriculum-r15-assisted-v2',
                'chuanliantui-mjlab-getup-curriculum-r15-assisted-v3'):
            if args.backend != 'mjlab' or args.report:
                parser.error('r15 playback currently uses --backend mjlab; CPU evaluation is not implemented')
            from .experiments.getup_curriculum import run
            run(args)
            return
    if args.command=='train' and args.resume_disable_knee_limit and not args.resume:
        parser.error('--resume-disable-knee-limit requires --resume')
    if args.command=='train' and args.resume_disable_knee_limit:
        parser.error('--resume-disable-knee-limit belongs to historical r12; current r14 changes the gate and commands')
    if args.command=='train' and args.resume_wide_height_gate:
        parser.error('--resume-wide-height-gate belongs to archived r13; current r14 also enables velocity commands')
    if args.command in ('play','eval') and args.seconds<=0:
        parser.error('--seconds must be positive')
    if args.command=='play' and args.backend=='mjlab' and args.report:
        parser.error('--report is available with --backend cpu; use eval for behavior reports')
    if args.command=='play' and args.backend=='cpu' and args.random_reset:
        parser.error('--random-reset currently requires the mjlab backend')
    if args.command=='export':
        from .policy import export
        print(json.dumps(export(args.checkpoint,args.output),indent=2));return
    if args.command=='eval' or args.command=='play' and args.backend=='cpu':
        from .policy import checkpoint_command_config
        from .policy import load_actor, checkpoint_is_legacy, checkpoint_uses_constant_modes, checkpoint_policy_joint_order, checkpoint_knee_limit_weight, checkpoint_reset_config, checkpoint_height_gate
        from .cpu import evaluate as legacy_evaluate
        from .reference_training.evaluate import evaluate as current_evaluate
        actor=load_actor(args.checkpoint) if args.checkpoint else None
        if checkpoint_is_legacy(args.checkpoint):
            result=legacy_evaluate(actor,args.seconds,args.render,args.report)
        else:
            result=current_evaluate(actor,args.seconds,args.render,args.report,
                dynamic_modes=not checkpoint_uses_constant_modes(args.checkpoint),
                policy_joint_order=checkpoint_policy_joint_order(args.checkpoint),
                knee_limit_weight=checkpoint_knee_limit_weight(args.checkpoint),
                reset_config=checkpoint_reset_config(args.checkpoint),height_gate=checkpoint_height_gate(args.checkpoint),
                command_config=checkpoint_command_config(args.checkpoint))
        print(json.dumps(result,indent=2));return
    from .registry import make_task
    from .policy import checkpoint_command_config
    from .policy import checkpoint_is_legacy, checkpoint_uses_constant_modes, checkpoint_policy_joint_order, checkpoint_knee_limit_weight, checkpoint_reset_config, checkpoint_height_gate
    legacy=args.command=='play' and checkpoint_is_legacy(args.checkpoint)
    env=make_task(args.task,num_envs=args.num_envs,device=args.device,seed=args.seed,randomize=args.command=='train',
        random_initial_pose=args.command=='train' or (args.command=='play' and args.random_reset),legacy=legacy,
        **({} if legacy else {'dynamic_modes':not checkpoint_uses_constant_modes(getattr(args,'checkpoint',None)),
            'policy_joint_order':checkpoint_policy_joint_order(getattr(args,'checkpoint',None)),
            'knee_limit_weight':checkpoint_knee_limit_weight(getattr(args,'checkpoint',None)),
            'reset_config':checkpoint_reset_config(getattr(args,'checkpoint',None)),
            'height_gate':checkpoint_height_gate(getattr(args,'checkpoint',None)),
            'command_config':checkpoint_command_config(getattr(args,'checkpoint',None))}))
    raw_env=env.unwrapped
    if args.command=='train':
        from .policy import runner_class
        runner_type=runner_class()
        if args.render:
            from .training_viewer import ViewerOptions, viewer_runner_class
            options=ViewerOptions(fps=args.viewer_fps,num_envs=args.num_envs)
            options.validate(env.num_envs)
            runner_type=viewer_runner_class(runner_type,options)
        if '/' in args.run_name or '\\' in args.run_name:raise ValueError('run-name must be a label')
        name=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+('_'+args.run_name if args.run_name else '')
        path=ROOT/'logs/mjlab/chuanliantui'/name;path.mkdir(parents=True,exist_ok=False)
        cfg=runner_cfg(args.seed);cfg['run_name']=args.run_name;cfg['max_iterations']=args.iterations
        (path/'interface.json').write_text(json.dumps(manifest(),indent=2)+'\n')
        (path/'runner.json').write_text(json.dumps(cfg,indent=2)+'\n')
        from .reference_training.entry import serializable
        (path/'environment.json').write_text(json.dumps(serializable(raw_env.cfg),indent=2)+'\n')
        print(f'Training output: {path}',flush=True)
        # Keep the dirty source state with this new run, including new assets.
        import subprocess
        import hashlib
        source_roots=('mjlab_training/','resources/robots/chuanliantui_new_1/','scripts/')
        if (ROOT/'.git').exists():
            changed=subprocess.check_output(['git','diff','HEAD','--name-only','-z'],cwd=ROOT).decode().split('\0')[:-1]
            added=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')[:-1]
            relevant=[name for name in sorted(set(changed+added)) if name.startswith(source_roots)]
            provenance=dict(
                head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
                branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip(),
                source_kind='git-workspace')
            source_diff=subprocess.check_output(['git','diff','HEAD','--','mjlab_training','resources/robots/chuanliantui_new_1','scripts'],cwd=ROOT)
        else:
            # SSH releases carry an immutable version record rather than .git.
            version=(ROOT/'source-version.txt').read_text()
            fields=dict(line.split('：',1) for line in version.splitlines() if '：' in line)
            head=fields.get('提交','')
            if len(head)!=40 or any(c not in '0123456789abcdef' for c in head):
                raise ValueError('Remote source snapshot is missing a valid source commit')
            provenance=dict(head=head,branch=None,source_kind='remote-release',release=fields.get('快照'))
            (path/'source-version.txt').write_text(version)
            relevant=sorted(str(p.relative_to(ROOT)) for name in source_roots for p in (ROOT/name).rglob('*')
                if p.is_file() and not p.is_symlink() and not set(p.relative_to(ROOT).parts)&
                   {'.venv','venv','__pycache__','logs','.git','.remote-training'}
                and p.suffix not in {'.pyc','.pyo'})
            source_diff=None
        hashes={}
        for name in relevant:
            source=ROOT/name
            if source.is_file() and not source.is_symlink():
                data=source.read_bytes();target=path/'source-snapshot'/name
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
                hashes[name]=hashlib.sha256(data).hexdigest()
        (path/'source-state.json').write_text(json.dumps(dict(
            **provenance,source_diff_available=source_diff is not None,
            changed_files_sha256=hashes,command=__import__('sys').argv),indent=2)+'\n')
        if source_diff is not None:(path/'source.diff').write_bytes(source_diff)
        runner=runner_type(env,cfg,str(path),device=args.device)
        if args.resume:
            if args.resume_wide_height_gate:
                runner.resume_with_wide_height_gate(str(args.resume))
                runner.save(str(path/'resume_state.pt'),infos={'checkpoint_kind':'resume_state'})
            elif args.resume_disable_knee_limit:
                runner.resume_without_knee_limit(str(args.resume))
                runner.save(str(path/'resume_state.pt'),infos={'checkpoint_kind':'resume_state'})
            else:runner.load(str(args.resume))
        runner.learn(args.iterations,init_at_random_ep_len=False);return
    if args.command=='check':
        for _ in range(args.steps):
            obs,rew,done,extras=env.step(torch.zeros(args.num_envs,6,device=args.device))
            assert obs['actor'].shape==(args.num_envs,35) and obs['critic'].shape==(args.num_envs,78)
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['critic']).all() and torch.isfinite(rew).all()
        print(json.dumps(dict(kind='finite physics/interface check; no training',steps=args.steps,**manifest()),indent=2));return
    from .policy import load_actor
    actor=load_actor(args.checkpoint,args.device) if args.checkpoint else None
    import mujoco
    import time
    viewer=None
    grid_viewer=None
    if args.render:
        if args.num_envs>1:
            from .training_viewer import NativeTrainingViewer,ViewerOptions
            from types import SimpleNamespace
            grid_viewer=NativeTrainingViewer(env,ViewerOptions(num_envs=args.num_envs))
            preview=SimpleNamespace(current_learning_iteration=-1)
            grid_viewer.open();grid_viewer.draw(preview,force=True)
        else:
            import mujoco.viewer
            viewer=mujoco.viewer.launch_passive(raw_env.sim.mj_model,raw_env.sim.mj_data)
    try:
        for _ in range(round(args.seconds/.02)):
            start=time.monotonic()
            if grid_viewer:
                grid_viewer.wait_if_paused(preview)
                if not grid_viewer.is_running():break
            with torch.inference_mode():action=actor(env.obs) if actor else torch.zeros(args.num_envs,6,device=args.device)
            env.step(action)
            if viewer:
                if not viewer.is_running():break
                d=raw_env.sim.mj_data;d.qpos[:]=raw_env.sim.data.qpos[0].cpu().numpy();d.qvel[:]=raw_env.sim.data.qvel[0].cpu().numpy()
                d.ctrl[:]=raw_env.sim.data.ctrl[0].cpu().numpy()
                mujoco.mj_forward(raw_env.sim.mj_model,d);viewer.sync();time.sleep(max(0,.02-time.monotonic()+start))
            if grid_viewer:
                grid_viewer.draw(preview);grid_viewer.pace(start)
    finally:
        if viewer:viewer.close()
        if grid_viewer:grid_viewer.close()
        env.close()

if __name__=='__main__':main()
