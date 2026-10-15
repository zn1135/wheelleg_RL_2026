"""Recover-r14 commands and historical r5–r13 replay/export execution.

Experimental profiles own their execution in experiments/; profiles.py routes them.
"""
from contextlib import ExitStack
from datetime import datetime
import json

import torch

from ...config import ROOT, manifest, runner_cfg


def run(args, parser):
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
        from ...policy import export
        print(json.dumps(export(args.checkpoint,args.output),indent=2));return
    if args.command=='eval' or args.command=='play' and args.backend=='cpu':
        from ...policy import checkpoint_command_config
        from ...policy import load_actor, checkpoint_is_legacy, checkpoint_uses_constant_modes, checkpoint_policy_joint_order, checkpoint_knee_limit_weight, checkpoint_reset_config, checkpoint_height_gate
        from ...cpu import evaluate as legacy_evaluate
        from ...reference_training.evaluate import evaluate as current_evaluate
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
    from ...registry import make_task
    from ...policy import checkpoint_command_config
    from ...policy import checkpoint_is_legacy, checkpoint_uses_constant_modes, checkpoint_policy_joint_order, checkpoint_knee_limit_weight, checkpoint_reset_config, checkpoint_height_gate
    legacy=args.command=='play' and checkpoint_is_legacy(args.checkpoint)
    env=make_task(args.task,num_envs=args.num_envs,device=args.device,seed=args.seed,randomize=args.command=='train',
        random_initial_pose=args.command=='train' or (args.command=='play' and args.random_reset),legacy=legacy,
        **({} if legacy else {'dynamic_modes':not checkpoint_uses_constant_modes(getattr(args,'checkpoint',None)),
            'policy_joint_order':checkpoint_policy_joint_order(getattr(args,'checkpoint',None)),
            'knee_limit_weight':checkpoint_knee_limit_weight(getattr(args,'checkpoint',None)),
            'reset_config':checkpoint_reset_config(getattr(args,'checkpoint',None)),
            'height_gate':checkpoint_height_gate(getattr(args,'checkpoint',None)),
            'command_config':checkpoint_command_config(getattr(args,'checkpoint',None))}))
    with ExitStack() as cleanup:
        cleanup.callback(env.close)
        raw_env=env.unwrapped
        if args.command=='train':
            from ...policy import runner_class
            runner_type=runner_class()
            if args.render:
                from ...training_viewer import ViewerOptions, viewer_runner_class
                options=ViewerOptions(fps=args.viewer_fps,num_envs=args.num_envs)
                options.validate(env.num_envs)
                runner_type=viewer_runner_class(runner_type,options,close_env_on_error=False)
            if '/' in args.run_name or '\\' in args.run_name:raise ValueError('run-name must be a label')
            name=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+('_'+args.run_name if args.run_name else '')
            path=ROOT/'logs/mjlab/chuanliantui'/name;path.mkdir(parents=True,exist_ok=False)
            cfg=runner_cfg(args.seed);cfg['run_name']=args.run_name;cfg['max_iterations']=args.iterations
            (path/'interface.json').write_text(json.dumps(manifest(),indent=2)+'\n')
            (path/'runner.json').write_text(json.dumps(cfg,indent=2)+'\n')
            from ...reference_training.entry import serializable
            (path/'environment.json').write_text(json.dumps(serializable(raw_env.cfg),indent=2)+'\n')
            print(f'Training output: {path}',flush=True)
            from ...utils.provenance import snapshot_source
            snapshot_source(path)
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
        from ...policy import load_actor
        actor=load_actor(args.checkpoint,args.device) if args.checkpoint else None
        import mujoco
        import time
        viewer=None
        grid_viewer=None
        if args.render:
            if args.num_envs>1:
                from ...training_viewer import NativeTrainingViewer,ViewerOptions
                from types import SimpleNamespace
                grid_viewer=NativeTrainingViewer(env,ViewerOptions(num_envs=args.num_envs))
                cleanup.callback(grid_viewer.close)
                preview=SimpleNamespace(current_learning_iteration=-1)
                grid_viewer.open();grid_viewer.draw(preview,force=True)
            else:
                import mujoco.viewer
                viewer=mujoco.viewer.launch_passive(raw_env.sim.mj_model,raw_env.sim.mj_data)
                cleanup.callback(viewer.close)
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
