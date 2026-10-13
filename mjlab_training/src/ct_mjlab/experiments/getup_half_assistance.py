"""Fresh r15 training with half-weight assistance; historical kernels stay frozen."""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from . import getup_curriculum as base, getup_orientation as orientation
from ..reference_training.entry import serializable
from ..reference_training.huananhu_observations import OBSERVATION_CONTRACT
from ..reference_training.pd_action import CONTROL_CONTRACT
from ..reference_training.runner import DirectMotorOnPolicyRunner

MAX_WEIGHT_RATIO = .5
INTERFACES = ('chuanliantui-mjlab-getup-curriculum-r15-half-assistance-v1',
              'chuanliantui-mjlab-getup-curriculum-r15-orientation-half-assistance-v1')


def manifest(strong_orientation=False):
    original = (orientation if strong_orientation else base).manifest()
    result = json.loads(json.dumps(original))
    result.update(interface=INTERFACES[int(strong_orientation)],
                  parent_interface=original['interface'],
                  parent_module_sha256=original['module_sha256'],
                  module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result['pull_up_assistance']['max_weight_ratio'] = MAX_WEIGHT_RATIO
    return result


def make_env_cfg(num_envs=4096, seed=42, play=False, stage=0, strong_orientation=False):
    cfg = (orientation if strong_orientation else base).make_env_cfg(num_envs, seed, play, stage)
    cfg.actions['pd'].max_weight_ratio = MAX_WEIGHT_RATIO
    return cfg


class HalfAssistanceVecEnv(base.GetupVecEnv):
    def __init__(self, num_envs=4096, seed=42, device='cuda:0', play=False, stage=0,
                 strong_orientation=False):
        self.strong_orientation = strong_orientation
        torch.manual_seed(seed)
        RslRlVecEnvWrapper.__init__(self, ManagerBasedRlEnv(
            make_env_cfg(num_envs, seed, play, stage, strong_orientation), device=device),
            clip_actions=None)
        self.obs = self.get_observations()


def validate_checkpoint(saved, strong_orientation):
    infos = saved.get('infos') or {}
    expected = manifest(strong_orientation)
    if (json.dumps(infos.get('interface'), sort_keys=True) != json.dumps(expected, sort_keys=True)
            or infos.get('experiment_profile_contract') != expected['interface']
            or infos.get('control_contract') != CONTROL_CONTRACT
            or infos.get('observation_contract') != OBSERVATION_CONTRACT):
        raise ValueError('Half-assistance resume requires the exact same half-assistance profile')
    return infos


class HalfAssistanceRunner(base.GetupRunner):
    def save(self, path, infos=None):
        command = self.env.unwrapped.command_manager.get_term('standup')
        contract = manifest(self.env.strong_orientation)
        DirectMotorOnPolicyRunner.save(self, path, {
            **(infos or {}), 'interface': contract,
            'observation_contract': OBSERVATION_CONTRACT,
            'experiment_profile_contract': contract['interface'],
            'getup_curriculum': command.curriculum.state_dict(),
            'completed_updates': self.current_learning_iteration + 1 if self._learning else self._completed_updates,
            'migration_history': self.migration_history,
            'environment': serializable(self.env.unwrapped.cfg)})

    def load(self, path, **kwargs):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        infos = validate_checkpoint(saved, self.env.strong_orientation)
        if infos['environment'] != serializable(self.env.unwrapped.cfg):
            raise ValueError('Half-assistance resume environment mismatch')
        result = DirectMotorOnPolicyRunner.load(self, path, **kwargs)
        self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]['lr'])
        self.migration_history = list(infos.get('migration_history', []))
        self.current_learning_iteration = self._completed_updates = infos['completed_updates']
        command = self.env.unwrapped.command_manager.get_term('standup')
        command.training_iteration = infos['completed_updates'] - self.env.unwrapped.common_step_counter // 24
        command.curriculum.load(infos['getup_curriculum'])
        command._update_command(None)
        from ..reference_training.huananhu_observations import reset_observations
        ids = torch.arange(self.env.num_envs, device=self.env.device)
        reset_observations(self.env.unwrapped, ids)
        self.env.unwrapped.observation_manager.reset(ids)
        self.env.unwrapped.obs_buf = self.env.unwrapped.observation_manager.compute(update_history=True)
        self.env.obs = self.env.get_observations()
        return result


def run(args):
    if any(getattr(args, flag, False) for flag in ('resume_contact_force_fix', 'resume_pose_bias_fix',
            'resume_orientation_weight', 'resume_wide_height_gate', 'resume_disable_knee_limit')):
        raise ValueError('Half-assistance training does not implicitly migrate historical models')
    if os.environ.get('WHEELLEG_HUANANHU_INIT_CHECKPOINT'):
        raise ValueError('External initialization is not allowed')
    if args.num_envs < 1 or (args.command == 'train' and args.iterations < 1) or (
            args.command == 'check' and args.steps < 1):
        raise ValueError('Environment, iteration and step counts must be positive')
    saved, stage = None, 0
    checkpoint = getattr(args, 'checkpoint', None) or getattr(args, 'resume', None)
    if checkpoint:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        interface = ((saved.get('infos') or {}).get('interface') or {}).get('interface')
        strong = interface == INTERFACES[1]
        infos = validate_checkpoint(saved, strong)
        if args.command == 'play':
            stage = infos['getup_curriculum']['stage']
    else:
        strong = getattr(args, 'training_profile', None) == 'getup-r15-orientation'
    env = HalfAssistanceVecEnv(args.num_envs, args.seed, args.device,
                              args.command == 'play', stage, strong)
    try:
        if args.command == 'train':
            if '/' in args.run_name or '\\' in args.run_name:
                raise ValueError('run-name must be a label')
            label = 'getup_r15_half_assistance' + ('_orientation' if strong else '')
            path = base.ROOT / 'logs/mjlab/chuanliantui' / (
                datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + label
                + ('_' + args.run_name if args.run_name else ''))
            path.mkdir(parents=True, exist_ok=False)
            cfg = base.runner_cfg(args.seed)
            cfg.update(run_name=label, experiment_name='chuanliantui_' + label,
                       max_iterations=args.iterations)
            for name, value in (('interface', manifest(strong)),
                                ('environment', serializable(env.unwrapped.cfg)), ('runner', cfg)):
                (path / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
            base.snapshot_source(path)
            runner_type = HalfAssistanceRunner
            if args.render:
                from ..training_viewer import ViewerOptions, viewer_runner_class
                runner_type = viewer_runner_class(runner_type, ViewerOptions(
                    fps=args.viewer_fps, num_envs=args.num_envs))
            runner = runner_type(env, cfg, str(path), device=args.device)
            if args.resume:
                runner.load(str(args.resume))
            runner.save(str(path / ('model_resume_initial.pt' if args.resume else 'model_initial.pt')))
            print(f'Training output: {path}', flush=True)
            runner.learn(args.iterations, init_at_random_ep_len=False)
        elif args.command == 'check':
            numerical = 0
            for _ in range(args.steps):
                obs, value, _, _ = env.step(torch.zeros(args.num_envs, 6, device=args.device))
                assert obs['actor'].shape == (args.num_envs, 35) and obs['critic'].shape == (args.num_envs, 78)
                assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(value).all()
                numerical += int(env.unwrapped.command_manager.get_term('standup').numerical_failure.sum())
            print('HALF_ASSISTANCE_CHECK ' + json.dumps(dict(num_envs=args.num_envs, steps=args.steps,
                finite=True, numerical=numerical, max_weight_ratio=MAX_WEIGHT_RATIO)), flush=True)
        else:
            if saved is None or not math.isfinite(args.seconds) or args.seconds <= 0:
                raise ValueError('Playback requires a checkpoint and positive finite seconds')
            from ..policy import actor_model
            actor = actor_model().to(args.device).eval()
            actor.load_state_dict(saved['actor_state_dict'], strict=True)
            if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
                raise ValueError('Nonfinite actor parameters')
            viewer = None
            if args.render:
                from ..training_viewer import NativeTrainingViewer, ViewerOptions
                viewer = NativeTrainingViewer(env, ViewerOptions(num_envs=args.num_envs))
                viewer.open()
            try:
                for _ in range(round(args.seconds / env.unwrapped.step_dt)):
                    started = time.monotonic()
                    if viewer:
                        viewer.wait_if_paused(SimpleNamespace(current_learning_iteration=-1))
                        if not viewer.is_running():
                            break
                    with torch.inference_mode():
                        env.step(actor(env.obs))
                    if viewer:
                        viewer.draw(SimpleNamespace(current_learning_iteration=-1))
                        viewer.pace(started)
            finally:
                if viewer:
                    viewer.close()
    finally:
        env.close()
