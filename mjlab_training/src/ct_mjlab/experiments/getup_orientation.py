"""Explicit r15-v3 resume with stronger orientation weights, unchanged formula."""
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

from . import getup_curriculum as base
from ..reference_training.entry import serializable
from ..reference_training.huananhu_observations import OBSERVATION_CONTRACT, reset_observations
from ..reference_training.pd_action import CONTROL_CONTRACT
from ..reference_training.runner import DirectMotorOnPolicyRunner

INTERFACE = 'chuanliantui-mjlab-getup-curriculum-r15-orientation-weight-v1'
SOURCE_MODULE_SHA256 = '9725ffa5830d9346fb77a491a6a9f312a1dd54d6d88cc628fe6611e63c470e1c'
ORIENTATION_WEIGHTS = (-30., -30., -30.)
WEIGHTS = tuple({**weights, 'orientation': ORIENTATION_WEIGHTS[stage]}
                for stage, weights in enumerate(base.WEIGHTS))


def reward(env, *, name):
    if name != 'orientation':
        raise ValueError('Only the orientation term may use this adapter')
    if not hasattr(env, '_getup_rewards'):
        env._getup_rewards = base.RewardState(env)
    stage = env.command_manager.get_term('standup').curriculum.stage
    # Reuse the exact raw formula, caching and numerical masking from r15-v3.
    return env._getup_rewards.compute(env)['orientation'] * ORIENTATION_WEIGHTS[stage]


def make_env_cfg(num_envs=4096, seed=42, play=False, stage=0):
    cfg = base.make_env_cfg(num_envs, seed, play, stage)
    cfg.rewards['orientation'].func = reward
    return cfg


def manifest():
    original = base.manifest()
    if original['module_sha256'] != SOURCE_MODULE_SHA256:
        raise ValueError('Orientation-only profile requires the frozen r15-v3 reward kernel')
    result = json.loads(json.dumps(original))
    result.update(interface=INTERFACE, weights=WEIGHTS,
                  module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  getup_module_sha256=SOURCE_MODULE_SHA256,
                  orientation_change=dict(formula='unchanged r15-v3',
                      previous_weights=[w['orientation'] for w in base.WEIGHTS],
                      weights=ORIENTATION_WEIGHTS, scope='orientation weights only'))
    return result


def validate_checkpoint(saved, *, allow_orientation_weight=False):
    if allow_orientation_weight:
        infos = base.validate_checkpoint(saved)
        if infos['interface']['module_sha256'] != SOURCE_MODULE_SHA256:
            raise ValueError('Unrecognized r15-v3 orientation migration source')
        return infos
    infos = saved.get('infos') or {}
    if (json.dumps(infos.get('interface'), sort_keys=True) != json.dumps(manifest(), sort_keys=True)
            or infos.get('control_contract') != CONTROL_CONTRACT
            or infos.get('observation_contract') != OBSERVATION_CONTRACT
            or infos.get('experiment_profile_contract') != INTERFACE):
        raise ValueError('Orientation profile mismatch; v3 requires --resume-orientation-weight')
    return infos


class OrientationVecEnv(base.GetupVecEnv):
    def __init__(self, num_envs=4096, seed=42, device='cuda:0', play=False, stage=0):
        torch.manual_seed(seed)
        RslRlVecEnvWrapper.__init__(self,
            ManagerBasedRlEnv(make_env_cfg(num_envs, seed, play, stage), device=device),
            clip_actions=None)
        self.obs = self.get_observations()


class OrientationRunner(base.GetupRunner):
    def save(self, path, infos=None):
        command = self.env.unwrapped.command_manager.get_term('standup')
        DirectMotorOnPolicyRunner.save(self, path, {
            **(infos or {}), 'interface': manifest(),
            'observation_contract': OBSERVATION_CONTRACT,
            'experiment_profile_contract': INTERFACE,
            'getup_curriculum': command.curriculum.state_dict(),
            'completed_updates': self.current_learning_iteration + 1 if self._learning else self._completed_updates,
            'migration_history': self.migration_history,
            'environment': serializable(self.env.unwrapped.cfg)})

    def load(self, path, *, allow_orientation_weight=False, **kwargs):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        infos = validate_checkpoint(saved, allow_orientation_weight=allow_orientation_weight)
        previous = json.loads(json.dumps(infos['environment']))
        current = serializable(self.env.unwrapped.cfg)
        if allow_orientation_weight:
            if previous['rewards']['orientation']['func'] != (
                    'ct_mjlab.experiments.getup_curriculum.reward'):
                raise ValueError('Unexpected source orientation reward')
            previous['rewards']['orientation']['func'] = current['rewards']['orientation']['func']
        if previous != current:
            raise ValueError('Resume may change only orientation weights, not physics/reset/PPO environment')
        result = DirectMotorOnPolicyRunner.load(self, path, **kwargs)
        self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]['lr'])
        self.migration_history = list(infos.get('migration_history', []))
        if allow_orientation_weight:
            self.migration_history.append(dict(change='increase orientation weights; preserve raw formula',
                from_interface=base.INTERFACE, to_interface=INTERFACE,
                source_checkpoint=str(Path(path).resolve()),
                source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                completed_updates=infos['completed_updates'],
                previous_weights=[w['orientation'] for w in base.WEIGHTS],
                weights=list(ORIENTATION_WEIGHTS)))
        self.current_learning_iteration = self._completed_updates = infos['completed_updates']
        command = self.env.unwrapped.command_manager.get_term('standup')
        command.training_iteration = infos['completed_updates'] - self.env.unwrapped.common_step_counter // 24
        command.curriculum.load(infos['getup_curriculum'])
        command._update_command(None)
        ids = torch.arange(self.env.num_envs, device=self.env.device)
        reset_observations(self.env.unwrapped, ids)
        self.env.unwrapped.observation_manager.reset(ids)
        self.env.unwrapped.obs_buf = self.env.unwrapped.observation_manager.compute(update_history=True)
        self.env.obs = self.env.get_observations()
        return result


def run(args):
    migration = getattr(args, 'resume_orientation_weight', False)
    if migration and (args.command != 'train' or not getattr(args, 'resume', None)):
        raise ValueError('Orientation migration requires train --resume')
    if os.environ.get('WHEELLEG_HUANANHU_INIT_CHECKPOINT'):
        raise ValueError('Explicit checkpoint required; external initialization is not allowed')
    if args.num_envs < 1 or (args.command == 'train' and args.iterations < 1) or (
            args.command == 'check' and args.steps < 1):
        raise ValueError('Environment, iteration and step counts must be positive')
    if args.command == 'play' and (not math.isfinite(args.seconds) or args.seconds <= 0):
        raise ValueError('Playback duration must be finite and positive')
    manifest()  # Verify the unchanged base reward kernel before building an env.
    saved, stage = None, 0
    if getattr(args, 'checkpoint', None):
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        stage = validate_checkpoint(saved)['getup_curriculum']['stage']
    env = OrientationVecEnv(args.num_envs, args.seed, args.device, args.command == 'play', stage)
    try:
        if args.command == 'train':
            label = args.run_name
            if '/' in label or '\\' in label:
                raise ValueError('run-name must be a label')
            path = base.ROOT / 'logs/mjlab/chuanliantui' / (
                datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_getup_r15_orientation'
                + ('_' + label if label else ''))
            path.mkdir(parents=True, exist_ok=False)
            cfg = base.runner_cfg(args.seed)
            cfg.update(run_name='getup_r15_orientation', experiment_name='chuanliantui_getup_r15_orientation',
                       max_iterations=args.iterations)
            for name, value in (('interface', manifest()), ('environment', serializable(env.unwrapped.cfg)),
                                ('runner', cfg)):
                (path / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
            base.snapshot_source(path)
            runner_type = OrientationRunner
            if args.render:
                from ..training_viewer import ViewerOptions, viewer_runner_class
                runner_type = viewer_runner_class(OrientationRunner, ViewerOptions(
                    fps=args.viewer_fps, num_envs=args.num_envs))
            runner = runner_type(env, cfg, str(path), device=args.device)
            if args.resume:
                runner.load(str(args.resume), allow_orientation_weight=migration)
                runner.save(str(path / 'model_resume_initial.pt'))
            else:
                runner.save(str(path / 'model_initial.pt'))
            print(f'Training output: {path}', flush=True)
            runner.learn(args.iterations, init_at_random_ep_len=False)
        elif args.command == 'check':
            numerical = 0
            for _ in range(args.steps):
                obs, value, _, _ = env.step(torch.zeros(args.num_envs, 6, device=args.device))
                assert obs['actor'].shape == (args.num_envs, 35) and obs['critic'].shape == (args.num_envs, 78)
                assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(value).all()
                numerical += int(env.unwrapped.command_manager.get_term('standup').numerical_failure.sum())
            print('ORIENTATION_CHECK ' + json.dumps(dict(num_envs=args.num_envs, steps=args.steps,
                  finite=True, numerical=numerical, profile=INTERFACE)), flush=True)
        else:
            if saved is None:
                raise ValueError('Orientation playback requires --checkpoint')
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
