"""Explicit V14 continuation with reachable height commands and velocity gating."""
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

from . import huananhu_v14 as base
from ..config import ROOT
from ..reference_training.entry import serializable
from ..reference_training.huananhu_observations import OBSERVATION_CONTRACT
from ..reference_training.pd_action import CONTROL_CONTRACT
from ..reference_training.runner import DirectMotorOnPolicyRunner

INTERFACE = 'chuanliantui-mjlab-scut-v14-height-v1'
SOURCE_MODULE_SHA256 = 'a0f65605edad045e13c06efdd5ee05822a6854cd65c64e5b43e8227b472d66d1'
MIGRATION_CHECKPOINT_SHA256 = '0201396ba38378abba8ba986aa8e32405d7ebdb874eecd119911ee354939fda7'
HEIGHT_RANGE = (.27, .34)
HEIGHT_GATE = (.05, .10)
GATED_TERMS = ('track_lin_vel_xy', 'track_lin_vel_xy_square',
               'track_ang_vel_z', 'track_ang_vel_z_square')


class RewardState(base.RewardState):
    def compute(self, env):
        if self.step == env.common_step_counter:
            return self.values
        values = super().compute(env)
        command = env.command_manager.get_term('standup')
        height = getattr(command, 'height_above_ground', self.robot.data.root_link_pos_w[:, 2])
        error = (height - command.command[:, 2]).abs()
        full, off = HEIGHT_GATE
        gate = torch.nan_to_num(((off - error) / (off - full)).clamp(0., 1.),
                                nan=0., posinf=0., neginf=0.)
        for name in GATED_TERMS:
            values[name] = values[name] * gate
        return values


def reward(env, *, name):
    if not hasattr(env, '_huananhu_rewards'):
        env._huananhu_rewards = RewardState(env)
    return env._huananhu_rewards.compute(env)[name]


def make_env_cfg(num_envs=4096, seed=42, play=False):
    cfg = base.make_env_cfg(num_envs, seed, play)
    cfg.commands['standup'].height_range = HEIGHT_RANGE
    for term in cfg.rewards.values():
        term.func = reward
    return cfg


def runner_cfg(seed=42):
    cfg = base.runner_cfg(seed)
    cfg['experiment_name'] = 'chuanliantui_scut_v14_height'
    return cfg


def manifest():
    original = base.manifest()
    if original['module_sha256'] != SOURCE_MODULE_SHA256:
        raise ValueError('Height continuation requires the frozen original V14 kernel')
    return dict(interface=INTERFACE, source=original,
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        height_range=HEIGHT_RANGE, velocity_height_gate=HEIGHT_GATE, gated_terms=GATED_TERMS,
        weights=base.WEIGHTS, environment=serializable(make_env_cfg()), ppo=runner_cfg(),
        pull_up_assistance=False, changes=['height command range only', 'four velocity reward gates only'])


def validate_checkpoint(saved, *, migration=False):
    if migration:
        infos = base.validate_checkpoint(saved)
        if infos['completed_updates'] != 1000:
            raise ValueError('This migration is for the evaluated 1000-update V14 source')
    else:
        infos = saved.get('infos') or {}
        if (json.dumps(infos.get('interface'), sort_keys=True) != json.dumps(manifest(), sort_keys=True)
                or infos.get('control_contract') != CONTROL_CONTRACT
                or infos.get('observation_contract') != OBSERVATION_CONTRACT
                or infos.get('experiment_profile_contract') != INTERFACE):
            raise ValueError('V14 height profile mismatch; original V14 requires --resume-v14-height-fix')
    if any(name not in saved for name in ('actor_state_dict', 'critic_state_dict',
                                         'optimizer_state_dict', 'iter')):
        raise ValueError('Full V14 continuation requires actor, critic and optimizer')
    updates = infos.get('completed_updates')
    if not isinstance(updates, int) or updates < 0:
        raise ValueError('Missing completed update counter')
    return infos


class HeightVecEnv(base.V14VecEnv):
    def __init__(self, num_envs=4096, seed=42, device='cuda:0', play=False):
        torch.manual_seed(seed)
        RslRlVecEnvWrapper.__init__(self,
            ManagerBasedRlEnv(make_env_cfg(num_envs, seed, play), device=device), clip_actions=None)
        self.obs = self.get_observations()


class HeightRunner(base.V14Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.migration_history = []

    def save(self, path, infos=None):
        updates = self.current_learning_iteration + 1 if self._learning else self._completed_updates
        DirectMotorOnPolicyRunner.save(self, path, {**(infos or {}), 'interface': manifest(),
            'observation_contract': OBSERVATION_CONTRACT, 'experiment_profile_contract': INTERFACE,
            'completed_updates': updates, 'environment': serializable(self.env.unwrapped.cfg),
            'migration_history': self.migration_history})

    def load(self, path, *, migration=False, **kwargs):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        infos = validate_checkpoint(saved, migration=migration)
        previous = json.loads(json.dumps(infos.get('environment')))
        current = serializable(self.env.unwrapped.cfg)
        if migration:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != MIGRATION_CHECKPOINT_SHA256:
                raise ValueError('Migration requires the evaluated model_999.pt fingerprint')
            if previous['commands']['standup']['height_range'] != [.20, .42]:
                raise ValueError('Unexpected original height range')
            previous['commands']['standup']['height_range'] = list(HEIGHT_RANGE)
            for term in previous['rewards'].values():
                if term['func'] != 'ct_mjlab.experiments.huananhu_v14.reward':
                    raise ValueError('Unexpected source reward function')
                term['func'] = 'ct_mjlab.experiments.huananhu_v14_height.reward'
        if previous != current:
            raise ValueError('Height migration permits only command range and reward gate adaptation')
        result = DirectMotorOnPolicyRunner.load(self, path, **kwargs)
        self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]['lr'])
        self._completed_updates = self.current_learning_iteration = infos['completed_updates']
        self.migration_history = list(infos.get('migration_history', []))
        if migration:
            self.migration_history.append(dict(source_checkpoint=str(Path(path).resolve()),
                source_sha256=MIGRATION_CHECKPOINT_SHA256, from_interface=base.INTERFACE,
                to_interface=INTERFACE, completed_updates=self._completed_updates,
                changes=manifest()['changes']))
        env = self.env.unwrapped
        command = env.command_manager.get_term('standup')
        command.training_iteration = self._completed_updates - env.common_step_counter // 24
        command._update_command(None)
        return result


def run(args):
    migration = getattr(args, 'resume_v14_height_fix', False)
    if os.environ.get('WHEELLEG_HUANANHU_INIT_CHECKPOINT'):
        raise ValueError('Explicit checkpoint or a random start required')
    if args.num_envs < 1 or (args.command == 'train' and args.iterations < 1) or (
            args.command == 'check' and args.steps < 1):
        raise ValueError('Counts must be positive')
    if args.command == 'play' and (not math.isfinite(args.seconds) or args.seconds <= 0):
        raise ValueError('Playback duration must be finite and positive')
    saved = None
    if getattr(args, 'checkpoint', None):
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        validate_checkpoint(saved)
    if getattr(args, 'resume', None):
        validate_checkpoint(torch.load(args.resume, map_location='cpu', weights_only=False), migration=migration)
    env = HeightVecEnv(args.num_envs, args.seed, args.device, args.command == 'play')
    try:
        if args.command == 'train':
            from .getup_curriculum import snapshot_source
            if '/' in args.run_name or '\\' in args.run_name:
                raise ValueError('run-name must be a label')
            path = ROOT / 'logs/mjlab/chuanliantui' / (
                datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_scut_v14_height'
                + ('_' + args.run_name if args.run_name else ''))
            path.mkdir(parents=True, exist_ok=False)
            cfg = runner_cfg(args.seed)
            cfg.update(run_name=args.run_name, max_iterations=args.iterations)
            for name, value in (('interface', manifest()), ('environment', serializable(env.unwrapped.cfg)),
                                ('runner', cfg)):
                (path / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
            snapshot_source(path)
            runner_type = HeightRunner
            if args.render:
                from ..training_viewer import ViewerOptions, viewer_runner_class
                runner_type = viewer_runner_class(HeightRunner, ViewerOptions(
                    fps=args.viewer_fps, num_envs=args.num_envs))
            runner = runner_type(env, cfg, str(path), device=args.device)
            if args.resume:
                runner.load(str(args.resume), migration=migration)
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
            print('V14_HEIGHT_CHECK ' + json.dumps(dict(finite=True, numerical=numerical,
                num_envs=args.num_envs, steps=args.steps, learning=False)), flush=True)
        else:
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
