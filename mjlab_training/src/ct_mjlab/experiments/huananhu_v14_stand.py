"""Explicit standing phase after the evaluated V14 height continuation."""
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

from . import huananhu_v14_height as base
from ..config import ROOT
from ..reference_training.entry import serializable
from ..reference_training.huananhu_observations import OBSERVATION_CONTRACT
from ..reference_training.pd_action import CONTROL_CONTRACT
from ..reference_training.runner import DirectMotorOnPolicyRunner

INTERFACE = 'chuanliantui-mjlab-scut-v14-stand-v1'
SOURCE_MODULE_SHA256 = '672647d77620c75ecfb03611fd6fe77267fe3850bba98649f66ac8c403d06bcc'
MIGRATION_CHECKPOINT_SHA256 = '1c3487a7da8d1fe74c418466ec4068f621c162e36406c14786c972a2839e7027'
HEIGHT_RANGE = (.30, .30)
HEIGHT_GATE = (.05, .10)
GATED_TERMS = ('track_lin_vel_xy', 'track_lin_vel_xy_square',
               'track_ang_vel_z', 'track_ang_vel_z_square')
HEIGHT_SIGMA = .01
LEARNING_RATE = 5e-5
WEIGHTS = {**base.base.WEIGHTS, 'track_height_exp_tight': 3., 'track_height_square': -3.}


class RewardState(base.RewardState):
    def compute(self, env):
        if self.step == env.common_step_counter:
            return self.values
        values = super().compute(env)
        command = env.command_manager.get_term('standup')
        height = getattr(command, 'height_above_ground', self.robot.data.root_link_pos_w[:, 2])
        error = height - command.command[:, 2]
        # Inherit the velocity gate exactly once, replacing only the tight height term.
        value = torch.exp(-error.square() / HEIGHT_SIGMA)
        values['track_height_exp_tight'] = torch.nan_to_num(value, nan=0., posinf=0., neginf=0.).masked_fill(
            command.numerical_failure, 0.)
        return values


def reward(env, *, name):
    if not hasattr(env, '_huananhu_rewards'):
        env._huananhu_rewards = RewardState(env)
    return env._huananhu_rewards.compute(env)[name]


def make_env_cfg(num_envs=4096, seed=42, play=False):
    cfg = base.make_env_cfg(num_envs, seed, play)
    cfg.commands['standup'].height_range = HEIGHT_RANGE
    cfg.commands['standup'].zero_velocity_commands = True
    cfg.commands['standup'].special_modes_enabled = False
    for name, term in cfg.rewards.items():
        term.func = reward
        term.weight = WEIGHTS[name]
    return cfg


def runner_cfg(seed=42):
    cfg = base.runner_cfg(seed)
    cfg['experiment_name'] = 'chuanliantui_scut_v14_stand'
    cfg['algorithm'].update(learning_rate=LEARNING_RATE, schedule='fixed')
    return cfg


def manifest():
    original = base.manifest()
    if original['module_sha256'] != SOURCE_MODULE_SHA256:
        raise ValueError('Standing phase requires the frozen V14 height profile')
    return dict(interface=INTERFACE, source=original,
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        height_range=HEIGHT_RANGE, velocity_height_gate=HEIGHT_GATE, gated_terms=GATED_TERMS,
        weights=WEIGHTS, height_sigma=HEIGHT_SIGMA, environment=serializable(make_env_cfg()), ppo=runner_cfg(),
        pull_up_assistance=False, changes=['fixed 0.30m zero-velocity standing phase; special modes off',
            'height exponential sigma 0.001 to 0.01; height weights +3/-3',
            'fixed 5e-5 learning rate; Adam state preserved'])


def validate_checkpoint(saved, *, migration=False):
    if migration:
        infos = base.validate_checkpoint(saved)
        if infos['completed_updates'] != 2000:
            raise ValueError('This migration is for the evaluated 2000-update V14 height source')
    else:
        infos = saved.get('infos') or {}
        if (json.dumps(infos.get('interface'), sort_keys=True) != json.dumps(manifest(), sort_keys=True)
                or infos.get('control_contract') != CONTROL_CONTRACT
                or infos.get('observation_contract') != OBSERVATION_CONTRACT
                or infos.get('experiment_profile_contract') != INTERFACE):
            raise ValueError('V14 stand profile mismatch; height V14 requires --resume-v14-stand-fix')
    if any(name not in saved for name in ('actor_state_dict', 'critic_state_dict',
                                         'optimizer_state_dict', 'iter')):
        raise ValueError('Full V14 continuation requires actor, critic and optimizer')
    updates = infos.get('completed_updates')
    if not isinstance(updates, int) or updates < 0:
        raise ValueError('Missing completed update counter')
    return infos


class StandVecEnv(base.HeightVecEnv):
    def __init__(self, num_envs=4096, seed=42, device='cuda:0', play=False):
        torch.manual_seed(seed)
        RslRlVecEnvWrapper.__init__(self,
            ManagerBasedRlEnv(make_env_cfg(num_envs, seed, play), device=device), clip_actions=None)
        self.obs = self.get_observations()


class StandRunner(base.HeightRunner):
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
                raise ValueError('Migration requires the evaluated model_1999.pt fingerprint')
            if previous['commands']['standup']['height_range'] != [.27, .34]:
                raise ValueError('Unexpected source height range')
            previous['commands']['standup']['height_range'] = list(HEIGHT_RANGE)
            previous['commands']['standup']['zero_velocity_commands'] = True
            previous['commands']['standup']['special_modes_enabled'] = False
            for name, term in previous['rewards'].items():
                if term['func'] != 'ct_mjlab.experiments.huananhu_v14_height.reward':
                    raise ValueError('Unexpected source reward function')
                term['func'] = 'ct_mjlab.experiments.huananhu_v14_stand.reward'
                term['weight'] = WEIGHTS[name]
        if previous != current:
            raise ValueError('Standing migration environment differs beyond the declared changes')
        result = DirectMotorOnPolicyRunner.load(self, path, **kwargs)
        if migration:
            for group in self.alg.optimizer.param_groups:
                group['lr'] = LEARNING_RATE
        self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]['lr'])
        if self.alg.learning_rate != LEARNING_RATE:
            raise ValueError('Standing continuation requires its fixed learning rate')
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
    migration = getattr(args, 'resume_v14_stand_fix', False)
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
    env = StandVecEnv(args.num_envs, args.seed, args.device, args.command == 'play')
    try:
        if args.command == 'train':
            from .getup_curriculum import snapshot_source
            if '/' in args.run_name or '\\' in args.run_name:
                raise ValueError('run-name must be a label')
            path = ROOT / 'logs/mjlab/chuanliantui' / (
                datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_scut_v14_stand'
                + ('_' + args.run_name if args.run_name else ''))
            path.mkdir(parents=True, exist_ok=False)
            cfg = runner_cfg(args.seed)
            cfg.update(run_name=args.run_name, max_iterations=args.iterations)
            for name, value in (('interface', manifest()), ('environment', serializable(env.unwrapped.cfg)),
                                ('runner', cfg)):
                (path / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
            snapshot_source(path)
            runner_type = StandRunner
            if args.render:
                from ..training_viewer import ViewerOptions, viewer_runner_class
                runner_type = viewer_runner_class(StandRunner, ViewerOptions(
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
            print('V14_STAND_CHECK ' + json.dumps(dict(finite=True, numerical=numerical,
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
