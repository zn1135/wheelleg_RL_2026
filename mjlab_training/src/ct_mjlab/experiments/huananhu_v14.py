"""Original SCUT V14 Flat-v0 recipe on the retained CAD robot and mjlab."""
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
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper

from ..config import ROOT, RECIPE, PARAMS, WEIGHTS, legacy_runner_cfg
from ..reference_training import huananhu_rewards
from ..reference_training.entry import manifest as hardware_manifest, serializable
from ..reference_training.huananhu_env_cfg import make_huananhu_env_cfg
from ..reference_training.huananhu_observations import OBSERVATION_CONTRACT
from ..reference_training.pd_action import CONTROL_CONTRACT
from ..reference_training.runner import DirectMotorOnPolicyRunner

INTERFACE = 'chuanliantui-mjlab-scut-v14-flat-v1'


def velocity_terms(command, linear_velocity, angular_velocity, pitch):
    """Original clipped exponential tracking, without later height gates."""
    still = command[:, 0].abs() < PARAMS['stand_still_deadzone_threshold']
    vx = torch.where(still, 0., command[:, 0])
    yaw = torch.where(command[:, 1].abs() < .1, 0., command[:, 1])
    error = vx - linear_velocity[:, 0] * pitch.cos()
    yaw_error = yaw - angular_velocity[:, 2]
    return {
        'track_lin_vel_xy': torch.exp(-error.clamp(
            -PARAMS['lin_vel_err_constraint'], PARAMS['lin_vel_err_constraint']).square()
            / PARAMS['lin_vel_xy_sigma']),
        'track_lin_vel_xy_square': (PARAMS['lin_vel_xy_square_sigma'] * error).square(),
        'track_ang_vel_z': torch.exp(-yaw_error.clamp(
            -PARAMS['ang_vel_err_constraint'], PARAMS['ang_vel_err_constraint']).square()
            / PARAMS['ang_vel_z_sigma']),
        'track_ang_vel_z_square': (PARAMS['ang_vel_z_square_sigma'] * yaw_error).square(),
        'stand_still_lin_vel': still.float() * linear_velocity[:, :2].abs().sum(-1),
    }


class RewardState(huananhu_rewards.RewardState):
    def compute(self, env):
        if self.step == env.common_step_counter:
            return self.values
        values = super().compute(env)
        data = self.robot.data
        command = env.command_manager.get_term('standup').command
        velocity = velocity_terms(command, data.root_link_lin_vel_b,
            data.root_link_ang_vel_b, torch.asin(data.projected_gravity_b[:, 0].clamp(-1., 1.)))
        invalid = env.command_manager.get_term('standup').numerical_failure
        values.update({name: torch.nan_to_num(value, nan=0., posinf=0., neginf=0.).masked_fill(invalid, 0.)
                       for name, value in velocity.items()})
        self.values = {name: values[name] for name in WEIGHTS}
        return self.values


def reward(env, *, name):
    if not hasattr(env, '_huananhu_rewards'):
        env._huananhu_rewards = RewardState(env)
    return env._huananhu_rewards.compute(env)[name]


def make_env_cfg(num_envs=4096, seed=42, play=False):
    cfg = make_huananhu_env_cfg(play=play, recover=False, dynamic_modes=False)
    cfg.scene.num_envs, cfg.seed = num_envs, seed
    command = cfg.commands['standup']
    command.training_stage = 'scut_v14_flat'
    command.height_range = tuple(PARAMS['height_range'])
    command.special_modes_enabled = True
    command.strict_recovery_metrics = True
    cfg.actions['pd'].max_wheel_velocity = PARAMS['max_wheel_vel']
    cfg.rewards = {name: RewardTermCfg(func=reward, weight=weight, params={'name': name})
                   for name, weight in WEIGHTS.items()}
    return cfg


def runner_cfg(seed=42):
    cfg = legacy_runner_cfg(seed)
    cfg.update(experiment_name='chuanliantui_scut_v14_flat', clip_actions=None)
    return cfg


def manifest():
    return dict(interface=INTERFACE, source_commit=RECIPE['commit'], source_task=RECIPE['task'],
        source_files_sha256=RECIPE['files'], source_recipe_sha256=hashlib.sha256(
            Path(__file__).parents[1].joinpath('scut_v14.json').read_bytes()).hexdigest(),
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        robot_and_backend=hardware_manifest(), weights=WEIGHTS,
        environment=serializable(make_env_cfg()), ppo=runner_cfg(),
        modes='constant normal, as original V14; no batch getup curriculum',
        velocity_height_gate=False, pull_up_assistance=False,
        backend_adaptations=['CAD robot, physical joints and 0-150 N gas springs retained',
            'Closed-chain-safe 2 ms physics / 20 ms control, rounded sensor/action delays',
            'Retargeted legal pose bank and collision-clear root height',
            'MuJoCo contact and friction mapping; original PhysX is not reproduced'])


def validate_checkpoint(saved):
    infos = saved.get('infos') or {}
    if (json.dumps(infos.get('interface'), sort_keys=True) != json.dumps(manifest(), sort_keys=True)
            or infos.get('control_contract') != CONTROL_CONTRACT
            or infos.get('observation_contract') != OBSERVATION_CONTRACT
            or infos.get('experiment_profile_contract') != INTERFACE):
        raise ValueError('Original V14 profile mismatch; r14/r15 checkpoints require their own profile')
    if any(name not in saved for name in ('actor_state_dict', 'critic_state_dict',
                                         'optimizer_state_dict', 'iter')):
        raise ValueError('V14 resume requires a complete actor/critic/optimizer checkpoint')
    updates = infos.get('completed_updates')
    if not isinstance(updates, int) or updates < 0:
        raise ValueError('Missing completed V14 update count')
    return infos


class V14VecEnv(RslRlVecEnvWrapper):
    def __init__(self, num_envs=4096, seed=42, device='cuda:0', play=False):
        torch.manual_seed(seed)
        super().__init__(ManagerBasedRlEnv(make_env_cfg(num_envs, seed, play), device=device),
                         clip_actions=None)
        self.obs = self.get_observations()

    def step(self, actions):
        self.obs, value, done, extra = super().step(actions)
        return self.obs, value, done, extra


class V14Runner(DirectMotorOnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._completed_updates = 0
        self._learning = False

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._learning = True
        try:
            result = super().learn(num_learning_iterations, init_at_random_ep_len=False)
            self._completed_updates = self.current_learning_iteration + 1
            return result
        finally:
            self._learning = False

    def save(self, path, infos=None):
        updates = self.current_learning_iteration + 1 if self._learning else self._completed_updates
        super().save(path, {**(infos or {}), 'interface': manifest(),
            'observation_contract': OBSERVATION_CONTRACT, 'experiment_profile_contract': INTERFACE,
            'completed_updates': updates, 'environment': serializable(self.env.unwrapped.cfg)})

    def load(self, path, **kwargs):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        infos = validate_checkpoint(saved)
        if infos.get('environment') != serializable(self.env.unwrapped.cfg):
            raise ValueError('V14 resume environment configuration mismatch')
        result = super().load(path, **kwargs)
        self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]['lr'])
        self._completed_updates = self.current_learning_iteration = infos['completed_updates']
        env = self.env.unwrapped
        command = env.command_manager.get_term('standup')
        command.training_iteration = self._completed_updates - env.common_step_counter // 24
        command._update_command(None)
        return result


def run(args):
    if os.environ.get('WHEELLEG_HUANANHU_INIT_CHECKPOINT'):
        raise ValueError('V14 requires its own checkpoint or a random start')
    if args.num_envs < 1 or (args.command == 'train' and args.iterations < 1) or (
            args.command == 'check' and args.steps < 1):
        raise ValueError('Environment, iteration and step counts must be positive')
    if args.command == 'play' and (not math.isfinite(args.seconds) or args.seconds <= 0):
        raise ValueError('Playback duration must be finite and positive')
    saved = None
    if getattr(args, 'checkpoint', None):
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        validate_checkpoint(saved)
    # Reject other versions before creating a new run directory or GPU environment.
    if getattr(args, 'resume', None):
        validate_checkpoint(torch.load(args.resume, map_location='cpu', weights_only=False))
    env = V14VecEnv(args.num_envs, args.seed, args.device, args.command == 'play')
    try:
        if args.command == 'train':
            from .getup_curriculum import snapshot_source
            if '/' in args.run_name or '\\' in args.run_name:
                raise ValueError('run-name must be a label')
            path = ROOT / 'logs/mjlab/chuanliantui' / (
                datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_scut_v14_flat'
                + ('_' + args.run_name if args.run_name else ''))
            path.mkdir(parents=True, exist_ok=False)
            cfg = runner_cfg(args.seed)
            cfg.update(run_name=args.run_name, max_iterations=args.iterations)
            for name, value in (('interface', manifest()), ('environment', serializable(env.unwrapped.cfg)),
                                ('runner', cfg)):
                (path / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
            snapshot_source(path)
            runner_type = V14Runner
            if args.render:
                from ..training_viewer import ViewerOptions, viewer_runner_class
                runner_type = viewer_runner_class(V14Runner, ViewerOptions(
                    fps=args.viewer_fps, num_envs=args.num_envs))
            runner = runner_type(env, cfg, str(path), device=args.device)
            if args.resume:
                runner.load(str(args.resume))
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
            print('V14_CHECK ' + json.dumps(dict(num_envs=args.num_envs, steps=args.steps,
                  finite=True, numerical=numerical, profile=INTERFACE)), flush=True)
        else:
            from ..policy import actor_model
            actor = actor_model().to(args.device).eval() if saved else None
            if actor is not None:
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
                        env.step(actor(env.obs) if actor else torch.zeros(args.num_envs, 6, device=args.device))
                    if viewer:
                        viewer.draw(SimpleNamespace(current_learning_iteration=-1))
                        viewer.pace(started)
            finally:
                if viewer:
                    viewer.close()
    finally:
        env.close()
