"""Getup_gym reward groups 2/3/4 on the physical Chuanliantui robot.

One batch-wide, monotonic curriculum; no policy updates in check/play.
Historical reference_training kernels are deliberately imported unchanged.
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper

from ..asset import POSES, cpu_model
from ..config import ROOT
from ..reference_training import huananhu_mdp
from ..reference_training.entry import manifest as base_manifest, runner_cfg, serializable
from ..reference_training.huananhu_env_cfg import make_huananhu_env_cfg
from ..reference_training.huananhu_mdp import HuananhuCommand, HuananhuCommandCfg
from ..reference_training.huananhu_observations import OBSERVATION_CONTRACT
from ..reference_training.model import ACTUATED_JOINTS
from ..reference_training.pd_action import CONTROL_CONTRACT
from ..reference_training.runner import DirectMotorOnPolicyRunner
from .pull_up_assistance import AssistedPdActionCfg, PARAMETERS as ASSISTANCE_PARAMETERS

INTERFACE = 'chuanliantui-mjlab-getup-curriculum-r15-assisted-v3'
LEGACY_POSE_INTERFACE = 'chuanliantui-mjlab-getup-curriculum-r15-assisted-v2'
LEGACY_POSE_MODULE_SHA256 = '41d395799656b1473e10a72d6755b54f793913e1b2bece6278547a3d8d230746'
LEGACY_CONTACT_INTERFACE = 'chuanliantui-mjlab-getup-curriculum-r15-assisted-v1'
LEGACY_CONTACT_MODULE_SHA256 = 'edc3b8ef35a7c6624913fdd1727777f2b50540d040c167b02fc762667023efac'
SOURCE_COMMIT = '238d2962a0fee8e3da4da914cadcea2be00ce402'
SOURCE_URL = f'https://github.com/2350575870/getup_gym/tree/{SOURCE_COMMIT}'
STAGES = ('righting', 'standing', 'walking')
HEIGHTS = (.27, .30, .30)
PARAMETERS = dict(ready_fraction=2 / 3, hold_s=2., minimum_updates=100,
                  height_tolerance_m=.03, tilt_deg=15., wheel_contact_n=5.,
                  base_contact_n=1., height_sigma_m=.12, velocity_sigma=.12)
# Source groups 2/3/4 without the fixed-default-angle penalty in v3.
WEIGHTS = (
    dict(termination=-200., orientation=-2.6, inverted=-.6, height=4.,
         acceleration=-2.5e-8, action_rate=-.06,
         symmetry=-.02, no_fly=-2., wheel_load=4.),
    dict(termination=-100., orientation=-2., inverted=-1., torque=-1e-6,
         acceleration=-2.5e-6, action_rate=-.02, symmetry=-.02,
         height=6.),
    dict(termination=-500., linear_tracking=7., yaw_tracking=4., orientation=-15.,
         torque=-1e-5, acceleration=-2.5e-6, leg_velocity=-.004,
         action_rate=-.1, no_fly=-.2, height=5.,
         wheel_distance=-2., velocity_band=1., motor_limits=-5.,
         torque_limits=-.05, action_smoothness=-.02, power=-1e-4),
)


class Curriculum:
    def __init__(self):
        self.stage = 0
        self.ready_steps = 0
        self.stage_steps = 0
        self.history = []
        self.last_step = -1

    def advance(self, ready_fraction, step, step_dt):
        if step == self.last_step:
            return False
        self.last_step = step
        self.stage_steps += 1
        self.ready_steps = self.ready_steps + 1 if ready_fraction >= PARAMETERS['ready_fraction'] else 0
        if (self.stage < 2 and self.stage_steps >= 24 * PARAMETERS['minimum_updates']
                and self.ready_steps >= math.ceil(PARAMETERS['hold_s'] / step_dt)):
            self.history.append(dict(from_stage=STAGES[self.stage], to_stage=STAGES[self.stage + 1],
                                     control_step=step, ready_fraction=ready_fraction))
            self.stage += 1
            self.ready_steps = self.stage_steps = 0
            return True
        return False

    def state_dict(self):
        return dict(stage=self.stage, ready_steps=self.ready_steps,
                    stage_steps=self.stage_steps, history=self.history)

    def load(self, saved):
        if (saved['stage'] not in (0, 1, 2) or saved['ready_steps'] < 0
                or saved['stage_steps'] < 0):
            raise ValueError('Invalid getup curriculum state')
        self.stage, self.ready_steps, self.stage_steps = (
            saved['stage'], saved['ready_steps'], saved['stage_steps'])
        self.history = list(saved['history'])
        self.last_step = -1


@dataclass(kw_only=True)
class GetupCommandCfg(HuananhuCommandCfg):
    curriculum_enabled: bool = True
    initial_stage: int = 0

    def build(self, env):
        super().build(env)  # Retain the base configuration validation.
        if self.initial_stage not in (0, 1, 2):
            raise ValueError('initial_stage must be 0, 1 or 2')
        return GetupCommand(self, env)


class GetupCommand(HuananhuCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.curriculum = Curriculum()
        self.curriculum.stage = cfg.initial_stage
        self.metrics.update({name: torch.zeros(self.num_envs, device=self.device)
                             for name in ('getup_stage', 'getup_ready_fraction',
                                          'pull_up_force_n', 'pull_up_budget_ratio')})

    def _update_command(self, env_ids):
        super()._update_command(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._command[ids, 2] = HEIGHTS[self.curriculum.stage]
        if self.curriculum.stage < 2:
            self._command[ids, :2] = 0.


def wheel_support_force(env):
    # mjlab netforce points primary (wheel) -> secondary (ground).
    # Ground support therefore has negative world-z in this sensor output.
    # Do not use abs(): a downward reaction must not count as upward support.
    return (-env.scene['wheel_contact'].data.force[..., 2]).clamp_min(0.)


def readiness(env):
    state = env.command_manager.get_term('standup')
    robot = env.scene['robot'].data
    height = robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    wheel = wheel_support_force(env)
    base = env.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
    action = env.action_manager.get_term('pd')
    independent = action.assistance_force <= action.robot_weight() * ASSISTANCE_PARAMETERS['promotion_max_weight_ratio']
    return (torch.isfinite(height) & ~state.numerical_failure & independent
            & ((height - state.command[:, 2]).abs() <= PARAMETERS['height_tolerance_m'])
            & (-robot.projected_gravity_b[:, 2] >= math.cos(math.radians(PARAMETERS['tilt_deg'])))
            & (wheel > PARAMETERS['wheel_contact_n']).all(-1)
            & (base <= PARAMETERS['base_contact_n']))


def failure(env):
    failed = huananhu_mdp.failure(env)
    state = env.command_manager.get_term('standup')
    # Source default permits whole-episode recovery, including torso contact.
    # Walking retains this robot's 20-step physical failure criterion.
    return state.numerical_failure if state.curriculum.stage < 2 else failed


def update_curriculum(env, env_ids=None):
    state = env.command_manager.get_term('standup')
    fraction = float(readiness(env).float().mean())
    state.metrics['getup_ready_fraction'].fill_(fraction)
    if state.cfg.curriculum_enabled and state.curriculum.advance(
            fraction, state.training_iteration * 24 + env.common_step_counter, env.step_dt):
        print('GETUP_STAGE ' + json.dumps(state.curriculum.history[-1]), flush=True)
    state.metrics['getup_stage'].fill_(state.curriculum.stage)
    # This event runs after reward computation. The new stage/target become
    # active together on the next observation and transition.
    state._update_command(None)


class RewardState:
    def __init__(self, env):
        self.robot = env.scene['robot']
        ids, names = self.robot.find_joints(ACTUATED_JOINTS, preserve_order=True)
        if tuple(names) != ACTUATED_JOINTS:
            raise ValueError('Unexpected physical motor order')
        self.ids = ids
        self.qadr = self.robot.indexing.joint_q_adr[ids]
        self.vadr = self.robot.indexing.joint_v_adr[ids]
        self.wheel_ids, _ = self.robot.find_bodies(('lfwheel', 'rfwheel'), preserve_order=True)
        self.previous_velocity = env.sim.data.qvel[:, self.vadr].clone()
        self.step = -1
        self.values = {}
        # Retarget wheel spacing from the solved nominal CAD pose, not random resets.
        model = cpu_model()
        data = mujoco.MjData(model)
        for name, value in POSES['nominal'].items():
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        left, right = (data.xpos[model.body(name).id] for name in ('lfwheel', 'rfwheel'))
        self.wheel_spacing = float(((left[:2] - right[:2]) ** 2).sum() ** .5)

    def compute(self, env):
        if self.step == env.common_step_counter:
            return self.values
        state = env.command_manager.get_term('standup')
        robot, action = self.robot.data, env.action_manager
        q, velocity = env.sim.data.qpos[:, self.qadr], env.sim.data.qvel[:, self.vadr]
        acceleration = (velocity - self.previous_velocity) / env.step_dt
        self.previous_velocity.copy_(velocity)
        legs = (0, 1, 3, 4)
        delta = q[:, legs] - robot.default_joint_pos[:, self.ids][:, legs]
        # The CAD's right motor coordinate signs mirror the left side.
        delta = delta * delta.new_tensor((1., 1., -1., -1.))
        wheel_z = wheel_support_force(env)
        dual = (wheel_z > 1.).all(-1)
        gravity = robot.projected_gravity_b
        xy = gravity[:, :2].square().sum(-1)
        height = robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        torque = action.get_term('pd').torques
        mass = env.sim.model.body_mass[:, self.robot.indexing.body_ids].sum(-1)
        wheel_position = robot.body_link_pos_w[:, self.wheel_ids, :2]
        spacing = (wheel_position[:, 0] - wheel_position[:, 1]).norm(dim=-1)
        lower_upper = robot.default_joint_pos_limits[:, self.ids][:, legs]
        middle = lower_upper.mean(-1)
        half = (lower_upper[..., 1] - lower_upper[..., 0]) * .45
        motor_limits = ((q[:, legs] - middle).abs() - half).clamp_min(0.).sum(-1)
        linear = robot.root_link_lin_vel_b
        command = state.command
        vx, target = linear[:, 0], command[:, 0]
        positive_band = (vx >= .8 * target) & (vx <= 1.1 * target) & (target >= 0.)
        negative_band = (vx <= .8 * target) & (vx >= 1.1 * target) & (target < 0.)
        self.values = dict(
            termination=env.reset_terminated.float(),
            height=torch.exp(-(height - command[:, 2]).abs() / .12),
            orientation=.02 * xy + 2. * (gravity[:, 2] + 1.).square()
                        - torch.exp(-xy / .25).clamp(0., .1),
            inverted=(-gravity[:, 2] < -.2).float(),
            symmetry=(delta[:, :2] - delta[:, 2:]).square().sum(-1),
            no_fly=(~dual).float(),
            wheel_load=torch.exp(-(wheel_z.abs().sum(-1) - mass * 10.).abs() * .05),
            linear_tracking=torch.exp(-((command[:, 0] - linear[:, 0]).square()
                                        + linear[:, 1].square()) / .12)
                            * (height > HEIGHTS[2] * .8),
            yaw_tracking=torch.exp(-(command[:, 1] - robot.root_link_ang_vel_b[:, 2]).square() / .12),
            torque=torque.square().sum(-1), acceleration=acceleration.square().sum(-1),
            leg_velocity=velocity[:, legs].square().sum(-1),
            action_rate=(action.action - action.prev_action).square().sum(-1),
            action_smoothness=(action.action - 2 * action.prev_action + action.prev_prev_action).square().sum(-1),
            wheel_distance=(self.wheel_spacing - .03 - spacing).clamp(0., 1.)
                           + (spacing - self.wheel_spacing - .03).clamp(0., 1.),
            velocity_band=torch.where(positive_band | negative_band, 1., -1.),
            motor_limits=motor_limits,
            torque_limits=(torque.abs() - torque.new_tensor((40., 40., 3.9, 40., 40., 3.9)) * .8).clamp_min(0.).sum(-1),
            power=(velocity.abs() * torque.abs()).sum(-1) * (height < HEIGHTS[2] * 14 / 15),
        )
        self.values = {name: torch.nan_to_num(value, nan=0., posinf=0., neginf=0.)
                       .masked_fill(state.numerical_failure, 0.)
                       for name, value in self.values.items()}
        self.step = env.common_step_counter
        return self.values


def reward(env, *, name):
    if not hasattr(env, '_getup_rewards'):
        env._getup_rewards = RewardState(env)
    stage = env.command_manager.get_term('standup').curriculum.stage
    return env._getup_rewards.compute(env)[name] * WEIGHTS[stage].get(name, 0.)


def reset_rewards(env, env_ids):
    if hasattr(env, '_getup_rewards'):
        rewards = env._getup_rewards
        rewards.previous_velocity[env_ids] = env.sim.data.qvel[env_ids][:, rewards.vadr]
        rewards.step = -1


def make_env_cfg(num_envs=4096, seed=42, play=False, stage=0):
    cfg = make_huananhu_env_cfg(play=play)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    action = asdict(cfg.actions['pd'])
    action.pop('class_type')
    cfg.actions['pd'] = AssistedPdActionCfg(**action, assistance_enabled=not play)
    cfg.commands['standup'] = GetupCommandCfg(
        **asdict(cfg.commands['standup']), curriculum_enabled=not play, initial_stage=stage)
    cfg.commands['standup'].height_range = (HEIGHTS[stage], HEIGHTS[stage])
    cfg.rewards = {name: RewardTermCfg(func=reward, weight=1., params={'name': name})
                   for name in sorted(set().union(*WEIGHTS))}
    cfg.events['getup_curriculum'] = EventTermCfg(func=update_curriculum, mode='step')
    cfg.events['getup_reset_rewards'] = EventTermCfg(func=reset_rewards, mode='reset')
    cfg.terminations['failure'] = TerminationTermCfg(func=failure)
    return cfg


def manifest():
    return dict(interface=INTERFACE, base=base_manifest(), source_url=SOURCE_URL,
                weights=WEIGHTS, stages=STAGES, heights=HEIGHTS, parameters=PARAMETERS,
                module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                recovery_termination='numerical and 20s timeout; walking adds 20-step contact/tilt',
                pull_up_assistance=ASSISTANCE_PARAMETERS.copy(),
                assistance_sha256=hashlib.sha256(Path(__file__).with_name('pull_up_assistance.py').read_bytes()).hexdigest(),
                wheel_support_convention='clamp(-primary_to_secondary_world_Fz, min=0)',
                pose_bias_contract=dict(enabled=False, removed_stages=STAGES))


class GetupVecEnv(RslRlVecEnvWrapper):
    def __init__(self, num_envs=4096, seed=42, device='cuda:0', play=False, stage=0):
        torch.manual_seed(seed)
        super().__init__(ManagerBasedRlEnv(make_env_cfg(num_envs, seed, play, stage), device=device),
                         clip_actions=None)
        self.obs = self.get_observations()

    def step(self, actions):
        self.obs, reward_value, done, extra = super().step(actions)
        return self.obs, reward_value, done, extra


def validate_checkpoint(saved, *, allow_contact_fix=False, allow_pose_bias_fix=False):
    infos = saved.get('infos') or {}
    # Migration comparisons must not mutate the live reward dictionaries.
    expected = json.loads(json.dumps(manifest()))
    if allow_contact_fix:
        raise ValueError('v1 contact migration uses the archived v2 release; v3 only migrates known v2 with --resume-pose-bias-fix')
    if allow_pose_bias_fix:
        expected['interface'] = LEGACY_POSE_INTERFACE
        expected['module_sha256'] = LEGACY_POSE_MODULE_SHA256
        expected['weights'][0]['pose_bias'] = -.06
        expected['weights'][1]['pose_bias'] = -.14
        expected['weights'][2]['pose_bias'] = -.12
        expected.pop('pose_bias_contract')
    if (json.dumps(infos.get('interface'), sort_keys=True) != json.dumps(expected, sort_keys=True)
            or infos.get('control_contract') != CONTROL_CONTRACT
            or infos.get('observation_contract') != OBSERVATION_CONTRACT
            or infos.get('experiment_profile_contract') != expected['interface']):
        raise ValueError('Getup source/configuration/control contract mismatch; implicit r14 resume is forbidden')
    return infos


class GetupRunner(DirectMotorOnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._completed_updates = 0
        self._learning = False
        self.migration_history = []

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._learning = True
        try:
            result = super().learn(num_learning_iterations, init_at_random_ep_len=False)
            self._completed_updates = self.current_learning_iteration + 1
            return result
        finally:
            self._learning = False

    def save(self, path, infos=None):
        command = self.env.unwrapped.command_manager.get_term('standup')
        super().save(path, {**(infos or {}), 'interface': manifest(),
                           'observation_contract': OBSERVATION_CONTRACT,
                           'experiment_profile_contract': INTERFACE,
                           'getup_curriculum': command.curriculum.state_dict(),
                           'completed_updates': self.current_learning_iteration + 1 if self._learning else self._completed_updates,
                           'migration_history': self.migration_history,
                           'environment': serializable(self.env.unwrapped.cfg)})

    def load(self, path, *, allow_contact_fix=False, allow_pose_bias_fix=False, **kwargs):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        infos = validate_checkpoint(saved, allow_contact_fix=allow_contact_fix,
                                    allow_pose_bias_fix=allow_pose_bias_fix)
        previous_environment = json.loads(json.dumps(infos['environment']))
        current_environment = serializable(self.env.unwrapped.cfg)
        if allow_pose_bias_fix:
            # The only allowed environment delta is removal of this reward.
            expected_term = dict(current_environment['rewards']['height'])
            expected_term['params'] = {'name': 'pose_bias'}
            removed = previous_environment['rewards'].pop('pose_bias', None)
            if removed != expected_term:
                raise ValueError('Unexpected legacy pose_bias reward configuration')
        if previous_environment != current_environment:
            raise ValueError('Getup resume environment configuration mismatch')
        result = super().load(path, **kwargs)
        self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]['lr'])
        self.migration_history = list(infos.get('migration_history', []))
        if allow_pose_bias_fix:
            self.migration_history.append(dict(
                change='remove fixed-default-angle pose_bias reward from all three stages',
                source_checkpoint=str(Path(path).resolve()),
                source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                from_interface=LEGACY_POSE_INTERFACE, to_interface=INTERFACE,
                completed_updates=infos['completed_updates']))
        self.current_learning_iteration = infos['completed_updates']
        self._completed_updates = infos['completed_updates']
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


def snapshot_source(path):
    """Keep the new run's complete simulation/PPO sources without touching old runs."""
    hashes = {}
    for name in ('mjlab_training', 'resources/robots/chuanliantui_new_1', 'scripts/agent'):
        for source in sorted((ROOT / name).rglob('*')):
            relative = source.relative_to(ROOT)
            if (not source.is_file() or source.is_symlink()
                    or set(relative.parts) & {'.venv', 'venv', '__pycache__', 'logs', '.git'}
                    or source.suffix in ('.pyc', '.pyo')):
                continue
            data = source.read_bytes()
            target = path / 'source-snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    (path / 'source-state.json').write_text(json.dumps(
        dict(source_kind='complete-filesystem-snapshot', files_sha256=hashes), indent=2) + '\n')


def run(args):
    contact_fix = getattr(args, 'resume_contact_force_fix', False)
    pose_bias_fix = getattr(args, 'resume_pose_bias_fix', False)
    if contact_fix:
        raise ValueError('Use archived v2 for v1 contact migration; v3 accepts --resume-pose-bias-fix from known v2')
    if pose_bias_fix and (args.command != 'train' or not getattr(args, 'resume', None)):
        raise ValueError('Pose bias migration requires train --resume <assisted-v2 checkpoint>')
    if contact_fix and (args.command != 'train' or not getattr(args, 'resume', None)):
        raise ValueError('Contact force migration requires train --resume <assisted-v1 checkpoint>')
    if args.num_envs < 1 or (args.command == 'train' and args.iterations < 1) or (
            args.command == 'check' and args.steps < 1):
        raise ValueError('Environment, iteration and step counts must be positive')
    if args.command == 'play' and (not math.isfinite(args.seconds) or args.seconds <= 0.):
        raise ValueError('Playback duration must be finite and positive')
    if os.environ.get('WHEELLEG_HUANANHU_INIT_CHECKPOINT'):
        raise ValueError('Getup curriculum requires its own checkpoint or a random start')
    checkpoint = getattr(args, 'checkpoint', None)
    stage = 0
    saved = None
    if checkpoint:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        stage = validate_checkpoint(saved)['getup_curriculum']['stage']
    env = GetupVecEnv(args.num_envs, args.seed, args.device, args.command == 'play', stage)
    try:
        if args.command == 'train':
            label = getattr(args, 'run_name', '')
            if '/' in label or '\\' in label:
                raise ValueError('run-name must be a label')
            path = ROOT / 'logs/mjlab/chuanliantui' / (
                datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_getup_r15' + ('_' + label if label else ''))
            path.mkdir(parents=True, exist_ok=False)
            cfg = runner_cfg(args.seed)
            cfg.update(run_name='getup_r15', experiment_name='chuanliantui_getup_r15',
                       max_iterations=args.iterations)
            (path / 'interface.json').write_text(json.dumps(manifest(), indent=2) + '\n')
            (path / 'environment.json').write_text(json.dumps(serializable(env.unwrapped.cfg), indent=2) + '\n')
            (path / 'runner.json').write_text(json.dumps(cfg, indent=2) + '\n')
            snapshot_source(path)
            runner_type = GetupRunner
            if getattr(args, 'render', False):
                from ..training_viewer import ViewerOptions, viewer_runner_class
                runner_type = viewer_runner_class(GetupRunner, ViewerOptions(
                    fps=getattr(args, 'viewer_fps', 30.), num_envs=args.num_envs))
            runner = runner_type(env, cfg, str(path), device=args.device)
            if getattr(args, 'resume', None):
                runner.load(str(args.resume), allow_pose_bias_fix=pose_bias_fix)
                runner.save(str(path / 'model_resume_initial.pt'))
            else:
                runner.save(str(path / 'model_initial.pt'))
            print(f'Training output: {path}', flush=True)
            runner.learn(args.iterations, init_at_random_ep_len=False)
        elif args.command == 'check':
            numerical = 0
            for _ in range(args.steps):
                obs, value, done, extra = env.step(torch.zeros(args.num_envs, 6, device=args.device))
                assert obs['actor'].shape == (args.num_envs, 35) and obs['critic'].shape == (args.num_envs, 78)
                assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(value).all()
                numerical += int(env.unwrapped.command_manager.get_term('standup').numerical_failure.sum())
            print('GETUP_CHECK ' + json.dumps(dict(num_envs=args.num_envs, steps=args.steps,
                                                  finite=True, numerical=numerical, profile=INTERFACE)), flush=True)
        else:
            if saved is None:
                raise ValueError('Getup playback requires --checkpoint')
            from ..policy import actor_model
            actor = actor_model().to(args.device).eval()
            actor.load_state_dict(saved['actor_state_dict'], strict=True)
            if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
                raise ValueError('Getup checkpoint contains nonfinite actor parameters')
            viewer = None
            if getattr(args, 'render', False):
                from ..training_viewer import NativeTrainingViewer, ViewerOptions
                viewer = NativeTrainingViewer(env, ViewerOptions(num_envs=args.num_envs))
                viewer.open()
            from types import SimpleNamespace
            import time
            preview = SimpleNamespace(current_learning_iteration=-1)
            try:
                for _ in range(round(args.seconds / env.unwrapped.step_dt)):
                    started = time.monotonic()
                    if viewer:
                        viewer.wait_if_paused(preview)
                        if not viewer.is_running():
                            break
                    with torch.inference_mode():
                        env.step(actor(env.obs))
                    if viewer:
                        viewer.draw(preview)
                        viewer.pace(started)
            finally:
                if viewer:
                    viewer.close()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('train', 'check', 'play'))
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-name', default='')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--resume-contact-force-fix', action='store_true')
    parser.add_argument('--resume-pose-bias-fix', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--viewer-fps', type=float, default=30.)
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.steps, args.seconds) <= 0:
        parser.error('Counts and duration must be positive')
    run(args)


if __name__ == '__main__':
    # Keep serialized function names canonical even when launched with -m.
    from importlib import import_module
    import_module(__spec__.name).main()
