"""Explicit V14 repair: righting -> two-wheel balance -> commanded height."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from datetime import datetime
import time
from types import SimpleNamespace

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from . import huananhu_v14 as base, huananhu_v14_height as source
from . import huananhu_v14_recovery as previous
from .wheel_support_goals import Curriculum, PARAMETERS, STAGES, predicates
from ..config import ROOT
from ..reference_training.control_modes import ControlMode
from ..reference_training.entry import serializable
from ..reference_training.huananhu_mdp import HuananhuCommand, HuananhuCommandCfg
from ..reference_training.huananhu_observations import OBSERVATION_CONTRACT
from ..reference_training.model import ACTUATED_JOINTS
from ..reference_training.pd_action import CONTROL_CONTRACT
from ..reference_training.runner import DirectMotorOnPolicyRunner

INTERFACE = 'chuanliantui-mjlab-v14-wheel-support-v1'
SOURCE_SHA256 = '672647d77620c75ecfb03611fd6fe77267fe3850bba98649f66ac8c403d06bcc'
MIGRATION_SHA256 = '56f31fb28778e6f966b1d0ea92b40b9b6698a95a69b535f5283b4f59e095514e'
WEIGHTS = {**base.WEIGHTS, 'wheel_balance': 2.}
HEIGHT_TERMS = ('track_height_exp_tight', 'track_height_square')


class BalanceMode(ControlMode):
    def advance_balance(self, balanced, failed, step_id):
        if self.last_step == step_id:
            return
        self.last_step = step_id
        self.failure_steps = torch.where(failed, self.failure_steps + 1, 0)
        entered = ~self.recover & (self.failure_steps >= math.ceil(.1 / self.step_dt))
        active = self.recover | entered
        self.stable_steps = torch.where(active & balanced, self.stable_steps + 1, 0)
        completed = active & (self.stable_steps >= math.ceil(PARAMETERS['success_hold_s'] / self.step_dt))
        self.recovered |= completed
        self.recover.copy_(active & ~completed)
        self.state_steps = torch.where(self.recover,
            torch.where(entered, 0, self.state_steps + 1), 0)


@dataclass(kw_only=True)
class RecoveryCommandCfg(HuananhuCommandCfg):
    curriculum_enabled: bool = True

    def build(self, env):
        super().build(env)
        return RecoveryCommand(self, env)


class RecoveryCommand(HuananhuCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.control_mode = BalanceMode(self.num_envs, self.device, env.step_dt)
        self.curriculum = Curriculum()
        self.goal_times = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_achieved = torch.zeros_like(self.goal_times, dtype=torch.bool)
        self.goal_step = -1
        self.goal_values = None
        self.metrics.update({name: torch.zeros(self.num_envs, device=self.device)
            for name in ('recovery_stage', 'stage_ready_fraction',
                         *[f'{s}_{m}' for s in STAGES for m in ('success', 'fraction')])})

    def reset(self, env_ids):
        result = super().reset(env_ids)
        self.goal_times[env_ids] = 0.
        self.goal_achieved[env_ids] = False
        self.goal_step = -1
        return result


def wheel_support_force(env):
    return (-env.scene['wheel_contact'].data.force[..., 2]).clamp_min(0.)


def robot_weight(env):
    return env.sim.model.body_mass[:, env.scene['robot'].indexing.body_ids].sum(-1) * 9.81


def nonwheel_support_force(env):
    return (-env.scene['nonwheel_ground'].data.force[..., 2]).clamp_min(0.).sum(-1)


def goals(env):
    state = env.command_manager.get_term('standup')
    robot = env.scene['robot'].data
    return predicates(gravity=robot.projected_gravity_b, wheel_support=wheel_support_force(env),
        wheel_contact=env.scene['wheel_contact'].data.found > 0,
        nonwheel_support=nonwheel_support_force(env), robot_weight=robot_weight(env),
        base_contact=env.scene['base_contact'].data.force.norm(dim=-1).amax(-1),
        linear_velocity=robot.root_link_lin_vel_b, angular_velocity=robot.root_link_ang_vel_b,
        height=robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2],
        target_height=state.command[:, 2], valid=~state.numerical_failure)


def failure(env):
    state = env.command_manager.get_term('standup')
    robot = env.scene['robot']
    if not hasattr(state, 'motor_v_adr'):
        ids, _ = robot.find_joints(ACTUATED_JOINTS, preserve_order=True)
        state.motor_v_adr = robot.indexing.joint_v_adr[ids]
    velocity = env.sim.data.qvel[:, state.motor_v_adr]
    linear, angular = robot.data.root_link_lin_vel_b, robot.data.root_link_ang_vel_b
    finite = torch.isfinite(env.sim.data.qpos).all(-1) & torch.isfinite(env.sim.data.qvel).all(-1)
    numerical = (~finite | (velocity.abs().amax(-1) > 500.)
                 | (linear.abs().amax(-1) > 100.) | (angular.abs().amax(-1) > 200.))
    state.numerical_failure.copy_(numerical)
    # Physical contact/tilt are recovery states in ALL three stages, not early failure.
    if state.goal_step != env.common_step_counter:
        state.goal_step = env.common_step_counter
        state.goal_values = values = goals(env)
        flags = torch.stack([values[s] for s in STAGES], -1)
        state.goal_times = torch.where(flags, state.goal_times + env.step_dt, 0.)
        state.goal_achieved |= state.goal_times >= PARAMETERS['success_hold_s']
        for i, name in enumerate(STAGES):
            state.metrics[f'{name}_success'].copy_(state.goal_achieved[:, i])
            state.metrics[f'{name}_fraction'] += flags[:, i].float() * env.step_dt / env.cfg.episode_length_s
        state.metrics['recovered'].copy_(state.goal_achieved[:, 1])
        state.metrics['ground_start'].copy_(state.ground_start)
        state.metrics['numerical_failure'].copy_(numerical)
        state.metrics['upright_fraction'] += values['righting'].float() * env.step_dt / env.cfg.episode_length_s
        height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        state.metrics['height_error'] += torch.nan_to_num((height - state.command[:, 2]).abs()) * env.step_dt / env.cfg.episode_length_s
        state.metrics['velocity_error'] += torch.nan_to_num(linear[:, 0].abs()) * env.step_dt / env.cfg.episode_length_s
        state.control_mode.advance_balance(values['wheel_balance'], ~values['supported_upright'], env.common_step_counter)
    # Give the complete recovery horizon, then penalize an unfinished current
    # stage rather than letting endless failed recovery end as a neutral timeout.
    exhausted = env.episode_length_buf >= math.ceil(env.cfg.episode_length_s / env.step_dt)
    unfinished = state.goal_times[:, state.curriculum.stage] < PARAMETERS['success_hold_s']
    return numerical | (exhausted & unfinished)


def update_curriculum(env, env_ids=None):
    state = env.command_manager.get_term('standup')
    # Called after rewards; stage and reward gates switch on the next transition.
    fraction = float(state.goal_values[STAGES[state.curriculum.stage]].float().mean())
    state.metrics['stage_ready_fraction'].fill_(fraction)
    if state.cfg.curriculum_enabled and state.curriculum.advance(fraction, env.common_step_counter, env.step_dt):
        print('RECOVERY_STAGE ' + json.dumps(state.curriculum.history[-1]), flush=True)
    state.metrics['recovery_stage'].fill_(state.curriculum.stage)


class RewardState(source.RewardState):
    def compute(self, env):
        if self.step == env.common_step_counter:
            return self.values
        values = super().compute(env)
        state = env.command_manager.get_term('standup')
        gravity = self.robot.data.projected_gravity_b
        # V14 xy-only orientation gives the same score when upright and inverted.
        upright = ((1. - gravity[:, 2]) * .5).clamp(0., 1.)
        for name in ('flat_orientation_x_exp', 'flat_orientation_y_exp'):
            values[name] *= upright
        for name in ('flat_orientation_x_v', 'flat_orientation_y_v'):
            values[name] += .5 * (gravity[:, 2] + 1.).square()
        # A5N touch on each wheel no longer earns full load-bearing credit.
        support = (wheel_support_force(env) / (.5 * robot_weight(env)[:, None])).clamp(0., 1.).amin(-1)
        support *= (env.scene['wheel_contact'].data.found > 0).all(-1)
        support *= nonwheel_support_force(env) <= PARAMETERS['maximum_nonwheel_weight_fraction'] * robot_weight(env)
        values['wheel_balance'] = support * upright if state.curriculum.stage >= 1 else torch.zeros_like(support)
        if state.curriculum.stage < 2:
            for name in HEIGHT_TERMS:
                values[name] = torch.zeros_like(support)
        else:
            height = self.robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
            error = height - state.command[:, 2]
            gate = goals(env)['supported_upright'].float()
            values['track_height_exp_tight'] = torch.exp(-error.abs() / .12) * gate
            values['track_height_square'] = (10. * error).square() * gate
        for name in source.GATED_TERMS:
            values[name] = torch.zeros_like(support)  # Dedicated getup/height task, zero velocity command.
        if state.curriculum.stage == 0:
            values['undesired_contact'] = torch.zeros_like(support)
        self.values = {name: torch.nan_to_num(value, nan=0., posinf=0., neginf=0.)
                       .masked_fill(state.numerical_failure, 0.) for name, value in values.items()}
        self.values['termination'] = env.reset_terminated.float()
        return self.values


def reward(env, *, name):
    if not hasattr(env, '_huananhu_rewards'):
        env._huananhu_rewards = RewardState(env)
    return env._huananhu_rewards.compute(env)[name]


def make_env_cfg(num_envs=4096, seed=42, play=False):
    cfg = source.make_env_cfg(num_envs, seed, play)
    command = asdict(cfg.commands['standup'])
    command.update(dynamic_modes=True, strict_recovery_metrics=False,
                   zero_velocity_commands=True, special_modes_enabled=False)
    cfg.commands['standup'] = RecoveryCommandCfg(**command, curriculum_enabled=not play)
    cfg.rewards = {name: RewardTermCfg(func=reward, weight=weight, params={'name': name})
                   for name, weight in WEIGHTS.items()}
    cfg.events['recovery_curriculum'] = EventTermCfg(func=update_curriculum, mode='step')
    cfg.terminations['failure'] = TerminationTermCfg(func=failure)
    # ContactMatch without entity uses the exact compiled terrain geom name.
    ground = ContactMatch(mode='geom', pattern='terrain')
    for sensor in cfg.scene.sensors:
        if sensor.name in ('wheel_contact', 'base_contact'):
            sensor.secondary = ground
            sensor.secondary_policy = 'error'
    cfg.scene.sensors += (ContactSensorCfg(name='nonwheel_ground',
        primary=ContactMatch(mode='body', pattern='^(base_link|[lr]f(0|1|00|01|02|03))$', entity='robot'),
        secondary=ground, secondary_policy='error', fields=('found', 'force'), reduce='netforce'),)
    return cfg


def runner_cfg(seed=42):
    cfg = source.runner_cfg(seed)
    cfg['experiment_name'] = 'chuanliantui_v14_wheel_support'
    return cfg


def manifest():
    original = source.manifest()
    if original['module_sha256'] != SOURCE_SHA256:
        raise ValueError('Frozen V14 height source changed')
    return dict(interface=INTERFACE, source=original,
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        goals_sha256=hashlib.sha256(Path(__file__).with_name('wheel_support_goals.py').read_bytes()).hexdigest(),
        previous_recovery=previous.manifest(),
        stages=STAGES, parameters=PARAMETERS, weights=WEIGHTS,
        environment=serializable(make_env_cfg()), ppo=runner_cfg(),
        modes='recover ends at sustained wheel balance, independent of commanded height',
        height_range=source.HEIGHT_RANGE, pull_up_assistance=False,
        termination='numerical immediately; physical recovery allowed until 20s timeout',
        height_formula='exp(-abs(error)/0.12) and (10*error)^2, supported upright, stage 2 only')


def validate_checkpoint(saved, *, migration=False):
    if migration:
        infos = previous.validate_checkpoint(saved)
        if infos['completed_updates'] != 3000:
            raise ValueError('Recovery migration requires the evaluated 3000-update recovery source')
    else:
        infos = saved.get('infos') or {}
        if (json.dumps(infos.get('interface'), sort_keys=True) != json.dumps(manifest(), sort_keys=True)
                or infos.get('control_contract') != CONTROL_CONTRACT
                or infos.get('observation_contract') != OBSERVATION_CONTRACT
                or infos.get('experiment_profile_contract') != INTERFACE):
            raise ValueError('Recovery profile mismatch; explicit known-source migration required')
        check = Curriculum()
        check.load(infos['recovery_curriculum'])
    if any(name not in saved for name in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'iter')):
        raise ValueError('Full actor/critic/optimizer checkpoint required')
    if not isinstance(infos.get('completed_updates'), int) or infos['completed_updates'] < 0:
        raise ValueError('Missing completed update count')
    return infos


def runtime_environment(cfg):
    value = serializable(cfg)
    terrain = value['scene'].get('terrain')
    if terrain:
        terrain['num_envs'] = value['scene']['num_envs']
        terrain['env_spacing'] = value['scene']['env_spacing']
    return value


class RecoveryVecEnv(base.V14VecEnv):
    def __init__(self, num_envs=4096, seed=42, device='cuda:0', play=False):
        torch.manual_seed(seed)
        RslRlVecEnvWrapper.__init__(self,
            ManagerBasedRlEnv(make_env_cfg(num_envs, seed, play), device=device), clip_actions=None)
        self.obs = self.get_observations()


class RecoveryRunner(base.V14Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.migration_history = []

    def save(self, path, infos=None):
        updates = self.current_learning_iteration + 1 if self._learning else self._completed_updates
        command = self.env.unwrapped.command_manager.get_term('standup')
        DirectMotorOnPolicyRunner.save(self, path, {**(infos or {}), 'interface': manifest(),
            'observation_contract': OBSERVATION_CONTRACT, 'experiment_profile_contract': INTERFACE,
            'completed_updates': updates, 'environment': serializable(self.env.unwrapped.cfg),
            'recovery_curriculum': command.curriculum.state_dict(), 'migration_history': self.migration_history})

    def load(self, path, *, migration=False, **kwargs):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        infos = validate_checkpoint(saved, migration=migration)
        env = self.env.unwrapped
        if migration:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != MIGRATION_SHA256:
                raise ValueError('Migration source fingerprint mismatch; only the evaluated recovery model2999 is accepted')
            expected = runtime_environment(previous.make_env_cfg(env.num_envs, env.cfg.seed))
            if infos['environment'] != expected:
                raise ValueError('Source environment does not match the frozen evaluated height profile')
        elif infos['environment'] != serializable(env.cfg):
            raise ValueError('Recovery resume environment mismatch')
        result = DirectMotorOnPolicyRunner.load(self, path, **kwargs)
        self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]['lr'])
        self._completed_updates = self.current_learning_iteration = infos['completed_updates']
        self.migration_history = list(infos.get('migration_history', []))
        command = env.command_manager.get_term('standup')
        if migration:
            self.migration_history.append(dict(source_sha256=MIGRATION_SHA256,
                from_interface=previous.INTERFACE, to_interface=INTERFACE,
                completed_updates=self._completed_updates, curriculum_restart=STAGES[0]))
        else:
            command.curriculum.load(infos['recovery_curriculum'])
        command.training_iteration = self._completed_updates - env.common_step_counter // 24
        command._update_command(None)
        return result
def run(args):
    migration = getattr(args, 'resume_v14_wheel_support', False)
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
    env = RecoveryVecEnv(args.num_envs, args.seed, args.device, args.command == 'play')
    try:
        if args.command == 'train':
            from .getup_curriculum import snapshot_source
            if '/' in args.run_name or '\\' in args.run_name:
                raise ValueError('run-name must be a label')
            path = ROOT / 'logs/mjlab/chuanliantui' / (
                datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_v14_wheel_support'
                + ('_' + args.run_name if args.run_name else ''))
            path.mkdir(parents=True, exist_ok=False)
            cfg = runner_cfg(args.seed)
            cfg.update(run_name=args.run_name, max_iterations=args.iterations)
            for name, value in (('interface', manifest()), ('environment', serializable(env.unwrapped.cfg)),
                                ('runner', cfg)):
                (path / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
            snapshot_source(path)
            runner_type = RecoveryRunner
            if args.render:
                from ..training_viewer import ViewerOptions, viewer_runner_class
                runner_type = viewer_runner_class(RecoveryRunner, ViewerOptions(
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
            print('V14_WHEEL_SUPPORT_CHECK ' + json.dumps(dict(finite=True, numerical=numerical,
                num_envs=args.num_envs, steps=args.steps, learning=False)), flush=True)
        else:
            env.unwrapped.command_manager.get_term('standup').curriculum.load(saved['infos']['recovery_curriculum'])
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

