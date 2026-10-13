"""Authorized A2500 height continuations; historical profiles stay unchanged."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper

from . import recovery_factors as source, huananhu_v14_height as height_base
from ..reference_training import huananhu_mdp, huananhu_rewards
from ..reference_training.entry import serializable
from ..reference_training.huananhu_observations import OBSERVATION_CONTRACT
from ..reference_training.pd_action import CONTROL_CONTRACT
from ..reference_training.runner import DirectMotorOnPolicyRunner
from ..wheel_balance_assessment import components

INTERFACE = 'chuanliantui-mjlab-height-curriculum-v1'
SOURCE_SHA256 = '706c8387cba4815b491be3f5914a5951881c2966edabbd5fbb90d0da9c9d3dbd'
CASES = ('curriculum', 'curriculum_knee')
RULE = dict(initial_level_m=.21, increment_m=.01, maximum_level_m=.34,
            minimum_updates=50, ready_fraction=2 / 3, hold_s=2.,
            height_tolerance_m=.03, reward_length_m=.12, knee_weight=-.5)


class HeightCurriculum:
    """Global level and dwell; actual completed optimizer updates, not rollout estimates."""
    def __init__(self):
        self.level_index = 0
        self.completed_updates = self.level_start_updates = 2500
        self.hold_s = self.ready_fraction = 0.
        self.last_step = -1
        self.history = []
        self.learning_active = False

    @property
    def level(self):
        return round(.21 + .01 * self.level_index, 2)

    @property
    def height_range(self):
        return (.27, .34) if self.level_index == 13 else (max(.20, self.level - .02), self.level)

    def advance(self, fraction, dt, step):
        if step == self.last_step:
            return None
        self.last_step, self.ready_fraction = step, float(fraction)
        eligible = self.completed_updates - self.level_start_updates >= 50
        if not self.learning_active:
            return None
        self.hold_s = self.hold_s + dt if fraction >= 2 / 3 else 0.
        if not eligible or self.level_index == 13 or self.hold_s + 1e-9 < 2.:
            return None
        old = self.level
        self.level_index += 1
        self.level_start_updates = self.completed_updates
        record = dict(previous_level_m=old, level_m=self.level, control_step=step,
                      completed_updates=self.completed_updates, ready_fraction=float(fraction),
                      hold_s=self.hold_s, next_resample_range_m=self.height_range)
        self.history.append(record)
        self.hold_s = 0.
        print('HEIGHT_LEVEL ' + json.dumps(record), flush=True)
        return record

    def state_dict(self):
        return dict(level_index=self.level_index, level_m=self.level,
                    completed_updates=self.completed_updates, level_start_updates=self.level_start_updates,
                    hold_s=self.hold_s, ready_fraction=self.ready_fraction, last_step=self.last_step,
                    history=list(self.history))

    def load(self, saved):
        if (not isinstance(saved.get('level_index'), int) or not 0 <= saved['level_index'] <= 13
                or saved.get('level_m') != round(.21 + .01 * saved['level_index'], 2)
                or not 2500 <= saved['level_start_updates'] <= saved['completed_updates'] <= 3000
                or not math.isfinite(saved['hold_s']) or saved['hold_s'] < 0
                or not 0 <= saved['ready_fraction'] <= 1):
            raise ValueError('Invalid persisted height curriculum')
        for key in ('level_index', 'completed_updates', 'level_start_updates', 'hold_s',
                    'ready_fraction', 'last_step', 'history'):
            setattr(self, key, saved[key])


@dataclass(kw_only=True)
class HeightCommandCfg(huananhu_mdp.HuananhuCommandCfg):
    curriculum_enabled: bool = True

    def build(self, env):
        # Retain the original validation, then build the independent subclass.
        huananhu_mdp.HuananhuCommandCfg.build(self, env)
        return HeightCommand(self, env)


class HeightCommand(huananhu_mdp.HuananhuCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.height_curriculum = HeightCurriculum()
        self.metrics.update({name: torch.zeros(self.num_envs, device=self.device) for name in
                             ('height_level_m', 'height_ready_fraction')})

    def _resample_command(self, env_ids):
        if not len(env_ids):
            return
        if not self.cfg.zero_velocity_commands:
            raise ValueError('Height curriculum requires zero velocity commands')
        bounds = self.height_curriculum.height_range if self.cfg.curriculum_enabled else self.cfg.height_range
        self.sampled[env_ids, :2] = 0.
        self.sampled[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(*bounds)
        self.heading[env_ids] = False
        self.standing[env_ids] = True
        self.special_mode[env_ids] = -1

    def observe_transition(self):
        env, course = self._env, self.height_curriculum
        if course.last_step == env.common_step_counter:
            return
        robot = env.scene['robot'].data
        height = getattr(self, 'height_above_ground', robot.root_link_pos_w[:, 2])
        values = components(gravity=robot.projected_gravity_b, wheel_support=source.wheel_force(env),
            wheel_contact=env.scene['wheel_ground_probe'].data.found > 0,
            nonwheel_support=source.other_force(env), robot_weight=source.weight(env),
            base_contact=env.scene['base_contact'].data.force.norm(dim=-1).amax(-1),
            linear_velocity=robot.root_link_lin_vel_b, angular_velocity=robot.root_link_ang_vel_b,
            height=height, target_height=self.command[:, 2], valid=~self.numerical_failure,
            parameters=source.PARAMETERS)
        valid = ~self.numerical_failure
        self.last_transition_numerical_failure_count = int(self.numerical_failure.sum())
        ready = values['wheel_load_bearing'] & values['near_level'] & values['target_height'] & valid
        denominator = int(valid.sum())
        fraction = float(ready.sum()) / denominator if denominator else 0.
        if self.cfg.curriculum_enabled:
            course.advance(fraction, env.step_dt, env.common_step_counter)
        else:
            course.ready_fraction, course.last_step = fraction, env.common_step_counter
        self.metrics['height_level_m'].fill_(course.level)
        self.metrics['height_ready_fraction'].fill_(fraction)
        env.extras['log']['HeightCurriculum/level_m'] = course.level
        env.extras['log']['HeightCurriculum/ready_fraction'] = fraction
        env.extras['log']['HeightCurriculum/completed_updates'] = course.completed_updates


class RewardState(height_base.RewardState):
    def __init__(self, env):
        super().__init__(env)
        # The original V14 selects 26 terms and drops the zero-weight knee
        # value. Obtain it from the unchanged reference kernel when requested.
        self.knee_kernel = huananhu_rewards.RewardState(env)

    def compute(self, env):
        if self.step == env.common_step_counter:
            return self.values
        values = dict(super().compute(env))
        state = env.command_manager.get_term('standup')
        height = getattr(state, 'height_above_ground', self.robot.data.root_link_pos_w[:, 2])
        value = torch.exp(-(height - state.command[:, 2]).abs() / .12)
        values['track_height_exp_tight'] = torch.nan_to_num(value, nan=0., posinf=0., neginf=0.).masked_fill(
            state.numerical_failure, 0.)
        values['knee_limit'] = self.knee_kernel.compute(env)['knee_limit']
        self.values = values
        if isinstance(state, HeightCommand):
            state.observe_transition()
        return values


def reward(env, *, name):
    if not hasattr(env, '_huananhu_rewards'):
        env._huananhu_rewards = RewardState(env)
    return env._huananhu_rewards.compute(env)[name]


def make_env_cfg(case, num_envs=4096, seed=42, play=False):
    if case not in CASES:
        raise ValueError('Unknown height curriculum case')
    cfg = source.make_env_cfg('baseline', num_envs, seed, play)
    settings = asdict(cfg.commands['standup'])
    settings.update(zero_velocity_commands=True, height_range=(.20, .21))
    cfg.commands['standup'] = HeightCommandCfg(**settings, curriculum_enabled=not play)
    cfg.rewards = {name: RewardTermCfg(func=reward,
        weight=(-.5 if name == 'knee_limit' and case == 'curriculum_knee' else term.weight),
        params={'name': name}) for name, term in cfg.rewards.items()}
    if case == 'curriculum_knee':
        cfg.rewards['knee_limit'] = RewardTermCfg(func=reward, weight=-.5, params={'name':'knee_limit'})
    return cfg


def interface(case):
    return f'{INTERFACE}:{case}'


def manifest(case):
    if case not in CASES:
        raise ValueError('Unknown case')
    return dict(interface=interface(case), case=case, source=source.manifest('baseline'),
        source_checkpoint_sha256=SOURCE_SHA256,
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rules=RULE,
        environment=serializable(make_env_cfg(case)), ppo=height_base.runner_cfg(42),
        advancement_predicate='wheel_load_bearing & near_level & target_height; nonnumerical denominator',
        level_changes='subsequent normal command resamples; no forced resample or reset')


def validate_source(path, num_envs=4096, seed=42):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Only the specified A2500 SHA256 is accepted')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    infos = source.validate_checkpoint(saved, 'baseline')
    if infos['completed_updates'] != 2500 or infos['environment'] != source.recovery.runtime_environment(source.make_env_cfg('baseline', num_envs, seed)):
        raise ValueError('A2500 population/seed/environment/count mismatch')
    return saved


def validate_checkpoint(saved, case):
    infos = saved.get('infos') or {}
    if (infos.get('interface') != manifest(case) or infos.get('experiment_profile_contract') != interface(case)
            or infos.get('observation_contract') != OBSERVATION_CONTRACT
            or infos.get('control_contract') != CONTROL_CONTRACT
            or not isinstance(infos.get('completed_updates'), int)
            or not 2500 <= infos['completed_updates'] <= 3000
            or any(k not in saved for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict','iter'))):
        raise ValueError('Height curriculum full checkpoint/interface mismatch')
    course = HeightCurriculum()
    course.load(infos['height_curriculum'])
    if course.completed_updates != infos['completed_updates']:
        raise ValueError('Curriculum and optimizer update counters differ')
    return infos


class HeightRunner(source.base.V14Runner):
    def __init__(self, *args, case, **kwargs):
        self.case, self.migration_history = case, []
        super().__init__(*args, **kwargs)
        self.logger.git_status_repos.clear()  # User prohibits even read-only Git commands.
        original_update = self.alg.update
        def update():
            result = original_update()
            self.env.unwrapped.command_manager.get_term('standup').height_curriculum.completed_updates += 1
            return result
        self.alg.update = update

    def save(self, path, infos=None):
        updates = self.current_learning_iteration + 1 if self._learning else self._completed_updates
        state = self.env.unwrapped.command_manager.get_term('standup')
        if state.height_curriculum.completed_updates != updates:
            raise ValueError('Save update count disagrees with actual optimizer updates')
        DirectMotorOnPolicyRunner.save(self, path, {**(infos or {}), 'interface': manifest(self.case),
            'experiment_profile_contract': interface(self.case), 'observation_contract': OBSERVATION_CONTRACT,
            'environment': serializable(self.env.unwrapped.cfg), 'completed_updates': updates,
            'height_curriculum': state.height_curriculum.state_dict(), 'migration_history': self.migration_history})

    def load(self, path, *, migration=False, **kwargs):
        raw = self.env.unwrapped
        saved = validate_source(path) if migration else torch.load(path, map_location='cpu', weights_only=False)
        if not migration:
            infos = validate_checkpoint(saved, self.case)
            if infos['environment'] != serializable(raw.cfg):
                raise ValueError('Height resume runtime configuration differs')
        result = DirectMotorOnPolicyRunner.load(self, path, **kwargs)
        self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]['lr'])
        self.current_learning_iteration = self._completed_updates = saved['infos']['completed_updates']
        state = raw.command_manager.get_term('standup')
        if migration:
            state.height_curriculum = HeightCurriculum()
            self.migration_history = [dict(source_sha256=SOURCE_SHA256, from_interface=source.interface('baseline'),
                to_interface=interface(self.case), completed_updates=2500)]
        else:
            state.height_curriculum.load(saved['infos']['height_curriculum'])
            self.migration_history = saved['infos'].get('migration_history', [])
        state.training_iteration = self._completed_updates - raw.common_step_counter // 24
        state._resample_command(torch.arange(raw.num_envs, device=raw.device))
        state._update_command(None)
        return result

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if num_learning_iterations != 500 or self._completed_updates != 2500:
            raise ValueError('Authorized budget: exactly A2500 +500 updates; no extra learning')
        course = self.env.unwrapped.command_manager.get_term('standup').height_curriculum
        course.learning_active = True
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len=False)
        finally:
            course.learning_active = False


def train(args):
    if args.iterations != 500 or args.num_envs != 4096 or args.seed != 42:
        raise ValueError('Only two authorized 4096-env seed42 +500-update cases')
    original = validate_source(args.resume)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    cfg = make_env_cfg(args.case)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device), clip_actions=None)
    try:
        runner_cfg = height_base.runner_cfg(42)
        runner_cfg.update(save_interval=100, max_iterations=500,
                          experiment_name='chuanliantui_height_curriculum', run_name=args.case)
        for name, value in (('interface',manifest(args.case)),('environment',serializable(cfg)),('runner',runner_cfg)):
            (args.run_dir / f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
        from .getup_curriculum import snapshot_source
        snapshot_source(args.run_dir)
        runner = HeightRunner(env, runner_cfg, str(args.run_dir), case=args.case, device=args.device)
        runner.load(args.resume, migration=True)
        initial = args.run_dir / 'model_resume_initial.pt'
        runner.save(str(initial))
        loaded = torch.load(initial,map_location='cpu',weights_only=False)
        equal = {k:source.state_equal(original[k],loaded[k]) for k in
                 ('actor_state_dict','critic_state_dict','optimizer_state_dict')}
        validate_checkpoint(loaded,args.case)
        if not all(equal.values()) or loaded['infos']['height_curriculum']['level_m'] != .21:
            raise ValueError('Full initial migration failed; no learning')
        verification = dict(state_equal=equal,source_sha256=SOURCE_SHA256,case=args.case,
            initial_updates=2500,target_updates=3000,initial_learning_rate=runner.alg.learning_rate,
            curriculum=loaded['infos']['height_curriculum'])
        (args.run_dir/'migration-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
        print('HEIGHT_MIGRATION '+json.dumps(verification),flush=True)
        runner.learn(500,init_at_random_ep_len=False)
    finally:
        env.close()
