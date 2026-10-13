"""Two authorized A2500 +1000 continuations; all historical kernels unchanged."""
import hashlib
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper

from . import recovery_factors as source, huananhu_v14 as base, huananhu_v14_height as height_base
from ..reference_training import huananhu_rewards
from ..reference_training.entry import serializable
from ..reference_training.huananhu_observations import OBSERVATION_CONTRACT
from ..reference_training.pd_action import CONTROL_CONTRACT
from ..reference_training.runner import DirectMotorOnPolicyRunner

INTERFACE = 'chuanliantui-mjlab-height-knee-v1'
SOURCE_SHA256 = '706c8387cba4815b491be3f5914a5951881c2966edabbd5fbb90d0da9c9d3dbd'
CASES = ('knee_gate', 'knee_only')
GATED_TERMS = height_base.GATED_TERMS


def height_gate(error, case):
    full, off = (.03, .06) if case == 'knee_gate' else (.05, .10)
    return torch.nan_to_num(((off - error.abs()) / (off - full)).clamp(0., 1.),
                            nan=0., posinf=0., neginf=0.)


class RewardState(base.RewardState):
    def __init__(self, env, case):
        super().__init__(env)
        if case not in CASES:
            raise ValueError('Unknown height-knee case')
        self.case = case
        self.knee_kernel = huananhu_rewards.RewardState(env)

    def compute(self, env):
        if self.step == env.common_step_counter:
            return self.values
        # Original V14 exponential velocity values, before ANY height gate.
        values = dict(super().compute(env))
        state = env.command_manager.get_term('standup')
        self.transition_numerical_failures = int(state.numerical_failure.sum())
        height = getattr(state, 'height_above_ground', self.robot.data.root_link_pos_w[:, 2])
        gate = height_gate(height - state.command[:, 2], self.case)
        for name in GATED_TERMS:
            values[name] = values[name] * gate
        values['knee_limit'] = self.knee_kernel.compute(env)['knee_limit']
        self.values = values
        return values


def reward(env, *, name, case):
    if not hasattr(env, '_huananhu_rewards'):
        env._huananhu_rewards = RewardState(env, case)
    if env._huananhu_rewards.case != case:
        raise ValueError('Mixed height-knee reward cases')
    return env._huananhu_rewards.compute(env)[name]


def make_env_cfg(case, num_envs=4096, seed=42, play=False):
    if case not in CASES:
        raise ValueError('Unknown case')
    cfg = source.make_env_cfg('baseline', num_envs, seed, play)
    cfg.rewards = {name: RewardTermCfg(func=reward, weight=term.weight,
        params={'name': name, 'case': case}) for name, term in cfg.rewards.items()}
    cfg.rewards['knee_limit'] = RewardTermCfg(func=reward, weight=-.5,
        params={'name': 'knee_limit', 'case': case})
    return cfg


def interface(case):
    return f'{INTERFACE}:{case}'


def manifest(case):
    if case not in CASES:
        raise ValueError('Unknown case')
    return dict(interface=interface(case), case=case, source=source.manifest('baseline'),
        source_checkpoint_sha256=SOURCE_SHA256,
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        knee_weight=-.5, knee_margin_rad=.15, knee_joints=['lf1', 'rf1'],
        velocity_height_gate=[.03, .06] if case == 'knee_gate' else [.05, .10],
        gated_terms=list(GATED_TERMS), environment=serializable(make_env_cfg(case)),
        ppo=height_base.runner_cfg(42), budget=dict(initial_updates=2500, added_updates=1000,
            final_updates=3500, num_envs=4096, seed=42),
        command_curriculum='original A: dash enabled; spin_low threshold3000 retained; no height curriculum')


def validate_source(path, num_envs=4096, seed=42):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Only the specified A2500 SHA256 is accepted')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    infos = source.validate_checkpoint(saved, 'baseline')
    if infos['completed_updates'] != 2500 or infos['environment'] != source.recovery.runtime_environment(
            source.make_env_cfg('baseline', num_envs, seed)):
        raise ValueError('A2500 environment/population/seed/count mismatch')
    return saved


def validate_checkpoint(saved, case):
    infos = saved.get('infos') or {}
    if (infos.get('interface') != manifest(case)
            or infos.get('experiment_profile_contract') != interface(case)
            or infos.get('observation_contract') != OBSERVATION_CONTRACT
            or infos.get('control_contract') != CONTROL_CONTRACT
            or infos.get('environment') != source.recovery.runtime_environment(make_env_cfg(case))
            or not isinstance(infos.get('completed_updates'), int)
            or not 2500 <= infos['completed_updates'] <= 3500
            or any(k not in saved for k in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'iter'))):
        raise ValueError('Height-knee full checkpoint/interface mismatch')
    return infos


class HeightKneeRunner(base.V14Runner):
    def __init__(self, *args, case, **kwargs):
        self.case, self.migration_history, self.actual_updates = case, [], 0
        super().__init__(*args, **kwargs)
        self.logger.git_status_repos.clear()
        original_update = self.alg.update
        def update():
            result = original_update()
            self.actual_updates += 1
            return result
        self.alg.update = update
        original_log = self.logger.log
        def log(**kw):
            result = original_log(**kw)
            # Save at completed2600/2700/.../3500, with zero-based filenames.
            if self.logger.writer is not None and (kw['it'] + 1) % 100 == 0:
                self.save(str(Path(self.logger.log_dir) / f"model_{kw['it']}.pt"))
            return result
        self.logger.log = log

    def save(self, path, infos=None):
        updates = self.current_learning_iteration + 1 if self._learning else self._completed_updates
        if updates != self.actual_updates:
            raise ValueError('Checkpoint count differs from actual PPO updates')
        if self._learning and updates % 100:
            return  # Ignore upstream's zero-based100 interval.
        if Path(path).exists():
            if self._learning and updates == 3500:
                return  # Upstream final save repeats the already saved3500 model.
            raise FileExistsError(path)
        DirectMotorOnPolicyRunner.save(self, path, {**(infos or {}), 'interface': manifest(self.case),
            'experiment_profile_contract': interface(self.case), 'observation_contract': OBSERVATION_CONTRACT,
            'environment': serializable(self.env.unwrapped.cfg), 'completed_updates': updates,
            'migration_history': self.migration_history})

    def load(self, path, *, migration=False, **kwargs):
        raw = self.env.unwrapped
        saved = validate_source(path, raw.num_envs, raw.cfg.seed) if migration else torch.load(
            path, map_location='cpu', weights_only=False)
        if not migration:
            infos = validate_checkpoint(saved, self.case)
            if infos['environment'] != serializable(raw.cfg):
                raise ValueError('Runtime environment differs from own checkpoint')
        result = DirectMotorOnPolicyRunner.load(self, path, **kwargs)
        self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]['lr'])
        self.current_learning_iteration = self._completed_updates = self.actual_updates = saved['infos']['completed_updates']
        self.migration_history = list(saved['infos'].get('migration_history', []))
        if migration:
            self.migration_history.append(dict(source_sha256=SOURCE_SHA256,
                from_interface=source.interface('baseline'), to_interface=interface(self.case), completed_updates=2500))
        state = raw.command_manager.get_term('standup')
        state.training_iteration = self._completed_updates - raw.common_step_counter // 24
        state._update_command(None)
        return result

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if num_learning_iterations != 1000 or self._completed_updates != 2500:
            raise ValueError('Authorized budget is exactly A2500 +1000; no extra learning')
        return super().learn(1000, init_at_random_ep_len=False)


def train(args):
    if args.iterations != 1000 or args.num_envs != 4096 or args.seed != 42:
        raise ValueError('Exactly two authorized4096-env seed42 +1000 cases')
    original = validate_source(args.resume)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    cfg = make_env_cfg(args.case)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device), clip_actions=None)
    try:
        runner_cfg = height_base.runner_cfg(42)
        runner_cfg.update(save_interval=100, max_iterations=1000,
            experiment_name='chuanliantui_height_knee', run_name=args.case)
        for name, value in (('interface', manifest(args.case)), ('environment', serializable(cfg)), ('runner', runner_cfg)):
            (args.run_dir / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
        from .getup_curriculum import snapshot_source
        snapshot_source(args.run_dir)
        runner = HeightKneeRunner(env, runner_cfg, str(args.run_dir), case=args.case, device=args.device)
        runner.load(args.resume, migration=True)
        initial = args.run_dir / 'model_resume_initial.pt'
        runner.save(str(initial))
        saved = torch.load(initial, map_location='cpu', weights_only=False)
        equal = {k: source.state_equal(original[k], saved[k]) for k in
            ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict')}
        validate_checkpoint(saved, args.case)
        if not all(equal.values()) or saved['infos']['completed_updates'] != 2500:
            raise ValueError('Full initial migration failed; no learning')
        verification = dict(state_equal=equal, source_sha256=SOURCE_SHA256, case=args.case,
            initial_updates=2500, target_updates=3500, initial_learning_rate=runner.alg.learning_rate)
        (args.run_dir / 'migration-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
        print('HEIGHT_KNEE_MIGRATION ' + json.dumps(verification), flush=True)
        runner.learn(1000, init_at_random_ep_len=False)
        assert runner.actual_updates == 3500
    finally:
        env.close()
