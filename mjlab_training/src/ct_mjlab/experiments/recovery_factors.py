"""Explicit, independent continuations for the 2000-update recovery audit."""
import hashlib
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

from . import huananhu_v14 as base, huananhu_v14_height as source
from . import huananhu_v14_recovery as recovery
from .wheel_support_goals import PARAMETERS, STAGES, predicates
from ..reference_training.entry import serializable
from ..reference_training.huananhu_observations import OBSERVATION_CONTRACT
from ..reference_training.pd_action import CONTROL_CONTRACT
from ..reference_training.runner import DirectMotorOnPolicyRunner

INTERFACE = 'chuanliantui-mjlab-recovery-factors-v1'
SOURCE_SHA256 = recovery.MIGRATION_SHA256
CASES = {
    'baseline': 'Continue the original height profile, with shared read-only load probes',
    'recovery_reference': 'Original recovery curriculum, with shared read-only load probes',
    'skip_righting': 'Only restart the recovery curriculum at wheel_balance',
    'keep_height': 'Only preserve original height rewards before the final stage',
    'keep_contact': 'Only preserve undesired_contact in the righting stage',
    'bearing_reward': 'Only replace the 5N-saturated wheel reward by ground load bearing',
}


def wheel_force(env):
    return (-env.scene['wheel_ground_probe'].data.force[..., 2]).clamp_min(0.)


def other_force(env):
    return (-env.scene['nonwheel_ground_probe'].data.force[..., 2]).clamp_min(0.).sum(-1)


def weight(env):
    return env.sim.model.body_mass[:, env.scene['robot'].indexing.body_ids].sum(-1) * 9.81


def goals(env):
    state, robot = env.command_manager.get_term('standup'), env.scene['robot'].data
    return predicates(gravity=robot.projected_gravity_b, wheel_support=wheel_force(env),
        wheel_contact=env.scene['wheel_ground_probe'].data.found > 0,
        nonwheel_support=other_force(env), robot_weight=weight(env),
        base_contact=env.scene['base_contact'].data.force.norm(dim=-1).amax(-1),
        linear_velocity=robot.root_link_lin_vel_b, angular_velocity=robot.root_link_ang_vel_b,
        height=robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2],
        target_height=state.command[:, 2], valid=~state.numerical_failure)


def add_probes(cfg):
    # Leave every original sensor intact. These probes never apply forces.
    ground = ContactMatch(mode='geom', pattern='terrain')
    cfg.scene.sensors += tuple(ContactSensorCfg(name=name,
        primary=ContactMatch(mode='body', pattern=pattern, entity='robot'),
        secondary=ground, secondary_policy='error', fields=('found', 'force'), reduce='netforce')
        for name, pattern in (
            ('wheel_ground_probe', '^(lfwheel|rfwheel)$'),
            ('nonwheel_ground_probe', '^(base_link|[lr]f(0|1|00|01|02|03))$')))
    return cfg


class RewardState(recovery.RewardState):
    def __init__(self, env, case):
        super().__init__(env)
        self.case = case

    def compute(self, env):
        if self.step == env.common_step_counter:
            return self.values
        values = super().compute(env)
        state = env.command_manager.get_term('standup')
        if self.case == 'keep_height' and state.curriculum.stage < 2:
            height = getattr(state, 'height_above_ground', self.robot.data.root_link_pos_w[:, 2])
            error = height - state.command[:, 2]
            # Exact original V14 formulas; final-stage recovery formulas stay intact.
            values['track_height_exp_tight'] = torch.exp(-error.clamp(-.15, .15).square() / .001)
            values['track_height_square'] = (10. * error).square()
        elif self.case == 'keep_contact' and state.curriculum.stage == 0:
            force = env.scene['undesired_contact'].data.force_history
            values['undesired_contact'] = (force.norm(dim=-1).amax(dim=(1, 2)) > 3.).float()
        elif self.case == 'bearing_reward' and state.curriculum.stage >= 1:
            upright = ((1. - self.robot.data.projected_gravity_b[:, 2]) * .5).clamp(0., 1.)
            support = (wheel_force(env) / (.5 * weight(env)[:, None])).clamp(0., 1.).amin(-1)
            support *= (env.scene['wheel_ground_probe'].data.found > 0).all(-1)
            support *= other_force(env) <= PARAMETERS['maximum_nonwheel_weight_fraction'] * weight(env)
            values['wheel_balance'] = support * upright
        self.values = {name: torch.nan_to_num(value, nan=0., posinf=0., neginf=0.)
                       .masked_fill(state.numerical_failure, 0.) for name, value in values.items()}
        self.values['termination'] = env.reset_terminated.float()
        return self.values


def reward(env, *, name, case):
    if not hasattr(env, '_huananhu_rewards'):
        env._huananhu_rewards = RewardState(env, case)
    if env._huananhu_rewards.case != case:
        raise ValueError('Mixed single-factor reward configurations')
    return env._huananhu_rewards.compute(env)[name]


def make_env_cfg(case, num_envs=4096, seed=42, play=False):
    if case not in CASES:
        raise ValueError('Unknown recovery factor')
    cfg = (source if case == 'baseline' else recovery).make_env_cfg(num_envs, seed, play)
    if case in ('keep_height', 'keep_contact', 'bearing_reward'):
        cfg.rewards = {name: RewardTermCfg(func=reward, weight=term.weight,
            params={'name': name, 'case': case}) for name, term in cfg.rewards.items()}
    return add_probes(cfg)


def initialize_stage(env, case):
    if case != 'baseline':
        state = env.command_manager.get_term('standup')
        state.curriculum.stage = int(case == 'skip_righting')


def interface(case):
    return f'{INTERFACE}:{case}'


def manifest(case):
    return dict(interface=interface(case), case=case, intervention=CASES[case],
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source=source.manifest(), recovery_reference=recovery.manifest(),
        goals_sha256=hashlib.sha256(Path(__file__).with_name('wheel_support_goals.py').read_bytes()).hexdigest(),
        environment=serializable(make_env_cfg(case)),
        initial_stage=None if case == 'baseline' else int(case == 'skip_righting'),
        evaluation_criteria=PARAMETERS)


def validate_checkpoint(saved, case):
    infos = saved.get('infos') or {}
    if (infos.get('interface') != manifest(case) or infos.get('experiment_profile_contract') != interface(case)
            or infos.get('observation_contract') != OBSERVATION_CONTRACT
            or infos.get('control_contract') != CONTROL_CONTRACT):
        raise ValueError('Factor checkpoint interface/source mismatch')
    if not isinstance(infos.get('completed_updates'), int):
        raise ValueError('Missing completed update count')
    if any(k not in saved for k in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'iter')):
        raise ValueError('Full training state required')
    return infos


def state_equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


def validate_source(path, num_envs=4096, seed=42):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Only the evaluated height model_1999 is accepted for factor migration')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    source.validate_checkpoint(saved)
    if saved['infos']['completed_updates'] != 2000:
        raise ValueError('Expected exactly 2000 completed source updates')
    expected = recovery.runtime_environment(source.make_env_cfg(num_envs, seed))
    if saved['infos']['environment'] != expected:
        raise ValueError('Source population/seed/environment mismatch')
    return saved


class FactorRunner(base.V14Runner):
    def __init__(self, *args, case, **kwargs):
        self.case = case
        self.migration_history = []
        super().__init__(*args, **kwargs)

    def save(self, path, infos=None):
        updates = self.current_learning_iteration + 1 if self._learning else self._completed_updates
        state = self.env.unwrapped.command_manager.get_term('standup')
        DirectMotorOnPolicyRunner.save(self, path, {**(infos or {}), 'interface': manifest(self.case),
            'experiment_profile_contract': interface(self.case), 'observation_contract': OBSERVATION_CONTRACT,
            'environment': serializable(self.env.unwrapped.cfg), 'completed_updates': updates,
            'recovery_curriculum': None if self.case == 'baseline' else state.curriculum.state_dict(),
            'migration_history': self.migration_history})

    def load(self, path, *, migration=False, **kwargs):
        raw = self.env.unwrapped
        if migration:
            saved = validate_source(path, raw.num_envs, raw.cfg.seed)
        else:
            saved = torch.load(path, map_location='cpu', weights_only=False)
            validate_checkpoint(saved, self.case)
            if saved['infos']['environment'] != serializable(raw.cfg):
                raise ValueError('Factor resume environment mismatch')
        result = DirectMotorOnPolicyRunner.load(self, path, **kwargs)
        self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]['lr'])
        self.current_learning_iteration = self._completed_updates = saved['infos']['completed_updates']
        self.migration_history = list(saved['infos'].get('migration_history', []))
        state = raw.command_manager.get_term('standup')
        if migration:
            initialize_stage(raw, self.case)
            self.migration_history.append(dict(source_sha256=SOURCE_SHA256,
                from_interface=source.INTERFACE, to_interface=interface(self.case), completed_updates=2000))
        elif self.case != 'baseline':
            state.curriculum.load(saved['infos']['recovery_curriculum'])
        state.training_iteration = self._completed_updates - raw.common_step_counter // 24
        state._update_command(None)
        return result


def train(args):
    if args.iterations < 1 or args.num_envs != 4096 or args.seed != 42:
        raise ValueError('This campaign uses positive updates, 4096 environments and seed42')
    validate_source(args.resume, args.num_envs, args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    cfg = make_env_cfg(args.case, args.num_envs, args.seed)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device), clip_actions=None)
    try:
        initialize_stage(env.unwrapped, args.case)
        runner_cfg = source.runner_cfg(args.seed)
        runner_cfg.update(save_interval=100, max_iterations=args.iterations,
                          experiment_name='chuanliantui_recovery_factors', run_name=args.case)
        for name, value in (('interface', manifest(args.case)), ('environment', serializable(cfg)),
                            ('runner', runner_cfg)):
            (args.run_dir / f'{name}.json').write_text(json.dumps(value, indent=2) + '\n')
        from .getup_curriculum import snapshot_source
        snapshot_source(args.run_dir)
        runner = FactorRunner(env, runner_cfg, str(args.run_dir), case=args.case, device=args.device)
        runner.load(args.resume, migration=True)
        initial = args.run_dir / 'model_resume_initial.pt'
        runner.save(str(initial))
        original = torch.load(args.resume, map_location='cpu', weights_only=False)
        loaded = torch.load(initial, map_location='cpu', weights_only=False)
        equal = {k: state_equal(original[k], loaded[k]) for k in (
            'actor_state_dict', 'critic_state_dict', 'optimizer_state_dict')}
        if not all(equal.values()):
            raise ValueError('Full source migration failed before learning')
        verification = dict(state_equal=equal, source_sha256=SOURCE_SHA256, case=args.case,
            initial_updates=2000, target_updates=2000 + args.iterations,
            initial_learning_rate=runner.alg.learning_rate, curriculum=loaded['infos']['recovery_curriculum'])
        (args.run_dir / 'migration-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
        print('FACTOR_MIGRATION ' + json.dumps(verification), flush=True)
        print('Training output: ' + str(args.run_dir), flush=True)
        runner.learn(args.iterations, init_at_random_ep_len=False)
    finally:
        env.close()
