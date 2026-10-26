"""Opt-in SMP reward on the existing 39/82 multimode PPO interface."""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ..envs.chuanliantui import multimode_mdp
from ..envs.chuanliantui.huananhu_poses import POSE_JOINT_NAMES
from ..envs.chuanliantui.model import MODEL_XML
from ..envs.chuanliantui.multimode_runner import MultimodeOnPolicyRunner
from ..envs.chuanliantui.multimode_observations import OBSERVATION_CONTRACT
from ..envs.chuanliantui.multimode_observations import configure_wheel_angle_observations
from ..wheel_angles import HANDOVER_OBSERVATION_CONTRACT, HANDOVER_WHEEL_ANGLES
from ..envs.chuanliantui.runner import DirectMotorOnPolicyRunner
from .features import FEATURE_DIM, SAMPLE_DT, MotionHistory, MotionStateReader
from .prior import load_prior

LEGACY_PROFILE = 'chuanliantui-smp-ppo-normal-ground-v1'
ADAPTIVE_STAND_PROFILE = 'chuanliantui-smp-ppo-normal-ground-stand-v2'
PROFILE = 'chuanliantui-smp-ppo-normal-ground-stand-unit-v3'
WHEEL_PROFILE = 'chuanliantui-smp-ppo-normal-ground-stand-wheel-handover-v4'
PPO_PROFILES = {OBSERVATION_CONTRACT: PROFILE, HANDOVER_OBSERVATION_CONTRACT: WHEEL_PROFILE}


def experiment_profile(env_cfg):
    from .isaac_standup_rewards import reward as isaac_reward
    if any(term.func is isaac_reward for term in env_cfg.rewards.values()):
        from .finetune import PROFILE_ISAAC, PROFILE_ISAAC_V2, PROFILE_ISAAC_V3
        if env_cfg.rewards['tracking_lin_vel'].params.get('convex_velocity'):
            return PROFILE_ISAAC_V3
        return PROFILE_ISAAC_V2 if env_cfg.rewards['tracking_lin_vel'].params.get('tight_velocity') else PROFILE_ISAAC
    if 'zero_command_brake' in env_cfg.rewards:
        from .finetune import PROFILE as FINETUNE_PROFILE, PROFILE_V2
        return PROFILE_V2 if env_cfg.rewards['zero_command_brake'].params.get('velocity_scale') == .01 else FINETUNE_PROFILE
    return PPO_PROFILES[observation_contract(env_cfg)]


def checkpoint_bc_finetune(path):
    if path is None:
        return False
    from .finetune import PROFILE as FINETUNE_PROFILE, PROFILE_V2, PROFILE_ISAAC, PROFILE_ISAAC_V2, PROFILE_ISAAC_V3
    saved = torch.load(path, map_location='cpu', weights_only=False)
    profile = (saved.get('infos') or {}).get('experiment_profile_contract')
    if profile == PROFILE_ISAAC_V3:
        return 'isaac-standup-v3'
    if profile == PROFILE_ISAAC_V2:
        return 'isaac-standup-v2'
    if profile == PROFILE_ISAAC:
        return 'isaac-standup'
    return 'v2' if profile == PROFILE_V2 else profile == FINETUNE_PROFILE


def observation_contract(env_cfg):
    flags=[]
    for name in ('actor','critic'):
        cfg=env_cfg.observations[name].terms['proprioception'].params.get('cfg')
        flags.append((getattr(cfg,'wheel_angle_displacement',False),getattr(cfg,'wheel_angle_handover_reset',False)))
    if flags==[(False,False)]*2:
        return OBSERVATION_CONTRACT
    if flags==[(True,True)]*2:
        return HANDOVER_OBSERVATION_CONTRACT
    raise ValueError('SMP PPO requires matching legacy39 or first-recovery wheel39 observations')


def policy_observation_contract(path, *, bc=False):
    """Select input semantics before creating an environment; equal dimensions are insufficient."""
    if path is None:
        return OBSERVATION_CONTRACT
    if bc:
        from ..bc import load_actor
        _,saved=load_actor(path)
        contract=saved['dataset']['observation_contract']
    else:
        infos=torch.load(path,map_location='cpu',weights_only=False).get('infos') or {}
        contract=infos.get('observation_contract')
        from .finetune import PROFILE as FINETUNE_PROFILE, PROFILE_V2, PROFILE_ISAAC, PROFILE_ISAAC_V2, PROFILE_ISAAC_V3
        allowed = {PPO_PROFILES.get(contract)}
        if contract == HANDOVER_OBSERVATION_CONTRACT:
            allowed.update((FINETUNE_PROFILE, PROFILE_V2, PROFILE_ISAAC, PROFILE_ISAAC_V2, PROFILE_ISAAC_V3))
        if infos.get('experiment_profile_contract') not in allowed:
            raise ValueError('Checkpoint is not a compatible fixed-unit SMP PPO profile')
        if contract==HANDOVER_OBSERVATION_CONTRACT and (
                infos.get('wheel_angles')!=HANDOVER_WHEEL_ANGLES or
                infos.get('xml_sha256')!=hashlib.sha256(MODEL_XML.read_bytes()).hexdigest()):
            raise ValueError('Wheel39 PPO checkpoint observation or XML differs')
    if contract not in PPO_PROFILES:
        raise ValueError(f'Unsupported SMP PPO observation contract: {contract!r}')
    return contract


def checkpoint_smp_weight(path):
    if path is None:
        return 1.
    weight = torch.load(path, map_location='cpu', weights_only=False)['infos']['smp']['weight']
    if weight not in (0., 1.):
        raise ValueError('SMP comparison supports reward weights 0 or 1')
    return weight


@dataclass(kw_only=True)
class SmpCommandCfg(multimode_mdp.MultimodeCommandCfg):
    def build(self, env):
        return SmpCommand(self, env)


class SmpCommand(multimode_mdp.MultimodeCommand):
    @property
    def policy_active(self):
        # 观测中的接管标志与实际电机门控共用同一配置，复位首拍也一致。
        hold_s = self._env.cfg.actions['pd'].ground_start_hold_s
        hold_steps = math.ceil(hold_s/self._env.step_dt-1e-9)
        return ~self.ground_start | (self._env.episode_length_buf >= hold_steps)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        # 当前阶段只学正常起立和零速站稳，重采样后也不放开移动。
        self.sampled[env_ids, :2] = 0.
        self.sampled[env_ids, 2] = .33648  # 教师0.30 m腿长对应的机身高度。
        self.heading[env_ids] = False
        self.special_mode[env_ids] = -1
        self.standing[env_ids] = True
        self.zero_hold[env_ids] = True

    def reset(self, env_ids):
        result = super().reset(env_ids)
        self.ground_command[env_ids] = 0.
        return result


def reset_pose(env, env_ids):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    robot = env.scene['robot']
    if not hasattr(env, '_smp_ground_poses'):
        with np.load(MODEL_XML.with_name('huananhu_pose_bank.npz')) as bank:
            poses = np.concatenate((bank['negative_left_pos'], bank['negative_right_pos']), -1)
        order = [POSE_JOINT_NAMES.index(name) for name in robot.joint_names]
        env._smp_ground_poses = torch.as_tensor(poses[:, order], device=env.device)
    rows = torch.randint(len(env._smp_ground_poses), (len(ids),), device=env.device)
    joint = env._smp_ground_poses[rows]
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] = env.scene.env_origins[ids]
    root[:, 2] += .25
    root[:, 3:7] = root.new_tensor([1., 0., 0., 0.])
    root[:, 7:] = 0.
    robot.write_joint_state_to_sim(joint, torch.zeros_like(joint), env_ids=ids)
    robot.write_root_state_to_sim(root, env_ids=ids)
    env.command_manager.get_term('standup').ground_start[ids] = True
    if hasattr(env, '_huananhu_rewards'):
        env._huananhu_rewards.step = -1
    if hasattr(env, '_isaac_standup_rewards'):
        env._isaac_standup_rewards.reset(ids)


def inverted(env):
    # 倒置立即结束本回合，自动复位；不执行倒置恢复。
    return env.scene['robot'].data.projected_gravity_b[:, 2] > 0.


class SmpReward:
    def __init__(self, cfg, env):
        self.prior, metadata = load_prior(cfg.params['prior_path'], env.device)
        if metadata['dataset']['xml_sha256'] != hashlib.sha256(MODEL_XML.read_bytes()).hexdigest():
            raise ValueError('SMP prior XML differs from the current robot')
        if abs(env.step_dt-SAMPLE_DT) > 1e-9:
            raise ValueError('SMP motion history requires 100 Hz control')
        self.prior.freeze()
        # 固定为已验证的单位尺度，不让初期随机策略改变评分标准。
        normalizer = self.prior.error_normalizer
        normalizer.mean_abs.fill_(1.)
        normalizer.count.zero_()
        normalizer.pending_count.zero_()
        normalizer.pending_sum.zero_()
        self.reader = MotionStateReader(env)
        self.history = MotionHistory(env.num_envs, env.device)
        self.generator = torch.Generator(device=env.device).manual_seed(env.cfg.seed + 7919)
        self.digest = hashlib.sha256(Path(cfg.params['prior_path']).read_bytes()).hexdigest()
        self.contract = dict(profile=experiment_profile(env.cfg), prior_sha256=self.digest,
                             weight=cfg.weight*cfg.params.get('reward_scale', 1.),
                             sample_dt=SAMPLE_DT, initial='normal-ground-zero-velocity',
                             command='zero-vx-zero-yaw',
                             ground_start_hold_s=env.cfg.actions['pd'].ground_start_hold_s,
                             normalizer_mode='fixed-unit', normalizer_scale=[1., 1., 1.],
                             normalizer_samples=0)
        self.last_reward = torch.zeros(env.num_envs, device=env.device)
        self.last_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._smp_reward = self

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.history.reset(ids)
        self.last_reward[ids] = 0.
        self.last_valid[ids] = False

    @torch.no_grad()
    def __call__(self, env, *, prior_path, training, reward_scale=1.):
        # RewardManager在自动复位前调用；刷新派生量后与离线特征使用同一物理时刻。
        env.sim.forward()
        windows, valid = self.history.append(self.reader.features())
        self.last_valid.copy_(valid)
        self.last_reward.zero_()
        raw_error = self.last_reward.new_zeros(())
        if valid.any():
            selected = windows[valid]
            noise = torch.randn(3, len(selected), selected.shape[1]*FEATURE_DIM,
                                device=env.device, generator=self.generator)
            errors = self.prior.score_errors(selected, noise=noise)
            raw_error = errors.mean()
            self.last_reward[valid] = torch.exp(-6*self.prior.error_normalizer(errors).mean(-1))
        env.extras['log']['SMP/reward'] = self.last_reward.mean()*reward_scale
        env.extras['log']['SMP/raw_error'] = raw_error
        env.extras['log']['SMP/error_scale'] = self.prior.error_normalizer.mean_abs.mean().clone()
        env.extras['log']['SMP/valid_fraction'] = valid.float().mean()
        env.extras['log']['SMP/normalizer_count'] = self.prior.error_normalizer.count.float().clone()
        # 对照组保留相同刷新、窗口和独立评分RNG，只取消SMP对PPO奖励的贡献。
        return self.last_reward*reward_scale


def configure_smp(env_cfg, train_cfg, prior_path, *, play=False, policy_contract=OBSERVATION_CONTRACT, reward_weight=1., bc_finetune=False):
    path = Path(prior_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'SMP prior not found: {path}')
    if reward_weight not in (0., 1.):
        raise ValueError('SMP comparison supports reward weights 0 or 1')
    if policy_contract not in PPO_PROFILES:
        raise ValueError(f'Unsupported SMP PPO observation contract: {policy_contract!r}')
    if policy_contract==HANDOVER_OBSERVATION_CONTRACT:
        configure_wheel_angle_observations(env_cfg,handover_reset=True)
    if observation_contract(env_cfg)!=policy_contract:
        raise ValueError('Configured SMP PPO observations differ from the requested contract')
    env_cfg.commands['standup'] = SmpCommandCfg()
    env_cfg.events['reset_pose'] = EventTermCfg(func=reset_pose, mode='reset')
    env_cfg.terminations['inverted'] = TerminationTermCfg(func=inverted)
    # 先刷新物理状态，再计算原任务奖励；dt仍由RewardManager统一乘一次。
    env_cfg.rewards = {'smp_motion': RewardTermCfg(func=SmpReward, weight=1.,
        params={'prior_path': str(path), 'training': not play,
                **({'reward_scale': 0.} if reward_weight == 0. else {})}), **env_cfg.rewards}
    train_cfg.experiment_name = ('chuanliantui_huananhu_multimode_smp_wheel_handover_v4'
        if policy_contract==HANDOVER_OBSERVATION_CONTRACT else 'chuanliantui_huananhu_multimode_smp_stand_unit_v3')
    if reward_weight == 0.:
        train_cfg.experiment_name += '_taskonly'
    if bc_finetune:
        if policy_contract != HANDOVER_OBSERVATION_CONTRACT:
            raise ValueError('BC refinement requires the wheel39 handover observation contract')
        from .finetune import configure_finetune
        configure_finetune(env_cfg, train_cfg, variant=bc_finetune)
    return env_cfg, train_cfg


class SmpOnPolicyRunner(MultimodeOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.observation_contract=observation_contract(env.unwrapped.cfg)
        self.experiment_profile=experiment_profile(env.unwrapped.cfg)
        from .finetune import PROFILE as FINETUNE_PROFILE, PROFILE_V2, PROFILE_ISAAC, PROFILE_ISAAC_V2, PROFILE_ISAAC_V3
        if self.experiment_profile in (FINETUNE_PROFILE, PROFILE_V2, PROFILE_ISAAC, PROFILE_ISAAC_V2, PROFILE_ISAAC_V3):
            from mjlab.utils.torch import configure_torch_backends
            configure_torch_backends(allow_tf32=False)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        if self.is_distributed or train_cfg['num_steps_per_env'] != 24:
            raise ValueError('SMP PPO currently supports one GPU and 24-step rollout batches')

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        from .finetune import ProtectedPPO
        if isinstance(self.alg, ProtectedPPO):
            # 基类保存的是末轮下标；续训从已完成轮数开始，避免重复轮号。
            self.current_learning_iteration = self.alg.completed_iterations
        if hasattr(self, 'phase_mixing') and not getattr(self, '_phase_mixing_ready', False):
            from .phase_sampling import prepare
            prepare(self)
        return super().learn(num_learning_iterations, init_at_random_ep_len=False)

    def enable_phase_mixing(self):
        from .finetune import PROFILE_ISAAC_V2, PROFILE_ISAAC_V3
        from .phase_sampling import CONTRACT
        if self.experiment_profile not in (PROFILE_ISAAC_V2, PROFILE_ISAAC_V3):
            raise ValueError('Phase mixing supports Isaac tight-velocity v2/v3 only')
        if not hasattr(self, 'phase_mixing'):
            self.phase_mixing = dict(contract=CONTRACT, prelude_runs=0, prelude_control_steps_total=0)
            self._phase_mixing_ready = False

    def initialize_from(self, path):
        raise ValueError('SMP PPO starts fresh or resumes an SMP PPO checkpoint; unset legacy init variables')

    def initialize_from_ppo(self, path):
        """Transfer the accepted Isaac actor; new rewards require a fresh critic/Adam."""
        from .finetune import (PROFILE_ISAAC, PROFILE_ISAAC_V2, PROFILE_ISAAC_V3,
                              RECIPE_ISAAC, RECIPE_ISAAC_V2, checked_recipe)
        saved = torch.load(path, map_location=self.device, weights_only=False)
        infos = saved.get('infos') or {}
        source_smp = {**(infos.get('smp') or {}), 'profile': self.experiment_profile}
        source_profile = infos.get('experiment_profile_contract')
        wanted_source = {PROFILE_ISAAC_V2: (PROFILE_ISAAC, RECIPE_ISAAC),
                         PROFILE_ISAAC_V3: (PROFILE_ISAAC_V2, RECIPE_ISAAC_V2)}.get(self.experiment_profile)
        if (wanted_source is None or source_profile != wanted_source[0] or
                policy_observation_contract(path) != self.observation_contract or
                source_smp != self.env.unwrapped._smp_reward.contract):
            raise ValueError('PPO reward transfer requires matching adjacent Isaac recipe and frozen prior')
        checked_recipe(saved.get('bc_finetune', {}).get('recipe', {}), wanted_source[1])
        if self.alg.optimizer.state or self.env.unwrapped.common_step_counter != 0:
            raise ValueError('PPO reward transfer requires a fresh run')
        self.alg._raw_actor.load_state_dict(saved['actor_state_dict'], strict=True)
        # 保留这轮已学到的动作；用它作软参考，价值网络重新学习新奖励。
        self.alg.initialize_reference()
        self.bc_init = infos['bc_init']
        self.ppo_init = dict(sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            source_profile=source_profile, source_iterations=saved['bc_finetune']['completed_iterations'],
            actor_only=True, critic_reset=True, optimizer_reset=True, reference='source_ppo_actor')

    def initialize_from_bc(self, path):
        from ..bc import load_actor
        actor, saved = load_actor(path, self.device)
        if self.alg.optimizer.state or self.env.unwrapped.common_step_counter != 0:
            raise ValueError('BC initialization requires a fresh PPO run')
        metadata = saved['dataset']
        if (metadata['observation_contract'] != self.observation_contract or
                metadata['xml_sha256'] != hashlib.sha256(MODEL_XML.read_bytes()).hexdigest()):
            raise ValueError('BC observation or robot XML differs')
        if self.observation_contract==HANDOVER_OBSERVATION_CONTRACT and metadata.get('wheel_angles')!=HANDOVER_WHEEL_ANGLES:
            raise ValueError('BC wheel-angle observation definition differs')
        root = Path(__file__).resolve().parents[4]
        for name, digest in metadata['control_sources'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
                raise ValueError('BC student action implementation differs')
        # 只初始化actor；critic和PPO优化器从新状态开始，不导入BC优化器。
        self.alg._raw_actor.load_state_dict(actor.state_dict(), strict=True)
        from .finetune import ProtectedPPO
        if isinstance(self.alg, ProtectedPPO):
            self.alg.initialize_reference()
        self.bc_init = dict(sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                            dataset_fingerprint=metadata['fingerprint'],
                            updates=saved.get('total_actor_updates', saved['iteration']),
                            action_std=.05,
                            **({'observation_contract':self.observation_contract,
                                'wheel_angles':HANDOVER_WHEEL_ANGLES} if self.observation_contract==HANDOVER_OBSERVATION_CONTRACT else {}))
        if isinstance(self.alg, ProtectedPPO):
            from .finetune import ACTION_STD
            self.bc_init['action_std'] = list(ACTION_STD)

    def save(self, path, infos=None):
        term = self.env.unwrapped._smp_reward
        DirectMotorOnPolicyRunner.save(self, path, {
            **(infos or {}), 'observation_contract': self.observation_contract,
            'experiment_profile_contract': self.experiment_profile,
            **({'wheel_angles':HANDOVER_WHEEL_ANGLES,
                'xml_sha256':hashlib.sha256(MODEL_XML.read_bytes()).hexdigest()}
               if self.observation_contract==HANDOVER_OBSERVATION_CONTRACT else {}),
            'huananhu_training_iteration': self._effective_training_iteration(),
            'smp': term.contract,
            'smp_normalizer': term.prior.error_normalizer.state_dict(),
            'smp_generator': term.generator.get_state(),
            **({'bc_init': self.bc_init} if hasattr(self, 'bc_init') else {}),
            **({'ppo_init': self.ppo_init} if hasattr(self, 'ppo_init') else {}),
            **({'phase_mixing': self.phase_mixing} if hasattr(self, 'phase_mixing') else {}),
        })

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        infos = saved.get('infos') or {}
        term = self.env.unwrapped._smp_reward
        if (infos.get('experiment_profile_contract') != self.experiment_profile or
                infos.get('observation_contract') != self.observation_contract or
                infos.get('smp') != term.contract):
            raise ValueError('Incompatible SMP PPO checkpoint or prior/reward contract')
        if self.observation_contract==HANDOVER_OBSERVATION_CONTRACT and (
                infos.get('wheel_angles')!=HANDOVER_WHEEL_ANGLES or
                infos.get('xml_sha256')!=hashlib.sha256(MODEL_XML.read_bytes()).hexdigest()):
            raise ValueError('Wheel39 PPO checkpoint observation or XML differs')
        normalizer = infos['smp_normalizer']
        if (int(normalizer['count']) != 0 or int(normalizer['pending_count']) != 0 or
                normalizer['pending_sum'].count_nonzero() or
                not torch.equal(normalizer['mean_abs'], torch.ones_like(normalizer['mean_abs']))):
            raise ValueError('Fixed-unit SMP checkpoint contains adaptive error statistics')
        restored = DirectMotorOnPolicyRunner.load(
            self, path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        owner = self._progress_owner()
        owner.training_iteration = int(infos['huananhu_training_iteration']) - (
            int(self.env.unwrapped.common_step_counter)//24)
        term.prior.error_normalizer.load_state_dict(infos['smp_normalizer'], strict=True)
        term.generator.set_state(infos['smp_generator'].cpu())
        if 'bc_init' in infos:
            self.bc_init = infos['bc_init']
        elif hasattr(self, 'bc_init'):
            del self.bc_init
        if 'ppo_init' in infos:
            self.ppo_init = infos['ppo_init']
        elif hasattr(self, 'ppo_init'):
            del self.ppo_init
        if 'phase_mixing' in infos:
            from .phase_sampling import CONTRACT
            if infos['phase_mixing']['contract'] != CONTRACT:
                raise ValueError('Incompatible episode sampling contract')
            self.enable_phase_mixing()
            self.phase_mixing = infos['phase_mixing']
            self._phase_mixing_ready = False
        elif hasattr(self, 'phase_mixing'):
            del self.phase_mixing
            self._phase_mixing_ready = False
        return restored
