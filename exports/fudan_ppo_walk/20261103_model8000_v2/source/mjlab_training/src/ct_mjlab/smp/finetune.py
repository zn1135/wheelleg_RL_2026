"""Opt-in, nominal-condition PPO refinement of the stable wheel39 BC actor."""
import copy
from dataclasses import replace

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from rsl_rl.algorithms import PPO

PROFILE = 'chuanliantui-smp-ppo-wheel39-bc-finetune-v1'
PROFILE_V2 = 'chuanliantui-smp-ppo-wheel39-bc-finetune-v2'
PROFILE_ISAAC = 'chuanliantui-smp-ppo-wheel39-bc-isaac-standup-v1'
PROFILE_ISAAC_V2 = 'chuanliantui-smp-ppo-wheel39-bc-isaac-standup-v2'
PROFILE_ISAAC_V3 = 'chuanliantui-smp-ppo-wheel39-bc-isaac-standup-v3'
ACTION_STD = (.001, .001, .005, .001, .001, .005)
WHEEL_STD_CHOICES = (.005, .01, .02, .05)
RECIPE = dict(version=PROFILE, action_std=list(ACTION_STD), fixed_std=True,
              critic_warmup_iterations=20, actor_lr=1e-6, critic_lr=1e-4,
              bc_loss_coef=1., actor_kl_limit=.01, fp32=True,
              physics='nominal', observation_noise=False, observation_delay=False,
              action_delay_seconds=[.002, .004], parking_reward='separate-brake-position-v1')
RECIPE_V2 = {**RECIPE, 'version': PROFILE_V2,
             'bc_action_scale': [.05, .05, .1, .05, .05, .1], 'braking_velocity_scale': .01}
RECIPE_ISAAC = {**RECIPE_V2, 'version': PROFILE_ISAAC,
                'parking_reward': 'isaac-standup-pre-unlock-current-height-v1'}
del RECIPE_ISAAC['braking_velocity_scale']
RECIPE_ISAAC_V2 = {**RECIPE_ISAAC, 'version': PROFILE_ISAAC_V2,
                   'parking_reward': 'isaac-standup-tight-xy-velocity-v2'}
RECIPE_ISAAC_V3 = {**RECIPE_ISAAC_V2, 'version': PROFILE_ISAAC_V3,
                   'action_std': [.001, .001, .02, .001, .001, .02],
                   'critic_warmup_iterations': 80,
                   'parking_reward': 'stood-convex-velocity-v3'}


def checked_recipe(saved, base):
    expected = base
    if 'wheel_exploration_std' in saved:
        wheel = saved['wheel_exploration_std']
        if base['version'] not in (PROFILE_ISAAC_V2, PROFILE_ISAAC_V3) or wheel not in WHEEL_STD_CHOICES:
            raise ValueError('Unsupported wheel exploration recipe')
        expected = {**base, 'action_std': [.001, .001, wheel, .001, .001, wheel],
                    'wheel_exploration_std': wheel}
    if saved != expected:
        raise ValueError('Incompatible BC refinement recipe')
    return copy.deepcopy(expected)


def parking_scores(speed, position_error, *, velocity_scale=.05):
    # 独立、连续的减速反馈，不再被远处的旧位置锚点乘成零。
    return 1./(1.+(speed/velocity_scale).square()), 1./(1.+(position_error/.1).square())


def parking_reward(env, *, position=False, velocity_scale=.05):
    state = env.command_manager.get_term('standup')
    data = env.scene['robot'].data
    active = ((state.command[:, :2].abs() < .05).all(-1) & ~state.recover
              & ~state.airborne & (state.jump_phase == 0)
              & (data.projected_gravity_b[:, 2] < -.9) & (state.stable_time >= .5))
    speed = data.root_link_lin_vel_b[:, :2].norm(dim=-1)
    error = (data.root_link_pos_w[:, :2]-state.zero_hold_anchor_xy).norm(dim=-1)
    brake, hold = parking_scores(speed, error, velocity_scale=velocity_scale)
    if position:
        return active.float()*state.zero_hold_anchor_valid.float()*hold
    return active.float()*brake


def configure_finetune(cfg, train, *, variant='v1'):
    variant = 'v1' if variant is True else variant
    if variant not in ('v1', 'v2', 'isaac-standup', 'isaac-standup-v2', 'isaac-standup-v3'):
        raise ValueError('Unknown BC refinement recipe')
    # 使用既有play配置中的名义物理；起立分布和电机门控由configure_smp保留。
    from ..envs.chuanliantui.multimode_env_cfg import make_multimode_env_cfg
    nominal = make_multimode_env_cfg(play=True)
    cfg.events = {name: term for name, term in cfg.events.items()
                  if name in ('reset_pose', 'reset_observations')}
    for name in ('motor_randomization', 'spring_force_bias_range', 'nominal_leg_offset_range'):
        setattr(cfg.actions['pd'], name, getattr(nominal.actions['pd'], name))
    cfg.actions['pd'].action_delay_s = None
    cfg.actions['pd'].action_delay_choices_s = (.002, .004)
    cfg.actions['pd'].randomize_delays = True
    obs_cfg = replace(cfg.observations['actor'].terms['proprioception'].params['cfg'], noise=False, delay=False)
    for group in cfg.observations.values():
        group.terms['proprioception'].params['cfg'] = obs_cfg
    cfg.events['reset_observations'].params['cfg'] = obs_cfg
    cfg.metrics['observation_delay_ms'].params['cfg'] = obs_cfg
    del cfg.rewards['zero_command_hold']
    cfg.rewards['zero_command_brake'] = RewardTermCfg(func=parking_reward, weight=2.)
    cfg.rewards['zero_command_position'] = RewardTermCfg(func=parking_reward, weight=1., params={'position': True})
    train.algorithm.class_name = 'ct_mjlab.smp.finetune:ProtectedPPO'
    train.algorithm.schedule = 'fixed'
    train.algorithm.entropy_coef = 0.
    train.experiment_name += '_bc_finetune_v1'
    if variant in ('v2', 'isaac-standup', 'isaac-standup-v2', 'isaac-standup-v3'):
        cfg.rewards['zero_command_brake'].params['velocity_scale'] = .01
        train.algorithm.class_name = 'ct_mjlab.smp.finetune:ProtectedPPOV2'
        train.experiment_name = train.experiment_name.removesuffix('_v1')+'_v2'
    if variant in ('isaac-standup', 'isaac-standup-v2', 'isaac-standup-v3'):
        from .isaac_standup_rewards import configure_rewards
        configure_rewards(cfg, tight_velocity=variant in ('isaac-standup-v2', 'isaac-standup-v3'),
                          convex_velocity=variant == 'isaac-standup-v3')
        train.algorithm.class_name = 'ct_mjlab.smp.finetune:ProtectedPPOIsaacStandup'
        train.experiment_name = train.experiment_name.removesuffix('_v2')+'_isaac_standup_v1'
        if variant == 'isaac-standup-v2':
            train.algorithm.class_name = 'ct_mjlab.smp.finetune:ProtectedPPOIsaacStandupV2'
            train.experiment_name = train.experiment_name.removesuffix('_v1')+'_v2'
        if variant == 'isaac-standup-v3':
            train.algorithm.class_name = 'ct_mjlab.smp.finetune:ProtectedPPOIsaacStandupV3'
            train.experiment_name = train.experiment_name.removesuffix('_v1')+'_v3'


class ProtectedPPO(PPO):
    """Value warmup, fixed small exploration, and a frozen BC action reference."""
    recipe = RECIPE
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_multi_gpu or self.rnd or self.symmetry or self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError('BC refinement supports the current single-GPU MLP PPO only')
        self.reference = copy.deepcopy(self._raw_actor.mlp).eval().requires_grad_(False)
        self.reference_ready = False
        self.completed_iterations = 0
        self.actor_updates = self.critic_updates = 0
        self._raw_actor.distribution.requires_grad_(False)
        self.optimizer = torch.optim.Adam([
            {'params': list(self.actor.mlp.parameters()), 'lr': self.recipe['actor_lr']},
            {'params': list(self.critic.parameters()), 'lr': self.recipe['critic_lr']}])
        self.learning_rate = self.recipe['actor_lr']

    def initialize_reference(self):
        self.reference.load_state_dict(self._raw_actor.mlp.state_dict())
        with torch.no_grad():
            self._raw_actor.distribution.std_param.copy_(torch.tensor(self.recipe['action_std'], device=self.device))
        self.reference_ready = True

    def set_wheel_std(self, wheel):
        if type(self).recipe['version'] not in (PROFILE_ISAAC_V2, PROFILE_ISAAC_V3) or wheel not in WHEEL_STD_CHOICES or self.storage.step:
            raise ValueError('Wheel noise change requires a fresh tight-velocity rollout')
        # 只改固定探索分布，保留动作均值、价值网络、Adam及冻结参考。
        self.recipe = {**type(self).recipe,
                       'action_std': [.001, .001, wheel, .001, .001, wheel],
                       'wheel_exploration_std': wheel}
        with torch.no_grad():
            self._raw_actor.distribution.std_param.copy_(torch.tensor(self.recipe['action_std'], device=self.device))

    def update(self):
        if not self.reference_ready:
            raise ValueError('BC refinement requires --bc-init or a refinement checkpoint')
        warmup = self.completed_iterations < self.recipe['critic_warmup_iterations']
        actor_stopped = warmup
        totals = dict(value=0., surrogate=0., bc_reference=0., kl=0.)
        batches = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            obs = batch.observations
            values = self.critic(obs)
            if self.use_clipped_value_loss:
                clipped = batch.values+(values-batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((values-batch.returns).square(), (clipped-batch.returns).square()).mean()
            else:
                value_loss = (values-batch.returns).square().mean()
            loss = self.value_loss_coef*value_loss
            surrogate = bc_loss = kl_mean = values.new_zeros(())
            if not actor_stopped:
                self.actor(obs, stochastic_output=True)
                kl_mean = self.actor.get_kl_divergence(batch.old_distribution_params,
                                                       self.actor.output_distribution_params).mean()
                # 超过本轮采样分布的距离后，只继续拟合价值网络。
                if kl_mean.detach() > self.recipe['actor_kl_limit']:
                    actor_stopped = True
                else:
                    log_prob = self.actor.get_output_log_prob(batch.actions)
                    ratio = (log_prob-batch.old_actions_log_prob.squeeze(-1)).exp()
                    advantage = batch.advantages.squeeze(-1)
                    if self.normalize_advantage_per_mini_batch:
                        advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
                    surrogate = torch.maximum(-advantage*ratio,
                        -advantage*ratio.clamp(1.-self.clip_param, 1.+self.clip_param)).mean()
                    with torch.no_grad():
                        reference = self.reference(obs['actor'])
                    mean, std = self.actor.output_distribution_params
                    # 探索噪声与允许偏离BC的动作尺度分开，避免把BC锁得过死。
                    scale = (mean.new_tensor(self.recipe['bc_action_scale'])
                             if 'bc_action_scale' in self.recipe else std)
                    bc_loss = .5*((mean-reference)/scale).square().mean()
                    loss = loss+surrogate+self.recipe['bc_loss_coef']*bc_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.mlp.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            actor_has_grad = any(p.grad is not None for p in self.actor.mlp.parameters())
            self.optimizer.step()
            self.actor_updates += int(actor_has_grad)
            self.critic_updates += 1
            for name, value in (('value', value_loss), ('surrogate', surrogate),
                                ('bc_reference', bc_loss), ('kl', kl_mean)):
                totals[name] += float(value.detach())
            batches += 1
        self.completed_iterations += 1
        self.storage.clear()
        return {**{k: v/batches for k, v in totals.items()}, 'critic_warmup': float(warmup),
                'actor_updates': self.actor_updates, 'critic_updates': self.critic_updates}

    def save(self):
        return {**super().save(), 'bc_finetune': dict(recipe=self.recipe,
            reference_state_dict=self.reference.state_dict(), reference_ready=self.reference_ready,
            completed_iterations=self.completed_iterations, actor_updates=self.actor_updates,
            critic_updates=self.critic_updates)}

    def load(self, loaded_dict, load_cfg=None, strict=True):
        saved = loaded_dict.get('bc_finetune')
        if not saved or not saved['reference_ready']:
            raise ValueError('Missing or incompatible BC refinement state')
        self.recipe = checked_recipe(saved['recipe'], type(self).recipe)
        std = loaded_dict['actor_state_dict']['distribution.std_param']
        if not torch.equal(std.cpu(), torch.tensor(self.recipe['action_std'], dtype=std.dtype)):
            raise ValueError('BC refinement requires the fixed small action noise')
        restored = super().load(loaded_dict, load_cfg, strict)
        self.reference.load_state_dict(saved['reference_state_dict'], strict=True)
        self.reference_ready = saved['reference_ready']
        for name in ('completed_iterations', 'actor_updates', 'critic_updates'):
            setattr(self, name, saved[name])
        return restored


class ProtectedPPOV2(ProtectedPPO):
    recipe = RECIPE_V2


class ProtectedPPOIsaacStandup(ProtectedPPO):
    recipe = RECIPE_ISAAC


class ProtectedPPOIsaacStandupV2(ProtectedPPO):
    recipe = RECIPE_ISAAC_V2


class ProtectedPPOIsaacStandupV3(ProtectedPPO):
    recipe = RECIPE_ISAAC_V3
