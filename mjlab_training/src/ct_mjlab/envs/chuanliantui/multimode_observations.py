"""Huananhu 39/82 fixed-scale observations with live control-mode signals."""

from dataclasses import replace

import torch

from .huananhu_observations import HuananhuObservationState
from .v14_observations import V14ObservationCfg
from ...wheel_odometry import WHEEL_OBSERVATION_CONTRACT, WHEEL_ODOMETRY, wheel_displacement


OBSERVATION_CONTRACT = "chuanliantui-huananhu-multimode-39-82-v1"
class MultimodeObservationState(HuananhuObservationState):
    def __init__(self, env, cfg):
        self.wheel_angle_handover_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        super().__init__(env, cfg)
        self.wheel_origin = self.wheel_angles().clone()

    def wheel_angles(self):
        return self.env.sim.data.qpos[:, self._motor_q_adr[4:6]]

    def raw_sensors(self):
        sensors = super().raw_sensors()
        if self.cfg.wheel_angle_displacement:
            if not hasattr(self, 'wheel_angle_origin'):
                self.wheel_angle_origin = self.wheel_angles().clone()
            # 原生轮轴方向、连续角差；放回位置组，共用原传感器延迟与噪声。
            sensors[:, 10:12] = self.wheel_angles() - self.wheel_angle_origin
        return sensors

    def reset(self, env_ids):
        self.wheel_angle_handover_latched[env_ids] = False
        if hasattr(self, 'wheel_angle_origin'):
            # 先改起点，再填充延迟历史，避免带入上一回合的轮位移。
            self.wheel_angle_origin[env_ids] = self.wheel_angles()[env_ids]
        super().reset(env_ids)
        if hasattr(self, 'wheel_origin'):
            self.wheel_origin[env_ids] = self.wheel_angles()[env_ids]

    def synchronize_wheel_handover(self):
        if not self.cfg.wheel_angle_handover_reset:
            return
        ready = self.command.ground_start & self.command.recovered & ~self.wheel_angle_handover_latched
        if ready.any():
            # 起立完成只锁一次；仅清两轮位置历史，不改其他观测或电机状态。
            self.wheel_angle_origin[ready] = self.wheel_angles()[ready]
            self.delay.history[:, ready, 10:12] = 0.
            self.wheel_angle_handover_latched[ready] = True

    def sample_substep(self):
        self.synchronize_wheel_handover()
        super().sample_substep()

    def wheel_travel(self):
        return wheel_displacement(self.wheel_angles(), self.wheel_origin)

    @property
    def command(self):
        return self.env.command_manager.get_term("standup")

    def mode(self):
        return self.command.mode()

    def actor(self):
        self.synchronize_wheel_handover()
        return torch.cat((super().actor(), self.command.extra_flags()), dim=-1)

    def critic(self):
        self.synchronize_wheel_handover()
        old = super().critic().clone()
        # The existing privileged height slot is terrain-relative in this task.
        old[:, 31] = self.command.height_above_ground.clamp(-10., 10.) * 5.
        return torch.cat((old, self.command.extra_flags()), dim=-1)


def get_state(env, cfg: V14ObservationCfg | None = None) -> MultimodeObservationState:
    if not hasattr(env, "_multimode_observations"):
        env._multimode_observations = MultimodeObservationState(env, cfg or V14ObservationCfg())
    return env._multimode_observations


def actor_observation(env, cfg: V14ObservationCfg | None = None):
    return get_state(env, cfg).actor()


def critic_observation(env, cfg: V14ObservationCfg | None = None):
    return get_state(env, cfg).critic()


def wheel_actor_observation(env, cfg: V14ObservationCfg | None = None):
    state = get_state(env, cfg)
    return torch.cat((state.actor(), state.wheel_travel()), dim=-1)


def wheel_critic_observation(env, cfg: V14ObservationCfg | None = None):
    state = get_state(env, cfg)
    return torch.cat((state.critic(), state.wheel_travel()), dim=-1)


def configure_wheel_odometry(cfg):
    # BC新观测单独版本化；旧39维PPO配置及旧模型回放保持兼容。
    cfg.observations['actor'].terms['proprioception'].func = wheel_actor_observation
    cfg.observations['critic'].terms['proprioception'].func = wheel_critic_observation


def reference_actor_observation(env, cfg=None, takeover_s=None):
    from ...teacher.reference import get_reference
    reference=get_reference(env,takeover_s=takeover_s).update()
    return torch.cat((wheel_actor_observation(env,cfg),reference.error),dim=-1)


def reference_critic_observation(env, cfg=None, takeover_s=None):
    from ...teacher.reference import get_reference
    reference=get_reference(env,takeover_s=takeover_s).update()
    return torch.cat((wheel_critic_observation(env,cfg),reference.error),dim=-1)


def configure_reference_observations(cfg, *, takeover_s=None):
    for name,func in (('actor',reference_actor_observation),('critic',reference_critic_observation)):
        term=cfg.observations[name].terms['proprioception']
        term.func=func;term.params={**term.params,'takeover_s':takeover_s}


def configure_wheel_angle_observations(cfg, *, handover_reset=False):
    actor=cfg.observations['actor'].terms['proprioception']
    observation_cfg=replace(actor.params.get('cfg',V14ObservationCfg()),wheel_angle_displacement=True,
                            wheel_angle_handover_reset=handover_reset)
    actor.func=actor_observation
    actor.params={'cfg':observation_cfg}
    critic=cfg.observations['critic'].terms['proprioception']
    critic.func=critic_observation
    critic.params={'cfg':observation_cfg}
    # 复位或子步采样可能先于首次观测，必须使用同一个新接口配置。
    for term in (cfg.events['reset_observations'],cfg.metrics['observation_delay_ms']):
        term.params={**term.params,'cfg':observation_cfg}


def reset_observations(env, env_ids, cfg=None):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    get_state(env,cfg).reset(env_ids)
    if hasattr(env,'_bc_reference_observer'):
        env._bc_reference_observer.reset(env_ids)


def sample_observations(env, cfg=None):
    state = get_state(env,cfg)
    state.sample_substep()
    if hasattr(env,'_bc_reference_observer'):
        env._bc_reference_observer.sample_substep()
    return state.delay.lags.float().mean(dim=-1) * env.physics_dt * 1000.
