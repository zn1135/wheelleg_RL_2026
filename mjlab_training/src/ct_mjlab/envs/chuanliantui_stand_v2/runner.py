"""Persist the independent Isaac curriculum without accepting old task checkpoints."""

import math
import torch
from mjlab.utils.torch import configure_torch_backends

from .observations import OBSERVATION_CONTRACT
from ..chuanliantui.runner import DirectMotorOnPolicyRunner
from .mdp import (CURRICULUM_VERSION, LEGACY_RECOVERY_CONTRACT, RECOVERY_CONTRACT,
                  ZERO_COMMAND_CONTRACT, TRACKING_COMMAND_CONTRACT, SPEED_COMMAND_CONTRACT, YAW_COMMAND_CONTRACT,
                  POSITION_COMMAND_CONTRACT)
from .rewards import RECIPE_VERSION, TRACKING_RECIPE_VERSION, ACTIVE_TRACKING_RECIPE_VERSION, TIGHT_TRACKING_RECIPE_VERSION


EXPERIMENT_PROFILE_CONTRACT = "chuanliantui-stand-v2-isaac32-history-upright-first-v4"
TRACKING_PROFILE_CONTRACT = "chuanliantui-stand-v2-isaac32-post-stand-tracking-v5"
SPEED_PROFILE_CONTRACT = "chuanliantui-stand-v2-isaac32-speed-slew-v6"
YAW_PROFILE_CONTRACT = "chuanliantui-stand-v2-isaac32-yaw-heading-hold-v7"
POSITION_PROFILE_CONTRACT = "chuanliantui-stand-v2-isaac32-yaw-position-hold-v8"


def task_contracts(tracking, speed=False, active_weight=False, tight=False, *, kernel_denominator=.02, tight_weight=.5,
                   yaw=False, yaw_rate_limit=.5, yaw_acceleration=1., heading_gain=1.5,
                   position=False, position_gain=1., position_velocity_limit=.1, position_tolerance=.02,
                   position_lock_speed=.05, position_lock_duration=.5):
    if tight and not active_weight:
        raise ValueError("tracking_tight_term requires tracking_weight_while_active")
    if yaw and not (tracking and speed):
        raise ValueError("yaw_control requires post_stand_tracking and speed_curriculum")
    if position and not yaw:
        raise ValueError("position_hold requires yaw_control")
    contracts = dict(experiment_profile_contract=POSITION_PROFILE_CONTRACT if position else YAW_PROFILE_CONTRACT if yaw else SPEED_PROFILE_CONTRACT if speed else TRACKING_PROFILE_CONTRACT if tracking else EXPERIMENT_PROFILE_CONTRACT,
        recipe_version=TIGHT_TRACKING_RECIPE_VERSION if tight else ACTIVE_TRACKING_RECIPE_VERSION if active_weight else TRACKING_RECIPE_VERSION if tracking else RECIPE_VERSION,
        command_contract=POSITION_COMMAND_CONTRACT if position else YAW_COMMAND_CONTRACT if yaw else SPEED_COMMAND_CONTRACT if speed else TRACKING_COMMAND_CONTRACT if tracking else ZERO_COMMAND_CONTRACT)
    if tight:
        contracts["tracking_tight_config"] = dict(kernel_denominator=kernel_denominator, weight=tight_weight)
    if yaw:
        if any(not math.isfinite(value) or value <= 0. for value in (yaw_rate_limit, yaw_acceleration, heading_gain)) or yaw_rate_limit < .1:
            raise ValueError('Yaw parameters must be finite and positive, with yaw_rate_limit >= 0.1')
        contracts["yaw_control_config"] = dict(yaw_rate_limit=yaw_rate_limit,
            yaw_acceleration=yaw_acceleration, heading_gain=heading_gain)
    if position:
        if any(not math.isfinite(value) or value <= 0. for value in
               (position_gain, position_velocity_limit, position_lock_speed, position_lock_duration)):
            raise ValueError('Position hold gain, limit, lock speed and duration must be finite and positive')
        if not math.isfinite(position_tolerance) or position_tolerance < 0.:
            raise ValueError('Position tolerance must be finite and nonnegative')
        contracts['position_hold_config'] = dict(position_gain=position_gain,
            position_velocity_limit=position_velocity_limit, position_tolerance=position_tolerance,
            position_lock_speed=position_lock_speed, position_lock_duration=position_lock_duration)
    return contracts


class StandV2OnPolicyRunner(DirectMotorOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        configure_torch_backends(allow_tf32=False)
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        return super().learn(num_learning_iterations, init_at_random_ep_len=False)

    def save(self, path, infos=None):
        env = self.env.unwrapped
        state = env.command_manager.get_term("standup")
        super().save(path, {**(infos or {}), "observation_contract": OBSERVATION_CONTRACT,
            **task_contracts(state.cfg.post_stand_tracking, state.cfg.speed_curriculum,
                             state.cfg.tracking_weight_while_active, state.cfg.tracking_tight_term,
                             kernel_denominator=state.cfg.tracking_tight_kernel_denominator,
                             tight_weight=state.cfg.tracking_tight_weight,
                             yaw=state.cfg.yaw_control,
                             **(state.yaw_config() if state.cfg.yaw_control else {}),
                             position=state.cfg.position_hold,
                             **(state.position_config() if state.cfg.position_hold else {})), "curriculum_version": CURRICULUM_VERSION,
            "recovery_contract": RECOVERY_CONTRACT,
            "standup_curriculum": state.checkpoint_state(),
            "physics_dt_s": env.physics_dt, "policy_dt_s": env.step_dt})

    def load(self, path, load_cfg=None, strict=True, map_location=None, *, allow_recovery_change=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        infos = checkpoint.get("infos") or {}
        recovery = infos.get("recovery_contract", LEGACY_RECOVERY_CONTRACT)
        if recovery not in (RECOVERY_CONTRACT, LEGACY_RECOVERY_CONTRACT):
            raise ValueError(f"Incompatible stand_v2 recovery_contract: {recovery!r}")
        if recovery != RECOVERY_CONTRACT and not allow_recovery_change:
            raise ValueError(f"stand_v2 NORMAL rule changed from {recovery} to {RECOVERY_CONTRACT}; "
                             "legacy weights require an explicit diagnostic comparison, not ordinary replay/resume")
        # 旧checkpoint缺少command_contract时只代表原零速配方，不静默改成跟踪。
        infos.setdefault("command_contract", ZERO_COMMAND_CONTRACT)
        state = self.env.unwrapped.command_manager.get_term("standup")
        tracking = state.cfg.post_stand_tracking
        for key, value in {"observation_contract": OBSERVATION_CONTRACT,
            **task_contracts(tracking, state.cfg.speed_curriculum, state.cfg.tracking_weight_while_active,
                state.cfg.tracking_tight_term, kernel_denominator=state.cfg.tracking_tight_kernel_denominator,
                tight_weight=state.cfg.tracking_tight_weight,
                yaw=state.cfg.yaw_control,
                **(state.yaw_config() if state.cfg.yaw_control else {}),
                position=state.cfg.position_hold,
                **(state.position_config() if state.cfg.position_hold else {})), "curriculum_version": CURRICULUM_VERSION,
            "physics_dt_s": .002, "policy_dt_s": .01}.items():
            if infos.get(key) != value:
                raise ValueError(f"Incompatible stand_v2 {key}: {infos.get(key)!r}, expected {value!r}")
        curriculum = infos["standup_curriculum"]
        if state.cfg.speed_curriculum and curriculum["speed_curriculum"]["config"] != state.speed_config():
            raise ValueError("Speed curriculum configuration differs from checkpoint")
        if state.cfg.yaw_control and curriculum.get('yaw_control') != state.yaw_config():
            raise ValueError('Yaw control curriculum configuration differs from checkpoint')
        if not state.cfg.yaw_control and 'yaw_control' in curriculum:
            raise ValueError('Yaw control checkpoint requires yaw control')
        if state.cfg.position_hold and curriculum.get('position_hold') != state.position_config():
            raise ValueError('Position hold curriculum configuration differs from checkpoint')
        if not state.cfg.position_hold and ('position_hold' in curriculum or 'position_hold_config' in infos):
            raise ValueError('Position hold checkpoint requires position hold')
        if recovery != RECOVERY_CONTRACT:
            print(f"[INFO] Loading {recovery} weights with current {RECOVERY_CONTRACT} mode rule", flush=True)
        del checkpoint
        restored = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        env = self.env.unwrapped
        env.command_manager.get_term("standup").load_checkpoint_state(curriculum)
        # The very first policy observation must contain the restored height command.
        env.observation_manager.reset()
        # Reset buffers are initialized but empty: explicitly backfill all five restored frames.
        env.obs_buf = env.observation_manager.compute(update_history=True)
        if load_cfg is None or load_cfg.get("iteration", False):
            self.current_learning_iteration += 1
        return restored
