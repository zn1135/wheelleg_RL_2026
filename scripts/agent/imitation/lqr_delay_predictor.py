"""Nominal-model preview using current state and pending PD commands only."""
from dataclasses import replace

import torch
from mjlab.managers.event_manager import EventTermCfg

from ct_mjlab.envs import task_registry
from ct_mjlab.smp.ppo import SmpCommandCfg, reset_pose
from ct_mjlab.teacher.controller import LqrVmcController
from ct_mjlab.teacher.env import TeacherAction, TeacherActionCfg


def copy_controller_state(target, source):
    for key, value in vars(source).items():
        if isinstance(value, torch.Tensor):
            getattr(target, key).copy_(value)
        elif hasattr(value, '__dict__'):
            copy_controller_state(getattr(target, key), value)


class QueuedPhysicsPredictor:
    def __init__(self, env, action):
        self.live, self.action = env, action
        cfg, train = task_registry.get_cfgs('chuanliantui', play=True)
        cfg.seed = 42
        cfg.commands['standup'] = SmpCommandCfg()
        cfg.events['reset_pose'] = EventTermCfg(func=reset_pose, mode='reset')
        cfg.rewards, cfg.metrics, cfg.terminations = {}, {}, {}
        cfg.actions['pd'] = replace(cfg.actions['pd'], action_delay_s=None,
            randomize_delays=False, action_delay_choices_s=(0., .002, .006, .008, .010))
        self.wrapped, _ = task_registry.make_env('chuanliantui', env_cfg=cfg, train_cfg=train,
                                               num_envs=env.num_envs, device=env.device)
        self.shadow = self.wrapped.unwrapped
        self.shadow.reset(seed=42)
        self.pd = self.shadow.action_manager.get_term('pd')
        self.teacher = TeacherAction(TeacherActionCfg(entity_name='robot', initial='ground',
            velocity_integral=True, ground_start_hold_s=.2), self.shadow)
        self.teacher.controller = LqrVmcController(env.num_envs, env.device, velocity_integral=True, dt=env.step_dt)
        self.calls = 0

    def predict(self, proposal, *, preview_lqr=False):
        real, shadow, source, pd = self.live, self.shadow, self.action, self.pd
        # 只从当前已知状态初始化影子世界，不读未来真实轨迹，不写真实世界。
        for name in ('qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'act', 'time', 'qfrc_applied', 'xfrc_applied'):
            if hasattr(real.sim.data, name):
                getattr(shadow.sim.data, name).copy_(getattr(real.sim.data, name))
        shadow.episode_length_buf.copy_(real.episode_length_buf)
        shadow.command_manager.get_term('standup').ground_start.copy_(
            real.command_manager.get_term('standup').ground_start)
        for name in ('_action_history', 'action_delay_steps', 'wheel_action_delay_steps',
                     'kp_scale', 'kd_scale', 'motor_strength', 'spring_damping', '_spring_force_bias',
                     '_motor_response_torque'):
            getattr(pd, name).copy_(getattr(source, name))
        pd._history_head = source._history_head
        pd.process_actions(proposal)
        future_dq = real.sim.data.qvel[:, source._motor_v_adr].clone()
        before = real.sim.data.qpos.clone() if self.calls == 0 else None
        depth = int(source.wheel_action_delay_steps.max())
        for tick in range(depth):
            pd.apply_actions()
            shadow.scene.write_data_to_sim()
            shadow.sim.step()
            shadow.scene.update(shadow.physics_dt)
            reached = source.wheel_action_delay_steps == tick+1
            future_dq[reached] = shadow.sim.data.qvel[reached][:, pd._motor_v_adr]
        predicted_torque = source.teacher.torques.clone()
        if preview_lqr:
            # 10ms轮延迟恰好一策略周期：预览下一次LQR状态；其他延迟不强行按10ms预览。
            assert depth*shadow.physics_dt == real.step_dt
            shadow.sim.forward()
            copy_controller_state(self.teacher.controller, source.teacher.controller)
            for name in ('previous_velocity', 'velocity_ready', '_control_ticks', 'control_active', 'command'):
                getattr(self.teacher, name).copy_(getattr(source.teacher, name))
            predicted_torque.copy_(self.teacher.compute_torques(dt=real.step_dt))
        if before is not None:
            torch.testing.assert_close(real.sim.data.qpos[:], before, atol=0, rtol=0)
        self.calls += 1
        return future_dq, predicted_torque

    def close(self):
        self.wrapped.close()
