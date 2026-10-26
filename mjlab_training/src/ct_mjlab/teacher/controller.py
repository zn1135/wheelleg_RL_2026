"""H7 big-machine LQR+VMC controller in calibrated H7 coordinates."""
import torch
from .gains import gain_matrix
from .mechanics import (DT, H7_LEG_LIMIT, H7_WHEEL_LIMIT, SUPPORT_FORCE, PD, Ramp,
                        force_map, gravity_torque, leg_geometry, spring_force, wrap)
from .estimators import AirDetector, VelocityFilter
from .standup import DONE, FAILED, Standup


class LqrVmcController:
    def __init__(self, n, device, *, velocity_integral=False, dt=DT):
        self.dt = dt
        self.rate_decay = .7**(dt/DT)
        self.velocity_integral = velocity_integral
        self.standup = Standup(n, device, dt=dt)
        self.air = AirDetector(n, device, dt=dt)
        self.velocity = VelocityFilter(n, device, dt=dt)
        self.length_pd = PD((n, 2), device, 2300., 25000., 5000., dt=dt)
        self.roll_pd = PD((n,), device, 500., 100., 5000., dt=dt)
        self.length_ramp = Ramp((n, 2), device, dt=dt)
        self.x = torch.zeros(n, 10, device=device)
        self.target = torch.zeros_like(self.x)
        self.u = torch.zeros(n, 4, device=device)
        self.contributions = torch.zeros(n, 4, 10, device=device)
        self.K = torch.zeros_like(self.contributions)
        self.gain_length = torch.full((n, 2), -1., device=device)
        self.filtered_rates = torch.zeros(n, 2, device=device)
        self.pos = torch.zeros(n, device=device)
        self.pos_armed = torch.zeros(n, device=device, dtype=torch.bool)
        self.position_reference_latched = torch.zeros_like(self.pos_armed)
        self.position_reference_event = torch.zeros_like(self.pos_armed)
        self.position_integrating = torch.zeros_like(self.pos_armed)
        self.yaw_target = torch.zeros(n, device=device)
        self.ready = torch.zeros_like(self.pos_armed)
        self.length = torch.zeros(n, 2, device=device)
        self.theta = torch.zeros_like(self.length)
        self.force = torch.zeros_like(self.length)
        self.swing_torque = torch.zeros_like(self.length)
        self.spring = torch.zeros_like(self.length)
        self.gravity = torch.zeros_like(self.length)
        self.raw_torque = torch.zeros(n, 6, device=device)
        self.command_length = torch.full((n,), .15, device=device)
        self.last_length_command = self.command_length.clone()

    def reset(self, ids, *, standing=False):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0
            elif isinstance(value, Standup):
                value.reset(ids, standing=standing)
            elif isinstance(value, (AirDetector, VelocityFilter, PD, Ramp)):
                value.reset(ids)
        self.gain_length[ids] = -1
        self.command_length[ids] = .15
        self.last_length_command[ids] = .15

    def step(self, q, dq, wheel_speed, pitch, roll, yaw, gyro_pitch, gyro_yaw,
             upright, acceleration, vertical_acceleration, feedback_torque, command, *, position_reference=None):
        """q,dq=(N,2,2), torque feedback same order, command=(vx,yaw_rate,leg_length)."""
        length, theta, rates, jac, geometry_valid = leg_geometry(q, dq)
        valid = geometry_valid.all(-1)
        inputs = torch.stack((pitch, roll, yaw, gyro_pitch, gyro_yaw, upright,
                              acceleration, vertical_acceleration), -1)
        measurements_finite = (torch.isfinite(inputs).all(-1) & torch.isfinite(command).all(-1) &
                               torch.isfinite(q).all(dim=(1, 2)) & torch.isfinite(dq).all(dim=(1, 2)) &
                               torch.isfinite(wheel_speed).all(-1) &
                               torch.isfinite(feedback_torque).all(dim=(1, 2)))
        valid &= measurements_finite
        self.length.copy_(length)
        self.theta.copy_(theta)
        self.filtered_rates.mul_(self.rate_decay).add_((1-self.rate_decay) * torch.stack((gyro_pitch, gyro_yaw), -1))
        omg = self.filtered_rates[:, 0]
        self.x[:, 2] = yaw
        self.x[:, 3] = self.filtered_rates[:, 1]
        self.x[:, 4] = theta[:, 0] - pitch
        self.x[:, 5] = rates[:, 0, 1] - omg
        self.x[:, 6] = theta[:, 1] - pitch
        self.x[:, 7] = rates[:, 1, 1] - omg
        self.x[:, 8] = pitch
        self.x[:, 9] = omg
        self.spring.copy_(spring_force(length))
        self.gravity.copy_(gravity_torque(length, theta - pitch[:, None]))
        old_air = self.air.flight.clone()
        air, landed = self.air.step(length, theta, rates, pitch, gyro_pitch, jac,
                                   feedback_torque, self.spring, self.gravity, vertical_acceleration,
                                   valid & self.ready & (self.standup.phase == DONE))
        changed_air = old_air != air
        self.yaw_target.copy_(torch.where(~self.ready | changed_air | (command[:, 1] != 0), yaw, self.yaw_target))
        changed_length = command[:, 2] != self.last_length_command
        self.command_length.copy_(torch.where(~self.ready | changed_length, command[:, 2], self.command_length))
        self.command_length[landed] = .20  # H7落地后重建腿长目标。
        self.last_length_command.copy_(command[:, 2])
        if position_reference is None:
            self.pos[changed_air] = 0
            self.pos_armed[changed_air] = False
        elif not self.velocity_integral:
            raise ValueError('Shared wheel reference requires velocity-integral control')
        self.position_reference_event.zero_()
        if position_reference is not None:
            self.position_reference_event.copy_(position_reference & ~self.position_reference_latched)
            self.position_reference_latched |= position_reference
        wheel_ground = wheel_speed - rates[..., 1] - omg[:, None]
        world_theta = theta - pitch[:, None]
        world_rate = rates[..., 1] - omg[:, None]
        raw_v = (wheel_ground * .0525 - length * world_rate * world_theta.cos() -
                 rates[..., 0] * world_theta.sin()).mean(-1)
        self.x[:, 1] = self.velocity.step(raw_v, acceleration, ~air)
        self.position_integrating.zero_()
        if self.velocity_integral:
            # 仿真跟踪模式：速度误差累积到原LQR位置槽；准备和腾空时不积分。
            active = ~air & valid & (self.standup.phase == DONE)
            if position_reference is not None:
                active &= position_reference
            self.position_integrating.copy_(active)
            self.pos += active * (self.x[:,1]-command[:,0]) * self.dt
            self.pos.clamp_(-3.5,3.5)
        else:
            reset_position = ~air & ((command[:, 0] != 0) | (self.pos.abs() >= 3.5))
            self.pos[reset_position] = 0
            self.pos_armed[reset_position] = False
            legs_ready = (wrap(world_theta).abs() <= .1).all(-1)
            arm = ~air & ~reset_position & legs_ready & ~self.pos_armed & (self.x[:, 1].abs() < .1)
            self.pos[arm] = 0
            self.pos_armed |= arm
            self.pos += (~air & ~reset_position & legs_ready & self.pos_armed) * self.x[:, 1] * self.dt
        prepare_f, prepare_t, preparing, handed_over = self.standup.step(
            length, theta, rates[..., 1], pitch, upright, valid)
        # 对应H7 LQR_Enable_Latch：正常起立交接重新锁定停车位置和朝向。
        # 新配对模式只由共享事件锁位置；原教师默认仍按H7内部交接锁定。
        position_reset = handed_over if position_reference is None else self.position_reference_event
        self.pos[position_reset] = 0
        self.pos_armed[position_reset] = False
        self.yaw_target[handed_over] = yaw[handed_over]
        self.x[:, 0] = self.pos
        self.target.zero_()
        # 仿真零速命令保持本次停车原点；H7的0.4m台架目标不作为移动命令。
        self.target[:, 0] = 0.
        self.target[:, 1] = command[:, 0]
        self.target[:, 2] = self.yaw_target
        self.target[:, 3] = command[:, 1]
        gain_length = length.clamp(.15, .31)
        update_gain = ((gain_length - self.gain_length).abs() > .0005).any(-1)
        k = gain_matrix(gain_length)
        self.K.copy_(torch.where(update_gain[:, None, None], k, self.K))
        self.gain_length.copy_(torch.where(update_gain[:, None], gain_length, self.gain_length))
        error = self.target - self.x
        error[:, 2] = wrap(error[:, 2])
        for col in (4, 6):
            error[:, col] = torch.where(air, wrap(error[:, col]), error[:, col])
        self.contributions.copy_(self.K * error[:, None, :])
        self.contributions[air, :2] = 0
        self.contributions[air, 2:, :4] = 0
        self.contributions[air, 2:, 8:] = 0
        self.u.copy_(self.contributions.sum(-1))
        self.u[:, :2].clamp_(-H7_WHEEL_LIMIT, H7_WHEEL_LIMIT)
        hip_limit = torch.where(air, 10., H7_LEG_LIMIT)[:, None]
        self.u[:, 2:] = torch.clamp(self.u[:, 2:], -hip_limit, hip_limit)
        restart_balance = preparing | handed_over | changed_air
        self.length_pd.reset(restart_balance)
        self.roll_pd.reset(restart_balance)
        self.length_ramp.reset(restart_balance)
        self.roll_pd.last[changed_air & ~air] = -roll[changed_air & ~air]
        self.roll_pd.ready[changed_air & ~air] = True
        target = self.length_ramp.step(length, self.command_length[:, None], .2)
        force = self.length_pd.step(length, target) + SUPPORT_FORCE
        roll_force = self.roll_pd.step(roll, torch.zeros_like(roll), prime=False)
        force[:, 0] += roll_force
        force[:, 1] -= roll_force
        force[air] = 0
        self.length_pd.reset(air | preparing)
        self.roll_pd.reset(air | preparing)
        self.length_ramp.reset(air | preparing)
        self.force.copy_(torch.where(preparing[:, None], prepare_f, force))
        self.swing_torque.copy_(torch.where(preparing[:, None], prepare_t, -self.u[:, 2:]))
        base = force_map(jac, self.force, self.swing_torque)
        # 原顺序：基础力矩→弹簧抵消→完整重力补偿→限幅。
        spring = force_map(jac, -self.spring, torch.zeros_like(length))
        base += torch.where((air & ~preparing)[:, None, None], 0., spring)
        base += force_map(jac, torch.zeros_like(length), self.gravity)
        self.raw_torque[:, :4] = base.flatten(1)
        self.raw_torque[:, 4:] = torch.where(preparing[:, None], 0., self.u[:, :2])
        finite = torch.isfinite(self.raw_torque).all(-1) & valid
        self.standup.fail(~measurements_finite | (~finite & valid), 5)
        output = self.raw_torque.clone()
        output[:, :4].clamp_(-H7_LEG_LIMIT, H7_LEG_LIMIT)
        output[:, 4:].clamp_(-H7_WHEEL_LIMIT, H7_WHEEL_LIMIT)
        output[~finite | (self.standup.phase == FAILED)] = 0
        self.ready.copy_(finite)
        return output
