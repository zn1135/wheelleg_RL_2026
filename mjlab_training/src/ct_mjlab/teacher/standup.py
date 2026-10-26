"""H7 normal EXTEND→REAR→RETRACT→SWING→DONE path; no inverted recovery."""
import math
import torch
from .mechanics import DT, PD, Ramp, SUPPORT_FORCE, wrap

IDLE, RETRACT, SWING, DONE, FAILED, REAR, EXTEND = range(7)
PHASE_NAMES = ('idle', 'retract', 'swing', 'balance', 'failed', 'rear', 'extend')
FAULT_NAMES = ('none', 'invalid_geometry', 'inverted', 'timeout', 'pitch_limit', 'nonfinite')


class Standup:
    def __init__(self, n, device, *, dt=DT):
        self.dt = dt
        self.phase = torch.zeros(n, device=device, dtype=torch.long)
        self.fault = torch.zeros_like(self.phase)
        self.retry = torch.zeros_like(self.phase)
        for name in ('elapsed', 'stable', 'trigger', 'retry_time', 'support'):
            setattr(self, name, torch.zeros(n, device=device))
        for name in ('length_cmd', 'angle_cmd', 'rear_position', 'rear_last', 'rear_goal'):
            setattr(self, name, torch.zeros(n, 2, device=device))
        self.path_ready = torch.zeros(n, device=device, dtype=torch.bool)
        self.length_pd = PD((n, 2), device, 2300., 25000., 5000., dt=dt)
        self.angle_pd = PD((n, 2), device, 45., 15., 10., dt=dt)
        self.speed_pd = PD((n, 2), device, 5., 5., 40., dt=dt)
        self.ramp = Ramp((n, 2), device, dt=dt)

    def reset(self, ids, *, standing=False):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0
            elif isinstance(value, (PD, Ramp)):
                value.reset(ids)
        if standing:
            self.phase[ids] = DONE

    def enter(self, mask, phase):
        self.phase[mask] = phase
        self.elapsed[mask] = 0
        self.stable[mask] = 0
        self.trigger[mask] = 0

    def fail(self, mask, reason):
        self.retry_time[mask & (self.phase != FAILED)] = 0
        self.phase[mask] = FAILED
        self.fault[mask] = reason

    def clear_pid(self, mask):
        self.length_pd.reset(mask)
        self.angle_pd.reset(mask)
        self.speed_pd.reset(mask)

    def step(self, length, theta, rate, pitch, upright, valid):
        previous = self.phase.clone()
        self.fail(~valid, 1)
        self.fail(upright <= 0, 2)  # 倒置锁停，只有显式复位才重新投入。
        retryable = (self.phase == FAILED) & ((self.fault == 3) | (self.fault == 4)) & (self.retry < 3)
        self.retry_time.copy_(torch.where(retryable, self.retry_time + self.dt, 0.))
        restart = retryable & (self.retry_time >= .1)
        old_retry = self.retry.clone()
        self.reset(restart)
        self.retry[restart] = old_retry[restart] + 1

        need = (theta.abs() > math.radians(50)).any(-1) | (pitch.abs() > math.radians(50))
        balance = self.phase == DONE
        self.trigger.copy_(torch.where(balance & need, self.trigger + self.dt, 0.))
        self.retry[balance & ~need] = 0
        restart = balance & (self.trigger >= .2)
        old_retry = self.retry.clone()
        self.reset(restart)
        self.retry[restart] = old_retry[restart]
        idle = self.phase == IDLE
        self.length_cmd[idle] = 0  # H7首次EXTEND的目标由后面的分支设置。
        self.clear_pid(idle)
        extended = (length >= .24).all(-1)
        rear_ready = (wrap(theta + 1.42).abs() <= .1).all(-1)
        retracted = (length <= .19).all(-1)
        self.enter(idle & ~extended, EXTEND)
        self.enter(idle & extended & ~rear_ready, REAR)
        self.enter(idle & extended & rear_ready & retracted, SWING)
        self.enter(idle & extended & rear_ready & ~retracted, RETRACT)

        rear = (self.phase == REAR) & self.path_ready
        self.rear_position[rear] += wrap(theta[rear] - self.rear_last[rear])
        self.rear_last[rear] = theta[rear]
        preparing = (self.phase != DONE) & (self.phase != FAILED)
        self.fail(preparing & (pitch.abs() > 1.), 4)
        preparing &= self.phase != FAILED
        self.elapsed += preparing * self.dt
        timeout = torch.where(self.phase == EXTEND, 4., torch.where(self.phase == REAR, 1., 1.5))
        expired = preparing & (self.elapsed > timeout)
        extend_expired = expired & (self.phase == EXTEND)
        self.fail(expired & ~extend_expired, 3)
        self.clear_pid(extend_expired)
        self.path_ready[extend_expired] = False
        self.support[extend_expired] = 0
        self.enter(extend_expired, REAR)

        ready = ((self.phase == EXTEND) & extended |
                 (self.phase == RETRACT) & retracted |
                 (self.phase == REAR) & self.path_ready &
                 ((self.rear_goal - self.rear_position).abs() <= .1).all(-1) &
                 ((self.rear_goal - self.angle_cmd).abs() <= .1).all(-1) |
                 (self.phase == SWING) & (wrap(theta).abs() <= .4).all(-1) &
                 (wrap(self.angle_cmd).abs() <= .4).all(-1))
        ready &= ~extend_expired
        to_swing = ready & (self.phase == RETRACT)
        self.enter(to_swing, SWING)
        self.stable.copy_(torch.where(ready & ~to_swing, self.stable + self.dt, 0.))
        advance = self.stable >= .05
        old_phase = self.phase.clone()
        transition = advance & ((old_phase == EXTEND) | (old_phase == REAR))
        self.clear_pid(transition)
        self.enter(transition & (old_phase == EXTEND) & ~rear_ready, REAR)
        can_retract = transition & ((old_phase == REAR) | rear_ready)
        self.enter(can_retract & retracted, SWING)
        self.enter(can_retract & ~retracted, RETRACT)
        self.enter(advance & (old_phase == SWING), DONE)

        preparing = (self.phase != DONE) & (self.phase != FAILED)
        unprimed = preparing & ~self.length_pd.ready.all(-1)
        self.length_cmd[unprimed & (self.phase == REAR)] = length[unprimed & (self.phase == REAR)]
        self.angle_cmd[unprimed] = wrap(theta[unprimed])
        new_path = (self.phase == REAR) & ~self.path_ready
        start = torch.remainder(wrap(theta - pitch[:, None]), 2 * math.pi)
        goal = torch.remainder(wrap(-1.42 - pitch[:, None]), 2 * math.pi)
        self.rear_position[new_path] = theta[new_path]
        self.rear_last[new_path] = theta[new_path]
        self.rear_goal[new_path] = (theta + goal - start)[new_path]
        self.angle_cmd[new_path] = theta[new_path]
        self.path_ready[new_path] = True
        rear = self.phase == REAR
        swing = self.phase == SWING
        self.support[swing] = (self.support[swing] + self.dt / .4).clamp(0, 1)
        self.length_cmd[self.phase == EXTEND] = .25
        self.length_cmd[(self.phase == RETRACT) | swing] = .15
        self.angle_cmd[rear] += (self.rear_goal[rear] - self.angle_cmd[rear]).clamp(-6.7 * self.dt, 6.7 * self.dt)
        self.angle_cmd[swing] = 0

        # 非准备状态的计算历史不消费；下一次进入准备会重新播种。
        angle = torch.where(rear[:, None], self.rear_position, wrap(theta))
        target = self.ramp.step(length, self.length_cmd, .75)
        force = self.length_pd.step(length, target)
        error = torch.where(rear[:, None], self.angle_cmd - angle, wrap(self.angle_cmd - angle))
        speed_target = self.angle_pd.step(torch.zeros_like(error), error)
        torque = self.speed_pd.step(rate, speed_target)
        torque[self.phase == RETRACT] = 0
        self.angle_pd.reset(self.phase == RETRACT)
        self.speed_pd.reset(self.phase == RETRACT)
        force += self.support[:, None] * (1 - wrap(angle).abs() / .6).clamp(0, 1) * SUPPORT_FORCE
        self.clear_pid(~preparing)
        self.ramp.reset(~preparing)
        return force, torque, preparing, (previous != DONE) & (self.phase == DONE)
