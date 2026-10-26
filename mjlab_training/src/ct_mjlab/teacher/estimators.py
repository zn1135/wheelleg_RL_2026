"""H7 slip.c velocity fusion and air_detection.c support estimator."""
import torch
from .mechanics import DT, wrap


class VelocityFilter:
    def __init__(self, n, device, *, dt=DT):
        self.dt = dt
        self.state = torch.zeros(n, 2, device=device)
        self.cov = torch.zeros(n, 2, 2, device=device)
        self.ready = torch.zeros(n, dtype=torch.bool, device=device)
        self.eye = torch.eye(2, device=device)

    def reset(self, ids):
        self.state[ids] = 0
        self.cov[ids] = 0
        self.ready[ids] = False

    def correct(self, axis, innovation, noise, enabled):
        gain = self.cov[:, :, axis] / (self.cov[:, axis, axis] + noise)[:, None]
        gain = gain * enabled[:, None]
        self.state.add_(gain * innovation[:, None])
        transform = self.eye[None].expand_as(self.cov).clone()
        transform[:, :, axis] -= gain
        self.cov.copy_(transform @ self.cov @ transform.transpose(-1, -2) +
                       noise[:, None, None] * gain[:, :, None] * gain[:, None, :])
        cross = (self.cov[:, 0, 1] + self.cov[:, 1, 0]) / 2
        self.cov[:, 0, 1] = cross
        self.cov[:, 1, 0] = cross

    def step(self, wheel_velocity, acceleration, contact):
        old = self.ready.clone()
        self.state[~old, 0] = wheel_velocity[~old]
        self.state[~old, 1] = acceleration[~old]
        self.cov[~old] = self.cov.new_tensor([[.1, 0], [0, .25]])
        initial_state, initial_cov = self.state.clone(), self.cov.clone()
        transition = self.cov.new_tensor([[1., self.dt], [0., 1.]])
        self.state[:, 0] += self.dt * self.state[:, 1]
        self.cov.copy_(transition @ self.cov @ transition.T +
                       self.cov.new_tensor([[.007, 0], [0, .01]])*(self.dt/DT))
        self.correct(1, acceleration - self.state[:, 1],
                     torch.full_like(acceleration, .25), old)
        innovation = wheel_velocity - self.state[:, 0]
        limit = (3 * (self.cov[:, 0, 0] + .01).sqrt()).clamp_min(.4)
        noise = .01 * (innovation / limit).square().clamp_min(1.)
        self.correct(0, innovation, noise, old & contact)
        self.state.copy_(torch.where(old[:, None], self.state, initial_state))
        self.cov.copy_(torch.where(old[:, None, None], self.cov, initial_cov))
        self.ready.fill_(True)
        return self.state[:, 0]


class AirDetector:
    def __init__(self, n, device, *, dt=DT):
        self.dt = dt
        for key in ('armed', 'flight', 'ready'):
            setattr(self, key, torch.zeros(n, dtype=torch.bool, device=device))
        self.elapsed = torch.zeros(n, device=device)
        for key in ('force', 'height_v', 'height_a', 'peak', 'compression_time'):
            setattr(self, key, torch.zeros(n, 2, device=device))

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0

    def step(self, length, theta, rates, pitch, pitch_rate, jac, motor_torque,
             spring, gravity, vertical_acceleration, permit):
        self.reset(~permit)
        old_flight, old_armed = self.flight.clone(), self.armed.clone()
        # 反馈电机力矩反解；奇异输入由控制器门控，不当作有效支撑。
        a, b = jac[..., 0, 0], jac[..., 1, 0]
        c, d = jac[..., 0, 1], jac[..., 1, 1]
        det = a*d - b*c
        safe = torch.where(det.abs() < 1e-6, torch.ones_like(det), det)
        force = (d * motor_torque[..., 0] - b * motor_torque[..., 1]) / safe
        torque = (a * motor_torque[..., 1] - c * motor_torque[..., 0]) / safe
        angle = wrap(theta - pitch[:, None])
        omega = rates[..., 1] - pitch_rate[:, None]
        pressure = (force + spring) * angle.cos() - (torque - gravity) * angle.sin() / length.clamp_min(.001)
        hv = rates[..., 0] * angle.cos() - length * omega * angle.sin()
        ha = torch.where(self.ready[:, None], self.height_a + self.dt / (.01 + self.dt) *
                         ((hv - self.height_v) / self.dt - self.height_a), 0.)
        raw = pressure + .65 * (9.81 + vertical_acceleration[:, None] - ha)
        filtered = torch.where(self.ready[:, None], self.force + self.dt / (.01 + self.dt) * (raw - self.force), raw)
        valid = permit & torch.isfinite(filtered).all(-1) & (det.abs() >= 1e-6).all(-1)
        self.force.copy_(torch.where(valid[:, None], filtered, self.force))
        self.height_v.copy_(hv)
        self.height_a.copy_(ha)
        mean = self.force.mean(-1)
        condition = torch.where(old_armed, mean < 10., mean > 50.)
        self.elapsed.copy_(torch.where(valid & ~old_flight & condition, self.elapsed + self.dt, 0.))
        reached = self.elapsed + 1e-6 >= .05
        self.armed |= reached & ~old_armed
        self.flight |= reached & old_armed
        self.elapsed[reached] = 0.
        taking_off = ~old_flight & self.flight
        self.peak.copy_(torch.where(taking_off[:, None], length, torch.maximum(self.peak, length)))
        compressed = (self.peak - length) >= .02
        self.compression_time.copy_(torch.where(valid[:, None] & old_flight[:, None] & compressed,
                                                self.compression_time + self.dt, 0.))
        landed = old_flight & (self.compression_time.max(-1).values + 1e-6 >= .10)
        self.flight[landed] = False
        self.ready.copy_(valid)
        self.reset(~permit)
        return self.flight, landed
