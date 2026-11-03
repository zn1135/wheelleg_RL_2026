"""Batch equivalents of H7 leg_solver, gas_spring and gravity_comp.

Coordinates here are H7 calibrated coordinates, NOT raw encoder readings.
No physical polarity/zero is inferred or changed by this module.
"""
import math
import torch


DT = .001
H7_LEG_LIMIT = 54.
H7_WHEEL_LIMIT = (.30 * 20.) * (15.5 / 19.2)
SUPPORT_FORCE = 20.534 * 9.81 * .5


def wrap(angle):
    result = torch.fmod(angle, 2 * math.pi)
    return torch.where(result > math.pi, result - 2 * math.pi,
                       torch.where(result < -math.pi, result + 2 * math.pi, result))


def leg_geometry(q, dq):
    """q/dq: (..., front/rear); returns L, theta, their rates, J and validity."""
    front, rear = q.unbind(-1)
    a = .21 * torch.stack((front.cos(), front.sin()), -1)
    b = .21 * torch.stack((rear.cos(), rear.sin()), -1)
    ab = b - a
    distance = ab.norm(dim=-1)
    disc = .25**2 - distance.square() / 4
    normal = torch.stack((-ab[..., 1], ab[..., 0]), -1) / distance.clamp_min(1e-6)[..., None]
    point = (a + b) / 2 + disc.clamp_min(0).sqrt()[..., None] * normal
    length = point.norm(dim=-1)
    phi = torch.atan2(point[..., 1], point[..., 0])
    theta = wrap(math.pi / 2 - phi)
    pa, pb = point - a, point - b
    fa = torch.atan2(pa[..., 1], pa[..., 0])
    fb = torch.atan2(pb[..., 1], pb[..., 0])
    alpha = fa.cos() * fb.sin() - fa.sin() * fb.cos()
    safe = torch.where(alpha.abs() < 1e-6, torch.ones_like(alpha), alpha)
    ka, kb = .21 * (fa - front).sin(), .21 * (fb - rear).sin()
    jx = torch.stack((ka * fb.sin(), -kb * fa.sin()), -1) / safe[..., None]
    jy = torch.stack((-ka * fb.cos(), kb * fa.cos()), -1) / safe[..., None]
    jl = phi.cos()[..., None] * jx + phi.sin()[..., None] * jy
    jt = (phi.sin()[..., None] * jx - phi.cos()[..., None] * jy) / length.clamp_min(.001)[..., None]
    jac = torch.stack((jl, jt), -2)
    det = jl[..., 0] * jt[..., 1] - jt[..., 0] * jl[..., 1]
    valid = ((distance > 1e-6) & (disc >= 0) & (length >= .001) &
             (alpha.abs() >= 1e-6) & (det.abs() >= 1e-6) &
             torch.isfinite(q).all(-1) & torch.isfinite(dq).all(-1) &
             torch.isfinite(jac).all(dim=(-1, -2)))
    rates = (jac @ dq[..., None]).squeeze(-1)
    return length, theta, rates, jac, valid


def force_map(jac, force, torque):
    # 虚功：关节力矩 = 腿长/腿角雅可比的转置 × 腿端力/摆矩。
    return (jac.transpose(-1, -2) @ torch.stack((force, torque), -1)[..., None]).squeeze(-1)


def spring_force(length):
    length = length.clamp(.05, .45)
    theta = torch.acos((.1066 - length.square()) / .105)
    span = (.0473512858 - .0206288453 * (theta - .213803).cos()).clamp_min(0).sqrt()
    return 150. * (.0103144227 * (theta - .213803).sin() * length /
                   (.0525 * theta.sin() * span).clamp_min(1e-6))


# machine_config.c gravity_nodes_big: (offset rad, radius m), L=.15:.01:.31.
GRAVITY_NODES = (
    (.14696847451069225, .06629277864141764), (.15233205965915383, .06716142047336403),
    (.15785793169809365, .06810379284592012), (.16353393531619057, .06910968094268703),
    (.16932790703247946, .07018361703417686), (.17523986903670066, .07130926798109767),
    (.18125438063671734, .07249103737704407), (.18735804573062775, .07371456640312009),
    (.1935440737919919, .07498472177717272), (.19981164355462372, .07629346564418214),
    (.20613804355334314, .07763950669601141), (.2125254867068889, .07902583438344704),
    (.21897323717751446, .08044798692322885), (.22546613138118995, .08189613177678175),
    (.23200655335571885, .08337530089900726), (.23859429875837357, .08488368158839482),
    (.2452153170175142, .08641615416112892),
)


def gravity_torque(length, world_angle):
    table = length.new_tensor(GRAVITY_NODES)
    index = ((length.clamp(.15, .31) - .15) / .01).clamp(0, 16)
    lo = index.long().clamp(max=15)
    values = table[lo] + (index - lo)[..., None] * (table[lo + 1] - table[lo])
    return 1.408 * 9.81 * values[..., 1] * (wrap(world_angle) - values[..., 0]).sin()


class PD:
    """H7 POSITION_PID with Ki=0 (all active big-machine controllers)."""
    def __init__(self, shape, device, kp, kd, limit, *, dt=DT):
        self.last = torch.zeros(shape, device=device)
        self.ready = torch.zeros(shape, device=device, dtype=torch.bool)
        # H7的D增益针对1ms差分；换周期时保持对应的连续时间微分系数。
        self.kp, self.kd, self.limit = kp, kd*(DT/dt), limit

    def reset(self, ids):
        self.last[ids] = 0
        self.ready[ids] = False

    def step(self, measured, target, *, angular=False, prime=True):
        error = target - measured
        if angular:
            error = wrap(error)
        delta = torch.where(self.ready | (not prime), error - self.last, 0.)
        # 原H7的D项没有除以dt；增益对应1ms节拍。
        output = (self.kp * error + self.kd * delta).clamp(-self.limit, self.limit)
        self.last.copy_(error)
        self.ready.fill_(True)
        return output


class Ramp:
    def __init__(self, shape, device, *, dt=DT):
        self.dt = dt
        self.value = torch.zeros(shape, device=device)
        self.ready = torch.zeros(shape, device=device, dtype=torch.bool)

    def reset(self, ids):
        self.value[ids] = 0
        self.ready[ids] = False

    def step(self, measured, target, rate):
        self.value.copy_(torch.where(self.ready, self.value, measured))
        self.value.add_((target - self.value).clamp(-rate * self.dt, rate * self.dt))
        self.ready.fill_(True)
        return self.value
