"""Independent, opt-in command integrator; never changes the actor or its history FIFO."""
import torch
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
REASONS = {
    'disabled': 0,
    'moving': 1,
    'turning': 2,
    'unstable': 3,
    'contact_warmup': 4,
    'unsupported': 5,
    'body_contact': 6,
    'waiting': 7,
    'active': 8,
    'reset': 9
}

def configuration():
    cfg = FudanWalkCfg.parking_integral
    return {k: getattr(cfg, k) for k in vars(cfg) if not k.startswith('_')}

def finite(*values):
    if any((not torch.isfinite(v).all() for v in values)):
        raise FloatingPointError('Non-finite parking integral input/state')

def feedback_velocity(raw, encoder=False):
    finite(raw)
    return raw / 2.0 if encoder else raw.clone()

class ParkingIntegral:

    def __init__(self, n, device='cpu', enabled=None):
        self.cfg = configuration()
        c = self.cfg
        self.enabled = torch.full((n,), c['enabled'], device=device, dtype=torch.bool) if enabled is None else torch.as_tensor(enabled, device=device, dtype=torch.bool)
        self.integral = torch.zeros(n, device=device)
        self.compensation = torch.zeros_like(self.integral)
        self.ticks = torch.zeros(n, device=device, dtype=torch.long)
        self.last_good = torch.zeros_like(self.enabled)
        self.active = self.last_good.clone()
        self.reason = torch.zeros_like(self.ticks)
        self.contacts = torch.zeros(round(c['contact_window_s'] / c['contact_dt']), n, 3, device=device)
        self.contact_count = torch.zeros_like(self.ticks)
        self.cursor = 0
        self.pending = False
        self.last_tick = -1
        self.updates = 0

    def reset(self, ids):
        self.integral[ids] = 0.0
        self.compensation[ids] = 0.0
        self.ticks[ids] = 0
        self.last_good[ids] = False
        self.active[ids] = False
        self.reason[ids] = REASONS['reset']
        self.contacts[:, ids] = 0.0
        self.contact_count[ids] = 0

    def observe_contacts(self, values):
        finite(values)
        for value in values:
            self.contacts[self.cursor] = value
            self.cursor = (self.cursor + 1) % len(self.contacts)
            self.contact_count.add_(1).clamp_(max=len(self.contacts))

    def conditions(self, raw, angles):
        finite(raw, angles, self.integral, self.compensation, self.contacts)
        mean = self.contacts.mean(0)
        reason = torch.full_like(self.ticks, REASONS['active'])
        reason[mean[:, 2] > self.cfg['body_contact_max_N']] = REASONS['body_contact']
        reason[(mean[:, :2] <= self.cfg['wheel_contact_min_N']).any(-1)] = REASONS['unsupported']
        reason[self.contact_count < len(self.contacts)] = REASONS['contact_warmup']
        reason[(angles.abs() >= self.cfg['angle_limit_deg']).any(-1)] = REASONS['unstable']
        reason[raw[:, 1] != 0] = REASONS['turning']
        reason[raw[:, 0] != 0] = REASONS['moving']
        reason[~self.enabled] = REASONS['disabled']
        return (reason == REASONS['active'], reason, mean)

    def begin(self, tick, raw, angles):
        if self.pending or tick != self.last_tick + 1:
            raise RuntimeError('Exactly one begin/end pair required per policy tick')
        self.last_tick = tick
        self.pending = True
        good, reason, mean = self.conditions(raw, angles)
        self.ticks = torch.where(good & self.last_good, self.ticks + 1, torch.zeros_like(self.ticks))
        self.last_good = good
        self.active = good & (self.ticks >= round(self.cfg['gate_duration_s'] / self.cfg['policy_dt']))
        self.integral[~good] = 0.0
        self.compensation.zero_()
        self.compensation[self.active] = (self.integral[self.active] * self.cfg['gain']).clamp(-self.cfg['compensation_limit'], self.cfg['compensation_limit'])
        reason[good & ~self.active] = REASONS['waiting']
        self.reason = reason
        actual = raw.clone()
        actual[:, 0] += self.compensation
        return (actual, mean)

    def end(self, raw, feedback, angles):
        if not self.pending:
            raise RuntimeError('Integral updated twice or before begin')
        finite(feedback)
        good, reason, _ = self.conditions(raw, angles)
        invalid = ~good
        self.integral[invalid] = 0.0
        self.compensation[invalid] = 0.0
        self.ticks[invalid] = 0
        self.last_good[invalid] = False
        self.active &= good
        self.reason[invalid] = reason[invalid]
        self.integral[self.active] = (self.integral[self.active] + (raw[:, 0] - feedback)[self.active] * self.cfg['policy_dt']).clamp(-self.cfg['integral_limit'], self.cfg['integral_limit'])
        self.pending = False
        self.updates += 1
