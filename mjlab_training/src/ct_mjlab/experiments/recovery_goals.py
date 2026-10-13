"""Separate righting, wheel balance and commanded height; no robot pose target."""
import math
import torch

PARAMETERS = dict(tilt_deg=15., wheel_contact_n=5., base_contact_n=1.,
                  linear_speed_m_s=.15, angular_speed_rad_s=.5,
                  height_tolerance_m=.03, success_hold_s=.5,
                  ready_fraction=2 / 3, curriculum_hold_s=2., minimum_updates=100)
STAGES = ('righting', 'wheel_balance', 'height_tracking')


def predicates(*, gravity, wheel_support, base_contact, linear_velocity,
               angular_velocity, height, target_height, valid):
    """World-up support uses clamp(-sensor Fz, 0), supplied by the caller."""
    p = PARAMETERS
    finite = (torch.isfinite(gravity).all(-1) & torch.isfinite(wheel_support).all(-1)
              & torch.isfinite(base_contact) & torch.isfinite(linear_velocity).all(-1)
              & torch.isfinite(angular_velocity).all(-1) & torch.isfinite(height)
              & torch.isfinite(target_height) & valid)
    righted = finite & (-gravity[:, 2] >= math.cos(math.radians(p['tilt_deg'])))
    supported = (righted & (wheel_support > p['wheel_contact_n']).all(-1)
                 & (base_contact <= p['base_contact_n']))
    balanced = (supported & (linear_velocity.norm(dim=-1) <= p['linear_speed_m_s'])
                & (angular_velocity.norm(dim=-1) <= p['angular_speed_rad_s']))
    tracking = balanced & ((height - target_height).abs() <= p['height_tolerance_m'])
    return dict(righting=righted, wheel_balance=balanced, height_tracking=tracking,
                supported_upright=supported)


class Curriculum:
    """Batch-wide promotion; the first two gates deliberately ignore height."""
    def __init__(self):
        self.stage = self.ready_steps = self.stage_steps = 0
        self.last_step = -1
        self.history = []

    def advance(self, fraction, step, step_dt):
        if self.last_step == step:
            return False
        self.last_step = step
        self.stage_steps += 1
        self.ready_steps = self.ready_steps + 1 if fraction >= PARAMETERS['ready_fraction'] else 0
        if (self.stage < 2 and self.stage_steps >= 24 * PARAMETERS['minimum_updates']
                and self.ready_steps >= math.ceil(PARAMETERS['curriculum_hold_s'] / step_dt)):
            self.history.append(dict(from_stage=STAGES[self.stage], to_stage=STAGES[self.stage + 1],
                                     control_step=step, ready_fraction=fraction))
            self.stage += 1
            self.stage_steps = self.ready_steps = 0
            return True
        return False

    def state_dict(self):
        return dict(stage=self.stage, ready_steps=self.ready_steps,
                    stage_steps=self.stage_steps, history=list(self.history))

    def load(self, saved):
        if (saved['stage'] not in (0, 1, 2) or saved['ready_steps'] < 0 or saved['stage_steps'] < 0):
            raise ValueError('Invalid recovery curriculum state')
        self.stage, self.ready_steps, self.stage_steps = (
            saved['stage'], saved['ready_steps'], saved['stage_steps'])
        self.history = list(saved['history'])
        self.last_step = -1
