"""Normal/recover state for the existing seven V14 observation slots.

This is a local standup extension: upstream flat V14 leaves recover unused.
State advances once per policy transition; observation reads are side effect free.
"""
import math

import torch

from .config import HEIGHT, STEP_DT, MODE_PARAMS


class ControlMode:
    def __init__(self, n, device):
        self.recover = torch.zeros(n, dtype=torch.bool, device=device)
        self.recovered = torch.zeros_like(self.recover)
        self.stable_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.failure_steps = torch.zeros_like(self.stable_steps)
        self.state_steps = torch.zeros_like(self.stable_steps)

    def reset(self, ids, ground_start=None):
        self.recover[ids] = False if ground_start is None else ground_start
        self.recovered[ids] = False
        self.stable_steps[ids] = 0
        self.failure_steps[ids] = 0
        self.state_steps[ids] = 0

    def advance(self, state):
        p = MODE_PARAMS
        finite = torch.isfinite(torch.stack((state['height'], state['roll'], state['pitch'],
                                            state['base_contact']), -1)).all(-1)
        failed = finite & ((state['base_contact'] > p['base_contact_n'])
                          | (state['roll'].abs() > math.radians(p['failure_tilt_deg']))
                          | (state['pitch'].abs() > math.radians(p['failure_tilt_deg']))
                          | (state['height'] < p['failure_height_m']))
        self.failure_steps = torch.where(failed, self.failure_steps + 1, 0)
        entered = ~self.recover & (self.failure_steps >= round(p['failure_hold_s'] / STEP_DT))
        active = self.recover | entered
        stable = (finite & (state['base_contact'] <= p['base_contact_n'])
                  & (state['roll'].abs() <= math.radians(p['stable_tilt_deg']))
                  & (state['pitch'].abs() <= math.radians(p['stable_tilt_deg']))
                  & ((state['height'] - HEIGHT).abs() <= p['height_tolerance_m'])
                  & torch.isfinite(state['wheel_contact']).all(-1)
                  & (state['wheel_contact'] > p['wheel_contact_n']).all(-1))
        self.stable_steps = torch.where(active & stable, self.stable_steps + 1, 0)
        completed = active & (self.stable_steps >= round(p['stable_hold_s'] / STEP_DT))
        self.recovered |= completed
        # Newly entered recovery has elapsed zero seconds at this observation.
        self.state_steps = torch.where(entered, 0, self.state_steps + 1)
        self.recover.copy_(active & ~completed)
        self.state_steps = torch.where(self.recover, self.state_steps, 0)

    def observation(self):
        result = torch.zeros(len(self.recover), 7, device=self.recover.device)
        result[:, 0] = (~self.recover).float()
        result[:, 3] = self.recover.float()
        result[:, 5] = self.recover.float() * HEIGHT * 5
        result[:, 6] = self.state_steps.float() * STEP_DT
        return result
