"""GPU replay switch: confirmed flight, wheel touchdown, then a settling interval."""
import numpy as np
import torch
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import LANDING


class LandingHandoff:
    def __init__(self, started, buffer_ticks=None):
        self.started = started
        self.buffer_ticks = LANDING['buffer_ticks'] if buffer_ticks is None else buffer_ticks
        self.air = torch.zeros_like(started)
        self.ascended = torch.zeros_like(started,dtype=torch.bool)
        self.touchdown = torch.full_like(started,-1)

    def active(self, tick):
        return (self.started>=0) & (tick>=self.started) & torch.where(self.touchdown<0,
            tick-self.started<LANDING['max_jump_ticks'], tick-self.touchdown<=self.buffer_ticks)

    def observe(self, tick, rows, active):
        contact = (rows[:,:,8:10]>.5).any(dim=(0,2))
        self.observe_signals(tick, contact, (rows[:,:,19]>.15).any(0), active)

    def observe_signals(self, tick, contact, rising, active):
        first = (active & (self.touchdown<0) & contact & self.ascended
                 & (self.air>=LANDING['air_ticks']))
        self.touchdown[first] = tick
        airborne = active & ~contact & (self.touchdown<0)
        self.air = torch.where(airborne,self.air+1,0)
        self.ascended = airborne & (self.ascended | rising)


def expected_landing_mode(trace, start, buffer_ticks=None, *, substeps=10):
    started = torch.tensor([start])
    handoff = LandingHandoff(started,buffer_ticks)
    expected = []
    for tick, rows in enumerate(torch.as_tensor(trace).reshape(-1,substeps,1,23)):
        active = handoff.active(tick)
        expected.append(bool(active.item()))
        handoff.observe(tick,rows,active)
    return np.array(expected), int(handoff.touchdown.item())
