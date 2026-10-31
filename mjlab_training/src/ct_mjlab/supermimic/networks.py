# SPDX-License-Identifier: MIT
# RunningNorm/MLP structure adapted from RadiantForever/supermimic_wheelbipe.
# Copyright (c) 2026 RadiantForever; see THIRD_PARTY_LICENSE.txt.
"""Physical dynamics and deployable actor; derived signals share one state."""

import torch
from torch import nn
from .analytic_reward import CONTEXT_DIM
from .state import (StateAdapter, STATE_DIM, PARAM_DIM, PHYS_DIM, DELTA_DIM,
                    SUBSTEPS, PHYSICAL, MEMORY, SENSORS, NOISE, SOLREF,
                    physical_delta, integrate_physical)


def model_context(context):
    # Unlimited joint bounds stay infinite in the exact reward, but never enter an MLP.
    return context.masked_fill(context.isinf(),0.)


def contact_parameters(parameters):
    """Encode raw solref as reference stiffness/10000 and damping/100.

    The 2ms timestep, REFSAFE and fixed contact impedance are checked by make_env.
    These are material coefficients before the solver's dmax factors, not an
    estimate of measured contact force. Raw replay/simulator parameters stay intact.
    """
    solref = parameters[...,SOLREF]
    timeconst = solref[...,0].clamp_min(.004)
    damping_ratio = solref[...,1].abs().clamp_min(1e-8)
    stiffness = torch.where(solref[...,0] > 0.,
        (timeconst*damping_ratio).square().reciprocal(),-solref[...,0])
    damping = torch.where(solref[...,1] > 0.,2./timeconst,-solref[...,1])
    contact = torch.stack((stiffness/10000.,damping/100.),-1)
    return torch.cat((parameters[...,:SOLREF.start],contact,parameters[...,SOLREF.stop:]),-1)


def normalize_contact_inputs(values, norm, parameter_offset):
    normalized = norm(values)
    lo, hi = parameter_offset+SOLREF.start, parameter_offset+SOLREF.stop
    # A constant training material must not turn a new physical value into 1e6 sigma.
    return torch.cat((normalized[...,:lo],values[...,lo:hi],normalized[...,hi:]),-1)


def mlp(inp, hidden, out, *, normalized=False):
    layers = []
    for width in hidden:
        layers.append(nn.Linear(inp, width))
        if normalized:
            layers.append(nn.LayerNorm(width))
        layers.append(nn.SiLU())
        inp = width
    layers.append(nn.Linear(inp, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=.01):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.zeros(()))
        self.eps = eps

    @torch.no_grad()
    def update(self, values):
        values = values.reshape(-1, self.mean.numel()).float()
        if not len(values):
            return
        count = self.count + len(values)
        delta = values.mean(0) - self.mean
        self.var.copy_((self.var*self.count + values.var(0, unbiased=False)*len(values)
            + delta.square()*self.count*len(values)/count) / count)
        self.mean.add_(delta * len(values)/count)
        self.count.copy_(count)

    def std(self):
        return self.var.clamp_min(self.eps**2).sqrt()

    def forward(self, values):
        return (values-self.mean)/self.std()


def make_actor(device):
    # Reuse the current deployable 125 -> estimator3 + actor25 -> action6 network.
    from tensordict import TensorDict
    from ct_mjlab.envs.chuanliantui_split.env_cfg import make_ppo_cfg
    from ct_mjlab.envs.chuanliantui_split.network import HistoryActor
    cfg = make_ppo_cfg('walk').actor
    dummy = TensorDict(dict(actor=torch.zeros(1,25), history=torch.zeros(1,125)), batch_size=[1])
    actor = HistoryActor(dummy, {'actor':['actor']}, 'actor', 6,
        hidden_dims=cfg.hidden_dims, activation=cfg.activation, distribution_cfg=None)
    # Start close to zero without importing any PPO weights or optimizer.
    with torch.no_grad():
        actor.mlp[-1].weight.mul_(.01)
        actor.mlp[-1].bias.zero_()
    return actor.to(device)


def act(actor, history):
    # Freeze encoder parameters in imagination, keeping gradients through its input history.
    velocity = actor.encoder(history)
    return actor.mlp(torch.cat((history[..., -25:], velocity), -1))


class WorldModel(nn.Module):
    """Predict five 2ms physical increments, never a separate observation/reward.

    All passive joints are retained. Forces are predicted once as physical
    channels, then reused by reward and termination. No physics solver is implied.
    """
    def __init__(self):
        super().__init__()
        self.adapter = StateAdapter()
        width = PHYS_DIM+PARAM_DIM+18+1
        self.net = mlp(width, (512,512), SUBSTEPS*DELTA_DIM, normalized=True)
        self.in_norm, self.delta_norm = RunningNorm(width), RunningNorm(DELTA_DIM)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def inputs(state, parameters, action, previous, previous_previous):
        physical = state[...,PHYSICAL]
        # Global XY and task commands cannot cause fixed-action physical changes.
        physical = torch.cat((torch.zeros_like(physical[...,:2]),physical[...,2:]),-1)
        return torch.cat((physical,contact_parameters(parameters),action,previous,previous_previous,
                          state[...,MEMORY][...,:1]*.01),-1)

    @torch.no_grad()
    def observe(self, w):
        inputs = self.inputs(w['state'],w['parameters'],w['action'],w['previous'],w['previous_previous'])
        valid = w['mask'].bool() & inputs.isfinite().all(-1) & w['path'].isfinite().all(-1)
        self.in_norm.update(inputs[valid])
        path = w['path'].reshape(*w['path'].shape[:-1],SUBSTEPS,PHYS_DIM)
        before = torch.cat((w['state'][...,None,PHYSICAL],path[...,:-1,:]),-2)
        delta = physical_delta(before,path)
        self.delta_norm.update(delta[valid])

    def step(self, state, parameters, action, next_command, context, previous, previous_previous, noise,
             *, return_path=False):
        inputs = self.inputs(state,parameters,action,previous,previous_previous)
        y = self.net(normalize_contact_inputs(inputs,self.in_norm,PHYS_DIM))
        increments = y.reshape(*y.shape[:-1],SUBSTEPS,DELTA_DIM)*self.delta_norm.std()+self.delta_norm.mean
        physical = state[...,PHYSICAL]
        path = []
        for t in range(SUBSTEPS):
            physical = integrate_physical(physical,increments[...,t,:])
            path.append(physical)
        path = torch.stack(path,-2)
        result = self.adapter.advance(state,parameters,path,next_command,action,previous,previous_previous,context,noise)
        return (*result,path) if return_path else result


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RunningNorm(STATE_DIM+PARAM_DIM+CONTEXT_DIM)
        self.net = mlp(STATE_DIM+PARAM_DIM+CONTEXT_DIM, (256,128), 1)

    @staticmethod
    def inputs(state, parameters, context):
        return torch.cat((state,contact_parameters(parameters),model_context(context)),-1)

    def forward(self, state, parameters, context):
        inputs = self.inputs(state,parameters,context)
        return self.net(normalize_contact_inputs(inputs,self.norm,STATE_DIM)).squeeze(-1)


class MotionCritic(nn.Module):
    """Value inputs share the motion model's available state, including delays."""
    STATE_WIDTH = 47+162+18+1

    def __init__(self):
        super().__init__()
        width = self.STATE_WIDTH+PARAM_DIM+CONTEXT_DIM
        self.norm = RunningNorm(width)
        self.net = mlp(width, (256,128), 1)

    @staticmethod
    def inputs(state, parameters, context):
        from .motion import physical47_from_full64
        motion_state = torch.cat((physical47_from_full64(state[...,PHYSICAL]),
            state[...,SENSORS],state[...,NOISE],state[...,MEMORY][...,:1]),-1)
        return torch.cat((motion_state,contact_parameters(parameters),model_context(context)),-1)

    def forward(self, state, parameters, context):
        inputs = self.inputs(state,parameters,context)
        return self.net(normalize_contact_inputs(inputs,self.norm,self.STATE_WIDTH)).squeeze(-1)
