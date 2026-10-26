"""Pure-PyTorch adaptation of MimicKit TinyStableMotionDiTModel.

Copyright 2025 Xue Bin Peng; Copyright 2024 HuggingFace Inc.; Apache-2.0.
Modified for the Wheel-Legged-Gym
30-channel robot features; removes diffusers dependency, conditioning and generation.
Upstream commit and licenses: third_party/mimickit_smp/NOTICE.md.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.heads = heads
        self.to_q, self.to_k, self.to_v = (nn.Linear(dim,dim,bias=False) for _ in range(3))
        self.to_out = nn.ModuleList((nn.Linear(dim,dim),nn.Dropout(0.)))

    def forward(self, x):
        n,t,d = x.shape
        q,k,v = [layer(x).reshape(n,t,self.heads,d//self.heads).transpose(1,2)
                 for layer in (self.to_q,self.to_k,self.to_v)]
        value = F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(n,t,d)
        return self.to_out[1](self.to_out[0](value))


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim,dim*8)

    def forward(self, x):
        value,gate = self.proj(x).chunk(2,-1)
        return value*F.silu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.ModuleList((SwiGLU(dim),nn.Dropout(0.),nn.Linear(dim*4,dim)))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class StableMotionDiTBlock(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim,eps=1e-5,elementwise_affine=False)
        self.attn1 = Attention(dim,heads)
        self.norm2 = nn.LayerNorm(dim,eps=1e-5,elementwise_affine=False)
        self.ff = FeedForward(dim)
        self.scale_shift_table = nn.Parameter(torch.randn(1,1,6,dim)/math.sqrt(dim))

    def forward(self, x, modulation):
        shift_a,scale_a,gate_a,shift_f,scale_f,gate_f = (
            self.scale_shift_table+modulation.reshape(len(x),1,6,-1)).unbind(-2)
        x = x+gate_a*self.attn1(self.norm1(x)*(1+scale_a)+shift_a)
        return x+gate_f*self.ff(self.norm2(x)*(1+scale_f)+shift_f)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1, self.linear_2 = nn.Linear(256,dim),nn.Linear(dim,dim)

    def forward(self, timestep):
        frequencies = torch.exp(-math.log(10000)*torch.arange(128,device=timestep.device).float()/128)
        phase = timestep.float()[:,None]*frequencies
        embedded = torch.cat((phase.cos(),phase.sin()),-1)
        return self.linear_2(F.silu(self.linear_1(embedded)))[:,None]


class TinyMotionDiT(nn.Module):
    def __init__(self, channels=30, num_layers=2, heads=4, head_dim=64):
        super().__init__()
        dim = heads*head_dim
        self.channels = channels
        self.transformer_blocks = nn.ModuleList([StableMotionDiTBlock(dim,heads) for _ in range(num_layers)])
        self.time_embedding = TimestepEmbedding(dim)
        self.time_linear = nn.Linear(dim,6*dim)
        self.preprocess_conv = nn.Conv1d(channels,channels,1,bias=False)
        self.proj_in = nn.Linear(channels,dim,bias=False)
        self.proj_out = nn.Linear(dim,channels,bias=False)
        self.postprocess_conv = nn.Conv1d(channels,channels,1,bias=False)
        position = torch.arange(32)[:,None]
        frequency = torch.exp(torch.arange(0,dim,2)*(-math.log(10000)/dim))
        encoding = torch.zeros(1,32,dim)
        encoding[0,:,0::2],encoding[0,:,1::2] = (position*frequency).sin(),(position*frequency).cos()
        self.register_buffer('position_encoding',encoding)

    def forward(self, hidden_states, timestep):
        n = len(hidden_states)
        x = hidden_states.reshape(n,-1,self.channels).transpose(1,2)
        x = (x+self.preprocess_conv(x)).transpose(1,2)
        x = self.proj_in(x)+self.position_encoding[:,:x.shape[1]]
        modulation = self.time_linear(F.silu(self.time_embedding(timestep)))
        for block in self.transformer_blocks:
            x = block(x,modulation)
        x = self.proj_out(x).transpose(1,2)
        return (x+self.postprocess_conv(x)).transpose(1,2).reshape(n,-1)
