"""MimicKit-compatible epsilon diffusion loss, EMA and SMP motion scoring.

Adapted from MimicKit (Apache-2.0), commit fbe1a36e878f036d6bf5521defb8eb3b8ab4170d.
Only the unconditional score path is implemented; generated state initialization is disabled.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .dit import TinyMotionDiT
from .features import FEATURE_DIM, FEATURE_SCHEMA, SAMPLE_DT, WINDOW

UPSTREAM_COMMIT = 'fbe1a36e878f036d6bf5521defb8eb3b8ab4170d'
SCORE_STEPS = (22,15,8)


@dataclass(frozen=True)
class PriorConfig:
    features: int = FEATURE_DIM
    window: int = WINDOW
    diffusion_steps: int = 50
    layers: int = 2
    heads: int = 4
    head_dim: int = 64
    ema_decay: float = .995
    ema_every: int = 10
    ema_after: int = 5000


class ErrorNormalizer(nn.Module):
    """Public DiffNormalizer: divide each noise-level error by its mean absolute value."""
    def __init__(self):
        super().__init__()
        self.register_buffer('count',torch.zeros((),dtype=torch.long))
        self.register_buffer('mean_abs',torch.ones(len(SCORE_STEPS)))
        self.register_buffer('pending_count',torch.zeros((),dtype=torch.long))
        self.register_buffer('pending_sum',torch.zeros(len(SCORE_STEPS)))

    @torch.no_grad()
    def record(self, errors):
        if errors.numel():
            self.pending_count.add_(len(errors))
            self.pending_sum.add_(errors.abs().sum(0))

    @torch.no_grad()
    def update(self):
        if self.pending_count > 0:
            total = self.count+self.pending_count
            self.mean_abs.copy_((self.count*self.mean_abs+self.pending_sum)/total)
            self.count.copy_(total)
            self.pending_count.zero_()
            self.pending_sum.zero_()

    def forward(self, errors):
        return errors/self.mean_abs.clamp_min(1e-4)


class MotionPrior(nn.Module):
    def __init__(self, config=PriorConfig()):
        super().__init__()
        if config.features != FEATURE_DIM or config.window != WINDOW or config.diffusion_steps != 50:
            raise ValueError('Incompatible SMP feature/window/diffusion schema')
        self.config = config
        self.denoiser = TinyMotionDiT(config.features,config.layers,config.heads,config.head_dim)
        self.ema_denoiser = deepcopy(self.denoiser).requires_grad_(False)
        self.register_buffer('ema_step',torch.zeros((),dtype=torch.long))
        self.register_buffer('feature_mean',torch.zeros(config.features))
        self.register_buffer('feature_std',torch.ones(config.features))
        # diffusers DDPMScheduler squaredcos_cap_v2, with beta capped at0.999.
        alpha_bar = lambda t: math.cos((t+.008)/1.008*math.pi/2)**2
        betas = torch.tensor([min(1-alpha_bar((i+1)/50)/alpha_bar(i/50),.999) for i in range(50)])
        self.register_buffer('alpha_cumprod',(1-betas).cumprod(0))
        self.error_normalizer = ErrorNormalizer()

    def normalize(self, windows):
        return (windows-self.feature_mean)/self.feature_std

    def add_noise(self, normalized, noise, timestep):
        alpha = self.alpha_cumprod[timestep][:,None]
        return alpha.sqrt()*normalized+(1-alpha).sqrt()*noise

    def loss(self, windows, *, timestep=None, noise=None):
        x = self.normalize(windows).flatten(1)
        if timestep is None:
            timestep = torch.randint(50,(len(x),),device=x.device)
        noise = torch.randn_like(x) if noise is None else noise
        prediction = self.denoiser(self.add_noise(x,noise,timestep),timestep)
        return F.l1_loss(prediction,noise)

    @torch.no_grad()
    def update_ema(self):
        step = int(self.ema_step)
        self.ema_step.add_(1)
        if step % self.config.ema_every:
            return
        if step <= self.config.ema_after:
            self.ema_denoiser.load_state_dict(self.denoiser.state_dict())
        else:
            for old,new in zip(self.ema_denoiser.parameters(),self.denoiser.parameters()):
                old.copy_(old*self.config.ema_decay+(1-self.config.ema_decay)*new)

    @torch.no_grad()
    def score_errors(self, windows, *, noise=None):
        if windows.shape[1:] != (WINDOW,FEATURE_DIM):
            raise ValueError('Expected continuous 10x30 motion windows')
        x = self.normalize(windows).flatten(1)
        errors = []
        for i,step in enumerate(SCORE_STEPS):
            timestep = torch.full((len(x),),step,dtype=torch.long,device=x.device)
            current_noise = torch.randn_like(x) if noise is None else noise[i]
            prediction = self.ema_denoiser(self.add_noise(x,current_noise,timestep),timestep)
            # Public epsilon→DDIM x0→epsilon path simplifies to this same prediction.
            errors.append((prediction-current_noise).square().mean(-1))
        return torch.stack(errors,-1)

    @torch.no_grad()
    def score(self, windows, *, noise=None, record=False):
        errors = self.score_errors(windows,noise=noise)
        reward = torch.exp(-6*self.error_normalizer(errors).mean(-1))
        if record:
            self.error_normalizer.record(errors)
        return reward

    def freeze(self):
        self.eval().requires_grad_(False)
        return self


def checkpoint(prior, *, iteration, dataset, optimizer, generator):
    return dict(format='ct_smp_prior_v1',feature_schema=FEATURE_SCHEMA,sample_dt=SAMPLE_DT,
                score_steps=list(SCORE_STEPS),score_scale=6.,feature_std_floor=.2,
                upstream_commit=UPSTREAM_COMMIT,model_config=asdict(prior.config),
                model_state_dict=prior.state_dict(),iteration=iteration,dataset=dataset,
                optimizer_state_dict=optimizer.state_dict(),generator_state=generator.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def load_prior(path, device='cpu', *, require_trained=True):
    saved = torch.load(path,map_location=device,weights_only=True)
    if (saved.get('format') != 'ct_smp_prior_v1' or saved.get('feature_schema') != FEATURE_SCHEMA or
            saved.get('sample_dt') != SAMPLE_DT or saved.get('upstream_commit') != UPSTREAM_COMMIT or
            saved.get('score_steps') != list(SCORE_STEPS) or saved.get('score_scale') != 6. or
            saved.get('feature_std_floor') != .2):
        raise ValueError('Incompatible SMP prior checkpoint')
    if require_trained and saved.get('iteration',0) < 1:
        raise ValueError('Untrained test fixture is not a motion prior')
    prior = MotionPrior(PriorConfig(**saved['model_config'])).to(device)
    prior.load_state_dict(saved['model_state_dict'],strict=True)
    return prior,saved
