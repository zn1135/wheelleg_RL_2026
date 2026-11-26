"""无固定时钟的弱动作提示，不向Actor增加相位或地形输入。"""
import hashlib
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[6]


class MotionHint:
    def __init__(self, cfg, device):
        path=ROOT/cfg.reference_motion_path
        if hashlib.sha256(path.read_bytes()).hexdigest()!=cfg.reference_motion_sha256:
            raise ValueError('Traditional motion reference fingerprint differs')
        with np.load(path,allow_pickle=False) as data:
            frames=torch.as_tensor(data['features'],device=device)
        if frames.ndim!=2 or frames.shape[1]!=8 or not torch.isfinite(frames).all():
            raise ValueError('Expected finite four-hip position/velocity reference')
        self.scale=frames.new_tensor([.35]*4+[4.]*4)
        self.frames=frames/self.scale
        self.norm=self.frames.square().sum(-1)

    def score(self, position, velocity):
        features=torch.cat((position,velocity),-1)/self.scale
        distance=(features.square().sum(-1,keepdim=True)+self.norm[None]-2*features@self.frames.T).clamp_min(0)
        # 任一相近运动片段均可得弱奖励，允许改变时序、左右先后及偏离参考。
        return torch.exp(-distance.min(-1).values/8.)
