"""GPU-only sampling of physically closed V14-style initial poses."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from .config import ASSET

BANK = Path(__file__).with_name('reset_poses.npz')

class RandomReset:
    def __init__(self, joint_names, device):
        self.device = device
        self.names = tuple(joint_names)
        with np.load(BANK, allow_pickle=False) as bank:
            if str(bank['model_hash']) != hashlib.sha256(ASSET.read_bytes()).hexdigest():
                raise ValueError('Reset pose bank asset mismatch; regenerate and verify it offline')
            names = bank['joint_names'].tolist()
            if set(names) != set(self.names):
                raise ValueError('Reset pose bank joint names do not match the robot')
            self.order = [names.index(name) for name in self.names]
            self.rows = {mode: {side: torch.as_tensor(bank[f'{mode}_{side}_pos'], device=device)
                for side in ('left','right')} for mode in ('normal','positive','negative')}
            self.bounds = {side: torch.as_tensor(bank[f'normal_{side}_tilt'], device=device)
                for side in ('left','right')}
            self.base_bound = float(bank['base_tilt'])

    def sample(self, count):
        ground = torch.rand(count, device=self.device) < .5
        positions, normal_bounds = [], []
        for side in ('left','right'):
            row = torch.randint(len(self.rows['normal'][side]), (count,), device=self.device)
            positive = torch.rand(count, device=self.device) < .6
            folded = torch.where(positive[:,None], self.rows['positive'][side][row], self.rows['negative'][side][row])
            positions.append(torch.where(ground[:,None], folded, self.rows['normal'][side][row]))
            normal_bounds.append(self.bounds[side][row])
        q = torch.cat(positions, dim=1)[:,self.order].clone()
        v = torch.zeros_like(q)
        for name in ('lf0','lf00','rf0','rf00'):
            v[:,self.names.index(name)] = torch.empty(count,device=self.device).uniform_(-np.pi/2,np.pi/2)
        for name in ('lfwheel','rfwheel'):
            col = self.names.index(name)
            q[:,col] = torch.empty(count,device=self.device).uniform_(-2*np.pi,2*np.pi)
            v[:,col] = torch.empty(count,device=self.device).uniform_(-50,50)
        minimum = torch.minimum(*normal_bounds).clamp(max=self.base_bound)
        normal_z = torch.maximum(torch.full_like(minimum,.35), .002-minimum)
        z = torch.where(ground, torch.full_like(normal_z,.25), normal_z)
        angles = torch.empty(count,3,device=self.device).uniform_(-1,1)
        angles *= torch.tensor([.15,.15,3.14],device=self.device)
        quat = quat_from_euler_xyz(*angles.unbind(-1))
        identity = torch.zeros_like(quat);identity[:,0] = 1.
        quat = torch.where(ground[:,None], identity, quat)
        return q, v, z, quat, ground
