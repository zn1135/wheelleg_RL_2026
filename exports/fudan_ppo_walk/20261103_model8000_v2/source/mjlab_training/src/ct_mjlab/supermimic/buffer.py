"""Time/environment ring buffer; windows stop at both failure and timeout."""

import torch
from .analytic_reward import FEATURE_DIM, CONTEXT_DIM
from .state import STATE_DIM, PARAM_DIM, PHYS_DIM, SUBSTEPS


class ReplayBuffer:
    FIELDS = dict(history=125, privileged=71, action=6, next_frame=25,
                  next_privileged=71, velocity=3, reward=0, done=0, failure=0,
                  features=FEATURE_DIM,context=CONTEXT_DIM,previous=6,previous_previous=6,
                  state=STATE_DIM,parameters=PARAM_DIM,next_state=STATE_DIM,path=SUBSTEPS*PHYS_DIM)

    def __init__(self, capacity, num_envs, device):
        self.capacity, self.num_envs, self.device = capacity, num_envs, device
        self.data = {key:torch.zeros((capacity,num_envs)+((dim,) if dim else ()),
            device=device, dtype=torch.bool if key in ('done','failure') else torch.float32)
            for key,dim in self.FIELDS.items()}
        self.ptr = self.size = 0
        # Keep entire environments out of fitting, including their rewards and norm updates.
        ids = torch.arange(num_envs,device=device)
        self.train_ids, self.validation_ids = ids[ids%10!=0], ids[ids%10==0]
        if not len(self.train_ids) or not len(self.validation_ids):
            raise ValueError('At least two environments are required')

    @torch.no_grad()
    def add(self, **transition):
        for key in self.FIELDS:
            self.data[key][self.ptr].copy_(transition[key])
        self.ptr = (self.ptr+1)%self.capacity
        self.size = min(self.size+1,self.capacity)

    def sample(self, batch, length, *, validation=False):
        if self.size < length:
            raise ValueError(f'Need {length} frames, have {self.size}')
        pool = self.validation_ids if validation else self.train_ids
        env = pool[torch.randint(len(pool),(batch,),device=self.device)]
        start = torch.randint(self.size-length+1,(batch,),device=self.device)
        age = start[:,None]+torch.arange(length,device=self.device)
        rows = (self.ptr-self.size+age)%self.capacity
        w = {key:value[rows,env[:,None]] for key,value in self.data.items()}
        w['mask'] = torch.cumprod(torch.cat((torch.ones(batch,1,device=self.device),
                                           (~w['done'][:,:-1]).float()),1),1)
        return w
