"""Restore normal/recover slots with the current per-environment height command."""
import math
import torch

PARAMETERS = dict(stable_hold_s=.5, stable_tilt_deg=15., height_tolerance_m=.03,
    wheel_contact_n=5., base_contact_n=1., failure_hold_s=.1,
    failure_tilt_deg=40., failure_height_m=.16)

class ControlMode:
    def __init__(self,n,device,step_dt=.02):
        self.step_dt=step_dt
        self.recover=torch.zeros(n,dtype=torch.bool,device=device)
        self.recovered=torch.zeros_like(self.recover)
        self.stable_steps=torch.zeros(n,dtype=torch.long,device=device)
        self.failure_steps=torch.zeros_like(self.stable_steps)
        self.state_steps=torch.zeros_like(self.stable_steps)
        self.last_step=None

    def reset(self,ids,ground_start):
        self.recover[ids]=ground_start
        self.recovered[ids]=False
        self.stable_steps[ids]=0;self.failure_steps[ids]=0;self.state_steps[ids]=0

    def advance(self,state,target_height,step_id):
        if self.last_step==step_id:return
        self.last_step=step_id;p=PARAMETERS
        finite=torch.isfinite(torch.stack((state['height'],state['roll'],state['pitch'],
            state['base_contact'],target_height),-1)).all(-1)
        failed=finite & ((state['base_contact']>p['base_contact_n'])
            | (state['roll'].abs()>math.radians(p['failure_tilt_deg']))
            | (state['pitch'].abs()>math.radians(p['failure_tilt_deg']))
            | (state['height']<p['failure_height_m']))
        self.failure_steps=torch.where(failed,self.failure_steps+1,0)
        entered=~self.recover & (self.failure_steps>=round(p['failure_hold_s']/self.step_dt))
        active=self.recover|entered
        stable=(finite & state['valid'] & (state['base_contact']<=p['base_contact_n'])
            & (state['roll'].abs()<=math.radians(p['stable_tilt_deg']))
            & (state['pitch'].abs()<=math.radians(p['stable_tilt_deg']))
            & ((state['height']-target_height).abs()<=p['height_tolerance_m'])
            & torch.isfinite(state['wheel_contact']).all(-1)
            & (state['wheel_contact']>p['wheel_contact_n']).all(-1))
        self.stable_steps=torch.where(active & stable,self.stable_steps+1,0)
        completed=active & (self.stable_steps>=round(p['stable_hold_s']/self.step_dt))
        self.recovered|=completed
        self.state_steps=torch.where(entered,0,self.state_steps+1)
        self.recover.copy_(active & ~completed)
        self.state_steps=torch.where(self.recover,self.state_steps,0)

    def observation(self,target_height):
        slots=target_height.new_zeros(len(self.recover),7)
        slots[:,0]=(~self.recover).float();slots[:,3]=self.recover.float()
        slots[:,5]=torch.where(self.recover,target_height*5,0.)
        slots[:,6]=self.state_steps.float()*self.step_dt
        return slots
