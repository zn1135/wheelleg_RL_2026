# Copyright (c) 2026 SCUTRobotLab. SPDX-License-Identifier: MIT
# Adapted from wheelbipe25_v3/env.py at the commit in scut_v14.json.
"""Backend-independent V14 observation, delay, action and active reward kernels."""
import torch
from .config import PARAMS as P, WEIGHTS, STEP_DT, HEIGHT

class Delay:
    """IsaacLab delay semantics: append current, clamp lookup to oldest available."""
    def __init__(self, n, dim, max_lag, device):
        self.buf=torch.zeros(n,max_lag+1,dim,device=device)
        self.lag=torch.ones(n,dtype=torch.long,device=device)
        self.count=torch.zeros(n,dtype=torch.long,device=device)
        self.ids=torch.arange(n,device=device)
    def reset(self, ids, lo, hi, randomize=True):
        self.buf[ids]=0; self.count[ids]=0
        self.lag[ids]=torch.randint(lo,hi,(len(ids),),device=self.buf.device) if randomize else lo
    def compute(self, value, ids=None):
        ids=self.ids if ids is None else ids
        self.buf[ids,1:]=self.buf[ids,:-1].clone(); self.buf[ids,0]=value[ids]
        lookup=torch.minimum(self.lag[ids],self.count[ids])
        self.count[ids]=(self.count[ids]+1).clamp(max=self.buf.shape[1]-1)
        return self.buf[ids,lookup]

def core_obs(ang, gravity, pos, vel, actions):
    n=len(pos); zero=pos.new_zeros(n,3); h=pos.new_full((n,1),HEIGHT)
    mode=pos.new_zeros(n,7);mode[:,0]=1
    return torch.cat((zero.clamp(-100,100),h.clamp(0,1)*5,
        ang.clamp(-100,100)*.5,gravity.clamp(-100,100),pos.clamp(-100,100),
        vel.clamp(-200,200)*.1,actions.clamp(-100,100),mode),-1)

class Recipe:
    def __init__(self, n, device, nominal, randomize=True):
        self.n=n; self.device=device;self.nominal=nominal;self.randomize=randomize
        self.actions=torch.zeros(n,6,device=device);self.previous=torch.zeros_like(self.actions)
        self.before=torch.zeros_like(self.actions)
        self.obs_delays=[Delay(n,d,4,device) for d in (3,3,6,6)]
        self.act_delays=[Delay(n,d,3,device) for d in (4,2)]
        self.obs_lags=torch.ones(n,4,device=device);self.act_lags=torch.ones(n,2,device=device)
        self.kp=torch.ones(n,6,device=device);self.kp[:,:4]*=10;self.kp[:,4:]=0
        self.kd=torch.ones(n,6,device=device);self.kd[:,4:]=.1
        self.effort=torch.tensor([40]*4+[3.9]*2,device=device).expand(n,-1).clone()
        self.effort_scale=torch.ones(n,6,device=device)
        self.material=torch.tensor([.5,.5,0.]*2,device=device).expand(n,-1).clone()
        self.mass_scale=torch.ones(n,1,device=device)
        self.bad_obs=torch.zeros(n,dtype=torch.bool,device=device)
        self.raw_obs_max=torch.zeros(n,device=device)
        self.delayed=[torch.zeros(n,d,device=device) for d in (3,3,6,6)]
    def reset(self, ids):
        self.actions[ids]=0;self.previous[ids]=0;self.before[ids]=0;self.bad_obs[ids]=False
        self.raw_obs_max[ids]=0
        for delay in self.obs_delays: delay.reset(ids,1,4,self.randomize)
        for delay in self.act_delays: delay.reset(ids,1,3,self.randomize)
        self.obs_lags=torch.stack([d.lag for d in self.obs_delays],-1).float()
        self.act_lags=torch.stack([d.lag for d in self.act_delays],-1).float()
    def torque(self, pos, vel, episode_steps):
        leg=(.5*self.actions[:,:4]+self.nominal[:4]).clamp(-torch.pi,torch.pi)
        wheel=(10*self.actions[:,4:]).clamp(-20,20)
        leg=self.act_delays[0].compute(leg);wheel=self.act_delays[1].compute(wheel)
        result=torch.cat((self.kp[:,:4]*(leg-pos[:,:4])-self.kd[:,:4]*vel[:,:4],
                          self.kd[:,4:]*(wheel-vel[:,4:])), -1)
        # SCUT's event scales computed effort BEFORE fixed motor saturation.
        result=(result*self.effort_scale).clamp(-self.effort,self.effort)
        return torch.where((episode_steps*STEP_DT<.2)[:,None],0,result)
    def observe_substep(self,s,ids=None):
        pos=s['pos']-self.nominal
        values=(s['ang'],s['gravity'],pos,s['vel'])
        for i,(delay,x) in enumerate(zip(self.obs_delays,values)):
            if ids is None:self.delayed[i]=delay.compute(x)
            else:self.delayed[i][ids]=delay.compute(x,ids)
    def observations(self,s,advance=True):
        pos=s['pos']-self.nominal;pos=pos.clone();pos[:,4:]=0
        values=(s['ang'],s['gravity'],pos,s['vel'])
        if advance:self.observe_substep(s)
        delayed=[x.clone() for x in self.delayed]
        delayed[2][:,4:]=0
        if self.randomize:
            scales=(.25,.05,.025,None)
            for i,scale in enumerate(scales):
                if scale is not None:delayed[i]=delayed[i]+(2*torch.rand_like(delayed[i])-1)*scale
            delayed[3]=delayed[3]+(2*torch.rand_like(delayed[3])-1)*delayed[3].new_tensor([.5]*4+[1]*2)
        actor=core_obs(*delayed,self.actions)
        clean=core_obs(*values,self.actions)
        extra=torch.cat((self.kp.clamp(-100,100),self.kd.clamp(-100,100),s['torque'].clamp(-100,100)*.05,
            self.obs_lags.clamp(-100,100),self.act_lags.clamp(-100,100),
            s['wheel_vel'].flatten(1).clamp(-100,100), (s['wheel_contact']>5).float(),
            self.mass_scale.clamp(-10,10), self.material.clamp(-100,100)), -1)
        # Source clean core28, linear velocity3, absolute height1, ctrl mode7, extra39.
        critic=torch.cat((clean[:,:28],s['lin'].clamp(-100,100),s['height'][:,None].clamp(-10,10)*5,
                          clean[:,28:],extra),-1)
        raw=torch.cat((*delayed,self.actions,*values,s['lin'],s['height'][:,None],extra),-1)
        self.bad_obs=~torch.isfinite(raw).all(-1);self.raw_obs_max=raw.abs().amax(-1)
        self.before.copy_(self.previous);self.previous.copy_(self.actions)
        return {'actor':torch.nan_to_num(actor,nan=0,posinf=0,neginf=0),
                'critic':torch.nan_to_num(critic,nan=0,posinf=0,neginf=0)}
    def reward_terms(self,s,terminated):
        sq=lambda x:(x*x).sum(-1)
        vel,acc,tau=s['vel'],s['acc'],s['torque'];g=s['gravity'];lin=s['lin'];ang=s['ang']
        second=self.actions-2*self.previous+self.before
        e=-lin[:,0]*torch.cos(s['pitch']);yaw=-ang[:,2];height=s['height']-HEIGHT
        fork=s['wheel_pos'][:,0,0]-s['wheel_pos'][:,1,0]
        terms=dict(termination=terminated.float(),leg_joint_acc=sq(acc[:,:4]),leg_joint_vel=sq(vel[:,:4]),
            joint_torque=sq(tau),wheel_acc=sq(acc[:,4:]),wheel_vel=sq(vel[:,4:]),
            wheel_power=(tau[:,4:]*vel[:,4:]).clamp(min=0).sum(-1),lin_vel_z=lin[:,2]**2,
            ang_vel_xy=sq(ang[:,:2]),action_rate=sq(self.actions-self.previous),
            action_smoothness_leg=sq(second[:,:4]),action_smoothness_wheel=sq(second[:,4:]),
            flat_orientation_y_v=((P['orientation_y_A']+P['orientation_y_bias'])*g[:,0])**2,
            flat_orientation_x_v=((P['orientation_x_A']+P['orientation_x_bias'])*g[:,1])**2,
            flat_orientation_y_exp=torch.exp(-g[:,0]**2/P['orientation_y_exp_sigma']),
            flat_orientation_x_exp=torch.exp(-g[:,1]**2/P['orientation_x_exp_sigma']),
            track_lin_vel_xy=torch.exp(-e.clamp(-P['lin_vel_err_constraint'],P['lin_vel_err_constraint'])**2/P['lin_vel_xy_sigma']),
            track_lin_vel_xy_square=(e*P['lin_vel_xy_square_sigma'])**2,
            track_ang_vel_z=torch.exp(-yaw.clamp(-P['ang_vel_err_constraint'],P['ang_vel_err_constraint'])**2/P['ang_vel_z_sigma']),
            track_ang_vel_z_square=(yaw*P['ang_vel_z_square_sigma'])**2,
            stand_still_lin_vel=lin[:,:2].abs().sum(-1),
            track_height_exp_tight=torch.exp(-height.clamp(-P['height_err_constraint'],P['height_err_constraint'])**2/P['height_tight_sigma']),
            track_height_square=(P['height_square_sigma']*height)**2,
            no_fork=(fork.abs()>P['no_fork_distance']).float(),no_fork_square=(fork*P['no_fork_square_sigma'])**2,
            undesired_contact=(s['bad_contact']>P['undesired_contact_force_threshold']).float())
        assert terms.keys()==WEIGHTS.keys(), (terms.keys()-WEIGHTS.keys(),WEIGHTS.keys()-terms.keys())
        return {k:torch.nan_to_num(v*WEIGHTS[k]*STEP_DT,nan=0,posinf=0,neginf=0) for k,v in terms.items()}
