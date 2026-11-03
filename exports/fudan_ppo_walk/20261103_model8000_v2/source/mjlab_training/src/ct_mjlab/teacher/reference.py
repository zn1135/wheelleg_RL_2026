"""Read-only reference observer; teacher labels and student inputs share one clock."""
import math

import torch
from .controller import LqrVmcController
from .env import TeacherAction, TeacherActionCfg, body_angles
from .standup import DONE


def controller_tensors(controller):
    for value in vars(controller).values():
        if isinstance(value, torch.Tensor):
            yield value
        elif hasattr(value, '__dict__'):
            yield from controller_tensors(value)


class ReferenceObserver:
    def __init__(self, env, *, takeover_s=None):
        if abs(env.step_dt-.01)>1e-9 or abs(env.physics_dt-.002)>1e-9:
            raise ValueError('Reference observer requires 100Hz observations and 500Hz physical control')
        self.env=env;self.takeover_s=takeover_s
        action=env.action_manager.get_term('pd')
        self.teacher=getattr(action,'teacher',None)
        if self.teacher is None:
            self.teacher=TeacherAction(TeacherActionCfg(entity_name='robot',initial='ground',
                velocity_integral=True,ground_start_hold_s=.2),env)
            self.teacher.controller=LqrVmcController(env.num_envs,env.device,velocity_integral=True,dt=env.step_dt)
        n=env.num_envs;device=env.device
        self.hold=torch.zeros(n,dtype=torch.long,device=device)
        self.synchronized=torch.zeros(n,dtype=torch.bool,device=device)
        self.takeover_done=torch.zeros_like(self.synchronized)
        self.valid=torch.zeros_like(self.synchronized);self.step=-1
        self.error=torch.zeros(n,1,device=device)
        self.events=torch.zeros(n,dtype=torch.long,device=device)
        # before, estimated_velocity_error, active, reset_before, reset_after, after, target
        self.state=torch.zeros(n,7,device=device)
        self.weight=float(env.sim.get_default_field('body_mass')[env.scene['robot'].indexing.body_ids].sum())*9.81
        self.reset(torch.arange(n,device=device))

    def reset(self, ids):
        self.teacher.reset(ids)
        self.teacher.command[ids]=self.teacher.command.new_tensor([0.,0.,.30])
        for value in (self.hold,self.synchronized,self.takeover_done,self.valid,self.error,self.events,self.state):
            value[ids]=0

    def sample_substep(self):
        robot=self.env.scene['robot'].data;g=robot.projected_gravity_b
        tilt=torch.maximum(torch.atan2(-g[:,1],-g[:,2]).abs(),torch.asin(g[:,0].clamp(-1,1)).abs())
        base=self.env.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
        wheel=-self.env.scene['wheel_contact'].data.force[:,:,2]
        weight=self.weight
        balanced=((tilt<=math.radians(35)) & (base<=1.) &
            (wheel>=max(.1*weight,5.)).all(-1) & (wheel.sum(-1)>=.8*weight) &
            (wheel.sum(-1)<=1.2*weight) & (robot.root_link_ang_vel_b[:,:2].norm(dim=-1)<=.5))
        self.hold.copy_(torch.where(balanced,self.hold+1,0))

    def update(self):
        tick=self.env.common_step_counter
        if tick!=self.step:
            self.valid.zero_();self.step=tick
        pending=~self.valid
        if not pending.any():
            return self
        # 同一全局拍只局部复位时，不让整批计算推进其他环境的滤波和起立历史。
        keep=~pending
        buffers=list(controller_tensors(self.teacher.controller))+[
            getattr(self.teacher,key) for key in ('command','torques','h7_torques','previous_velocity',
                'velocity_ready','_control_ticks','control_active')]
        restored=[(value,value[keep].clone()) for value in buffers if value.ndim and value.shape[0]==self.env.num_envs] if keep.any() else []
        c=self.teacher.controller;before=c.pos.clone();old_air=c.air.flight.clone()
        supported=pending & ~self.synchronized & (self.hold>=500)
        # 已正常交接的教师不能额外重锁；只修复旁路估计器落后的阶段。
        stable=supported & (c.standup.phase!=DONE)
        takeover=torch.zeros_like(stable)
        if self.takeover_s is not None:
            takeover=pending & ~self.takeover_done & (self.hold>=500) & (
                self.env.episode_length_buf==round(self.takeover_s/self.env.step_dt))
        anchor=stable | takeover
        if anchor.any():
            c.standup.reset(anchor,standing=True);c.pos[anchor]=0.
            c.yaw_target[anchor]=body_angles(self.env.scene['robot'].data.root_link_quat_w)[2][anchor]
        self.synchronized|=supported;self.takeover_done|=takeover
        phase=c.standup.phase.clone()
        command=self.env.command_manager.get_term('standup').command
        self.teacher.command[:,:2]=command[:,:2]
        # 当前零速起立任务的0.33648m机身目标对应0.30m腿长。
        self.teacher.command[:,2]=(command[:,2]-.03648).clamp(.15,.31)
        self.teacher.compute_torques(dt=self.env.step_dt)
        changed_air=old_air!=c.air.flight
        handed_over=(phase!=DONE)&(c.standup.phase==DONE)
        hold=~self.teacher.control_active
        active=(phase==DONE)&~c.air.flight&c.ready&~hold
        state=torch.stack((before,c.x[:,1]-self.teacher.command[:,0],active.float(),(anchor|changed_air).float(),
            (handed_over|hold).float(),c.pos,c.target[:,0]),-1)
        events=stable.long()+2*takeover.long()+4*handed_over.long()+8*changed_air.long()+16*hold.long()
        self.state[pending]=state[pending];self.events[pending]=events[pending]
        self.error[pending]=(c.target[:,0]-c.x[:,0])[pending,None]
        for value,snapshot in restored:value[keep]=snapshot
        self.valid[pending]=True
        return self


def get_reference(env, *, takeover_s=None):
    if not hasattr(env,'_bc_reference_observer'):
        env._bc_reference_observer=ReferenceObserver(env,takeover_s=takeover_s)
    observer=env._bc_reference_observer
    if observer.takeover_s!=takeover_s:
        raise ValueError('Actor and critic must use the same reference reset schedule')
    return observer
