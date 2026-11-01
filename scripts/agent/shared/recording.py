"""SuperMimic共用500Hz物理采集；不提供独立行走PPO诊断入口。"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sample_physics(env):
    recorder = getattr(env, '_walk_diagnosis', None)
    if recorder is not None:
        recorder.sample()
    return torch.zeros(env.num_envs, device=env.device)


class Recorder:
    def __init__(self, raw, steps):
        self.raw, self.pd = raw, raw.action_manager.get_term('pd')
        self.robot = raw.scene['robot']
        self.state = raw.command_manager.get_term('standup')
        self.live = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
        self.count, self.layout = 0, {}
        self.trace = None
        self.steps = steps
        self.original_apply = self.pd.apply_actions
        self.hips = self.robot.find_bodies(('lf0','rf0'), preserve_order=True)[0]
        self.wheels = self.robot.find_bodies(('lfwheel','rfwheel'), preserve_order=True)[0]
        names = ('lf0','lf00','rf0','rf00','lf1','rf1','lfwheel','rfwheel')
        ids = self.robot.find_joints(names, preserve_order=True)[0]
        self.qadr = self.robot.indexing.joint_q_adr[ids]
        self.vadr = self.robot.indexing.joint_v_adr[ids]
        self.joint_names = names

    def apply(self):
        pd = self.pd
        self.original_apply()
        delayed = pd._action_history[(pd._history_head-pd.action_delay_steps)
            % len(pd._action_history), pd._env_ids]
        legs = pd._leg_action_ids
        desired = pd._nominal_pos[:, legs] + pd.cfg.leg_position_scale*delayed[:, legs]
        target = desired.clamp(pd._reference_pos[:, legs]-3.14, pd._reference_pos[:, legs]+3.14)
        self.pending = dict(leg_target=target.clone(), leg_unclipped=desired.clone(),
            pd_torque=pd._torques.clone(), applied_command=self.state.command.clone())

    def sample(self):
        from mjlab.utils.lab_api.math import quat_apply_inverse
        pd, raw, robot = self.pd, self.raw, self.robot
        data = robot.data
        force = raw.scene['wheel_contact'].data.force
        support = -force[:,:,2]
        relative = data.body_link_pos_w[:,self.wheels]-data.body_link_pos_w[:,self.hips]
        quat = data.root_link_quat_w[:,None,:].expand(-1,2,-1)
        relative = quat_apply_inverse(quat.reshape(-1,4),relative.reshape(-1,3)).reshape(-1,2,3)
        root = raw.sim.data.qpos[:,robot.indexing.free_joint_q_adr]
        root_velocity = raw.sim.data.qvel[:,robot.indexing.free_joint_v_adr]
        qw,qx,qy,qz = root[:,3:7].unbind(-1)
        roll = torch.atan2(2.*(qw*qx+qy*qz), 1.-2.*(qx.square()+qy.square()))
        pitch = torch.asin((2.*(qw*qy-qz*qx)).clamp(-1.,1.))
        height = root[:,2]-raw.scene.env_origins[:,2]
        gz = (-1.+2.*(root[:,4].square()+root[:,5].square())).clamp(-1.,1.)
        base = raw.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
        fields = dict(valid=self.live[:,None].float(), height=height[:,None],
            tilt=torch.acos(-gz)[:,None].rad2deg(), base_contact=base[:,None],
            true_velocity=data.root_link_lin_vel_b, support=support,
            roll_pitch=torch.stack((roll,pitch),-1).rad2deg(),
            angular_velocity_xy=root_velocity[:,3:5],
            root_linear_velocity_w=root_velocity[:,:3],
            swing=torch.atan2(-relative[...,0],-relative[...,2]).rad2deg(),
            leg_length=relative.norm(dim=-1), joint_q=raw.sim.data.qpos[:,self.qadr],
            joint_dq=raw.sim.data.qvel[:,self.vadr],
            actual_torque=raw.sim.data.qfrc_actuator[:,pd._motor_v_adr],
            friction_ratio=force[:,:,:2].norm(dim=-1)/support.clamp_min(5.),
            down_ticks=self.state.walk_down_ticks[:,None].float(),
            encoder_velocity=self.encoder, true_velocity_before=self.true_before,
            raw_action=self.action, **self.pending)
        # 专用反馈对照按策略拍锁存输入，便于离线核对唯一干预。
        for name in ('policy_velocity','policy_wheel_velocity'):
            if hasattr(self,name):
                fields[name] = getattr(self,name)
        if self.trace is None:
            start = 0
            for name,value in fields.items():
                self.layout[name] = [start,start+value.shape[1]]
                start += value.shape[1]
            self.trace = torch.empty(self.steps,raw.num_envs,start,device=raw.device)
        self.trace[self.count] = torch.cat(tuple(fields.values()),-1)
        self.count += 1
