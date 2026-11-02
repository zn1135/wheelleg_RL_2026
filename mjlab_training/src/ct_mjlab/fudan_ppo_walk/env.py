"""mjlab physics with a separate Fudan task loop, no legacy command or reward path."""
from dataclasses import dataclass, field, replace
import math
import re
import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from ct_mjlab.envs.chuanliantui.huananhu_env_cfg import make_huananhu_env_cfg
from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdAction, HuananhuPdActionCfg
from ct_mjlab.envs.chuanliantui.pd_action import ChuanliantuiPdAction
from .reset_geometry import ResetGeometry, nominal_reset_pose
from .contract import MOTORS, LEGS, WEIGHTS, POLICY_DT, RESET_CONTRACT, RESET_V2_CONTRACT
from .disturbances import DisturbanceConfig,DisturbanceState,SENSOR_SLOTS,from_task_config


@dataclass(kw_only=True)
class PhysicalActionCfg(HuananhuPdActionCfg):
    class_type: type = field(default_factory=lambda: PhysicalAction)
    reference_randomization: bool = True
    disturbances: DisturbanceConfig = field(default_factory=from_task_config)


class PhysicalAction(HuananhuPdAction):
    def __init__(self, cfg, env):
        super().__init__(replace(cfg,motor_randomization=False,randomize_delays=False,
            action_delay_s=0.,nominal_leg_offset_range=(0.,0.),ground_start_hold_s=0.),env)
        self.cfg=cfg
        if cfg.reference_randomization:
            self.kp_scale.uniform_(.95,1.05)
            self.kd_scale.uniform_(.95,1.05)
            self.motor_strength.uniform_(.95,1.05)
            self.nominal_offset.uniform_(-.03,.03)
            self._nominal_pos.add_(self.nominal_offset)
        self.last_sample_q = env.sim.data.qpos[:,self._motor_q_adr].clone()
        self.sample_velocity = torch.zeros_like(self.last_sample_q)
        self.substep_rows = []
        self.record_substeps = False
        self.rebuild_reset_poses()
        self.disturbances=DisturbanceState(cfg.disturbances,self.num_envs,self.device,
            env.cfg.seed if env.cfg.seed is not None else 1)
        self.disturbances.reset(slice(None),self.sensor_frame())
        self._base_id=int(env.sim.mj_model.body('robot/base_link').id)
        self.push_force_scale=torch.zeros(self.num_envs,device=self.device)

    def sensor_frame(self):
        q=self._env.sim.data.qpos
        gravity=quat_apply_inverse(q[:,3:7],q.new_tensor([0.,0.,-1.]).expand(self.num_envs,-1))
        return torch.cat((self._env.sim.data.qvel[:,3:6]*.25,gravity,
            (q[:,self._motor_q_adr]-self.nominal_positions)[:,LEGS],self.sample_velocity*.05),-1)

    def apply_actions(self):
        if self.cfg.disturbances.action_delay:
            self._control_actions.copy_(self.disturbances.action(self._raw_actions))
        super().apply_actions()
        if self.cfg.disturbances.pushes:
            force=self.disturbances.force(self._env.sim.data.qpos[:,3:7],self.push_force_scale)
            # 世界坐标系COM外力，每次仅保持一个2ms物理拍；下一拍明确清零。
            self._env.sim.data.xfrc_applied[:,self._base_id,:3]=force
            self._env.sim.data.xfrc_applied[:,self._base_id,3:]=0.

    def rebuild_reset_poses(self):
        # Parameters are sampled once per environment; reuse the legal pose at later resets.
        solver=ResetGeometry()
        poses=[solver.solve(row) for row in self.nominal_positions.detach().cpu().numpy()]
        names=self._entity.joint_names
        self.reset_joint_positions=torch.tensor([[p['joint_positions'][name] for name in names] for p in poses],
            dtype=self._nominal_pos.dtype,device=self.device)
        self.reset_root_height=torch.tensor([p['height'] for p in poses],dtype=self._nominal_pos.dtype,device=self.device)

    def _ground_hold(self):
        return torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)

    def process_actions(self, actions):
        self._raw_actions.copy_(actions.clamp(-100.,100.))
        self._control_actions.copy_(self._raw_actions)
        self.substep_rows.clear()

    def reset(self, env_ids=None):
        ChuanliantuiPdAction.reset(self,env_ids)
        ids=slice(None) if env_ids is None else env_ids
        self._motor_response_torque[ids]=0.
        self.action_delay_steps[ids]=0
        self.wheel_action_delay_steps[ids]=0
        if hasattr(self,'last_sample_q'):
            self.last_sample_q[ids]=self._env.sim.data.qpos[ids][:,self._motor_q_adr]
            self.sample_velocity[ids]=0.
        if hasattr(self,'disturbances'):
            self.disturbances.reset(ids,self.sensor_frame())
            if self.cfg.disturbances.pushes:self._env.sim.data.xfrc_applied[ids,self._base_id]=0.


def current_sensors(env):
    return env.scene['robot'].data.projected_gravity_b


def sample_substep(env):
    action=env.action_manager.get_term('pd')
    q=env.sim.data.qpos[:,action._motor_q_adr]
    action.sample_velocity.copy_((q-action.last_sample_q+math.pi).remainder(2*math.pi).sub(math.pi)/env.physics_dt)
    action.last_sample_q.copy_(q)
    if action.cfg.disturbances.sensor_delay:action.disturbances.sample(action.sensor_frame())
    if action.record_substeps:
        from .measurement import gpu_row
        action.substep_rows.append(gpu_row(env).clone())
    return action.sample_velocity.abs().amax(-1)


def reset_pose(env,env_ids):
    robot=env.scene['robot']
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    root=robot.data.default_root_state[env_ids].clone()
    action=env.action_manager.get_term('pd')
    root[:,:3]=env.scene.env_origins[env_ids]
    root[:,2]+=action.reset_root_height[env_ids]
    root[:,7:13]=torch.empty(len(env_ids),6,device=env.device).uniform_(-.5,.5)
    robot.write_root_state_to_sim(root,env_ids=env_ids)
    robot.write_joint_state_to_sim(action.reset_joint_positions[env_ids],
        torch.zeros_like(robot.data.default_joint_vel[env_ids]),env_ids=env_ids)


@requires_model_fields('body_mass','body_inertia','body_ipos','geom_friction','geom_solref',recompute=RecomputeLevel.set_const)
def randomize(env,env_ids):
    robot=env.scene['robot'];model=env.sim.mj_model;n=env.num_envs;device=env.device
    ids=robot.indexing.body_ids;base=int(ids[robot.body_names.index('base_link')])
    mass=env.sim.get_default_field('body_mass')[ids].expand(n,-1).clone()
    local=robot.body_names.index('base_link')
    add=torch.empty(n,device=device).uniform_(-1.,2.)
    mass[:,local]+=add
    scale=torch.empty_like(mass).uniform_(.9,1.1)
    com=torch.empty(n,3,device=device).uniform_(-.02,.02)
    buckets=torch.empty(64,device=device).uniform_(.6,1.4)
    friction=buckets[torch.randint(64,(n,),device=device)]
    restitution=torch.empty(n,device=device).uniform_(.6,1.)
    env.sim.model.body_mass[:,ids]=mass*scale
    env.sim.model.body_inertia[:,ids]=env.sim.get_default_field('body_inertia')[ids]*scale[...,None]
    env.sim.model.body_ipos[:,base]=env.sim.get_default_field('body_ipos')[base]+com
    geoms=torch.tensor([i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in set(ids.tolist())],device=device)
    env.sim.model.geom_friction[:,geoms,0]=((friction+.5)*.5)[:,None]
    # PhysX恢复系数映射为软接触阻尼；只近似碰撞能量耗散，非同一接触模型。
    effective=(restitution+.5)*.5
    log_e=effective.log();zeta=-log_e/(math.pi**2+log_e.square()).sqrt()
    env.sim.model.geom_solref[:,geoms]=torch.stack((torch.full_like(zeta,-10000.),-200*zeta),-1)[:,None,:]
    env._fudan_parameters=dict(base_mass=mass[:,local],com=com,friction=friction,restitution=restitution)


def make_env_cfg(*,play=False,disturbances=None):
    cfg=make_huananhu_env_cfg(play=True)
    cfg.scene.num_envs=20 if play else 512
    cfg.sim.mujoco.timestep=.002;cfg.decimation=5;cfg.episode_length_s=20.
    cfg.commands={};cfg.curriculum={};cfg.rewards={};cfg.terminations={}
    # Reuse only geometry, sensors and physical actuation; replace every task hook.
    cfg.events={'reset_pose':EventTermCfg(func=reset_pose,mode='reset')}
    if not play:cfg.events['fudan_randomization']=EventTermCfg(func=randomize,mode='startup')
    original=cfg.actions['pd']
    cfg.actions={'pd':PhysicalActionCfg(entity_name='robot',nominal_leg_positions=original.nominal_leg_positions,
        wheel_stiffness=.1,wheel_damping=0.,motor_randomization=False,randomize_delays=False,
        action_delay_s=0.,ground_start_hold_s=0.,spring_force_bias_range=(0.,0.),
        standup_landing_gate=False,standup_pushes=False,reference_randomization=not play,
        disturbances=disturbances or from_task_config())}
    cfg.observations={'actor':ObservationGroupCfg(terms={'sensors':ObservationTermCfg(func=current_sensors)},
        concatenate_terms=True,enable_corruption=False,nan_policy='error')}
    cfg.metrics={'motor_sampling':MetricsTermCfg(func=sample_substep,per_substep=True,reduce='last')}
    pose=nominal_reset_pose()
    initial=cfg.scene.entities['robot'].init_state
    initial.pos=(0.,0.,pose['height'])
    initial.joint_pos={f'^{re.escape(name)}$':value for name,value in pose['joint_positions'].items()}
    return cfg


class FudanEnv:
    def __init__(self,num_envs=512,seed=1,device='cuda:0',randomized=True,physics_env=None,reset_contract=RESET_CONTRACT,disturbances=None):
        if reset_contract not in (RESET_CONTRACT,RESET_V2_CONTRACT):
            raise ValueError('Unsupported PD-centred reset/history contract')
        self.reset_contract=reset_contract
        from mjlab.envs import ManagerBasedRlEnv
        cfg=make_env_cfg(play=not randomized,disturbances=disturbances);cfg.scene.num_envs=num_envs;cfg.seed=seed
        self.env=ManagerBasedRlEnv(cfg,device=device) if physics_env is None else physics_env
        num_envs,device=self.env.num_envs,self.env.device
        self.num_envs,self.device=num_envs,device
        self.robot=self.env.scene['robot'];self.action=self.env.action_manager.get_term('pd')
        self.disturbance_config=self.action.cfg.disturbances
        if disturbances is not None and self.disturbance_config!=disturbances:
            raise ValueError('Wrapped physics environment has a different disturbance configuration')
        self.update_push_scale()
        self.randomized=randomized
        self.commands=torch.zeros(num_envs,3,device=device)
        self.history=torch.zeros(num_envs,125,device=device)
        self._history_needs_fill=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.previous=torch.zeros(num_envs,6,2,device=device)
        self.previous_vel=torch.zeros(num_envs,6,device=device)
        self.previous_position=torch.zeros(num_envs,3,device=device)
        self.fail_ticks=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.sums=torch.zeros(num_envs,2,device=device)
        self.speed_limit=2.
        self.episode_steps=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.total_steps=0
        ids,_=self.robot.find_joints(('lf0','lf1','rf0','rf1'),preserve_order=True)
        self.front_q=self.robot.indexing.joint_q_adr[ids]
        model=self.env.sim.mj_model
        joint_ids=[model.joint(f'robot/{name}').id for name in ('lf0','lf1','rf0','rf1')]
        limits=torch.tensor(model.jnt_range[joint_ids],device=device,dtype=torch.float32)
        mid=limits.mean(-1);half=(limits[:,1]-limits[:,0])*.485
        self.front_limits=torch.stack((mid-half,mid+half),-1)
        self.hip_ids=torch.tensor([model.joint('robot/'+n).id for n in ('lf0','rf0')],device=device)
        self.wheel_ids=torch.tensor([model.body('robot/'+n).id for n in ('lfwheel','rfwheel')],device=device)
        self.previous_position.copy_(self.robot.data.root_link_pos_w)
        self._update_state()

    def update_push_scale(self):
        # 参考以批量base质量均值将速度尺度转为单拍冲量，不套用旧任务推力经验。
        if self.disturbance_config.pushes:
            mass=self.env.sim.model.body_mass[...,self.action._base_id].mean()
            self.action.push_force_scale.fill_(float(mass)*self.disturbance_config.max_push_vel/.002)

    def resample(self,ids):
        self.commands[ids,0]=torch.empty(len(ids),device=self.device).uniform_(-self.speed_limit,self.speed_limit)
        self.commands[ids,1]=torch.empty(len(ids),device=self.device).uniform_(-2.,2.)
        self.commands[ids,2]=torch.empty(len(ids),device=self.device).uniform_(.22,.30)

    def reset_idx(self,ids):
        if len(ids)==0:return
        if self.total_steps%2000==0 and (self.sums[ids,0].mean()/2000>.7*.01
                and self.sums[ids,1].mean()/2000>.7*.01*.8):
            self.speed_limit=min(2.5,self.speed_limit+.1)
        self.env.reset(env_ids=ids)
        self.resample(ids)
        self.previous[ids]=0.;self.previous_vel[ids]=0.;self.fail_ticks[ids]=0;self.sums[ids]=0.
        self.episode_steps[ids]=0
        self.previous_position[ids]=self.robot.data.root_link_pos_w[ids]
        self._update_state(ids)
        if hasattr(self.action,'disturbances'):
            self.action.disturbances.sensors[:,ids]=self.action.sensor_frame()[ids]
        if self.reset_contract==RESET_V2_CONTRACT:
            self.history[ids]=self.frame()[ids].repeat(1,5)
        else:
            # 等本拍唯一一次观测采样后，选中的环境用同一带噪/限幅帧填满。
            self._history_needs_fill[ids]=True

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs,device=self.device))
        # Reference BaseTask.reset advances one zero-action 10ms step before random episode phases.
        obs,critic,_,_,_,history=self.step(torch.zeros(self.num_envs,6,device=self.device))
        return obs,critic,history

    def _update_state(self,ids=None):
        d=self.robot.data
        position=d.root_link_pos_w
        velocity=self.action.sample_velocity
        values=dict(position=position,
            linear=quat_apply_inverse(d.root_link_quat_w,(position-self.previous_position)/POLICY_DT),
            angular=d.root_link_ang_vel_b,gravity=d.projected_gravity_b,
            height=position[:,2]-self.env.scene.env_origins[:,2],velocity=velocity,
            acceleration=(self.previous_vel-velocity)/POLICY_DT)
        for name,value in values.items():
            if ids is None or not hasattr(self,name):setattr(self,name,value.clone())
            else:getattr(self,name)[ids]=value[ids]

    def frame(self):
        q=self.env.sim.data.qpos[:,self.action._motor_q_adr]
        return torch.cat((self.angular*.25,self.gravity,self.commands*torch.tensor([2.,.25,5.],device=self.device),
            (q-self.action.nominal_positions)[:,LEGS],self.velocity*.05,self.action.raw_action),-1)

    def observations(self):
        clean=self.frame()
        p=getattr(self.env,'_fudan_parameters',None)
        if p is None:
            mass=torch.zeros(self.num_envs,1,device=self.device);com=torch.zeros(self.num_envs,3,device=self.device)
            friction=torch.full_like(mass,.5);restitution=torch.full_like(mass,.5)
        else:
            mass=(p['base_mass']-p['base_mass'].mean())[:,None];com=p['com']
            friction=p['friction'][:,None];restitution=p['restitution'][:,None]
        critic=torch.cat((self.linear*2.,clean,self.previous[:,:,0],self.previous[:,:,1],self.acceleration*.0025,
            ((self.height-.5).clamp(-1.,1.)*5.)[:,None].expand(-1,77),self.action.torques*.05,
            mass,com,self.action.nominal_offset,friction,restitution),-1).clamp(-100.,100.)
        obs=clean.clone()
        if getattr(self,'disturbance_config',DisturbanceConfig()).sensor_delay:
            obs[:,SENSOR_SLOTS]=self.action.disturbances.observed()
        if self.randomized:
            scales=obs.new_tensor([.05]*6+[0.]*3+[.02]*4+[.075]*6+[0.]*6)
            noise=(torch.rand_like(obs)*2-1)*scales
            if hasattr(self,'noise_mask'):noise*=self.noise_mask[:,None]
            obs+=noise
        clipped=obs.clamp(-100.,100.)
        self._history_frame=obs if self.reset_contract==RESET_V2_CONTRACT else clipped
        return clipped,critic

    def reward(self):
        error=(self.commands[:,0]-self.linear[:,0]).square()
        yaw_error=(self.commands[:,1]-self.angular[:,2]).square()
        delta=self.env.sim.data.xpos[:,self.wheel_ids]-self.env.sim.data.xanchor[:,self.hip_ids]
        quaternion=self.robot.data.root_link_quat_w[:,None,:].expand(-1,2,-1)
        local=quat_apply_inverse(quaternion.reshape(-1,4),delta.reshape(-1,3)).reshape(-1,2,3)
        theta=torch.atan2(local[:,:,0],-local[:,:,2])
        q=self.env.sim.data.qpos[:,self.front_q]
        limit=((self.front_limits[:,0]-q).clamp_min(0.)+(q-self.front_limits[:,1]).clamp_min(0.)).sum(-1)
        action=self.action.raw_action
        raw=dict(tracking_lin_vel=torch.exp(-error/.25),tracking_lin_vel_enhance=torch.exp(-error/2.5)-1.,
            tracking_ang_vel=torch.exp(-yaw_error/.25),tracking_ang_vel_enhance=torch.exp(-yaw_error/2.5)-1.,
            base_height=torch.exp(-(self.height-self.commands[:,2]).square()/.001),
            nominal_state=(theta[:,0]-theta[:,1]).square(),lin_vel_z=self.linear[:,2].square(),
            ang_vel_xy=self.angular[:,:2].square().sum(-1),orientation=self.gravity[:,:2].square().sum(-1),
            dof_vel=self.velocity[:,LEGS].square().sum(-1),dof_acc=self.acceleration.square().sum(-1),
            torques=self.action.torques.square().sum(-1),action_rate=(action-self.previous[:,:,0]).square().sum(-1),
            action_smooth=(action[:,LEGS]-2*self.previous[:,LEGS,0]+self.previous[:,LEGS,1]).square().sum(-1),
            collision=torch.zeros_like(error),dof_pos_limits=limit)
        terms={k:(v*WEIGHTS[k]*POLICY_DT).clamp(-POLICY_DT,POLICY_DT) for k,v in raw.items()}
        self.sums[:,0]+=terms['tracking_lin_vel'];self.sums[:,1]+=terms['tracking_ang_vel']
        return torch.stack(list(terms.values())).sum(0),terms

    def step(self,actions,*,auto_reset=True,resample=True):
        if not torch.isfinite(actions).all():raise FloatingPointError('Non-finite policy action')
        self.env.step(actions)
        self.total_steps+=1;self.episode_steps+=1
        self._update_state()
        if not torch.isfinite(self.env.sim.data.qpos).all() or not torch.isfinite(self.env.sim.data.qvel).all():
            raise FloatingPointError('Non-finite physical state')
        if resample:self.resample((self.episode_steps.remainder(500)==0).nonzero().flatten())
        bad=self.gravity[:,2]>-.1
        self.fail_ticks=torch.where(bad,self.fail_ticks+1,0)
        timeout=self.episode_steps>2000
        done=(self.fail_ticks>100)|timeout
        reward,terms=self.reward()
        if auto_reset and done.any():self.reset_idx(done.nonzero().flatten())
        obs,critic=self.observations()
        self.history=torch.cat((self.history[:,25:],self._history_frame),-1)
        if self.reset_contract==RESET_CONTRACT:
            self.history[self._history_needs_fill]=self._history_frame[self._history_needs_fill].repeat(1,5)
            self._history_needs_fill.zero_()
        self.previous[:,:,1]=self.previous[:,:,0];self.previous[:,:,0]=self.action.raw_action
        self.previous_vel.copy_(self.velocity);self.previous_position.copy_(self.position)
        return obs,critic,reward,done,{'time_outs':timeout,'terms':terms},self.history

    def close(self):
        self.env.close()
