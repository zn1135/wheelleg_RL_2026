"""SCUT direct-RL task port to mjlab Scene/Simulation and RSL-RL VecEnv.

A direct environment preserves V14's action/reward/reset/observation ordering.
It uses mjlab's batched MuJoCo Warp physics, entities and contact sensors.
"""
import numpy as np
import mujoco
import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, euler_xyz_from_quat
from mjlab.managers.event_manager import RecomputeLevel
from .asset import robot_spec, POSES
from .config import JOINTS, DT, DECIMATION, STEP_DT, manifest
from .recipe import Recipe

class ChuanliantuiEnv(VecEnv):
    num_actions=6
    max_episode_length=1000
    def __init__(self,num_envs=32,device='cuda:0',seed=42,randomize=True):
        self.num_envs=num_envs;self.device=torch.device(device);self.randomize=randomize
        torch.manual_seed(seed)
        self.cfg=dict(manifest(),num_envs=num_envs,seed=seed,randomize=randomize)
        robot=EntityCfg(spec_fn=robot_spec,init_state=EntityCfg.InitialStateCfg(pos=(0,0,.15),
            joint_pos={f'^{name}$':value for name,value in POSES['reset'].items()},joint_vel={'.*':0.0}),sort_actuators=False,
            articulation=EntityArticulationInfoCfg(actuators=(
                XmlActuatorCfg(target_names_expr=JOINTS),
                XmlActuatorCfg(target_names_expr=('left_gas_spring_tendon','right_gas_spring_tendon'),transmission_type='tendon'),)))
        sensor=ContactSensorCfg(name='contacts',primary=ContactMatch(mode='body',pattern='.*',entity='robot'),
                                fields=('force',),reduce='netforce',history_length=3)
        self.scene=Scene(SceneCfg(num_envs=num_envs,entities={'robot':robot},
            terrain=TerrainEntityCfg(terrain_type='plane'),sensors=(sensor,)),device=device)
        self.sim=Simulation(num_envs=num_envs,cfg=SimulationCfg(nconmax=128,njmax=512,
            mujoco=MujocoCfg(timestep=DT,cone='elliptic',impratio=10,iterations=100,tolerance=1e-10)),
            spec=self.scene.spec,device=device)
        self.scene.initialize(self.sim.mj_model,self.sim.model,self.sim.data)
        self.robot=self.scene['robot'];m=self.sim.mj_model;names=self.robot.joint_names
        self.jids=[names.index(n) for n in JOINTS]
        self.ctrl=[m.actuator('robot/'+n+'_motor').id for n in JOINTS]
        self.gas=[m.actuator('robot/'+side+'_gas_spring_motor').id for side in ('left','right')]
        self.wheel=[self.robot.body_names.index(n) for n in JOINTS[4:]]
        self.base=self.robot.body_names.index('base_link')
        self.base_id=m.body('robot/base_link').id
        cn=[name.split('/')[-1] for name in self.scene['contacts'].primary_names]
        self.contact_base=cn.index('base_link')
        self.contact_wheel=[cn.index(n) for n in JOINTS[4:]]
        self.contact_bad=[i for i,n in enumerate(cn) if n not in JOINTS[4:]]
        nominal=torch.tensor([POSES['nominal'][n] for n in JOINTS],device=device)
        self.recipe=Recipe(num_envs,device,nominal,randomize)
        self.episode_length_buf=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.duration=torch.zeros_like(self.episode_length_buf)
        self.acc=torch.zeros(num_envs,6,device=device)
        self.common_step_counter=0
        self.last_numerical_reset=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.last_gain_sample=torch.full_like(self.episode_length_buf,-720)
        self.push_timer=torch.zeros(num_envs,device=device);self.wrench_timer=torch.zeros_like(self.push_timer)
        self.external_wrench_b=torch.zeros(num_envs,6,device=device)
        self.term_sums={k:torch.zeros(num_envs,device=device) for k in __import__('ct_mjlab.config',fromlist=['WEIGHTS']).WEIGHTS}
        self._startup_randomization()
        self.reset(torch.arange(num_envs,device=device))
        self.obs=self._observations()
    @property
    def unwrapped(self):return self
    def _startup_randomization(self):
        if not self.randomize:return
        self.sim.expand_model_fields(('body_mass','body_inertia','body_ipos','geom_friction','geom_solref','dof_frictionloss','dof_damping'))
        m=self.sim.mj_model;model=self.sim.model;n=self.num_envs
        uniform=lambda shape,lo,hi:torch.empty(shape,device=self.device).uniform_(lo,hi)
        bids=self.robot.indexing.body_ids
        for i,name in enumerate(self.robot.body_names):
            lo,hi=(.9,1.3) if name=='base_link' else (.9,1.1)
            factor=uniform((n,),lo,hi);bid=int(bids[i]);model.body_mass[:,bid]*=factor
            # IsaacLab mass event recomputes inertia at the same density ratio.
            model.body_inertia[:,bid]*=factor[:,None]
            if name=='base_link':self.recipe.mass_scale[:,0]=factor
        model.body_ipos[:,self.base_id]+=uniform((n,3),-1,1)*torch.tensor([.04,.02,.02],device=self.device)
        # PhysX static/dynamic/restitution are sampled as 64 shared buckets.
        for body_names,ranges in [(('base_link',),((.01,.1),(.01,.1),(.02,.2))),
                                   (JOINTS[4:],((.5,1.2),(.4,1.),(.02,.2)))]:
            table=torch.stack([uniform((64,),*r) for r in ranges],-1)
            table[:,1]=torch.minimum(table[:,0],table[:,1])
            for j,name in enumerate(body_names):
                samples=table[torch.randint(0,64,(n,),device=self.device)]
                if name!='base_link':self.recipe.material[:,j*3:(j+1)*3]=samples
                bid=m.body('robot/'+name).id
                for gid in np.where(m.geom_bodyid==bid)[0]:
                    model.geom_friction[:,gid,0]=samples[:,1]
                    # Source ground e=0 with multiply combine: effective e=0.
                    # Keep sampled e in critic material slots; critically damp
                    # ground contacts instead of making sampled e cause bounce.
                    model.geom_solref[:,gid,0]=.02;model.geom_solref[:,gid,1]=1.
        for name in self.robot.joint_names:
            jid=m.joint('robot/'+name).id;vid=m.jnt_dofadr[jid]
            if name in JOINTS[:4]:fr,vis=(.25,1.),(.05,.2)
            elif name in JOINTS[4:]:fr,vis=(.05,.25),(0,.01)
            else:fr,vis=(.05,.1),(.01,.025)
            # PhysX static and dynamic joint friction are equal in the recipe.
            model.dof_frictionloss[:,vid]+=uniform((n,),*fr)
            model.dof_damping[:,vid]+=uniform((n,),*vis)
        self.sim.recompute_constants(RecomputeLevel.set_const)
    def reset(self,ids):
        self.sim.reset(ids);self.scene.reset(ids)
        d=self.robot.data
        root=d.default_root_state[ids].clone()
        root[:,:3]+=self.scene.env_origins[ids]
        d.write_root_state(root,ids)
        d.write_joint_state(d.default_joint_pos[ids],d.default_joint_vel[ids],env_ids=ids)
        self.episode_length_buf[ids]=0;self.duration[ids]=0;self.recipe.reset(ids)
        self.acc[ids]=0
        self.sim.data.xfrc_applied[ids]=0
        self.external_wrench_b[ids]=0
        self.push_timer[ids]=torch.empty(len(ids),device=self.device).uniform_(5,10)
        self.wrench_timer[ids]=torch.empty(len(ids),device=self.device).uniform_(5,10)
        eligible=ids[self.common_step_counter-self.last_gain_sample[ids]>=720]
        if self.randomize and len(eligible):
            self.recipe.kp[eligible,:4]=10*torch.empty(len(eligible),4,device=self.device).uniform_(.75,1.25)
            self.recipe.kd[eligible]=torch.empty(len(eligible),6,device=self.device).uniform_(.75,1.25)*torch.tensor([1]*4+[.1]*2,device=self.device)
            self.recipe.effort_scale[eligible,:4]=torch.empty(len(eligible),4,device=self.device).uniform_(.8,1.1)
            self.recipe.effort_scale[eligible,4:]=torch.empty(len(eligible),2,device=self.device).uniform_(.9,1.1)
            self.last_gain_sample[eligible]=self.common_step_counter
        for value in self.term_sums.values():value[ids]=0
        self.sim.forward()
    def state(self):
        d=self.robot.data;q=d.root_link_quat_w
        rpy=euler_xyz_from_quat(q)
        relative=d.body_link_pos_w[:,self.wheel]-d.root_link_pos_w[:,None]
        wpos=quat_apply_inverse(q[:,None].expand(-1,2,-1),relative)
        relative_vel=d.body_com_lin_vel_w[:,self.wheel]-d.root_com_lin_vel_w[:,None]
        wvel=quat_apply_inverse(q[:,None].expand(-1,2,-1),relative_vel);wvel[:,:,1]=0
        contacts=self.scene['contacts'].data
        force=torch.linalg.vector_norm(contacts.force,dim=-1)
        history=contacts.force_history
        peaks=torch.linalg.vector_norm(history,dim=-1).amax(2) if history is not None else force
        return dict(pos=d.joint_pos[:,self.jids],vel=d.joint_vel[:,self.jids],acc=self.acc,
            torque=self.sim.data.actuator_force[:,self.ctrl],lin=d.root_com_lin_vel_b,
            ang=d.root_link_ang_vel_b,gravity=d.projected_gravity_b,height=d.root_link_pos_w[:,2],
            pitch=(rpy[1]+torch.pi)%(2*torch.pi)-torch.pi,roll=(rpy[0]+torch.pi)%(2*torch.pi)-torch.pi,
            wheel_pos=wpos,wheel_vel=wvel,wheel_contact=peaks[:,self.contact_wheel],
            bad_contact=peaks[:,self.contact_bad].amax(-1),base_contact=force[:,self.contact_base])
    def _observations(self,advance=True):return TensorDict(self.recipe.observations(self.state(),advance),batch_size=[self.num_envs])
    def get_observations(self):return self.obs
    def _events(self):
        if not self.randomize:return
        self.push_timer-=STEP_DT;self.wrench_timer-=STEP_DT
        ids=(self.push_timer<=0).nonzero().flatten()
        if len(ids):
            velocity=self.robot.data.root_com_vel_w[ids].clone()
            velocity[:,:2]=torch.empty(len(ids),2,device=self.device).uniform_(-.25,.25)
            self.robot.write_root_com_velocity_to_sim(velocity,env_ids=ids)
            self.push_timer[ids]=torch.empty(len(ids),device=self.device).uniform_(5,10)
        ids=(self.wrench_timer<=0).nonzero().flatten()
        if len(ids):
            wrench=torch.empty(len(ids),6,device=self.device).uniform_(-1,1)*torch.tensor([10]*3+[1]*3,device=self.device)
            self.external_wrench_b[ids]=wrench
            self.wrench_timer[ids]=torch.empty(len(ids),device=self.device).uniform_(5,10)
    def _write_external_wrench(self):
        # SCUT set_external_force_and_torque defaults to body-local vectors.
        q=self.robot.data.root_link_quat_w
        self.sim.data.xfrc_applied[:,self.base_id,:3]=quat_apply(q,self.external_wrench_b[:,:3])
        self.sim.data.xfrc_applied[:,self.base_id,3:]=quat_apply(q,self.external_wrench_b[:,3:])
    def step(self,actions):
        self.recipe.actions.copy_(actions)
        for _ in range(DECIMATION):
            d=self.robot.data
            previous_vel=d.joint_vel[:,self.jids].clone()
            self.sim.data.ctrl[:,self.ctrl]=self.recipe.torque(d.joint_pos[:,self.jids],d.joint_vel[:,self.jids],self.episode_length_buf)
            self.sim.data.ctrl[:,self.gas]=150
            self._write_external_wrench()
            self.sim.step();self.scene.update(DT)
            self.acc=(d.joint_vel[:,self.jids]-previous_vel)/DT
            self.sim.forward()
            self.recipe.observe_substep(self.state())
        self.sim.forward();self.episode_length_buf+=1;self.common_step_counter+=1
        s=self.state()
        immediate=(~torch.isfinite(torch.cat((s['vel'],s['lin'],s['ang']),-1)).all(-1)
                   |(s['vel'].abs().amax(-1)>500)|(s['lin'].abs().amax(-1)>100)|(s['ang'].abs().amax(-1)>200))
        if self.common_step_counter>10:
            immediate|=self.recipe.bad_obs|(self.recipe.raw_obs_max>120)
        self.last_numerical_reset.copy_(immediate)
        ordinary=(s['base_contact']>1)|(s['roll'].abs()>torch.deg2rad(torch.tensor(40.,device=self.device)))|(s['pitch'].abs()>torch.deg2rad(torch.tensor(40.,device=self.device)))
        self.duration=torch.where(ordinary,self.duration+1,0)
        terminated=(self.duration>=20)|immediate;timeout=self.episode_length_buf>=self.max_episode_length-1
        terms=self.recipe.reward_terms(s,terminated)
        reward=sum(terms.values());reward=torch.where(immediate,0,reward)
        for k,v in terms.items():self.term_sums[k]+=v
        done=terminated|timeout;ids=done.nonzero().flatten()
        extras={'time_outs':timeout}
        if len(ids):
            extras['log']={f'Reward/{k}':v[ids].mean()/20 for k,v in self.term_sums.items()}
            self.reset(ids)
            self.recipe.observe_substep(self.state(),ids)
        self._events();self.sim.forward();self.obs=self._observations(advance=False)
        return self.obs,reward,done.long(),extras
    def close(self):pass
