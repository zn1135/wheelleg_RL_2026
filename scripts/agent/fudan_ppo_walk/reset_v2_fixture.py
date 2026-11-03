"""A non-physical backend for exercising the actual Fudan reset/step/input code."""
from types import SimpleNamespace, MethodType
import numpy as np
import torch
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv,PhysicalAction,reset_pose
from ct_mjlab.envs.fudan_ppo_walk.reset_geometry import ResetGeometry
from ct_mjlab.fudan_ppo_walk.contract import MOTORS,RESET_V2_CONTRACT
from ct_mjlab.envs.fudan_ppo_walk.commands import initialize,LEGACY_COMMAND_SAMPLING


class ModelFacade:
    def __init__(self,model):self.model=model
    def __getattr__(self,name):return getattr(self.model,name)
    def body(self,name):return self.model.body(name.removeprefix('robot/') if isinstance(name,str) else name)
    def actuator(self,name):return self.model.actuator(name.removeprefix('robot/') if isinstance(name,str) else name)


def fixture(n=4,reset_contract=RESET_V2_CONTRACT):
    solver=ResetGeometry();model=solver.model
    pose=solver.solve([solver.baseline[name] for name in MOTORS])
    env=object.__new__(FudanEnv)
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, policy_dt, episode_steps
    env.cfg=FudanWalkCfg()
    env.dt=policy_dt(env.cfg)
    env.max_episode_length=episode_steps(env.cfg)
    env.reset_contract=reset_contract
    env.num_envs=n;env.device='cpu';env.total_steps=0;env.speed_limit=2.;env.randomized=True
    initialize(env,dict(LEGACY_COMMAND_SAMPLING))
    env.episode_steps=torch.zeros(n,dtype=torch.long);env.fail_ticks=torch.zeros(n,dtype=torch.long)
    env.commands=torch.zeros(n,3);env.history=torch.zeros(n,125)
    env._history_needs_fill=torch.zeros(n,dtype=torch.bool)
    env.previous=torch.zeros(n,6,2);env.previous_vel=torch.zeros(n,6);env.previous_position=torch.zeros(n,3);env.sums=torch.zeros(n,2)
    q=torch.tensor(np.repeat(model.qpos0[None],n,axis=0),dtype=torch.float32);v=torch.zeros(n,model.nv)
    origins=torch.zeros(n,3);origins[:,0]=torch.arange(n)*10.
    q[:,:3]=origins;q[:,2]+=pose['height'];q[:,3:7]=torch.tensor([1.,0.,0.,0.])
    names=solver.joint_names;joint_q=torch.tensor([solver.qaddr[name] for name in names]);joint_v=torch.tensor([int(model.joint(name).dofadr[0]) for name in names])
    joints=torch.tensor([[pose['joint_positions'][name] for name in names]]*n)
    q[:,joint_q]=joints
    mq=torch.tensor([solver.qaddr[name] for name in MOTORS]);mv=torch.tensor([int(model.joint(name).dofadr[0]) for name in MOTORS])
    nominal=torch.tensor([[pose['joint_positions'][name] for name in MOTORS]]*n)
    springs=['left_spring2_joint','right_spring2_joint']
    action=SimpleNamespace(nominal_positions=nominal,_nominal_pos=nominal,nominal_offset=torch.zeros(n,6),
        raw_action=torch.zeros(n,6),torques=torch.zeros(n,6),_control_actions=torch.zeros(n,6),_motor_q_adr=mq,_motor_v_adr=mv,
        last_sample_q=q[:,mq].clone(),sample_velocity=torch.zeros(n,6),
        _spring_ids=torch.tensor([model.actuator(name).id for name in ('left_gas_spring_motor','right_gas_spring_motor')]),
        _spring_q_adr=torch.tensor([solver.qaddr[name] for name in springs]),
        _spring_v_adr=torch.tensor([int(model.joint(name).dofadr[0]) for name in springs]),
        kp_scale=torch.ones(n,6),kd_scale=torch.ones(n,6),motor_strength=torch.ones(n,6),
        _wheel_response=torch.tensor([model.actuator(name+'_motor').gainprm[:3] for name in ('lfwheel','rfwheel')],dtype=torch.float32),
        reference_positions=torch.tensor(model.qpos0[mq.numpy()],dtype=torch.float32)[None],
        device='cpu',reset_joint_positions=joints,reset_root_height=torch.full((n,),pose['height']))
    root_default=torch.zeros(n,13);root_default[:,3]=1.;root_default[:,2]=pose['height']
    data=SimpleNamespace(default_root_state=root_default,default_joint_vel=torch.zeros(n,len(names)),
        root_link_pos_w=q[:,:3],root_link_quat_w=q[:,3:7],root_link_ang_vel_b=v[:,3:6],
        projected_gravity_b=torch.tensor([[0.,0.,-1.]]*n))
    def write_root(value,*,env_ids):
        q[env_ids,:7]=value[:,:7];v[env_ids,:6]=value[:,7:13]
    def write_joints(pos,vel,*,env_ids):
        q[env_ids[:,None],joint_q[None]]=pos;v[env_ids[:,None],joint_v[None]]=vel
    robot=SimpleNamespace(data=data,joint_names=names,body_names=[model.body(i).name for i in range(1,model.nbody)],
        indexing=SimpleNamespace(joint_q_adr=joint_q,body_ids=torch.arange(1,model.nbody)),
        write_root_state_to_sim=write_root,write_joint_state_to_sim=write_joints)
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import DisturbanceConfig
    action.cfg=SimpleNamespace(task_cfg=env.cfg,disturbances=DisturbanceConfig())
    action._entity=robot;action._raw_actions=action.raw_action
    action.rebuild_reset_poses=MethodType(PhysicalAction.rebuild_reset_poses,action)
    scene={'robot':robot};scene=Scene(scene);scene.env_origins=origins;scene.update=lambda **kw:None
    sim=SimpleNamespace(mj_model=ModelFacade(model),data=SimpleNamespace(qpos=q,qvel=v),
        model=SimpleNamespace(**{k:torch.as_tensor(np.repeat(np.asarray(getattr(model,k))[None],n,axis=0),dtype=torch.float32) for k in ('body_mass','body_inertia','body_ipos','geom_friction','geom_solref')}),
        recompute_constants=lambda *a:None,forward=lambda:None,sense=lambda:None)
    env.action=action;env.robot=robot;env.events=[];env.mock_physics_calls=0;env.reset_calls=[];env.resample_calls=[]
    env.env=SimpleNamespace(sim=sim,scene=scene,device='cpu',num_envs=n,
        cfg=SimpleNamespace(_fudan_task_cfg=env.cfg),
        episode_length_buf=torch.zeros(n,dtype=torch.long),common_step_counter=0,
        action_manager=SimpleNamespace(get_term=lambda name:action))
    def physics_step(actions):
        env.events.append('mock_physics_step');env.mock_physics_calls+=1
        action.raw_action.copy_(actions.clamp(-100,100));action._control_actions.copy_(action.raw_action)
        q[:,:3]+=v[:,:3]*.01
        env.env.episode_length_buf+=1;env.env.common_step_counter+=1
    def physics_reset(*,env_ids):
        env.events.append('physical_reset');env.reset_calls.append(env_ids.clone())
        reset_pose(env.env,env_ids)
        env.env.episode_length_buf[env_ids]=0
        action.raw_action[env_ids]=0.;action._control_actions[env_ids]=0.;action.sample_velocity[env_ids]=0.
        action.last_sample_q[env_ids]=q[env_ids][:,mq]
    env.env.step=physics_step;env.env.reset=physics_reset
    def resample(self,ids):
        if len(ids):
            self.events.append('command_resample');self.resample_calls.append(ids.clone())
        FudanEnv.resample(self,ids)
    env.resample=MethodType(resample,env);env.reward=lambda:(torch.zeros(n),{})
    env.close=lambda:None
    env.previous_position.copy_(data.root_link_pos_w);FudanEnv._update_state(env)
    return env


class Scene(dict):
    pass


def runner_configuration(runner):
    """给object.__new__构造的检查夹具补齐生产runner配置，不执行优化器。"""
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, FudanWalkCfgPPO, section_dict
    runner.train_cfg = FudanWalkCfgPPO()
    runner.ppo_config = section_dict(runner.train_cfg.algorithm)
    runner.num_steps_per_env = runner.train_cfg.runner.num_steps_per_env
    if hasattr(runner, 'env') and not hasattr(runner.env, 'cfg'):
        runner.env.cfg = FudanWalkCfg()
