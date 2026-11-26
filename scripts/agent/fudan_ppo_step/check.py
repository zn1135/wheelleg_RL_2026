"""Static migration/geometry checks and bounded no-learning GPU precheck."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import mujoco
import torch
from ct_mjlab.envs.chuanliantui.fudan_ppo_step.config import FudanStepCfg, SOURCE, SOURCE_SHA, recipe_record
from ct_mjlab.envs.chuanliantui.fudan_ppo_step.terrain import PlatformCfg
from ct_mjlab.fudan_ppo_walk.contract import make_network
from ct_mjlab.envs.chuanliantui.fudan_ppo_step.runner import initialize_policy

ROOT = Path(__file__).resolve().parents[3]


def source_fingerprint():
    paths = sorted((ROOT/'mjlab_training/src/ct_mjlab').rglob('*.py'))
    paths += sorted((ROOT/'scripts/agent/fudan_ppo_step').glob('*.py'))
    paths += sorted((ROOT/'resources/robots/chuanliantui_new_1/mjlab').glob('*'))
    paths += sorted((ROOT/'resources/robots/chuanliantui_new_1/meshes').glob('*'))
    paths += [ROOT/'sim2sim/rmuc2027_arena.py', ROOT/'scripts/agent/motion/acceptance.py',
              ROOT/'scripts/agent/motion/assess.py']
    return hashlib.sha256(b''.join(str(p.relative_to(ROOT)).encode()+p.read_bytes()
                          for p in paths if p.is_file())).hexdigest()


def static_checks(output):
    if getattr(FudanStepCfg().step,'allow_temporary_support',False):
        from .check_contact_exploration import static_checks as check_exploration
        return check_exploration(output)
    from ct_mjlab.envs import task_registry
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
    from ct_mjlab.fudan_ppo_walk.policy import load_policy
    assert 'fudan_ppo_step' in task_registry.names()
    old = FudanWalkCfg()
    cfg = FudanStepCfg()
    assert FudanWalkCfg().control.leg_stiffness == old.control.leg_stiffness
    assert cfg.rewards.scales is not old.rewards.scales
    assert hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest() == SOURCE_SHA
    policy = make_network()
    report = initialize_policy(policy, ROOT/SOURCE, cfg)
    reference, _ = load_policy(ROOT/SOURCE)
    torch.manual_seed(31001)
    obs, history = torch.randn(100, 25), torch.randn(100, 125)
    with torch.inference_mode():
        a = policy.act_inference(obs, history)[0]
        b = reference.act_inference(obs, history)[0]
    assert torch.equal(a, b)
    geometry = []
    for level, height in enumerate(cfg.step.heights):
        spec = mujoco.MjSpec();spec.worldbody.add_body(name='terrain')
        terrain_output = PlatformCfg(mode='up', size=cfg.step.patch_size).function(level/(len(cfg.step.heights)-1),spec,np.random.default_rng(1))
        model = spec.compile();data = mujoco.MjData(model);mujoco.mj_forward(model,data)
        def ray(x,y):
            return 1.-mujoco.mj_ray(model,data,np.array([x,y,1.]),np.array([0.,0.,-1.]),None,True,-1,None)
        assert abs(ray(2.8,2.))<1e-7
        assert abs(ray(5.,2.)-height)<1e-7
        geometry.append(dict(level=level,height=height,geoms=model.ngeom,spawn=terrain_output.origin.tolist()))
    from sim2sim.rmuc2027_arena import command_schedule, scheduled_command
    sequence=output/'commands.json'
    sequence.write_text(json.dumps([dict(duration_s=.20,command=[0.,0.,.30]),
                                   dict(duration_s=.20,command=[.15,0.,.30])]))
    schedule=command_schedule(sequence,.01)
    assert scheduled_command(schedule,19)==[0.,0.,.30] and scheduled_command(schedule,20)==[.15,0.,.30]
    from ct_mjlab.envs.chuanliantui.fudan_ppo_step.train import train
    rejected=SimpleNamespace(resume=None,ppo_init=None,log_dir=output/'must_not_start',
        precheck=output/'missing.json',num_envs=20,iterations=None,deadline=None)
    with patch.object(task_registry,'make_env',side_effect=AssertionError('No GPU construction without budget')):
        try:train(rejected)
        except ValueError as exc:assert 'budget' in str(exc)
        else:raise AssertionError('Missing training budget accepted')
    assert not rejected.log_dir.exists()
    return dict(migration=report,actor_output_max_difference=0.,geometry=geometry,
                contact_logic=contact_logic_checks(),
                independent_legs=independent_leg_reward_checks(),
                reference_reward=reference_reward_checks(),
                scoring=scoring_checks(),online_success=online_success_check(),
                logging=logging_checks(output),
                command_sequence_boundaries=True,missing_budget_rejected_before_gpu=True,
                old_task_configuration_unchanged=True,optimizer_updates=0)


def independent_leg_reward_checks():
    from ct_mjlab.envs.chuanliantui.fudan_ppo_step.env import StepEnv
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk import posture_reward
    n=8;env=object.__new__(StepEnv)
    env.cfg=FudanStepCfg();env.dt=.01
    env.terrain=SimpleNamespace(terrain_types=torch.tensor([0,1,1,1,2,2,1,1]))
    env.position=torch.tensor([[1.2,0.,.3],[0.,0.,.3],[1.2,0.,.3],[1.2,0.,.3],
                               [-.6,0.,.3],[-.6,0.,.3],[1.2,0.,.3],[1.2,0.,.3]])
    env.normal_support=torch.tensor([True,True,True,False,True,False,True,True])
    env.contacts=SimpleNamespace(step_contact=torch.tensor([False,False,False,True,False,True,False,False]))
    env.ray_failed=torch.zeros(n,dtype=torch.bool)
    # 最后两例交换左右腿、改成同步姿态，验证没有额外指定动作顺序。
    theta=torch.tensor([[.2,-.2]]*6+[[-.2,.2],[.2,.2]])
    wheel=torch.zeros(n,2,3);wheel[:,:,0]=theta.sin();wheel[:,:,2]=-theta.cos()
    env.env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(n,3)),
        sim=SimpleNamespace(data=SimpleNamespace(xpos=wheel,xanchor=torch.zeros(n,2,3),qpos=torch.zeros(n,4))))
    env.wheel_ids=env.hip_ids=torch.tensor([0,1]);env.front_q=torch.arange(4)
    env.front_limits=torch.tensor([[-1.,1.]]*4)
    env.robot=SimpleNamespace(data=SimpleNamespace(root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]*n)))
    env.commands=torch.zeros(n,3);env.commands[:,2]=.3
    env.linear=torch.zeros(n,3);env.angular=torch.zeros(n,3)
    env.height=torch.full((n,),.3);env.velocity=torch.zeros(n,6);env.acceleration=torch.zeros(n,6)
    env.action=SimpleNamespace(raw_action=torch.zeros(n,6),torques=torch.zeros(n,6))
    env.previous=torch.zeros(n,6,2);env.gravity=torch.tensor([[0.,0.,-1.]]*n);env.sums=torch.zeros(n,2)
    env.posture_reward_profile=posture_reward.from_task_config(env.cfg)
    _,baseline=FudanEnv.reward(env)
    total,terms=StepEnv.reward(env)
    assert set(terms)==set(env.cfg.rewards.scales)
    assert (baseline['nominal_state'][:7]<0).all()
    assert torch.equal(terms['nominal_state'][:2],baseline['nominal_state'][:2])
    assert (terms['nominal_state'][2:]==0).all()
    assert total[2]==total[6] and total[2]==total[7]
    assert all(torch.equal(value,baseline[key]) for key,value in terms.items()
               if key not in ('nominal_state','base_height','base_height_enhance','collision'))
    # 直接执行真实速度奖励，确认慢走到跟上指令始终有得分改善。
    speeds=torch.tensor([0.,.05,.2,.4,.6,.8,1.,1.2])
    env.commands[:,0]=1.;env.linear[:,0]=speeds
    _,velocity_terms=FudanEnv.reward(env)
    progress=velocity_terms['tracking_lin_vel']+velocity_terms['tracking_lin_vel_enhance']
    assert (progress[1:7]>progress[:6]).all()
    env.commands[:,0]=2.5
    _,maximum_terms=FudanEnv.reward(env)
    maximum=maximum_terms['tracking_lin_vel']+maximum_terms['tracking_lin_vel_enhance']
    assert (maximum[1:]>maximum[:-1]).all()
    return dict(flat_and_far_edge_constraint_retained=True,
        up_and_down_near_edge_independent_before_liftoff=True,
        left_right_mirror_and_synchronous_allowed=True,other_reward_terms_unchanged=True,
        actual_velocity_reward_improves_from_stall=True,
        speed_probes_m_s=speeds.tolist(),initial_cap_reward=progress.tolist(),maximum_cap_reward=maximum.tolist(),
        scope='Actual reward with synthetic poses; no physical climbing validation')


def reference_reward_checks():
    from ct_mjlab.envs.chuanliantui.fudan_ppo_step.env import StepEnv
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk import posture_reward
    cfg=FudanStepCfg();n=4
    assert 'roll' not in cfg.rewards.scales and 'pitch' not in cfg.rewards.scales
    assert cfg.rewards.scales['orientation']==-10. and cfg.rewards.scales['tracking_ang_vel_enhance']==0.
    fake=object.__new__(StepEnv);fake.cfg=cfg;fake.dt=.01
    fake.height=torch.tensor([.35,.32,.35,.35]);fake.commands=torch.tensor([[0.,0.,.35]]*n)
    fake.position=torch.tensor([[1.2,0.,.35]]*n);fake.terrain=SimpleNamespace(terrain_types=torch.ones(n,dtype=torch.long))
    fake.env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(n,3)))
    fake.normal_support=torch.tensor([True,True,False,True]);fake.ray_failed=torch.tensor([False,False,False,True])
    fake.contacts=SimpleNamespace(step_contact=torch.zeros(n,dtype=torch.bool))
    theta=torch.tensor(5.*np.pi/180)
    fake.gravity=torch.tensor([[0.,-torch.sin(theta),-torch.cos(theta)]]*n)
    fake.posture_reward_profile=posture_reward.from_task_config(cfg)
    raw=posture_reward.penalties(fake)['orientation']
    orientation=(raw*cfg.rewards.scales['orientation']*fake.dt).clamp(-fake.dt,fake.dt)
    baseline={name:torch.zeros(n) for name in cfg.rewards.scales};baseline['orientation']=orientation
    with patch.object(FudanEnv,'reward',return_value=(torch.zeros(n),baseline)):
        total,terms=StepEnv.reward(fake)
    error=(fake.height[:3]-.35).square()
    torch.testing.assert_close(terms['base_height'][:3],torch.exp(-error/.001)*.01)
    assert (terms['base_height_enhance']==0).all()
    assert terms['base_height'][0]==terms['base_height'][2]  # 抬轮不直接清掉高度奖励。
    assert abs(float(terms['orientation'][0])-(-.00075961235))<1e-9
    assert total[3]==0 and all(value[3]==0 for value in terms.values())
    return dict(source_commit=cfg.step.reference_commit,gravity_orientation_formula_verified=True,
        unit_height_amplitude_and_disabled_enhance_verified=True,height_continues_without_double_support=True,
        missing_ray_reward_zero=True,scope='Isolated actual reward formulas; not physical stair success')


def contact_logic_checks():
    from ct_mjlab.envs.chuanliantui.fudan_ppo_step.contacts import ContactMonitor
    def sensor(slots):
        return SimpleNamespace(data=SimpleNamespace(force=torch.zeros(2,slots,3),pos=torch.zeros(2,slots,3),
            normal=torch.zeros(2,slots,3),found=torch.zeros(2,slots)))
    scene={key:sensor(slots) for key,slots in [('step_base',1),('step_legs',2),('step_wheels',8)]}
    scene['robot']=SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.tensor([0])))
    fake=SimpleNamespace(num_envs=2,device='cpu',physics_dt=.001,scene=scene,
        sim=SimpleNamespace(model=SimpleNamespace(body_mass=torch.full((2,1),13.4)),
                            data=SimpleNamespace(qvel=torch.zeros(2,6))))
    monitor=ContactMonitor(fake,FudanStepCfg().step)
    # 两根杆方向相反时，真实接触不能被向量合力抵消。
    scene['step_legs'].data.force[:,0,0]=2.
    scene['step_legs'].data.force[:,1,0]=-2.
    for _ in range(210):monitor.update()
    assert monitor.failed.all() and (monitor.peak[:,1]==2.).all()
    before=monitor.count[0].clone();monitor.reset(torch.tensor([1]))
    assert torch.equal(before,monitor.count[0]) and not monitor.failed[1]
    monitor.reset(slice(None));scene['step_legs'].data.force.zero_()
    scene['step_base'].data.force[:,:,2]=20.
    for _ in range(110):monitor.update()
    assert monitor.failed.all()
    monitor.reset(slice(None));scene['step_base'].data.force.fill_(.1)
    for _ in range(250):monitor.update()
    assert not monitor.failed.any() and not monitor.count.any()
    return dict(opposing_leg_contacts_detected=True,continuous_and_heavy_support_terminate=True,
                subthreshold_ignored=True,partial_reset_isolated=True)


def scoring_checks():
    from .evaluate import score
    n=800;shape=(n,3);wheel=np.zeros((n,2,3));wheel[:,:,0]=np.minimum(np.arange(n)[:,None]*.003,1.8)
    wheel[:,0,1]=.21;wheel[:,1,1]=-.21;wheel[:,:,2]=.26
    command=np.zeros(shape);command[:600,0]=.3;command[:,2]=.3
    gravity=np.zeros(shape);gravity[:,2]=-1.
    velocity=np.zeros(shape);velocity[:600,0]=.3
    trace=dict(valid=np.ones(n,bool),origin=np.zeros(3),position=np.zeros(shape),wheel_pos=wheel,
        linear_world=velocity,linear_body=velocity,angular_body=np.zeros(shape),gravity=gravity,
        command=command,force=np.zeros((n,10,2)),support=np.ones((n,2),bool),
        support_height=np.full((n,2),.2),weight=np.array(131.4))
    sample=dict(id='synthetic',mode='up',height=.2)
    assert score(sample,trace,duration=8.)['passed']
    trace['force'][650:675,:,0]=20.
    assert not score(sample,trace,duration=8.)['passed']
    assert not score(sample,trace,duration=8.,max_approach_speed=3.)['passed']
    trace['force'][:]=0.;trace['linear_world']=velocity.copy();trace['linear_world'][400:450,0]=1.
    assert not score(sample,trace,duration=8.)['passed']
    assert score(sample,trace,duration=8.,max_approach_speed=3.)['passed']
    trace['linear_world']=velocity;trace['valid'][:]=False;trace['valid'][:100]=True
    assert not score(sample,trace,duration=8.)['passed']
    trace['valid'][:]=True;trace['terrain_valid']=np.ones(n,bool);trace['terrain_valid'][400]=False
    assert score(sample,trace,duration=8.)['reason']=='missing_terrain'
    return dict(two_second_stop=True,body_support_rejected=True,fast_collision_rejected=True,
                incomplete_rejected=True,missing_terrain_rejected=True,scope='Synthetic scoring fixtures, not physical validation')


def online_success_check():
    from ct_mjlab.envs.chuanliantui.fudan_ppo_step.env import StepEnv
    from math import sin,cos,radians
    theta=radians(4.)
    fake=SimpleNamespace(cfg=FudanStepCfg(),modes=torch.tensor([1]),dt=.01,
        env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(1,3)),
            sim=SimpleNamespace(data=SimpleNamespace(xpos=torch.tensor([[[1.8,.21,.26],[1.8,-.21,.26]]])))),
        wheel_ids=torch.tensor([0,1]),parking_time=torch.tensor([1.99]),parking_ticks=torch.tensor([199]),landing_ticks=torch.tensor([10]),
        terrain=SimpleNamespace(terrain_levels=torch.tensor([7])),support_height=torch.tensor([[.2,.2]]),
        contacts=SimpleNamespace(supported=torch.ones(1,2,dtype=torch.bool),failed=torch.tensor([False]),
            step_contact=torch.tensor([False]),step_speed_peak=torch.tensor([0.])),
        crossed=torch.tensor([True]),max_approach_speed=torch.tensor([.3]),
        robot=SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_w=torch.zeros(1,3))),
        position=torch.tensor([[1.8,0,.5]]),normal_support=torch.tensor([True]),commands=torch.tensor([[0.,0.,.3]]),
        gravity=torch.tensor([[sin(theta),-sin(theta)*cos(theta),-cos(theta)**2]]))
    StepEnv._progress(fake)
    assert fake.success.item() and fake.parking_ticks.item()==200
    fake.contacts.step_contact[:]=True
    StepEnv._progress(fake)
    assert not fake.success.item() and fake.parking_ticks.item()==0
    fake.contacts.step_contact[:]=False;fake.parking_ticks[:]=199
    fake.max_approach_speed[:]=fake.cfg.step.approach_speed[1]
    StepEnv._progress(fake)
    assert fake.success.item()
    fake.parking_ticks[:]=199;fake.max_approach_speed[:]=fake.cfg.step.max_approach_speed+.01
    StepEnv._progress(fake)
    assert not fake.success.item()
    return dict(exact_200_tick_stop=True,roll4_pitch4_allowed=True,contact_breaks_stop=True,
                configured_approach_speed_allowed=True,overspeed_rejected=True,
                scope='Actual online predicate with synthetic states; not physical validation')


def logging_checks(output):
    from ct_mjlab.envs.chuanliantui.fudan_ppo_step.runner import StepRunner
    row=dict(env=0,mode=1,level=0,success=False,failure=True,reverse=True,crossed=False,max_approach_speed=.3,
        contact_rate=[.5,0.],peak_N=[20.,0.],longest_s=[.1,0.],peak_position_w=[[1.,2.,3.],[0.,0.,0.]])
    current=dict(contact_rate=torch.zeros(1,2),peak_N=torch.zeros(1,2),longest_s=torch.zeros(1,2),
                 peak_position_w=torch.zeros(1,2,3))
    scalars={}
    fake=SimpleNamespace(iteration=1,log_dir=output/'synthetic_logging',
        writer=SimpleNamespace(add_scalar=lambda key,value,step:scalars.__setitem__(key,float(value))),
        env=SimpleNamespace(cfg=FudanStepCfg(),episode_rows=[row],
            terrain=SimpleNamespace(terrain_levels=torch.zeros(1,dtype=torch.long)),
            contacts=SimpleNamespace(metrics=lambda ids:current)))
    fake.log_dir.mkdir()
    StepRunner.log_task_metrics(fake)
    assert scalars['Contact/body/peak_N']==20. and scalars['Contact/body/contact_rate']>0.
    assert scalars['Contact/body/peak_position_x_m']==1. and not fake.env.episode_rows
    return dict(completed_episode_contact_retained_after_reset=True,peak_position_logged=True,
                scope='Synthetic logger fixture; no training update')


def contact_friction(env):
    sim=env.env.sim;model=sim.mj_model
    terrain=model.body('terrain').id
    count=int(sim.wp_data.nacon.numpy()[0])
    pairs=sim.wp_data.contact.geom.numpy()[:count]
    worlds=sim.wp_data.contact.worldid.numpy()[:count]
    mask=(model.geom_bodyid[np.maximum(pairs,0)]==terrain).any(-1)
    actual=sim.wp_data.contact.friction.numpy()[:count,:2][mask]
    expected=env.env._fudan_parameters['friction'].cpu().numpy()[worlds[mask],None]
    err=float(np.max(np.abs(actual-expected))) if len(actual) else 0.
    assert err<1e-6
    return dict(contacts=int(mask.sum()),max_error=err)


def gpu_checks(args, report):
    if getattr(FudanStepCfg().step,'allow_temporary_support',False):
        from .check_contact_exploration import gpu_checks as check_exploration
        return check_exploration(args,report)
    from ct_mjlab.envs.chuanliantui.fudan_ppo_step.env import StepEnv
    from ct_mjlab.envs.chuanliantui.fudan_ppo_step.runner import StepRunner
    cfg=FudanStepCfg();env=None;runner=None;viewer=None
    if args.friction is not None:
        cfg.domain_rand.ground_friction_range=[args.friction,args.friction]
    torch.set_num_threads(1)
    torch.manual_seed(31001)
    try:
        env=StepEnv(args.num_envs,seed=31001,cfg=cfg,randomized=True)
        original_types=env.modes.clone()
        # 核实际GPU参数，不以配置写了小范围代替随机化生效。
        low_random={}
        for field,key in [('kp_scale','kp_range'),('kd_scale','kd_range'),
                          ('motor_strength','motor_strength_range'),('nominal_offset','nominal_offset_range')]:
            values=getattr(env.action,field);low,high=getattr(cfg.domain_rand,key)
            assert values.min()>=low-1e-7 and values.max()<=high+1e-7
            low_random[field]=[float(values.min()),float(values.max())]
        robot=env.robot;body_ids=robot.indexing.body_ids
        default_mass=env.env.sim.get_default_field('body_mass')[body_ids]
        lower=default_mass*cfg.domain_rand.mass_scale_range[0]
        upper=default_mass*cfg.domain_rand.mass_scale_range[1]
        local=robot.body_names.index('base_link')
        lower[local]=(default_mass[local]+cfg.domain_rand.added_mass_range[0])*cfg.domain_rand.mass_scale_range[0]
        upper[local]=(default_mass[local]+cfg.domain_rand.added_mass_range[1])*cfg.domain_rand.mass_scale_range[1]
        actual_mass=env.env.sim.model.body_mass[:,body_ids]
        assert (actual_mass>=lower[None]-1e-5).all() and (actual_mass<=upper[None]+1e-5).all()
        com=env.env._fudan_parameters['com']
        assert com.abs().max()<=.003+1e-7
        mu=env.env._fudan_parameters['friction'];low,high=cfg.domain_rand.ground_friction_range
        assert mu.min()>=low-1e-6 and mu.max()<=high+1e-6
        low_random.update(actual_body_mass_within_bounds=True,actual_com_abs_max_m=float(com.abs().max()),
            friction=[float(mu.min()),float(mu.max())],noise_scales=cfg.normalization.noise_scales)
        # 复用真实复位路径检查跟踪不合格不提速、合格提速以及最高上限。
        original_steps=env.total_steps;ids=torch.arange(args.num_envs,device=env.device)
        env.total_steps=env.max_episode_length;env.sums.zero_();env.reset_idx(ids)
        assert env.speed_limit==1.
        env.sums[:]=env.max_episode_length*env.dt;env.reset_idx(ids)
        assert abs(env.speed_limit-1.1)<1e-9
        env.speed_limit=2.49;env.sums[:]=env.max_episode_length*env.dt;env.reset_idx(ids)
        assert env.speed_limit==2.5
        env.speed_limit=cfg.commands.velocity_initial;env.total_steps=original_steps;env.reset_idx(ids)
        stairs=env.modes!=0
        assert (env.commands[stairs,0]>=.3).all() and (env.commands[stairs,0]<=1.).all()
        if time.time()>=args.deadline:raise TimeoutError('GPU precheck deadline expired')
        runner=StepRunner(env,log_dir=args.output/'migration')
        migration=runner.initialize_from_ppo(ROOT/SOURCE)
        initial=runner.save('model_initial.pt')
        from ct_mjlab.fudan_ppo_walk.policy import load_policy
        restored,_=load_policy(initial,device=env.device)
        with torch.inference_mode():
            probe=torch.zeros(4,25,device=env.device);history_probe=torch.zeros(4,125,device=env.device)
            assert torch.equal(restored.act_inference(probe,history_probe)[0],
                               runner.policy.act_inference(probe,history_probe)[0])
        del restored,probe,history_probe
        assert not runner.alg.optimizer.state and not runner.alg.extra_optimizer.state
        # 平地77槽与原重复高度逐值核对；网络迁移已在静态检查中核对。
        from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
        ids=torch.arange(args.num_envs,device=env.device)
        env.randomized=False
        env.terrain.terrain_types.zero_();env.terrain.terrain_levels.zero_()
        env.reset_idx(ids);env.env.sim.forward();env.env.scene.update(dt=0.);env.env.sim.sense();env._update_state()
        actual_obs,actual_critic=env.observations();old_obs,old_critic=FudanEnv.observations(env)
        assert torch.equal(actual_obs,old_obs)
        torch.testing.assert_close(actual_critic,old_critic,atol=1e-6,rtol=0.)
        # 覆盖所有课程，包括带侧障碍的最高层；随后检查局部复位隔离。
        env.randomized=True
        env.terrain.terrain_types.copy_(original_types)
        env.terrain.terrain_levels[:] = ids.remainder(len(cfg.step.heights))
        env.reset_idx(ids)
        # 本机台阶课程采用正向；平地仍允许前后行驶。
        if cfg.step.reverse_probability==0.:
            stairs=env.modes!=0
            assert not env.reverse.any() and (env.commands[stairs,0]>0).all()
        if cfg.step.height_range[0]==cfg.step.height_range[1]:
            stairs=env.modes!=0
            torch.testing.assert_close(env.commands[stairs,2],
                torch.full_like(env.commands[stairs,2],cfg.step.height_range[0]),atol=0.,rtol=0.)
        if cfg.step.approach_speed[0]==cfg.step.approach_speed[1]:
            stairs=env.modes!=0
            torch.testing.assert_close(env.commands[stairs,0].abs(),
                torch.full_like(env.commands[stairs,0],cfg.step.approach_speed[0]),atol=0.,rtol=0.)
        flat=env.modes==0
        assert (env.commands[flat,0].abs()<=cfg.commands.stop_speed_range[1]).all()
        env.env.sim.forward();env.env.scene.update(dt=0.);env.env.sim.sense();env._update_state()
        obs,critic=env.observations()
        assert obs.shape==(args.num_envs,25) and critic.shape==(args.num_envs,141)
        assert env.terrain_samples.shape==(args.num_envs,77)
        if args.viewer:
            import viser
            from mjlab.viewer.viser.scene import MjlabViserScene
            viewer=viser.ViserServer(host='127.0.0.1',port=args.port,label='Single platform GPU precheck')
            scene=MjlabViserScene(viewer,env.env.sim.mj_model,args.num_envs,
                sim_model=env.env.sim.model,expanded_fields=env.env.sim.expanded_fields)
            scene.show_only_selected=False;scene.create_scene_gui(camera_distance=6.,show_debug_viz_control=False)
            print(f'20-environment precheck: http://127.0.0.1:{viewer.get_port()}',flush=True)
        model=env.env.sim.mj_model
        ground=env.env._terrain_friction_ids
        expected=env.env._fudan_parameters['friction']
        assert torch.allclose(env.env.sim.model.geom_friction[:,ground,0],expected[:,None].expand(-1,len(ground)))
        positions=env.env.sim.data.qpos.clone()
        sub=ids[::3]
        env.reset_idx(sub)
        untouched=ids[ids.remainder(3)!=0]
        assert torch.equal(positions[untouched],env.env.sim.data.qpos[untouched])
        # 上/下台×正向/倒车均须朝目标运动；转身和速度符号必须一起变。
        env.randomized=False
        env.terrain.terrain_types[:]=1+ids.remainder(2)
        env.env._step_fixed_reverse=(ids//2).remainder(2).bool()
        env.env._step_fixed_yaw=torch.zeros(args.num_envs,device=env.device)
        env.reset_idx(ids)
        heading=1.-2.*env.env.sim.data.qpos[:,6].square()
        direction=torch.where(env.modes==2,-1.,1.)
        assert (heading*env.commands[:,0]*direction>0).all()
        assert (env.commands[:,2]>=cfg.step.height_range[0]).all() and (env.commands[:,2]<=cfg.step.height_range[1]).all()
        # 通行升级与严格停车分开，防止探索噪声把课程永远锁在最低级。
        env.finished[:2]=True;env.crossed[0]=True;env.crossed[1]=False;env.success[:2]=False
        env.contacts.failed[:2]=False;env.terrain.terrain_levels[:2]=2
        env.reset_idx(ids[:2])
        assert env.terrain.terrain_levels[:2].tolist()==[3,1]
        env.finished[:2]=True;env.crossed[:2]=False;env.terrain.terrain_levels[:2]=1
        env.reset_idx(ids[:2])
        assert env.terrain.terrain_levels[:2].tolist()==[1,1]
        fixed_types=env.modes.clone()
        env.finished[:]=True;env.crossed[:]=True;env.contacts.failed[:]=False
        env.terrain.terrain_levels[:]=len(cfg.step.heights)-1
        env.reset_idx(ids)
        assert torch.equal(env.modes,fixed_types)
        assert ((env.terrain.terrain_levels>=0)&(env.terrain.terrain_levels<len(cfg.step.heights))).all()
        assert (env.terrain.terrain_levels[env.modes!=0]>=1).all()
        assert (env.terrain.terrain_levels<len(cfg.step.heights)-1).any()
        del env.env._step_fixed_reverse,env.env._step_fixed_yaw
        env.randomized=True
        fixed_types=env.modes.clone();env.reset_idx(ids)
        assert torch.equal(env.modes,fixed_types)
        # 直立机器人在场边时，扫描网格可先于根位置越界；必须有真实边框地面。
        corner=env.env.sim.data.qpos[0].clone()
        corner[:3]=corner.new_tensor([-len(cfg.step.heights)*cfg.step.patch_size[0]/2+.2,
                                     -3*cfg.step.patch_size[1]/2+.2,.3])
        corner[3:7]=corner.new_tensor([.92387953,0.,0.,.38268343])
        env.env.sim.data.qpos[0]=corner
        env.env.sim.forward();env.env.scene.update(dt=0.);env.env.sim.sense()
        assert (env.env.scene['step_scan'].data.distances[0]>=0).all()
        torch.testing.assert_close(env.env.scene['step_scan'].data.hit_pos_w[0,:,2],torch.zeros(77,device=env.device),atol=1e-5,rtol=0.)
        env.reset_idx(ids[:1])
        # 模拟一条失效射线：只终止对应世界，不允许奖励、成功或超时bootstrap。
        broken=env.env.scene['step_scan'].data
        broken.distances[0,0]=-1.
        env._update_state()
        assert env.ray_failed[0] and not env.ray_failed[1:].any() and not env.normal_support[0]
        reward,terms=env.reward()
        assert reward[0]==0 and all(value[0]==0 for value in terms.values())
        with patch.object(env.env,'step',return_value=None):
            _,_,reward,done,info,_=env.step(torch.zeros(args.num_envs,6,device=env.device),auto_reset=False,resample=False)
        assert done[0] and not done[1:].any() and not info['time_outs'][0]
        assert not info['terrain_valid'][0] and reward[0]==0
        env.reset_idx(ids[:1])
        assert not env.ray_failed.any()
        contact_rows=[];started=time.perf_counter()
        for step in range(args.steps):
            if time.time()>=args.deadline:raise TimeoutError('GPU precheck deadline expired')
            # 真实零动作检查，不拿未训练的策略越障结果当验收。
            env.step(torch.zeros(args.num_envs,6,device=env.device))
            if viewer is not None and step%3==0:scene.update(env.env.sim.data)
            contact_rows.append(contact_friction(env))
        assert sum(r['contacts'] for r in contact_rows)>0
        assert torch.isfinite(env.env.sim.data.qpos).all() and torch.isfinite(env.env.sim.data.qvel).all()
        assert runner.primary_updates==runner.encoder_updates==0
        assert env.contacts.total_ticks==args.steps*cfg.control.decimation
        # 走一轮真正的PPO采样/回报计算接口，但禁止反传和优化器更新。
        obs,critic,history=env.reset()
        with torch.inference_mode():
            for _ in range(runner.num_steps_per_env):
                if time.time()>=args.deadline:raise TimeoutError('Rollout smoke check deadline expired')
                action=runner.alg.act(obs,history,critic)
                obs,critic,reward,done,info,history=env.step(action)
                runner.alg.process_env_step(reward,done,info,obs)
            runner.alg.compute_returns(torch.cat((critic,runner.policy.encode(history)),-1))
            # 覆盖完整探索动作/复位过程，不能仅用确定性动作和不足一秒的rollout检查。
            for _ in range(512):
                if time.time()>=args.deadline:raise TimeoutError('Stochastic reset precheck deadline expired')
                action=runner.policy.act(obs,history)
                obs,critic,_,_,_,history=env.step(action)
        frozen=torch.load(ROOT/SOURCE,map_location=env.device,weights_only=False)['model_state_dict']
        assert all(torch.equal(value,frozen[key]) for key,value in runner.policy.state_dict().items())
        assert not runner.alg.optimizer.state and not runner.alg.extra_optimizer.state
        report.update(kind='step_gpu_precheck',num_envs=args.num_envs,recipe=recipe_record(cfg),
            physics_steps=env.contacts.total_ticks,zero_action_physics_steps=args.steps*cfg.control.decimation,
            rollout_steps_without_update=runner.num_steps_per_env,optimizer_updates=0,
            elapsed_sampling_s=time.perf_counter()-started,ngeom=model.ngeom,
            terrain_collision_geoms=len(ground),contact_friction=contact_rows,
            partial_reset_unchanged_other_worlds=True,ray_samples=77,migration=migration,
            forward_reverse_spawn_and_commands_checked=True,
            default_step_reverse_probability=cfg.step.reverse_probability,
            default_forward_only_checked=cfg.step.reverse_probability==0.,
            default_step_height_range=list(cfg.step.height_range),
            default_fixed_step_height_checked=cfg.step.height_range[0]==cfg.step.height_range[1],
            default_step_speed_range=list(cfg.step.approach_speed),
            default_fixed_step_speed_checked=cfg.step.approach_speed[0]==cfg.step.approach_speed[1],
            flat_speed_within_preserved_range=True,
            actual_low_randomization=low_random,speed_curriculum_reset_path_checked=True,
            initial_step_command_range=[.3,1.],
            independent_height_commands_within_range=True,landing_curriculum_independent_of_parking=True,
            fixed_terrain_types_across_reset=True,maximum_level_rehearsal_checked=True,
            minimum_stair_level_survives_failures_and_rehearsal=True,
            border_corner_rays_verified=True,stochastic_policy_steps_without_update=512,
            constraint_capacity=env.env.sim.wp_data.njmax,
            constraint_peak=int(env.contacts.constraint_peak),
            global_contact_capacity=env.env.sim.wp_data.naconmax,
            global_contact_peak=int(env.contacts.global_contact_peak),
            capacity_overflow=bool(env.contacts.capacity_overflow.any()),
            missing_ray_fails_only_affected_world=True,missing_ray_reward_zero=True,ray_failure_clears_on_reset=True,
            injected_missing_ray_terminates_without_bootstrap=True,
            flat_actor_and_critic_inputs_equivalent=True,
            initialized_checkpoint=str(initial),validation_backend='mujoco_warp_gpu',
            cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(),
            contact_physical_ticks_total=env.contacts.total_ticks,all_networks_unchanged_after_rollout=True,
            saved_policy_load_and_fixed_inference_equal=True,
            scope='Finite reset/rays/friction/contact/migration checks; not successful platform climbing')
    finally:
        if viewer is not None:viewer.stop()
        if runner is not None:runner.writer.close()
        if env is not None:env.close()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--num-envs',type=int,default=20)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--deadline',type=float)
    parser.add_argument('--friction',type=float,choices=(.4,1.2))
    parser.add_argument('--viewer',action='store_true',help='Show the actual GPU states in Viser')
    parser.add_argument('--port',type=int,default=8791)
    args=parser.parse_args()
    if args.gpu and args.deadline is None:parser.error('--gpu requires --deadline')
    if args.viewer and not args.gpu:parser.error('--viewer requires --gpu')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,kind='step_static_check',optimizer_updates=0,physics_steps=0)
    try:
        # 正式检查禁止任何优化器更新；整个运行里都有效。
        with ExitStack() as stack:
            stack.enter_context(patch('torch.optim.Adam.step',side_effect=AssertionError('No learning in checks')))
            report['static']=static_checks(args.output)
            if args.gpu:gpu_checks(args,report)
        report.update(passed=True,source_fingerprint=source_fingerprint())
    except BaseException as exc:
        report['error']=repr(exc)
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('static','recipe','contact_friction')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
