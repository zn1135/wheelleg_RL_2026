#!/usr/bin/env python3
"""Legacy r5 finite migration checks. For r6 use check_chuanliantui_reference_training.py.

Run with mjlab_training/.venv/bin/python. The independent reward oracle executes
SCUT's original _get_rewards AST, stopping before its manager/logging side effects.
Disabled reward helper stubs do not affect the active terms under comparison.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import numpy as np
import torch
from tensordict import TensorDict
from ct_mjlab.config import RECIPE, PARAMS, WEIGHTS, JOINTS, legacy_manifest as manifest, legacy_runner_cfg as runner_cfg, STEP_DT, DT, DECIMATION, LEG_REFERENCE
from ct_mjlab.recipe import Recipe, Delay, core_obs
from ct_mjlab.cpu import CpuPlayback
from ct_mjlab.gas_spring import gas_spring_force
from ct_mjlab.modes import ControlMode
from ct_mjlab.policy import actor_model, load_actor, export, check_manifest
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_from_euler_xyz

DEFAULT_SOURCE=Path('/home/zn/文档/wheeled-legged_RL-main_huananhu/source/agent_tasks/agent_tasks/direct/wheelbipe')

def check_control_modes():
    # Exercise thresholds and interrupted holds, including recovery after a fall.
    mode=ControlMode(2,'cpu');ids=torch.arange(2)
    mode.reset(ids,torch.tensor([True,False]))
    assert torch.allclose(mode.observation()[0],torch.tensor([0.,0,0,1,0,1.1,0]))
    assert torch.equal(mode.observation()[1],torch.tensor([1.,0,0,0,0,0,0]))
    state=dict(height=torch.full((2,),.22),roll=torch.zeros(2),pitch=torch.zeros(2),
               base_contact=torch.zeros(2),wheel_contact=torch.full((2,2),6.))
    for _ in range(24):mode.advance(state)
    assert mode.recover.tolist()==[True,False]
    # Exactly threshold 5 N is insufficient; interrupted stability restarts hold.
    state['wheel_contact'][0,1]=5.
    mode.advance(state);assert mode.stable_steps[0]==0
    state['wheel_contact'][0,1]=6.
    for _ in range(24):mode.advance(state)
    assert mode.recover[0]
    mode.advance(state)
    assert mode.recover.tolist()==[False,False] and mode.recovered.tolist()==[True,False]
    assert mode.observation()[:,6].tolist()==[0,0]
    state['base_contact'][1]=2.
    for _ in range(4):mode.advance(state)
    assert not mode.recover[1]
    mode.advance(state)
    assert mode.recover.tolist()==[False,True] and mode.observation()[1,6]==0
    mode.advance(state)
    assert torch.isclose(mode.observation()[1,6],torch.tensor(.02))
    state['base_contact'][1]=0.
    state['height'][1]=.18
    for _ in range(25):mode.advance(state)
    assert mode.recover[1], 'insufficient height marked as recovered'
    # Observation calls must not advance timers or reset other environments.
    before=mode.observation().clone()
    assert torch.equal(before,mode.observation())
    mode.reset(torch.tensor([0]),torch.tensor([True]))
    assert torch.equal(mode.observation()[1],before[1])
    state['height'][1]=.22
    for _ in range(25):mode.advance(state)
    assert not mode.recover.any() and mode.recovered.all()
    # Actor and critic carry identical live mode values, without noise/delay.
    recipe=Recipe(2,'cpu',torch.zeros(6),randomize=False)
    recipe.reset(ids,torch.tensor([True,False]))
    complete=dict(state,pos=torch.zeros(2,6),vel=torch.zeros(2,6),ang=torch.zeros(2,3),
        gravity=torch.tensor([[0.,0,-1.]]*2),lin=torch.zeros(2,3),torque=torch.zeros(2,6),
        wheel_vel=torch.zeros(2,2,3))
    obs=recipe.observations(complete,advance=False)
    assert torch.equal(obs['actor'][:,28:35],recipe.mode.observation())
    assert torch.equal(obs['critic'][:,32:39],obs['actor'][:,28:35])
    state['pitch'][:]=torch.pi
    for _ in range(5):mode.advance(state)
    assert mode.recover.all(), 'inverted pose was treated as upright'
    return dict(stable_hold_steps=25,failure_hold_steps=5,reset_and_reentry_checked=True,
                actor_critic_equal=True,observation_reads_side_effect_free=True)

def source_oracle(source,state,recipe,terminated, *, height_gate=False):
    tree=ast.parse((source/'wheelbipe25_v3/env.py').read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_get_rewards')
    cut=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='reward_terms' for t in n.targets))
    method.body=method.body[:cut]+ast.parse("return {k[4:]: v for k, v in locals().items() if k.startswith('rew_')}").body
    method.decorator_list=[];method.returns=None
    module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    namespace={'torch':torch,'euler_xyz_from_quat':euler_xyz_from_quat,
               'wrap_to_pi':lambda x:(x+torch.pi)%(2*torch.pi)-torch.pi}
    exec(compile(module,'SCUT original _get_rewards','exec'),namespace)
    n=len(state['pos']);zero=torch.zeros(n);cfg=NS(**PARAMS)
    if not hasattr(cfg,'leg_front_rear_range'):cfg.leg_front_rear_range=(-3.14,3.14)
    if not hasattr(cfg,'soft_range_scale'):cfg.soft_range_scale=.9
    d=NS(joint_acc=state['acc'],applied_torque=state['torque'],root_lin_vel_b=state['lin'],
        root_ang_vel_b=state['ang'],projected_gravity_b=state['gravity'],
        root_quat_w=quat_from_euler_xyz(state['roll'],state['pitch'],zero),
        body_pos_w=state['wheel_pos'],soft_joint_pos_limits=torch.tensor([[[-3.,3.]]*6]).expand(n,-1,-1))
    env=NS(cfg=cfg,robot=NS(data=d),num_envs=n,device='cpu',step_dt=STEP_DT,
        reset_terminated=terminated,episode_length_buf=torch.zeros(n),ground_z_est=zero,
        _actuate_idx=list(range(6)),_legs_act_idx=list(range(4)),_legs_front_idx=[0,2],_legs_rear_idx=[1,3],
        _wheel_idx=[4,5],_wheel_link_idx=[0,1],_left_right_leg_joint_pair_idx=([0,1],[2,3]),
        _undesired_contact_link_idx=[0],joint_pos=state['pos'],joint_vel=state['vel'],
        _actions=recipe.actions.clone(),_previous_actions=recipe.previous.clone(),
        _before_previous_actions=recipe.before.clone(),_previous_applied_torque=torch.zeros(n,6),
        _before_previous_applied_torque=torch.zeros(n,6),leg_actions=torch.zeros(n,4),command=torch.zeros(n,3))
    env.contact_sensor=NS(data=NS(net_forces_w_history=torch.zeros(n,3,3,3)))
    env.contact_sensor.data.net_forces_w_history[:,0,0,0]=state['bad_contact']
    env._get_root_quat_inv_and_wheel_pos_b=lambda:(None,state['wheel_pos'],state['wheel_pos'],state['wheel_vel'],state['wheel_vel'])
    env._update_ground_height_estimate=lambda:None
    env._get_policy_action_slices=lambda:(slice(0,4),slice(4,6))
    env._get_observed_height=lambda x:state['height']
    env._get_height_reward_reference_height=lambda x,y:x
    env._get_height_reward_target_height=lambda:torch.full((n,),.22)
    env._use_absolute_height=lambda:True;env._use_leg_length_height=lambda:False
    for key in ('height','orientation_x','orientation_y'):setattr(env,f'_get_vel_{key}_gate_enabled',lambda:False)
    cfg.vel_height_gate_mode='linear_band'
    env._get_vel_height_gate_enabled=lambda:height_gate
    env._get_vel_height_gate_full_error=lambda:.03
    env._get_vel_height_gate_zero_error=lambda:.06
    env._as_reward_gate_tensor=lambda value,reference:torch.full_like(reference,value)
    env._get_wheel_relative_ground_heights_raw=lambda:torch.zeros(n,2)
    env._get_wheel_motor_z_axis_align_error_sq=lambda:zero
    env._get_wheel_contact_force_peaks=lambda x:state['wheel_contact']
    env._get_wheel_air_spin_reward=lambda *args:zero
    env._debug_print_undesired_contacts=lambda *args:None
    env._get_rear2_rear1_joint_limit_terms=lambda:(zero,zero,zero)
    raw=namespace['_get_rewards'](env)
    return {k:torch.nan_to_num(raw[k]*WEIGHTS[k]*STEP_DT,nan=0,posinf=0,neginf=0) for k in WEIGHTS}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--gpu',action='store_true');p.add_argument('--out',type=Path)
    p.add_argument('--num-envs',type=int,default=4)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();torch.manual_seed(7)
    if args.num_envs<4 or args.steps<64:p.error('finite GPU check requires at least 4 environments and 64 steps')
    mode_checks=check_control_modes()
    for name,sha in RECIPE['files'].items():assert hashlib.sha256((args.source/name).read_bytes()).hexdigest()==sha,name
    n=64;nominal=torch.zeros(6);recipe=Recipe(n,'cpu',nominal,randomize=False)
    recipe.actions.normal_();recipe.previous.normal_();recipe.before.normal_()
    state={k:torch.randn(n,d) for k,d in [('pos',6),('vel',6),('acc',6),('torque',6),('lin',3),('ang',3),('gravity',3)]}
    state.update(height=torch.rand(n)*.4,pitch=torch.randn(n)*.2,roll=torch.randn(n)*.2,
        wheel_pos=torch.randn(n,2,3),wheel_vel=torch.randn(n,2,3),wheel_contact=torch.rand(n,2)*10,
        bad_contact=torch.rand(n)*5)
    terminated=torch.rand(n)>.5
    expected=source_oracle(args.source,state,recipe,terminated);got=recipe.reward_terms(state,terminated)
    errors={k:float((got[k]-expected[k]).abs().max()) for k in expected}
    assert max(errors.values())<1e-6,errors
    recipe.reset(torch.arange(n));obs=recipe.observations(state)
    assert obs['actor'].shape==(n,35) and obs['critic'].shape==(n,78)
    clean=core_obs(state['ang'],state['gravity'],state['pos'].clone().index_fill(1,torch.tensor([4,5]),0),state['vel'],recipe.actions)
    assert torch.equal(obs['actor'],clean)
    assert torch.equal(obs['critic'][:,:28],clean[:,:28])
    delay=Delay(2,1,4,'cpu');delay.lag[:]=torch.tensor([1,3])
    for step in range(6):
        result=delay.compute(torch.full((2,1),float(step)))[:,0]
        assert result.tolist()==[max(0,step-1),max(0,step-3)]
    delay.reset(torch.tensor([1]),1,4,False)
    assert delay.count.tolist()==[4,0] and torch.count_nonzero(delay.buf[1])==0
    # Action scaling, two independent delay groups, zero torque during first .2s.
    recipe.actions[:]=1
    assert torch.count_nonzero(recipe.torque(state['pos'],state['vel'],torch.zeros(n)))==0
    torque=recipe.torque(torch.zeros(n,6),torch.zeros(n,6),torch.full((n,),11))
    assert torch.allclose(torque,torch.tensor([30.,30.,30.,30.,1.,1.]).expand(n,-1))
    recipe.effort_scale[:]=.8
    torque=recipe.torque(torch.zeros(n,6),torch.zeros(n,6),torch.full((n,),11))
    assert torch.allclose(torque,torch.tensor([24.,24.,24.,24.,.8,.8]).expand(n,-1))
    torque=recipe.torque(torch.full((n,6),-1000.),torch.zeros(n,6),torch.full((n,),11))
    assert torch.equal(torque[:,:4],torch.full((n,4),40.))
    # Motor-coordinate invariance: target limits track ref rather than zero.
    reference=torch.tensor([*LEG_REFERENCE,0.,0.],dtype=torch.float32)
    control=Recipe(1,'cpu',reference,randomize=False)
    control.reset(torch.tensor([0]))
    assert (DT,DECIMATION,STEP_DT)==(.002,10,.02)
    assert torch.equal(control.kp,torch.tensor([[60.,60.,60.,60.,0.,0.]]))
    assert torch.equal(control.kd,torch.tensor([[2.,2.,2.,2.,.1,.1]]))
    damping=control.torque(reference[None],torch.ones(1,6),torch.tensor([11]))
    assert torch.allclose(damping,torch.tensor([[-2.,-2.,-2.,-2.,-.1,-.1]]))
    # Reduce gains to observe clamped position targets without motor saturation.
    control.kp[:,:4]=.1
    for sign in (-1.,1.):
        control.actions[:,:4]=sign*100
        for _ in range(2):
            torque=control.torque(reference[None],torch.zeros(1,6),torch.tensor([11]))
        assert torch.allclose(torque[:,:4],torch.full((1,4),sign*.314),atol=1e-6)
    # Independent prescribed curve: compression 0/45/90 mm -> 0/75/150 N.
    q=np.array([.01,-.035,-.08,-.2]);v=np.zeros(4)
    assert np.allclose(gas_spring_force(q,v),[0,75,150,150])
    assert np.allclose(gas_spring_force(np.zeros(2),np.array([1.,-10.])),[0,150])
    assert np.allclose(gas_spring_force(np.array([-.2]),np.array([1.])),[100])
    assert np.allclose(gas_spring_force(torch.tensor(q),torch.tensor(v)).numpy(),[0,75,150,150])
    cpu=CpuPlayback();max_pin=0.;motor_response=0.;spring_min=150.;spring_max=0.
    assert cpu.obs['actor'][0,31]==1 and cpu.obs['actor'][0,28]==0
    assert (cpu.model.njnt,cpu.model.neq,cpu.model.ntendon,cpu.model.nu)==(19,6,0,8)
    # Read actual constraint parameters from both compiled backends, not constants
    # from the Python controller. All six connects use a safe 5 ms response.
    assert np.allclose(cpu.model.eq_solref,np.array([[.005,1.]]*6))
    assert np.allclose(cpu.model.eq_solimp,np.array([[.95,.99,.001,.5,2.]]*6))
    assert cpu.model.eq_solref[:,0].min()>=2*DT
    pair_names=[(side+'_rear_pin'+pin,side+'_front_pin'+pin)
                for side in ('left','right') for pin in ('1','2')]
    pair_names.extend((side+'_spring_tip',side+'_gas_spring_lower') for side in ('left','right'))
    cpu_peaks=np.zeros(6)
    observe=cpu.recipe.observe_substep
    def observe_cpu_substep(*a,**kw):
        for j,(first,second) in enumerate(pair_names):
            cpu_peaks[j]=max(cpu_peaks[j],np.linalg.norm(
                cpu.data.site(first).xpos-cpu.data.site(second).xpos))
        return observe(*a,**kw)
    cpu.recipe.observe_substep=observe_cpu_substep
    for pose in ('reset','nominal'):
        from ct_mjlab.asset import POSES
        import mujoco
        d=mujoco.MjData(cpu.model)
        for name,value in POSES[pose].items():d.qpos[cpu.model.jnt_qposadr[cpu.model.joint(name).id]]=value
        mujoco.mj_forward(cpu.model,d)
        for side in ('left','right'):
            assert np.linalg.norm(d.site(side+'_spring_tip').xpos-d.site(side+'_gas_spring_lower').xpos)<1e-8
    for i in range(60):
        action=torch.tensor([[.05,-.03,-.05,.03,.1,-.1]])
        s,done,terms=cpu.step(action)
        assert np.isfinite(cpu.data.qpos).all() and torch.isfinite(s['vel']).all()
        motor_response=max(motor_response,float(s['vel'].abs().max()))
        for side in ('left','right'):
            error=np.linalg.norm(cpu.data.site(side+'_spring_tip').xpos-cpu.data.site(side+'_gas_spring_lower').xpos)
            max_pin=max(max_pin,error)
            for pin in ('1','2'):
                error=np.linalg.norm(cpu.data.site(side+'_rear_pin'+pin).xpos-cpu.data.site(side+'_front_pin'+pin).xpos)
                max_pin=max(max_pin,error)
        force=cpu.data.actuator_force[cpu.gas]
        assert np.all((force>=0)&(force<=150))
        assert np.allclose(cpu.data.qfrc_actuator[cpu.spring_v],force)
        spring_min=min(spring_min,float(force.min()));spring_max=max(spring_max,float(force.max()))
    max_pin=max(max_pin,float(cpu_peaks.max()))
    assert max_pin<.005 and motor_response>0,(max_pin,motor_response)
    with tempfile.TemporaryDirectory(prefix='ct-mjlab-export-') as tmp:
        path=Path(tmp)/'model.pt';actor=actor_model()
        torch.save({'actor_state_dict':actor.state_dict(),'infos':{'interface':manifest()}},path)
        exported=export(path,Path(tmp)/'policy.onnx')
        load_actor(path)
        try:check_manifest(manifest(),training=True)
        except ValueError:pass
        else:raise AssertionError('r5 PPO resume into r6 accepted')
        incompatible=manifest();incompatible['interface']='chuanliantui-mjlab-huananhu-v14-flat-standup-r4'
        torch.save({'actor_state_dict':actor.state_dict(),'infos':{'interface':incompatible}},path)
        try:load_actor(path)
        except ValueError:pass
        else:raise AssertionError('r4 checkpoint accepted after PD/zero/timing change')
    # Runner construction verifies library config compatibility, without learn().
    from ct_mjlab.environment import ChuanliantuiEnv
    gpu_max_pin=0.
    if args.gpu:
        env=ChuanliantuiEnv(args.num_envs,'cuda:0',seed=args.seed,randomize=True)
        assert np.allclose(env.sim.mj_model.eq_solref,cpu.model.eq_solref)
        assert np.allclose(env.sim.mj_model.eq_solimp,cpu.model.eq_solimp)
        initial_ground_count=int(env.ground_start.sum())
        assert torch.equal(env.obs['actor'][:,31].bool(),env.ground_start)
        assert torch.equal(env.obs['actor'][:,28:35],env.obs['critic'][:,32:39])
        from ct_mjlab.policy import runner_class
        runner=runner_class()(env,runner_cfg(),log_dir=None,device='cuda:0')
        pins=[(env.sim.mj_model.site('robot/'+side+'_rear_pin'+pin).id,
               env.sim.mj_model.site('robot/'+side+'_front_pin'+pin).id)
              for side in ('left','right') for pin in ('1','2')]
        pins.extend((env.sim.mj_model.site('robot/'+side+'_spring_tip').id,
                     env.sim.mj_model.site('robot/'+side+'_gas_spring_lower').id) for side in ('left','right'))
        # Capture every 2 ms substep, including states before an automatic reset.
        # Observing only policy boundaries can miss the largest pin separation.
        first_ids=torch.tensor([a for a,b in pins],device='cuda:0')
        second_ids=torch.tensor([b for a,b in pins],device='cuda:0')
        gpu_peaks=torch.zeros(6,device='cuda:0')
        forward=env.sim.forward
        def observe_gpu_forward(*a,**kw):
            forward(*a,**kw)
            error=torch.linalg.vector_norm(env.sim.data.site_xpos[:,first_ids]-env.sim.data.site_xpos[:,second_ids],dim=-1)
            gpu_peaks.copy_(torch.maximum(gpu_peaks,error.amax(0)))
        env.sim.forward=observe_gpu_forward
        episode_resets=0
        for _ in range(args.steps):
            obs,rew,done,extras=env.step(torch.zeros(args.num_envs,6,device='cuda:0'))
            episode_resets+=int(done.sum())
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['critic']).all() and torch.isfinite(rew).all()
            assert torch.equal(obs['actor'][:,28:35],env.recipe.mode.observation())
            assert torch.equal(obs['actor'][:,28:35],obs['critic'][:,32:39])
            assert torch.equal(obs['actor'][:,28]+obs['actor'][:,31],torch.ones(args.num_envs,device='cuda:0'))
            assert 'time_outs' in extras
            assert not env.last_numerical_reset.any(), 'numerical reset hid unstable physics'
            assert torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all()
            force=env.sim.data.actuator_force[:,env.gas]
            assert ((force>=0)&(force<=150)).all()
            for a,b in pins:
                gpu_max_pin=max(gpu_max_pin,float(torch.linalg.vector_norm(
                    env.sim.data.site_xpos[:,a]-env.sim.data.site_xpos[:,b],dim=-1).max()))
        env.sim.forward=forward
        gpu_max_pin=max(gpu_max_pin,float(gpu_peaks.max()))
        assert gpu_max_pin<.005,gpu_max_pin
        env.external_wrench_b[:,0]=1.
        env._write_external_wrench()
        # Independent native MuJoCo rotation matrix, not the implementation's
        # quaternion helper, is the oracle for the body-local force conversion.
        rotation=env.sim.data.xmat[:,env.base_id].reshape(args.num_envs,3,3)
        expected_force=(rotation@env.external_wrench_b[:,:3,None]).squeeze(-1)
        assert torch.allclose(env.sim.data.xfrc_applied[:,env.base_id,:3],expected_force)
        # Partial episode reset clears only selected histories and duration.
        counts=[d.count.clone() for d in env.recipe.obs_delays]
        mode_before=env.recipe.mode.observation().clone()
        untouched=torch.arange(args.num_envs,device='cuda:0')
        untouched=untouched[untouched!=1]
        env.reset(torch.tensor([1],device='cuda:0'))
        assert env.duration[1]==0
        assert torch.equal(env.recipe.mode.observation()[untouched],mode_before[untouched])
        assert env.recipe.mode.recover[1]==env.ground_start[1]
        assert env.recipe.mode.state_steps[1]==0 and not env.recipe.mode.recovered[1]
        assert torch.count_nonzero(env.external_wrench_b[1])==0
        for d,count in zip(env.recipe.obs_delays,counts):
            assert d.count[1]==0 and torch.equal(d.count[untouched],count[untouched])
        env.close()
    record=dict(interface=manifest(),source_reward_max_abs_error=errors,cpu_max_pin_error_m=max_pin,
        closure_checks=dict(threshold_m=.005,sampled_every_physics_substep=True,
            solref=cpu.model.eq_solref.tolist(),solimp=cpu.model.eq_solimp.tolist(),
            cpu_pair_max_m=cpu_peaks.tolist(),
            gpu_pair_max_m=gpu_peaks.cpu().tolist() if args.gpu else None,
            gpu_num_envs=args.num_envs if args.gpu else None,
            gpu_steps=args.steps if args.gpu else None,gpu_seed=args.seed if args.gpu else None,
            gpu_initial_ground_count=initial_ground_count if args.gpu else None,
            gpu_episode_resets=episode_resets if args.gpu else None,
            note='zero requested actions still engage position PD; ordinary resets are not standing success'),
        control_mode_checks=mode_checks,
        reference_control_checks=dict(gains_checked=True,ref_relative_limits_checked=True,
            physics_dt_s=DT,decimation=DECIMATION,policy_dt_s=STEP_DT),
        cpu_spring_force_min_n=spring_min,cpu_spring_force_max_n=spring_max,
        cpu_max_joint_speed=motor_response,onnx_max_abs_error=exported['max_abs_error'],
        gpu_checked=args.gpu,gpu_max_pin_error_m=gpu_max_pin if args.gpu else None,
        scope='finite math/physics/interface checks; no training or behavior acceptance')
    if args.out:args.out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
