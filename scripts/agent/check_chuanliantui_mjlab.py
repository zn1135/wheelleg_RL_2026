#!/usr/bin/env python3
"""Finite migration checks. No PPO learning or policy behavior acceptance.

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
from ct_mjlab.config import RECIPE, PARAMS, WEIGHTS, JOINTS, manifest, runner_cfg, STEP_DT
from ct_mjlab.recipe import Recipe, Delay, core_obs
from ct_mjlab.cpu import CpuPlayback
from ct_mjlab.policy import actor_model, load_actor, export
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_from_euler_xyz

DEFAULT_SOURCE=Path('/home/zn/文档/wheeled-legged_RL-main_huananhu/source/agent_tasks/agent_tasks/direct/wheelbipe')

def source_oracle(source,state,recipe,terminated):
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
    args=p.parse_args();torch.manual_seed(7)
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
    assert torch.allclose(torque,torch.tensor([5.,5.,5.,5.,1.,1.]).expand(n,-1))
    recipe.effort_scale[:]=.8
    torque=recipe.torque(torch.zeros(n,6),torch.zeros(n,6),torch.full((n,),11))
    assert torch.allclose(torque,torch.tensor([4.,4.,4.,4.,.8,.8]).expand(n,-1))
    torque=recipe.torque(torch.full((n,6),-1000.),torch.zeros(n,6),torch.full((n,),11))
    assert torch.equal(torque[:,:4],torch.full((n,4),40.))
    cpu=CpuPlayback();max_pin=0.;motor_response=0.
    for i in range(60):
        action=torch.tensor([[.05,-.03,-.05,.03,.1,-.1]])
        s,done,terms=cpu.step(action)
        assert np.isfinite(cpu.data.qpos).all() and torch.isfinite(s['vel']).all()
        motor_response=max(motor_response,float(s['vel'].abs().max()))
        for side in ('left','right'):
            for pin in ('1','2'):
                error=np.linalg.norm(cpu.data.site(side+'_rear_pin'+pin).xpos-cpu.data.site(side+'_front_pin'+pin).xpos)
                max_pin=max(max_pin,error)
    assert max_pin<.005 and motor_response>0,(max_pin,motor_response)
    with tempfile.TemporaryDirectory(prefix='ct-mjlab-export-') as tmp:
        path=Path(tmp)/'model.pt';actor=actor_model()
        torch.save({'actor_state_dict':actor.state_dict(),'infos':{'interface':manifest()}},path)
        exported=export(path,Path(tmp)/'policy.onnx')
        incompatible=manifest();incompatible['interface']='legacy-virtual'
        torch.save({'actor_state_dict':actor.state_dict(),'infos':{'interface':incompatible}},path)
        try:load_actor(path)
        except ValueError:pass
        else:raise AssertionError('legacy checkpoint accepted')
    # Runner construction verifies library config compatibility, without learn().
    from ct_mjlab.environment import ChuanliantuiEnv
    gpu_max_pin=0.
    if args.gpu:
        env=ChuanliantuiEnv(4,'cuda:0',randomize=True)
        from ct_mjlab.policy import runner_class
        runner=runner_class()(env,runner_cfg(),log_dir=None,device='cuda:0')
        pins=[(env.sim.mj_model.site('robot/'+side+'_rear_pin'+pin).id,
               env.sim.mj_model.site('robot/'+side+'_front_pin'+pin).id)
              for side in ('left','right') for pin in ('1','2')]
        for _ in range(64):
            obs,rew,done,extras=env.step(torch.zeros(4,6,device='cuda:0'))
            assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['critic']).all() and torch.isfinite(rew).all()
            assert 'time_outs' in extras
            assert not env.last_numerical_reset.any(), 'numerical reset hid unstable physics'
            assert torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all()
            for a,b in pins:
                gpu_max_pin=max(gpu_max_pin,float(torch.linalg.vector_norm(
                    env.sim.data.site_xpos[:,a]-env.sim.data.site_xpos[:,b],dim=-1).max()))
        assert gpu_max_pin<.005,gpu_max_pin
        env.external_wrench_b[:,0]=1.
        env._write_external_wrench()
        # Independent native MuJoCo rotation matrix, not the implementation's
        # quaternion helper, is the oracle for the body-local force conversion.
        rotation=env.sim.data.xmat[:,env.base_id].reshape(4,3,3)
        expected_force=(rotation@env.external_wrench_b[:,:3,None]).squeeze(-1)
        assert torch.allclose(env.sim.data.xfrc_applied[:,env.base_id,:3],expected_force)
        # Partial episode reset clears only selected histories and duration.
        counts=[d.count.clone() for d in env.recipe.obs_delays]
        env.reset(torch.tensor([1],device='cuda:0'))
        assert env.duration[1]==0
        assert torch.count_nonzero(env.external_wrench_b[1])==0
        for d,count in zip(env.recipe.obs_delays,counts):
            assert d.count[1]==0 and torch.equal(d.count[[0,2,3]],count[[0,2,3]])
        env.close()
    record=dict(interface=manifest(),source_reward_max_abs_error=errors,cpu_max_pin_error_m=max_pin,
        cpu_max_joint_speed=motor_response,onnx_max_abs_error=exported['max_abs_error'],
        gpu_checked=args.gpu,gpu_max_pin_error_m=gpu_max_pin if args.gpu else None,
        scope='finite math/physics/interface checks; no training or behavior acceptance')
    if args.out:args.out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
