"""冻结 Jump 权重。按 PD 更新拍记录控制目标，在原物理子步核对实际执行力矩。"""
import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.fudan_ppo_walk import PhysicalAction
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.env import JumpEnv
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha
from . import evaluate, handoff, measure_flat
from .attribute_handoff80 import timeline

MOTORS = ('lf0','lf00','rf0','rf00')


def run(checkpoint, reference, output, *, keep_flat=False):
    output.mkdir(parents=True,exist_ok=False)
    walk,jump,saved=evaluate.policies(checkpoint,'cuda:0')
    baseline_sha=json.loads((reference.parent/'report.json').read_text())['model_sha256']
    if sha(checkpoint)!=baseline_sha:
        assert (saved.get('resume_from') or {}).get('sha256')==baseline_sha
    apply=PhysicalAction.apply_actions
    frame=JumpEnv.trace_frame
    rows=[]
    held={}
    meta={}

    def probe_pd(action):
        refreshed=action._pd_tick==0
        apply(action)
        legs=action._leg_action_ids
        sim=action._env.sim
        if refreshed:
            delayed=action._action_history[(action._history_head-action.action_delay_steps)%len(action._action_history),
                                           action._env_ids][:,legs]
            center=action._reference_pos[:,legs]
            target=(action._nominal_pos[:,legs]+action.cfg.leg_position_scale*delayed).clamp(center-3.14,center+3.14)
            q=sim.data.qpos[:,action._motor_q_adr[legs]]
            v=sim.data.qvel[:,action._motor_v_adr[legs]]
            strength=action.motor_strength[:,legs]
            prop=action.cfg.leg_stiffness*action.kp_scale[:,legs]*(target-q)
            damp=-action.cfg.leg_damping*action.kd_scale[:,legs]*v
            held.update(target=target,p=prop*strength,d=damp*strength,request=(prop+damp)*strength)
        held['refreshed']=refreshed

    def probe_frame(env):
        trace=frame(env)
        sim=env.env.sim
        a=env.action
        legs=a._leg_action_ids
        if 'motor_ids' not in meta:
            meta['motor_ids']=[sim.mj_model.actuator('robot/'+name+'_motor').id for name in MOTORS]
            meta['limit']=a._effort_limits[legs].cpu().tolist()
            meta['base_slot']=env.env.scene['jump_ground'].primary_names.index('base_link')
            for j,n in enumerate(MOTORS):
                assert sim.mj_model.joint('robot/'+n).qposadr[0]==int(a._motor_q_adr[legs[j]])
            assert not a.cfg.disturbances.action_delay and a.cfg.pd_decimation==5
        delta=sim.data.xpos[:,env.wheel_ids]-sim.data.xanchor[:,env.hip_ids]
        quat=env.robot.data.root_link_quat_w[:,None].expand(-1,2,-1)
        local=quat_apply_inverse(quat.reshape(-1,4),delta.reshape(-1,3)).reshape(-1,2,3)
        length=local[:,:,(0,2)].norm(dim=-1)
        ids=meta['motor_ids']
        base_force=env.env.scene['jump_ground'].data.force[:,meta['base_slot']].norm(dim=-1,keepdim=True)
        rows.append(torch.cat((held['target'],sim.data.qpos[:,a._motor_q_adr[legs]],held['p'],held['d'],
            held['request'],a._torques[:,legs],sim.data.ctrl[:,ids],sim.data.actuator_force[:,ids],
            length,torch.full_like(length[:,:1],float(held['refreshed'])),base_force),dim=1))
        return trace

    with TemporaryDirectory() as tmp, patch.object(handoff,'LANDING',dict(handoff.LANDING,buffer_ticks=8)), \
         patch.object(PhysicalAction,'apply_actions',probe_pd),patch.object(JumpEnv,'trace_frame',probe_frame):
        work=Path(tmp)/'flat'
        profile=measure_flat.collect_profile(work,'walk_jump_walk',walk,jump,landing_handoff=True)
        batch=torch.stack(rows).cpu().numpy()
        cases=[]
        for i in range(20):
            name=f'start{i:02}.npz'
            with np.load(work/name) as new,np.load(reference/name) as old:
                assert all(np.array_equal(old[k],new[k]) for k in ('qpos0','qvel0','history0'))
                t,mode=new['trace'],new['mode']
                active_steps=np.flatnonzero(mode)
                if keep_flat:
                    # 直接用执行交接的首次真实承重拍，避免把随后的弹跳选作首跳。
                    confirmed=-1
                    if len(active_steps):
                        expected,confirmed=handoff.expected_landing_mode(t,int(active_steps[0]))
                        assert np.array_equal(expected,mode)
                    if confirmed<0:
                        cases.append(dict(case=i,measured=False,initial_exact=True,reason='no_confirmed_touchdown'))
                        continue
                    first=10*confirmed
                    wheel=first+int(np.flatnonzero((t[first:first+10,8:10]>.5).any(1))[0])
                    switch=10*(int(active_steps[-1])+1)
                    if wheel+1000>len(t):
                        cases.append(dict(case=i,measured=False,initial_exact=True,reason='incomplete_landing_window'))
                        continue
                    other=np.flatnonzero(t[wheel:wheel+1000,10]>1.)+wheel
                else:
                    _,wheel,switch,other=timeline(t,mode)
                _,ow,os,oo=timeline(old['trace'],old['mode'])
                assert wheel+60<switch and len(oo)
                sl=slice(wheel-30,wheel+81)
                arr=batch[sl,i]
                target,q,p,d,req,limited,ctrl,actual=np.split(arr[:,:32],8,axis=1)
                limit=np.array(meta['limit'])
                assert np.max(abs(limited-np.clip(req,-limit,limit)))<1e-4
                assert np.max(abs(ctrl-limited))<1e-4 and np.max(abs(actual-ctrl))<1e-4
                assert np.max(abs(req-p-d))<1e-4
                active=mode[np.arange(wheel,wheel+61)//10]
                assert active.all() and arr.shape==(111,36) and np.isfinite(arr).all()
                np.savez_compressed(output/name,ms=np.arange(-30,81),target=target,q=q,p=p,d=d,requested=req,
                    applied=actual,ctrl=ctrl,limit=limit,length=arr[:,32:34],refresh=arr[:,34],
                    trace=t[sl],wheel_ms=wheel,switch_ms=switch,first_nonwheel_ms=int(other[0]) if len(other) else -1)
                cases.append(dict(case=i,measured=True,wheel_ms=wheel,original_wheel_ms=ow,hit_delta_ms=int(other[0]-wheel) if len(other) else None,
                    original_hit_delta_ms=int(oo[0]-ow),switch_delta_ms=switch-wheel,original_switch_delta_ms=os-ow,
                    exact_trace=np.array_equal(old['trace'],t),exact_mode=np.array_equal(mode,old['mode'])))
        if keep_flat:
            np.savez_compressed(output/'base_contact.npz',force_N=batch[:,:,35])
            shutil.copytree(work,output/'flat')
            (output/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    (output/'capture.json').write_text(json.dumps(dict(motors=MOTORS,checkpoint=sha(checkpoint),
        walk_sha=sha(evaluate.WALK),dt_ms=1,pd_dt_ms=5,policy_dt_ms=10,optimizer_updates=0,**meta,cases=cases),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('checkpoint','reference','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--keep-flat',action='store_true')
    args=parser.parse_args()
    run(args.checkpoint,args.reference,args.output,keep_flat=args.keep_flat)
