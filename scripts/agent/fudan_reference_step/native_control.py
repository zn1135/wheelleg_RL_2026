"""MuJoCo CPU diagnostic cross-check, explicitly not final GPU acceptance."""
import argparse
import json
from pathlib import Path
import math
import time
import mujoco
import numpy as np
from .replay import module,scene,write,mujoco_rotation,sha


def execute(out,label,case,height=.2,policy_dt=.004,seconds=8.):
    directory=out/'runs'/label;directory.mkdir(parents=True,exist_ok=False)
    ref=module(out);args=ref.build_arg_parser().parse_args([])
    args.xml=str(scene(out,'single',height,8.));args.onnx=str(out/'source/mujoco/actor/yuntai/上台阶3_angz+.onnx');args.device='cpu'
    args.sim_dt=.002;args.policy_dt=policy_dt
    state=ref.CommandState();state.cmd_height=case.get('cmd_height',.27)
    rt=ref.BinglianRuntime(args,state);m,d=rt.m,rt.d
    yaw=math.radians(case.get('angle',0));d.qpos[:3]=[-.9,-.9*math.tan(yaw),.21];d.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
    mujoco.mj_forward(m,d)
    dt=float(m.opt.timestep);every=round(policy_dt/dt)
    wbids=[m.body(n).id for n in ('l_wheel_Link','r_wheel_Link')];base=m.body('base_Link_del').id
    motor_ids=[rt.handles.lf0,rt.handles.lw,rt.handles.rf0,rt.handles.rw,rt.handles.lf20,rt.handles.rf20]
    record={k:[] for k in ['time','qpos','qvel','wheel_pos','wheel_force','wheel_support','wheel_side_force','base_force','leg_force','base_contact_pos','leg_contact_pos','motor_torque','virtual_q','actions','command','pitch_deg','roll_deg','virtual_torque']}
    started=time.perf_counter()
    for tick in range(round(seconds/dt)):
        if tick%every==0:
            state.cmd_lin_vel=max(0,min(1,(tick*dt-1)/.5))*case['speed']
            if d.xpos[wbids,0].min()>.45:state.cmd_lin_vel=0.
            a,g,q,qd=rt.build_state();obs=rt.build_obs(a,g,q,qd);rt.update_history(obs);action=rt.run_policy(obs)
        else:_,_,q,qd=rt.build_state()
        rt.apply_ctrl(rt.compute_control(action,q,qd));mujoco.mj_step(m,d)
        wf=np.zeros((2,3));support=np.zeros(2);side=np.zeros(2);bf=lf=0.;bp=np.zeros(3);lp=bp.copy()
        for j in range(d.ncon):
            c=d.contact[j];a,b=m.geom_bodyid[c.geom];force=np.zeros(6);mujoco.mj_contactForce(m,d,j,force);f=c.frame.reshape(3,3).T@force[:3];mag=np.linalg.norm(f)
            if a and b:continue
            for body,sign in ((a,-1),(b,1)):
                if body in wbids:
                    k=wbids.index(body);wf[k]+=sign*f
                    if abs(c.frame[2])>.7:support[k]+=mag
                    else:side[k]=max(side[k],mag)
                if body==base and mag>bf:bf=mag;bp=c.pos.copy()
                if body>0 and body!=base and body not in wbids and mag>lf:lf=mag;lp=c.pos.copy()
        R=mujoco_rotation(d.qpos[3:7]);item=dict(time=d.time,qpos=d.qpos.copy(),qvel=d.qvel.copy(),wheel_pos=d.xpos[wbids].copy(),wheel_force=wf,wheel_support=support,wheel_side_force=side,
            base_force=bf,leg_force=lf,base_contact_pos=bp,leg_contact_pos=lp,motor_torque=d.ctrl[motor_ids].copy(),virtual_q=q.copy(),actions=action.copy(),command=[state.cmd_lin_vel,0,state.cmd_height],
            pitch_deg=np.rad2deg(np.arcsin(np.clip(-R[2,0],-1,1))),roll_deg=np.rad2deg(np.arctan2(R[2,1],R[2,2])),virtual_torque=rt.active_p_gains*(action*np.array([.5,.5,0,.5,.5,0])+rt.active_default_dof_pos-q)+rt.active_d_gains*(action*np.array([0,0,10,0,0,10])-qd))
        for key,val in item.items():record[key].append(val)
        if not np.isfinite(d.qpos).all() or d.warning.number.any():raise FloatingPointError('Native MuJoCo numerical warning')
    np.savez_compressed(directory/(case['id']+'.npz'),**{k:np.asarray(v) for k,v in record.items()},complete=True,error='')
    mujoco.mj_saveModel(m,str(directory/'model.mjb'))
    write(directory/'metadata.json',dict(kind='single',height=height,width=8.,physics_dt=dt,pd_dt=dt,policy_dt=policy_dt,physics_backend='native_mujoco_cpu_diagnostic',mujoco=mujoco.__version__,specs=[case],onnx_sha256=sha(Path(args.onnx)),optimizer_updates=0))
    write(directory/'execution.json',dict(complete=[True],worlds=1,physics_steps=len(record['time']),wall_s=time.perf_counter()-started,optimizer_updates=0))
    print('NATIVE_FINISHED',label,time.perf_counter()-started,flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    execute(a.output,'native160',dict(id='v2.00_a+00',speed=2.,angle=0),height=.16)
    execute(a.output,'native200',dict(id='v2.00_a-20',speed=2.,angle=-20),height=.2)


if __name__=='__main__':main()
