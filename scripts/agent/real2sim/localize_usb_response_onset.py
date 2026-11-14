"""Locate first observable replay discrepancies; do not infer motor faults from ordering."""
import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .replay_usb_pd_targets import sha,write


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output;out.mkdir(exist_ok=False);root=a.experiment;run=root/'wheel-mechanics-validation-01'
    source=root/'mujoco-target-replay-01/inputs.npz'
    with np.load(source) as f:real={k:f[k].copy() for k in f.files}
    plan=json.loads((run/'plan.json').read_text())
    with np.load(run/'traces.npz') as f:v=f['motor_velocity'].copy();q=f['qpos'].copy();request=f['request'].copy();starts=f['starts'].copy()
    model=mujoco.MjModel.from_binary_path(str(root/'mujoco-target-replay-01/model.mjb'));data=mujoco.MjData(model)
    qi=np.asarray([int(model.joint('robot/'+name).qposadr[0]) for name in ('lf0','lf00','lfwheel','rf0','rf00','rfwheel')]);legs=[0,1,3,4];wheels=[2,5];n=len(starts)
    horizons=[0,1,2,5,10,25,50,100];rows=[]
    for j,c in enumerate(plan['profiles']):
        row=dict(profile=c,horizons=[])
        for k in horizons:
            e=v[k,j*n:(j+1)*n]-real['velocity'][starts+k]
            eq=np.rad2deg(q[k,j*n:(j+1)*n][:,qi[legs]]-real['current'][starts+k][:,legs]);dv=real['velocity'][starts+k]-real['velocity'][starts]
            row['horizons'].append(dict(elapsed_ms=k*2,velocity_error_RMS_rad_s=np.sqrt(np.mean(e**2,axis=0)).tolist(),
                observed_delta_velocity_RMS_rad_s=np.sqrt(np.mean(dv**2,axis=0)).tolist(),
                leg_position_error_RMS_deg=np.sqrt(np.mean(eq**2,axis=0)).tolist(),
                maximum_velocity_error_rad_s=float(abs(e).max()),maximum_leg_position_error_deg=float(abs(eq).max())))
        rows.append(row)
    # 只做初态运动学检查：主动关节、闭链销点位置与相对速度是否自洽，不推进CPU物理。
    with np.load(root/'short-window-final-01/trajectories.npz') as f:initial_v=f['initial_qvel'][:n].copy()
    closure=[];closure_speed=[]
    for i in range(n):
        data.qpos[:]=q[0,i];mujoco.mj_kinematics(model,data);mujoco.mj_comPos(model,data)
        for side in ('left','right'):
            for pin in (1,2):
                first=model.site(f'robot/{side}_rear_pin{pin}').id;second=model.site(f'robot/{side}_front_pin{pin}').id
                j1=np.zeros((3,model.nv));j2=np.zeros_like(j1);mujoco.mj_jacSite(model,data,j1,None,first);mujoco.mj_jacSite(model,data,j2,None,second)
                closure.append(float(np.linalg.norm(data.site_xpos[first]-data.site_xpos[second])))
                closure_speed.append(float(np.linalg.norm((j1-j2)@initial_v[i])))
    initial_torque_error=float(abs(request[0,:n]-real['recorded_torque'][starts]).max())
    w=int(np.flatnonzero(starts==8500)[0]);example=[]
    for k in horizons:
        example.append(dict(elapsed_ms=k*2,simulated_delta_velocity_rad_s=(v[k,w]-v[0,w]).tolist(),
            observed_delta_velocity_rad_s=(real['velocity'][8500+k]-real['velocity'][8500]).tolist(),velocity_error_rad_s=(v[k,w]-real['velocity'][8500+k]).tolist()))
    alignment=json.loads((root/'early-torque-alignment-01/result.json').read_text());shift_rows=[]
    for part in ('first16s','last8s'):
        for ms in (2,10,20):
            b=next(e for e in alignment['rows'][0]['errors'] if e['part']==part and e['time_ms']==ms)
            for group,metric in (('wheels','wheels_RMS_rad_s'),('legs','legs_RMS_rad_s'),('all','legs_RMS_rad_s')):
                choices=[(r['profile'],next(e for e in r['errors'] if e['part']==part and e['time_ms']==ms)) for r in alignment['rows'] if r['profile']['group']==group or r['profile']['offset_ms']==0]
                best=min(choices,key=lambda x:x[1][metric]);shift_rows.append(dict(part=part,elapsed_ms=ms,group=group,best_offset_ms=best[0]['offset_ms'],
                    legs_error_ratio=best[1]['legs_RMS_rad_s']/b['legs_RMS_rad_s'],wheels_error_ratio=best[1]['wheels_RMS_rad_s']/b['wheels_RMS_rad_s']))
    write(out/'summary.json',dict(motor_order=['lf0','lf00','lfwheel','rf0','rf00','rfwheel'],rows=rows,window17_example=example,
        initial_request_error_Nm=initial_torque_error,initial_pin_closure_max_m=max(closure),initial_pin_relative_speed_max_m_s=max(closure_speed),
        alignment_best_by_group=shift_rows,limitations=['2ms is logging/simulation resolution, not a causal ordering of real motor faults.',
        'Motor sample age/current and base translation not measured; cannot separate drive response, plant forces and exact initial support.',
        'IMU is100Hz held data; no sub-10ms attitude onset ranking. Shift selection uses already-inspected single-session data.'],
        new_physics_steps=0,training_updates=0,source_sha256={str(path):sha(path) for path in (source,run/'traces.npz',root/'early-torque-alignment-01/result.json')}))
    np.savez_compressed(out/'window17.npz',time_ms=np.arange(101)*2,real_velocity=real['velocity'][8500:8601],sim_velocity=v[:,w],request=real['recorded_torque'][8500:8600])
    fig,axes=plt.subplots(3,2,figsize=(11,8),sharex=True)
    for i,ax in enumerate(axes.flat):
        ax.plot(np.arange(11)*2,real['velocity'][8500:8511,i]-real['velocity'][8500,i],label='Measured',color='black',lw=2)
        ax.plot(np.arange(11)*2,v[:11,w,i]-v[0,w,i],label='Original replay',color='#008bc4',lw=1.5)
        ax.set_title(['Left-front leg','Left-rear leg','Left wheel','Right-front leg','Right-rear leg','Right wheel'][i]);ax.grid(alpha=.25);ax.set_ylabel('Change in speed (rad/s)');ax.legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Elapsed time (ms)')
    fig.suptitle('First20ms of log17.000s window / same initial joint positions and speeds\nMotor-axis signs retained; recorded torque input, no policy inference',fontsize=11);fig.tight_layout();fig.savefig(out/'early-response.png',dpi=150);plt.close(fig)
    shutil.copy2(__file__,out/'source.py');print(json.dumps(dict(initial_request_error_Nm=initial_torque_error,initial_pin_closure_max_m=max(closure),initial_pin_relative_speed_max_m_s=max(closure_speed),example10ms=example[3]),indent=2))


if __name__=='__main__':main()
