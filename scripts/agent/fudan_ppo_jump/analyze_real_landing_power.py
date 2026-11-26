"""实机两跳的落地姿态/支撑时序及40Nm同路径做功余量，离线诊断。"""
import argparse
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .analyze_real022 import read_csv, matrix
from .probe_takeoff_work import ROOT, write_json
from .render_real_log import nearest
from ct_mjlab.envs.chuanliantui.model import load_model, MODEL_XML
from ct_mjlab.envs.chuanliantui.fudan_ppo_walk.reset_geometry import mesh_vertices
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha

ORDER = ('lf', 'lr', 'rf', 'rr')
MODEL_SIGNS = np.array([1., 1., -1., -1.])
EXTENSION_SIGNS = np.array([-1., 1., -1., 1.])


def extension(torque_model):
    return np.stack((torque_model[:, 1]-torque_model[:, 0], torque_model[:, 2]-torque_model[:, 3]), axis=1)


def interpolate(time, values, target):
    return np.stack([np.interp(target, time, values[:, j]) for j in range(values.shape[1])], axis=1)


def work_summary(time, angle, request, feedback, end, mass, *, ground_window=False):
    grid = np.r_[0., time[(time>0)&(time<end)], end]
    positions = interpolate(time, angle, grid)
    tau_feedback = interpolate(time, feedback, grid)
    source = (np.searchsorted(time, grid[:-1], side='right')-1).clip(0)
    dq = np.diff(positions, axis=0)
    wr = request[source]*dq
    wf = .5*(tau_feedback[:-1]+tau_feedback[1:])*dq
    cap = 40*abs(dq)
    span = np.diff(grid)
    request_gap = float(cap.sum()-wr.sum())
    feedback_gap = float(cap.sum()-wf.sum())
    return dict(end_s=float(end), request_work_J=float(wr.sum()), reported_feedback_work_J=float(wf.sum()),
        same_path_40Nm_bound_J=float(cap.sum()), request_fraction_of_bound=float(wr.sum()/cap.sum()),
        reported_feedback_fraction_of_bound=float(wf.sum()/cap.sum()),
        request_work_by_motor_J=wr.sum(0).tolist(), reported_feedback_work_by_motor_J=wf.sum(0).tolist(),
        reported_positive_work_J=float(wf.clip(0).sum()), reported_negative_work_J=float(wf.clip(max=0).sum()),
        axis_absolute_travel_rad=abs(dq).sum(0).tolist(),
        time_fraction_request_atleast38Nm=((span[:, None]*(abs(request[source])>=38)).sum(0)/span.sum()).tolist(),
        unrequested_same_path_work_J=request_gap,
        request_minus_reported_work_J=float(wr.sum()-wf.sum()),
        same_path_minus_reported_work_J=feedback_gap,
        height_conversion_uses_earlier_takeoff_boundary=ground_window,
        ideal_extra_COM_height_from_unrequested_work_at_model_mass_cm=100*request_gap/(mass*9.81) if ground_window else None,
        ideal_extra_COM_height_from_total_feedback_gap_at_model_mass_cm=100*feedback_gap/(mass*9.81) if ground_window else None)


def run(archive_path, alignment, pose_file, output):
    output.mkdir(parents=True, exist_ok=False)
    aligned = json.loads(alignment.read_text())
    assert sha(archive_path) == aligned['archive_sha256']
    model=load_model();data=mujoco.MjData(model);mass=float(model.body_mass.sum())
    test_time=np.linspace(0.,.1,21)
    test_angle=np.tile((2*test_time)[:,None],(1,4))
    test_torque=np.full_like(test_angle,20.)
    test_work=work_summary(test_time,test_angle,test_torque,test_torque,.083,mass)
    assert abs(test_work['request_work_J']-4*20*2*.083)<1e-12
    assert abs(test_work['reported_feedback_work_J']-4*20*2*.083)<1e-12
    wheel_ids=[model.body(n).id for n in ('lfwheel','rfwheel')]
    geoms=[[i for i in range(model.ngeom) if model.geom_bodyid[i]==bid and (model.geom_contype[i] or model.geom_conaffinity[i])] for bid in wheel_ids]
    poses=np.load(pose_file)
    result=dict(archive_sha256=sha(archive_path), alignment_source=str(alignment), pose_source=str(pose_file),
        model_mass_example_kg=mass, mass_is_real_measurement=False, model_sha256=sha(MODEL_XML), trials=[],
        constant_torque_partial_interval_check_J=test_work['request_work_J'],
        wheel_radius_example_m=.0525,
        work_definition='Request held to next telemetry sample; reported torque averaged across actual joint displacement. Same firmware coordinates for angle and torque.',
        limitations=['No load-cell or shaft-torque calibration; reported motor torque is an estimate.',
            'The 40Nm work bound holds the recorded joint path fixed and is not a realizable control or a global jump maximum.',
            'Additional work becomes additional COM height only under a lossless vertical-energy assumption; real mass unprovided.',
            'Video contacts are manual intervals at30fps; alignment has no hardware common clock.'],
        physics_steps=0, learning_updates=0)
    traces=[]
    with zipfile.ZipFile(archive_path) as archive:
        for row in aligned['trials']:
            trial=row['trial'];s=read_csv(archive,f'试次{trial}-H7反馈.csv');p=read_csv(archive,f'试次{trial}-H7控制.csv');n=read_csv(archive,f'试次{trial}-模型输入动作.csv')
            t=s['rel_s'];pt=p['rel_s'];angle=matrix(s,[f'hip_pos_rad_{j}' for j in range(4)])
            request=matrix(s,[f'request_Nm_{j}' for j in range(4)]);feedback=matrix(s,[f'hip_feedback_Nm_{j}' for j in range(4)])
            pair=nearest(s['board_us'],p['board_us']);valid=(pt>=0)&(pt<1)&(p['fault']==0)
            control_request=matrix(p,[f'request_Nm_{j}' for j in range(4)])
            pair_error=float(abs(control_request[valid]-request[pair[valid]]).max())
            assert pair_error < 1e-6
            quaternion=matrix(s,[f'quat_{j}' for j in range(4)])
            attitude=np.rad2deg(Rotation.from_quat(quaternion[:,[1,2,3,0]]).as_euler('xyz'))
            lengths=matrix(s,['left_leg_0','right_leg_0'])
            drive=extension(request*MODEL_SIGNS);reported_drive=extension(feedback*MODEL_SIGNS)
            wheel_speed=matrix(p,['current_4','current_5'])*[1.,-1.]
            wheel_torque=matrix(p,['pd_Nm_4','pd_Nm_5'])*[1.,-1.]
            off=np.array(row['takeoff_bracket_frames'])/30-row['video_offset_s']
            land=np.array(row['touchdown_bracket_frames'])/30-row['video_offset_s']
            peak=row['legs'][0]['extension_peak_log_s']
            windows={label:work_summary(t,angle,request,feedback,end,mass,
                     ground_window=label=='earlier_video_takeoff_boundary') for label,end in (
                ('earlier_video_takeoff_boundary',off[0]),('later_video_takeoff_boundary',off[1]),('maximum_extension',peak))}
            lags=[]
            for j,name in enumerate(ORDER):
                req=(request[:,j]*EXTENSION_SIGNS[j]>=20)&(t>=0)&(t<peak)
                fb=(feedback[:,j]*EXTENSION_SIGNS[j]>=20)&(t>=0)&(t<peak)
                req_i=np.flatnonzero(req);fb_i=np.flatnonzero(fb)
                lags.append(dict(motor=name,request20_at_ms=float(t[req_i[0]]*1000),feedback20_at_ms=float(t[fb_i[0]]*1000),
                    observed_feedback_lag_ms=float((t[fb_i[0]]-t[req_i[0]])*1000)))
            # 世界竖直轮底差不受未知机身平移影响。
            pose_mask=poses['case']==trial-1
            qt=(poses['board_us'][pose_mask]-row['first_jump_board_us'])/1e6
            bottoms=[]
            for q in poses['qpos'][pose_mask]:
                data.qpos[:]=q;mujoco.mj_kinematics(model,data)
                bottoms.append([min(float(mesh_vertices(model,data,g)[:,2].min()) for g in side) for side in geoms])
            bottoms=np.asarray(bottoms);bottom_delta=np.interp(t,qt,bottoms[:,0]-bottoms[:,1])
            snapshots=[]
            for label,at in [('before_contact_40ms',land[0]-.04),('contact_earlier',land[0]),('contact_midpoint',land.mean()),
                             ('contact_later',land[1]),('contact_plus40ms',land[1]+.04),('contact_plus80ms',land[1]+.08),('contact_plus150ms',land[1]+.15)]:
                snapshots.append(dict(label=label,time_s=float(at),
                    roll_pitch_deg=interpolate(t,attitude[:,:2],np.array([at]))[0].tolist(),
                    lengths_cm=(interpolate(t,lengths,np.array([at]))[0]*100).tolist(),
                    left_minus_right_bottom_cm=float(np.interp(at,t,bottom_delta)*100),
                    wheel_forward_rad_s=interpolate(pt,wheel_speed,np.array([at]))[0].tolist(),
                    wheel_rim_speed_example_m_s=(interpolate(pt,wheel_speed,np.array([at]))[0]*.0525).tolist(),
                    wheel_forward_model_torque_Nm=interpolate(pt,wheel_torque,np.array([at]))[0].tolist(),
                    requested_extension_difference_Nm=interpolate(t,drive,np.array([at]))[0].tolist(),
                    reported_extension_difference_Nm=interpolate(t,reported_drive,np.array([at]))[0].tolist()))
            near=(t>=land[0]-.04)&(t<=land[1]+.04)
            airborne_descent=(t>=.4)&(t<=land[0])
            first100=(t>=land[1])&(t<=land[1]+.1)
            jump=np.flatnonzero((n['model']=='5200')&(n['rel_s']>=0))
            returned=int(jump[-1])+1
            return_time=float(n['rel_s'][returned])
            result['trials'].append(dict(trial=trial, takeoff_log_interval_s=off.tolist(), contact_log_interval_s=land.tolist(),
                return_walk_s=return_time, first_contact_to_walk_ms=((return_time-land[::-1])*1000).tolist(),
                work=windows, request_feedback_20Nm_timings=lags, snapshots=snapshots,
                control_and_feedback_request_match_error_Nm=pair_error,
                precontact_roll_pitch_peak_deg=abs(attitude[airborne_descent,:2]).max(0).tolist(),
                requested_retraction_direction_fraction_near_contact=(drive[near]<0).mean(0).tolist(),
                minimum_leg_length_first100ms_after_contact_cm=(lengths[first100].min(0)*100).tolist(),
                faults_near_contact=np.unique(s['fault'][near|first100]).astype(int).tolist()))
            command_mask=(n['rel_s']>=land[0]-.04)&(n['rel_s']<=land[1]+.1)
            yaw_command=n['obs_7'][command_mask]/.25
            result['trials'][-1]['landing_command_yaw_range_rad_s']=[float(yaw_command.min()),float(yaw_command.max())]
            traces.append((trial,t,request,feedback,lengths,attitude,drive,reported_drive,off,land,return_time,bottom_delta,pt,wheel_speed))
            np.savez_compressed(output/f'trial{trial}.npz',relative_s=t,angle_rad=angle,request_Nm=request,reported_feedback_Nm=feedback,
                leg_lengths_m=lengths,roll_pitch_yaw_deg=attitude,requested_extension_difference_Nm=drive,
                reported_extension_difference_Nm=reported_drive,left_minus_right_wheel_bottom_m=bottom_delta)
    write_json(output/'result.json',result)
    plt.rcParams['font.family']=['Noto Sans CJK SC','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(5,2,figsize=(15,14),constrained_layout=True)
    for col,(trial,t,req,fb,length,attitude,drive,reported,off,land,returned,bottom_delta,pt,wheel_speed) in enumerate(traces):
        early=(t>=0)&(t<=.22);late=(t>=.35)&(t<=.9)
        for j,name in enumerate(ORDER):
            axes[0,col].plot(t[early]*1000,req[early,j]*EXTENSION_SIGNS[j],label=name,color=f'C{j}')
            axes[0,col].plot(t[early]*1000,fb[early,j]*EXTENSION_SIGNS[j],color=f'C{j}',ls='--',alpha=.7)
        axes[0,col].axhline(40,color='grey',ls=':');axes[0,col].axvspan(off[0]*1000,off[1]*1000,color='grey',alpha=.12)
        axes[0,col].set_title(f'试次{trial}：起跳请求实线 / 反馈虚线');axes[0,col].set_ylabel('伸展方向电机力矩 / N·m');axes[0,col].set_xlabel('触发后 / ms');axes[0,col].legend(ncol=4)
        axes[1,col].plot(t[late],length[late,0]*100,label='左腿');axes[1,col].plot(t[late],length[late,1]*100,label='右腿');axes[1,col].set_ylabel('实际腿长 / cm')
        axes[2,col].plot(t[late],attitude[late,0],label='横滚');axes[2,col].plot(t[late],attitude[late,1],label='俯仰');axes[2,col].set_ylabel('机身姿态 / °')
        for side,name in enumerate(('左腿','右腿')):
            axes[3,col].plot(t[late],drive[late,side],color=f'C{side}',label=name+'请求')
            axes[3,col].plot(t[late],reported[late,side],color=f'C{side}',ls='--',alpha=.7)
        axes[3,col].set_ylabel('伸展差动力矩 / N·m');axes[3,col].axhline(0,color='grey',lw=.8)
        wheel_window=(pt>=.35)&(pt<=.9)
        for side,name in enumerate(('左轮','右轮')):
            axes[4,col].plot(pt[wheel_window],wheel_speed[wheel_window,side]*.0525,label=name)
        axes[4,col].set_ylabel('轮缘速度 / m/s');axes[4,col].axhline(0,color='grey',lw=.8)
        for ax in axes[1:,col]:
            ax.axvspan(land[0],land[1],color='green',alpha=.12);ax.axvline(returned,color='black',ls=':',lw=1)
            ax.set_xlabel('触发后 / s');ax.legend()
        for ax in axes[:,col]:ax.grid(alpha=.2)
    fig.suptitle('灰色为视频离地边界，绿色为首次触地区间，黑虚线为返回WALK；反馈力矩未独立标定',fontsize=14)
    fig.savefig(output/'landing-and-power.png',dpi=150);plt.close(fig)
    print(json.dumps(result['trials'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--alignment',type=Path,required=True)
    parser.add_argument('--poses',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.archive,args.alignment,args.poses,args.output)
