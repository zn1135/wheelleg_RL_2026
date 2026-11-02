"""Read completed timing/action probes and report measured effects without new dynamics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def first_difference(left,right,threshold,dt):
    difference=np.abs(left-right).reshape(len(left),-1)
    indices=np.flatnonzero((difference>threshold).any(axis=1))
    return dict(threshold=threshold,first_time_s=None if not len(indices) else int(indices[0])*dt,
        maximum_abs_difference=float(difference.max()))


def averaged_body_velocity(data,k):
    if k==0:return None
    poses=data['states_qpos'];q=poses[k*5,3:7].astype(np.float64);q/=np.linalg.norm(q)
    world=(poses[k*5,:3]-poses[(k-1)*5,:3])/.01
    cross=np.cross(q[1:],world)
    return world-2.*q[0]*cross+2.*np.cross(q[1:],cross)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.experiment;args.output.mkdir(parents=True,exist_ok=False)
    budget=json.loads((root/'budget.json').read_text());mapping=json.loads((Path(budget['source_recording'])/'fixtures/mapping.json').read_text())
    runs={};meta={};inputs={};summaries={}
    v=np.asarray(mapping['motor_v']);wheel_response=np.asarray(mapping['wheel_response'])
    gains=np.array([1.,1.,wheel_response[0,0],1.,1.,wheel_response[1,0]])
    for folder in ('runs','repeat-audit'):
        for path in sorted((root/folder).glob('*.npz')):
            name=path.stem;meta[name]=json.loads(path.with_suffix('.json').read_text())
            assert meta[name]['complete'] and meta[name]['pd_steps']==500 and meta[name]['states']==501
            with np.load(path,allow_pickle=False) as saved:runs[name]={k:saved[k].copy() for k in saved.files}
            d=runs[name];assert all(np.isfinite(value).all() for value in d.values())
            assert d['pd_actuator_force_Nm'].shape==(500,6) and d['policy_history'].shape==(100,125)
            assert np.max(np.abs(d['pd_actuator_force_Nm']-d['pd_generalized_force_Nm']))<1e-5
            gain_error=float(np.max(np.abs(d['pd_actuator_force_Nm']-d['pd_motor_ctrl']*gains)))
            assert gain_error<2e-5
            pitch=d['states_row'][:,13];k=int(np.abs(pitch).argmax());force=d['states_row'][:,16]
            found=np.flatnonzero(force>10.)
            summaries[name]=dict(pitch_peak_deg=float(pitch[k]),pitch_peak_s=k*.002,
                body_force_peak_N=float(force.max()),first_body_force_over10N_s=None if not len(found) else int(found[0])*.002,
                applied_force_gain_check_max_error_Nm=gain_error,
                warmstart_change_due_to_extra_forward_max=float(np.abs(d['pd_qacc_warmstart_after_refresh']-d['pd_qacc_warmstart_before_refresh']).max()),
                estimator_evaluated=meta[name]['estimator_evaluated'])
            inputs[str(path)]=sha(path)
    assert len(runs)==10
    a=runs['cpu_pd_closed'];b=runs['cpu_end_closed'];g=runs['gpu_closed']
    identical={field:bool(np.array_equal(a[field],b[field])) for field in
        ('states_qpos','states_qvel','policy_action_raw','policy_obs','policy_estimated_velocity_scaled','pd_actuator_force_Nm')}
    assert all(identical.values())
    cpu_self={name:bool(np.array_equal(runs[name]['states_qpos'],runs[name.replace('_closed','_self_actions')]['states_qpos'])) for name in ('cpu_pd_closed','cpu_end_closed')}
    assert all(cpu_self.values())
    assert np.array_equal(g['policy_action_raw'],runs['gpu_self_actions']['policy_action_raw'])
    differences={field:[first_difference(g[field],a[field],eps,.01 if field.startswith('policy') else .002)
        for eps in (1e-4,.01,.1,.5,1.)] for field in ('policy_obs','policy_action_raw','policy_estimated_velocity_scaled','pd_actuator_force_Nm')}
    groups=dict(gyro=slice(0,3),gravity=slice(3,6),command=slice(6,9),leg_position=slice(9,13),
        motor_velocity=slice(13,19),previous_action=slice(19,25))
    group_diff={name:first_difference(g['policy_obs'][:,slot],a['policy_obs'][:,slot],1e-4,.01) for name,slot in groups.items()}
    assert group_diff['command']['maximum_abs_difference']==0.
    timeline={};limits={}
    for name in ('gpu_closed','cpu_pd_closed'):
        d=runs[name];timeline[name]=[]
        for k in (0,4,10,15,20,25,26,28,30,31):
            p=k*5
            timeline[name].append(dict(time_s=k*.01,estimated_vx_m_s=float(d['policy_estimated_velocity_scaled'][k,0]/2),
                actual_vx_m_s=float(d['policy_true_velocity_body'][k,0]),action_raw=d['policy_action_raw'][k].tolist(),
                position_difference_velocity_body=None if k==0 else averaged_body_velocity(d,k).tolist(),
                pd_target=d['pd_pd_target'][p].tolist(),actual_torque_Nm=d['pd_actuator_force_Nm'][p].tolist(),
                wheel_qvel_rad_s=d['pd_qvel_pre'][p,v[[2,5]]].tolist(),wheel_nominal_torque_Nm=d['pd_nominal_pd_torque'][p,[2,5]].tolist()))
        before_contact=slice(100,157)  # .200 <= t < .314, same interval for both engines.
        velocity=d['pd_qvel_pre'][before_contact][:,v[[2,5]]]
        nominal=d['pd_nominal_pd_torque'][before_contact][:,[2,5]]
        actual=d['pd_actuator_force_Nm'][before_contact][:,[2,5]]
        along=np.maximum(np.sign(nominal)*velocity,0.)
        zero=(np.abs(actual)<.01)&(np.abs(nominal)>1.)
        limits[name]=dict(window_s=[.2,.314],wheel_response=wheel_response.tolist(),
            zero_actual_despite_nominal_over1Nm_fraction=zero.mean(axis=0).tolist(),
            at_or_over_drive_cutoff_fraction=(along>=wheel_response[:,1]).mean(axis=0).tolist(),
            mean_abs_actual_torque_Nm=np.abs(actual).mean(axis=0).tolist(),
            leg_nominal_torque_at40Nm_fraction=(np.abs(d['pd_nominal_pd_torque'][:157][:,[0,1,3,4]])>=39.99).mean(axis=0).tolist(),
            wheel_target_at150_fraction=(np.abs(d['pd_pd_target'][:157][:,[2,5]])>=150.).mean(axis=0).tolist())
    paired_initial=dict(qpos_equal=bool(np.array_equal(g['states_qpos'][0],a['states_qpos'][0])),
        qvel_equal=bool(np.array_equal(g['states_qvel'][0],a['states_qvel'][0])),
        obs_max_abs_difference=float(np.abs(g['policy_obs'][0]-a['policy_obs'][0]).max()),
        action_max_abs_difference=float(np.abs(g['policy_action_raw'][0]-a['policy_action_raw'][0]).max()))
    assert paired_initial['qpos_equal'] and paired_initial['qvel_equal']
    r=g['policy_estimated_velocity_scaled']/2.;truth=g['policy_true_velocity_body']
    source=Path(budget['source_recording'])/'traces/diagnostic_2.npz'
    with np.load(source) as original:
        original_peak=float(np.abs(original['roll_pitch_deg'][:,1]).max())
        original_pose_diff=float(np.abs(original['qpos'][:101]-g['states_qpos'][::5]).max())
    result=dict(scope='Same7000-v2 diagnostic_2 initial state,1s probes; no production correction or behavior acceptance',
        summaries=summaries,paired_initial=paired_initial,refresh_timing_state_and_action_identical=identical,
        cpu_self_replays_bit_identical=cpu_self,first_differences=differences,observation_group_differences=group_diff,
        timeline=timeline,motor_limits=limits,
        initial_estimate_m_s=r[0].tolist(),initial_actual_velocity_m_s=truth[0].tolist(),
        gpu_original_peak_deg=original_peak,gpu_new_peak_deg=abs(summaries['gpu_closed']['pitch_peak_deg']),
        gpu_repeat_peak_deg=abs(summaries['gpu_repeat']['pitch_peak_deg']),gpu_original_vs_new_pose_max_difference=original_pose_diff,
        gpu_self_replay_action_identical=True,gpu_self_replay_reproduces_closed_state=bool(np.array_equal(g['states_qpos'],runs['gpu_self_actions']['states_qpos'])),
        conclusions=['CPU forward placement does not cause the extra pitch in this case; pose/action/actuator force match bit-for-bit',
            'Only contact-force reporting changes with forward placement',
            'Startup estimate has wrong forward-speed sign on both engines; later CPU estimate underestimates backward speed',
            'CPU commands reach wheel-speed target limits and actual wheel effort intermittently vanishes near the calibrated drive cutoff',
            'Long fixed-action replay is unstable even on the same GPU; cannot use cross-engine replay peaks as a clean causal decomposition'],
        limitations=['No intervention selectively corrected estimator outputs or the drive curve; causal contribution sizes are not established',
            'Fixed policy actions keep local PD/spring feedback active; this is not identical applied-torque playback',
            'GPU closed repeats have small numerical differences and new instrumented baseline differs from old12.24deg recording',
            'One initial state for1s; no full15-case regression or new training'],
        inputs_sha256=inputs,additional_physics_steps=0,additional_model_forwards=0,optimizer_updates=0)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,11),constrained_layout=True)
    colors={'gpu_closed':'#167e57','cpu_pd_closed':'#c94333','cpu_end_closed':'#343d48'}
    for name in ('gpu_closed','cpu_pd_closed','cpu_end_closed'):
        d=runs[name];t=np.arange(501)*.002
        axes[0,0].plot(t,d['states_row'][:,13],label=name,color=colors[name],ls='--' if name=='cpu_end_closed' else '-')
    for name in ('gpu_closed','cpu_pd_closed'):
        d=runs[name];t=np.arange(100)*.01;tp=np.arange(500)*.002;c=colors[name]
        axes[0,1].plot(t,d['policy_estimated_velocity_scaled'][:,0]/2,color=c,label=name+' estimated')
        axes[0,1].plot(t,d['policy_true_velocity_body'][:,0],color=c,ls='--',label=name+' true')
        axes[1,0].plot(tp,d['pd_pd_target'][:,2],color=c,label=name)
        axes[1,1].plot(tp,d['pd_pd_target'][:,1],color=c,label=name)
        axes[2,0].plot(tp,d['pd_actuator_force_Nm'][:,2],color=c,label=name,alpha=.85,lw=.9)
    for name,label,color,style in [('gpu_closed','GPU closed','#167e57','-'),('gpu_self_actions','GPU own actions replay','#7c4197','-'),
            ('cpu_pd_closed','CPU closed','#c94333','-'),('cpu_pd_self_actions','CPU own actions replay','#343d48','--')]:
        axes[2,1].plot(np.arange(501)*.002,runs[name]['states_row'][:,13],label=label,color=color,ls=style)
    titles=[('Changing CPU refresh timing does not change pitch','Pitch (deg)'),
        ('Velocity estimate differs from actual motion','Body vx (m/s)'),('Left wheel target reaches -150rad/s','Target speed (rad/s)'),
        ('Leg motor lf00 target also diverges','Target angle (rad)'),('Actual left wheel torque near drive cutoff','Applied torque (Nm)'),
        ('Own-action replay: CPU exact, GPU open-loop unstable','Pitch (deg)')]
    for axis,(title,ylabel) in zip(axes.flat,titles):
        axis.set(title=title,ylabel=ylabel,xlabel='Time (s)');axis.grid(alpha=.2);axis.legend(fontsize=7)
    for axis in [axes[0,1],axes[1,0],axes[1,1],axes[2,0]]:axis.set_xlim(0.,.5)
    axes[1,0].axhline(-150.,color='#777777',ls=':',lw=1)
    fig.savefig(args.output/'timing-feedback.png',dpi=150);plt.close(fig)
    lines=['# 同初态刷新时序与动作/力矩对照','',
        '**主要结论：CPU每2ms与每10ms刷新，姿态、关节速度、策略输入/动作和实际电机力矩逐位相同，峰值均24.337°。刷新频率不是这例额外倾角的原因；此前优先怀疑该项，被本次对照否定。接触力记录的相位/峰值会改变，不应与运动状态改变混为一谈。**','',
        '按用户授权，同一7000轮v2模型/diagnostic_2初态，最初8组1s探针，因新GPU基线与旧录像略不同，又在同一截止内补2组必要重复性核对。初始qpos/qvel逐位相同，夹具所有数组核对，PD/物理/模型/奖励/生产源保持；只有独立诊断中的forward位置是对照因素。','',
        '| 运行 | 最大pitch | 峰值时刻 | 机身接触峰值(N) |', '|---|---:|---:|---:|']
    for name,row in summaries.items():lines.append(f"| {name} | {row['pitch_peak_deg']:.3f}° | {row['pitch_peak_s']:.3f}s | {row['body_force_peak_N']:.2f} |")
    lines.extend(['','## 新记录给出的放大链路','',
        f"初始真实vx={truth[0,0]:.6f}m/s，估速={r[0,0]:.6f}m/s，方向相反；两个引擎几乎相同。encoder训练目标为速度×2，报告已除2还原为m/s。首次历史来自重复首帧，无法包含真正的过去运动；但没有做只修正估速的干预，不能把错误全部归给历史填充。",'',
        '约.04s起，陀螺仪/电机速度输入和动作出现可见分叉；命令槽始终一致。到.30s，GPU估速−1.509、实际瞬时−1.378m/s；CPU估速−.955、实际瞬时−1.429m/s。训练的估速监督采用最近10ms位置差分，而非瞬时qvel；用同一口径重算，此刻CPU为−1.408m/s、GPU为−1.359m/s，CPU偏慢仍然明显，不能用采样定义差别解释。复位首帧缺少过去位置，原训练代码将该差分状态初始化为0，这与真实初速不同，初始估速误差应连同冷启动可观测性理解。CPU同时输出更大的轮子动作和不同的腿目标，随后姿态过冲。估速误差与动作变化形成了可观测链条，但二者可能互相放大，尚未独立干预确认各自贡献。','',
        'CPU轮速目标从.260s起顶到±150rad/s，GPU在机身触地前没有顶到此上限；.30s左轮目标CPU−150、GPU−124.17rad/s。实际轮速却接近电机响应模型的62.54/63.71rad/s驱动截止，名义PD仍在请求驱动力时，实际力矩会间歇归零。例如.28s CPU双轮名义力矩−3.9/+3.9Nm，实际两轮均0；.30s CPU左轮仍0，GPU左轮约−4.82Nm。该力矩曲线是原有G/V/p物理响应，本次未修改。不能把它未经对照就称作唯一根因，也不能直接提高物理边界。','',
        '四腿在起步触地前没有撞到±40Nm名义力矩限制；因此不能把本例简单解释为腿力矩限幅不足。轮子实际扭矩还包含XML固定增益，约7.2Nm峰值并不等于名义PD的3.9Nm限制被代码漏掉；两者均保存并核对了实际actuator_force与广义关节力。','',
        '## 对照有效性与限制','',
        'CPU自身动作重放与闭环轨迹逐位一致；两种刷新时序的所有姿态/动作也一致。GPU闭环新采集两次峰值10.822°/10.826°，与旧12.244°有差异，所以本次因果比较使用同时记录的新基线，不把旧新差别归因于时序。GPU相同初态、重放自己的相同动作仍会随微小数值差逐步偏离，1s达到44.8°；CPU重放GPU动作达到47.8°。这种固定动作试验撤掉了神经策略反馈，连同引擎自身也会发散，不能用这些峰值干净地分摊CPU/GPU或估速的因果贡献。固定raw action也仍保留本地PD和弹簧反馈，并非固定实际力矩。','',
        '记录包含：100Hz当前25/历史125/估速/真实速度/raw和裁剪后动作；500Hz pre-qpos/qvel、PD目标（腿rad/轮rad/s）、名义PD力矩、实际ctrl/actuator_force/qfrc_actuator、采样电机速度、qacc_warmstart及独立当前姿态。固定动作分支没有执行估速，相关字段为占位值且metadata标estimator_evaluated=false，不当作零估速。','',
        '下一步优先做同一保存状态、同实际力矩输入的单个2ms响应对照，重点核.20～.31s转速触边附近，避免长开环重放的发散混淆；之后再用仅在诊断中校正估速的分支验证是否能减少触边和过冲。正式模型/控制/物理不据本次单例自动修改，也未追加训练。','',
        '[六项时间曲线](timing-feedback.png)、[完整指标JSON](report.json)、[原始记录目录](../runs/)、[预算与实际计数](../completion.json)。'])
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    for path,digest in inputs.items():assert sha(Path(path))==digest
    print(json.dumps(dict(refresh_timing_identical=identical,initial_velocity_estimate=r[0].tolist(),initial_velocity_true=truth[0].tolist(),
        motor_limits=limits,additional_physics_steps=0,additional_model_forwards=0),indent=2))


if __name__=='__main__':main()
