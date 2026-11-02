"""Offline timing/source audit of backward-start pitch overshoot; no policy or physics."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def first(mask):
    indices=np.flatnonzero(mask)
    return None if not len(indices) else int(indices[0])


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);root=args.experiment
    budget=json.loads((root/'budget.json').read_text());source=Path(budget['source_recording'])
    prior=json.loads((root/'metrics.json').read_text())
    for path,digest in prior['inputs_sha256'].items():assert sha(Path(path))==digest
    checkpoint=Path(budget['checkpoint']);assert sha(checkpoint)==budget['checkpoint_sha256']
    frozen=checkpoint.parent/'source/ct_mjlab/fudan_ppo_walk'
    contract=ast.parse((frozen/'contract.py').read_text())
    weights=next(n.value for n in contract.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WEIGHTS' for t in n.targets))
    weights={k.arg:ast.literal_eval(k.value) for k in weights.keywords}
    assert weights['orientation']==-100. and weights['collision']==-1.
    reward=lambda path:next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='reward')
    assert ast.dump(reward(frozen/'env.py'),include_attributes=False)==ast.dump(reward(Path('mjlab_training/src/ct_mjlab/fudan_ppo_walk/env.py')),include_attributes=False)
    paths={'gpu':source,'cpu2ms':root/'cpu2ms','cpu05ms':root/'cpu05ms'}
    traces={};inputs={};rows={};sid='diagnostic_2'
    for name,path in paths.items():
        file=path/'traces'/(sid+'.npz');inputs[str(file)]=sha(file)
        with np.load(file,allow_pickle=False) as data:traces[name]={k:data[k].copy() for k in data.files}
        d=traces[name];t=d['time'];pitch=d['roll_pitch_deg'][:,1];body=d['base_contact']
        assert len(t)==3501 and np.allclose(t,np.arange(3501)*.002,rtol=0,atol=1e-9)
        contact=first(body>10.);peak=int(np.abs(pitch).argmax());wheel=first((d['wheel_support']>1.).any(axis=1))
        recovered=None
        for i in range(peak,len(t)-100):
            if (np.abs(pitch[i:i+100])<=1.).all():recovered=i;break
        pre=slice(0,contact if contact is not None else len(t))
        rows[name]=dict(pitch_peak_deg=float(pitch[peak]),pitch_peak_tick=peak,pitch_peak_s=float(t[peak]),
            first_body_contact_tick=contact,first_body_contact_s=None if contact is None else float(t[contact]),
            pitch_immediately_before_body_contact_deg=None if contact is None else float(pitch[contact-1]),
            maximum_precontact_pitch_deg=float(np.abs(pitch[pre]).max()),
            body_force_peak_N=float(body.max()),first_wheel_support_tick=wheel,first_wheel_support_s=float(t[wheel]),
            recovery_below1deg_held200ms_s=None if recovered is None else float(t[recovered]),
            after1s_max_pitch_deg=float(np.abs(pitch[500:]).max()),
            minimum_height_m=float(d['position'][:,2].min()),minimum_height_s=float(t[d['position'][:,2].argmin()]),
            maximum_abs_pitch_rate_before_contact_rad_s=float(np.abs(d['angular_body'][pre,1]).max()))
    qpos=traces['gpu']['qpos'][0]
    assert all(np.array_equal(qpos,d['qpos'][0]) for d in traces.values())
    with np.load(source/'fixtures'/(sid+'_initial.npz'),allow_pickle=False) as initial:
        root_velocity=initial['qvel'][:6].tolist();initial_height=float(initial['qpos'][2])
        assert all(np.array_equal(d['position'][0],initial['qpos'][:3]) for d in traces.values())
    paired=[]
    for ts in (.01,.02,.03,.04,.05,.08,.1,.15,.2,.25,.3,.31,.312,.314,.318,.32,.35,.5,.75,1.,2.):
        k=round(ts/.002);pose=round(ts/.01)
        values={name:dict(pitch_deg=float(d['roll_pitch_deg'][k,1]),height_m=float(d['position'][k,2]),
            vx_m_s=float(d['linear_body'][k,0]),pitch_rate_rad_s=float(d['angular_body'][k,1]),
            body_contact_N=float(d['base_contact'][k]),total_wheel_support_N=float(d['wheel_support'][k].sum())) for name,d in traces.items()}
        paired.append(dict(time_s=ts,engines=values,
            cpu2ms_vs_gpu_qpos_max_abs=float(np.abs(traces['cpu2ms']['qpos'][pose]-traces['gpu']['qpos'][pose]).max())))
    pose_pitch={}
    for name,d in traces.items():
        q=d['qpos'][32,3:7].astype(np.float64)
        q/=np.linalg.norm(q)
        pose_pitch[name]=float(np.rad2deg(np.arcsin(np.clip(2.*(q[0]*q[2]-q[1]*q[3]),-1.,1.))))
        assert abs(pose_pitch[name]-d['roll_pitch_deg'][160,1])<1e-4
    runtime=Path(importlib.util.find_spec('mjlab').origin).parent
    manager_path=runtime/'envs/manager_based_rl_env.py'
    manager=ast.parse(manager_path.read_text());step=next(n for n in ast.walk(manager) if isinstance(n,ast.FunctionDef) and n.name=='step')
    decimation=next(n for n in step.body if isinstance(n,ast.For))
    def is_forward(n):
        return isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='forward'
    assert not any(is_forward(n) for n in ast.walk(decimation))
    assert any(is_forward(n) for statement in step.body[step.body.index(decimation)+1:] for n in ast.walk(statement))
    native_path=root/'source/native.py';native=ast.parse(native_path.read_text())
    native_step=next(n for n in ast.walk(native) if isinstance(n,ast.FunctionDef) and n.name=='step')
    pd_loop=next(n for n in native_step.body if isinstance(n,ast.For))
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='mj_forward' for n in ast.walk(pd_loop))
    other_cases=[]
    protocol=json.loads((source/'protocol.json').read_text())
    for sample in protocol['samples']:
        name=sample['id']
        with np.load(root/'cpu2ms/traces'/(name+'.npz'),allow_pickle=False) as data:
            pitch=data['roll_pitch_deg'][:,1];k=int(np.abs(pitch).argmax())
            other_cases.append(dict(sample_id=name,case=sample['case']['id'],pitch_peak_deg=float(pitch[k]),time_s=float(data['time'][k])))
    saturation_deg=float(np.rad2deg(np.arcsin(np.sqrt(.01))))
    penalties={str(angle):float(np.clip(-100.*np.sin(np.deg2rad(angle))**2*.01,-.01,.01)) for angle in (6.,12.243959426879883,24.33704103059529)}
    result=dict(scope='Offline analysis of existing7000-v2 fixed0.30m diagnostic_2; other15-case summary for context',
        checkpoint_sha256=budget['checkpoint_sha256'],initial=dict(height_m=initial_height,root_qvel=root_velocity,
            root_qvel_order=['world_vx','world_vy','world_vz','body_wx','body_wy','body_wz'],command_vx_m_s=-2.,command_height_m=.30),
        same_initial_qpos=True,engines=rows,timeline=paired,all_cpu2ms_cases=other_cases,
        independent_qpos_pitch_at032s_deg=pose_pitch,
        timing_audit=dict(native_forward_every_pd_s=.002,gpu_full_forward_every_policy_s=.01,
            gpu_substep_contact_sensor_is_preintegration_derived_data=True,
            native_substep_contact_is_poststep_forward_derived_data=True,
            first_wheel_force_tick_difference_is_not_proof_of_physical_contact_time_difference=True,
            additional_forward_effect_on_dynamics_not_tested=True,
            runtime_manager_source=str(manager_path),runtime_manager_sha256=sha(manager_path),recorded_native_source_sha256=sha(native_path)),
        source_facts=dict(command_step_without_ramp=True,evaluation_root_velocity_is_random_not_stationary=True,
            actor_has_no_true_linear_velocity_input=True,initial_history_has_no_distinct_past_frames_in_nominal_case=True,
            orientation_weight=-100.,orientation_per_step_clip=[-.01,.01],orientation_saturates_at_total_tilt_deg=saturation_deg,
            pitch_only_per_step_penalty=penalties,collision_raw_is_zero=True,
            assessment_v2_was_applied_after_training_not_used_as_training_reward=True),
        established=['Substantial pitch exists before body contact; impact is not the initial trigger',
            'GPU also overshoots, so the problem is not exclusive to native CPU',
            'Initial poses match and early poses nearly match; recorded trajectory divergence grows during wheel-ground contact',
            'Independent saved-qpos pitch confirms an actual posture difference, not only contact-force reporting phase',
            'Changing CPU integration2ms to0.5ms retains23deg overshoot; timestep alone is not an adequate explanation'],
        hypotheses=['Abrupt command from moving/rotating fresh reset plus cold history triggers startup controller overshoot',
            'Contact/constraint and closed-loop feedback differences amplify overshoot on native CPU',
            'Clipped orientation cost and zero body-collision reward give limited incentive to suppress peak amplitude'],
        unresolved=['No saved raw actions, estimator predictions or per-substep joint velocities/torques',
            'Cannot uniquely split action response, estimator error, contact/constraint solver or forward-call placement contributions',
            'GPU/CPU substep force samples have different refresh phases; a2ms force onset difference is not causal proof',
            'Other speed cases have different initial velocities; not a same-initial-state speed sweep',
            'Cold-history effects are plausible, but actual estimator error was not recorded'],
        physics_steps=0,model_forwards=0,optimizer_updates=0,mesh_rendering_steps=0,
        inputs_sha256=inputs,frozen_reward_source_sha256=sha(frozen/'env.py'))
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    colors={'gpu':'#188451','cpu2ms':'#c44736','cpu05ms':'#3567bb'}
    for name,d in traces.items():
        mask=d['time']<=1.;t=d['time'][mask]
        for axis,values in [(axes[0,0],d['roll_pitch_deg'][mask,1]),(axes[0,1],d['position'][mask,2]),
                (axes[1,0],d['linear_body'][mask,0]),(axes[1,1],d['base_contact'][mask])]:
            axis.plot(t,values,color=colors[name],label=name,lw=1.3)
    axes[0,0].set(title='Pitch overshoot starts before body contact',ylabel='Body pitch (deg)')
    axes[0,1].set(title='Body height undershoots during the same startup',ylabel='Base height (m)')
    axes[0,1].axhline(.30,color='#777777',ls='--',lw=1)
    axes[1,0].set(title='Immediate -2m/s request from a moving reset',ylabel='Measured body vx (m/s)')
    axes[1,0].axhline(-2.,color='#777777',ls='--',lw=1)
    axes[1,1].set(title='Brief body contact follows the pitch buildup',ylabel='Body contact-force norm (N)')
    for axis in axes.flat:
        axis.axvline(rows['cpu2ms']['first_body_contact_s'],color=colors['cpu2ms'],ls=':',lw=1)
        axis.set_xlabel('Time after first command (s)');axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.savefig(args.output/'startup-timeline.png',dpi=160);plt.close(fig)
    lines=['# 后退2m/s起步大倾角：已有轨迹分析','',
        '结论：这是起步过冲，随后恢复；GPU已存在12.24°过冲，原生CPU把它放大至24.34°。机身触地发生在大倾角已经形成之后，不能倒因为“先撞地才倾斜”。CPU0.5ms仍有23.00°，不能只归因于2ms积分。','',
        '本次仅已有NPZ/训练归档源码分析，0新物理、0策略前向、0更新、0网格渲染；所有输入哈希复核不变。','',
        '| 引擎 | 最大pitch | 峰值时间 | 首次机身力>10N | 触地前一拍pitch | 恢复≤1°持续200ms |',
        '|---|---:|---:|---:|---:|---:|']
    for name,row in rows.items():
        contact='未发生' if row['first_body_contact_s'] is None else f"{row['first_body_contact_s']:.3f}s"
        before='—' if row['pitch_immediately_before_body_contact_deg'] is None else f"{row['pitch_immediately_before_body_contact_deg']:.2f}°"
        lines.append(f"| {name} | {row['pitch_peak_deg']:.2f}° | {row['pitch_peak_s']:.3f}s | {contact} | {before} | {row['recovery_below1deg_held200ms_s']:.3f}s |")
    lines.extend(['','## 已确定的触发条件和先后顺序','',
        f'此例不是静止站稳后平滑后退：初始vx={root_velocity[0]:.3f}m/s、vy={root_velocity[1]:.3f}m/s、vz={root_velocity[2]:.3f}m/s，pitch角速度={root_velocity[4]:.3f}rad/s、yaw角速度={root_velocity[5]:.3f}rad/s；合法闭链根高{initial_height:.3f}m，t=0直接请求vx=-2m/s、高度.30m，没有速度斜坡或先站稳阶段。评估首次历史五帧来自同一时刻；名义例无噪声，因此旧4净1噪此时实际五帧一致。Actor不直接读真实线速度；初始估速是否错误未记录，不能声称已测到。','',
        'GPU/CPU初始qpos逐位相同；t=.03s最大qpos差仅约1.95e−6。首次记录轮支撑>1N分别CPU2ms .032s、GPU与CPU0.5ms .034s；但这2ms差异受下述记录相位影响，不能单独当作物理接触早晚的证据。到.20s，CPU/GPU pitch为−5.15°/−3.45°；.30s为−21.02°/−10.57°，两端vx却接近−1.4m/s。CPU在.310s已−23.20°且机身力0，.314s才出现753N接触，随后角速度反向并恢复。触地是大角度已形成之后发生的，也改变后续速度/姿态，不是最初的倾斜诱因。','',
        '独立从100Hz保存的qpos四元数重算.320s俯仰，GPU−11.59°、CPU2ms−24.23°，与500Hz记录一致。因此大角度差是真实保存姿态差，不只是接触力记录相位造成的假象。','',
        'CPU根高最小约.227m，目标.30m也发生明显下冲。GPU/CPU的腿部姿态和支撑控制需共同检查，不能用简化“加速就必须倾斜24°”解释机身角度。','',
        '## 具体实现差异：刷新与记录时序','',
        '当前本机mjlab的ManagerBasedRlEnv.step在5个2ms子步中执行step→scene.update→substep metrics，整个10ms结束后才调用一次完整sim.forward；源代码明确说明子步接触等派生量来自积分前的计算。原生Native.step则每个2ms之后做完整mj_forward，再取姿态/接触。forward重新计算动力学与约束而不推进时间，故两端子步接触读数的相位不完全相同；其对约束求解缓存和实际轨迹的影响尚未做动态A/B。不要把这处明确差异直接写成已证实的唯一根因，但应优先核对。策略周期结束时两端都会刷新，不能概括为GPU所有策略输入都慢2ms。','',
        '## 已确认的训练约束特性','',
        '归档7000训练源的姿态项为clip(-100×(gx²+gy²)×.01,-.01,.01)。总倾角约5.74°就达到单步-.01上限，纯pitch6°、12.24°和24.34°单步姿态罚都为-.01；持续更久仍会增加累计惩罚。该项不再随峰值角度增大而加重；collision原始项为0，没有额外机身接触惩罚。它们能解释策略为何可能接受短暂大角度代价，但不能单凭公式证明唯一原因。原训练倾覆终止条件也不会在24°立即触发。评分v2是在7000训练结束后修改的，不能反过来造成这次训练策略。','',
        '## 当前判断和建议','',
        '最符合现有证据的是“突变指令/运动初态下的策略起步过冲，加上CPU接触约束与闭环差异放大”；精确到“某个电机力矩方向错误”“估速误差多少”或“哪个solver参数错误”仍缺数据。另一后退−2.5m/s样本只有1.34°峰值，但初态不同，不能把它当严格速度因果对照，也不能概括为后退始终不稳。','',
        '下一步优先做同一diagnostic_2初态的0～1s短诊断，同时记录估速、六路raw action、PD目标、实际力矩、关节速度及支撑/约束数据；将forward调用与力采样相位作为显式对照因素，固定动作比较CPU/GPU首次轮地接触前后，分开求解差异和闭环反馈放大，再决定是否修改启动指令或策略奖励。成功判据：查清首次输入/动作/受力分叉的具体拍，并检验仅对齐时序能否降低约24°峰值且不丢原15例。本次未执行新动态诊断，也未调整forward时序。','',
        '[四条时间曲线](startup-timeline.png)、[原始事件和逐拍JSON](report.json)。'])
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    for path,digest in inputs.items():assert sha(Path(path))==digest
    print(json.dumps(dict(engines=rows,orientation_saturation_deg=saturation_deg,physics_steps=0,model_forwards=0,optimizer_updates=0),indent=2))


if __name__=='__main__':main()
