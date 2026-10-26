#!/usr/bin/env python3
"""Assess exact-switch parking branches, teacher validity and training coverage; zero updates."""
import argparse,json
from pathlib import Path
import numpy as np
from ct_mjlab.smp.dataset import MotionDataset
from check_lqr_student_closed_loop import digest

MODES=('student-restored','warm','relock','balance')


def coverage(f):
    leg=np.sqrt(np.square(f[:,[18,19,21,22]]).mean(-1))
    stable=(abs(f[:,0]-.33648)<=.03)&(f[:,3]<-.97)&(leg<=.05)
    moving=np.linalg.norm(f[:,4:6],axis=-1)>.3;turning=abs(f[:,9])>.3
    return dict(frames=len(f),stable=int(stable.sum()),stable_moving=int((stable&moving).sum()),
        stable_moving_turning=int((stable&moving&turning).sum()))


def assess(args):
    out=args.run;reports={m:json.loads((out/m/'result.json').read_text()) for m in MODES}
    traces={m:np.load(out/m/'trace.npz') for m in MODES};b=traces['student-restored']
    try:
        signature=np.load(out/'student-paired/switch-check.npz');results={}
        for mode,r in reports.items():
            assert r['completed'] and r['optimizer_updates']==0 and r['diagnostic_only'] and not r['heldout_labels_for_training']
            assert digest(out/mode/'trace.npz')==r['trace_sha256']
            assert r['switch_state_sha256']==reports['student-restored']['switch_state_sha256']
            assert r['checkpoint_sha256']==reports['student-restored']['checkpoint_sha256']
            d=traces[mode]
            with np.load(out/mode/'switch-check.npz') as check:
                for key in signature.files:np.testing.assert_array_equal(check[key],signature[key])
            np.testing.assert_array_equal(d['observation'][1000],b['observation'][1000])
            np.testing.assert_allclose(d['student_action'][1000],b['student_action'][1000],atol=2e-6,rtol=2e-6)
            rows=r['rows'];valid_before=signature['teacher_phase']==3
            result={k:r[k] for k in ('num_envs','full_30s','final_balance','parking','smooth_parking')}
            # 趴地不动也可能满足运动学阈值；停车结论必须同时保持双轮平衡。
            result['kinematic_stop']=r.get('kinematic_stop',result['parking'])
            result['parking']=sum(row['parking'] and row['final_balance'] for row in rows)
            result['original_teacher_done_count']=int(valid_before.sum())
            result['parking_original_teacher_done']=sum(row['parking'] and row['final_balance'] for row,v in zip(rows,valid_before) if v)
            result['parking_original_teacher_not_done']=sum(row['parking'] and row['final_balance'] for row,v in zip(rows,valid_before) if not v)
            for key in ('last5s_mean_speed_m_s','last5s_mean_abs_yaw_rate_rad_s','last5s_leg_velocity_rms_rad_s','last5s_xy_excursion_m'):
                values=[row[key] for row in rows if row[key] is not None]
                result[key]=dict(min=min(values),median=float(np.median(values)),max=max(values)) if values else None
            result['teacher_phase_at_first_control_after_switch']={str(int(k)):int(v) for k,v in zip(*np.unique(d['teacher_phase'][1000],return_counts=True))}
            supported=np.asarray([row['final_balance'] for row in rows])
            late=(d['time'][1:]>25)[:,None]&d['alive'][1:]&supported[None]&(d['teacher_phase'][1:]==3)
            if late.any():
                measured=d['features'][:-1,:,4][late];estimated=d['teacher_x'][1:,:,1][late]
                result['late_pre_step_forward_velocity']=dict(measured_mean_m_s=float(measured.mean()),
                    teacher_estimate_mean_m_s=float(estimated.mean()),mean_abs_difference_m_s=float(abs(measured-estimated).mean()))
            results[mode]=result
        np.testing.assert_allclose(traces['warm']['teacher_action'][1000],b['teacher_action'][1000],atol=2e-6,rtol=2e-6)
        original_done=signature['teacher_phase']==3
        hidden=dict(identical_actor39_at_switch=True,
            changed_only_pos_yaw_references=True,
            wheel_target_difference_rad_s=(abs(traces['relock']['teacher_action'][1000,original_done][:,[2,5]]-
                traces['warm']['teacher_action'][1000,original_done][:,[2,5]]).mean(0)*10).tolist(),
            scope='Controlled hidden-reference intervention, not proof that a39D zero-speed policy cannot exist.')
        valid=(b['time']>25)[:,None]&(b['teacher_phase']==3)&b['alive']
        query=dict(valid_late_frames=int(valid.sum()),
            wheel_target_difference_rad_s=(abs((b['teacher_action']-b['student_action'])[:,:,[2,5]][valid]).mean(0)*10).tolist(),
            teacher_requested_wheel_torque_nm=b['teacher_requested'][:,:,[2,5]][valid].mean(0).tolist(),
            student_requested_wheel_torque_nm=b['student_requested'][:,:,[2,5]][valid].mean(0).tolist(),
            teacher_position_error_mean=float((b['teacher_target_state'][:,:,0]-b['teacher_x'][:,:,0])[valid].mean()),
            teacher_speed_error_mean=float((b['teacher_target_state'][:,:,1]-b['teacher_x'][:,:,1])[valid].mean()),
            nominal_teacher_wheel_request_power_w=float((b['teacher_requested'][1:][:,:,[2,5]]*b['motor_dq'][:-1][:,:,[2,5]]).sum(-1)[valid[1:]].mean()),
            note='Nominal same-state requests before motor-response gain; torque sign alone does not prove braking. Actual takeover outcomes are reported separately.')
        dataset=MotionDataset([args.dataset]);features=[]
        for e in dataset.splits['train']:
            with np.load(e['path']) as d:features.append(d['features'].copy())
        cov=dict(definition='Height .33648+/- .03m, gravity_b.z<-.97, RMS leg velocity<=.05rad/s; moving>.3m/s, turning>.3rad/s',
            base_training=coverage(np.concatenate(features)))
        with np.load(args.corrections/'trace.npz') as d:cov['old_valid_corrections']=coverage(d['features'][d['label_valid']&d['alive']])
        cov['current_heldout_late5s_diagnostic_only']=coverage(b['features'][b['time']>25].reshape(-1,30))
        (out/'coverage.json').write_text(json.dumps(cov,indent=2)+'\n')
        numerical=json.loads((out/'restore-check.json').read_text())
        result=dict(artifact_checks_passed=True,optimizer_updates=0,identical_switch_physics_observation_fifo=True,
            strict_numeric_restore_passed=numerical['passed'],strict_numeric_restore_differences=numerical['differences'],
            restore_only_control_parking=results['student-restored']['parking'],
            independent_branch_switch_s=10.,common_prefix='Use canonical student-paired0..10s; other branch0..10s was discarded warmup before exact restore.',
            results=results,hidden_reference=hidden,student_state_teacher_queries=query,coverage=cov,
            source_sha256=digest(__file__),diagnostic_only=True,training_labels_exported=False)
        (out/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
        plot(out,traces)
        print(json.dumps(result,indent=2))
    finally:
        for d in traces.values():d.close()


def plot(out,traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    names={'student-restored':'恢复后学生继续控制','warm':'原历史教师接管','relock':'教师重新锁定停车参考','balance':'教师进入平衡并锁定参考'}
    b=np.load(out/'student-paired/trace.npz');env=0
    for mode,d in traces.items():
        f=d['features'][:,env].copy();alive=d['alive'][:,env].copy()
        f[:1000]=b['features'][:1000,env];alive[:1000]=b['alive'][:1000,env]
        speed=np.linalg.norm(f[:,4:6],axis=-1);yaw=f[:,9]
        speed[~alive]=np.nan;yaw[~alive]=np.nan
        axes[0].plot(d['time'],speed,label=names[mode]);axes[1].plot(d['time'],yaw,label=names[mode])
    axes[0].set_ylabel('水平速度 m/s',fontproperties=font);axes[1].set_ylabel('偏航角速度 rad/s',fontproperties=font)
    axes[1].set_xlabel('时间 s',fontproperties=font)
    for ax in axes:
        ax.axvline(10,color='black',linestyle='--',label='10秒接管');ax.axhline(0,color='gray',linewidth=.7);ax.grid(alpha=.2)
        ax.legend(prop=font,fontsize=8)
    fig.tight_layout();fig.savefig(out/'parking-traces.png',dpi=160);plt.close(fig);b.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--corrections',type=Path,required=True)
    assess(p.parse_args())
