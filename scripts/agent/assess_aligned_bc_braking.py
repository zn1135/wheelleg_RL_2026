#!/usr/bin/env python3
"""Recompute shared-reference braking evidence and accept only real train-pose labels."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from ct_mjlab.bc import (load_actor,ORDER,ActionDataset,PRESERVE_SAMPLING,
                        ALIGNED_LABEL_REFERENCE,check_handover_wheel_pairs,check_aligned_position_pairs)
from check_lqr_student_closed_loop import digest
from prepare_bc_wheel_dataset import audit_sources


@torch.inference_mode()
def assess(args):
    torch.set_num_threads(1)
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    actor,saved=load_actor(args.checkpoint,'cuda:0');plan=json.loads((args.run/'plan.json').read_text())
    before={k:v.clone() for k,v in actor.state_dict().items()}
    train=set(saved['dataset']['group_split']['train']);heldout=set(saved['dataset']['group_split']['validation'])
    assert len(train)==39 and len(heldout)==10 and not train&heldout
    for path,sha in plan['immutable_sha256'].items():assert digest(path)==sha,path
    results={};prefix={};probes={};summaries={};cases={};max_model_error=0.;max_angle_error=0.
    for branch in ('student','teacher'):
        directory=args.run/branch;r=json.loads((directory/'result.json').read_text());results[branch]=r
        assert r['completed'] and r['optimizer_updates']==0 and r['heldout_used'] is False and r['models_unchanged']
        assert r['checkpoint_sha256']==digest(args.checkpoint) and r['trace_sha256']==digest(directory/'trace.npz')
        assert r['label_reference']==ALIGNED_LABEL_REFERENCE and r['duration_s']==30. and r['takeover_s']==10.
        assert {x['group_id'] for x in r['rows']}==train and r['num_envs']==156
        audit_sources(directory,r)
        with np.load(directory/'trace.npz') as archive:d={k:archive[k] for k in archive.files}
        assert not d['actuator_torque'][:20].any()
        assert not d['observation'][:,:,:3].any()
        np.testing.assert_allclose(d['observation'][1:,:,22:28][d['alive'][1:]],
            np.clip(d['applied_action'][:-1,:,ORDER],-100,100)[d['alive'][1:]],atol=2e-6,rtol=2e-6)
        angle=check_handover_wheel_pairs(d,d['source_q'],d['observation'],d['alive'])
        max_angle_error=max(max_angle_error,float(abs(angle.clip(-100,100)[d['alive']]-d['observation'][...,14:16][d['alive']]).max()))
        check_aligned_position_pairs(d,d['alive'])
        for tick in np.linspace(0,len(d['time'])-1,16).round().astype(int):
            actual=actor.mlp(torch.from_numpy(d['observation'][tick].copy()).cuda()).cpu().numpy()
            np.testing.assert_allclose(actual,d['student_action'][tick],atol=2e-6,rtol=2e-6)
            max_model_error=max(max_model_error,float(abs(actual-d['student_action'][tick]).max()))
        if branch=='student':
            assert r['correction_schema']=='ct_bc_diagnostic_queries_v1' and not d['teacher_selected'].any() and not d['label_valid'].any()
            np.testing.assert_array_equal(d['applied_action'],d['student_action'])
            prefix={k:d[k][:1000].copy() for k in ('qpos','qvel','observation','student_action')}
            initial={k:d[k].copy() for k in ('initial_qpos','initial_qvel')}
        else:
            assert r['teacher_queries_only'] is False and r['correction_schema']=='ct_bc_corrections_v1'
            assert not d['teacher_selected'][:1000].any()
            np.testing.assert_array_equal(d['applied_action'],np.where(d['teacher_selected'][...,None],d['label'],d['student_action']))
            for key,value in initial.items():np.testing.assert_array_equal(value,d[key])
            differences={k:float(abs(value-d[k][:1000]).max()) for k,value in prefix.items()}
            allowed=np.asarray([x['qualified'] for x in r['rows']])[None,:]
            assert not (d['label_valid']&~allowed).any() and not d['label_valid'][:1000].any()
            assert d['teacher_selected'][d['label_valid']].all()
            np.testing.assert_array_equal(d['label'][d['label_valid']],d['applied_action'][d['label_valid']])
            assert not d['teacher_fault'][d['label_valid']].any()
        pre_root=np.concatenate((d['initial_qpos'][None],d['qpos'][:-1]))
        pre_velocity=np.concatenate((d['initial_qvel'][None,:,:3],d['qvel'][:-1,:,:3]))
        pre_features=np.concatenate((np.zeros_like(d['features'][:1]),d['features'][:-1]))
        rows=[];metrics=[];reference_rows=[]
        for row in r['rows']:
            i=row['env'];valid=d['alive'][:,i];late=(d['time']>25.)&valid
            f=d['features'][:,i];q=d['qpos'][:,i]
            values=dict(speed=float(np.linalg.norm(f[late,4:6],axis=-1).mean()) if late.any() else None,
                drift=float(np.linalg.norm(q[late,:2]-q[late,:2][0],axis=-1).max()) if late.any() else None,
                leg_rms=float(np.sqrt(np.square(d['motor_dq'][late,i][:,[0,1,3,4]]).mean())) if late.any() else None,
                yaw=float(abs(f[late,9]).mean()) if late.any() else None)
            for key,value in values.items():
                if valid.all():assert abs(value-row['quality'][key])<1e-10
            passed=bool(valid.all() and row['quality']['strict_rise'] and row['quality']['final_balance'] and row['quality']['finite'] and
                not row['quality']['body_contact'] and row['quality']['height']<=.03 and row['quality']['tilt']<=np.deg2rad(25) and
                all(values[k] is not None and values[k]<=limit for k,limit in (('speed',.01),('drift',.02),('leg_rms',.02),('yaw',.02))))
            if branch=='teacher':assert passed==row['qualified']
            rows.append(dict(env=i,group_id=row['group_id'],qualified=passed,**values))
            if late.any():metrics.append(values)
            # 清零时刻相同不代表三种位移数值相同；均由动作前真实状态独立计算。
            latch=d['wheel_angle_handover_latched'][:,i]
            event=np.flatnonzero(latch & ~np.concatenate(([False],latch[:-1])) & valid)
            if len(event):
                t=event[0];quat=pre_root[t,i,3:7];w,x,y,z=quat
                heading=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
                root_travel=((pre_root[:,i,:2]-pre_root[t,i,:2])*np.array([np.cos(heading),np.sin(heading)])).sum(-1)
                wheel_travel=.03*(angle[:,i,0]-angle[:,i,1]);position=d['lqr_position_error_used'][:,i]
                for seconds in (10.,20.,29.99):
                    k=round(seconds/.01)
                    w,x,y,z=pre_root[k,i,3:7]
                    yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
                    if valid[k]:reference_rows.append(dict(env=i,time_s=seconds,wheel_travel_m=float(wheel_travel[k]),
                        actual_forward_travel_m=float(root_travel[k]),lqr_position_m=float(position[k]),
                        lqr_minus_wheel_m=float(position[k]-wheel_travel[k]),
                        lqr_velocity_m_s=float(d['teacher_x'][k,i,1]),actual_velocity_b_m_s=float(pre_features[k,i,4]),
                        actual_velocity_heading_m_s=float(pre_velocity[k,i,0]*np.cos(yaw)+pre_velocity[k,i,1]*np.sin(yaw))))
        summaries[branch]=dict(num_envs=156,full_30s=sum(x['quality']['full'] for x in r['rows']),
            strict_rise=sum(x['quality']['strict_rise'] for x in r['rows']),final_balance=sum(x['quality']['final_balance'] for x in r['rows']),
            qualified=sum(x['qualified'] for x in rows),qualified_pose_groups=len({x['group_id'] for x in rows if x['qualified']}),
            valid_correction_frames=int(d['label_valid'].sum()),
            late5s={k:dict(mean=float(np.mean([x[k] for x in metrics])),min=min(x[k] for x in metrics),max=max(x[k] for x in metrics)) for k in ('speed','drift','leg_rms','yaw')},
            metrics_scope='Available last5s frames; only complete qualified episodes may enter training.')
        cases[branch]=rows
        (directory/'reference-comparison.json').write_text(json.dumps(reference_rows,indent=2)+'\n')
        with np.load(directory/'probes.npz') as p:probes[branch]={k:p[k] for k in p.files}
        assert digest(directory/'probes.npz')==r['probes_sha256'] and r['max_cloned_teacher_vs_live_action_difference']<=2e-6
        del d
    feedback=[];baseline=[];p=probes['student']
    for time in results['student']['anchor_times_s']:
        tick=round(time/.01);valid=p[f'{tick}.valid']
        entry=dict(time_s=time,valid_anchors=int(valid.sum()))
        for owner in ('teacher','student'):
            torque=p[f'{tick}.base.{owner}.torque'][valid]
            entry[owner+'_wheel_forward_request_mean_nm']=float(((torque[:,2]-torque[:,5])*.5).mean())
        baseline.append(entry)
    for variable in results['student']['perturbations']:
        name=variable['variable'];amplitude=variable['amplitude'];derivatives={owner:[] for owner in ('teacher','student')}
        for time in results['student']['anchor_times_s']:
            tick=round(time/.01);valid=p[f'{tick}.valid']
            for owner in derivatives:
                plus=p[f'{tick}.{name}.1.{owner}.torque'];minus=p[f'{tick}.{name}.-1.{owner}.torque']
                values=((plus-minus)/(2*amplitude))[valid]
                derivatives[owner].append((values[:,2]-values[:,5])*.5)
        derivatives={k:np.concatenate(v) for k,v in derivatives.items()};a=derivatives['teacher'];b=derivatives['student'];active=abs(a)>1e-4
        feedback.append(dict(variable=name,amplitude=amplitude,anchors=len(a),
            teacher_derivative_median=float(np.median(a)),student_derivative_median=float(np.median(b)),
            sign_agreement=float((a[active]*b[active]>0).mean()) if active.any() else None,
            signed_gain_ratio_median=float(np.median(b[active]/a[active])) if active.any() else None,
            scope='Encoder-angle and LQR-integral interventions deliberately hold other history fixed; they are different variables, not a physical trajectory equivalence.'))
    # 只把通过全轨迹筛选的实际教师段交给既有正式加载器；旁路查询禁止进入训练。
    dataset_check=None
    if summaries['teacher']['valid_correction_frames']:
        current=ActionDataset([args.dataset],args.previous_corrections,action_sampling=PRESERVE_SAMPLING,require_reference=True)
        assert current.description['fingerprint']==saved['dataset']['fingerprint']
        data=ActionDataset([args.dataset],[*args.previous_corrections,args.run/'teacher'],action_sampling=PRESERVE_SAMPLING,require_reference=True)
        for old,new in zip(current.frames['validation'],data.frames['validation']):torch.testing.assert_close(old,new,atol=0,rtol=0)
        for old,new in zip(current.sample('validation',2048,torch.Generator().manual_seed(714)),
                data.sample('validation',2048,torch.Generator().manual_seed(714))):torch.testing.assert_close(old,new,atol=0,rtol=0)
        dataset_check=dict(passed=True,dataset=data.description,optimizer_updates=0,
            previous_dataset_matches_checkpoint=True,validation_tensors_and_seed714_sample_unchanged=True,
            previous_corrections=[str(p) for p in args.previous_corrections])
    for key,value in before.items():torch.testing.assert_close(value,actor.state_dict()[key],atol=0,rtol=0)
    result=dict(artifact_checks_passed=True,optimizer_updates=0,smp_updates=0,ppo_updates=0,train_groups=39,heldout_groups_untouched=10,
        checkpoint_sha256=digest(args.checkpoint),checkpoint_total_updates=saved['total_actor_updates'],
        summary=summaries,cases=cases,feedback=feedback,baseline_requests_on_student_states=baseline,
        identical_initial_states=True,prefix_max_differences=differences,prefix_bitwise_equal=all(x==0 for x in differences.values()),
        max_actual_actor_action_difference=max_model_error,max_wheel_angle_error_rad=max_angle_error,
        cloned_teacher_vs_live_checked=True,shared_reference_recurrence_checked=True,models_and_immutable_inputs_unchanged=True,
        corrected_dataset=dataset_check,assessor_sha256=digest(__file__),
        scope='Nominal train-only39x4 x30s. Local nominal requests and actual closed loops are distinct. No optimizer, physical perturbation,10ms stress,heldout expansion,CPU/H7 or generalization claim.')
    (args.run/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('corrected_dataset','cases')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','checkpoint','dataset'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous-corrections',type=Path,action='append',default=[])
    assess(p.parse_args())
