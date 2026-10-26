#!/usr/bin/env python3
"""Audit live braking branches, 500Hz PD/force/FIFO and hidden-reference wheel requests."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from ct_mjlab.bc import load_actor,ORDER
from ct_mjlab.teacher.mechanics import H7_WHEEL_LIMIT
from check_lqr_student_closed_loop import digest
from check_bc_braking_takeover import MODES,WHEEL_FIELDS


@torch.inference_mode()
def assess(args):
    r=json.loads((args.run/'result.json').read_text());plan=json.loads((args.run/'plan.json').read_text())
    assert r['completed'] and r['diagnostic_only'] and r['optimizer_updates']==0 and not r['training_labels_exported'] and not r['heldout_used']
    assert r['trace_sha256']==digest(args.run/'trace.npz') and r['wheel_trace_sha256']==digest(args.run/'wheel-500hz.npz')
    assert r['switch_sha256']==digest(args.run/'switch.npz') and r['checkpoint_sha256']==digest(args.checkpoint)
    for name,h in r['source_sha256'].items():assert digest(args.run/'snapshot'/name)==h
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    actor,saved=load_actor(args.checkpoint,'cuda:0');n=r['base_cases'];entry=round(r['takeover_s']/.01)
    train={e['group_id'] for e in saved['dataset']['splits']['train']}
    validation={e['group_id'] for e in saved['dataset']['splits']['validation']}
    assert set(plan['group_ids'])==train and not train&validation
    assert all(row['group_id'] in train for row in r['rows']) and len(validation)==10
    with np.load(args.run/'trace.npz') as archive:d={k:archive[k] for k in archive.files}
    with np.load(args.run/'wheel-500hz.npz') as archive:w={k:archive[k] for k in archive.files}
    with np.load(args.run/'switch.npz') as archive:s={k:archive[k] for k in archive.files}
    for key in ('initial_qpos','initial_qvel'):
        for mode in range(1,len(MODES)):np.testing.assert_array_equal(d[key][:n],d[key][mode*n:(mode+1)*n])
    prefix={}
    for key in ('qpos','qvel','observation','student_action'):
        f=d[key][:entry].reshape(entry,len(MODES),n,-1)
        prefix[key]=float(np.max(np.abs(f[:,1:]-f[:,:1])))
    switch_difference={key:float(np.max(np.abs(value.reshape(len(MODES),n,-1)[1:]-value.reshape(len(MODES),n,-1)[:1])))
        for key,value in s.items() if key in ('qpos','qvel','observation')}
    assert not d['teacher_selected'][:entry].any() and not d['teacher_selected'][:,:n].any()
    np.testing.assert_array_equal(d['applied_action'],np.where(d['teacher_selected'][:,:,None],d['teacher_action'],d['student_action']))
    np.testing.assert_allclose(d['observation'][1:,:,22:28],np.clip(d['applied_action'][:-1,:,ORDER],-100,100),atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(d['observation_time'],d['time']-.01,atol=1e-8,rtol=0)
    np.testing.assert_array_equal(d['observation_time'],d['action_time'])
    assert not d['observation'][:,:,:3].any() and not d['actual_torque'][:20].any()
    actor_differences=[]
    for tick in np.linspace(0,len(d['time'])-1,16).round().astype(int):
        p=actor.mlp(torch.from_numpy(d['observation'][tick].copy()).cuda()).cpu().numpy()
        np.testing.assert_allclose(p,d['student_action'][tick],atol=2e-6,rtol=2e-6)
        actor_differences.append(float(np.abs(p-d['student_action'][tick]).max()))
    assert list(w['field_names'])==list(WHEEL_FIELDS)
    values=w['values'];steps=np.arange(len(values))[:,None]
    delay=np.array([round(row['wheel_delay_s']/.002) for row in r['rows']])[None,:]
    generated=(steps-delay)//5
    target=d['applied_action'][np.maximum(generated,0),np.arange(len(r['rows']))[None,:]][:,:,[2,5]]*10
    target[generated<0]=0.;target=np.clip(target,-150,150)
    target_error=float(np.abs(target-values[:,:,1]).max());assert target_error<2e-5
    nominal=np.clip(float(w['nominal_stiffness'])*(target-values[:,:,0])-float(w['nominal_damping'])*values[:,:,0],-3.9,3.9)
    nominal[:100]=0.
    pd_error=float(np.abs(nominal-values[:,:,2]).max());assert pd_error<2e-6
    params=w['response_parameters'];fraction=np.clip(1.-np.maximum(np.sign(nominal)*values[:,:,0],0.)/params[:,1],0.,1.)
    force=nominal*np.power(fraction,params[:,2])*params[:,0]
    force_error=float(np.abs(force-values[:,:,3]).max());assert force_error<2e-5
    assert not values[:100,:,2:4].any()
    np.testing.assert_array_equal(values[4::5,:,3],d['actual_torque'][:,:,[2,5]])
    summaries={};late=d['time']>25.;post=d['time']>10.
    for mode in MODES:
        rows=[row for row in r['rows'] if row['mode']==mode]
        for row in rows:
            i=row['env'];f=d['features'][:,i];q=d['qpos'][:,i]
            metrics=dict(last5s_leg_velocity_rms_rad_s=float(np.sqrt(np.square(d['motor_dq'][late,i][:,[0,1,3,4]]).mean())),
                last5s_mean_speed_m_s=float(np.linalg.norm(f[late,4:6],axis=-1).mean()),
                last5s_mean_abs_yaw_rate_rad_s=float(np.abs(f[late,9]).mean()),
                last5s_xy_excursion_m=float(np.linalg.norm(q[late,:2]-q[late,:2][0],axis=-1).max()))
            for key,x in metrics.items():assert abs(x-row[key])<1e-10
            assert row['qualified']==all(row['checks'].values())
            if row['entry_hold_substeps']:
                assert row['entry_hold_substeps']==s['balance_hold'][i]>=500
                assert row['old_teacher_phase']==s['teacher_phase'][i]
                assert d['teacher_phase'][entry,i]==3
                if mode=='relock':assert d['entry_reference_after'][i,0]==0.
                else:np.testing.assert_array_equal(d['entry_reference_before'][i],d['entry_reference_after'][i])
            if row['qualified']:
                assert (d['balance_hold'][-1,i]>=500 and not d['done'][:,i].any() and
                    metrics['last5s_mean_speed_m_s']<=.01 and metrics['last5s_mean_abs_yaw_rate_rad_s']<=.02 and
                    metrics['last5s_leg_velocity_rms_rad_s']<=.02 and metrics['last5s_xy_excursion_m']<=.02)
        summary=dict(r['summary'][mode])
        summary['late5s']={key:dict(min=min(row[key] for row in rows),median=float(np.median([row[key] for row in rows])),max=max(row[key] for row in rows)) for key in metrics}
        ids=np.array([row['env'] for row in rows]);offset=MODES.index(mode)*n
        full=np.array([row['checks']['full_30s'] for row in rows])
        mask=late[:,None]&d['alive'][:,ids]&full[None,:]&(d['teacher_phase'][:,ids]==3)&~d['teacher_air'][:,ids]
        if mask.any():
            pre=np.concatenate((np.zeros_like(d['features'][:1,ids]),d['features'][:-1,ids]))
            summary['teacher_speed_estimate_minus_actual_mean_m_s']=float((d['teacher_x'][:,ids,1]-pre[:,:,4])[mask].mean())
            summary['teacher_position_integral_mean_m']=float(d['teacher_x'][:,ids,0][mask].mean())
        summaries[mode]=summary
    # 同一个学生实测状态上，仅去掉位置/朝向误差项；不恢复物理或改估计器历史。
    cols=d['wheel_contributions'][:,:n];unsaturated=d['wheel_unsaturated'][:,:n]
    convert=lambda u:np.clip(np.clip(u,-H7_WHEEL_LIMIT,H7_WHEEL_LIMIT)*np.array([1.,-1.],dtype=np.float32),-3.9,3.9)
    valid=(d['teacher_phase'][:,:n]==3)&~d['teacher_air'][:,:n]&d['alive'][:,:n]
    original=convert(unsaturated);alt=convert(unsaturated-cols[:,:,:,0]-cols[:,:,:,2])
    np.testing.assert_allclose(original[valid],d['teacher_requested'][:,:n][:,:,[2,5]][valid],atol=2e-6,rtol=2e-6)
    at=valid[entry];delta=alt[entry]-original[entry]
    hidden=dict(same_state_frames_at_switch=int(at.sum()),
        position_integral_at_switch_range_m=[float(d['teacher_x'][entry,:n,0][at].min()),float(d['teacher_x'][entry,:n,0][at].max())],
        position_only_unsaturated_wheel_change_mean_nm=np.abs(cols[entry,:n,:,0][at]).mean(0).tolist(),
        yaw_only_unsaturated_wheel_change_mean_nm=np.abs(cols[entry,:n,:,2][at]).mean(0).tolist(),
        relocked_wheel_request_change_mean_abs_nm=np.abs(delta[at]).mean(0).tolist(),
        relocked_pd_velocity_target_change_mean_abs_rad_s=(np.abs(delta[at])/float(w['nominal_stiffness'])).mean(0).tolist(),
        exact_same_measurement_counterfactual=True,
        scope='Analytic wheel LQR contribution intervention before motor response, with identical measured state/filter history; not proof that a39D zero-speed controller cannot exist.')
    baseline_mask=late[:,None]&valid
    query=dict(valid_late_frames=int(baseline_mask.sum()),
        teacher_vs_student_wheel_target_mean_abs_rad_s=(np.abs(d['teacher_action'][:,:n][:,:,[2,5]]-d['student_action'][:,:n][:,:,[2,5]])[baseline_mask]*10).mean(0).tolist(),
        teacher_keep_reference_nominal_wheel_request_mean_nm=d['teacher_requested'][:,:n][:,:,[2,5]][baseline_mask].mean(0).tolist(),
        teacher_relocked_same_state_nominal_wheel_request_mean_nm=alt[baseline_mask].mean(0).tolist(),
        student_nominal_wheel_request_mean_nm=d['student_requested'][:,:n][:,:,[2,5]][baseline_mask].mean(0).tolist())
    transient={}
    for mode_index,mode in enumerate(MODES):
        select=(w['time']>10.)&(w['time']<=10.5);a=values[select,mode_index*n:(mode_index+1)*n]
        transient[mode]=dict(first_half_second_actual_wheel_power_mean_w=float((a[:,:,3]*a[:,:,0]).sum(-1).mean()),
            first_half_second_actual_wheel_torque_mean_nm=a[:,:,3].mean((0,1)).tolist(),
            note='Mechanical wheel power uses actual torque and pre-step speed; actual stopping judged by full supported trajectories.')
    result=dict(artifact_checks_passed=True,optimizer_updates=0,diagnostic_only=True,training_labels_exported=False,
        training_groups=len(train),heldout_groups_untouched=len(validation),summary=summaries,
        identical_initial_states=True,prefix_max_differences=prefix,switch_max_differences=switch_difference,
        prefix_bitwise_equal=all(x==0 for x in prefix.values()),numeric_snapshot_restoration=False,
        max_actor_action_difference=max(actor_differences),max_fifo_target_difference_rad_s=target_error,
        max_nominal_pd_difference_nm=pd_error,max_actual_motor_force_difference_nm=force_error,
        wheel_physics_steps=len(values),hidden_reference=hidden,student_state_queries=query,first_half_second=transient,
        checkpoint_sha256=r['checkpoint_sha256'],trace_sha256=r['trace_sha256'],wheel_trace_sha256=r['wheel_trace_sha256'],
        assessor_sha256=digest(__file__))
    (args.run/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(args.run/'same-state-counterfactual.npz',teacher_keep_request=original,teacher_relocked_request=alt,valid=valid,time=d['time'])
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);assess(p.parse_args())
