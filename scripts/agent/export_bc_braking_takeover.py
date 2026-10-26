#!/usr/bin/env python3
"""Export only qualified relocked teacher segments and audit BC loader compatibility."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from ct_mjlab.bc import ActionDataset,load_actor,LEGACY_SAMPLING,LABEL_REFERENCE
from check_lqr_student_closed_loop import digest


def export(args):
    torch.set_num_threads(1)
    report=json.loads((args.run/'result.json').read_text());assessment=json.loads((args.run/'assessment.json').read_text())
    assert assessment['artifact_checks_passed'] and report['optimizer_updates']==0 and not report['heldout_used']
    assert digest(args.run/'trace.npz')==report['trace_sha256'] and digest(args.checkpoint)==report['checkpoint_sha256']
    actor,saved=load_actor(args.checkpoint);profile=saved['dataset'].get('action_sampling',LEGACY_SAMPLING)
    base=ActionDataset([args.dataset],args.corrections,action_sampling=profile)
    assert base.description['fingerprint']==saved['dataset']['fingerprint']
    source_rows=[row for row in report['rows'] if row['mode']=='relock'];ids=[row['env'] for row in source_rows]
    qualified=np.array([row['qualified'] for row in source_rows]);args.output.mkdir(parents=True,exist_ok=False)
    with np.load(args.run/'trace.npz') as d:
        keys=('qpos','qvel','features','motor_dq','observation','alive','done','teacher_phase','teacher_fault','teacher_selected')
        arrays={key:d[key][:,ids].copy() for key in keys}
        arrays.update(action=d['student_action'][:,ids].copy(),label=d['teacher_action'][:,ids].copy(),
            applied_action=d['applied_action'][:,ids].copy(),source_q=d['source_q'][:,ids].copy(),source_dq=d['source_dq'][:,ids].copy())
        time=d['time'];eligible=(time>report['takeover_s'])[:,None]&arrays['alive']&arrays['teacher_selected']&(arrays['teacher_phase']==3)&~d['teacher_clipped'][:,ids]
        arrays['label_valid']=eligible&qualified[None,:]
        assert not arrays['label_valid'][time<=10.].any() and not arrays['label_valid'][:,~qualified].any()
        np.testing.assert_array_equal(arrays['applied_action'][arrays['label_valid']],arrays['label'][arrays['label_valid']])
        np.testing.assert_allclose(arrays['observation'][1:,:,22:28],np.clip(arrays['applied_action'][:-1,:,[0,1,3,4,2,5]],-100,100),atol=2e-6,rtol=2e-6)
        np.savez_compressed(args.output/'trace.npz',**arrays,time=time,observation_time=d['observation_time'],action_time=d['action_time'],
            initial_qpos=d['initial_qpos'][ids],initial_qvel=d['initial_qvel'][ids],
            nominal_motor_positions=d['nominal_motor_positions'],reference_motor_positions=d['reference_motor_positions'])
    rows=[dict(row,env=i,source_env=row['env']) for i,row in enumerate(source_rows)]
    child=dict(completed=True,correction_schema='ct_bc_corrections_v1',split='train',optimizer_updates=0,
        rows=rows,num_envs=len(rows),qualified=int(qualified.sum()),valid_correction_frames=int(arrays['label_valid'].sum()),
        teacher_queries_only=False,independent_student_control=False,heldout_used=False,
        label_reference=LABEL_REFERENCE,
        observation_contract=base.description['observation_contract'],
        **({'wheel_odometry':base.description['wheel_odometry']} if base.observation_dim == 40 else {}),
        source_actor_scope='BC15000 learner0..10s then relocked live teacher10..30s; only qualified teacher segments labeled.',
        base_motion_fingerprint=base.motion.fingerprint,dataset_fingerprint=base.description['fingerprint'],
        checkpoint_sha256=report['checkpoint_sha256'],xml_sha256=report['xml_sha256'],takeover_s=report['takeover_s'],duration_s=report['duration_s'],
        quality_thresholds=json.loads((args.run/'plan.json').read_text())['quality_thresholds'],
        source_trace_sha256=report['trace_sha256'],source_assessment_sha256=digest(args.run/'assessment.json'),
        trace_sha256=digest(args.output/'trace.npz'),exporter_sha256=digest(__file__))
    (args.output/'result.json').write_text(json.dumps(child,indent=2)+'\n')
    # 历史清单缺少参考版本时，由专用审计器统一导出，不能在这里混用新旧口径。
    if 'label_reference' not in base.description:
        child.pop('label_reference')
        (args.output/'result.json').write_text(json.dumps(child,indent=2)+'\n')
    mixed=ActionDataset([args.dataset],[*args.corrections,args.output],action_sampling=profile)
    for a,b in zip(base.frames['validation'],mixed.frames['validation']):torch.testing.assert_close(a,b,atol=0,rtol=0)
    for a,b in zip(base.sample('validation',2048,torch.Generator().manual_seed(714)),mixed.sample('validation',2048,torch.Generator().manual_seed(714))):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    added=len(mixed.frames['train'][0])-len(base.frames['train'][0]);assert added==child['valid_correction_frames']
    result=dict(passed=True,optimizer_updates=0,qualified=child['qualified'],accepted_pose_groups=len({r['group_id'] for r in rows if r['qualified']}),
        valid_frames=added,heldout_tensors_and_fixed_sampling_unchanged=True,actual_teacher_labels_checked=True,
        prefix_and_rejected_labels_excluded=True,base_fingerprint=base.description['fingerprint'],augmented_fingerprint=mixed.description['fingerprint'],
        class_frame_counts=mixed.description['class_frame_counts'],export_trace_sha256=child['trace_sha256'],source_trace_sha256=report['trace_sha256'],
        exporter_sha256=digest(__file__),scope='Training-only action corrections, no new BC/SMP/PPO optimization; whole trajectories not exported as SMP positive episodes.')
    (args.output/'check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--corrections',type=Path,action='append',default=[])
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);export(p.parse_args())
