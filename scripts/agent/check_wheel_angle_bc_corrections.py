#!/usr/bin/env python3
"""Audit actual teacher recovery labels for wheel-angle BC; performs no update."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from ct_mjlab.bc import ActionDataset,load_actor,ORDER
from prepare_bc_wheel_dataset import audit_sources,digest,write


@torch.inference_mode()
def check(args):
    torch.set_num_threads(1)
    base=ActionDataset([args.dataset],require_reference=True)
    actor,saved=load_actor(args.checkpoint,'cuda:0')
    assert saved['dataset']['fingerprint']==base.description['fingerprint']
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    before={k:v.clone() for k,v in actor.state_dict().items()}
    groups=set(base.description['group_split']['train'])
    sources=[]
    for directory in args.corrections:
        report=json.loads((directory/'result.json').read_text())
        evidence=audit_sources(directory,report)
        assert report['completed'] and report['optimizer_updates']==0
        assert report['teacher_queries_only'] is False and not report['heldout_used']
        assert {r['group_id'] for r in report['rows']}==groups and len(report['rows'])==156
        assert report['base_motion_fingerprint']==base.motion.fingerprint
        assert report['checkpoint_sha256']==digest(args.checkpoint)
        assert report['trace_sha256']==digest(directory/'trace.npz')
        qualified=np.array([r['qualified'] for r in report['rows']])
        with np.load(directory/'trace.npz') as d:
            valid=d['label_valid']&d['alive']
            assert valid.sum()==report['valid_correction_frames']>0
            post=d['time']>report['takeover_s']
            allowed=np.broadcast_to(qualified,valid.shape)
            if report.get('label_scope')=='qualified_full_recoveries_plus_rise_only_v1':
                rise=np.array([r['rise_qualified'] for r in report['rows']])
                allowed=allowed|((d['time']<=2.)[:,None]&rise[None,:])
            assert not valid[~post].any() and not (valid&~allowed).any()
            assert d['teacher_selected'][post][d['alive'][post]].all() and not d['teacher_selected'][~post].any()
            assert d['teacher_selected'][valid].all() and not d['target_clipped'][valid].any()
            assert not d['teacher_fault'][valid].any() and not (d['teacher_phase'][valid]==4).any()
            assert d['alive'][:,qualified].all() and not d['done'][:-1,qualified].any()
            assert not d['actuator_torque'][:20].any()
            np.testing.assert_array_equal(d['applied_action'],np.where(d['teacher_selected'][...,None],d['label'],d['student_action']))
            np.testing.assert_array_equal(d['applied_action'][valid],d['label'][valid])
            np.testing.assert_allclose(d['observation_time'],d['action_time'],atol=1e-8,rtol=0)
            continuous=d['alive'][1:]&d['alive'][:-1]
            np.testing.assert_allclose(d['source_q'][1:][continuous],d['motor_q'][:-1][continuous],atol=2e-6,rtol=2e-6)
            np.testing.assert_allclose(d['observation'][1:,...,22:28][continuous],
                d['applied_action'][:-1,...,ORDER][continuous].clip(-100,100),atol=2e-6,rtol=2e-6)
            expected=(d['source_q'][...,[2,5]]-d['source_q'][0][...,[2,5]]).clip(-100,100)
            np.testing.assert_allclose(d['observation'][...,14:16][valid],expected[valid],atol=2e-6,rtol=2e-6)
            action_difference=[]
            for tick in np.linspace(0,len(d['time'])-1,16).round().astype(int):
                predicted=actor.mlp(torch.from_numpy(d['observation'][tick].copy()).cuda()).cpu().numpy()
                np.testing.assert_allclose(predicted,d['student_action'][tick],atol=2e-6,rtol=2e-6)
                action_difference.append(float(np.max(abs(predicted-d['student_action'][tick]))))
            first=int(np.flatnonzero(post)[0])
            error=(d['student_action'][first,qualified]-d['label'][first,qualified])
            sources.append(dict(path=str(directory),qualified=int(qualified.sum()),candidate_count=len(qualified),
                qualified_groups=len({r['group_id'] for r in report['rows'] if r['qualified']}),
                valid_frames=int(valid.sum()),rise_frames=int(valid[d['time']<=2.].sum()),
                rise_only_candidates=sum(r.get('rise_qualified',False) and not r['qualified'] for r in report['rows']),
                intervention_action_rmse=np.sqrt(np.square(error).mean(0)).tolist(),
                max_recomputed_student_action_difference=max(action_difference),
                trace_sha256=report['trace_sha256'],source_audit=evidence))
    mixed=ActionDataset([args.dataset],args.corrections,require_reference=True)
    for original,augmented in zip(base.frames['validation'],mixed.frames['validation']):
        torch.testing.assert_close(original,augmented,atol=0,rtol=0)
    for original,augmented in zip(base.sample('validation',2048,torch.Generator().manual_seed(714)),
            mixed.sample('validation',2048,torch.Generator().manual_seed(714))):
        torch.testing.assert_close(original,augmented,atol=0,rtol=0)
    assert mixed.motion.fingerprint==base.motion.fingerprint
    for name,value in before.items():torch.testing.assert_close(value,actor.state_dict()[name],atol=0,rtol=0)
    result=dict(passed=True,optimizer_updates=0,sources=sources,dataset=mixed.description,
        checkpoint_sha256=digest(args.checkpoint),heldout_tensors_and_fixed_sample_unchanged=True,
        training_pose_groups_only=True,actual_teacher_labels_only=True,zero_startup_force=True,
        added_training_frames=len(mixed.frames['train'][0])-len(base.frames['train'][0]),
        checker_sha256=digest(Path(__file__)),
        scope='Screened real teacher interventions after frozen student prefixes, aligned wheel-angle/previous-action inputs and exclusion of heldout poses; no policy or SMP training')
    write(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='dataset'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--corrections',type=Path,action='append',required=True);check(p.parse_args())
