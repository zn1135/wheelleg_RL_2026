#!/usr/bin/env python3
"""Audit recollected shared-reference pairs and assemble all original BC source classes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from ct_mjlab.bc import ActionDataset,BRAKING_SAMPLING,check_reference_pairs,load_actor,expand_actor,ORDER
from prepare_bc_wheel_dataset import audit_sources


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def continuity(data):
    alive=data['alive'];mask=alive[1:]&alive[:-1]
    np.testing.assert_array_equal(data['reference_state'][1:,...,0][mask],data['reference_state'][:-1,...,5][mask])
    check_reference_pairs(data,data['observation'],alive)
    assert not data['actuator_torque'][:20].any()
    np.testing.assert_allclose(data['observation_time'],data['action_time'],atol=1e-8,rtol=0)
    np.testing.assert_allclose(data['observation_time'],data['time']-.01,atol=1e-8,rtol=0)
    np.testing.assert_allclose(data['source_q'][1:][mask],data['motor_q'][:-1][mask],atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(data['observation'][1:,...,22:28][mask],
        data['applied_action'][:-1,...,ORDER][mask].clip(-100,100),atol=2e-6,rtol=2e-6)


@torch.inference_mode()
def prepare(args):
    torch.set_num_threads(1)
    assert len(args.corrections)==len(args.models)==3
    original=ActionDataset([args.original]);split={s:sorted({e['group_id'] for e in original.motion.splits[s]}) for s in ('train','validation')}
    args.output.mkdir(parents=True,exist_ok=False)
    episodes=[];audits=[];manifest=None
    for source in args.teacher:
        m=read(source);assert m['group_split']==split
        audits.append(dict(source=str(source),**audit_sources(source.parent,m)))
        manifest=m if manifest is None else manifest
        for row in m['episodes']:
            with np.load(source.parent/row['path']) as d:continuity(d)
            target=args.output/'paired'/'accepted'/Path(row['path']).name;target.parent.mkdir(parents=True,exist_ok=True)
            os.link(source.parent/row['path'],target)
            episodes.append(dict(row,path='accepted/'+target.name))
    write(args.output/'paired/dataset.json',dict(manifest,episodes=episodes,assembly_sources=audits))
    base=ActionDataset([args.output/'paired/dataset.json'],require_reference=True)
    assert all({e['group_id'] for e in base.motion.splits[s]}==set(split[s]) for s in split)
    corrections=[];counts=[]
    for source,model,total_updates in zip(args.corrections,args.models,(5000,10000,15000),strict=True):
        report=read(source/'result.json');evidence=audit_sources(source,report)
        assert report['completed'] and report['optimizer_updates']==0 and not report['heldout_used']
        assert report['source_base_motion_fingerprint']==original.motion.fingerprint
        assert report['trace_sha256']==digest(source/'trace.npz')
        assert len(report['rows'])==156 and {r['group_id'] for r in report['rows']}==set(split['train'])
        actor,saved=load_actor(model,'cuda:0')
        assert saved.get('total_actor_updates',saved['iteration'])==total_updates
        assert digest(model)==report['checkpoint_sha256']
        for s in split:assert {e['group_id'] for e in saved['dataset']['splits'][s]}==set(split[s])
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends()
        with np.load(source/'trace.npz') as d:
            continuity(d);valid=d['label_valid']&d['alive']
            assert valid.sum()==report['valid_correction_frames'] and not (valid&d['target_clipped']).any()
            np.testing.assert_array_equal(d['applied_action'],np.where(d['teacher_selected'][...,None],d['label'],d['student_action']))
            for tick in np.linspace(0,len(d['time'])-1,16).round().astype(int):
                predicted=actor.mlp(torch.from_numpy(d['observation'][tick,:,:39].copy()).cuda()).cpu().numpy()
                np.testing.assert_allclose(predicted,d['student_action'][tick],atol=2e-6,rtol=2e-6)
            if report['teacher_queries_only']:
                assert not d['teacher_selected'].any()
            else:
                qualified=np.array([r['qualified'] for r in report['rows']])
                assert not valid[:1000].any() and not valid[:,~qualified].any()
                assert d['teacher_selected'][valid].all() and not d['teacher_fault'][valid].any()
                assert (d['reference_events'][1000,qualified]&2).all()
            counts.append(dict(source=str(source),candidates=len(report['rows']),qualified=report['qualified'],
                valid_frames=int(valid.sum()),teacher_queries_only=report['teacher_queries_only'],audit=evidence))
        target=args.output/source.name;target.mkdir()
        os.link(source/'trace.npz',target/'trace.npz')
        write(target/'result.json',dict(report,base_motion_fingerprint=base.motion.fingerprint,
            checked_assembly=dict(original_report_sha256=digest(source/'result.json'),assembler_sha256=digest(__file__))))
        corrections.append(target)
    mixed=ActionDataset([args.output/'paired/dataset.json'],corrections,action_sampling=BRAKING_SAMPLING,require_reference=True)
    old,saved=load_actor(args.init,'cuda:0');expanded=expand_actor(old,41,'cuda:0')
    inputs,_=mixed.sample('validation',2048,torch.Generator().manual_seed(714));inputs=inputs.cuda()
    expected=old.mlp(inputs[:,:39].contiguous());actual=expanded.mlp(inputs)
    torch.testing.assert_close(expected,actual,atol=2e-5,rtol=2e-5)
    changed=inputs.clone();changed[:,39:]=torch.linspace(-100,100,len(changed),device='cuda')[:,None]
    torch.testing.assert_close(actual,expanded.mlp(changed),atol=0,rtol=0)
    for name,value in old.state_dict().items():
        new=expanded.state_dict()[name]
        if new.shape!=value.shape:
            torch.testing.assert_close(new[:,:39],value,atol=0,rtol=0);assert not new[:,39:].any()
        else:torch.testing.assert_close(new,value,atol=0,rtol=0)
    result=dict(passed=True,optimizer_updates=0,dataset=mixed.description,correction_sources=counts,
        original_group_split=split,base_episodes=len(episodes),all_original_pose_groups_preserved=True,
        old_learned_parameters_preserved=True,new_input_weights_zero=True,
        max_initialization_action_difference=float((expected-actual).abs().max()),
        checkpoint_sha256=digest(args.init),assembler_sha256=digest(__file__),
        scope='Recollected sources, aligned pre-step reference recurrence, actual/query ownership, no training or heldout correction labels')
    write(args.output/'check.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('dataset','original_group_split')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--original',type=Path,required=True)
    p.add_argument('--teacher',type=Path,action='append',required=True);p.add_argument('--corrections',type=Path,action='append',required=True)
    p.add_argument('--models',type=Path,action='append',required=True);p.add_argument('--init',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);prepare(p.parse_args())
