#!/usr/bin/env python3
"""Audit fresh wheel-angle actor39 teacher pairs and assemble fixed pose splits."""
import argparse,json,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ct_mjlab.bc import ActionDataset,LEGACY_SAMPLING,ALIGNED_LABEL_REFERENCE,build_actor,check_handover_wheel_pairs
from ct_mjlab.scripts.train_bc import train
from ct_mjlab import wheel_angles
from prepare_bc_wheel_dataset import audit_sources,digest,read,write


def prepare(args):
    torch.set_num_threads(1)
    original=ActionDataset([args.original])
    expected={s:{e['group_id'] for e in original.motion.splits[s]} for s in ('train','validation')}
    poses={}
    for e in original.motion.episodes:
        with np.load(e['path']) as d:poses[e['group_id']]=(d['initial_qpos'].copy(),d['initial_qvel'].copy())
    args.output.mkdir(parents=True,exist_ok=False)
    episodes=[];audits=[];manifest=None
    for source in args.dataset:
        m=read(source);profile=wheel_angles.PROFILES[m['observation_contract']]
        assert m['wheel_angles']==profile['wheel_angles'] and m['action_pair_schema']==profile['pair_schema']
        if manifest is not None:assert m['observation_contract']==manifest['observation_contract']
        assert read(source.parent/'report.json')['complete']
        evidence=audit_sources(source.parent,m);audits.append(dict(source=str(source),**evidence))
        assert all(set(m['group_split'][s])==expected[s] for s in expected)
        manifest=m if manifest is None else manifest
        for row in m['episodes']:
            path=source.parent/row['path'];assert digest(path)==row['sha256']
            with np.load(path) as d:
                assert d['alive'].all() and not d['done'][:-1].any() and len(d['time'])==2000
                assert d['teacher_selected'].all() and not d['fault'].any() and not d['target_clipped'].any()
                assert not d['actuator_torque'][:20].any() and d['observation'].shape==(2000,39)
                assert not any(key.startswith('reference_') for key in d.files)
                np.testing.assert_array_equal(d['action'],d['label'])
                np.testing.assert_array_equal(d['action'],d['applied_action'])
                np.testing.assert_allclose(d['initial_qpos'][2:],poses[row['group_id']][0][2:],atol=0,rtol=0)
                np.testing.assert_array_equal(d['initial_qvel'],poses[row['group_id']][1])
                angles=(check_handover_wheel_pairs(d,d['source_motor_q'],d['observation']).clip(-100,100)
                    if m['observation_contract']==wheel_angles.HANDOVER_OBSERVATION_CONTRACT else
                    (d['source_motor_q'][:,[2,5]]-d['source_motor_q'][0,[2,5]]).clip(-100,100))
                np.testing.assert_allclose(d['observation'][:,14:16],angles,atol=2e-6,rtol=2e-6)
            target=args.output/'paired'/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
            os.link(path,target);episodes.append(row)
    assert len({e['episode_id'] for e in episodes})==len(episodes)
    combined=args.output/'paired/dataset.json'
    write(combined,dict(manifest,episodes=episodes,assembly_sources=audits))
    data=ActionDataset([combined],require_reference=True)
    actual={s:{e['group_id'] for e in data.motion.splits[s]} for s in expected}
    assert all(actual[s]<=expected[s] and actual[s] for s in expected)
    # 共享参考重采保留原始分组；不把未通过停车筛选的示范补入训练集。
    if manifest['label_reference']!=ALIGNED_LABEL_REFERENCE:assert actual['train']==expected['train']
    # 同为39维的旧接口也必须被拒绝，检查应在创建优化器／输出目录之前结束。
    try:
        train(SimpleNamespace(iterations=1,batch_size=2,save_every=1,seed=42,device='cpu',
            dataset=[combined],corrections=[],action_sampling=LEGACY_SAMPLING,init=args.old_checkpoint,
            expand_observations=False,output=args.output/'forbidden-old-init',lr=.001))
    except ValueError as error:
        assert 'observation contracts differ' in str(error);rejection=str(error)
    else:raise AssertionError('Old39 initialization must not silently enter new wheel-angle39 training')
    assert not (args.output/'forbidden-old-init').exists()
    actor=build_actor(observation_dim=39);before={k:v.clone() for k,v in actor.state_dict().items()}
    obs,target=data.sample('train',32,torch.Generator().manual_seed(42))
    loss=(actor.mlp(obs)-target).square().mean();loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
    for name,value in actor.state_dict().items():torch.testing.assert_close(value,before[name],atol=0,rtol=0)
    result=dict(passed=True,optimizer_updates=0,observation_contract=manifest['observation_contract'],
        dataset=data.description,teacher_episodes=len(episodes),all_original_training_pose_groups_preserved=actual['train']==expected['train'],
        qualified_pose_group_counts={s:len(v) for s,v in actual.items()},
        missing_qualified_training_groups=sorted(expected['train']-actual['train']),
        missing_qualified_validation_groups=sorted(expected['validation']-actual['validation']),
        evaluation_pose_manifest=str(args.original),evaluation_pose_manifest_sha256=digest(args.original),
        same_dimension_old39_initialization_rejected=True,old_initialization_error=rejection,
        original_pose_group_counts={s:len(v) for s,v in expected.items()},
        actor_dim=39,critic_dim=82,finite_loss_and_gradients=True,parameters_unchanged=True,
        actual_teacher_labels_only=True,startup_actual_force_zero=True,no_LQR_observation_sidecar_fields=True,
        source_audits=audits,checker_sha256=digest(__file__),
        scope='Fresh teacher pairing, wheel-angle fields, fixed split, source integrity and zero-update training precheck; no behavior claim')
    write(args.output/'check.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='dataset'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,action='append',required=True)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--old-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);prepare(p.parse_args())
