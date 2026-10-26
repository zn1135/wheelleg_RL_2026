#!/usr/bin/env python3
"""Audit historical teacher ownership, then derive encoder travel without relabeling actions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from ct_mjlab.bc import ActionDataset, BRAKING_SAMPLING, LABEL_REFERENCE, WHEEL_PAIR_SCHEMA, check_label_reference
from ct_mjlab.wheel_odometry import WHEEL_OBSERVATION_CONTRACT, WHEEL_ODOMETRY, wheel_displacement

CONTROLLER = 'mjlab_training/src/ct_mjlab/teacher/controller.py'
CONTROLLER_SHA = 'dd3a83fafd789ee53f3a6053d9cfa80c688859661ef7a5ffb0b6d622067bd74b'
SHARED_CONTROLLER_SHA = '495e7e0b8fe8bf1505d771df0996b95bf408aa8064960b9afe7745eef85302b8'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def audit_sources(directory, report):
    for name, sha in report['source_sha256'].items():
        assert digest(directory/'snapshot'/name) == sha, name
    actual=report['source_sha256'][CONTROLLER]
    assert actual in (CONTROLLER_SHA,SHARED_CONTROLLER_SHA)
    return dict(manifest_sha256=digest(directory/('dataset.json' if 'episodes' in report else 'result.json')),
                controller_sha256=actual, snapshot_files=len(report['source_sha256']))


def add_travel(arrays, source_q):
    old = arrays['observation']
    assert old.shape[-1] == 39
    angles = source_q[..., [2,5]]
    # 第一拍就是保存的回合初态；取动作生成前编码器，不借用动作后的状态。
    travel = wheel_displacement(angles, angles[0])
    arrays['observation'] = np.concatenate((old, travel), -1)
    np.testing.assert_array_equal(arrays['observation'][..., :39], old)
    assert not travel[0].any()


def prepare(args):
    torch.set_num_threads(1)
    original = ActionDataset([args.dataset])
    manifest = read(args.dataset)
    assert manifest['initial'] == 'ground' and manifest['ground_start_hold_s'] == .2
    assert manifest['source_controller'] == 'live_lqr_vmc_through_student_pd'
    base_audit = audit_sources(args.dataset.parent, manifest)
    assert read(args.dataset.parent/'report.json')['complete']
    args.output.mkdir(parents=True, exist_ok=False)
    common = dict(label_reference=LABEL_REFERENCE, observation_contract=WHEEL_OBSERVATION_CONTRACT,
                  wheel_odometry=WHEEL_ODOMETRY)
    derived = dict(tool_sha256=digest(__file__), source_manifest_sha256=digest(args.dataset),
                   scope='Only append pre-step wheel travel; original teacher actions and first39 channels unchanged')
    episodes = []
    handovers = 0
    for record in manifest['episodes']:
        with np.load(args.dataset.parent/record['path']) as d:
            arrays = {k:d[k] for k in d.files}
        assert (arrays['phase'][-1] == 3) and not arrays['fault'].any() and arrays['alive'].all()
        # 最后一拍允许20秒合法超时；提前结束仍拒收。
        assert not arrays['done'][:-1].any() and abs(float(arrays['time'][-1])-20.) < 1e-8
        assert not (arrays['phase'] == 4).any()
        transitions = (arrays['phase'][1:] == 3) & (arrays['phase'][:-1] != 3)
        assert transitions.sum() == 1
        handovers += int(transitions.sum())
        add_travel(arrays, arrays['source_motor_q'])
        # 仍是同一段物理轨迹，保留排序和分组；新文件哈希使数据指纹改变。
        identity = record['episode_id']
        new = dict(record, episode_id=identity, path=f'accepted/{identity}.npz')
        metadata = dict(json.loads(str(arrays['metadata'])), **common,
                        action_pair_schema=WHEEL_PAIR_SCHEMA, derived_observation=derived)
        metadata.update({k:v for k,v in new.items() if k != 'sha256'})
        arrays['metadata'] = np.asarray(json.dumps(metadata))
        target = args.output/'paired'/new['path'];target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        new['sha256'] = digest(target);episodes.append(new)
    new_manifest = dict(manifest, **common, episodes=episodes, action_pair_schema=WHEEL_PAIR_SCHEMA,
                        derived_observation=derived)
    write(args.output/'paired/dataset.json', new_manifest)
    augmented_base = ActionDataset([args.output/'paired/dataset.json'], require_reference=True)
    train_groups = {e['group_id'] for e in original.motion.splits['train']}
    audits = []
    sources = (args.braking, args.takeover/'corrections')
    for source, name in zip(sources, ('braking69', 'braking121')):
        report = read(source/'result.json')
        assert report['completed'] and report['optimizer_updates'] == 0 and not report['heldout_used']
        assert report['teacher_queries_only'] is False
        assert report['base_motion_fingerprint'] == original.motion.fingerprint
        assert report['trace_sha256'] == digest(source/'trace.npz')
        if name == 'braking69':
            evidence = audit_sources(source, report)
            text = (source/'snapshot/scripts/agent/collect_bc_braking.py').read_text()
            helper = (source/'snapshot/scripts/agent/check_lqr_student_closed_loop.py').read_text()
            assert 'self.synchronize_balance(ready)' in text and 'switch & probe' in text
            assert 'self.teacher.controller.pos[mask] = 0.' in helper
            assert 'self.teacher.controller.yaw_target[mask] = body_angles' in helper
            assert '(probe[\'hold\'][mask] >= 500)' in helper
            assert read(source/'check.json')['passed']
            evidence['anchor_evidence'] = 'Frozen executed source and measured entry support; no recorded numeric anchor array in this historical run'
        else:
            parent = read(args.takeover/'result.json')
            evidence = audit_sources(args.takeover, parent)
            assert read(args.takeover/'assessment.json')['artifact_checks_passed']
            assert digest(args.takeover/'trace.npz') == parent['trace_sha256'] == report['source_trace_sha256']
            assert digest(args.takeover/'assessment.json') == report['source_assessment_sha256']
            assert digest(args.takeover/'switch.npz') == parent['switch_sha256']
            ids = [row['source_env'] for row in report['rows']]
            assert all(row['mode'] == 'relock' for row in report['rows'])
            with np.load(args.takeover/'trace.npz') as d, np.load(args.takeover/'switch.npz') as switch:
                after = d['entry_reference_after'][ids]
                assert not after[:,0].any()
                w,x,y,z = switch['qpos'][ids,3:7].T
                yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
                np.testing.assert_allclose(after[:,1], yaw, atol=3e-7, rtol=2e-6)
            evidence['anchor_evidence'] = 'Recorded zero position integral and current yaw at measured takeover'
        with np.load(source/'trace.npz') as d:
            arrays = {k:d[k] for k in d.files}
        valid = arrays['label_valid'] & arrays['alive']
        accepted = np.array([row['qualified'] for row in report['rows']])
        assert accepted.sum() == report['qualified']
        assert not valid[:,~accepted].any() and not valid[arrays['time']<=report['takeover_s']].any()
        assert arrays['teacher_selected'][valid].all() and (arrays['teacher_phase'][valid] == 3).all()
        assert not arrays['teacher_fault'][valid].any()
        np.testing.assert_array_equal(arrays['label'][valid], arrays['applied_action'][valid])
        for row in report['rows']:
            assert row['group_id'] in train_groups
            if row['qualified']:
                assert row['entry_hold_substeps'] >= 500 and all(row['checks'].values())
        add_travel(arrays, arrays['source_q'])
        target = args.output/name;target.mkdir()
        np.savez_compressed(target/'trace.npz', **arrays)
        updated = dict(report, **common, base_motion_fingerprint=augmented_base.motion.fingerprint,
            trace_sha256=digest(target/'trace.npz'), derived_observation=dict(derived,
                source_manifest_sha256=digest(source/'result.json'), source_trace_sha256=report['trace_sha256']))
        write(target/'result.json', updated)
        audits.append(dict(source=str(source), output=name, evidence=evidence,
            accepted_episodes=int(accepted.sum()), frames=int(valid.sum()), source_trace_sha256=report['trace_sha256']))
    mixed = ActionDataset([args.output/'paired/dataset.json'], [args.output/'braking69', args.output/'braking121'],
                          action_sampling=BRAKING_SAMPLING, require_reference=True)
    # 验证组不变；新增编码器通道逐帧可重算，原39维和标签必须逐位相同。
    for old, new in zip(original.frames['validation'], mixed.frames['validation']):
        torch.testing.assert_close(old, new[:, :old.shape[1]], atol=0, rtol=0)
    for split in ('train', 'validation'):
        assert {e['group_id'] for e in original.motion.splits[split]} == {e['group_id'] for e in mixed.motion.splits[split]}
    for old, new in zip(original.sample('validation',2048,torch.Generator().manual_seed(714)),
                        mixed.sample('validation',2048,torch.Generator().manual_seed(714))):
        torch.testing.assert_close(old, new[:, :old.shape[1]], atol=0, rtol=0)
    try:
        check_label_reference([new_manifest], [read(args.excluded/'result.json')], required=True)
    except ValueError:
        excluded_rejected = True
    else:
        raise AssertionError('Historical shadow queries must not enter unified data')
    video_manifest = read(args.takeover/'videos.json')
    for movie in video_manifest['movies']:
        source = args.takeover/movie['path'];assert digest(source) == movie['sha256']
        shutil.copy2(source, args.output/movie['path'])
    write(args.output/'videos.json', dict(video_manifest, provenance='Existing audited source trajectories, copied unchanged; no new behavior run'))
    result = dict(passed=True, optimizer_updates=0, label_reference=LABEL_REFERENCE,
        wheel_odometry=WHEEL_ODOMETRY, dataset=mixed.description, normal_handover_episodes=handovers,
        base_source=base_audit, correction_sources=audits, original_validation39_and_targets_bitwise_unchanged=True,
        original_fixed_validation_sample714_preserved=True, original_group_split_preserved=True,
        excluded=dict(path=str(args.excluded), trace_sha256=digest(args.excluded/'trace.npz'),
            rejected_by_loader_contract=excluded_rejected, reason='Shadow history queries; no actual teacher ownership'),
        scope='Historical source audit and exact encoder-derived observations; no rewritten actions, new rollout or new model; wheel travel is not LQR position integral or world displacement',
        tool_sha256=digest(__file__))
    write(args.output/'check.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('dataset', 'base_source','correction_sources')}, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--braking',type=Path,required=True)
    p.add_argument('--takeover',type=Path,required=True);p.add_argument('--excluded',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);prepare(p.parse_args())
