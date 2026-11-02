"""Offline v1/v2 comparison of all paired diagnostic cases, preserving old results."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .score_v2 import acceptance_manifest, summarize_recording


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transition(before, after):
    assert before.keys() == after.keys()
    return dict(retained_passed=[k for k in before if before[k] and after[k]],
        newly_passed=[k for k in before if not before[k] and after[k]],
        regressed=[k for k in before if before[k] and not after[k]],
        still_not_met=[k for k in before if not before[k] and not after[k]])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--height030', type=Path)
    parser.add_argument('--protected-source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protected = json.loads(args.protected_source_manifest.read_text())
    assert all(sha(Path(path)) == digest for path,digest in protected.items())
    inputs = {}
    reports = {}
    original_protocols = {}
    changes = {}
    records = [('iter6000',args.before),('iter7000',args.after)]
    if args.height030 is not None:records.append(('iter7000_height030',args.height030))
    for name,directory in records:
        protocol_path = directory/'protocol.json'
        original_path = directory/'diagnostic_summary.json'
        protocol = json.loads(protocol_path.read_text())
        assert protocol['mode'] == 'diagnostic_not_formal_acceptance'
        original_protocols[name] = protocol
        old = json.loads(original_path.read_text())
        old_rows = {r['sample_id']:r for r in old['samples']}
        assert set(old_rows) == {s['id'] for s in protocol['samples']} and old['total'] == len(old_rows)
        video_index_path = directory.parent/(directory.name+'-video')/'index.json'
        index = json.loads(video_index_path.read_text())
        assert not index['missing'] and index['mode'] == protocol['mode']
        budget_path = directory.parent/'replay-budget.json'
        budget = json.loads(budget_path.read_text())
        assert sha(Path(budget['checkpoint'])) == index['checkpoint_sha256'] == budget['checkpoint_sha256']
        input_paths = [protocol_path,original_path,video_index_path,budget_path,Path(budget['checkpoint'])]
        chapters = {c['sample_id']:c for v in index['videos'] for c in v['chapters']}
        assert set(chapters) == set(old_rows)
        for video in index['videos']:
            path = Path(video['path'])
            assert sha(path) == video['sha256']
            input_paths.append(path)
        for sid in old_rows:
            path = directory/'traces'/(sid+'.npz')
            assert sha(path) == chapters[sid]['trajectory_sha256'] and chapters[sid]['complete']
            input_paths.append(path)
            input_paths.append(directory/'fixtures'/(sid+'_initial.npz'))
        input_paths.extend([directory/'fixtures/model.mjb',directory/'fixtures/mapping.json'])
        inputs.update({str(p):sha(p) for p in input_paths})
        report = summarize_recording(directory)
        for row in report['samples']:
            previous = old_rows[row['sample_id']]
            # 逐字段复现旧成绩，不只核对原来的七个争议ID。
            for key,value in row['v1_diagnostic'].items():assert previous[key] == value, (name,row['sample_id'],key)
            assert row['segments'] == previous['segments']
            assert row.get('stop') == previous.get('stop')
        (args.output/(name+'-v1-preserved.json')).write_bytes(original_path.read_bytes())
        (args.output/(name+'-v2.json')).write_text(json.dumps(report,indent=2)+'\n')
        before = {r['sample_id']:r['passed'] for r in old['samples']}
        after = {r['sample_id']:r['passed'] for r in report['samples']}
        changes[name] = dict(v1_passed=sum(before.values()),v2_passed=sum(after.values()),
            total=len(after),complete=report['complete'],numerical_failures=report['numerical_failures'],
            v1_to_v2=transition(before,after),checkpoint_sha256=index['checkpoint_sha256'])
        reports[name] = report
    assert original_protocols['iter6000']['samples'] == original_protocols['iter7000']['samples']
    assert len(original_protocols['iter6000']['samples']) == 20
    fixture_arrays = 0
    for sample in original_protocols['iter6000']['samples']:
        sid = sample['id']
        with np.load(args.before/'fixtures'/(sid+'_initial.npz'),allow_pickle=False) as before, np.load(args.after/'fixtures'/(sid+'_initial.npz'),allow_pickle=False) as after:
            assert set(before.files) == set(after.files)
            for key in before.files:
                assert np.array_equal(before[key],after[key]), (sid,key)
                fixture_arrays += 1
    paired = {}
    for version in ('v1','v2'):
        values = []
        for name in ('iter6000','iter7000'):
            values.append({r['sample_id']:(r['v1_diagnostic']['passed'] if version=='v1' else r['passed']) for r in reports[name]['samples']})
        paired[version] = transition(*values)
    assert all(sha(Path(path)) == digest for path,digest in inputs.items()), 'Existing artifacts changed during offline scoring'
    assert all(sha(Path(path)) == digest for path,digest in protected.items()), 'Protected source changed'
    result = dict(protocol=acceptance_manifest(),scope='All20 paired nominal diagnostics; optional0.30m15 cases reported separately',
        runs=changes,paired6000_to7000=paired,initial_fixture_arrays_bit_identical=fixture_arrays,
        original_artifact_hashes_preserved=True,v1_scores_reproduced_all_fields=True,
        tracking_and_stop_metrics_unchanged=True,protected_source_unchanged=True,
        physics_steps=0,model_forwards=0,optimizer_updates=0,final_acceptance='pending_user_video_review',
        prior_user_video_review=dict(recording='iter7000',sample_ids=[r['sample_id'] for r in reports['iter7000']['samples'] if not r['v1_diagnostic']['passed']],
            decision='User previously reported no observed failure in these seven existing clips',
            scope='Does not cover unreviewed0.30m cases, full987 matrix, CPU or hardware'))
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'input-sha256.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (args.output/'protocol-v2.json').write_text(json.dumps(acceptance_manifest(),indent=2)+'\n')
    lines = ['# 复旦验收v2：既有轨迹离线对照','',
        '未进行仿真、策略推理或训练；原v1成绩逐字段复现，原模型/轨迹/视频哈希保留。6000和7000的全部20个相同ID配对，0.30m另列，不混分母。','',
        '| 模型／范围 | 完整 | 原v1技术达标 | v2技术达标 | v1→v2新增 | 回退 | 仍未达标 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,row in changes.items():
        change = row['v1_to_v2']
        lines.append(f"| {name} | {row['complete']}/{row['total']} | {row['v1_passed']}/{row['total']} | {row['v2_passed']}/{row['total']} | {len(change['newly_passed'])} | {len(change['regressed'])} | {len(change['still_not_met'])} |")
    lines.extend(['','下表按同一v2比较6000→7000，明确保留／新增／回退；不是只比较v1→v2的标准变化。','',
        f"- 保留：{paired['v2']['retained_passed']}",f"- 新增：{paired['v2']['newly_passed']}",
        f"- 回退：{paired['v2']['regressed']}",f"- 两者仍未达标：{paired['v2']['still_not_met']}",'',
        '| 范围 | ID／工况 | v1 | v2 | v2原因 | 起步>10N单次最长／累计(ms) | 机身峰值(N) |',
        '|---|---|---|---|---|---:|---:|'])
    for name,report in reports.items():
        for row in report['samples']:
            startup = row['body_contact']['startup_over10N']
            contact = row['body_contact']
            lines.append(f"| {name} | {row['sample_id']} / {row['case']['id']} | {'达标' if row['v1_diagnostic']['passed'] else '未达标'} | {'达标' if row['passed'] else '未达标'} | {', '.join(row['failure_reasons']) or '—'} | {startup['maximum_run_ticks']*2}/{startup['total_ticks']*2} | {contact['raw_peak_N']:.2f} |")
    lines.extend(['','所有原速度/yaw/高度/俯仰/停车指标逐字段不变。原瞬时轮载、旧fell合并标记保留在v1_diagnostic，不作为v2隐藏否决；原始峰值和持续区间仍保留。',
        '用户此前认可7000轮原七个争议片段“未观察到失败”，该意见保留；本次离线数字不能代替新增视频确认，更不能代表全987例、CPU或实机通过。',
        '建议先核对本表仍未达标项的具体持续时间，再决定是否优化行为；不能为了旧瞬时标记直接改奖励或加训。',
        '', '[逐例与配对JSON](comparison.json)、[协议JSON](protocol-v2.json)、[原输入哈希](input-sha256.json)。'])
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(runs=changes,paired=paired,fixture_arrays=fixture_arrays,physics_steps=0,model_forwards=0,optimizer_updates=0),indent=2))


if __name__ == '__main__':
    main()
