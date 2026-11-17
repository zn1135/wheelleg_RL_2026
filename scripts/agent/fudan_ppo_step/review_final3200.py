"""Verify final v7 evidence and compare unchanged stair/flat protocols offline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_or_verify(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, path
    else:
        with path.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)


def read_recording(folder, audit):
    protocol = json.loads((folder / 'protocol.json').read_text())
    assert json.loads(audit.read_text())['passed']
    assert protocol['all_finite'] and protocol['all_networks_unchanged']
    assert protocol['optimizer_updates'] == 0 and protocol['backend'] == 'mujoco_warp_gpu'
    assert sha(folder / 'source_diagnose_reverse.py') == protocol['script_sha256']
    for case in protocol['cases']:
        assert sha(folder / 'traces' / (case['id'] + '.npz')) == protocol['trace_sha256'][case['id']]
    return protocol, json.loads((folder / 'metrics.json').read_text())


def groups(rows):
    return [dict(height_m=h, total=len(chosen), landed=sum(r['landed'] for r in chosen),
                 full_parked=sum(r['task_score']['passed'] for r in chosen),
                 strict_parked=sum(r['parked'] for r in chosen))
            for h in sorted({r['height'] for r in rows if r['height'] > 0})
            for chosen in [[r for r in rows if r['height'] == h]]]


def pair(before, after, old_protocol, new_protocol):
    assert old_protocol['cases'] == new_protocol['cases']
    for key in ('recipe', 'seconds', 'command_height_m', 'high_speed', 'max_approach_speed',
                'diagnostic_action_delay_ms', 'diagnostic_sensor_delay_ms'):
        assert old_protocol[key] == new_protocol[key], key
    differences = []
    for case in old_protocol['cases']:
        with np.load(before / 'traces' / (case['id'] + '.npz')) as old, \
             np.load(after / 'traces' / (case['id'] + '.npz')) as new:
            delta = float(np.abs(old['initial'] - new['initial']).max())
        differences.append(dict(id=case['id'], max_abs=delta, passed=delta < 2e-6))
    return dict(initial_tolerance=2e-6, all_initials_matched=all(r['passed'] for r in differences),
                max_initial_difference=max(r['max_abs'] for r in differences), differences=differences)


def detail(folder, case, row):
    with np.load(folder / 'traces' / (case['id'] + '.npz')) as trace:
        n = int(trace['valid'].sum())
        force = trace['force'][:n].reshape(-1, 2)
        pitch = np.degrees(np.arcsin(np.clip(trace['gravity'][:n, 0], -1, 1)))
        roll = np.degrees(np.arctan2(-trace['gravity'][:n, 1], -trace['gravity'][:n, 2]))
        speed = np.linalg.norm(trace['linear_world'][:n, :2], axis=-1)
        x = trace['wheel_pos'][:n, :, 0] - trace['origin'][0] - 1.2
        stops = np.flatnonzero(abs(trace['command'][:n, :2]).max(-1) < 1e-6)
        tail = slice(max(0, n - 200), n)
        # 同时保留停车各条件占比，用来辨别回退、支撑、姿态或速度瓶颈。
        conditions = dict(clearance=(x.min(-1) > .15),
            target_surface=(abs(trace['support_height'][:n] - case['height']) < .01).all(-1),
            normal_support=trace['normal_support'][:n].astype(bool),
            no_nonwheel_contact=~(trace['force'][:n] > 1).any(axis=(1, 2)),
            speed=speed <= .05, pitch=abs(pitch) <= 5, roll=abs(roll) <= 5)
        return dict(id=case['id'], height_m=case['height'], command_speed=case['speed'],
            angle_deg=case['angle'], friction=case['friction'], landed=row['landed'],
            full_parked=row['task_score']['passed'], stop_reason=row['stop_reason'],
            stop_command_s=float(stops[0] * .01) if len(stops) else None,
            first_wheel_top_support_s=row['first_wheel_top_support_s'],
            stable_stop_s=row['task_score']['stable_stop_s'],
            approach_speed_peak=row['task_score']['max_approach_speed'],
            nonwheel_peak_N=force.max(0).tolist(), nonwheel_time_fraction=(force > 1).mean(0).tolist(),
            final_wheel_edge_x_m=x[-1].tolist(), final_wheel_z_m=trace['wheel_pos'][n-1, :, 2].tolist(),
            wheel_z_max_m=trace['wheel_pos'][:n, :, 2].max(0).tolist(),
            final_pitch_deg=float(pitch[-1]), final_roll_deg=float(roll[-1]),
            final_speed=float(speed[-1]), tail_s=min(n, 200) * .01,
            tail_condition_fractions={key: float(value[tail].mean()) for key, value in conditions.items()})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    args = parser.parse_args()
    root = args.recording.resolve()
    assert json.loads((root / 'execution.json').read_text())['passed']
    core_p, core = read_recording(root / 'core', root / 'audit-core/audit.json')
    nominal_p, nominal = read_recording(root / 'nominal', root / 'audit-nominal/audit.json')
    receipt = json.loads((root / 'input-receipt.json').read_text())
    for protocol in (core_p, nominal_p):
        assert protocol['source_sha256'] == receipt['checkpoint_sha256'] == sha(Path(protocol['source_checkpoint']))
    previous = ROOT / 'artifacts/fudan_ppo_step/20261117-v7-effect200-01/evidence'
    comparisons = {}
    for label, name in [('walk3000', 'baseline'), ('step200', 'candidate')]:
        folder = previous / name
        old_p, old = read_recording(folder, previous / ('audit-' + name) / 'audit.json')
        comparisons[label] = dict(groups=groups(old), pairing=pair(folder, root / 'core', old_p, core_p))
    old1300 = ROOT / 'artifacts/fudan_ppo_step/20261117-v7-replay1300-01'
    old_p, old = read_recording(old1300 / 'replay', old1300 / 'audit/audit.json')
    comparisons['step1300'] = dict(groups=groups(old),
        pairing=pair(old1300 / 'replay', root / 'nominal', old_p, nominal_p))
    flat = json.loads((root / 'flat47/acceptance/acceptance.json').read_text())
    runtime = json.loads((root / 'flat47/runtime-audit.json').read_text())
    assert flat['total'] == 47 and runtime['passed'] and runtime['optimizer_updates'] == 0
    flat_plan = json.loads((root / 'flat47/plan.json').read_text())
    assert flat_plan['sha256'] == receipt['checkpoint_sha256']
    # 将已采集的高速子集交给统一入口评分，明确保留完整矩阵缺测标记。
    cases = [dict(id=c['id'], mode='up', height=c['height'], speed=c['speed'],
                  angle_deg=c['angle'], friction=c['friction'], reverse=False)
             for c in core_p['cases'] if c['height'] > 0]
    initial, mu = [], []
    for case in cases:
        with np.load(root / 'core/traces' / (case['id'] + '.npz')) as trace:
            q = trace['initial'].copy()
            q[:3] -= trace['origin']
            initial.append(q)
        mu.append(case['friction'])
    cohort = hashlib.sha256(np.array(initial).tobytes() + np.array(mu, dtype=np.float32).tobytes()).hexdigest()
    unified_protocol = dict(task='fudan_ppo_step', backend='mujoco_warp_gpu', samples=cases,
        duration_s=core_p['seconds'], dt=.01, recipe=core_p['recipe'],
        checkpoint_sha256=receipt['checkpoint_sha256'], cohort_sha256=cohort,
        source_fingerprint=core_p['source_fingerprint'], trace_sha256=core_p['trace_sha256'],
        source_protocol_sha256=sha(root / 'core/protocol.json'), optimizer_updates=0,
        scope='Derived recording manifest of the already collected forward high-speed subset; full matrix incomplete')
    write_or_verify(root / 'core/step-protocol.json', unified_protocol)
    write_or_verify(root / 'flat47/gpu/step-policy-receipt.json',
        dict(checkpoint_sha256=receipt['checkpoint_sha256'], optimizer_updates=0,
             source_plan_sha256=sha(root / 'flat47/plan.json')))
    if not (root / 'unified-acceptance').exists():
        with (root / 'unified-accept.log').open('x') as stream:
            subprocess.run([str(ROOT / 'scripts/mjlab.sh'), 'accept', '--task', 'fudan_ppo_step',
                '--recording', str(root / 'core'), '--output', str(root / 'unified-acceptance'),
                '--flat-recording', str(root / 'flat47/gpu'),
                '--flat-baseline', str(previous / 'baseline-flat47/gpu')], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=stream, stderr=subprocess.STDOUT, check=True)
    unified = json.loads((root / 'unified-acceptance/acceptance.json').read_text())
    assert unified['details']['checkpoint_sha256'] == receipt['checkpoint_sha256']
    assert unified['flat_regression']['cohort_sha256'] == flat['cohort_sha256']
    assert unified['flat_regression']['samples'] == flat['samples']
    assert not unified['step_matrix_complete'] and not unified['full_task_accepted']
    assert unified['technical_passed'] == sum(r['task_score']['passed'] for r in core if r['height'] > 0)
    details = [detail(root / 'core', case, next(r for r in core if r['id'] == case['id']))
               for case in core_p['cases'] if case['height'] > 0]
    videos = []
    import imageio.v2 as imageio
    for folder in sorted(path for path in root.glob('video-*') if path.is_dir()):
        media = json.loads((folder / 'media.json').read_text())
        path = folder / 'replay-1x.mp4'
        assert sha(path) == media['video_sha256']
        recording = root / ('nominal' if 'nominal' in folder.name else 'core')
        assert sha(recording / 'traces' / (media['case'] + '.npz')) == media['trace_sha256']
        with imageio.get_reader(path) as reader:
            assert reader.get_meta_data()['fps'] == media['fps']
            count = sum(1 for _ in reader)
        assert count == media['frames']
        videos.append(dict(directory=folder.name, **media, full_decode_passed=True))
    result = dict(evidence_verified=True, checkpoint_sha256=receipt['checkpoint_sha256'],
        core_groups=groups(core), nominal_groups=groups(nominal), comparisons=comparisons,
        flat_passed=flat['technical_passed'], flat_total=47, flat_comparison=flat['comparison'],
        flat_failed=[r for r in flat['samples'] if not r['passed']], core_details=details,
        videos=videos, optimizer_updates=0, full_task_accepted=False,
        scope='Frozen nominal GPU subsets; no full randomized matrix, full arena, get-up or hardware acceptance.')
    with (root / 'review.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ('core_groups', 'nominal_groups', 'flat_passed', 'flat_comparison')}))


if __name__ == '__main__':
    main()
