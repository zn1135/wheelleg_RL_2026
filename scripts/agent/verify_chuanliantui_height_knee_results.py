"""Recompute saved scans/acceptance/curves and verify models/videos; no simulation."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
sys.dont_write_bytecode = True
import imageio.v2 as imageio
import numpy as np
import torch
from ct_mjlab.experiments import height_knee as h
from ct_mjlab.wheel_balance_assessment import paired_acceptance

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'artifacts/height-knee-training-20261013'
HEIGHTS = (.20, .22, .25, .27, .30, .34)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, tolerance=1e-7):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=1e-7)


def finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value,dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value,(list,tuple)):
        return all(finite_tree(x) for x in value)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert args.output.resolve().is_relative_to(DATA) and args.output.resolve() != DATA
    args.output.mkdir(parents=True, exist_ok=False)
    job = DATA/'campaign-results'
    reference = read(ROOT/'artifacts/balance-acceptance-v2-20261012/baseline/evaluation.json')
    original = h.validate_source(ROOT/'artifacts/height-response-a2500-20261012/models/model_2499.pt')
    launch = read(job/'launch.json')
    original_status = read(job/'status.json')
    status = read(job/'assessment-status.json')
    report = dict(started=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), learning=False,
        simulation=False, rows=[], scans=[], videos=[], models={}, training={}, exit_code=None,
        command=[sys.executable, *sys.argv])
    try:
        assert status['state'] == 'completed' and not status['failures']
        assert (job/'worker.exit-code').read_text().strip() == '1'
        assert original_status['state'] == 'coordinator_failed' and original_status['error'] == '2599'
        assert (job/'assessment.exit-code').read_text().strip() == '0'
        report['original_coordinator_failure'] = original_status
        check = read(job/'preflight/preflight.json')
        assert check['passed'] and check['frozen_sources_unchanged'] and check['exit_code'] == 0
        freeze = read(job/'frozen-release.json')
        current_bundle = read(DATA/'release-v2-bundle.json')
        assert freeze['files'] == current_bundle['files']
        assert all(digest(ROOT/k) == v for k,v in freeze['files'].items())
        report['frozen_release'] = freeze
        protected = read(DATA/'protected-before.json')
        assert all(digest(ROOT/k) == v for k,v in protected.items())
        report['unchanged_existing_files'] = len(protected)
        for case in h.CASES:
            group = status['groups'][case]
            assert group['completed_updates'] == 3500 and group['training_pid_absent'] and group['final_log_complete']
            assert group['exit_code'] == (0 if case=='knee_only' else None)
            assert group['exit_code_limitation'] is None if case=='knee_only' else group['exit_code_limitation']
            for folder in (job/'preflight'/case, DATA/'models'/case):
                initial = torch.load(folder/'model_resume_initial.pt', map_location='cpu', weights_only=False)
                assert h.validate_checkpoint(initial, case)['completed_updates'] == 2500
                assert all(h.source.state_equal(original[k], initial[k]) for k in
                    ('actor_state_dict','critic_state_dict','optimizer_state_dict'))
            replay = read(job/'source-replay'/case/'evaluation.json')
            assert replay['exit_code'] == 0 and replay['completed_updates'] == 2500
            assert paired_acceptance(replay, reference) == replay['acceptance']
            assert replay['acceptance']['balance_retention_pass']
            assert replay['poses']['normal']['sustained_last5s']['wheel_balance'] == 64
            assert replay['poses']['ground']['sustained_last5s']['wheel_balance'] == 60
            curve = read(job/case/'training-curve.json')
            assert len(curve['checkpoint_metrics']) == 10
            low = min(curve['checkpoint_metrics'], key=lambda x:(x['upright_fraction'], x['completed_updates']))
            assert low == curve['lowest_saved_checkpoint']
            points = sorted({3000, 3500, low['completed_updates']})
            assert points == status['groups'][case]['evaluated_updates']
            report['training'][case] = dict(curve=curve, status=status['groups'][case], run=launch['runs'][case])
            for updates in range(2600,3501,100):
                model = DATA/'models'/case/f'model_{updates-1}.pt'
                saved = torch.load(model,map_location='cpu',weights_only=False)
                assert h.validate_checkpoint(saved,case)['completed_updates'] == updates
                assert all(finite_tree(saved[k]) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict'))
                report['models'][f'{case}/{model.name}'] = dict(sha256=digest(model), completed_updates=updates)
            for updates in points:
                folder = job/case/f'eval-{updates}'
                evaluation = read(folder/'evaluation.json')
                assert evaluation['exit_code'] == 0 and evaluation['completed_updates'] == updates
                assert len(evaluation['records']) == 12
                assert digest(DATA/'models'/case/f'model_{updates-1}.pt') == evaluation['checkpoint_sha256']
                assert paired_acceptance(evaluation,reference) == evaluation['acceptance']
                for pose in ('normal','ground'):
                    rows = []
                    for command in HEIGHTS:
                        row = evaluation['records'][f'{pose}:{case}:{command:.2f}']
                        raw = read(folder/f'{pose}-h{command:.2f}.json')
                        assert raw['summary'] == row and row['numerical_failures'] == 0
                        assert row['initial_states_restored'] and row['initial_state_sha256'] == reference['poses'][pose]['initial_state_sha256']
                        height = np.array(raw['tail_height'],dtype=np.float32)
                        assert height.shape == (250,64) and np.isfinite(height).all()
                        close(height.mean(),row['mean_height_m'])
                        close(height.std(),row['pooled_height_std_m'])
                        close(height.mean(0),row['per_env_mean_height_m'])
                        for name,count,samples in (('wheel_balance','stable_balance','stable_samples'),
                                ('height_tracking','height_tracking','height_tracking_samples')):
                            ids = np.where(np.asarray(row['component_fractions'][name]) >= .95)[0].tolist()
                            assert ids == row[samples] and len(ids) == row[count]
                        for name, values in raw['tail_knee_constraints'].items():
                            values = np.array(values)
                            assert values.shape == (250,64,2) and np.isfinite(values).all()
                            close(values.mean((0,1)),row['knee_limit_distributions'][name]['mean'])
                        knees = raw['tail_knee_constraints']
                        close((np.array(knees['knee_limit_constraint_count'])>0).mean((0,1)),
                            row['knee_limit_activation_fraction'])
                        close(np.array(knees['knee_limit_generalized_torque']).mean((0,1)),
                            row['knee_limit_generalized_torque_mean_nm'])
                        rows.append(row)
                        report['scans'].append(dict(case=case,updates=updates,**row))
                    fit = evaluation['fits'][pose]
                    x = [r['command_height_m'] for r in rows]
                    close(np.polyfit(x,[r['mean_height_m'] for r in rows],1)[0],fit['slope_m_per_m'])
                    common = sorted(set.intersection(*(set(r['stable_samples']) for r in rows)))
                    assert common == fit['common_balanced_samples'] and len(common) == fit['common_balanced_count']
                    if common:
                        close(np.polyfit(x,[np.mean(np.array(r['per_env_mean_height_m'])[common]) for r in rows],1)[0],
                            fit['common_balanced_slope_m_per_m'])
                    else:
                        assert fit['common_balanced_slope_m_per_m'] is None
                    row = next(r for r in rows if r['command_height_m'] == .30)
                    report['rows'].append(dict(case=case,updates=updates,pose=pose,
                        mean_height_m=row['mean_height_m'],balance=row['stable_balance'],height_tracking=row['height_tracking'],
                        height_band_only=int((np.array(row['component_fractions']['target_height'])>=.95).sum()),
                        fit=fit,acceptance=evaluation['acceptance']['poses'][pose],verdict=evaluation['acceptance']['verdict'],
                        component_failures={k:64-int((np.array(v)>=.95).sum()) for k,v in row['component_fractions'].items()},
                        knee_limit_activation_fraction=row['knee_limit_activation_fraction'],
                        knee_limit_generalized_torque_mean_nm=row['knee_limit_generalized_torque_mean_nm']))
                    video = folder/f'comparison-{pose}.mp4'
                    with imageio.get_reader(video) as reader:
                        meta = reader.get_meta_data()
                        assert reader.count_frames() == 200
                        close(meta['fps'],10.)
                        close(meta['duration'],20.)
                        assert list(meta['size']) == [640,320]
                        assert reader.get_data(0).shape == reader.get_data(199).shape == (320,640,3)
                    report['videos'].append(dict(path=str(video.relative_to(ROOT)),sha256=digest(video),
                        frames=200,fps=10,duration_s=20,size=[640,320]))
            final = read(job/case/'eval-3500/evaluation.json')
            triggered = max(x['sustained_last5s']['height_tracking'] for x in final['poses'].values()) >= 32
            robustness = status['groups'][case]['robustness']
            if triggered:
                assert robustness == read(job/case/'robustness/robustness.json')
                assert robustness['exit_code'] == 0 and len(robustness['records']) == 3
                for row in robustness['records']:
                    assert row['num_envs'] == 256
                    assert row['falls_before5s']+row['falls_at_or_after5s'] == row['falls_total']
            else:
                assert not robustness['executed']
        report['exit_code'] = 0
        print(json.dumps(dict(exit_code=0, scans=len(report['scans']), videos=len(report['videos']),
            rows=[{k:r[k] for k in ('case','updates','pose','balance','height_tracking','mean_height_m','verdict')} for r in report['rows']])))
    finally:
        report['ended'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        if report['exit_code'] is None:
            report['exit_code'] = 1
        (args.output/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
