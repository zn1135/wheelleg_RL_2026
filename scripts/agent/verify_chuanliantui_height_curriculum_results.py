"""Verify saved height-curriculum results without simulation or learning.

Read raw tails, checkpoints, release hashes and videos; write only a new
directory under artifacts/height-curriculum-training-20261012.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True

import imageio.v2 as imageio
import numpy as np
import torch
from ct_mjlab.experiments import height_curriculum as h
from ct_mjlab.wheel_balance_assessment import paired_acceptance

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'artifacts/height-curriculum-training-20261012'
HEIGHTS = (.20, .22, .25, .27, .30, .34)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def close(actual, expected, tolerance=1e-7):
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=1e-7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == DATA or not output.is_relative_to(DATA):
        parser.error('Output must be a new subdirectory of the campaign artifacts')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(started=now(), ended=None, exit_code=None, learning=False,
                  simulation=False, command=shlex.join([sys.executable, *sys.argv]),
                  rows=[], scans=[], videos=[], models={}, inputs_sha256={})
    try:
        manifest = read(DATA / 'campaign-launch/frozen-release.json')['files']
        assert len(manifest) == 127
        assert all(digest(ROOT / path) == value for path, value in manifest.items())
        report['frozen127_sources_unchanged'] = True
        reference = read(ROOT / 'artifacts/balance-acceptance-v2-20261012/baseline/evaluation.json')
        source = h.validate_source(ROOT / 'artifacts/height-response-a2500-20261012/models/model_2499.pt')
        status = read(DATA / 'campaign-launch/status.json')
        assert status['state'] == 'completed' and not status['failures']
        assert (DATA / 'campaign-launch/worker.exit-code').read_text().strip() == '0'
        assert len(status['commands']) == 8
        assert all(item['exit_code'] == 0 for item in status['commands'])
        report['campaign'] = dict(started=status['started'], ended=status['ended'],
                                  worker_exit_code=0, commands=status['commands'])
        report['preflight'] = {}
        for folder in ('local-preflight-v2', 'remote-preflight-v1'):
            proof = read(DATA / folder / 'preflight.json')
            assert proof['passed'] and proof['exit_code'] == 0 and proof['frozen_sources_unchanged']
            assert proof['curriculum_checks']['passed']
            for item in proof['cases'].values():
                assert item['finite'] and item['all_reward_terms_finite']
                assert item['numerical_failures'] == 0 and item['configuration_isolation']
                if 'complete_state_equal' in item:
                    assert all(item['complete_state_equal'].values()) and item['own_checkpoint_reload']
            report['preflight'][folder] = dict(passed=True, cases=proof['cases'])
        for case in h.CASES:
            initial = torch.load(DATA / 'models' / case / 'model_resume_initial.pt',
                                 map_location='cpu', weights_only=False)
            h.validate_checkpoint(initial, case)
            assert all(h.source.state_equal(source[key], initial[key]) for key in
                       ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'))
            assert status['groups'][case]['exit_code'] == 0
            completion = read(DATA / 'campaign-results' / case / 'training-completion.json')
            assert completion['height_level_records'] == completion['height_curriculum']['history']
            assert completion['height_curriculum']['completed_updates'] == 3000
            levels = [json.loads(line.split('HEIGHT_LEVEL ', 1)[1]) for line in
                      (DATA / 'campaign-results' / case / 'training.log').read_text().splitlines()
                      if line.startswith('HEIGHT_LEVEL ')]
            assert levels == completion['height_level_records']
            report.setdefault('training', {})[case] = completion
            for point, name, updates in (('early101', 'model_2600.pt', 2601),
                                         ('final500', 'model_2999.pt', 3000)):
                folder = DATA / 'campaign-results' / case / point
                path = folder / 'evaluation.json'
                evaluation = read(path)
                report['inputs_sha256'][str(path.relative_to(ROOT))] = digest(path)
                assert evaluation['exit_code'] == 0 and evaluation['completed_updates'] == updates
                assert len(evaluation['records']) == 12 and evaluation['device'] == 'cuda:0'
                model_path = DATA / 'models' / case / name
                saved = torch.load(model_path, map_location='cpu', weights_only=False)
                info = h.validate_checkpoint(saved, case)
                assert info['completed_updates'] == updates
                assert json.loads(json.dumps(info['height_curriculum'])) == evaluation['height_curriculum']
                assert digest(model_path) == evaluation['checkpoint_sha256']
                assert all(torch.isfinite(value).all() for value in saved['actor_state_dict'].values())
                report['models'][f'{case}/{name}'] = dict(sha256=digest(model_path),
                    completed_updates=updates, level_m=info['height_curriculum']['level_m'])
                assert paired_acceptance(evaluation, reference) == evaluation['acceptance']
                for pose in ('normal', 'ground'):
                    rows = []
                    for command in HEIGHTS:
                        row = evaluation['records'][f'{pose}:{case}:{command:.2f}']
                        raw_path = folder / f'{pose}-h{command:.2f}.json'
                        raw = read(raw_path)
                        report['inputs_sha256'][str(raw_path.relative_to(ROOT))] = digest(raw_path)
                        assert row == raw['summary'] and row['numerical_failures'] == 0
                        height = np.asarray(raw['tail_height'], dtype=np.float32)
                        assert height.shape == (250, 64) and np.isfinite(height).all()
                        close(height.mean(), row['mean_height_m'])
                        close(height.std(), row['pooled_height_std_m'])
                        close(height.mean(0), row['per_env_mean_height_m'])
                        fractions = row['component_fractions']
                        for value in fractions.values():
                            fraction = np.asarray(value)
                            assert fraction.shape == (64,) and ((0 <= fraction) & (fraction <= 1)).all()
                            close(fraction * 250, np.rint(fraction * 250), 2e-5)
                        for metric, count, samples in (
                            ('wheel_balance', 'stable_balance', 'stable_samples'),
                            ('height_tracking', 'height_tracking', 'height_tracking_samples')):
                            ids = np.where(np.asarray(fractions[metric]) >= .95)[0].tolist()
                            assert ids == row[samples] and len(ids) == row[count]
                        knees = raw['tail_knee_constraints']
                        for key, values in knees.items():
                            values = np.asarray(values)
                            assert values.shape == (250, 64, 2) and np.isfinite(values).all()
                            close(values.mean((0, 1)), row['knee_limit_distributions'][key]['mean'])
                        active = np.asarray(knees['knee_limit_constraint_count']) > 0
                        close(active.mean((0, 1)), row['knee_limit_activation_fraction'])
                        close(np.asarray(knees['knee_limit_generalized_torque']).mean((0, 1)),
                              row['knee_limit_generalized_torque_mean_nm'])
                        rows.append(row)
                        report['scans'].append(dict(case=case, updates=updates, **row))
                    fit = evaluation['fits'][pose]
                    x = [row['command_height_m'] for row in rows]
                    close(np.polyfit(x, [row['mean_height_m'] for row in rows], 1)[0],
                          fit['slope_m_per_m'])
                    common = sorted(set.intersection(*(set(row['stable_samples']) for row in rows)))
                    assert common == fit['common_balanced_samples'] and len(common) == fit['common_balanced_count']
                    if common:
                        close(np.polyfit(x, [np.mean(np.asarray(row['per_env_mean_height_m'])[common])
                                            for row in rows], 1)[0], fit['common_balanced_slope_m_per_m'])
                    else:
                        assert fit['common_balanced_slope_m_per_m'] is None
                    row = next(row for row in rows if row['command_height_m'] == .30)
                    report['rows'].append(dict(case=case, point=point, updates=updates, pose=pose,
                        level_m=evaluation['height_curriculum']['level_m'],
                        mean_height_m=row['mean_height_m'], balance=row['stable_balance'],
                        height_tracking=row['height_tracking'], fit=fit,
                        acceptance=evaluation['acceptance']['poses'][pose],
                        component_failures={key:64 - int((np.asarray(value) >= .95).sum())
                                            for key, value in row['component_fractions'].items()},
                        component_failure_samples={key:np.where(np.asarray(value) < .95)[0].tolist()
                                            for key, value in row['component_fractions'].items()},
                        knee_limit_activation_fraction=row['knee_limit_activation_fraction'],
                        knee_limit_generalized_torque_mean_nm=row['knee_limit_generalized_torque_mean_nm']))
        for item in read(DATA / 'video-verification.json')['videos']:
            path = ROOT / item['path']
            assert digest(path) == item['sha256']
            with imageio.get_reader(path) as reader:
                metadata = reader.get_meta_data()
                assert reader.count_frames() == 200
                close(metadata['fps'], 10.)
                close(metadata['duration'], 20.)
                assert list(metadata['size']) == [640, 320]
                assert reader.get_data(0).shape == reader.get_data(199).shape == (320, 640, 3)
            report['videos'].append(item)
        assert len(report['scans']) == 48 and len(report['videos']) == 8
        report['exit_code'] = 0
        print(json.dumps(dict(exit_code=0, raw_cases=48, videos=8,
                              frozen127_sources_unchanged=True,
                              rows=[{key:row[key] for key in ('case', 'updates', 'pose',
                                  'balance', 'height_tracking', 'mean_height_m')}
                                    for row in report['rows']]), ensure_ascii=False))
    finally:
        report['ended'] = now()
        if report['exit_code'] is None:
            report['exit_code'] = 1
        (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False,
            indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
