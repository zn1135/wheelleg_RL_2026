"""Review existing video-disputed flags; never change scoring or run physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .diagnose_load_failures import intervals
from .score import score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_summary(wheels, weight):
    ratio = wheels.sum(axis=1) / weight
    bad = (wheels < np.maximum(.1 * weight, 5.)[:, None]).any(axis=1)
    bad |= (ratio < .8) | (ratio > 1.2)
    return dict(total_mean=float(ratio.mean()), total_min=float(ratio.min()),
                total_max=float(ratio.max()), bad_fraction=float(bad.mean())), bad


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--video-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.recording / 'diagnostic_summary.json').read_text())
    video_index = json.loads(args.video_index.read_text())
    videos = {row['sample_id']: row for row in video_index['chapters']}
    rows, traces = [], {}
    for old in original['samples']:
        if old['passed']:
            continue
        sid = old['sample_id']
        path = args.recording / 'traces' / f'{sid}.npz'
        digest = sha(path)
        with np.load(path, allow_pickle=False) as saved:
            trace = {key: saved[key].copy() for key in saved.files}
        recomputed = score({'id': sid, 'case': old['case']}, trace)
        for key in ('passed', 'complete', 'finite', 'fell', 'terminated', 'balance'):
            assert recomputed[key] == old[key], (sid, key)
        assert sha(Path(videos[sid]['path'])) == videos[sid]['sha256']
        duration = old['case']['duration']
        mask = (trace['time'] >= duration - 1. - 1e-9) & (trace['time'] < duration - 1e-9)
        wheels, weight = trace['wheel_support'][mask], trace['weight'][mask]
        raw, bad = load_summary(wheels, weight)
        raw['intervals'] = intervals(bad, trace['time'][mask])
        sensitivity = {}
        for milliseconds in (20, 50):
            # 只作平滑敏感性诊断，不用它替换原始成绩或设定新验收门槛。
            count = round(milliseconds / 2)
            kernel = np.ones(count) / count
            averaged = np.column_stack([np.convolve(wheels[:, i], kernel, 'valid') for i in range(2)])
            averaged_weight = np.convolve(weight, kernel, 'valid')
            sensitivity[str(milliseconds)] = load_summary(averaged, averaged_weight)[0]
            sensitivity[str(milliseconds)]['valid_windows'] = len(averaged_weight)
        body = trace['base_contact']
        peak = int(body.argmax())
        measurement = (trace['time'] >= 2.) & (trace['time'] < duration)
        row = dict(sample_id=sid, case=old['case']['id'],
            original_technical_passed=False, original_fell_flag=old['fell'],
            classification='brief_startup_body_contact' if old['fell'] else 'high_frequency_load_flag',
            tracking_passed=all(s['passed'] for s in recomputed['segments']),
            segments=recomputed['segments'], final_balance=recomputed['balance'],
            full_pitch_max_deg=float(np.abs(trace['roll_pitch_deg'][:, 1]).max()),
            full_roll_max_deg=float(np.abs(trace['roll_pitch_deg'][:, 0]).max()),
            measurement_body_force_max_N=float(body[measurement].max()),
            body_force_peak_N=float(body[peak]), body_force_peak_time_s=float(trace['time'][peak]),
            body_over10N=intervals(body > 10., trace['time']),
            final1s_raw_load=raw, averaging_sensitivity_ms=sensitivity,
            trace_sha256=digest, video=videos[sid]['path'], video_sha256=videos[sid]['sha256'])
        rows.append(row)
        traces[sid] = trace
        assert sha(path) == digest
    assert len(rows) == 7 and set(videos) == {row['sample_id'] for row in rows}
    report = dict(scope='Offline review of the seven existing 7000-iteration GPU clips',
        checkpoint_sha256=video_index['checkpoint_sha256'],
        user_review='User reports no observed failure in these seven videos',
        original_score_unchanged=True, thresholds_changed=False,
        original_technical_passed=original['technical_passed'], original_total=original['total'],
        raw_sample_period_s=.002, video_frame_period_s=.02,
        physics_steps=0, model_forwards=0, optimizer_updates=0,
        limitations=['Smoothing is diagnostic, not a new pass criterion',
            'Contact-force norm is not vertical force or signed impact impulse',
            'Zero support force alone does not prove geometric wheel lift',
            'No full-rate action, torque or joint-velocity trace is available to establish causality',
            'Only these seven nominal GPU cases were reviewed, not full randomization or hardware'],
        samples=rows)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    high = traces['diagnostic_6']
    mask = (high['time'] >= 6.) & (high['time'] < 7.)
    t = high['time'][mask]
    ratio = high['wheel_support'][mask].sum(axis=1) / high['weight'][mask]
    axes[0, 0].plot(t, ratio, lw=.8, color='#7d8895', label='Raw, every 2 ms')
    axes[0, 0].plot(np.convolve(t, np.ones(10)/10, 'valid'),
        np.convolve(ratio, np.ones(10)/10, 'valid'), color='#1768ac', label='20 ms moving mean')
    axes[0, 0].axhspan(.8, 1.2, color='#299b59', alpha=.12, label='Original load band')
    axes[0, 0].set(title='Forward 2.5 m/s: final-second wheel load', ylabel='Total wheel support / weight')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(t, high['roll_pitch_deg'][mask, 1], color='#1768ac')
    axes[0, 1].set(title='Same case: body pitch remains small', ylabel='Pitch (degrees)')
    contact = traces['diagnostic_9']
    start = contact['time'] < 1.
    axes[1, 0].plot(contact['time'][start], contact['base_contact'][start], color='#cb493b')
    axes[1, 0].axhline(10., color='#777777', ls='--', label='Original contact threshold')
    axes[1, 0].set(title='Backward turn: brief startup body contact', ylabel='Body contact-force norm (N)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(contact['time'], contact['roll_pitch_deg'][:, 1], color='#1768ac')
    axes[1, 1].axvline(2., color='#777777', ls='--', label='Tracking window starts')
    axes[1, 1].set(title='Same turn: recovery before measurement window', ylabel='Pitch (degrees)')
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel('Time (s)')
        axis.grid(alpha=.2)
    fig.savefig(args.output / 'review.png', dpi=160)
    plt.close(fig)
    print(json.dumps([dict(id=r['sample_id'], case=r['case'],
        raw_bad_fraction=r['final1s_raw_load']['bad_fraction'],
        max_bad_ms=r['final1s_raw_load']['intervals']['maximum_run_ms'],
        mean20_bad=r['averaging_sensitivity_ms']['20']['bad_fraction'],
        mean50_bad=r['averaging_sensitivity_ms']['50']['bad_fraction'],
        tracking_passed=r['tracking_passed'], body_peak=r['body_force_peak_N']) for r in rows], indent=2))


if __name__ == '__main__':
    main()
