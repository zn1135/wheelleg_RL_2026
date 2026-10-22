#!/usr/bin/env python3
"""Finite offline checks for real2sim dynamics; no serial connection or training."""
import argparse
import binascii
import csv
import json
from pathlib import Path
import sys
import tempfile

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'sim2sim'))
import real2sim_dynamics_huananhu as dynamics
import real2sim_huananhu as mirror


def packet(sequence, tick, q, *, stale=False, imu_stale=False, pulse=False, response_mode=0,
           applied=False, wrong_controller=False):
    f = np.zeros(160, dtype='<f4')
    f[3:6] = 7101, 2, 0
    for offset, value in ((6, sequence), (8, tick), (48, 1), (52, (tick-20) % 2**32),
                          (103, sequence), (105, tick), (107, 1)):
        f[offset:offset+2] = int(value) & 65535, (int(value) >> 16) & 65535
    sign, zero = mirror.load_mapping(dynamics.CONFIG)
    f[10:14] = q/sign[:4]+zero
    f[22], f[29], f[47], f[109], f[154] = 1, 127, 3, 15, 3
    f[122:126] = q
    if stale:
        f[142] = 200
    if imu_stale:
        f[30] = 200
    if pulse:
        f[136] = 3
        f[150] = 8000
    f[158] = response_mode
    if applied:
        f[109] = 61
        f[114:116], f[130:132] = [3., -3.], [2., -2.]
        value = .1*(f[114:116]-f[130:132])
        if response_mode == 1:
            value += np.asarray(mirror.WHEEL_RESPONSE_CATALOG['1']['friction_nm'])*np.tanh(f[130:132]/.5)
        f[136:138] = value*sign[4:]
        if wrong_controller:
            f[136] += .05
    f[-1] = binascii.crc_hqx(f[:-1].tobytes(), 0xffff)
    return f.tobytes()+mirror.TAIL


def make_session(directory, q, broken=False, response_mode=0, applied=False, wrong_controller=False):
    recorder = mirror.SessionRecorder(directory, dynamics.sim.MODEL_ASSET, dynamics.CONFIG)
    recorder.record('connected', elapsed_ns=0)
    for i in range(65):
        if broken and i == 40:
            recorder.record('disconnected', elapsed_ns=i*10_000_000-2)
            recorder.record('connected', elapsed_ns=i*10_000_000-1)
        seq = (2**32-5+i+(i >= 30 if broken else 0)) % 2**32
        tick = (2**32-20+i*10) % 2**32
        if broken and i >= 50:
            seq, tick = i-50, (i-50)*10
        raw = packet(seq, tick, q, stale=broken and i == 20, imu_stale=broken and i == 21, pulse=10 <= i <= 14,
                     response_mode=response_mode, applied=applied, wrong_controller=wrong_controller)
        if broken and i == 8:
            bad = bytearray(raw); bad[80] ^= 1
            recorder.record('data', elapsed_ns=i*10_000_000, data_hex=bad.hex())
        recorder.record('data', elapsed_ns=i*10_000_000, data_hex=raw.hex())
    recorder.finish(elapsed_ns=650_000_000)


def run_checks(directory):
    directory = Path(directory)
    env = dynamics.sim.Playback()
    q = env.data.qpos[env.q][dynamics.N2I[:4]].copy()
    make_session(directory/'clean', q)
    make_session(directory/'broken', q, broken=True)
    clean, broken = dynamics.Session(directory/'clean'), dynamics.Session(directory/'broken')
    make_session(directory/'compensated', q, response_mode=1, applied=True)
    compensated = dynamics.Session(directory/'compensated')
    assert compensated.wheel_response_mode == 1 and compensated.wheel_response_verified == 65
    assert np.array_equal(compensated.wheel_friction, mirror.WHEEL_RESPONSE_CATALOG['1']['friction_nm'])
    predictor = dynamics.BodyPredictor(compensated.model_path, compensated.config, compensated.wheel_friction)
    predictor.d.qvel[predictor.env.v[dynamics.sim.WHEELS]] = [2., -2.]
    torque, _ = predictor.env.controls(np.zeros(6))
    expected = -.1*np.array([2., -2.])+compensated.wheel_friction*np.tanh(np.array([2., -2.])/.5)
    np.testing.assert_allclose(torque[dynamics.sim.WHEELS], expected, rtol=0, atol=1e-12)
    make_session(directory/'wrong-controller', q, response_mode=1, applied=True, wrong_controller=True)
    try:
        dynamics.Session(directory/'wrong-controller')
    except ValueError as exc:
        assert 'control snapshot disagrees' in str(exc)
    else:
        raise AssertionError('wrong wheel controller accepted')
    meta_path = directory/'compensated/meta.json'
    meta = json.loads(meta_path.read_text());meta.pop('wheel_response_catalog')
    dynamics.save_json(meta_path, meta)
    try:
        dynamics.Session(directory/'compensated')
    except ValueError as exc:
        assert 'catalog missing or changed' in str(exc)
    else:
        raise AssertionError('new controller accepted without archived profile coefficients')
    make_session(directory/'unknown-controller', q, response_mode=2)
    try:
        dynamics.Session(directory/'unknown-controller')
    except ValueError as exc:
        assert 'unknown wheel response' in str(exc)
    else:
        raise AssertionError('unknown controller accepted')
    mixed = mirror.SessionRecorder(directory/'mixed-controller', dynamics.sim.MODEL_ASSET, dynamics.CONFIG)
    mixed.record('connected', elapsed_ns=0)
    for i in range(14):
        raw = packet(i, i*10, q, response_mode=int(i >= 7), applied=True)
        mixed.record('data', elapsed_ns=i*10_000_000, data_hex=raw.hex())
    mixed.finish(elapsed_ns=140_000_000)
    try:
        dynamics.Session(directory/'mixed-controller')
    except ValueError as exc:
        assert 'Mixed or unknown' in str(exc)
    else:
        raise AssertionError('mixed controller session accepted without segmentation')
    assert len(clean.f) == 65 and np.all(np.diff(clean.ms) == 10), 'natural tick wrap corrupted time'
    assert broken.summary['missing'] == 1 and broken.summary['restarts'] == 1
    assert broken.summary['crc_rejected_candidates'] >= 1
    assert not broken.valid[20] and not broken.continuous(19, 21)
    assert not broken.valid[21], 'stale IMU source accepted'
    assert not broken.continuous(29, 31) and not broken.continuous(39, 41)
    pulses, rejected, _ = dynamics.pulses_from_session(broken)
    assert not pulses and rejected, 'stale/gapped pulse window was accepted'
    good_pulses, rejected, _ = dynamics.pulses_from_session(clean)
    assert len(good_pulses) == 1 and not rejected
    pulse = good_pulses[0]
    parameters = np.array([[.004, .003, .12], [.004, .003, .12]])
    a, b = pulse.predict(parameters, 'request')
    score = dynamics.pulse_metrics(pulse, a, b)['velocity_error_rad_s']['mean']
    pulse.actual_v += 1
    assert abs(dynamics.pulse_metrics(pulse, a, b)['velocity_error_rad_s']['mean']-(score-1)) < 1e-10
    pulse.actual_q[:] = np.nan; pulse.actual_v[:] = np.nan
    clean.f[15:, 18:22] = np.nan
    aa, bb = pulse.predict(parameters, 'request')
    assert np.array_equal(a, aa) and np.array_equal(b, bb), 'future observations leaked into physics'
    assert pulse.fixture.m.nv == 2 and pulse.fixture.m.neq == 0
    mujoco.mj_forward(pulse.fixture.m, pulse.fixture.d)
    drift = float(np.max(abs(pulse.fixture.d.xpos-pulse.fixture.fixed_positions)))
    assert drift < 1e-12 and pulse.fixture.bake_error_m < 1e-12
    fixture = pulse.fixture
    fixture.m.opt.gravity[:] = 0
    fixture.parameters(parameters)
    mujoco.mj_forward(fixture.m, fixture.d)
    mass = np.empty((2, 2)); mujoco.mj_fullM(fixture.m, fixture.d, mass)
    _, v = fixture.predict(np.zeros((500, 2)), np.zeros(2), np.array((20., 0.)), parameters)
    time = np.arange(501)*.001
    j = mass[fixture.v[0], fixture.v[0]]
    expected = np.maximum(0, (20+.12/.003)*np.exp(-.003/j*time)-.12/.003)
    ode_error = float(max(abs(expected-v[:, 0])))
    assert ode_error < .2, ode_error
    # Changing sign must change acceleration sign in model coordinates.
    _, positive = fixture.predict(np.full((10, 2), .8), np.zeros(2), np.zeros(2), parameters)
    _, negative = fixture.predict(np.full((10, 2), -.8), np.zeros(2), np.zeros(2), parameters)
    assert np.all(positive[-1] > 0) and np.all(negative[-1] < 0)
    candidate = directory/'candidate.xml'
    integrity = dynamics.export_candidate(dynamics.sim.MODEL_ASSET, candidate, parameters)
    predictor = dynamics.BodyPredictor(candidate, dynamics.CONFIG)
    initial_q, initial_v = predictor.d.qpos.copy(), predictor.d.qvel.copy()
    actions = np.zeros((50, 6))
    pq, pv = predictor.predict(initial_q, initial_v, actions, 'lower_contact')
    predictor.pose.data.qpos[:] = np.nan
    qq, vv = predictor.predict(initial_q, initial_v, actions, 'lower_contact')
    assert np.array_equal(pq, qq) and np.array_equal(pv, vv)
    stream = directory/'clean/stream.jsonl'
    stream.write_bytes(stream.read_bytes()[:-9])
    try:
        dynamics.Session(directory/'clean')
    except ValueError as exc:
        assert 'hash mismatch' in str(exc)
    else:
        raise AssertionError('truncated recording accepted')
    try:
        dynamics.numeric_expression('__import__("os")')
    except ValueError:
        pass
    else:
        raise AssertionError('arbitrary code accepted in numeric expression')
    return {'raw_stream_crc_wrap_gap_reboot_disconnect_stale_truncation': True,
            'wheel_profile_archive_prediction_and_wrong_controller_rejection': True,
            'fixture_nv': 2, 'fixture_bake_error_m': pulse.fixture.bake_error_m,
            'fixture_position_drift_m': drift, 'ode_consistency_max_rad_s': ode_error,
            'bench_and_full_body_future_observation_independence': True,
            'candidate_integrity': integrity}


def verify_run(directory):
    """Recalculate every saved endpoint/statistic and rerun sampled predictions."""
    directory = Path(directory)
    report = json.loads((directory/'report.json').read_text())
    assert report['complete'] and json.loads((directory/'run.json').read_text())['complete']
    for name, digest in report['source_snapshot'].items():
        assert dynamics.identity(directory/'source'/name)['sha256'] == digest
    assert dynamics.identity(directory/'source'/Path(dynamics.__file__).name)['sha256'] == report['program']['sha256']
    rl_report = json.loads((directory/'rl-report.json').read_text())
    bench_report = json.loads((directory/'bench-report.json').read_text())
    candidate = directory/'candidate.xml'
    assert dynamics.identity(candidate)['sha256'] == bench_report['candidate_integrity']['sha256']
    args = report['command_arguments']
    resolve = lambda p: Path(p) if Path(p).is_absolute() else ROOT/p
    s = dynamics.Session(resolve(args['rl_session']), config=resolve(args['machine_config']))
    r = dynamics.RecordedRL(s)
    envs = {name: dynamics.BodyPredictor(path, s.config, s.wheel_friction) for name, path in
            (('original', s.model_path), ('candidate', candidate))}
    with np.load(directory/'rl-states.npz') as saved:
        qs, vs = saved['qpos'], saved['qvel']
        ix, cases, names = saved['row'], saved['case'], saved['model']
    lookup = {(int(i), str(c), str(m)): k for k, (i, c, m) in enumerate(zip(ix, cases, names))}
    assert len(lookup) == len(qs) == rl_report['evaluated_windows']*len(dynamics.BODY_CASES)*2
    columns = ([n+'_v_error' for n in mirror.MOTORS]+[n+'_q_error_deg' for n in mirror.MOTORS[:4]]
               +[n+'_constant_v_error' for n in mirror.MOTORS]+['pitch_error_deg', 'constant_gyro_pitch_error_deg'])
    grouped, count, max_error = {}, 0, 0.
    actual_pitch = {}
    with (directory/'rl-predictions.csv').open() as f:
        for row in csv.DictReader(f):
            i, h = int(row['row']), int(row['horizon_ms'])
            j, step = i+h//10, h//2
            k = lookup[i, row['case'], row['model']]
            env = envs[row['model']].env
            assert s.output_ms[j]-s.output_ms[i] == h and s.continuous(i, j)
            assert np.all((s.f[i:j+1, 109].astype(int) & 63) == 61), 'gated/failed output included in RL comparison'
            v = vs[k, step, env.v][dynamics.N2I]
            q = qs[k, step, env.q][dynamics.N2I[:4]]
            err = v-s.f[j, 126:132]
            qerr = np.degrees(q-s.f[j, 122:126])
            if j not in actual_pitch:
                actual_pitch[j] = dynamics.pitch(r.attitude(s.output_ms[j]))
            continuation = qs[k, 0].copy()
            mujoco.mj_integratePos(env.model, continuation, vs[k, 0], h/1000)
            measured = dict(zip(columns, np.r_[err, qerr, s.f[i, 126:132]-s.f[j, 126:132],
                np.degrees(dynamics.pitch(qs[k, step, 3:7])-actual_pitch[j]),
                np.degrees(dynamics.pitch(continuation[3:7])-actual_pitch[j])]))
            max_error = max(max_error, max(abs(float(row[c])-value) for c, value in measured.items()))
            max_error = max(max_error, max(abs(float(row[n+'_v_real'])-s.f[j, 126+a]) for a, n in enumerate(mirror.MOTORS)))
            max_error = max(max_error, max(abs(float(row[n+'_v_pred'])-v[a]) for a, n in enumerate(mirror.MOTORS)))
            for coverage in ('all', 'extrapolated' if int(row['extrapolated_over_37']) else 'within_bench'):
                key = f"{row['case']}/{h}/{row['model']}/{coverage}"
                values = grouped.setdefault(key, {c: [] for c in columns})
                for c, value in measured.items():
                    values[c].append(value)
            count += 1
    assert count == rl_report['evaluated_windows']*len(dynamics.BODY_CASES)*2*len(dynamics.HORIZONS)
    assert max_error < 1e-9, max_error
    stat_error, compared = 0., 0
    for key, values in grouped.items():
        assert rl_report['summaries'][key]['windows'] == len(values[columns[0]])
        for field, data in values.items():
            for statistic, value in dynamics.stats(data).items():
                stat_error = max(stat_error, abs(value-rl_report['summaries'][key][field][statistic]))
                compared += 1
    assert stat_error < 1e-9, stat_error
    for key, values in rl_report['summaries'].items():
        if key not in grouped:
            assert values['windows'] == 0 and all(values[c] is None for c in columns)
    regeneration = []
    for case in dynamics.BODY_CASES:
        for name, predictor in envs.items():
            selected = np.flatnonzero((cases == case) & (names == name))
            for k in selected[[0, len(selected)//2, -1]]:
                actions, _ = r.actions(int(ix[k]), dynamics.BODY_CASES[case][2])
                q, v = predictor.predict(qs[k, 0], vs[k, 0], actions, case)
                delta = float(max(np.max(abs(q-qs[k])), np.max(abs(v-vs[k]))))
                assert delta < 1e-10, (case, name, int(ix[k]), delta)
                regeneration.append(delta)
    b = dynamics.Session(resolve(args['bench_session']), config=resolve(args['machine_config']))
    pulses, rejected, _ = dynamics.pulses_from_session(b)
    assert not rejected
    params = np.array(bench_report['fits']['request/forward']['parameters'])
    bench_error = 0.
    with np.load(directory/'bench-states.npz') as saved:
        for pulse in pulses:
            for name, pmat in (('original', np.zeros((2, 3))), ('candidate', params)):
                q, v = pulse.predict(pmat, 'request')
                bench_error = max(bench_error, float(np.max(abs(q-saved[f'p{pulse.number}_{name}_q']))),
                                  float(np.max(abs(v-saved[f'p{pulse.number}_{name}_v']))))
    assert bench_error < 1e-10, bench_error
    high = np.any(abs(s.v[:, 4:]) > 37, axis=1)
    phases, phase_counts = np.unique(s.f[high, 154], return_counts=True)
    return {'rl_endpoints_verified': count, 'rl_statistics_recalculated': compared,
            'max_saved_endpoint_difference': max_error, 'max_summary_difference': stat_error,
            'full_body_trajectories_regenerated': len(regeneration), 'max_regenerated_difference': max(regeneration),
            'bench_trajectories_regenerated': len(pulses)*2, 'max_bench_difference': bench_error,
            'whole_session_high_speed_frames': int(high.sum()),
            'high_speed_frames_by_recovery_phase': dict(zip(phases.astype(int).astype(str), phase_counts.astype(int).tolist())),
            'current_program': dynamics.identity(Path(dynamics.__file__)), 'checker': dynamics.identity(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', type=Path, help='verify an existing complete dynamics run')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    with tempfile.TemporaryDirectory(prefix='real2sim-dynamics-check-') as directory:
        report = {'ok': True, 'scope': 'finite offline software/physics checks; no hardware acceptance',
                  'results': run_checks(directory)}
    if args.run:
        report['recorded_run_verification'] = verify_run(args.run)
    report['checker'] = dynamics.identity(Path(__file__))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dynamics.save_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
