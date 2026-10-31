#!/usr/bin/env python3
"""CPU-only phased summary of saved 30s matched walk-feedback traces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


FIXED_KEYS = ('qpos', 'qvel', 'nominal', 'kp', 'kd', 'strength', 'mass', 'friction')
PROTOCOL_KEYS = ('checkpoint_iteration', 'seed', 'seconds',
                 'warmup', 'stop_at', 'height', 'dt', 'policy_dt', 'weight_N',
                 'layout', 'joint_names', 'cases', 'timing', 'schedule', 'wheel_feedback',
                 'wheel_radius_m', 'wheel_vx_coefficient_m')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()


def distribution(values):
    observed = [(i, float(v)) for i, v in enumerate(values) if v is not None]
    if not observed:
        return dict(observed_environments=0, missing_environments=len(values),
                    minimum=None, median=None, p95=None, maximum=None, worst_environment=None)
    a = np.asarray([v for _, v in observed])
    return dict(observed_environments=len(a), missing_environments=len(values)-len(a),
        minimum=float(a.min()), median=float(np.median(a)), p95=float(np.percentile(a, 95)),
        maximum=float(a.max()), worst_environment=observed[int(np.argmax(a))][0])


def array_receipt(value, name):
    return dict(saved_name=name, shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def scalar_metrics(data, env, mask, speed):
    if not mask.any():
        return None
    vx = data['vx'][mask, env]
    velocity = data['world_velocity'][mask, env, :2]
    horizontal = np.linalg.norm(velocity, axis=-1)
    pitch = abs(data['pitch'][mask, env])
    height = data['height'][mask, env]
    error = None if speed is None else abs(vx-speed)
    return dict(vx_mean_m_s=float(vx.mean()), vx_mean_bias_m_s=None if speed is None else float((vx-speed).mean()),
        vx_mae_m_s=None if error is None else float(error.mean()),
        vx_abs_error_p95_m_s=None if error is None else float(np.percentile(error, 95)),
        vx_abs_error_max_m_s=None if error is None else float(error.max()),
        world_horizontal_speed_max_m_s=float(horizontal.max()),
        world_horizontal_speed_p95_m_s=float(np.percentile(horizontal, 95)),
        world_horizontal_speed_over_0_05_duration_s=float((horizontal > .05).sum()*data['dt']),
        world_abs_vx_max_m_s=float(abs(velocity[:, 0]).max()),
        world_abs_vy_max_m_s=float(abs(velocity[:, 1]).max()),
        pitch_abs_max_deg=float(pitch.max()), pitch_abs_p95_deg=float(np.percentile(pitch, 95)),
        pitch_abs_over_5_duration_s=float((pitch > 5.).sum()*data['dt']),
        height_mean_m=float(height.mean()), height_mae_m=float(abs(height-data['target_height']).mean()),
        actual_leg_velocity_abs_max_rad_s=float(abs(data['joint_dq'][mask, env, :4]).max()),
        angular_velocity_xy_abs_max_rad_s=float(abs(data['angular_velocity_xy'][mask, env]).max()))


def spectrum_hint(value, dt):
    if len(value)*dt < 3.-dt:
        return None
    channel = int(np.argmax(np.ptp(value, axis=0)))
    v = value[:, channel].astype(float)
    x = np.linspace(-1., 1., len(v))
    v = v-v.mean()-x*(x@v/(x@x))
    power = abs(np.fft.rfft(v*np.hanning(len(v))))**2
    frequency = np.fft.rfftfreq(len(v), dt)
    band = (frequency >= .5) & (frequency <= 10.)
    dominant = float(frequency[np.flatnonzero(band)[np.argmax(power[band])]])
    selected = band & (abs(frequency-dominant) <= .5)
    return dict(motor_index=channel, dominant_hz=dominant,
        resolution_hz=1/(len(v)*dt),
        dominant_plusminus_0_5Hz_power_fraction=float(power[selected].sum()/max(power[band].sum(), 1e-20)),
        interpretation='Descriptive recurrence hint, not a confirmed periodicity or task acceptance threshold.')


def phase_summary(data, env, start, end, speed):
    times, dt = data['times'], data['dt']
    planned = (times > start+1e-9) & (times <= end+1e-9)
    mask = planned & data['valid'][:, env]
    complete = bool(mask.sum() == planned.sum() and planned.any())
    metrics = scalar_metrics(data, env, mask, speed)
    windows = []
    # 只把完整1秒首回合窗口用于摆幅，缺样不按零摆幅处理。
    for i in range(int(np.floor(end-start+1e-9))):
        a, b = start+i, start+i+1.
        intended = (times > a+1e-9) & (times <= b+1e-9)
        live = intended & data['valid'][:, env]
        full = bool(live.sum() == intended.sum() and intended.any())
        motor = None if not full else np.rad2deg(np.ptp(np.unwrap(data['motor'][live, env], axis=0), axis=0))
        swing = None if not full else np.ptp(data['swing'][live, env], axis=0)
        length = None if not full else np.ptp(data['length'][live, env], axis=0)
        windows.append(dict(start_s=a, end_s=b, complete=full, observed_samples=int(live.sum()),
            motor_peak_to_peak_deg=None if motor is None else motor.tolist(),
            four_motor_max_peak_to_peak_deg=None if motor is None else float(motor.max()),
            virtual_leg_swing_peak_to_peak_deg=None if swing is None else swing.tolist(),
            virtual_leg_swing_max_peak_to_peak_deg=None if swing is None else float(swing.max()),
            leg_length_peak_to_peak_m=None if length is None else length.tolist(),
            leg_length_max_peak_to_peak_m=None if length is None else float(length.max())))
    streak, longest, first = 0, 0, None
    for i, window in enumerate(windows):
        over = window['complete'] and window['four_motor_max_peak_to_peak_deg'] > 2.
        streak = streak+1 if over else 0
        longest = max(longest, streak)
        if streak == 3 and first is None:
            first = i-2
    hint = None
    if first is not None:
        a, b = windows[first]['start_s'], windows[first+2]['end_s']
        segment = (times > a+1e-9) & (times <= b+1e-9)
        hint = spectrum_hint(np.rad2deg(np.unwrap(data['motor'][segment, env], axis=0)), dt)
    def maximum(key):
        observed = [w[key] for w in windows if w[key] is not None]
        return None if not observed else max(observed)
    return dict(start_s=start, end_s=end, expected_samples=int(planned.sum()),
        observed_samples=int(mask.sum()), complete=complete,
        metrics=metrics if complete else None,
        partial_observed_metrics=metrics if not complete else None,
        one_second_windows=windows,
        one_second_motor_max_peak_to_peak_deg=maximum('four_motor_max_peak_to_peak_deg'),
        one_second_virtual_leg_swing_max_peak_to_peak_deg=maximum('virtual_leg_swing_max_peak_to_peak_deg'),
        one_second_leg_length_max_peak_to_peak_m=maximum('leg_length_max_peak_to_peak_m'),
        longest_consecutive_1s_motor_amplitude_over_proposed_2deg=longest,
        persistent_motion_amplitude_proposal_flag=None if not complete else bool(longest >= 3),
        partial_observed_persistent_motion_amplitude_proposal_flag=bool(longest >= 3) if not complete else None,
        periodicity_spectrum_hint=hint)


def transition_pitch(data, env, start, end):
    times = data['times']
    planned = (times > start+1e-9) & (times <= end+1e-9)
    mask = planned & data['valid'][:, env]
    if not mask.any():
        return dict(event_s=start, observed_samples=0, complete=False,
                    pitch_abs_max_deg=None, first_return_after_exceedance_s=None,
                    final_observed_return_after_last_exceedance_s=None)
    t, pitch = times[mask], abs(data['pitch'][mask, env])
    over = pitch > 5.
    first = np.flatnonzero(over)
    first_return, final_return = 0., 0.
    if len(first):
        returns = np.flatnonzero((np.arange(len(t)) > first[0]) & ~over)
        first_return = None if not len(returns) else float(t[returns[0]]-start)
        final_return = None if first[-1]+1 >= len(t) else float(t[first[-1]+1]-start)
    return dict(event_s=start, observed_until_s=float(t[-1]), observed_samples=len(t),
        complete=bool(mask.sum() == planned.sum()), pitch_abs_max_deg=float(pitch.max()),
        pitch_abs_over_5_duration_s=float(over.sum()*data['dt']),
        first_exceedance_after_event_s=None if not len(first) else float(t[first[0]]-start),
        first_return_after_exceedance_s=first_return,
        final_observed_return_after_last_exceedance_s=final_return,
        allowed_excursion_duration_s=None,
        interpretation='Observed recovery only; allowed start/stop excursion duration is unconfirmed. A phase boundary does not grant an exemption.')


def load_measurement(label, prefix):
    prefix = Path(prefix).resolve()
    metadata, trace = prefix.with_suffix('.json'), prefix.with_suffix('.npz')
    meta = json.loads(metadata.read_text())
    assert meta['parameter_updates'] == 0 and meta['actor_unchanged'] is True
    assert meta['encoder_unchanged'] is True
    assert meta['actor_state_sha256'] == meta['actor_state_sha256_after']
    for key in ('checkpoint_sha256', 'actor_state_sha256'):
        assert len(meta[key]) == 64
        int(meta[key], 16)
    mode = meta['feedback_mode']
    assert mode in ('encoder', 'wheel-vx') and meta['velocity_source'] == mode
    assert meta['seconds'] == 30. and meta['warmup'] == 2. and meta['stop_at'] == 16.
    assert meta['height'] == .26 and meta['dt'] == .002 and meta['policy_dt'] == .01
    assert meta['seed'] == 43
    wheel = meta['wheel_feedback']
    assert wheel['radius_m'] == .06 and wheel['forward_signs'] == [1, -1]
    assert wheel['actor_indices'] == [15, 18] and wheel['observation_scale'] == .05
    coefficient = .5*wheel['radius_m']
    assert meta['wheel_radius_m'] == .06 and meta['wheel_vx_coefficient_m'] == coefficient
    protocol = {key: meta[key] for key in PROTOCOL_KEYS}
    with np.load(trace, allow_pickle=False) as saved:
        state, receipts, missing = {}, {}, []
        for key in FIXED_KEYS:
            names = [n for n in ('fixed_'+key, 'initial_'+key) if n in saved.files]
            if not names:
                missing.append(key)
                continue
            value = saved[names[0]]
            assert np.isfinite(value).all()
            assert all(equal(value, saved[n]) for n in names)
            state[key] = value
            receipts[key] = array_receipt(value, names[0])
        initial = {key: saved['initial_'+key] for key in ('qpos', 'qvel') if 'initial_'+key in saved.files}
        saved_final_live = saved['qpos_live'][-1] if 'qpos_live' in saved.files else None
        values = saved['values']
    assert values.ndim == 3 and values.shape[:2] == (15000, 96) and np.isfinite(values).all()
    def get(name):
        return values[..., slice(*meta['layout'][name])]
    times = (np.arange(len(values))+1)*meta['dt']
    valid_values = get('valid')[..., 0]
    assert ((valid_values == 0.) | (valid_values == 1.)).all()
    valid = valid_values > .5
    assert valid[0].all() and not (valid[1:] & ~valid[:-1]).any()
    ends = np.asarray(meta['episode_end'])
    assert ends.shape == (96,) and np.isfinite(ends).all()
    assert ((ends > 0.) & (ends <= 30.)).all()
    assert all(isinstance(done, bool) for done in meta['episode_done'])
    done = np.asarray(meta['episode_done'], dtype=bool)
    assert done.shape == (96,) and meta['survived'] == int((~done).sum())
    assert not ((ends < 30.-1e-6) & ~done).any()
    if saved_final_live is not None:
        assert np.array_equal(saved_final_live, ~done), 'Final qpos mask differs from episode_done'
    assert np.array_equal(valid.sum(0), np.rint(ends/meta['dt']).astype(int))
    used, encoder, wheels = get('policy_velocity'), get('encoder_velocity'), get('policy_wheel_velocity')
    assert used.shape[-1] == encoder.shape[-1] == 3 and wheels.shape[-1] == 2
    assert np.allclose(used[..., 1:], encoder[..., 1:], rtol=0., atol=1e-7)
    expected_vx = encoder[..., 0] if mode == 'encoder' else coefficient*(wheels[..., 0]-wheels[..., 1])
    assert np.allclose(used[..., 0], expected_vx, rtol=1e-6, atol=1e-7)
    roll_pitch = get('roll_pitch')
    assert roll_pitch.shape[-1] == 2
    assert np.allclose(get('applied_command')[..., 2][valid], .26, rtol=0., atol=1e-7)
    joint_ids = [meta['joint_names'].index(n) for n in ('lf0', 'lf00', 'rf0', 'rf00')]
    data = dict(times=times, dt=meta['dt'], target_height=.26, valid=valid,
        vx=get('true_velocity')[..., 0], world_velocity=get('root_linear_velocity_w'),
        pitch=roll_pitch[..., 1], height=get('height')[..., 0],
        motor=get('joint_q')[..., joint_ids], joint_dq=get('joint_dq'),
        angular_velocity_xy=get('angular_velocity_xy'), swing=get('swing'), length=get('leg_length'))
    assert data['world_velocity'].shape[-1] == 3 and data['joint_dq'].shape[-1] == 8
    assert data['angular_velocity_xy'].shape[-1] == 2
    cases, all_ids, expected_cases = [], [], {(speed, acceleration) for speed in (0., .5, -.5) for acceleration in (.5, 2.)}
    assert len(meta['cases']) == 6
    for case in meta['cases']:
        speed, acceleration = case['speed'], case['acceleration']
        assert (speed, acceleration) in expected_cases
        expected_cases.remove((speed, acceleration))
        ids = case['env_ids']
        assert len(ids) == 16
        all_ids.extend(ids)
        ramp_end, brake_end = 2.+abs(speed)/acceleration, 16.+abs(speed)/acceleration
        phase_specs = [('initial', 0., 2., 0.), ('start', 2., ramp_end+1.5, speed),
            ('target_platform', ramp_end+1.5, 16., speed), ('brake', 16., brake_end+1.5, 0.),
            ('post_stop_platform', brake_end+1.5, 30., 0.)]
        environments = []
        for env in ids:
            phases = {name: phase_summary(data, env, a, b, ref) for name, a, b, ref in phase_specs}
            for name, a, b, ref in phase_specs:
                phase = phases[name]
                metrics = phase['metrics']
                phase['threshold_observations'] = dict(
                    confirmed_pitch_abs_max_le5=None if metrics is None else bool(metrics['pitch_abs_max_deg'] <= 5.),
                    confirmed_zero_speed_norm_max_le0_05=None if metrics is None or ref != 0. else bool(metrics['world_horizontal_speed_max_m_s'] <= .05),
                    proposed_nonzero_speed_mae_le0_05_and_p95_le0_10=None if metrics is None or name != 'target_platform' or speed == 0. else bool(metrics['vx_mae_m_s'] <= .05 and metrics['vx_abs_error_p95_m_s'] <= .10),
                    proposed_all_full_1s_motor_amplitudes_le2deg=None if not phase['complete'] or not phase['one_second_windows'] else bool(phase['one_second_motor_max_peak_to_peak_deg'] <= 2.),
                    transient_pitch_permitted_duration_unconfirmed=name in ('initial', 'start', 'brake'))
            zero_requested = np.ones(len(times), dtype=bool) if speed == 0. else ((times <= 2.+1e-9) | (times > 16.+1e-9))
            zero_mask = zero_requested & valid[:, env]
            zero_metrics = scalar_metrics(data, env, zero_mask, 0.)
            all_metrics = scalar_metrics(data, env, valid[:, env], None)
            environments.append(dict(env=env, survived=bool(not done[env]),
                episode_end_s=float(ends[env]), phases=phases,
                zero_requested_samples=int(zero_requested.sum()), zero_requested_observed_samples=int(zero_mask.sum()),
                zero_requested_observed_metrics=zero_metrics,
                first_episode_observed_metrics=all_metrics,
                pitch_transitions=dict(start=transition_pitch(data, env, 2., 16.),
                                       stop=transition_pitch(data, env, 16., 30.))))
        case_summary = dict(case=case['name'], speed_m_s=speed, acceleration_m_s2=acceleration,
            environments=16, survived=sum(e['survived'] for e in environments),
            per_environment=environments, phases={})
        for name, a, b, ref in phase_specs:
            phase_rows = [e['phases'][name] for e in environments]
            metric_names = list(environments[0]['first_episode_observed_metrics'])
            aggregate = dict(start_s=a, end_s=b, speed_reference_m_s=ref,
                complete_environments=sum(p['complete'] for p in phase_rows),
                metrics={key: distribution([None if p['metrics'] is None else p['metrics'][key] for p in phase_rows])
                         for key in metric_names})
            for key in ('one_second_motor_max_peak_to_peak_deg', 'one_second_virtual_leg_swing_max_peak_to_peak_deg',
                        'one_second_leg_length_max_peak_to_peak_m'):
                aggregate[key] = distribution([p[key] if p['complete'] else None for p in phase_rows])
            aggregate['persistent_motion_amplitude_proposal_flag_environments'] = sum(
                p['persistent_motion_amplitude_proposal_flag'] is True for p in phase_rows)
            aggregate['metric_worst_environment_indices_are_case_local'] = True
            aggregate['threshold_observation_counts'] = {}
            for key in ('confirmed_pitch_abs_max_le5', 'confirmed_zero_speed_norm_max_le0_05',
                        'proposed_nonzero_speed_mae_le0_05_and_p95_le0_10',
                        'proposed_all_full_1s_motor_amplitudes_le2deg'):
                flags = [p['threshold_observations'][key] for p in phase_rows]
                aggregate['threshold_observation_counts'][key] = dict(true=sum(v is True for v in flags),
                    false=sum(v is False for v in flags), unknown_or_not_applicable=sum(v is None for v in flags))
            case_summary['phases'][name] = aggregate
        case_summary['all_observed_zero_request_samples'] = dict(
            complete_environments=sum(e['zero_requested_observed_samples'] == e['zero_requested_samples'] for e in environments),
            metrics={key: distribution([None if e['zero_requested_observed_metrics'] is None else e['zero_requested_observed_metrics'][key] for e in environments])
                     for key in ('world_horizontal_speed_max_m_s', 'world_horizontal_speed_p95_m_s',
                                 'world_horizontal_speed_over_0_05_duration_s', 'world_abs_vx_max_m_s', 'world_abs_vy_max_m_s')},
            interpretation='All observed zero-request samples, including initial and braking. Failed episodes may be incomplete; observed maxima remain lower bounds, not a pass.')
        case_summary['first_episode_observed_pitch'] = {
            key: distribution([e['first_episode_observed_metrics'][key] for e in environments])
            for key in ('pitch_abs_max_deg', 'pitch_abs_p95_deg', 'pitch_abs_over_5_duration_s')}
        case_summary['pitch_transitions'] = {}
        for event in ('start', 'stop'):
            transitions = [e['pitch_transitions'][event] for e in environments]
            case_summary['pitch_transitions'][event] = dict(
                complete_environments=sum(t['complete'] for t in transitions),
                metrics={key: distribution([t.get(key) for t in transitions]) for key in
                    ('pitch_abs_max_deg', 'pitch_abs_over_5_duration_s',
                     'first_return_after_exceedance_s', 'final_observed_return_after_last_exceedance_s')},
                permitted_duration_s=None)
        cases.append(case_summary)
    assert not expected_cases and sorted(all_ids) == list(range(96))
    return dict(label=label, feedback_mode=mode, measurement=str(prefix), checkpoint=meta['checkpoint'],
        checkpoint_sha256=meta['checkpoint_sha256'], actor_state_sha256=meta.get('actor_state_sha256'),
        actor_state_sha256_after=meta['actor_state_sha256_after'], encoder_unchanged=True,
        parameter_updates=0, actor_unchanged=True, saved_fixed_arrays=receipts,
        missing_fixed_arrays=missing, saved_initial_arrays={k: array_receipt(v, 'initial_'+k) for k, v in initial.items()},
        missing_initial_arrays=[k for k in ('qpos', 'qvel') if k not in initial],
        input_files={str(p): digest(p) for p in (metadata, trace)},
        feedback_formula_max_abs_residual_m_s=float(abs(used[..., 0]-expected_vx).max()),
        preserved_vy_vz_max_abs_residual_m_s=float(abs(used[..., 1:]-encoder[..., 1:]).max()),
        cases=cases), protocol, state


def main(args):
    labels = [label for label, _ in args.measurement]
    assert len(labels) >= 2 and len(set(labels)) == len(labels) and args.baseline in labels
    assert not args.output.exists(), 'Use a new JSON; preserve previous results'
    runs, states, protocol, checkpoint_receipts = [], [], None, {}
    for label, prefix in args.measurement:
        run, current, state = load_measurement(label, prefix)
        if protocol is None:
            protocol = current
        else:
            assert current == protocol, 'Protocol differs beyond feedback mode and its velocity_source alias'
            assert run['checkpoint_sha256'] == runs[0]['checkpoint_sha256']
            if run['actor_state_sha256'] is not None and runs[0]['actor_state_sha256'] is not None:
                assert run['actor_state_sha256'] == runs[0]['actor_state_sha256']
            for key in FIXED_KEYS:
                for previous in states:
                    if key in state and key in previous:
                        assert equal(state[key], previous[key]), f'{label}: {key} differs'
        checkpoint = Path(run['checkpoint']).resolve()
        if str(checkpoint) not in checkpoint_receipts:
            checkpoint_receipts[str(checkpoint)] = digest(checkpoint) if checkpoint.exists() else None
        assert checkpoint_receipts[str(checkpoint)] in (None, run['checkpoint_sha256'])
        runs.append(run)
        states.append(state)
    baseline = next(r for r in runs if r['label'] == args.baseline)
    assert baseline['feedback_mode'] == 'encoder'
    for run in runs:
        lost = []
        for case, old_case in zip(run['cases'], baseline['cases'], strict=True):
            assert case['case'] == old_case['case']
            old_alive = {e['env'] for e in old_case['per_environment'] if e['survived']}
            alive = {e['env'] for e in case['per_environment'] if e['survived']}
            case['lost_baseline_survivor_envs'] = sorted(old_alive-alive)
            case['gained_survivor_envs'] = sorted(alive-old_alive)
            case['lost_baseline_descriptive_threshold_observation_envs'] = {}
            for phase_name in case['phases']:
                comparisons = {}
                for criterion in ('confirmed_pitch_abs_max_le5', 'confirmed_zero_speed_norm_max_le0_05',
                                  'proposed_nonzero_speed_mae_le0_05_and_p95_le0_10',
                                  'proposed_all_full_1s_motor_amplitudes_le2deg'):
                    def retained(source):
                        return {e['env'] for e in source['per_environment'] if e['survived']
                            and e['phases'][phase_name]['threshold_observations'][criterion] is True}
                    comparisons[criterion] = sorted(retained(old_case)-retained(case))
                case['lost_baseline_descriptive_threshold_observation_envs'][phase_name] = comparisons
            lost.extend(case['lost_baseline_survivor_envs'])
        run['lost_baseline_survivor_envs'] = lost
        run['survived'] = sum(c['survived'] for c in run['cases'])
    report = dict(verified=True, baseline=args.baseline, script=str(Path(__file__).resolve()),
        script_sha256=digest(Path(__file__)), invocation=[sys.executable]+sys.argv,
        parameter_updates=0, physics_steps=0, protocol=protocol,
        protocol_except_feedback_mode_equal=True,
        feedback_mode_alias='velocity_source carries the same encoder/wheel-vx label; both validated as one intervention.',
        checkpoint_files_sha256=checkpoint_receipts,
        actual_checkpoint_sha_verified=all(v is not None for v in checkpoint_receipts.values()),
        actor_invariance_evidence='Recorder metadata asserts tensor equality, saved actor SHA equality when available, and zero updates; CPU summary does not rerun a policy.',
        all_saved_common_fixed_arrays_bitwise_equal=True,
        all_eight_required_fixed_arrays_bitwise_equal=None if any(r['missing_fixed_arrays'] for r in runs) else True,
        initial_qpos_qvel_bitwise_equal=None if any(r['missing_initial_arrays'] for r in runs) else True,
        missing_state_array_rule='Missing arrays are unknown; runtime equality claims do not replace saved-array comparisons.',
        confirmed_thresholds=dict(zero_request_world_horizontal_speed_max_m_s=.05, steady_pitch_abs_max_deg=5.,
            transient_pitch_allowed_duration_s=None, percentile_rule='P95 is descriptive; it never relaxes a confirmed instantaneous bound.'),
        unconfirmed_proposals=dict(nonzero_vx_mae_m_s=.05, nonzero_vx_absolute_error_p95_m_s=.10,
            motor_max_peak_to_peak_per_1s_deg=2., persistent_motion_minimum_consecutive_1s_windows=3,
            status='Proposals only, not user-confirmed task acceptance.'),
        window_rule='Sample windows are (start,end]. Target/post-stop descriptive platforms start1.5s after nominal ramp completion. These windows grant no exemption from confirmed bounds or unspecified transient limits.',
        speed_rule='Nonzero tracking uses root-link body vx versus fixed request; parking drift uses true free-joint world horizontal norm. Start/brake errors remain descriptive during the ramp.',
        missing_data_rule='All96 initial environments retained. Complete-window metrics are null when samples are missing; partial observed metrics are explicitly separate. Failures are never dropped from denominators.',
        motor_order=['lf0', 'lf00', 'rf0', 'rf00'], roll_pitch_order=['roll', 'pitch'],
        acceptance_decision_performed=False, formal_balance_acceptance_performed=False,
        scope='Matched nominal zero-update feedback diagnosis; no reward/physics change or new training. Wheel-based vx is a candidate feedback input, not an accepted deployment or full-task model.',
        runs=runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(verified=True, output=str(args.output.resolve()),
        all_fixed_arrays_verified=report['all_eight_required_fixed_arrays_bitwise_equal'],
        runs=[dict(label=r['label'], survived=r['survived'], lost_baseline_survivors=len(r['lost_baseline_survivor_envs'])) for r in runs]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', nargs=2, action='append', required=True,
                        metavar=('LABEL', 'PATH_PREFIX'))
    parser.add_argument('--baseline', default='encoder')
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
