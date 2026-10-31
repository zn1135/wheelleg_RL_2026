#!/usr/bin/env python3
"""Independently check saved target-slew traces; no environment or learning."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LEGS = (0, 1, 3, 4)
WHEELS = (2, 5)
JOINTS = ('lf0', 'lf00', 'rf0', 'rf00')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def near(actual, expected, description, atol=2e-5):
    error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected)), initial=0.))
    require(error <= atol, f'{description}: maximum error {error} > {atol}')
    return error


def source_check(directory, report, repository):
    candidates = [directory/'assessment-source.py', directory/'source.py',
        repository/'scripts/agent/split/assess_getup_target_slew.py']
    matching = [str(p) for p in candidates if p.is_file() and digest(p) == report['source_sha256']]
    require(bool(matching), 'assessment source SHA has no matching saved/current file')
    checkpoint = Path(report['checkpoint'])
    require(digest(checkpoint) == report['checkpoint_sha256'], 'checkpoint SHA')
    manifests = []
    for launch in (repository/'artifacts').glob('*/launch.json'):
        try:
            data = json.loads(launch.read_text())
            if Path(data.get('log_dir', '')).resolve() == checkpoint.parent.resolve():
                candidate = launch.parent/'source-sha256.json'
                if candidate.is_file():
                    manifests.append(candidate)
        except (ValueError, OSError):
            continue
    require(bool(manifests), 'training source manifest not found')
    expected = json.loads(manifests[0].read_text())
    frozen = directory/'frozen-runtime/mjlab_training/src/ct_mjlab'
    snapshot = checkpoint.parent/'snapshot/ct_mjlab'
    for relative, sha in expected.items():
        require(digest(frozen/relative) == sha, f'frozen source: {relative}')
        require(digest(snapshot/relative) == sha, f'training snapshot: {relative}')
    physical = json.loads((manifests[0].parent/'physical-files-sha256.json').read_text())
    for relative, sha in physical.items():
        require(digest(repository/relative) == sha, f'physical resource: {relative}')
    require(len(expected) == report['source_files_verified'], 'source count')
    require(len(physical) == report['physical_files_verified'], 'physical count')
    return dict(assessment_source=matching[0], training_manifest=str(manifests[0]),
        frozen_and_snapshot_files=len(expected), physical_files=len(physical))


def check(directory, repository):
    report = json.loads((directory/'result.json').read_text())
    provenance = source_check(directory, report, repository)
    require(digest(directory/'trace.npz') == report['trace_sha256'], 'trace SHA')
    require(digest(directory/'trajectory.npz') == report['trajectory_sha256'], 'trajectory SHA')
    with np.load(directory/'trace.npz') as saved:
        values, control = saved['values'], saved['control']
        fields, control_fields = list(saved['fields']), list(saved['control_fields'])
        weight = saved['weight_N'].copy()
        initial_qpos, initial_qvel = saved['initial_qpos'].copy(), saved['initial_qvel'].copy()
    column = lambda name: values[:, :, fields.index(name)]
    group = lambda names: values[:, :, [fields.index(name) for name in names]]
    cgroup = lambda prefix, joints=range(6): control[:, :, [control_fields.index(f'{prefix}_{j}') for j in joints]]
    dt, policy_dt, rate = report['physics_dt_s'], report['policy_dt_s'], report['rate_rad_s']
    count, environments = values.shape[:2]
    require(dt == .002 and policy_dt == .01, '500Hz/100Hz periods')
    require(environments == report['num_envs'] and count == round(report['seconds']/dt), 'trace dimensions')
    require(len(control) == 500, 'first-second control sample count')
    require(np.isfinite(values).all() and np.isfinite(control).all(), 'finite traces')
    require((weight > 0.).all() and np.isfinite(weight).all(), 'weight')
    require(report['parameter_updates'] == 0 and report['actor_unchanged']
        and report['strict_native_load'], 'frozen actor / strict load receipts')
    require(report['raw_control_change_max'] == 0., 'wrapper leaves raw/control buffers unchanged receipt')
    live, hold = column('live') > .5, column('hold') > .5
    limited, finished = column('limited') > .5, column('finished') > .5
    has_stood = column('has_stood') > .5
    require(not (np.diff(live.astype(int), axis=0) > 0).any(), 'first-episode mask never revives')
    require(hold[:100].all() and not hold[100:500][live[100:500]].any(), 'exact 0.2s hold boundary')
    target = group([f'target_{j}' for j in JOINTS])
    desired = group([f'desired_{j}' for j in JOINTS])
    reference = np.broadcast_to(np.asarray(report['controller_meta']['reference'])[:, LEGS],
        (environments,4))
    low, high = reference-3.14, reference+3.14
    require((target >= low[None]-2e-6).all() and (target <= high[None]+2e-6).all(), 'original target bounds')
    meta = report['controller_meta']
    q = cgroup('q', JOINTS)
    qd, raw, processed, delayed = [cgroup(prefix) for prefix in ('qd', 'raw', 'control', 'delayed')]
    nominal, command, actual = cgroup('nominal', JOINTS), cgroup('command'), cgroup('actual')
    require(meta['leg_stiffness'] == 60. and meta['leg_damping'] == 2., 'original leg gains')
    require(not np.asarray(meta['leg_delay_steps']).any() and not np.asarray(meta['wheel_delay_steps']).any(), 'nominal zero-delay protocol')
    near(delayed, processed, 'FIFO zero-delay samples')
    near(nominal, np.broadcast_to(nominal[:1], nominal.shape), 'nominal angles restored before observation')
    original_target = np.clip(nominal+meta['leg_position_scale']*delayed[:, :, LEGS], low, high)
    near(desired[:500], original_target, 'original delayed/clamped target')
    alpha_state = np.zeros((environments, 4), dtype=np.float32)
    ema_error = 0.
    for start in range(0, 500, 5):
        active = live[start]
        block = slice(start, start+5)
        near(raw[block][:, active], np.broadcast_to(raw[start:start+1, active], raw[block][:, active].shape), 'raw held over five physics substeps')
        alpha_state = .2*alpha_state+.8*raw[start, :, LEGS].T
        alpha_state[hold[start]] = 0.
        ema_error = max(ema_error, near(processed[start, active][:, LEGS], alpha_state[active], 'one EMA update per policy tick'))
        near(processed[block][:, active], np.broadcast_to(processed[start:start+1, active], processed[block][:, active].shape), 'control held over five substeps')
    near(processed[:, :, WHEELS], raw[:, :, WHEELS], 'wheel actions bypass EMA')
    kp, kd, strength = [np.asarray(meta[k]) for k in ('kp_scale', 'kd_scale', 'strength')]
    demands = np.zeros(command.shape, dtype=np.float64)
    demands[:, :, LEGS] = (meta['leg_stiffness']*kp[:, LEGS]*(target[:500]-q)
        - meta['leg_damping']*kd[:, LEGS]*qd[:, :, LEGS])*strength[:, LEGS]
    wheel_target = np.clip(meta['wheel_velocity_scale']*delayed[:, :, WHEELS],
        -meta['max_wheel_velocity'], meta['max_wheel_velocity'])
    demands[:, :, WHEELS] = kd[:, WHEELS]*(meta['wheel_stiffness']*(wheel_target-qd[:, :, WHEELS])
        -meta['wheel_damping']*qd[:, :, WHEELS])*strength[:, WHEELS]
    expected_command = np.clip(demands, -np.asarray([40.,40.,3.9,40.,40.,3.9]), np.asarray([40.,40.,3.9,40.,40.,3.9]))
    expected_command[hold[:500]] = 0.
    pd_error = near(command, expected_command, 'six-motor original PD / strength / effort limits', atol=1e-4)
    actual_error = near(actual[:, :, LEGS], command[:, :, LEGS], 'actual leg force equals original PD command')
    require((command[hold[:500]] == 0.).all() and (actual[hold[:500]] == 0.).all(), 'waiting all six actual motors zero')
    starts = live & ~hold & ~np.concatenate((np.zeros((1,environments), dtype=bool), (live & ~hold)[:-1]), axis=0)
    require(starts.sum(0).tolist() == [1]*environments, 'one first-episode release per environment')
    if rate == 0.:
        require(not limited[live].any() and not finished[live].any(), 'baseline has no limiter states')
        near(target[live], desired[live], 'baseline original target')
    else:
        expected_limited = live & ~hold & ~np.concatenate((np.zeros((1,environments), dtype=bool), finished[:-1]), axis=0)
        require(np.array_equal(limited[live], expected_limited[live]), 'limiter active until caught up after stand')
        first_indices = np.argmax(starts, axis=0)
        for i, first in enumerate(first_indices):
            require(first < 500, 'first activation in control window')
            near(target[first,i], np.clip(q[first,i], low[i], high[i]), 'measured-angle latch with original clamp', atol=2e-6)
        advancing = limited[1:] & live[1:] & ~starts[1:]
        expected_target = np.clip(target[:-1]+np.clip(desired[1:]-target[:-1], -rate*dt, rate*dt), low, high)
        near(target[1:][advancing], expected_target[advancing], 'per-2ms target recurrence', atol=3e-6)
        delta = np.abs(target[1:]-target[:-1])
        require((delta[advancing] <= rate*dt+3e-6).all(), '500Hz slew bound')
        require(not (np.diff(finished.astype(int), axis=0)[live[1:] & live[:-1]] < 0).any(), 'finish latch monotonic within first episode')
        finishing = finished & ~np.concatenate((np.zeros((1,environments), dtype=bool), finished[:-1]), axis=0) & live
        require(has_stood[finishing].all(), 'bypass only after first stand')
        near(target[finishing], desired[finishing], 'bypass handoff caught original target', atol=2e-6)
        near(target[live & ~limited], desired[live & ~limited], 'bypass returns original target')
    support = group(['left_support_N', 'right_support_N'])
    good = ((support >= np.maximum(.1*weight,5.)[None,:,None]).all(-1)
        & (support.sum(-1) >= .8*weight) & (support.sum(-1) <= 1.2*weight)
        & (column('base_N') <= 1.) & (column('max_roll_pitch_rad') <= np.deg2rad(35.))
        & (column('roll_pitch_speed_rad_s') <= .5) & live & np.isfinite(values).all(-1))
    rows, criterion_notes = [], []
    for i, recorded in enumerate(report['rows']):
        require(recorded['env'] == i, 'ordered environment rows')
        last_bad = np.flatnonzero(~good[:,i])
        tail = count-(last_bad[-1]+1) if len(last_bad) else count
        full = bool(live[:,i].all() and recorded['end_reason'] != 'failure'
            and abs(recorded['end_s']-report['seconds']) < 1e-6)
        balanced = full and tail >= round(1./dt)
        stood = recorded['first_stood_s'] is not None
        require(recorded['stood_and_survived_40s'] == (stood and full), 'stood/full episode report')
        near(recorded['final_balance_hold_s'], tail*dt, 'final continuous v3.1 hold duration')
        if bool(recorded['final_balanced_v31']) != balanced:
            require(not stood and not recorded['final_balanced_v31'] and balanced,
                'v3.1 report mismatch beyond historical extra has_stood gate')
            criterion_notes.append(f'env{i}: report added has_stood; independent pure v3.1 takes precedence')
        first = int(np.flatnonzero(starts[:,i])[0])
        first_error = np.rad2deg(target[first,i]-q[first,i])
        near(recorded['first_active_target_error_deg'], first_error, 'saved first target error')
        mask = live[:500,i] & ~hold[:500,i]
        torque = actual[:,i][:,LEGS]
        saturation = np.abs(torque) >= 39.999
        failure = recorded['first_failure'] if recorded['end_reason']=='failure' else None
        rows.append(dict(env=i, first_stood_s=recorded['first_stood_s'], full_episode=full,
            stood_and_survived_40s=bool(stood and full), final_balanced_v31=bool(balanced),
            stood_and_balanced=bool(stood and balanced), final_balance_hold_s=tail*dt,
            first_active_target_error_deg=first_error.tolist(), first_active_leg_torque_Nm=torque[first].tolist(),
            first_active_peak_abs_Nm=float(np.abs(torque[first]).max()),
            first_second_peak_abs_Nm=float(np.abs(torque[mask]).max()),
            any_leg_saturation_duration_s=float((saturation.any(-1)&mask).sum()*dt),
            per_leg_saturation_duration_s=(saturation[mask].sum(0)*dt).tolist(),
            end_s=recorded['end_s'], end_reason=recorded['end_reason'],
            airborne_failure=bool(failure and failure['airborne_ticks']>=20),
            inverted_failure=bool(failure and failure['inverted'])))
    with np.load(directory/'trajectory.npz') as trajectory:
        require(np.array_equal(trajectory['qpos'][0], initial_qpos), 'trajectory initial qpos')
        near(trajectory['time_s'], np.arange(round(report['seconds']/policy_dt)+1)*policy_dt, 'trajectory time')
    summary = dict(directory=str(directory), rate_rad_s=rate, num_envs=environments, seconds=report['seconds'],
        seed=report['seed'], checkpoint_sha256=report['checkpoint_sha256'], checks_passed=True,
        provenance=provenance, criterion_notes=criterion_notes, rows=rows,
        max_ema_error=ema_error, max_pd_error_Nm=pd_error, actual_leg_error_Nm=actual_error,
        observation_check_scope='Saved raw samples, EMA/FIFO and restored nominal checked; actor output before process_actions was not separately sampled.',
        delay_check_scope='Saved nominal play protocol has zero delay; nonzero delay and synthetic local-reset invariants require separate controller checks.')
    return summary, (initial_qpos, initial_qvel, weight, report['controller_meta'])


def compare(summaries, inputs):
    comparisons = []
    for candidate in summaries:
        if candidate['rate_rad_s'] == 0.:
            continue
        baselines = [r for r in summaries if r['rate_rad_s']==0. and r['num_envs']==candidate['num_envs']
            and r['seed']==candidate['seed'] and r['seconds']==candidate['seconds']]
        if not baselines:
            comparisons.append(dict(candidate=candidate['directory'], status='pending_matching_baseline'))
            continue
        require(len(baselines)==1, 'unique matching baseline')
        baseline = baselines[0]
        require(candidate['checkpoint_sha256']==baseline['checkpoint_sha256'], 'paired model SHA')
        for name, a, b in zip(('initial_qpos','initial_qvel','weight_N'),
                inputs[candidate['directory']][:3], inputs[baseline['directory']][:3]):
            require(np.array_equal(a,b), f'paired exact {name}')
        require(inputs[candidate['directory']][3] == inputs[baseline['directory']][3], 'paired original controller metadata')
        paired = dict(candidate=candidate['directory'], baseline=baseline['directory'],
            rate_rad_s=candidate['rate_rad_s'], num_envs=candidate['num_envs'], initial_states_exact=True)
        lost_any = False
        for criterion in ('stood_and_survived_40s','final_balanced_v31','stood_and_balanced'):
            old = {r['env'] for r in baseline['rows'] if r[criterion]}
            new = {r['env'] for r in candidate['rows'] if r[criterion]}
            lost, gained = sorted(old-new), sorted(new-old)
            paired[criterion] = dict(baseline_ids=sorted(old), candidate_ids=sorted(new), lost_ids=lost, gained_ids=gained)
            lost_any |= bool(lost)
        for criterion in ('airborne_failure','inverted_failure'):
            paired[criterion] = dict(baseline_ids=[r['env'] for r in baseline['rows'] if r[criterion]],
                candidate_ids=[r['env'] for r in candidate['rows'] if r[criterion]])
        paired['per_environment'] = [dict(env=old['env'], baseline={key:old[key] for key in (
                'first_stood_s','end_s','first_active_peak_abs_Nm','first_second_peak_abs_Nm',
                'any_leg_saturation_duration_s')}, candidate={key:new[key] for key in (
                'first_stood_s','end_s','first_active_peak_abs_Nm','first_second_peak_abs_Nm',
                'any_leg_saturation_duration_s')})
            for old,new in zip(baseline['rows'],candidate['rows'])]
        paired['status'] = 'regressed_rejected' if lost_any else 'no_observed_old_success_loss'
        comparisons.append(paired)
    return comparisons


def main(args):
    require(not args.output.exists(), 'output must be a new file')
    repository = Path(__file__).resolve().parents[3]
    directories = sorted({p.parent for p in args.root.rglob('result.json') if (p.parent/'trace.npz').is_file()})
    summaries, inputs, errors = [], {}, []
    for directory in directories:
        try:
            summary, initial = check(directory, repository)
            summaries.append(summary)
            inputs[str(directory)] = initial
        except (AssertionError, KeyError, ValueError, OSError) as error:
            errors.append(dict(directory=str(directory), error=str(error)))
    try:
        comparisons = compare(summaries, inputs)
    except (AssertionError, KeyError, ValueError) as error:
        errors.append(dict(comparison_error=str(error)))
        comparisons = []
    result = dict(checks_passed=not errors, completed_measurements=len(directories), runs=summaries,
        comparisons=comparisons, errors=errors, checker_sha256=digest(Path(__file__)),
        scope='Independent saved-trace arithmetic and paired first-episode comparison; no simulation, training or full-distribution claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(checks_passed=not errors, completed_measurements=len(directories), errors=errors,
        comparisons=comparisons), indent=2), flush=True)
    return bool(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
