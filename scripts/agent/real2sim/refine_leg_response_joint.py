"""Finite multi-trajectory diagnostic parameter search; never trains a policy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np


SEED = 20261104
ARMATURE = .015795
DIMENSIONS = 13


def logarithm(value, low, high):
    return np.log(np.clip(value, low, high) / low) / np.log(high / low)


def values(profile, key, count, default):
    result = np.asarray(profile.get(key, default), dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f'{key} must contain {count} finite values')
    return result


def encode(profile, side, family):
    """Map a seed into the bounded search space; physical offsets/gains are absent."""
    reference = values(profile, 'spring_reference_force_n', 2, [150.] * 2)
    stiffness = values(profile, 'spring_stiffness_n_m', 2, [0.] * 2)
    spring_damping = values(profile, 'spring_damping_ns_m', 2, [50.] * 2)
    prefix = 'motor' if family == 'independent' else 'coordinate'
    friction_key = 'friction_nm' if family == 'independent' else 'coordinate_friction_nm'
    damping_key = 'damping_nms_rad' if family == 'independent' else 'coordinate_damping_nms_rad'
    friction = values(profile, friction_key, 4, [2.] * 4)
    damping = values(profile, damping_key, 4, [.2] * 4)
    positive = values(profile, prefix + '_dynamic_positive_nm', 4, friction)
    negative = values(profile, prefix + '_dynamic_negative_nm', 4, friction)
    velocity = values(profile, prefix + '_velocity_scale_rad_s', 4, [.02] * 4)
    result = [(reference[side] - 50.) / 200., stiffness[side] / 3500., spring_damping[side] / 500.]
    for index in (2 * side, 2 * side + 1):
        static = max(friction[index], 1e-12)
        result.extend([logarithm(static, .2, 12.), (positive[index] / static - .1) / .9,
                       (negative[index] / static - .1) / .9,
                       logarithm(velocity[index], .002, .1), logarithm(damping[index], .01, 8.)])
    return np.clip(result, 0., 1.)


def decode(left, right, family):
    profile = dict(armature_kg_m2=[ARMATURE] * 4, spring_reference_force_n=[],
                   spring_stiffness_n_m=[], spring_force_limit_n=[600.] * 2,
                   spring_damping_ns_m=[])
    prefix = 'motor' if family == 'independent' else 'coordinate'
    friction_key = 'friction_nm' if family == 'independent' else 'coordinate_friction_nm'
    damping_key = 'damping_nms_rad' if family == 'independent' else 'coordinate_damping_nms_rad'
    for key in (friction_key, damping_key, prefix + '_dynamic_positive_nm',
                prefix + '_dynamic_negative_nm', prefix + '_velocity_scale_rad_s'):
        profile[key] = []
    for vector in (left, right):
        x = np.asarray(vector, dtype=float)
        if x.shape != (DIMENSIONS,) or not np.isfinite(x).all() or np.any((x < 0.) | (x > 1.)):
            raise ValueError('Each side must have 13 normalized parameters in [0,1]')
        profile['spring_reference_force_n'].append(float(50. + 200. * x[0]))
        profile['spring_stiffness_n_m'].append(float(3500. * x[1]))
        profile['spring_damping_ns_m'].append(float(500. * x[2]))
        for offset in (3, 8):
            static = .2 * 60. ** x[offset]
            profile[friction_key].append(float(static))
            profile[prefix + '_dynamic_positive_nm'].append(float(static * (.1 + .9 * x[offset + 1])))
            profile[prefix + '_dynamic_negative_nm'].append(float(static * (.1 + .9 * x[offset + 2])))
            profile[prefix + '_velocity_scale_rad_s'].append(float(.002 * 50. ** x[offset + 3]))
            profile[damping_key].append(float(.01 * 800. ** x[offset + 4]))
    if family == 'coordinate':
        # 同向/反向坐标承担可变阻力，原四电机仅保留固定小阻力。
        profile.update(friction_nm=[.02] * 4, damping_nms_rad=[.2] * 4)
    return profile


def score(trials, side):
    """Equal weight per trajectory, input branch and joint; peak/end are worst case."""
    section = slice(2 * side, 2 * side + 2)
    squared_rmse = []
    peaks = []
    terminals = []
    for trial in trials:
        if not trial.get('initial_static_equilibrium_feasible',True):return np.inf
        if trial['feedback_blocked']:
            return np.inf
        loss = np.asarray(trial['minimum_discrete_dissipation_j'], dtype=float)
        if loss.shape != (2,) or not np.isfinite(loss).all() or loss[side] < -1e-7:
            return np.inf
        for prefix in ('', 'feedback_'):
            rmse_key = 'all_motor_rmse_deg' if not prefix else 'feedback_motor_rmse_deg'
            for key in (rmse_key, prefix + 'peak_error_deg', prefix + 'final_error_deg'):
                metric = np.asarray(trial[key], dtype=float)
                if metric.shape != (4,) or not np.isfinite(metric).all():
                    return np.inf
            squared_rmse.extend(np.asarray(trial[rmse_key])[section] ** 2)
            peaks.extend(np.abs(np.asarray(trial[prefix + 'peak_error_deg'])[section]))
            terminals.extend(np.abs(np.asarray(trial[prefix + 'final_error_deg'])[section]))
    if not squared_rmse:
        raise ValueError('At least one trajectory is required')
    return float(np.sqrt(np.mean(squared_rmse)) + .05 * max(peaks) + .1 * max(terminals))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def finite_scores(scores):
    return [float(value) if np.isfinite(value) else None for value in scores]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_info(reference, rc_inputs):
    report = json.loads((reference / 'report.json').read_text())
    if report['constraint_check'] != 'stiff' or report['dt_s'] != .001:
        raise ValueError('Only the stiff 1ms diagnostic references are supported')
    tree = ET.parse(reference / 'fixed-base-model.xml').getroot()
    base = tree.find("./worldbody/body[@name='base_link']")
    if base is None or base.find('joint') is not None or base.find('freejoint') is not None:
        raise ValueError('Independent side composition requires a fixed robot base')
    if tree.find('.//freejoint') is not None or tree.find(".//joint[@type='free']") is not None:
        raise ValueError('Free-base models cannot use independent side composition')
    with np.load(reference / 'replay.npz') as saved:
        duration = float(saved['time_s'][-1])
    with np.load(rc_inputs) as rc:
        if str(rc['stream_sha256']) != report['stream_sha256']:
            raise ValueError(f'RC input stream mismatch: {reference}')
    return dict(reference=str(reference.resolve()), rc_inputs=str(rc_inputs.resolve()),
                duration_s=duration, stream_sha256=report['stream_sha256'],
                report_sha256=sha256(reference / 'report.json'), rc_inputs_sha256=sha256(rc_inputs),
                model_sha256=sha256(reference / 'fixed-base-model.xml'))


def run_candidates(parameters, output, references, friction_law='coulomb'):
    """GPU subprocesses run sequentially so they never contend for the same device."""
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'parameters.json'
    write_json(source, parameters)
    reports = []
    for index, reference in enumerate(references):
        directory = output / f'trajectory-{index:02d}'
        command = [sys.executable, '-m', 'scripts.agent.real2sim.fit_leg_response_gpu',
                   '--reference', reference['reference'], '--parameters', str(source),
                   '--fit-until-s', str(reference['duration_s']), '--feedback-validation',
                   '--rc-inputs', reference['rc_inputs'], '--output', str(directory)]
        if friction_law != 'coulomb':command.extend(['--friction-law',friction_law])
        if reference.get('memory_history'):command.extend(['--memory-history',reference['memory_history']])
        write_json(output / f'trajectory-{index:02d}-command.json', command)
        with (output / f'trajectory-{index:02d}.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        report_path = directory / 'report.json'
        report = json.loads(report_path.read_text())
        if (report['physics_backend'] != 'mujoco_warp_gpu'
                or report['objective_branches'] != ['recorded_torque', 'recorded_target_feedback']
                or len(report['trials']) != len(parameters)):
            raise ValueError(f'Incomplete GPU dual-branch report: {report_path}')
        if report['fit_window_from_start_s'] != [0., reference['duration_s']]:
            raise ValueError(f'GPU report did not score the entire input: {report_path}')
        for candidate, trial in zip(parameters, report['trials']):
            if trial['parameters'] != candidate:
                raise ValueError(f'GPU report candidate order mismatch: {report_path}')
        reports.append(report)
        print(json.dumps(dict(completed_trajectory=index, report=str(report_path))), flush=True)
    scores = np.asarray([[score([report['trials'][index] for report in reports], side)
                          for side in range(2)] for index in range(len(parameters))])
    report_paths = [str((output / f'trajectory-{index:02d}' / 'report.json').resolve())
                    for index in range(len(references))]
    write_json(output / 'scores.json', dict(report_paths=report_paths,
               candidate_side_scores_deg=[finite_scores(row) for row in scores],
               rejected_candidate_sides=np.argwhere(~np.isfinite(scores)).tolist()))
    return scores, report_paths


def selfcheck():
    rng = np.random.default_rng(SEED)
    for family in ('independent', 'coordinate'):
        for _ in range(30):
            vectors = rng.uniform(0., 1., (2, DIMENSIONS))
            profile = decode(*vectors, family)
            for side in range(2):
                assert np.allclose(encode(profile, side, family), vectors[side], atol=1e-12)
            assert profile['armature_kg_m2'] == [ARMATURE] * 4
            assert profile['spring_force_limit_n'] == [600.] * 2
    def trial(rmse):
        return dict(all_motor_rmse_deg=[rmse] * 4, feedback_motor_rmse_deg=[rmse] * 4,
                    peak_error_deg=[5.] * 4, feedback_peak_error_deg=[6.] * 4,
                    final_error_deg=[-2.] * 4, feedback_final_error_deg=[1.] * 4,
                    minimum_discrete_dissipation_j=[0., 0.], feedback_blocked=False)
    good = [trial(1.), trial(3.)]
    assert np.isclose(score(good, 0), np.sqrt(5.) + .05 * 6. + .1 * 2.)
    assert np.isclose(score(good, 1), score(good * 2, 1))
    good[0]['minimum_discrete_dissipation_j'][0] = -.01
    assert np.isinf(score(good, 0)) and np.isfinite(score(good, 1))
    good[0]['feedback_blocked'] = True
    assert np.isinf(score(good, 1))
    print(json.dumps(dict(selfcheck='passed', families=['independent', 'coordinate'],
                         dimensions_per_side=DIMENSIONS, encoding_trials=60, physics_steps=0)))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--references', type=Path, nargs='+', action='append')
    parser.add_argument('--rc-inputs', type=Path, nargs='+', action='append',
                        help='Optional paths in reference order; default is each reference/rc-inputs.npz')
    parser.add_argument('--seed-parameters', type=Path, help='JSON list of at most population parameter profiles')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--generations', type=int, default=4)
    parser.add_argument('--population', type=int, default=32)
    parser.add_argument('--family', choices=('independent', 'coordinate'), default='independent')
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if not args.references or args.seed_parameters is None or args.output is None:
        parser.error('--references, --seed-parameters and --output are required')
    if not 1 <= args.generations <= 6 or not 8 <= args.population <= 64:
        parser.error('Use 1..6 generations and 8..64 candidates')
    paths = [path for group in args.references for path in group]
    rc_paths = ([path for group in args.rc_inputs for path in group] if args.rc_inputs
                else [path / 'rc-inputs.npz' for path in paths])
    if len(paths) != len(rc_paths) or len(set(path.resolve() for path in paths)) != len(paths):
        parser.error('References must be unique and RC paths must have the same length')
    references = [reference_info(path, rc) for path, rc in zip(paths, rc_paths)]
    seeds = json.loads(args.seed_parameters.read_text())
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= args.population:
        parser.error('Seed JSON must be a nonempty list no larger than population')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output / Path(__file__).name)
    shutil.copy2(args.seed_parameters, args.output / 'seed-parameters.json')
    manifest = dict(scope='fixed-base no-contact development fit; unaccepted diagnostic parameters',
                    accepted=False, seed=SEED, family=args.family, references=references,
                    generations_limit=args.generations, population=args.population,
                    dimensions_per_side=DIMENSIONS,
                    objective='Equal-weight trajectory/branch/joint RMS +0.05 worst peak +0.1 worst terminal (degrees)',
                    rejection='Any blocked feedback branch or per-side discrete dissipation below -1e-7 J',
                    fixed_armature_kg_m2=ARMATURE, torque_gains_changed=False, joint_offsets_changed=False,
                    source_sha256=sha256(Path(__file__)), network_forwards=0, policy_optimizer_updates=0)
    write_json(args.output / 'manifest.json', manifest)
    rng = np.random.default_rng(SEED)
    populations = []
    for side in range(2):
        encoded = np.asarray([encode(seed, side, args.family) for seed in seeds])
        population = np.clip(rng.normal(encoded[np.arange(args.population) % len(encoded)], .18), 0., 1.)
        population[:len(encoded)] = encoded
        explore = min(max(2, args.population // 8), args.population - len(encoded))
        if explore:
            population[-explore:] = rng.uniform(0., 1., (explore, DIMENSIONS))
        populations.append(population)
    best_scores = np.full(2, np.inf)
    best_vectors = np.zeros((2, DIMENSIONS))
    best_sources = [None, None]
    history = []
    stagnation = 0
    for generation in range(args.generations):
        parameters = [decode(populations[0][index], populations[1][index], args.family)
                      for index in range(args.population)]
        scores, report_paths = run_candidates(parameters, args.output / f'generation-{generation:02d}', references)
        improved = False
        for side in range(2):
            order = np.argsort(scores[:, side])
            finite = order[np.isfinite(scores[order, side])]
            if len(finite) and scores[finite[0], side] < best_scores[side] - 1e-9:
                best_scores[side] = scores[finite[0], side]
                best_vectors[side] = populations[side][finite[0]].copy()
                best_sources[side] = dict(generation=generation, candidate_index=int(finite[0]),
                                          report_paths=report_paths)
                improved = True
            if len(finite):
                elite = populations[side][finite[:max(4, args.population // 6)]]
                center = elite.mean(axis=0)
                deviation = np.maximum(elite.std(axis=0), .055) * .9
                populations[side] = np.clip(rng.normal(center, deviation, (args.population, DIMENSIONS)), 0., 1.)
            else:
                populations[side] = rng.uniform(0., 1., (args.population, DIMENSIONS))
            if np.isfinite(best_scores[side]):
                populations[side][0] = best_vectors[side]
            explore = max(2, args.population // 8)
            populations[side][-explore:] = rng.uniform(0., 1., (explore, DIMENSIONS))
        stagnation = 0 if improved else stagnation + 1
        history.append(dict(generation=generation, report_paths=report_paths,
                            best_side_scores_deg=finite_scores(best_scores), sources=list(best_sources),
                            stagnant_generations=stagnation))
        write_json(args.output / 'progress.json', history)
        print(json.dumps(history[-1]), flush=True)
        if stagnation >= 2:
            break
    if not np.isfinite(best_scores).all():
        write_json(args.output / 'report.json', dict(manifest, status='no_valid_candidate', history=history))
        raise RuntimeError('No valid candidate for both sides; see saved rejection scores')
    selected = decode(*best_vectors, args.family)
    write_json(args.output / 'selected-parameters.json', selected)
    # 两侧分别选出的参数重新合并后，必须重新跑全部轨迹确认，不能拼接已有评分冒充复核。
    combined_scores, combined_reports = run_candidates([selected], args.output / 'combined-verification', references)
    difference = np.abs(combined_scores[0] - best_scores)
    composition_matches = bool(np.isfinite(combined_scores).all() and np.all(difference <= .02))
    write_json(args.output / 'report.json', dict(manifest, status='completed_unaccepted', history=history,
               selected_parameters=selected, side_composition=best_sources,
               source_side_scores_deg=finite_scores(best_scores),
               combined_side_scores_deg=finite_scores(combined_scores[0]),
               combined_verification_reports=combined_reports,
               composition_score_difference_deg=finite_scores(difference),
               composition_replay_matches_within_0_02_deg=composition_matches,
               stop_reason='two stagnant generations' if stagnation >= 2 else 'generation budget reached',
               limitations=['Parameters are effective fitted values, not independently measured component properties',
                            'All references used for selection are development data',
                            'Side composition applies only to this fixed-base, contact-disabled diagnostic']))
    if not composition_matches:
        raise RuntimeError('Combined single-candidate replay did not reproduce side selection within .02 degrees')


if __name__ == '__main__':
    main()
