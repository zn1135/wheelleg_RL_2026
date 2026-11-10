"""Bounded full-trajectory Dahl resistance search; no policy training or hardware access."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from scripts.agent.real2sim import refine_leg_response_joint as joint


DIMENSIONS = 15


def encode(profile, side):
    reference = joint.values(profile, 'spring_reference_force_n', 2, [150.] * 2)
    slope = joint.values(profile, 'spring_stiffness_n_m', 2, [0.] * 2)
    spring_damping = joint.values(profile, 'spring_damping_ns_m', 2, [50.] * 2)
    static = joint.values(profile, 'friction_nm', 4, [8.] * 4)
    positive = joint.values(profile, 'motor_dynamic_positive_nm', 4, static)
    negative = joint.values(profile, 'motor_dynamic_negative_nm', 4, static)
    velocity = joint.values(profile, 'motor_velocity_scale_rad_s', 4, [.02] * 4)
    stiffness = joint.values(profile, 'stiffness_nm_rad', 4, [250.] * 4)
    damping = joint.values(profile, 'damping_nms_rad', 4, [1.] * 4)
    vector = [(reference[side] - 50.) / 200., slope[side] / 5000., spring_damping[side] / 500.]
    for index in (2 * side, 2 * side + 1):
        capacity = max(static[index], 1e-12)
        vector.extend([joint.logarithm(capacity, .5, 30.),
                       (positive[index] / capacity - .02) / .98,
                       (negative[index] / capacity - .02) / .98,
                       joint.logarithm(velocity[index], .002, 1.),
                       joint.logarithm(stiffness[index], 20., 3000.),
                       joint.logarithm(damping[index], .01, 8.)])
    return np.clip(vector, 0., 1.)


def decode(left, right, force_limits=(600., 600.)):
    profile = dict(friction_nm=[], damping_nms_rad=[], stiffness_nm_rad=[],
                   motor_dynamic_positive_nm=[], motor_dynamic_negative_nm=[],
                   motor_velocity_scale_rad_s=[], armature_kg_m2=[joint.ARMATURE] * 4,
                   spring_reference_force_n=[], spring_stiffness_n_m=[],
                   spring_damping_ns_m=[], spring_force_limit_n=list(force_limits))
    for vector in (left, right):
        x = np.asarray(vector, dtype=float)
        if x.shape != (DIMENSIONS,) or not np.isfinite(x).all() or np.any((x < 0.) | (x > 1.)):
            raise ValueError('Each side requires 15 finite normalized parameters in [0,1]')
        profile['spring_reference_force_n'].append(float(50. + 200. * x[0]))
        profile['spring_stiffness_n_m'].append(float(5000. * x[1]))
        profile['spring_damping_ns_m'].append(float(500. * x[2]))
        for offset in (3, 9):
            static = .5 * 60. ** x[offset]
            stiffness = 20. * 150. ** x[offset + 4]
            # 离散储能的下限与K相关；B仍可在下限以上拟合，不调力矩倍率或零点。
            damping = max(.0005 * stiffness + .05, .01 * 800. ** x[offset + 5])
            profile['friction_nm'].append(float(static))
            profile['motor_dynamic_positive_nm'].append(float(static * (.02 + .98 * x[offset + 1])))
            profile['motor_dynamic_negative_nm'].append(float(static * (.02 + .98 * x[offset + 2])))
            profile['motor_velocity_scale_rad_s'].append(float(.002 * 500. ** x[offset + 3]))
            profile['stiffness_nm_rad'].append(float(stiffness))
            profile['damping_nms_rad'].append(float(damping))
    return profile


def run_candidates(parameters, output, references):
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'parameters.json'
    joint.write_json(source, parameters)
    reports = []; report_paths = []
    for index, reference in enumerate(references):
        directory = output / f'trajectory-{index:02d}'
        command = [sys.executable, '-m', 'scripts.agent.real2sim.fit_leg_response_gpu',
                   '--reference', reference['reference'], '--parameters', str(source),
                   '--fit-until-s', str(reference['duration_s']), '--feedback-validation',
                   '--rc-inputs', reference['rc_inputs'], '--friction-law', 'elastic', '--output', str(directory)]
        joint.write_json(output / f'trajectory-{index:02d}-command.json', dict(argv=command,
                         inputs_sha256={str(source.resolve()): joint.sha256(source),
                                        reference['rc_inputs']: reference['rc_inputs_sha256']}))
        with (output / f'trajectory-{index:02d}.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        path = directory / 'report.json'
        report = json.loads(path.read_text())
        if (report['physics_backend'] != 'mujoco_warp_gpu' or report['friction_law'] != 'elastic'
                or report['objective_branches'] != ['recorded_torque', 'recorded_target_feedback']
                or len(report['trials']) != len(parameters)
                or report['fit_window_from_start_s'] != [0., reference['duration_s']]):
            raise ValueError(f'Incomplete full-trajectory dual-branch Dahl report: {path}')
        for candidate, trial in zip(parameters, report['trials']):
            if trial['parameters'] != candidate:
                raise ValueError(f'Candidate order mismatch: {path}')
        reports.append(report); report_paths.append(str(path.resolve()))
        print(json.dumps(dict(completed_trajectory=index, report=str(path))), flush=True)
    scores = np.asarray([[joint.score([report['trials'][index] for report in reports], side)
                          for side in range(2)] for index in range(len(parameters))])
    joint.write_json(output / 'scores.json', dict(report_paths=report_paths,
                     candidate_side_scores_deg=[joint.finite_scores(row) for row in scores],
                     rejected_candidate_sides=np.argwhere(~np.isfinite(scores)).tolist()))
    return scores, report_paths


def selfcheck():
    rng = np.random.default_rng(joint.SEED)
    for _ in range(60):
        profile = decode(*rng.uniform(0., 1., (2, DIMENSIONS)))
        encoded = [encode(profile, side) for side in range(2)]
        reconstructed = decode(*encoded)
        for key in profile:
            assert np.allclose(profile[key], reconstructed[key], rtol=1e-12, atol=1e-12), key
        static = np.asarray(profile['friction_nm'])
        assert np.all(np.asarray(profile['motor_dynamic_positive_nm']) <= static)
        assert np.all(np.asarray(profile['motor_dynamic_negative_nm']) <= static)
        assert np.all(np.asarray(profile['damping_nms_rad']) >= .0005 * np.asarray(profile['stiffness_nm_rad']) + .05)
        assert profile['armature_kg_m2'] == [joint.ARMATURE] * 4
    joint.selfcheck()
    print(json.dumps(dict(elastic_selfcheck='passed', dimensions_per_side=DIMENSIONS,
                         semantic_encoding_trials=60, physics_steps=0)))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--references', type=Path, nargs='+', action='append')
    parser.add_argument('--rc-inputs', type=Path, nargs='+', action='append')
    parser.add_argument('--seed-parameters', type=Path, nargs='+', action='append')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--population', type=int, default=32)
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(); return
    if not args.references or not args.seed_parameters or args.output is None:
        parser.error('--references, --seed-parameters and --output are required')
    if not 1 <= args.generations <= 4 or not 8 <= args.population <= 32:
        parser.error('Use 1..4 generations and 8..32 candidates')
    paths = [path for group in args.references for path in group]
    rc_paths = ([path for group in args.rc_inputs for path in group] if args.rc_inputs
                else [path / 'rc-inputs.npz' for path in paths])
    if len(paths) != len(rc_paths) or len(set(path.resolve() for path in paths)) != len(paths):
        parser.error('References must be unique and have one corresponding RC cache each')
    references = [joint.reference_info(path, rc) for path, rc in zip(paths, rc_paths)]
    seed_paths = [path for group in args.seed_parameters for path in group]
    seeds = []
    for path in seed_paths:
        saved = json.loads(path.read_text())
        seeds.extend(saved if isinstance(saved, list) else [saved])
    if not seeds or len(seeds) > args.population:
        parser.error('Combined seed files must contain 1..population parameter profiles')
    force_limits = np.maximum(600., np.max([joint.values(seed, 'spring_force_limit_n', 2, [600.] * 2)
                                          for seed in seeds], axis=0)).tolist()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output / Path(__file__).name)
    for index, path in enumerate(seed_paths):
        shutil.copy2(path, args.output / f'seed-parameters-{index:02d}.json')
    manifest = dict(scope='Fixed-base no-contact full-trajectory Dahl development fit', accepted=False,
                    seed=joint.SEED, family='independent_motor_elastic', references=references,
                    dimensions_per_side=DIMENSIONS, generations_limit=args.generations, population=args.population,
                    objective='Equal-weight trajectory/branch/joint RMS +0.05 worst peak +0.1 worst terminal (degrees)',
                    rejection='Feedback blocked or per-side minimum discrete dissipation below -1e-7 J',
                    initial_preload='Backend uses only initial pose/velocity and first recorded request',
                    fixed_armature_kg_m2=joint.ARMATURE, torque_gains_changed=False, joint_offsets_changed=False,
                    spring_force_limit_n=force_limits, source_sha256=joint.sha256(Path(__file__)),
                    network_forwards=0, policy_optimizer_updates=0)
    joint.write_json(args.output / 'manifest.json', manifest)
    rng = np.random.default_rng(joint.SEED)
    populations = []
    for side in range(2):
        encoded = np.asarray([encode(seed, side) for seed in seeds])
        population = np.clip(rng.normal(encoded[np.arange(args.population) % len(encoded)], .18), 0., 1.)
        population[:len(encoded)] = encoded
        explore = min(max(2, args.population // 8), args.population - len(encoded))
        if explore:
            population[-explore:] = rng.uniform(0., 1., (explore, DIMENSIONS))
        populations.append(population)
    best_scores = np.full(2, np.inf); best_vectors = np.zeros((2, DIMENSIONS))
    best_sources = [None, None]; history = []; stagnation = 0
    for generation in range(args.generations):
        parameters = [decode(populations[0][index], populations[1][index], force_limits)
                      for index in range(args.population)]
        scores, report_paths = run_candidates(parameters, args.output / f'generation-{generation:02d}', references)
        improved = False
        for side in range(2):
            order = np.argsort(scores[:, side]); finite = order[np.isfinite(scores[order, side])]
            if len(finite) and scores[finite[0], side] < best_scores[side] - 1e-9:
                best_scores[side] = scores[finite[0], side]
                best_vectors[side] = populations[side][finite[0]].copy()
                best_sources[side] = dict(generation=generation, candidate_index=int(finite[0]), report_paths=report_paths)
                improved = True
            if len(finite):
                elite = populations[side][finite[:max(4, args.population // 6)]]
                center = elite.mean(axis=0); deviation = np.maximum(elite.std(axis=0), .055) * .9
                populations[side] = np.clip(rng.normal(center, deviation, (args.population, DIMENSIONS)), 0., 1.)
            else:
                populations[side] = rng.uniform(0., 1., (args.population, DIMENSIONS))
            if np.isfinite(best_scores[side]):
                populations[side][0] = best_vectors[side]
            explore = max(2, args.population // 8)
            populations[side][-explore:] = rng.uniform(0., 1., (explore, DIMENSIONS))
        stagnation = 0 if improved else stagnation + 1
        history.append(dict(generation=generation, report_paths=report_paths,
                            best_side_scores_deg=joint.finite_scores(best_scores), sources=list(best_sources),
                            stagnant_generations=stagnation))
        joint.write_json(args.output / 'progress.json', history)
        print(json.dumps(history[-1]), flush=True)
        if stagnation >= 2:
            break
    if not np.isfinite(best_scores).all():
        joint.write_json(args.output / 'report.json', dict(manifest, status='no_valid_candidate', history=history))
        raise RuntimeError('No passive unblocked candidate for both sides; see saved rejection scores')
    selected = decode(*best_vectors, force_limits)
    joint.write_json(args.output / 'selected-parameters.json', selected)
    combined, report_paths = run_candidates([selected], args.output / 'combined-verification', references)
    difference = np.abs(combined[0] - best_scores)
    matches = bool(np.isfinite(combined).all() and np.all(difference <= .02))
    joint.write_json(args.output / 'report.json', dict(manifest, status='completed_unaccepted', history=history,
                     selected_parameters=selected, side_composition=best_sources,
                     source_side_scores_deg=joint.finite_scores(best_scores),
                     combined_side_scores_deg=joint.finite_scores(combined[0]), combined_verification_reports=report_paths,
                     composition_score_difference_deg=joint.finite_scores(difference),
                     composition_replay_matches_within_0_02_deg=matches,
                     stop_reason='two stagnant generations' if stagnation >= 2 else 'generation budget reached',
                     limitations=['Effective fitted dynamics; not separately measured friction or spring properties',
                                  'All search references are development data, not independent hardware acceptance',
                                  'Leg composition applies only to this fixed-base no-contact model']))
    if not matches:
        raise RuntimeError('Combined single-candidate replay differs from side selection by more than .02 degrees')


if __name__ == '__main__':
    main()
