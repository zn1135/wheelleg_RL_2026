"""Finite multi-recording GPU search of motor/seal memory with causal initialization."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.agent.real2sim import refine_leg_response_joint as joint


DIM = 17


def encode(p, side):
    x = [(p['spring_reference_force_n'][side]-80.)/140., p['spring_stiffness_n_m'][side]/5000.,
         joint.logarithm(p['spring_damping_ns_m'][side], 20., 250.),
         joint.logarithm(p['spring_seal_static_n'][side], 100., 400.),
         joint.logarithm(p['spring_seal_stiffness_n_m'][side], 5000., 150000.),
         p['spring_seal_positive_n'][side]/p['spring_seal_static_n'][side],
         p['spring_seal_negative_n'][side]/p['spring_seal_static_n'][side]]
    for i in (2*side, 2*side+1):
        x.extend([joint.logarithm(p['friction_nm'][i], .2, 8.),
                  joint.logarithm(p['stiffness_nm_rad'][i], 30., 600.),
                  p['motor_dynamic_positive_nm'][i]/p['friction_nm'][i],
                  p['motor_dynamic_negative_nm'][i]/p['friction_nm'][i],
                  joint.logarithm(p['damping_nms_rad'][i], .1, 3.)])
    return np.clip(x, 0., 1.)


def decode(left, right):
    p = dict(friction_nm=[], damping_nms_rad=[], stiffness_nm_rad=[], motor_dynamic_positive_nm=[],
             motor_dynamic_negative_nm=[], motor_velocity_scale_rad_s=[.02]*4, armature_kg_m2=[.015795]*4,
             spring_reference_force_n=[], spring_stiffness_n_m=[], spring_damping_ns_m=[], spring_force_limit_n=[600.]*2,
             spring_seal_static_n=[], spring_seal_stiffness_n_m=[], spring_seal_positive_n=[], spring_seal_negative_n=[],
             spring_seal_velocity_scale_m_s=[.001]*2, preload_rule='minimum_energy_shared')
    for x in (left, right):
        p['spring_reference_force_n'].append(float(80.+140.*x[0])); p['spring_stiffness_n_m'].append(float(5000.*x[1]))
        p['spring_damping_ns_m'].append(float(20.*12.5**x[2]))
        capacity = float(100.*4.**x[3]); p['spring_seal_static_n'].append(capacity)
        p['spring_seal_stiffness_n_m'].append(float(5000.*30.**x[4]))
        p['spring_seal_positive_n'].append(capacity*max(float(x[5]), .01))
        p['spring_seal_negative_n'].append(capacity*max(float(x[6]), .01))
        for offset in (7, 12):
            f = float(.2*40.**x[offset]); k = float(30.*20.**x[offset+1])
            p['friction_nm'].append(f); p['stiffness_nm_rad'].append(k)
            p['motor_dynamic_positive_nm'].append(f*max(float(x[offset+2]), .05))
            p['motor_dynamic_negative_nm'].append(f*max(float(x[offset+3]), .05))
            # 离散储能仍由实际回放检查；此下界限制显式弹性状态的数值振荡。
            p['damping_nms_rad'].append(max(float(.1*30.**x[offset+4]), .0005*k+.05))
    return p


def evaluate(parameters, output, refs):
    values, paths = joint.run_candidates(parameters, output, refs, friction_law='elastic')
    return values, paths


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=Path, required=True)
    parser.add_argument('--generations', type=int, default=6)
    parser.add_argument('--population', type=int, default=48)
    args = parser.parse_args(); assert 1 <= args.generations <= 8 and 12 <= args.population <= 64
    root = Path('artifacts/real2sim/h7-joint-response-goal-20261104-04-smooth')
    entries = [('artifacts/real2sim/h7-joint-response-refit-20261104-01/input-wide', None, root/'history-wide/history.npz'),
               ('artifacts/real2sim/h7-response-align-20261104-01/final-elastic-replay',
                'artifacts/real2sim/h7-response-align-20261104-01/rc-inputs.npz', root/'history-original/history.npz')]
    refs = []
    for path, rc, history in entries:
        path = Path(path); item = joint.reference_info(path, Path(rc) if rc else path/'rc-inputs.npz')
        item.update(memory_history=str(history.resolve()), memory_history_sha256=joint.sha256(history)); refs.append(item)
    seeds = json.loads(args.seeds.read_text()); assert 1 <= len(seeds) <= args.population
    args.output.mkdir(parents=True, exist_ok=False); shutil.copy2(__file__, args.output/Path(__file__).name)
    joint.write_json(args.output/'manifest.json', dict(scope='Two full development trajectories, two free-response input branches',
        references=refs, generations=args.generations, population=args.population, dimensions_per_side=DIM,
        objective='Equal trajectory/branch/joint RMS +0.05 worst peak +0.1 worst terminal; acceptance gates unchanged',
        future_measurements_used_for_initialization=False, physical_calibration_changed=False, policy_training_updates=0,
        seed=20261104, accepted=False))
    rng = np.random.default_rng(20261104); population = []
    for side in (0, 1):
        centers = np.asarray([encode(p, side) for p in seeds])
        x = np.clip(rng.normal(centers[np.arange(args.population)%len(centers)], .15), 0., 1.)
        x[:len(centers)] = centers; x[-4:] = rng.uniform(0., 1., (4, DIM)); population.append(x)
    profiles = [decode(population[0][i], population[1][i]) for i in range(args.population)]
    values, paths = evaluate(profiles, args.output/'generation-00', refs); history = []
    for generation in range(args.generations):
        best = np.argmin(values, axis=0)
        entry = dict(generation=generation, best_side_scores=joint.finite_scores(values[best, [0, 1]]), reports=paths)
        history.append(entry); joint.write_json(args.output/'progress.json', history); print(json.dumps(entry), flush=True)
        selected = decode(population[0][best[0]], population[1][best[1]])
        joint.write_json(args.output/'selected-parameters.json', selected)
        if generation == args.generations-1: break
        proposed = []
        for side in (0, 1):
            x = population[side]; next_x = []
            for i in range(args.population):
                a, b, c = rng.choice([j for j in range(args.population) if j != i], 3, replace=False)
                mutant = x[best[side]]+.6*(x[a]-x[b]) if i%3 == 0 else x[a]+.8*(x[b]-x[c])
                mask = rng.random(DIM) < .8; mask[rng.integers(DIM)] = True
                next_x.append(np.clip(np.where(mask, mutant, x[i]), 0., 1.))
            proposed.append(np.asarray(next_x))
        profiles = [decode(proposed[0][i], proposed[1][i]) for i in range(args.population)]
        new_values, paths = evaluate(profiles, args.output/f'generation-{generation+1:02d}', refs)
        for side in (0, 1):
            better = new_values[:, side] < values[:, side]
            population[side][better] = proposed[side][better]; values[better, side] = new_values[better, side]
    final, paths = evaluate([selected], args.output/'combined-verification', refs)
    joint.write_json(args.output/'report.json', dict(status='candidate_unaccepted', selected=selected, history=history,
        combined_scores=joint.finite_scores(final[0]), combined_reports=paths, source_scores=history[-1]['best_side_scores'],
        limitations=['Slow and fast regressions remain required', 'All existing recordings are development data',
                     'Initial history uses a conditional static allocation at its first pose']))


if __name__ == '__main__': main()
