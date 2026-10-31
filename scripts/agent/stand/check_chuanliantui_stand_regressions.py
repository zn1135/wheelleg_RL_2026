#!/usr/bin/env python3
"""Require every ever-passing fixed case, including successes from separate repeats."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.agent.stand.analyze_chuanliantui_stand import case_key, identity


def check(manifest_path, candidate_path):
    manifest = json.loads(manifest_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    task = manifest.get('task', 'chuanliantui_stand')
    if task not in ('chuanliantui_stand', 'chuanliantui_stand_ground_report') or candidate.get('task') != task:
        raise ValueError('Candidate must match the fixed or separate ground regression manifest')
    actual = {case_key(row): row for row in candidate['episodes']}
    rows = []
    for required in manifest['cases']:
        key = tuple(required['case'])
        row = actual.get(key)
        success = required['successes'][0]
        reference = json.loads((manifest_path.parent / success['report']).read_text())
        original = success['row']
        same_contract = (candidate['criteria'] == manifest.get('criteria', reference['criteria']) and
            candidate['xml']['sha256'] == reference['xml']['sha256'] and
            candidate['physics_dt_s'] == reference['physics_dt_s'] and
            candidate['policy_dt_s'] == reference['policy_dt_s'])
        same_initial, same_pose, same_physics, same_initial_velocity = False, False, False, None
        if row is not None:
            with np.load(original['trajectory']) as left, np.load(row['trajectory']) as right:
                same_pose = np.array_equal(left['qpos'][0, original['env']], right['qpos'][0, row['env']])
                same_initial = same_pose
                if task == 'chuanliantui_stand_ground_report':
                    # Ground resets contain real random initial velocities; preserve those too.
                    same_initial_velocity = np.array_equal(left['initial_qvel'][original['env']],
                        right['initial_qvel'][row['env']])
                    same_initial = (same_initial and same_initial_velocity
                        and row.get('initial_state_mode') == original.get('initial_state_mode') == 'ground')
            with np.load(Path(original['trajectory']).parent / 'runtime_model.npz') as left, \
                 np.load(Path(row['trajectory']).parent / 'runtime_model.npz') as right:
                same_physics = all(field in right and np.array_equal(left[field][original['env']],
                    right[field][row['env']]) for field in left.files)
        same_impulse = row is not None and row['vector'] == original['vector']
        retained = row is not None and row['passed'] and same_contract and same_initial and same_physics and same_impulse
        rows.append({'case': list(key), 'present': row is not None,
            'candidate_passed': bool(row is not None and row['passed']),
            'criteria_xml_and_timing_match_manifest': same_contract,
            'reference_criteria_version': reference.get('criteria_version', 'stand-angle-displacement-v1'),
            'candidate_criteria_version': candidate.get('criteria_version', 'stand-angle-displacement-v1'),
            'criteria_migration': manifest.get('criteria_migration'),
            'initial_qpos_identical': same_pose, 'saved_physical_parameters_identical': same_physics,
            'initial_qvel_identical_for_ground': same_initial_velocity,
            'impulse_vector_unchanged': same_impulse,
            'retained': bool(retained), 'reference_success_reports': [x['report'] for x in required['successes']]})
    return {'passed': all(row['retained'] for row in rows), 'required_cases': len(rows),
        'retained_cases': sum(row['retained'] for row in rows), 'cases': rows,
        'candidate': identity(candidate_path), 'manifest': identity(manifest_path),
        'scope': 'Union of real historical successes, not a fabricated combined baseline. Failed repeats remain failures.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    result = check(args.manifest, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('passed', 'required_cases', 'retained_cases')}))
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
