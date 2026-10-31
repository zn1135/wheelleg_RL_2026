#!/usr/bin/env python3
"""Compare the same policy's fixed set before/after adding mixed training resets."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.agent.stand.analyze_chuanliantui_stand import case_key, identity


def compare(before, after):
    checks = []
    def record(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})

    for key in ('checkpoint', 'xml'):
        record(key + '_sha256_unchanged', before[key]['sha256'] == after[key]['sha256'])
    for key in ('criteria', 'seeds', 'profiles', 'suite', 'expected_cases', 'physics_dt_s', 'policy_dt_s'):
        record(key + '_unchanged', before[key] == after[key])
    record('both_are_near_start_matrices', before.get('initial_state_mode', 'near') ==
           after.get('initial_state_mode') == 'near')
    old = {case_key(row): row for row in before['episodes']}
    new = {case_key(row): row for row in after['episodes']}
    record('same_complete_case_set', old.keys() == new.keys() and
           len(old) == len(before['episodes']) == len(after['episodes']) == before['expected_cases'])
    if old.keys() != new.keys():
        return {'passed': False, 'checks': checks}
    groups = defaultdict(list)
    retained = []
    for key, row in old.items():
        candidate = new[key]
        record(f'case_{key}_impulse_unchanged', row['vector'] == candidate['vector'] and
               row['delivered_s'] == candidate['delivered_s'])
        if row['passed']:
            record(f'case_{key}_previous_success_retained', candidate['passed'])
            retained.append({'case': list(key), 'passed_before': True, 'passed_after': candidate['passed']})
        groups[row['trajectory']].append((row, candidate))
    for path, pairs in groups.items():
        new_paths = {candidate['trajectory'] for _, candidate in pairs}
        record(path + '_batch_members_unchanged', len(new_paths) == 1)
        if len(new_paths) != 1:
            continue
        other = Path(next(iter(new_paths)))
        with np.load(path) as left, np.load(other) as right:
            record(path + '_initial_qpos_identical', np.array_equal(left['qpos'][0], right['qpos'][0]))
            old_velocity = left['initial_qvel'] if 'initial_qvel' in left else np.zeros_like(right['initial_qvel'])
            record(path + '_initial_qvel_identical', np.array_equal(old_velocity, right['initial_qvel']))
        with np.load(Path(path).parent / 'runtime_model.npz') as left, \
             np.load(other.parent / 'runtime_model.npz') as right:
            record(path + '_runtime_field_names_identical', set(left.files) == set(right.files))
            for field in left.files:
                record(path + '_' + field + '_identical', field in right and np.array_equal(left[field], right[field]))
    return {'passed': all(item['passed'] for item in checks), 'checks': checks,
            'previous_successes': retained, 'case_count': len(old),
            'scope': 'Same policy, fixed criteria/seeds/cases, actual initial states and physical parameters; not a new policy acceptance result.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    before, after = (json.loads(path.read_text()) for path in (args.before, args.after))
    if any(report.get('task') != 'chuanliantui_stand' for report in (before, after)):
        parser.error('Both inputs must be fixed standing-task assessments')
    result = compare(before, after)
    result.update(before=identity(args.before), after=identity(args.after))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'checks': len(result['checks']),
                      'failures': [item['name'] for item in result['checks'] if not item['passed']]}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
