#!/usr/bin/env python3
"""Report original-bank ground get-up separately; never replace fixed acceptance."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from scripts.agent.stand import assess_chuanliantui_stand as assess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=('nominal', 'randomized', 'both'), default='both')
    parser.add_argument('--seeds', default='42,43')
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--cases-per-seed', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not args.checkpoint.is_file() or min(args.num_envs, args.cases_per_seed) < 1:
        parser.error('Existing checkpoint and positive environment counts are required')
    seeds = [int(value) for value in args.seeds.split(',') if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error('Provide distinct nonempty seeds')
    directory = args.output.with_suffix('')
    if args.output.exists() or directory.exists():
        parser.error('Choose a new report path')
    directory.mkdir(parents=True)
    args.start_mode = 'ground'
    profiles = ('nominal', 'randomized') if args.profile == 'both' else (args.profile,)
    cases = assess.make_cases('hold', hold_envs=args.cases_per_seed)
    rows = []
    for profile in profiles:
        for seed in seeds:
            for offset in range(0, len(cases), args.num_envs):
                batch = assess.evaluate_batch(args, cases[offset:offset + args.num_envs],
                    profile, seed, offset // args.num_envs, directory)
                with np.load(batch[0]['trajectory']) as archive:
                    raw, times = archive['physics'], archive['physics_time_s']
                    for row in batch:
                        before = times <= (row['anchor_s'] if row['anchor_s'] is not None else row['ended_s'])
                        trace = raw[before, row['env']]
                        row.update(getup_qualified=row['anchor_s'] is not None,
                            getup_qualification_s=row['anchor_s'],
                            observed_base_contact_before_getup=bool((trace[:, 18] > .5).any()),
                            minimum_height_before_getup_m=float(trace[:, 16].min()) if len(trace) else None)
                rows.extend(batch)
                print(json.dumps({'profile': profile, 'seed': seed, 'batch': offset // args.num_envs,
                    'getup_qualified': sum(row['getup_qualified'] for row in batch),
                    'hold_passed': sum(row['passed'] for row in batch), 'cases': len(batch)}), flush=True)
    from ct_mjlab.envs.chuanliantui.model import MODEL_XML
    report = {'kind': __doc__, 'task': 'chuanliantui_stand_ground_report',
        'checkpoint': assess.identity(args.checkpoint), 'xml': assess.identity(MODEL_XML),
        'initial_state_mode': 'original ground reset with new-task ground-clearance correction',
        'target_command': [0., 0., .27], 'profiles': profiles, 'seeds': seeds,
        'physics_dt_s': .002, 'policy_dt_s': .01, 'physics_capture_hz': 500,
        'expected_cases': len(cases) * len(profiles) * len(seeds), 'completed_cases': len(rows),
        'getup_qualified_cases': sum(row['getup_qualified'] for row in rows),
        'stand_and_hold_passed_cases': sum(row['passed'] for row in rows),
        'full_acceptance_passed': False,
        'qualification': 'strict body stability for0.5s within5s; no leg-speed eligibility condition',
        'hold': 'same20s bodyXYspeed<=.05m/s and eachlegabsvelocity<=.5rad/s; separate ground report',
        'criteria_version': assess.CRITERIA_VERSION, 'criteria': assess.acceptance_criteria(),
        'sources': [assess.identity(Path(__file__)), assess.identity(Path(assess.__file__)),
                    assess.identity(MODEL_XML.parent / 'huananhu_pose_bank.npz')],
        'command': sys.argv, 'episodes': rows,
        'scope': 'Separate ground-start learning report; never replaces the fixed42/43 and44/45 matrices.'}
    args.output.write_text(json.dumps(report, default=assess.json_default, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in
        ('getup_qualified_cases', 'stand_and_hold_passed_cases', 'completed_cases', 'full_acceptance_passed')}))
    if not all(row['getup_qualified'] for row in rows):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
