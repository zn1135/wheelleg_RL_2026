"""Frozen5000 nominal GPU straight tracking at +/-2.5 and +/-3m/s; no learning."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report-only', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    if a.report_only:
        summarize(out)
        return
    out.mkdir(parents=True, exist_ok=False)
    from scripts.agent.fudan_ppo_walk import inspect_randomization_4000 as frozen
    run = frozen.ROOT/'logs/mjlab/fudan_ppo_walk/20261116_high_yaw10_resume4000_to5000_01'
    model = run/'model_final.pt'
    assert frozen.sha(model) == '010106ac0b2ee356a630ab27ebe1c85ab2a6c843d21a1b4362ab81f1aafe0222'
    for name, digest in json.loads((run/'source_sha256.json').read_text()).items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            assert frozen.sha(run/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/')) == digest
        elif name.startswith('resources/'):
            assert frozen.sha(frozen.ROOT/name) == digest
    frozen.RUN = run
    sys.meta_path.insert(0, frozen.FrozenTrainingSource())
    import numpy as np
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_record
    from scripts.agent.fudan_ppo_walk import evaluate
    from scripts.agent.motion.acceptance import canonical
    torch.set_num_threads(1)
    source = torch.load(model, map_location='cpu', weights_only=False)
    conditions = replace(from_record(source['disturbances']), pushes=False,
                         external_force_range=(0., 0.), external_torque_range=(0., 0.))
    cases = []
    for height in (.22, .26, .30):
        for direction in (-1, 1):
            for speed in (2.5, 3.):
                # 两个速度沿用同case哈希和种子，确保同高度同方向的初态逐值配对。
                case = dict(id=f'forward_compare_{direction}_{height:g}',
                    category='forward' if direction > 0 else 'backward', duration=7.,
                    segments=[dict(start=0., duration=7., command=[direction*speed, 0., height])])
                cases.append(dict(id=f'vx{direction*speed:g}_h{height:g}__50001', case=case, seed=50001, randomized=False))
    request = SimpleNamespace(checkpoint=model, output=out/'gpu', deadline=time.time()+600,
        diagnostic=True, diagnostic_height=None, batch_size=12, selection=None,
        leg_load_profile='legacy', response_parameters=None)
    with patch.object(evaluate, 'evaluation_items', return_value=cases), \
         patch.object(evaluate, 'from_sim2sim_config', return_value=conditions), \
         patch('torch.optim.Adam.step', side_effect=AssertionError('Speed probe cannot learn')):
        evaluate.gpu(request)
    summarize(out, frozen.LOADED)


def summarize(out, modules=None):
    import numpy as np
    from scripts.agent.motion.acceptance import canonical
    from scripts.agent.fudan_ppo_walk import inspect_randomization_4000 as frozen
    cases = json.loads((out/'gpu/protocol.json').read_text())['samples']
    model = frozen.ROOT/'logs/mjlab/fudan_ppo_walk/20261116_high_yaw10_resume4000_to5000_01/model_final.pt'
    result = canonical(out/'gpu', 'fudan_ppo_walk')
    if not (out/'acceptance.json').exists():
        frozen.write(out/'acceptance.json', result)
    rows = []
    for sample, score in zip(cases, result['samples']):
        with np.load(out/'gpu/traces'/(sample['id']+'.npz')) as trace:
            selected = (trace['time'] >= 2.) & (trace['time'] < 7.)
            row = dict(sample_id=sample['id'], command=sample['case']['segments'][0]['command'],
                vx_mean_m_s=float(trace['linear_body'][selected, 0].mean()),
                vx_mae_m_s=float(np.abs(trace['linear_body'][selected, 0]-sample['case']['segments'][0]['command'][0]).mean()),
                yaw_mae_rad_s=float(np.abs(trace['angular_body'][selected, 2]).mean()),
                pitch_peak_deg=float(np.abs(trace['roll_pitch_deg'][selected, 1]).max()),
                technical_passed=score['passed'], failure_reasons=score['failure_reasons'],
                fell=score['v1_diagnostic']['fell'], complete=score['complete'])
            rows.append(row)
    paired = total = 0
    differences = []
    for i in range(0, len(cases), 2):
        with np.load(out/'gpu/fixtures'/(cases[i]['id']+'_initial.npz')) as left, \
             np.load(out/'gpu/fixtures'/(cases[i+1]['id']+'_initial.npz')) as right:
            for key in left.files:
                total += 1
                if np.array_equal(left[key], right[key]):
                    paired += 1
                else:
                    differences.append(dict(pair=i//2, field=key,
                        maximum_absolute_difference=float(np.abs(left[key]-right[key]).max())))
    frozen.write(out/'summary.json', dict(checkpoint=str(model), checkpoint_sha256=frozen.sha(model),
        rows=rows, matched_initial_arrays=paired, total_initial_arrays=total,
        initial_bitwise_equal=not differences, initial_differences=differences,
        optimizer_updates=0, frozen_modules=modules,
        scope='12 nominal flat-ground GPU cases; 2s response +5s measurement; no external disturbances; not slope capability or full acceptance'))
    for row in rows:
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
