"""Select one complete candidate, then independently replay all4 sessions with canonical physics."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.agent.real2sim import fit_leg_repaired_loop as fit


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False); fit.wp.init()
    parent = json.loads((args.pool/'report.json').read_text()); records = []
    for item in parent['history']:
        generation = item['generation']; original = args.pool/f'generation-{generation:02d}'/'original'
        wide = Path(item['reused_wide'])
        assert fit.audit.digest(wide/'report.json') == item['reused_report_sha256']
        assert fit.audit.digest(wide/'trials.npz') == item['reused_trace_sha256']
        a = json.loads((original/'report.json').read_text()); b = json.loads((wide/'report.json').read_text())
        scores = (np.asarray(a['side_scores'])+np.asarray(b['side_scores']))/2
        for i, trial in enumerate(a['results']):
            assert trial['parameters'] == b['results'][i]['parameters']
            records.append(dict(generation=generation, index=i, parameters=trial['parameters'],
                                score=float(scores[i].mean()), side_scores=scores[i].tolist(),
                                batch_original=trial, batch_wide=b['results'][i]))
    selected = min(records, key=lambda r:r['score']); parameters = selected['parameters']
    fit.search.write_json(args.output/'selection.json', selected)
    fit.search.write_json(args.output/'selected-parameters.json', parameters)
    final = {}; steps = 0; parity = {}
    for name in fit.REFERENCES:
        inputs = fit.input_data(name)
        _, result = fit.run([parameters], inputs, args.output/('final-'+name), full=True)
        final[name] = result['results'][0]; steps += result['new_gpu_environment_steps']
        if name in ('original', 'wide'):
            batch = selected['batch_'+name]
            parity[name] = {branch:float(np.max(abs(np.asarray(final[name][branch]['rmse_deg'])
                                  -batch[branch]['rmse_deg']))) for branch in ('recorded_torque', 'recorded_target')}
    protected = fit.audit.protected_sources(Path('artifacts/real2sim/h7-joint-response-goal-20261104-05-breakaway'), [])
    fit.search.write_json(args.output/'report.json', dict(selected_parameters=parameters, final=final,
        candidates_evaluated=len(records), selection='One complete candidate minimizing equal-trajectory mean of left/right response scores; never splice independently selected sides',
        selected_pool_candidate={k:selected[k] for k in ('generation', 'index', 'score', 'side_scores')},
        single_vs_batch_max_joint_rmse_difference_deg=parity,
        new_gpu_environment_steps=steps, reused_valid_gpu_environment_steps=42163200,
        verified_gpu_environment_steps=42163200+steps, protected_source_checks=protected,
        canonical_shared_inertia_for_all_sessions=True, accepted=False, shared_model_modified=False,
        policy_training_updates=0, hardware_access=False, pool_report=str(args.pool/'report.json'),
        pool_report_sha256=fit.audit.digest(args.pool/'report.json')))
    for source in (Path(__file__), Path(fit.__file__)):
        shutil.copy2(source, args.output/source.name)


if __name__ == '__main__': main()
