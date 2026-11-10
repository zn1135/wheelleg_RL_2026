"""Select one fixed diagnostic profile across paired development trajectories; no integration."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.agent.real2sim.refine_leg_response_joint import score


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--report-pairs',type=Path,required=True,help='JSON list of [wide-report, original-report] paths')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    pairs=json.loads(args.report_pairs.read_text());rows=[];identities=None
    for paths in pairs:
        assert len(paths)==2
        reports=[json.loads(Path(p).read_text()) for p in paths]
        current=[(r['stream_sha256'],r['source_model_sha256'],r['steps_per_world']) for r in reports]
        if identities is None:identities=current
        assert current==identities,'Selection must use the same two complete input trajectories'
        assert all(r['physics_backend']=='mujoco_warp_gpu' for r in reports)
        assert len(reports[0]['trials'])==len(reports[1]['trials'])
        for index,trial in enumerate(reports[0]['trials']):
            assert trial['parameters']==reports[1]['trials'][index]['parameters']
            assert np.allclose(trial['parameters']['armature_kg_m2'],.015795),'Keep physical inertia fixed'
            values=[score([r['trials'][index] for r in reports],side) for side in (0,1)]
            rows.append(dict(reports=paths,index=index,side_scores_deg=[float(x) if np.isfinite(x) else None for x in values],
                             objective_deg=float(np.mean(values)) if np.isfinite(values).all() else None,
                             parameters=trial['parameters']))
    eligible=[r for r in rows if r['objective_deg'] is not None]
    assert eligible,'No candidate passed data/passivity/feedback eligibility'
    selected=min(eligible,key=lambda r:r['objective_deg'])
    args.output.mkdir(parents=True,exist_ok=False)
    result=dict(scope='one fixed profile chosen using two development recordings; not independent acceptance',
                objective='mean of per-side equal trajectory/branch/joint RMS + .05 worst peak + .1 worst terminal',
                candidate_comparisons=len(rows),eligible_comparisons=len(eligible),selected=selected,trials=rows,
                new_environment_steps=0,policy_training_updates=0)
    (args.output/'selection.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output/'selected-parameters.json').write_text(json.dumps(selected['parameters'],indent=2)+'\n')
    (args.output/'single-candidate.json').write_text(json.dumps([selected['parameters']],indent=2)+'\n')
    print(json.dumps({k:v for k,v in selected.items() if k!='parameters'},indent=2))


if __name__=='__main__':main()
