"""Freeze the lowest whole-trajectory joint RMS candidate from an ordered GPU pool."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.agent.real2sim import fit_leg_repaired_loop as fit


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);fit.wp.init(); reports={};hashes={};reused=0
    for name in ('original','wide'):
        path=args.pool/'pool'/name/'report.json';report=json.loads(path.read_text())
        assert report['solver_profile']=='ordered' and report['numerical_regularization']=='original'
        assert fit.audit.digest(path.with_name('model.xml'))==report['model_sha256']
        for source,value in report['input_sha256'].items():assert fit.audit.digest(Path(source))==value
        reports[name]=report;hashes[str(path)]=fit.audit.digest(path);reused+=report['new_gpu_environment_steps']
    scores=[]
    for a,b in zip(reports['original']['results'],reports['wide']['results']):
        assert a['parameters']==b['parameters'];rmse=[];allowed=True
        for result in (a,b):
            allowed &= not result['feedback_blocked']
            for branch in ('recorded_torque','recorded_target'):
                row=result[branch];rmse.extend(row['rmse_deg'])
                allowed &= row['original_assembly_branch'] and row['limit_excess_deg']<=.1
        scores.append(float(np.sqrt(np.mean(np.asarray(rmse)**2))) if allowed else 1e6)
    selected=int(np.argmin(scores));parameters=reports['original']['results'][selected]['parameters']
    fit.search.write_json(args.output/'selection.json',dict(index=selected,combined_joint_rms_deg=scores[selected],all_scores=scores,
        objective='Equal weight per trajectory, input branch and joint; minimum sqrt(mean(joint_RMSE^2)); peak and terminal reported separately',
        tight_copy_tolerance_is_not_a_selection_gate=True))
    fit.search.write_json(args.output/'selected-parameters.json',parameters);steps=0;final={}
    for name in fit.REFERENCES:
        _,report=fit.run([parameters],fit.input_data(name),args.output/('final-'+name),full=True,solver_profile='ordered')
        final[name]=report['results'][0];steps+=report['new_gpu_environment_steps']
    fit.search.write_json(args.output/'report.json',dict(candidates_evaluated=len(scores),selected_index=selected,
        selected_parameters=parameters,final=final,combined_fit_joint_rms_deg=scores[selected],
        solver_profile='ordered',new_gpu_environment_steps=steps,reused_valid_gpu_environment_steps=reused,
        verified_gpu_environment_steps=reused+steps,pool_report_hashes=hashes,
        protected_source_checks=fit.audit.protected_sources(Path('artifacts/real2sim/h7-joint-response-goal-20261104-05-breakaway'),[]),
        accepted=False,best_available_candidate=True,shared_model_modified=False,hardware_access=False,
        development_sessions=['original','wide'],regression_sessions=['slow','fast'],policy_training_updates=0))
    from scripts.agent.real2sim import ordered_leg_constraints as ordered
    for path in (Path(__file__),Path(fit.__file__),Path(ordered.__file__)):
        shutil.copy2(path,args.output/path.name)


if __name__=='__main__':main()
