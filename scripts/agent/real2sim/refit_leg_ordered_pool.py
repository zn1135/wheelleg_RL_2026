"""Re-score the existing finite candidate pool with ordered constraints, then freeze one complete profile."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.agent.real2sim import fit_leg_repaired_loop as fit


def quality(a,b):
    values=[]; average=[]
    for result in (a,b):
        if result['feedback_blocked']:return (1e6,1e6)
        for branch in ('recorded_torque','recorded_target'):
            row=result[branch]
            if not row['original_assembly_branch'] or row['limit_excess_deg']>.1:return (1e6,1e6)
            values.extend(np.asarray(row['rmse_deg'])/3.)
            values.extend(np.asarray(row['peak_deg'])/5.)
            values.extend(np.asarray(row['terminal_deg']))
            average.extend(row['rmse_deg'])
    return float(max(values)),float(np.mean(average))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);fit.wp.init()
    protected=fit.audit.protected_sources(Path('artifacts/real2sim/h7-joint-response-goal-20261104-05-breakaway'),[])
    profiles=[];hashes={}
    for generation in range(4):
        path=args.pool/f'generation-{generation:02d}-parameters.json';profiles.extend(json.loads(path.read_text()))
        hashes[str(path)]=fit.audit.digest(path)
    assert len(profiles)==96
    fit.search.write_json(args.output/'parameters.json',profiles)
    inputs={name:fit.input_data(name) for name in fit.REFERENCES}; reports={};steps=0
    for name in ('original','wide'):
        _,report=fit.run(profiles,inputs[name],args.output/'pool'/name,solver_profile='ordered')
        reports[name]=report;steps+=report['new_gpu_environment_steps']
    scores=[quality(a,b) for a,b in zip(reports['original']['results'],reports['wide']['results'])]
    selected=min(range(len(scores)),key=lambda i:scores[i]);parameters=profiles[selected]
    fit.search.write_json(args.output/'selection.json',dict(index=selected,quality=scores[selected],all_quality=scores,
        objective='Minimize worst normalized joint RMSE/3, peak/5, terminal/1 over both fit recordings and input branches; mean RMSE breaks ties',
        clone_tolerance_is_not_a_selection_gate=True))
    fit.search.write_json(args.output/'selected-parameters.json',parameters)
    final={};parity={}
    for name in fit.REFERENCES:
        _,report=fit.run([parameters],inputs[name],args.output/('final-'+name),full=True,solver_profile='ordered')
        final[name]=report['results'][0];steps+=report['new_gpu_environment_steps']
        if name in reports:
            parity[name]={branch:float(np.max(abs(np.asarray(final[name][branch]['rmse_deg'])
                -reports[name]['results'][selected][branch]['rmse_deg']))) for branch in ('recorded_torque','recorded_target')}
    assert fit.audit.protected_sources(Path('artifacts/real2sim/h7-joint-response-goal-20261104-05-breakaway'),[])==protected
    fit.search.write_json(args.output/'report.json',dict(candidates_evaluated=len(profiles),selected_index=selected,
        selected_parameters=parameters,final=final,selection_quality=scores[selected],
        single_vs_batch_max_joint_rmse_difference_deg=parity,new_gpu_environment_steps=steps,
        protected_source_checks=protected,source_parameter_pool_hashes=hashes,
        solver_profile='ordered',accepted=False,shared_model_modified=False,hardware_access=False,
        development_sessions=['original','wide'],regression_sessions=['slow','fast'],policy_training_updates=0))
    from scripts.agent.real2sim import ordered_leg_constraints as ordered
    for path in (Path(__file__),Path(fit.__file__),Path(ordered.__file__)):
        shutil.copy2(path,args.output/path.name)


if __name__=='__main__':main()
