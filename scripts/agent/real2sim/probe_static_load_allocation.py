"""Test static-compatible load/friction allocations on the original dynamic recordings."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.agent.real2sim import fit_leg_repaired_loop as fit
from scripts.agent.real2sim.refit_joint_from_static_sweep import quality


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    previous=json.loads((args.previous/'report.json').read_text())
    (args.output/'inputs').symlink_to((args.previous/'inputs').resolve(),target_is_directory=True)
    inputs={}
    for name in tuple(fit.REFERENCES):
        path=args.output/'inputs'/name
        fit.REFERENCES[name]=(str(path),str(path/'rc-inputs.npz'))
        inputs[name]=fit.input_data(name)
    envelope=previous['static_envelope'];profiles=[];labels=[]
    for seed_name,seed in [('refitted',previous['selected_parameters']),('previous_motor_parameters',previous['baseline']['wide']['parameters'])]:
        for offset in (-20.,0.,20.):
            for share in (0.,.5,1.):
                profile=copy.deepcopy(seed);profile.update(copy.deepcopy(envelope))
                profile['spring_reference_force_n']=[v+offset for v in envelope['spring_reference_force_n']]
                profile['spring_friction_n']=[v*share for v in envelope['spring_friction_n']]
                profile['spring_friction_stiffness_n_m']=[v*share for v in envelope['spring_friction_stiffness_n_m']]
                profiles.append(profile);labels.append(dict(motor_seed=seed_name,pressure_offset_n=offset,slide_friction_share=share))
    fit.search.write_json(args.output/'generation-00-parameters.json',profiles)
    fit.search.write_json(args.output/'allocation-labels.json',labels)
    fit.wp.init();trials=[[] for _ in profiles];total=0
    for name in ('original','wide'):
        _,report=fit.run(profiles,inputs[name],args.output/'generation-00'/name,solver_profile='ordered')
        total+=report['new_gpu_environment_steps']
        for i,row in enumerate(report['results']):trials[i].append(dict(results=[row]))
    scores=np.array([quality(rows) for rows in trials]);index=int(np.argmin(scores.mean(1)))
    selected=profiles[index]
    fit.search.write_json(args.output/'selected-parameters.json',selected)
    fit.search.write_json(args.output/'allocation-selection.json',dict(index=index,label=labels[index],side_scores=scores.tolist(),
        reason='Static directional envelopes do not uniquely separate pressure, slider resistance and joint resistance',
        pressure_offsets_within_half_gap_over_observed_compression=True))
    final={}
    for name in inputs:
        _,report=fit.run([selected],inputs[name],args.output/('final-'+name),full=True,solver_profile='ordered')
        final[name]=report['results'][0];total+=report['new_gpu_environment_steps']
        (args.output/('baseline-'+name)).symlink_to((args.previous/('baseline-'+name)).resolve(),target_is_directory=True)
        fit.search.write_json(args.output/'partial-results.json',dict(final=final,gpu_environment_steps=total))
    for p,h in previous['protected_sources'].items():assert fit.audit.digest(Path(p))==h
    fit.search.write_json(args.output/'report.json',dict(final=final,baseline=previous['baseline'],
        selected_parameters=selected,history=[dict(generation=0)],static_envelope=envelope,
        previous_report=str((args.previous/'report.json').resolve()),previous_report_sha256=fit.audit.digest(args.previous/'report.json'),
        reused_baseline_reports=True,new_gpu_environment_steps=total,protected_sources=previous['protected_sources'],
        development_sessions=['original','wide'],regression_sessions=['slow','fast'],
        independent_hardware_holdout=False,shared_model_modified=False,hardware_access=False,policy_training_updates=0))
    for source in (Path(__file__),Path(fit.__file__)):
        shutil.copy2(source,args.output/source.name)
    print(json.dumps(dict(selected_index=index,selected_label=labels[index],scores=scores[index].tolist())),flush=True)


if __name__=='__main__':main()
