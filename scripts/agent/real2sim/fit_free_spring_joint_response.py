"""Finite unconstrained-specification spring/joint response search on original recordings."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.agent.real2sim import fit_leg_repaired_loop as fit


def encode(profile):
    rows=[]
    for side in (0,1):
        row=[profile['spring_reference_force_n'][side]/600.,
             (profile['spring_stiffness_n_m'][side]+10000.)/30000.,
             profile['spring_damping_ns_m'][side]/1500.]
        for j in (2*side,2*side+1):
            static=profile['friction_nm'][j]
            row.extend([fit.search.logarithm(static,.02,30.),
                (profile['motor_dynamic_positive_nm'][j]/static-.02)/.98,
                (profile['motor_dynamic_negative_nm'][j]/static-.02)/.98,
                fit.search.logarithm(profile['motor_velocity_scale_rad_s'][j],.001,.5),
                fit.search.logarithm(profile['damping_nms_rad'][j],.001,15.)])
        row.append(fit.search.logarithm(profile.get('spring_friction_n',[.01,.01])[side],.001,300.))
        rows.append(row)
    return np.clip(rows,0.,1.)


def decode(vectors):
    profile=dict(armature_kg_m2=[.015795]*4,spring_force_limit_n=[600.]*2)
    for key in ('spring_reference_force_n','spring_stiffness_n_m','spring_damping_ns_m','spring_friction_n',
                'friction_nm','motor_dynamic_positive_nm','motor_dynamic_negative_nm','motor_velocity_scale_rad_s','damping_nms_rad'):
        profile[key]=[]
    for row in vectors:
        profile['spring_reference_force_n'].append(float(600.*row[0]))
        profile['spring_stiffness_n_m'].append(float(-10000.+30000.*row[1]))
        profile['spring_damping_ns_m'].append(float(1500.*row[2]))
        for offset in (3,8):
            static=.02*1500.**row[offset]
            profile['friction_nm'].append(float(static))
            profile['motor_dynamic_positive_nm'].append(float(static*(.02+.98*row[offset+1])))
            profile['motor_dynamic_negative_nm'].append(float(static*(.02+.98*row[offset+2])))
            profile['motor_velocity_scale_rad_s'].append(float(.001*500.**row[offset+3]))
            profile['damping_nms_rad'].append(float(.001*15000.**row[offset+4]))
        profile['spring_friction_n'].append(float(.001*300000.**row[13]))
    return profile


def cost(rows):
    values=[]
    for row in rows:
        r=row['recorded_torque']
        if not r['original_assembly_branch'] or r['limit_excess_deg']>.1:return 1e6
        values.extend(np.square(r['rmse_deg']))
    return float(np.sqrt(np.mean(values)))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    previous=Path('artifacts/real2sim/h7-static-constrained-joint-refit-20261105-02')
    (args.output/'inputs').symlink_to((previous/'inputs').resolve(),target_is_directory=True)
    protected=json.loads((previous/'report.json').read_text())['protected_sources']
    inputs={}
    for name in tuple(fit.REFERENCES):
        path=previous/'inputs'/name;fit.REFERENCES[name]=(str(path),str(path/'rc-inputs.npz'));inputs[name]=fit.input_data(name)
    pool=Path('artifacts/real2sim/h7-ordered-refit-20261104-01/pool')
    old=[json.loads((pool/name/'report.json').read_text()) for name in ('original','wide')]
    assert [r['parameters'] for r in old[0]['results']]==[r['parameters'] for r in old[1]['results']]
    ranks=sorted(range(len(old[0]['results'])),key=lambda i:cost([d['results'][i] for d in old]))
    seeds=[old[0]['results'][i]['parameters'] for i in ranks[:4]]
    vectors=np.array([encode(seeds[i%4]) for i in range(12)])
    rng=np.random.default_rng(2026110503)
    for i in range(4,12):
        perturb=rng.normal(0.,.025 if i<8 else .08,(2,14));vectors[i]=np.clip(vectors[i]+perturb,0.,1.)
    fit.wp.init();total=0;all_rows=[]

    def evaluate(values,generation):
        nonlocal total
        profiles=[decode(v) for v in values]
        if generation==0:profiles[0]=copy.deepcopy(seeds[0])
        fit.search.write_json(args.output/f'generation-{generation:02d}-parameters.json',profiles)
        results=[[] for _ in profiles]
        for name in ('original','wide'):
            _,r=fit.run(profiles,inputs[name],args.output/f'generation-{generation:02d}'/name,solver_profile='ordered')
            total+=r['new_gpu_environment_steps']
            for i,row in enumerate(r['results']):results[i].append(row)
        scores=np.array([cost(r) for r in results])
        all_rows.extend(dict(score=float(s),parameters=p,generation=generation,index=i) for i,(s,p) in enumerate(zip(scores,profiles)))
        fit.search.write_json(args.output/'progress.json',dict(generation=generation,best_joint_rmse_deg=float(scores.min()),new_gpu_environment_steps=total))
        return scores

    scores=evaluate(vectors,0)
    best=vectors[int(np.argmin(scores))]
    proposed=vectors.copy()
    for i in range(12):
        a,b=rng.choice(np.delete(np.arange(12),i),2,replace=False)
        mutant=vectors[i]+.5*(best-vectors[i])+.5*(vectors[a]-vectors[b])
        if i<4:mutant=best+rng.normal(0.,.01 if i<2 else .03,(2,14))
        proposed[i]=np.clip(np.where(rng.random((2,14))<.65,mutant,vectors[i]),0.,1.)
    evaluate(proposed,1)
    selected=min(all_rows,key=lambda r:r['score']);profile=selected['parameters']
    fit.search.write_json(args.output/'selection.json',dict(selected=selected,all_evaluations=all_rows,
        objective='Equal original/wide trajectory and joint RMSE under identical recorded torques; no static-load or manufacturer prior',
        numerical_bounds=dict(reference_n=[0,600],stiffness_n_m=[-10000,20000],spring_damping_ns_m=[0,1500],motor_static_friction_nm=[.02,30.])))
    fit.search.write_json(args.output/'selected-parameters.json',profile)
    final={}
    for name in inputs:
        _,r=fit.run([profile],inputs[name],args.output/('final-'+name),full=True,solver_profile='ordered')
        total+=r['new_gpu_environment_steps'];final[name]=r['results'][0]
        (args.output/('baseline-'+name)).symlink_to((previous/('baseline-'+name)).resolve(),target_is_directory=True)
        fit.search.write_json(args.output/'partial-results.json',dict(final=final,new_gpu_environment_steps=total))
    assert {p:fit.audit.digest(Path(p)) for p in protected}==protected
    fit.search.write_json(args.output/'report.json',dict(final=final,selected_parameters=profile,new_gpu_environment_steps=total,
        protected_sources=protected,gas_spring_measured_parameters_fixed=False,static_fit_penalty=False,
        model_family='Free linear effective spring including negative slope, native directional joint friction and damping',
        reference_force_at_compression_m=.005,force_clip_n=[0,600],physical_parameters_identified=False,
        development_sessions=['original','wide'],regression_sessions=['slow','fast'],
        history=[dict(generation=0),dict(generation=1)],shared_model_modified=False,hardware_access=False,policy_training_updates=0,
        baseline=json.loads((previous/'report.json').read_text())['baseline']))
    for p in (Path(__file__),Path(fit.__file__)):shutil.copy2(p,args.output/p.name)


if __name__=='__main__':main()
