"""Finite full-trajectory differential evolution of directional breakaway resistance."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.agent.real2sim import refine_leg_response_joint as joint


DIM=12


def encode(p,side):
    f=np.asarray(p['friction_nm']);bias=np.asarray(p.get('motor_friction_bias_nm',[0.]*4))
    extra=p.get('motor_static_excess_nm',[1.]*4);decay=p.get('motor_static_decay_m',[.02]*2)
    values=[(p['spring_reference_force_n'][side]-50.)/200.,p['spring_stiffness_n_m'][side]/7000.,
            joint.logarithm(p.get('spring_damping_ns_m',[50.]*2)[side],10.,500.),joint.logarithm(decay[side],.005,.1)]
    for j in (side*2,side*2+1):
        values.extend([joint.logarithm(f[j]+bias[j],.02,8.),joint.logarithm(f[j]-bias[j],.02,8.),
                       joint.logarithm(p['damping_nms_rad'][j],.05,5.),joint.logarithm(extra[j],.1,40.)])
    return np.clip(values,0.,1.)


def decode(left,right):
    p=dict(friction_nm=[],motor_friction_bias_nm=[],damping_nms_rad=[],motor_static_excess_nm=[],motor_static_decay_m=[],
           motor_static_velocity_scale_rad_s=[.02]*4,armature_kg_m2=[.015795]*4,spring_reference_force_n=[],spring_stiffness_n_m=[],
           spring_damping_ns_m=[],spring_force_limit_n=[600.]*2,spring_extension_stop_m=[.007]*2,spring_extension_margin_m=[.001]*2)
    for x in (left,right):
        p['spring_reference_force_n'].append(float(50.+200.*x[0]));p['spring_stiffness_n_m'].append(float(7000.*x[1]))
        p['spring_damping_ns_m'].append(float(10.*50.**x[2]));p['motor_static_decay_m'].append(float(.005*20.**x[3]))
        for offset in (4,8):
            positive=.02*400.**x[offset];negative=.02*400.**x[offset+1]
            p['friction_nm'].append(float((positive+negative)/2));p['motor_friction_bias_nm'].append(float((positive-negative)/2))
            p['damping_nms_rad'].append(float(.05*100.**x[offset+2]));p['motor_static_excess_nm'].append(float(.1*400.**x[offset+3]))
    return p


def score(reports,side):
    section=slice(2*side,2*side+2)
    base=joint.score(reports,side)
    if not np.isfinite(base):return base
    span=[]
    for r in reports:
        real=np.asarray(r['real_motor_span_deg'])[section]
        for key in ('all_motor_span_deg','feedback_motor_span_deg'):
            span.extend(abs(np.asarray(r[key])[section]-real))
    return base+.08*float(np.sqrt(np.mean(np.square(span))))


def evaluate(parameters,output,references):
    _,paths=joint.run_candidates(parameters,output,references)
    reports=[json.loads(Path(path).read_text()) for path in paths]
    values=np.asarray([[score([r['trials'][i] for r in reports],side) for side in (0,1)] for i in range(len(parameters))])
    joint.write_json(output/'selection-scores.json',dict(scores=[joint.finite_scores(row) for row in values],reports=paths))
    return values,paths


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--references',nargs='+',type=Path,required=True)
    parser.add_argument('--rc-inputs',nargs='+',type=Path,required=True);parser.add_argument('--seeds',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--population',type=int,default=48)
    parser.add_argument('--generations',type=int,default=8)
    args=parser.parse_args();assert len(args.references)==len(args.rc_inputs) and 12<=args.population<=64 and 1<=args.generations<=12
    refs=[joint.reference_info(p,rc) for p,rc in zip(args.references,args.rc_inputs)]
    seeds=json.loads(args.seeds.read_text());assert len(seeds)<=args.population
    args.output.mkdir(parents=True,exist_ok=False)
    joint.write_json(args.output/'manifest.json',dict(scope='fixed-base full-trajectory development fit only',references=refs,population=args.population,generations=args.generations,
                     objective='joint both-input RMSE/peak/terminal score +0.08 activity-span RMSE; original final gates unchanged',seed=20261104,accepted=False,physics_source='fit_leg_response_gpu',policy_training_updates=0))
    rng=np.random.default_rng(20261104);population=[]
    for side in (0,1):
        seed=np.asarray([encode(p,side) for p in seeds]);x=np.clip(rng.normal(seed[np.arange(args.population)%len(seed)],.15),0,1)
        x[:len(seed)]=seed;x[-4:]=rng.uniform(0,1,(4,DIM));population.append(x)
    profiles=[decode(population[0][i],population[1][i]) for i in range(args.population)]
    values,paths=evaluate(profiles,args.output/'generation-00',refs);history=[]
    for generation in range(args.generations):
        best=np.argmin(values,axis=0);entry=dict(generation=generation,best_side_scores=[float(values[best[s],s]) for s in (0,1)],reports=paths)
        history.append(entry);joint.write_json(args.output/'progress.json',history);print(json.dumps(entry),flush=True)
        selected=decode(population[0][best[0]],population[1][best[1]])
        joint.write_json(args.output/'selected-parameters.json',selected)
        if generation==args.generations-1:break
        trials=[]
        for side in (0,1):
            x=population[side];proposed=[]
            for i in range(args.population):
                pool=[j for j in range(args.population) if j!=i];a,b,c=rng.choice(pool,3,replace=False)
                if i%3==0:mutant=x[best[side]]+.6*(x[a]-x[b])
                else:mutant=x[a]+.8*(x[b]-x[c])
                cross=rng.random(DIM)<.8;cross[rng.integers(DIM)]=True
                proposed.append(np.clip(np.where(cross,mutant,x[i]),0,1))
            trials.append(np.asarray(proposed))
        profiles=[decode(trials[0][i],trials[1][i]) for i in range(args.population)]
        candidate_scores,paths=evaluate(profiles,args.output/f'generation-{generation+1:02d}',refs)
        for side in (0,1):
            improve=candidate_scores[:,side]<values[:,side]
            population[side][improve]=trials[side][improve];values[improve,side]=candidate_scores[improve,side]
    final_scores,final_paths=evaluate([selected],args.output/'combined-verification',refs)
    joint.write_json(args.output/'report.json',dict(status='candidate_unaccepted',selected=selected,history=history,combined_scores=joint.finite_scores(final_scores[0]),
                     combined_reports=final_paths,source_scores=history[-1]['best_side_scores'],new_policy_training_updates=0,
                     limitations=['All selection traces are development data','Composite-side selection is valid only for fixed-base contact-disabled model','Full four-trace gates remain required']))


if __name__=='__main__':main()
