"""Finite, seeded candidate refinement on one development trajectory; no policy training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def logarithm(value,low,high):
    return np.log(np.clip(value,low,high)/low)/np.log(high/low)


def encode(profile,side):
    c=profile['coordinate_friction_nm'];k=profile.get('coordinate_stiffness_nm_rad',[0.]*4)
    return np.array([profile.get('spring_bias_n',[0.,0.])[side]/150,
                     logarithm(c[2*side],.1,2.),0. if k[2*side]==0 else .25+.75*logarithm(k[2*side],.1,10.),
                     logarithm(c[2*side+1],4.,30.),logarithm(k[2*side+1],20.,1000.),
                     logarithm(profile['damping_nms_rad'][2*side],.1,3.)])


def decode(left,right):
    profile=dict(friction_nm=[.02]*4,damping_nms_rad=[],armature_kg_m2=[.015795]*4,
                 spring_bias_n=[],coordinate_friction_nm=[],coordinate_stiffness_nm_rad=[])
    for x in (left,right):
        common=.1*20**x[1];kc=0. if x[2]<.25 else .1*100**((x[2]-.25)/.75)
        difference=4.*7.5**x[3];kd=20.*50**x[4]
        damping=max(.1*30**x[5],kd*.00025+.05)
        profile['spring_bias_n'].append(float(x[0]*150.))
        profile['coordinate_friction_nm'].extend([float(common),float(difference)])
        profile['coordinate_stiffness_nm_rad'].extend([float(kc),float(kd)])
        profile['damping_nms_rad'].extend([float(damping)]*2)
    return profile


def score(trial,side):
    indices=slice(side*2,side*2+2)
    rmse=np.r_[trial['all_motor_rmse_deg'][indices],trial['feedback_motor_rmse_deg'][indices]]
    peak=max(*trial['peak_error_deg'][indices],*trial['feedback_peak_error_deg'][indices])
    final=max(*np.abs(trial['final_error_deg'][indices]),*np.abs(trial['feedback_final_error_deg'][indices]))
    if trial['feedback_blocked'] or trial.get('minimum_discrete_dissipation_j',[0.,0.])[side]<-1e-7:return np.inf
    return float(np.sqrt(np.mean(rmse**2))+.05*peak+.1*final)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--seed-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--generations',type=int,default=4)
    parser.add_argument('--population',type=int,default=32)
    args=parser.parse_args();assert 1<=args.generations<=6 and 8<=args.population<=64
    args.output.mkdir(parents=True,exist_ok=False)
    seed=json.loads(args.seed_report.read_text())['best']['parameters']
    report=json.loads((args.reference/'report.json').read_text())
    rng=np.random.default_rng(20261104)
    centers=np.array([encode(seed,side) for side in (0,1)])
    deviation=np.full((2,6),.2)
    populations=[]
    for side in range(2):
        population=np.clip(rng.normal(centers[side],deviation[side],size=(args.population,6)),0,1)
        population[args.population//2:]=rng.uniform(0,1,size=(args.population-args.population//2,6))
        population[0]=centers[side];populations.append(population)
    best_scores=np.full(2,np.inf);best_vectors=centers.copy();best_sources=[None,None];history=[];stagnation=0
    for generation in range(args.generations):
        parameters=[decode(populations[0][i],populations[1][i]) for i in range(args.population)]
        source=args.output/f'parameters-{generation:02d}.json';source.write_text(json.dumps(parameters,indent=2)+'\n')
        out=args.output/f'generation-{generation:02d}'
        command=[sys.executable,'-m','scripts.agent.real2sim.fit_leg_response_gpu','--reference',str(args.reference),
                 '--parameters',str(source),'--fit-until-s',str(report['duration_s']),'--feedback-validation',
                 '--rc-inputs',str(args.reference/'rc-inputs.npz'),'--output',str(out)]
        with (args.output/f'generation-{generation:02d}.log').open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        trials=json.loads((out/'report.json').read_text())['trials']
        improved=False
        for side in range(2):
            scores=np.array([score(trial,side) for trial in trials]);order=np.argsort(scores)
            if scores[order[0]]<best_scores[side]:
                best_scores[side]=scores[order[0]];best_vectors[side]=populations[side][order[0]]
                best_sources[side]=dict(generation=generation,index=int(order[0]));improved=True
            elite=populations[side][order[:max(4,args.population//6)]]
            centers[side]=elite.mean(0);deviation[side]=np.maximum(elite.std(0),.05)*.9
            populations[side]=np.clip(rng.normal(centers[side],deviation[side],size=(args.population,6)),0,1)
            populations[side][0]=best_vectors[side]
            populations[side][-4:]=rng.uniform(0,1,size=(4,6))
        history.append(dict(generation=generation,best_side_scores_deg=best_scores.tolist(),sources=list(best_sources)))
        (args.output/'progress.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps(history[-1]),flush=True)
        stagnation=0 if improved else stagnation+1
        if stagnation>=2:break
    selected=decode(*best_vectors)
    (args.output/'selected-parameters.json').write_text(json.dumps(selected,indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(dict(scope='fixed-base development fit; no independent acceptance',
         seed=20261104,objective='per-side both-input RMSE +0.05peak +0.1terminal; reject negative discrete dissipation',
         histories=history,selected=selected,side_composition=best_sources,
         reason='two legs independent only in this fixed-base, no-contact diagnostic',
         maximum_generations=args.generations,population=args.population,network_forwards=0,policy_optimizer_updates=0),indent=2)+'\n')


if __name__=='__main__':main()
