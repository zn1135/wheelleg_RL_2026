"""Finite seeded search of effective plant parameters; frozen control, no policy training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import hashlib

PROGRAM_SOURCE=Path(__file__).read_bytes()

import numpy as np


def encode(p):
    values=np.r_[p['static_nm'],p['kinetic_nm'],np.log(np.maximum(p['damping_nms_rad'],.1)),
                 p['bias_delta_nm'],p['length_stiffness_delta_nm_rad'],p.get('cross_stiffness_delta_nm_rad',0),
                 p.get('length_quadratic_delta_nm_rad2',0),np.log(np.maximum(np.broadcast_to(p.get('friction_recovery_s',0),(2,)),.01))]
    if p.get('seal_friction_route')=='dahl':
        fk=p['seal_kinetic_n']
        values=np.r_[values,p.get('seal_force_bias_n',0)+fk,fk,p['seal_static_n']-fk,
                     np.log(p['seal_bristle_stiffness_n_m']),np.log(max(p['seal_damping_ns_m'],1))]
    return values


def decode(x,seed):
    p=dict(seed)
    p.update(static_nm=x[:2].tolist(),kinetic_nm=np.minimum(x[2:4],x[:2]).tolist(),
             damping_nms_rad=np.exp(x[4:6]).tolist(),bias_delta_nm=x[6:8].tolist(),
             length_stiffness_delta_nm_rad=float(x[8]),cross_stiffness_delta_nm_rad=float(x[9]),
             length_quadratic_delta_nm_rad2=float(x[10]),friction_recovery_s=np.exp(x[11:13]).tolist())
    if len(x)>13:
        p.update(seal_force_bias_n=float(np.clip(x[13]-x[14],-100,150)),seal_kinetic_n=float(x[14]),
                 seal_static_n=float(x[14]+x[15]),seal_bristle_stiffness_n_m=float(np.exp(x[16])),
                 seal_damping_ns_m=float(np.exp(x[17])))
    return p


def search_score(trial):
    values=[]
    for branch in trial['branches'].values():
        score=np.hypot(branch['length_rmse_mm'],branch['angle_rmse_deg']/.2)
        score+=.1*(branch['length_max_error_mm']/3+branch['angle_max_error_deg']/.5)
        values.append(float(score)+(1000 if branch['guard_stop_s'] is not None else 0))
    return max(values)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--seed-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=4)
    parser.add_argument('--population',type=int,default=40)
    parser.add_argument('--seconds',type=float,default=35.)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((args.seed_run/'report.json').read_text())
    for trial in source['trials']:trial['search_score']=search_score(trial)
    best=min(source['trials'],key=lambda p:p['search_score']);seed=best['parameters'];center=encode(seed)
    spread=np.array([.25,.25,.2,.2,.3,.3,.2,.2,2,1.5,7,.5,.5])
    if seed.get('seal_friction_route')=='dahl':
        spread[:4]=0.;spread[11:13]=0.  # Keep the constant motor-bearing resistance and inactive aging terms fixed.
    low=np.array([.1,.1,0,0,np.log(.1),np.log(.1),-5,-5,-10,-10,-10,np.log(.05),np.log(.05)])
    high=np.array([8,8,6,6,np.log(30),np.log(30),5,5,20,10,80,np.log(30),np.log(30)])
    if seed.get('seal_friction_route')=='dahl':
        spread=np.r_[spread,12,12,12,.5,.5]
        low=np.r_[low,-100,1,0,np.log(20000),np.log(1)]
        high=np.r_[high,400,200,150,np.log(1000000),np.log(15000)]
    active=np.flatnonzero(spread>0);covariance=np.diag(spread[active]**2)
    worker=Path(__file__).with_name('replay_length_response_gpu.py');worker_source=worker.read_bytes()
    (args.output/Path(__file__).name).write_bytes(PROGRAM_SOURCE)
    (args.output/'worker-source.py').write_bytes(worker_source)
    rng=np.random.default_rng(20261109);history=[];total=0
    for generation in range(args.rounds):
        vectors=np.tile(center,(args.population,1))
        vectors[:,active]+=rng.multivariate_normal(np.zeros(len(active)),covariance,size=args.population)
        vectors=np.clip(vectors,low,high)
        vectors[0]=encode(best['parameters'])
        ps=[decode(v,seed) for v in vectors]
        parameters=args.output/f'parameters-{generation:02d}.json';parameters.write_text(json.dumps(ps,indent=2)+'\n')
        run=args.output/f'round-{generation:02d}'
        assert worker.read_bytes()==worker_source,'Replay source changed during search'
        with (args.output/f'round-{generation:02d}.log').open('w') as log:
            subprocess.run([sys.executable,'-m','scripts.agent.real2sim.replay_length_response_gpu','--inputs',str(args.inputs),
                            '--parameters',str(parameters),'--seconds',str(args.seconds),'--output',str(run)],stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((run/'report.json').read_text());total+=result['new_gpu_environment_steps']
        assert result.get('program_sha256')==hashlib.sha256(worker_source).hexdigest()
        for trial in result['trials']:trial['search_score']=search_score(trial)
        ranked=sorted(result['trials'],key=lambda p:p['search_score'])
        if generation==0 or ranked[0]['search_score']<best['search_score']:best=ranked[0]
        elite=np.array([encode(v['parameters']) for v in ranked[:max(4,args.population//6)]])
        center=.5*encode(best['parameters'])+.5*elite.mean(0)
        empirical=np.cov(elite[:,active],rowvar=False,bias=True)
        covariance=.65*covariance+.35*empirical
        summary=dict(generation=generation,best=best,round_best_search_score=ranked[0]['search_score'],
                     score_definition='worst branch: hypot(length_RMSE_mm,angle_RMSE_deg/.2)+.1*(length_max_mm/3+angle_max_deg/.5)',
                     seconds=args.seconds,new_gpu_environment_steps=total)
        history.append(summary)
        (args.output/'progress.json').write_text(json.dumps(history,indent=2)+'\n')
        (args.output/'best-parameters.json').write_text(json.dumps([best['parameters']],indent=2)+'\n')
        print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
