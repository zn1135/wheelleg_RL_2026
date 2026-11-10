"""Bounded curve search by free-response error; no policy learning or production update."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def score(trial,objective='length_angle'):
    r=trial['branches']['recorded_torque'];c=trial['branches']['recorded_target']
    if objective=='length':
        return float(r['length_rmse_mm'])+1000*(r['guard_stop_s'] is not None)
    value=np.hypot(r['length_rmse_mm'],r['angle_rmse_deg']/.2)
    value+=.1*(r['length_max_error_mm']/3+r['angle_max_error_deg']/.5)
    value+=.25*max(c['length_rmse_mm']-5,0)+2*max(c['angle_rmse_deg']-.5,0)
    return float(value)+1000*sum(x['guard_stop_s'] is not None for x in (r,c))


def encode(p):
    return np.r_[p['spring_curve_force_n'],np.log(p['seal_damping_ns_m']),p['static_nm']]


def decode(x,seed):
    p=dict(seed);p['spring_curve_force_n']=np.maximum.accumulate(x[:5]).tolist()
    p['seal_damping_ns_m']=float(np.exp(x[5]))
    p['static_nm']=x[6:8].tolist();p['kinetic_nm']=x[6:8].tolist()
    return p


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--seed-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--population',type=int,default=16)
    parser.add_argument('--objective',choices=('length_angle','length'),default='length_angle')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    trials=json.loads((args.seed_run/'report.json').read_text())['trials']
    baseline=next(t['parameters'] for t in trials if 'spring_curve_force_n' not in t['parameters'])
    evaluate=lambda t:score(t,args.objective)
    best=min((t for t in trials if 'spring_curve_force_n' in t['parameters']),key=evaluate)
    seed=best['parameters'];center=encode(seed);spread=np.r_[np.full(5,8.),.3,.18,.18]
    if args.objective=='length':
        seed=dict(seed,spring_curve_interpolation='pchip');best=dict(best,parameters=seed)
        spread=np.r_[np.full(5,2.),.15,.1,.1]
    low=np.r_[np.zeros(5),np.log(10.),.05,.05];high=np.r_[np.full(5,600.),np.log(10000.),5.,5.]
    worker=Path(__file__).with_name('replay_length_response_gpu.py');source=worker.read_bytes()
    helper=Path(__file__).with_name('nonlinear_spring.py');helper_source=helper.read_bytes()
    (args.output/'worker.py').write_bytes(source);(args.output/helper.name).write_bytes(helper_source)
    (args.output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    rng=np.random.default_rng(20261110);history=[];steps=0
    for generation in range(args.rounds):
        x=np.clip(center+rng.normal(size=(args.population-2,8))*spread,low,high)
        ps=[baseline,best['parameters']]+[decode(row,seed) for row in x]
        path=args.output/f'parameters-{generation:02d}.json';path.write_text(json.dumps(ps,indent=2)+'\n')
        run=args.output/f'round-{generation:02d}'
        assert worker.read_bytes()==source and helper.read_bytes()==helper_source
        with (args.output/f'round-{generation:02d}.log').open('w') as log:
            subprocess.run([sys.executable,'-m','scripts.agent.real2sim.replay_length_response_gpu',
                '--inputs',str(args.inputs),'--parameters',str(path),'--seconds','35','--output',str(run)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((run/'report.json').read_text())
        assert result['program_sha256']==hashlib.sha256(source).hexdigest()
        (run/helper.name).write_bytes(helper_source)
        (run/'auxiliary-source.json').write_text(json.dumps({str(helper):hashlib.sha256(helper_source).hexdigest()})+'\n')
        steps+=result['new_gpu_environment_steps']
        ranked=sorted((t for t in result['trials'] if 'spring_curve_force_n' in t['parameters']),key=evaluate)
        if evaluate(ranked[0])<evaluate(best):best=ranked[0]
        elite=np.array([encode(t['parameters']) for t in ranked[:4]])
        center=.6*encode(best['parameters'])+.4*elite.mean(0)
        spread=np.maximum(spread*.65,elite.std(0))
        record=dict(generation=generation,objective=args.objective,best_score=evaluate(best),best=best,
            baseline_score=evaluate(result['trials'][0]),new_gpu_environment_steps=steps)
        history.append(record);(args.output/'progress.json').write_text(json.dumps(history,indent=2)+'\n')
        (args.output/'best-parameters.json').write_text(json.dumps([best['parameters']],indent=2)+'\n')
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
