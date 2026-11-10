"""Conditional inverse-load seed fit of spring-seal memory, using a causal banded recurrence."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import least_squares

from scripts.agent.real2sim import fit_leg_inverse_curve as inverse
from scripts.agent.real2sim.elastic_joint_resistance import advance


def decode(x):
    static=np.exp(x[3]);stiffness=np.exp(x[4])
    return dict(reference=x[0]*150.,slope=x[1]*1000.,damping=max(x[2]*100.,stiffness*.0005+5.),
                static=static,stiffness=stiffness,positive=static*x[5],negative=static*x[6],scale=np.exp(x[7]),
                motor_friction=x[8:10],motor_damping=x[10:12])


def trace(q,v,p,initial):
    capacity=np.where(v[:-1]>=0,p['positive'],p['negative'])
    capacity+=(p['static']-capacity)*np.exp(-(v[:-1]/p['scale'])**2)
    delta=np.diff(q);factor=np.exp(-p['stiffness']*abs(delta)/capacity)
    rhs=np.r_[initial,np.sign(delta)*capacity*(1-factor)]
    band=np.zeros((2,len(q)));band[0]=1.;band[1,:-1]=-factor
    return solve_banded((1,0),band,rhs,check_finite=False)


def outputs(datasets,side,p):
    sl=slice(2*side,2*side+2);result=[]
    for data in datasets:
        q=data['qpos'][:,inverse.SLIDE_INDEX[side]];v=data['qvel'][:,inverse.SLIDE_INDEX[side]]
        basis=data['spring_basis_nm_per_n'][:,sl,side]
        force=np.clip(p['reference']+p['slope']*(data['compression'][:,side]-.005)-p['damping']*v,0.,600.)
        initial_force=np.clip(p['reference']+p['slope']*(data['compression'][0,side]-.005),0.,600.)
        residual=data['drive_torque_nm'][0,sl]-data['holding_torque_nm'][0,sl]+basis[0]*(data['base_spring_force_n'][0,side]-initial_force)
        initial=np.clip(np.dot(-basis[0],residual)/np.dot(basis[0],basis[0]),-p['static'],p['static'])
        state=trace(q,v,p,initial)
        motor_v=data['velocity'][:,sl]
        load=basis*(force-state)[:,None]+np.sign(motor_v)*p['motor_friction']+motor_v*p['motor_damping']
        result.append(dict(name=data['name'],load=load,force=force,state=state,target=data['target'][:,sl],
                           velocity=motor_v,initial=float(initial)))
    return result


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path('artifacts/real2sim/h7-joint-response-refit-20261104-01')
    data=[inverse.load('wide',root/'inverse-preload-hypothesis'),inverse.load('original',Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/inverse-seeds/inverse-original')),
          inverse.load('slow',root/'inverse-near-slow'),inverse.load('fast',root/'inverse-near-fast')]
    bounds=(np.r_[.2,0.,0.,np.log(20.),np.log(1000.),.01,.01,np.log(.0001),[.01]*2,[.05]*2],
            np.r_[2.,6.,10.,np.log(600.),np.log(300000.),1.,1.,np.log(.05),[8.]*2,[8.]*2])
    starts=[np.r_[1.,.75,1.,np.log(200.),np.log(10000.),.2,.6,np.log(.001),[1.5]*2,[.2]*2],
            np.r_[.85,2.,2.,np.log(300.),np.log(50000.),.3,.5,np.log(.005),[1.]*2,[.5]*2]]
    fitted=[];runs=[]
    for side in (0,1):
        side_runs=[]
        for start in starts:
            def residual(x):
                p=decode(x);values=[]
                for o in outputs(data,side,p):
                    error=o['load']-o['target'];moving=abs(o['velocity'])>.03;quiet=abs(o['velocity'])<.005
                    moving[:10]=moving[-10:]=False;quiet[:10]=quiet[-10:]=False
                    weights=np.broadcast_to(1./np.sqrt(np.maximum(moving.sum(0),1)),moving.shape)
                    values.append(error[moving]*weights[moving])
                    passive=o['load']-np.sign(o['velocity'])*p['motor_friction']-o['velocity']*p['motor_damping']
                    excess=np.maximum(abs(passive-o['target'])-p['motor_friction'],0.)
                    wq=np.broadcast_to(1./np.sqrt(np.maximum(quiet.sum(0),1)),quiet.shape)
                    values.append(.3*excess[quiet]*wq[quiet])
                return np.concatenate(values)
            fit=least_squares(residual,start,bounds=bounds,max_nfev=150,loss='soft_l1',f_scale=.1)
            p=decode(fit.x);metrics={o['name']:[float(np.sqrt(np.mean((o['load'][abs(o['velocity'][:,j])>.03,j]-o['target'][abs(o['velocity'][:,j])>.03,j])**2))) for j in range(2)] for o in outputs(data,side,p)}
            side_runs.append(dict(parameters={k:(v.tolist() if isinstance(v,np.ndarray) else float(v)) for k,v in p.items()},nfev=fit.nfev,cost=float(np.linalg.norm(residual(fit.x))),moving_rmse_nm=metrics))
            print(json.dumps(dict(side=side,nfev=fit.nfev,metrics=metrics)),flush=True)
        best=min(side_runs,key=lambda x:x['cost']);fitted.append(best['parameters']);runs.append(side_runs)
    profile=dict(friction_nm=[],damping_nms_rad=[],armature_kg_m2=[.015795]*4,spring_force_limit_n=[600.]*2)
    fields={'reference':'spring_reference_force_n','slope':'spring_stiffness_n_m','damping':'spring_damping_ns_m',
            'static':'spring_seal_static_n','stiffness':'spring_seal_stiffness_n_m','positive':'spring_seal_positive_n',
            'negative':'spring_seal_negative_n','scale':'spring_seal_velocity_scale_m_s'}
    for p in fitted:
        profile['friction_nm'].extend(p['motor_friction']);profile['damping_nms_rad'].extend(p['motor_damping'])
        for key,target in fields.items():profile.setdefault(target,[]).append(p[key])
    # 递推使用测得位移仅产生种子，正式验收必须另做GPU自由积分。
    checks=[]
    for side,p in enumerate(fitted):
        for source,o in zip(data,outputs(data,side,p)):
            q=source['qpos'][:,inverse.SLIDE_INDEX[side]];v=source['qvel'][:,inverse.SLIDE_INDEX[side]];state=o['initial'];error=0.
            for i,delta in enumerate(np.diff(q)):
                dynamic=p['positive'] if v[i]>=0 else p['negative'];limit=dynamic+(p['static']-dynamic)*np.exp(-(v[i]/p['scale'])**2)
                state,_=advance(np.array([state]),np.array([delta]),np.array([limit]),np.array([p['stiffness']]),capacity=np.array([p['static']]))
                state=float(state[0]);error=max(error,abs(state-o['state'][i+1]))
            checks.append(error)
    assert max(checks)<1e-8
    (args.output/'parameters.json').write_text(json.dumps([profile],indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(dict(scope='conditional inverse-load seed only; not free-replay acceptance',environment_steps=0,runs=runs,maximum_recurrence_error_n=max(checks)),indent=2)+'\n')


if __name__=='__main__':main()
