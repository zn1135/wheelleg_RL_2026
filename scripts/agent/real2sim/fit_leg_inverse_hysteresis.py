"""Finite CPU-only measured-path Dahl fits for GPU seeds, never free-replay acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.optimize import least_squares
import warp as wp

from scripts.agent.real2sim import fit_leg_inverse_curve as curve

wp.init()


@wp.kernel
def dahl_path(q:wp.array2d(dtype=wp.float64),v:wp.array2d(dtype=wp.float64),
              parameters:wp.array2d(dtype=wp.float64),initial:wp.array(dtype=wp.float64),
              output:wp.array2d(dtype=wp.float64),loss:wp.array2d(dtype=wp.float64)):
    motor=wp.tid()
    state=initial[motor]
    static=parameters[motor,0];positive=parameters[motor,1];negative=parameters[motor,2]
    stiffness=parameters[motor,3];scale=parameters[motor,4]
    output[0,motor]=state;loss[0,motor]=wp.float64(0.)
    for i in range(1,q.shape[0]):
        delta=q[i,motor]-q[i-1,motor]
        direction=wp.float64(0.)
        if delta>wp.float64(0.):direction=wp.float64(1.)
        elif delta<wp.float64(0.):direction=wp.float64(-1.)
        dynamic=positive
        if v[i-1,motor]<wp.float64(0.):dynamic=negative
        capacity=dynamic+(static-dynamic)*wp.exp(-wp.pow(v[i-1,motor]/scale,wp.float64(2.)))
        factor=wp.exp(-stiffness*wp.abs(delta)/capacity)
        target=direction*capacity
        updated=target+(state-target)*factor
        work=capacity*wp.abs(delta)+direction*(state-target)*capacity/stiffness*(wp.float64(1.)-factor)
        stored=(updated*updated-state*state)/(wp.float64(2.)*stiffness)
        loss[i,motor]=work-stored
        output[i,motor]=updated;state=updated


class Paths:
    def __init__(self,datasets,side):
        self.side=side;self.paths=[]
        sl=slice(2*side,2*side+2)
        for data in datasets:
            q=np.ascontiguousarray(data['qpos'][:,curve.MOTOR_INDEX[sl]])
            v=np.ascontiguousarray(data['velocity'][:,sl])
            item=dict(data=data,q=wp.array(q,dtype=wp.float64,device='cpu'),v=wp.array(v,dtype=wp.float64,device='cpu'),
                      output=wp.zeros(q.shape,dtype=wp.float64,device='cpu'),loss=wp.zeros(q.shape,dtype=wp.float64,device='cpu'))
            self.paths.append(item)
        self.parameters=wp.zeros((2,5),dtype=wp.float64,device='cpu');self.initial=wp.zeros(2,dtype=wp.float64,device='cpu')

    def evaluate(self,p):
        sl=slice(2*self.side,2*self.side+2);outputs=[]
        parameters=np.column_stack((p['static'],p['positive'],p['negative'],p['stiffness'],p['scale']))
        self.parameters.assign(parameters)
        for path in self.paths:
            data=path['data'];side=self.side
            force=np.clip(p['reference']+p['spring_k']*(data['compression'][:,side]-.005)-p['spring_c']*data['qvel'][:,curve.SLIDE_INDEX[side]],0.,400.)
            basis=data['spring_basis_nm_per_n'][:,sl,side]
            spring_load=basis*force[:,None]
            # Initial preload comes only from the first stationary pose/request.
            static_target=data['drive_torque_nm'][0,sl]-data['holding_torque_nm'][0,sl]+data['spring_basis_nm_per_n'][0,sl,side]*data['base_spring_force_n'][0,side]
            initial=np.clip(static_target-basis[0]*force[0],-p['static'],p['static'])
            self.initial.assign(initial)
            wp.launch(dahl_path,dim=2,inputs=[path['q'],path['v'],self.parameters,self.initial,path['output'],path['loss']],device='cpu')
            state=path['output'].numpy().copy();dissipation=path['loss'].numpy().copy()
            prediction=spring_load+state+data['velocity'][:,sl]*p['damping']
            outputs.append(dict(data=data,prediction=prediction,force=force,state=state,minimum_dissipation=float(np.min(dissipation)),
                                total_dissipation=np.sum(dissipation,axis=0),initial=initial,initial_unclipped=static_target-basis[0]*force[0]))
        return outputs


def encode(seed,side,dynamic):
    sl=slice(side*2,side*2+2)
    static=np.asarray(seed['friction_nm'][sl]);stiffness=np.array([200.,200.]);b=np.maximum(seed['damping_nms_rad'][sl],.02)
    base=np.r_[seed['spring_reference_force_n'][side]/150.,seed['spring_stiffness_n_m'][side]/1000.,seed['spring_damping_ns_m'][side]/100.,np.log(static),np.log(stiffness),b]
    if dynamic:
        pos=np.clip(np.array(seed['motor_dynamic_positive_nm'][sl])/static,.05,1.)
        neg=np.clip(np.array(seed['motor_dynamic_negative_nm'][sl])/static,.05,1.)
        return np.r_[base,pos,neg,np.log([.03,.03])]
    return base


def decode(x,dynamic):
    p=dict(reference=x[0]*150.,spring_k=x[1]*1000.,spring_c=x[2]*100.,static=np.exp(x[3:5]),
           stiffness=np.exp(x[5:7]),damping=x[7:9])
    p['positive']=p['static']*x[9:11] if dynamic else p['static']
    p['negative']=p['static']*x[11:13] if dynamic else p['static']
    p['scale']=np.exp(x[13:15]) if dynamic else np.array([.03,.03])
    return p


def bounds(dynamic):
    low=np.r_[0.,0.,0.,np.log([.2,.2]),np.log([2.,2.]),[0.,0.]]
    high=np.r_[2.,5.,5.,np.log([20.,20.]),np.log([3000.,3000.]),[10.,10.]]
    if dynamic:return np.r_[low,[.02]*4,np.log([.005]*2)],np.r_[high,[1.]*4,np.log([.3]*2)]
    return low,high


def evaluate_metrics(outputs,side):
    sl=slice(side*2,side*2+2);result={}
    for o in outputs:
        data=o['data'];error=o['prediction']-data['target'][:,sl]
        mask=np.ones(error.shape,bool);mask[:10]=mask[-10:]=False
        moving=(abs(data['velocity'][:,sl])>.03)&mask
        result[data['name']]=dict(all_rmse_nm=np.sqrt(np.mean(error[10:-10]**2,axis=0)).tolist(),
             moving_rmse_nm=[float(np.sqrt(np.mean(error[moving[:,i],i]**2))) for i in range(2)],
             initial_state_nm=o['initial'].tolist(),initial_unclipped_nm=o['initial_unclipped'].tolist(),
             minimum_discrete_dissipation_j=o['minimum_dissipation'],total_dissipation_j=o['total_dissipation'].tolist(),
             force_range_n=[float(o['force'].min()),float(o['force'].max())])
    return result


def as_profile(left,right):
    profile=dict(spring_reference_force_n=[],spring_stiffness_n_m=[],spring_damping_ns_m=[],spring_force_limit_n=[400.,400.],
                 friction_nm=[],stiffness_nm_rad=[],motor_dynamic_positive_nm=[],motor_dynamic_negative_nm=[],
                 motor_velocity_scale_rad_s=[],damping_nms_rad=[],armature_kg_m2=[.015795]*4)
    for p in (left,right):
        for field,source in [('spring_reference_force_n','reference'),('spring_stiffness_n_m','spring_k'),('spring_damping_ns_m','spring_c')]:profile[field].append(float(p[source]))
        for field,source in [('friction_nm','static'),('stiffness_nm_rad','stiffness'),('motor_dynamic_positive_nm','positive'),('motor_dynamic_negative_nm','negative'),('motor_velocity_scale_rad_s','scale'),('damping_nms_rad','damping')]:profile[field].extend(np.asarray(p[source]).tolist())
    return profile


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=Path,required=True)
    parser.add_argument('--original',type=Path,required=True);parser.add_argument('--max-evaluations-per-side',type=int,default=300)
    args=parser.parse_args();assert 40<=args.max_evaluations_per_side<=300
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    datasets=[curve.load('wide',curve.ROOT/'inverse-preload-hypothesis'),curve.load('original',args.original),
              curve.load('slow',curve.ROOT/'inverse-near-slow'),curve.load('fast',curve.ROOT/'inverse-near-fast')]
    seed=json.loads(args.seed.read_text())[0];runs={};selected={}
    for side in range(2):
        train=Paths(datasets[:2],side);all_paths=Paths(datasets,side)
        budget=args.max_evaluations_per_side;selected[side]={}
        for dynamic in (False,True):
            initial=encode(seed,side,dynamic);lo,hi=bounds(dynamic)
            initial=np.clip(initial,lo+1e-8,hi-1e-8)
            if dynamic and 'constant' in selected[side]:
                c=selected[side]['constant'];initial[:9]=np.r_[c['reference']/150.,c['spring_k']/1000.,c['spring_c']/100.,np.log(c['static']),np.log(c['stiffness']),c['damping']]
            starts=[initial]
            if dynamic:
                starts.append(encode(seed,side,True))
                slow=encode(seed,side,True);slow[:3]=[1.,2.,.5];slow[3:5]=np.log([12.,12.]);slow[5:7]=np.log([60.,60.]);slow[13:15]=np.log([.08,.08]);starts.append(slow)
                fast=encode(seed,side,True);fast[:3]=[.8,2.,1.];fast[3:5]=np.log([12.,12.]);fast[5:7]=np.log([600.,600.]);fast[13:15]=np.log([.02,.02]);starts.append(fast)
            attempts=[];best=None
            for attempt,initial in enumerate(starts):
                limit=min(60,budget//4) if not dynamic else max(1,budget//(len(starts)-attempt))
                call_count=[0]
                def residual(x):
                    call_count[0]+=1
                    outputs=train.evaluate(decode(x,dynamic));values=[]
                    for o in outputs:
                        e=o['prediction']-o['data']['target'][:,side*2:side*2+2]
                        # Equal trajectory/joint weight. At-rest samples constrain state evolution.
                        values.append(e[10:-10].ravel()/np.sqrt(len(e)-20))
                    return np.concatenate(values)
                result=least_squares(residual,np.clip(initial,lo+1e-8,hi-1e-8),bounds=(lo,hi),loss='soft_l1',f_scale=.02,max_nfev=limit,
                                     ftol=2e-6,xtol=2e-6,gtol=2e-6,diff_step=1e-4)
                budget-=result.nfev
                attempts.append(dict(start=attempt,nfev=result.nfev,residual_calls_including_jacobian=call_count[0],cost=result.cost))
                if best is None or result.cost<best.cost:best=result
            result=best;p=decode(result.x,dynamic);label='dynamic' if dynamic else 'constant';selected[side][label]=p
            evaluated=all_paths.evaluate(p)
            runs[f'side{side}-{label}']=dict(parameters={k:np.asarray(v).tolist() for k,v in p.items()},
                   nfev=sum(a['nfev'] for a in attempts),attempts=attempts,status=result.status,cost=result.cost,optimality=result.optimality,
                   metrics=evaluate_metrics(evaluated,side),fitting_datasets=['wide','original'],fixed_development_crosschecks=['slow','fast'])
            arrays={}
            for o in evaluated:
                for key in ('prediction','force','state'):arrays[o['data']['name']+'_'+key]=o[key]
            np.savez_compressed(args.output/f'side{side}-{label}.npz',**arrays)
            print(json.dumps(dict(side=side,model=label,nfev=runs[f'side{side}-{label}']['nfev'],cost=result.cost,metrics=runs[f'side{side}-{label}']['metrics'])),flush=True)
        assert sum(runs[f'side{side}-{name}']['nfev'] for name in ('constant','dynamic'))<=args.max_evaluations_per_side
    profiles=[as_profile(selected[0][label],selected[1][label]) for label in ('constant','dynamic')]
    (args.output/'parameters.json').write_text(json.dumps(profiles,indent=2)+'\n')
    report=dict(scope='measured-path inverse fit for GPU seeds; no free replay and no physical parameter acceptance',
         recurrence='exact displacement-domain Dahl; g(v)=Fd_direction+(Fs-Fd_direction)*exp(-(v/vs)^2)',
         initial_state='only first pose and recorded initial request/static load; clipped to static capacity',
         fit_budget_per_side=args.max_evaluations_per_side,environment_steps=0,policy_optimizer_updates=0,
         implementation_device='Warp CPU; CUDA_VISIBLE_DEVICES empty',wall_time_s=time.monotonic()-start,
         runtime_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runs=runs,
         limitations=['Measured positions advance friction only during inverse fitting; GPU seed still requires free integration',
                      'Initial inverse load depends on nominal mass and support assumptions',
                      'Derivative smoothing is 110ms and can affect short start transients'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
