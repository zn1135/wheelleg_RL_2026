"""Finite compression-dependent resistance inverse fits; CPU only and no free replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from scripts.agent.real2sim import fit_leg_inverse_curve as curve


def design(data,side,length):
    sl=slice(side*2,side*2+2);v=data['velocity'][:,sl];basis=data['spring_basis_nm_per_n'][:,sl,side]
    compression=data['compression'][:,side]
    scale=np.ones(len(v)) if length is None else np.exp(-compression/length)
    columns=[basis,basis*((compression-.005)/.09)[:,None],basis*(-data['qvel'][:,curve.SLIDE_INDEX[side]]/.1)[:,None]]
    for component in ((v>0)*scale[:,None],-(v<0).astype(float)*scale[:,None],v):
        for i in range(2):
            c=np.zeros_like(v);c[:,i]=component[:,i];columns.append(c)
    matrix=np.stack(columns,axis=-1);target=data['target'][:,sl]
    mask=abs(v)>.03;mask[:10]=mask[-10:]=False
    weight=np.broadcast_to(1./np.sqrt(np.maximum(mask.sum(0),1))[None,:],mask.shape)
    return matrix,target,mask,weight


def decode(x,length):
    return dict(spring_reference_force_n=float(x[0]),spring_stiffness_n_m=float(x[1]/.09),spring_damping_ns_m=float(x[2]*10.),
                dynamic_positive_at_zero_compression_nm=x[3:5].tolist(),dynamic_negative_at_zero_compression_nm=x[5:7].tolist(),
                damping_nms_rad=x[7:9].tolist(),compression_decay_length_m=length)


def evaluate(data,side,x,length):
    matrix,target,mask,_=design(data,side,length);prediction=matrix@x;error=prediction-target;sl=slice(side*2,side*2+2)
    force=x[0]+x[1]/.09*(data['compression'][:,side]-.005)-x[2]*10.*data['qvel'][:,curve.SLIDE_INDEX[side]]
    resistance=target-data['spring_basis_nm_per_n'][:,sl,side]*force[:,None];power=resistance*data['velocity'][:,sl]
    total=np.zeros_like(mask,bool);total[10:-10]=True
    scale=np.ones(len(mask)) if length is None else np.exp(-data['compression'][:,side]/length)
    return dict(moving_rmse_nm=[float(np.sqrt(np.mean(error[mask[:,i],i]**2))) for i in range(2)],
                moving_p95_error_nm=[float(np.percentile(abs(error[mask[:,i],i]),95)) for i in range(2)],
                negative_resistance_work_j=np.trapezoid(np.minimum(power,0),data['time_s'],axis=0).tolist(),
                negative_power_fraction=[float(np.mean(power[mask[:,i],i]<-.01)) for i in range(2)],
                spring_force_range_n=[float(force.min()),float(force.max())],
                compression_range_m=[float(data['compression'][:,side].min()),float(data['compression'][:,side].max())],
                friction_scale_range=[float(scale.min()),float(scale.max())])


def fit(datasets,all_data,side,length):
    matrices=[];targets=[]
    for data in datasets:
        a,b,mask,w=design(data,side,length);matrices.append(a[mask]*w[mask,None]);targets.append(b[mask]*w[mask])
    a=np.concatenate(matrices);b=np.concatenate(targets);weights=np.ones(len(b));iterations=6
    # The grid changes only the geometric decay, never gain, sign, zero or mass.
    low=np.zeros(9);high=np.r_[300.,450.,100.,np.full(4,150.),np.full(2,30.)]
    for _ in range(iterations):
        result=lsq_linear(a*weights[:,None],b*weights,bounds=(low,high),tol=1e-11,max_iter=200)
        residual=a@result.x-b;scale=max(.5*np.median(abs(residual))/.6745,1e-4)
        weights=np.sqrt(np.minimum(1.,2.*scale/np.maximum(abs(residual),1e-12)))
    x=result.x;metrics={d['name']:evaluate(d,side,x,length) for d in all_data}
    train_score=float(np.sqrt(np.mean(np.concatenate([np.asarray(metrics[d['name']]['moving_rmse_nm'])**2 for d in datasets]))))
    return dict(parameters=decode(x,length),metrics=metrics,balanced_train_rmse_nm=train_score,
                normalized_design_condition=float(np.linalg.cond(a)),irls_iterations=iterations,
                optimizer_iterations=result.nit,active_bounds=result.active_mask.tolist(),
                minimum_spring_force_n=min(m['spring_force_range_n'][0] for m in metrics.values()),
                maximum_spring_force_n=max(m['spring_force_range_n'][1] for m in metrics.values()))


def profile(left,right):
    p=dict(spring_reference_force_n=[],spring_stiffness_n_m=[],spring_damping_ns_m=[],spring_force_limit_n=[400.,400.],
           friction_nm=[],motor_dynamic_positive_nm=[],motor_dynamic_negative_nm=[],damping_nms_rad=[],
           armature_kg_m2=[.015795]*4,motor_velocity_scale_rad_s=[.03]*4,motor_compression_decay_length_m=[])
    for r in (left,right):
        x=r['parameters'];positive=x['dynamic_positive_at_zero_compression_nm'];negative=x['dynamic_negative_at_zero_compression_nm']
        for key in ('spring_reference_force_n','spring_stiffness_n_m','spring_damping_ns_m'):p[key].append(x[key])
        p['motor_compression_decay_length_m'].append(x['compression_decay_length_m'] or 1e6)
        p['motor_dynamic_positive_nm'].extend(np.maximum(positive,.02).tolist());p['motor_dynamic_negative_nm'].extend(np.maximum(negative,.02).tolist())
        p['friction_nm'].extend((1.1*np.maximum(np.maximum(positive,negative),.02)).tolist());p['damping_nms_rad'].extend(x['damping_nms_rad'])
    return p


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--original',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    datasets=[curve.load('wide',curve.ROOT/'inverse-preload-hypothesis'),curve.load('original',args.original),curve.load('slow',curve.ROOT/'inverse-near-slow'),curve.load('fast',curve.ROOT/'inverse-near-fast')]
    grid=[None,.005,.01,.02,.04,.08];runs={};selected=[]
    for side in range(2):
        side_runs=[]
        for length in grid:
            key=f'side{side}-lambda{length}';runs[key]=fit(datasets[:2],datasets,side,length);side_runs.append((key,runs[key]));print(key,runs[key]['balanced_train_rmse_nm'],flush=True)
        ordered=sorted(side_runs[1:],key=lambda item:item[1]['balanced_train_rmse_nm']);selected.append(ordered)
    candidates=[profile(selected[0][rank][1],selected[1][rank][1]) for rank in range(2)]
    candidates.append(profile(runs['side0-lambdaNone'],runs['side1-lambdaNone']))
    labels=[dict(left=selected[0][rank][0],right=selected[1][rank][0]) for rank in range(2)]+[dict(left='side0-lambdaNone',right='side1-lambdaNone')]
    (args.output/'parameters.json').write_text(json.dumps(candidates,indent=2)+'\n');(args.output/'seed-sources.json').write_text(json.dumps(labels,indent=2)+'\n')
    result=dict(scope='conditional measured-path inverse fit seeds; no free simulation or physical parameter acceptance',
      formula='F=Fref+k*(x-.005)-C*v_slide; dry resistance=Fd_direction_at_x0*exp(-x/lambda); plus B*v_motor',
      initial_static_friction='unknown; moving samples only, GPU Fs0=1.1*max(Fd0 positive,Fd0 negative)',
      fitting_datasets=['wide','original'],fixed_development_crosschecks=['slow','fast'],lambda_grid_m=grid,
      environment_steps=0,policy_optimizer_updates=0,runtime_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runs=runs,
      caveats=['Resistance depends on leg geometry; this is an effective coupled approximation, not isolated motor friction',
               'Zero-compression resistance is extrapolated below the observed minimum of about3.6mm',
               'Spring curves and direction-dependent resistance remain correlated in inverse dynamics'])
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    for p in candidates:
        assert np.all(np.array(p['motor_dynamic_positive_nm'])<=p['friction_nm']) and np.all(np.array(p['motor_dynamic_negative_nm'])<=p['friction_nm'])
    verification=[]
    for side in range(2):
        for length in grid:
            run=runs[f'side{side}-lambda{length}'];p=run['parameters'];x=np.r_[p['spring_reference_force_n'],p['spring_stiffness_n_m']*.09,p['spring_damping_ns_m']/10,p['dynamic_positive_at_zero_compression_nm'],p['dynamic_negative_at_zero_compression_nm'],p['damping_nms_rad']]
            for data in datasets:
                a,_,_,_=design(data,side,length);sl=slice(side*2,side*2+2);v=data['velocity'][:,sl];s=np.ones(len(v)) if length is None else np.exp(-data['compression'][:,side]/length)
                force=x[0]+x[1]/.09*(data['compression'][:,side]-.005)-x[2]*10*data['qvel'][:,curve.SLIDE_INDEX[side]]
                direct=data['spring_basis_nm_per_n'][:,sl,side]*force[:,None]+np.where(v>0,x[3:5],-x[5:7])*s[:,None]+v*x[7:9]
                difference=float(np.max(abs(a@x-direct)));assert difference<1e-12;verification.append(difference)
    (args.output/'verification.json').write_text(json.dumps(dict(scope='load-model algebra only',maximum_reconstruction_error_nm=max(verification),checks=len(verification),environment_steps=0),indent=2)+'\n')


if __name__=='__main__':main()
