"""Fit a bounded state-dependent conservative load plus nonnegative resistance; no integration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares,lsq_linear

from scripts.agent.real2sim import fit_leg_inverse_curve as inverse
from scripts.agent.real2sim import leg_load_potential as potential


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path('artifacts/real2sim/h7-joint-response-refit-20261104-01')
    sources=[('wide',root/'inverse-preload-hypothesis'),('original',Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/inverse-seeds/inverse-original')),
             ('slow',root/'inverse-near-slow'),('fast',root/'inverse-near-fast')]
    datasets=[inverse.load(name,path) for name,path in sources]
    designs=[]
    for data in datasets:
        q=data['qpos'][:,inverse.MOTOR_INDEX];v=data['velocity'];basis=data['spring_basis_nm_per_n']
        spring=150.-50.*data['qvel'][:,inverse.SLIDE_INDEX]
        target=data['target']-np.einsum('tij,tj->ti',basis,spring)
        matrix=np.zeros((len(q),4,24))
        for column in range(16):
            coefficients=np.zeros((2,8));coefficients.flat[column]=1.
            matrix[:,:,column]=potential.evaluate(q,coefficients)[1]
        for j in range(4):
            matrix[:,j,16+j]=np.sign(v[:,j]);matrix[:,j,20+j]=v[:,j]
        moving=abs(v)>.03;moving[:10]=moving[-10:]=False
        quiet=abs(v)<.005;quiet[:10]=quiet[-10:]=False
        designs.append((data,matrix,target,moving,quiet))
    low=np.r_[np.full(16,-20.),np.full(4,.05),np.full(4,.05)]
    high=np.r_[np.full(16,20.),np.full(4,10.),np.full(4,10.)]
    results=[];profiles=[]
    for regularization in (.01,.03,.1):
        # 只用移动样本初始化；静止负载是摩擦区间，不能强行等于零。
        aa=[];bb=[]
        for _,matrix,target,moving,_ in designs:
            w=np.broadcast_to(1./np.sqrt(moving.sum(0)),moving.shape)
            aa.append(matrix[moving]*w[moving,None]);bb.append(target[moving]*w[moving])
        a=np.concatenate(aa);b=np.concatenate(bb)
        initial=lsq_linear(a,b,bounds=(low,high)).x
        def residual(x):
            values=[]
            for _,matrix,target,moving,quiet in designs:
                error=matrix@x-target
                values.append(error[moving]*np.broadcast_to(1./np.sqrt(moving.sum(0)),moving.shape)[moving])
                passive=matrix[:,:,:16]@x[:16]-target
                excess=np.maximum(abs(passive)-x[16:20],0.)
                values.append(.5*excess[quiet]*np.broadcast_to(1./np.sqrt(quiet.sum(0)),quiet.shape)[quiet])
            return np.r_[np.concatenate(values),regularization*x[:16]]
        fit=least_squares(residual,initial,bounds=(low,high),max_nfev=150)
        x=fit.x;metrics={}
        for data,matrix,target,moving,quiet in designs:
            error=matrix@x-target
            metrics[data['name']]=dict(moving_rmse_nm=[float(np.sqrt(np.mean(error[moving[:,j],j]**2))) for j in range(4)])
        base=dict(load_potential_coefficients_nm=x[:16].reshape(2,8).tolist(),friction_nm=x[16:20].tolist(),
                  damping_nms_rad=x[20:24].tolist(),armature_kg_m2=[.015795]*4,
                  spring_reference_force_n=[150.]*2,spring_stiffness_n_m=[0.]*2,spring_damping_ns_m=[50.]*2,spring_force_limit_n=[600.]*2)
        for static_ratio in (1.,1.5,2.5):
            profile=dict(base);profile['friction_nm']=(x[16:20]*static_ratio).tolist()
            profile.update(motor_dynamic_positive_nm=x[16:20].tolist(),motor_dynamic_negative_nm=x[16:20].tolist(),motor_velocity_scale_rad_s=[.02]*4)
            profiles.append(profile)
        results.append(dict(regularization=regularization,parameters=base,metrics=metrics,nfev=fit.nfev))
    rng=np.random.default_rng(7);q=rng.normal(size=(100,4));c=np.asarray(results[0]['parameters']['load_potential_coefficients_nm']);_,force=potential.evaluate(q,c)
    errors=[]
    for j in range(4):
        step=np.zeros(4);step[j]=1e-6
        derivative=(potential.evaluate(q+step,c)[0]-potential.evaluate(q-step,c)[0])/2e-6
        errors.append(float(np.max(abs(derivative-force[:,j]))))
    assert max(errors)<1e-7
    (args.output/'parameters.json').write_text(json.dumps(profiles,indent=2)+'\n')
    report=dict(scope='effective conservative-load correction fitted from current state; not physical component calibration or autonomous replay acceptance',
                force_uses_time=False,force_uses_future_reference=False,all_data_are_development=True,environment_steps=0,
                source_hashes={str(path/'inverse-loads.npz'):hashlib.sha256((path/'inverse-loads.npz').read_bytes()).hexdigest() for _,path in sources},
                gradient_check_max_error_nm=max(errors),results=results)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(gradient_error=max(errors),fits=[r['metrics'] for r in results]),indent=2))


if __name__=='__main__':main()
