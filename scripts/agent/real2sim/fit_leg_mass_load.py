"""Conditional rigid-body load sensitivity, without physics integration or source-model edits."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from scripts.agent.real2sim import fit_leg_inverse_curve as inverse


GROUPS=[['lf0'],['lf1'],['lfwheel'],['lf00','lf01','lf02','lf03'],
        ['rf0'],['rf1'],['rfwheel'],['rf00','rf01','rf02','rf03']]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path('artifacts/real2sim/h7-joint-response-refit-20261104-01')
    sources=[('wide',root/'inverse-preload-hypothesis'),('original',Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/inverse-seeds/inverse-original')),
             ('slow',root/'inverse-near-slow'),('fast',root/'inverse-near-fast')]
    models=[]
    for group in [None]+GROUPS:
        model=mujoco.MjModel.from_xml_path(str(root/'input-wide/fixed-base-model.xml'))
        model.dof_armature[[0,10,5,14]]=.015795
        if group:
            ids=[model.body(name).id for name in group]
            model.body_mass[ids]*=2.;model.body_inertia[ids]*=2.
        mujoco.mj_setConst(model,mujoco.MjData(model));models.append((model,mujoco.MjData(model)))
    all_data=[]
    for name,path in sources:
        dataset=inverse.load(name,path);indices=np.arange(10,len(dataset['qpos'])-10,3);basis=[]
        model,data=models[0];mv=inverse.MOTOR_INDEX;wheel=[model.joint(n).dofadr[0] for n in ('lfwheel','rfwheel')]
        passive=[i for i in range(model.nv) if i not in list(mv)+wheel]
        for index in indices:
            q=dataset['qpos'][index];v=dataset['qvel'][index];a=dataset['qacc'][index]
            data.qpos[:]=q;data.qvel[:]=v;mujoco.mj_forward(model,data)
            mask=data.efc_type[:data.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY
            jac=data.efc_J.reshape(data.nefc,model.nv)[mask]
            tangent=np.zeros((model.nv,4));tangent[mv]=np.eye(4)
            tangent[passive]=np.linalg.lstsq(jac[:,passive],-jac[:,mv],rcond=None)[0]
            forces=[]
            for altered,work in models:
                work.qpos[:]=q;work.qvel[:]=v;work.ctrl[:]=0.;mujoco.mj_forward(altered,work)
                inertia=np.empty(altered.nv);mujoco.mj_mulM(altered,work,inertia,a)
                forces.append(tangent.T@(inertia+work.qfrc_bias))
            basis.append((np.asarray(forces)[1:]-forces[0]).T)
        a,b,moving,labels,low,high=inverse.design(dataset,fit_damping=True)
        a=a[indices];b=b[indices];moving=moving[indices];basis=np.asarray(basis)
        # 基线已含名义质量，额外列拟合质量相对名义值的差。
        a=np.concatenate((a,basis),axis=-1);low=np.r_[low,np.full(8,-.75)];high=np.r_[high,np.full(8,3.)]
        all_data.append(dict(name=name,a=a,b=b,mask=moving,labels=labels,low=low,high=high))
        np.savez_compressed(args.output/(name+'-regressor.npz'),a=a,b=b,mask=moving,indices=indices,mass_basis=basis)
    runs=[];profiles=[]
    for regularization in (.1,.5,2.):
        aa=[];bb=[]
        for d in all_data:
            w=np.broadcast_to(1./np.sqrt(np.maximum(d['mask'].sum(0),1)),d['mask'].shape)
            aa.append(d['a'][d['mask']]*w[d['mask'],None]);bb.append(d['b'][d['mask']]*w[d['mask']])
        a=np.concatenate(aa);b=np.concatenate(bb);prior=np.zeros((8,a.shape[1]));prior[:,-8:]=regularization*np.eye(8)
        result=lsq_linear(np.r_[a,prior],np.r_[b,np.zeros(8)],bounds=(all_data[0]['low'],all_data[0]['high']))
        x=result.x;profile=inverse.decode(x[:-8],all_data[0]['labels'],fit_damping=True)
        profile.update(friction_nm=np.maximum(profile['motor_dynamic_positive_nm'],profile['motor_dynamic_negative_nm']).tolist(),motor_velocity_scale_rad_s=[.02]*4,armature_kg_m2=[.015795]*4,spring_force_limit_n=[600.]*2,
                       diagnostic_body_mass_scales={body:float(1+x[-8+j]) for j,group in enumerate(GROUPS) for body in group})
        metrics={}
        for d in all_data:
            error=d['a']@x-d['b'];metrics[d['name']]=[float(np.sqrt(np.mean(error[d['mask'][:,j],j]**2))) for j in range(4)]
        runs.append(dict(regularization=regularization,parameters=profile,moving_rmse_nm=metrics));profiles.append(profile)
    (args.output/'parameters.json').write_text(json.dumps(profiles,indent=2)+'\n')
    report=dict(scope='conditional mass-load sensitivity only; masses are not measured or adopted',environment_steps=0,groups=GROUPS,runs=runs)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
