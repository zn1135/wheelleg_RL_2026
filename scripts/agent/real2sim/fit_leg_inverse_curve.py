"""Offline joint-load fit for GPU seeds; measured states never force a replay.

The fit is conditional on the existing measured-pose inverse dynamics and nominal mass.
Static friction at rest is an interval, not zero and not independently identified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

ROOT=Path('artifacts/real2sim/h7-joint-response-refit-20261104-01')
MOTOR_INDEX=np.array([0,10,5,14])
SLIDE_INDEX=np.array([4,9])


def load(name,path):
    with np.load(path/'inverse-loads.npz') as saved:
        data={key:saved[key].copy() for key in saved.files}
    report=json.loads((path/'report.json').read_text())
    assert np.array_equal(data['motor_indices'],MOTOR_INDEX), 'Unexpected measured-pose motor order'
    assert np.isfinite(data['spring_basis_nm_per_n']).all()
    data['name']=name
    data['source']=str(path)
    data['stream_sha256']=report['stream_sha256']
    data['velocity']=data['qvel'][:,MOTOR_INDEX]
    data['compression']=np.maximum(.010-data['qpos'][:,SLIDE_INDEX],0.)
    # The stored nominal residual excludes the old zero-based spring. Add it back
    # so the fitted force uses its own reference, stiffness and damping directly.
    data['target']=data['residual_nm']+np.einsum('tij,tj->ti',data['spring_basis_nm_per_n'],data['base_spring_force_n'])
    return data


def design(data,common_spring=False,fit_damping=False,fixed_reference=None,coordinate=False):
    v=data['velocity'];n=len(v);basis=data['spring_basis_nm_per_n']
    transform=np.kron(np.eye(2),np.array([[.5,.5],[.5,-.5]]))
    if coordinate:
        v=v@transform.T
        basis=np.einsum('tij,ik->tkj',basis,np.linalg.inv(transform))
        target=data['target']@np.linalg.inv(transform)
    else:target=data['target'].copy()
    columns=[];labels=[];lower=[];upper=[]
    side_count=1 if common_spring else 2
    for key,component,maximum in (
        ('spring_reference_force_n',np.ones((n,2)),300.),
        ('spring_progression_force_n',(data['compression']-.005)/.09,450.),
        ('spring_damping_scaled_n',-data['qvel'][:,SLIDE_INDEX]/.1,100.)):
        if key=='spring_reference_force_n' and fixed_reference is not None:
            target-=np.einsum('tij,tj->ti',basis,np.full((n,2),fixed_reference))
            continue
        if key=='spring_damping_scaled_n' and not fit_damping:
            target-=np.einsum('tij,tj->ti',basis,-50.*data['qvel'][:,SLIDE_INDEX])
            continue
        for side in range(side_count):
            c=np.zeros((n,4))
            for leg in (range(2) if common_spring else [side]):c+=basis[:,:,leg]*component[:,leg,None]
            columns.append(c);labels.append((key,side));lower.append(0.);upper.append(maximum)
    for key,component,maximum in (
        ('dynamic_positive_nm',(v>0).astype(float),20.),
        ('dynamic_negative_nm',-(v<0).astype(float),20.),
        ('viscous_nms_rad',v,30.)):
        for i in range(4):
            c=np.zeros((n,4));c[:,i]=component[:,i]
            columns.append(c);labels.append((key,i));lower.append(0.);upper.append(maximum)
    a=np.stack(columns,axis=-1)
    moving=np.abs(v)>.03
    moving[:10]=moving[-10:]=False
    return a,target,moving,labels,np.array(lower),np.array(upper)


def decode(x,labels,common_spring=False,fit_damping=False,fixed_reference=None):
    p=dict(spring_reference_force_n=[fixed_reference or 0.]*2,spring_stiffness_n_m=[0.]*2,
           spring_damping_ns_m=[0. if fit_damping else 50.]*2,
           motor_dynamic_positive_nm=[0.]*4,motor_dynamic_negative_nm=[0.]*4,damping_nms_rad=[0.]*4)
    for value,(key,i) in zip(x,labels):
        if key=='spring_reference_force_n':target=key;factor=1.
        elif key=='spring_progression_force_n':target='spring_stiffness_n_m';factor=1./.09
        elif key=='spring_damping_scaled_n':target='spring_damping_ns_m';factor=10.
        elif key=='viscous_nms_rad':target='damping_nms_rad';factor=1.
        else:target='motor_'+key;factor=1.
        for j in (range(2) if common_spring and key.startswith('spring') else [i]):p[target][j]=float(value*factor)
    return p


def prediction(data,p,coordinate=False):
    force=np.asarray(p['spring_reference_force_n'])+np.asarray(p['spring_stiffness_n_m'])*(data['compression']-.005)-np.asarray(p['spring_damping_ns_m'])*data['qvel'][:,SLIDE_INDEX]
    v=data['velocity'];basis=data['spring_basis_nm_per_n']
    if coordinate:
        transform=np.kron(np.eye(2),np.array([[.5,.5],[.5,-.5]]));v=v@transform.T
        basis=np.einsum('tij,ik->tkj',basis,np.linalg.inv(transform))
    friction=np.where(v>0,p['motor_dynamic_positive_nm'],-np.asarray(p['motor_dynamic_negative_nm']))+v*p['damping_nms_rad']
    return np.einsum('tij,tj->ti',basis,force)+friction,force


def evaluate(data,p,coordinate=False):
    predicted,force=prediction(data,p,coordinate)
    _,target,moving,*_=design(data,coordinate=coordinate)
    # design removes the fixed 50 Ns/m term; compare the original full target.
    target=data['target'].copy();v=data['velocity']
    if coordinate:
        transform=np.kron(np.eye(2),np.array([[.5,.5],[.5,-.5]]));target=target@np.linalg.inv(transform);v=v@transform.T
    error=predicted-target
    residual=target-np.einsum('tij,tj->ti',data['spring_basis_nm_per_n'],force) if not coordinate else None
    quiet=np.abs(v)<.005;quiet[:10]=quiet[-10:]=False
    static=[]
    for i in range(4):
        values=np.abs(residual[quiet[:,i],i]) if residual is not None else np.empty(0)
        static.append(float(np.percentile(values,99)) if len(values) else None)
    return dict(moving_rmse_nm=[float(np.sqrt(np.mean(error[moving[:,i],i]**2))) for i in range(4)],
                moving_samples=moving.sum(0).tolist(),quiet_abs_load_p99_nm=static,
                force_range_n=np.stack((force.min(0),force.max(0))).tolist(),
                mean_force_n=force.mean(0).tolist(),
                moving_error_p95_nm=[float(np.percentile(np.abs(error[moving[:,i],i]),95)) for i in range(4)])


def fit(datasets,common_spring=False,fit_damping=False,fixed_reference=None,coordinate=False,prior_weight=0.):
    arrays=[design(d,common_spring,fit_damping,fixed_reference,coordinate) for d in datasets]
    labels=arrays[0][3];a=[];b=[]
    for matrix,target,mask,*_ in arrays:
        weights=np.broadcast_to(1./np.sqrt(np.maximum(mask.sum(0),1))[None,:],mask.shape)
        a.append(matrix[mask]*weights[mask,None]);b.append(target[mask]*weights[mask])
    a=np.concatenate(a);b=np.concatenate(b)
    if prior_weight:
        for j,(key,_) in enumerate(labels):
            if key=='spring_reference_force_n':
                row=np.zeros(len(labels));row[j]=prior_weight/150.
                a=np.vstack((a,row));b=np.r_[b,prior_weight]
    robust=np.ones(len(b));x=None
    for _ in range(6):
        result=lsq_linear(a*robust[:,None],b*robust,bounds=(arrays[0][4],arrays[0][5]),tol=1e-10,max_iter=100)
        x=result.x;error=a@x-b
        # Balanced trajectories and Huber weighting protect derivative/transition outliers.
        scale=max(.5*np.median(np.abs(error))/0.6745,1e-4)
        robust=np.sqrt(np.minimum(1.,2.*scale/np.maximum(np.abs(error),1e-12)))
    p=decode(x,labels,common_spring,fit_damping,fixed_reference)
    singular=np.linalg.svd(a,compute_uv=False)
    return dict(parameters=p,reference_force_offset_from_150_n=(np.array(p['spring_reference_force_n'])-150.).tolist(),
                moving_load_fit={d['name']:evaluate(d,p,coordinate) for d in ALL},
                normalized_design_singular_values=singular.tolist(),normalized_design_condition=float(singular[0]/max(singular[-1],1e-30)),
                minimum_or_maximum_bounds=[dict(parameter=labels[i],value=float(x[i]),bound='minimum' if x[i]<arrays[0][4][i]+1e-5 else 'maximum') for i in range(len(x)) if min(x[i]-arrays[0][4][i],arrays[0][5][i]-x[i])<1e-5],
                fit_datasets=[d['name'] for d in datasets],common_spring=common_spring,fit_damping=fit_damping,
                fixed_reference_force_n=fixed_reference,coordinate_friction=coordinate,prior_weight=prior_weight)


def main():
    global ALL
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--original',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    ALL=[load('wide',ROOT/'inverse-preload-hypothesis'),load('slow',ROOT/'inverse-near-slow'),
         load('fast',ROOT/'inverse-near-fast'),load('original',args.original)]
    runs={}
    for shared in (False,True):
        for damping in (False,True):
            for fixed in (None,150.):
                key=f'joint-shared{shared}-damping{damping}-reference{fixed}'
                runs[key]=fit(ALL,shared,damping,fixed)
    for coordinate in (False,True):
        for d in ALL:
            runs[f'only-{d["name"]}-coordinate{coordinate}']=fit([d],False,True,None,coordinate)
    runs['joint-prior150']=fit(ALL,False,True,None,False,prior_weight=1.)
    runs['joint-coordinate']=fit(ALL,False,True,None,True)
    for omitted in ALL:
        runs[f'leave-out-{omitted["name"]}']=fit([d for d in ALL if d is not omitted],False,True,None)
    result=dict(scope='conditional inverse-load seed fit only; no simulation or physical parameter acceptance',
                formula='x=max(0,0.010-q_slide); F=F_reference+k*(x-0.005)-c*v_slide; positive/negative dry friction + b*v',
                initial_static_friction='unknown interval; quiet samples excluded from moving fit and used only to estimate capacity',
                source_files={d['name']:dict(path=d['source'],stream_sha256=d['stream_sha256'],inverse_sha256=hashlib.sha256((Path(d['source'])/'inverse-loads.npz').read_bytes()).hexdigest()) for d in ALL},
                limits='No force clipping occurs in fitted family; bounds permit discovery of incompatible or negative trajectories and report force range',
                environment_steps=0,policy_optimizer_updates=0,runs=runs)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    seeds=[]
    for label in ('joint-sharedFalse-dampingFalse-referenceNone','joint-prior150','joint-sharedFalse-dampingTrue-reference150.0','joint-sharedTrue-dampingFalse-referenceNone'):
        p=dict(runs[label]['parameters']);capacity=[]
        for i in range(4):
            quiet=max(r['quiet_abs_load_p99_nm'][i] or 0. for r in runs[label]['moving_load_fit'].values())
            capacity.append(max(p['motor_dynamic_positive_nm'][i],p['motor_dynamic_negative_nm'][i],quiet)*1.1+.1)
        p.update(friction_nm=capacity,motor_velocity_scale_rad_s=[.03]*4,armature_kg_m2=[.015795]*4,spring_force_limit_n=[400.]*2)
        seeds.append(p)
    (args.output/'parameters.json').write_text(json.dumps(seeds,indent=2)+'\n')
    (args.output/'seed-labels.json').write_text(json.dumps(['joint-sharedFalse-dampingFalse-referenceNone','joint-prior150','joint-sharedFalse-dampingTrue-reference150.0','joint-sharedTrue-dampingFalse-referenceNone'],indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),seeds=seeds),indent=2))


if __name__=='__main__':main()


def piecewise_fit(datasets,knots,common_spring=False,fit_damping=False):
    """Finite monotone continuous curve diagnostic, with no trajectory-time offsets."""
    arrays=[];labels=[]
    for data in datasets:
        n=len(data['velocity']);basis=data['spring_basis_nm_per_n'];v=data['velocity'];columns=[]
        current_labels=[]
        components=[np.ones_like(data['compression'])]
        components.extend([np.clip((data['compression']-low)/(high-low),0.,1.) for low,high in zip(knots[:-1],knots[1:])])
        for side in range(1 if common_spring else 2):
            for i,component in enumerate(components):
                column=np.zeros((n,4))
                for leg in (range(2) if common_spring else [side]):column+=basis[:,:,leg]*component[:,leg,None]
                columns.append(column);current_labels.append(('spring_force_increment_n',side,i))
        target=data['target'].copy()
        if fit_damping:
            for side in range(2):
                columns.append(basis[:,:,side]*(-data['qvel'][:,SLIDE_INDEX[side],None]/.1));current_labels.append(('spring_damping_scaled_n',side,None))
        else:target-=np.einsum('tij,tj->ti',basis,-50.*data['qvel'][:,SLIDE_INDEX])
        for key,component in (('dynamic_positive_nm',(v>0).astype(float)),('dynamic_negative_nm',-(v<0).astype(float)),('viscous_nms_rad',v)):
            for i in range(4):
                column=np.zeros((n,4));column[:,i]=component[:,i];columns.append(column);current_labels.append((key,i,None))
        matrix=np.stack(columns,axis=-1);mask=np.abs(v)>.03;mask[:10]=mask[-10:]=False
        weight=np.broadcast_to(1./np.sqrt(np.maximum(mask.sum(0),1))[None,:],mask.shape)
        arrays.append((matrix[mask]*weight[mask,None],target[mask]*weight[mask]));labels=current_labels
    a=np.concatenate([x[0] for x in arrays]);b=np.concatenate([x[1] for x in arrays]);robust=np.ones(len(b))
    upper=np.array([300. if k=='spring_force_increment_n' else 100. if k=='spring_damping_scaled_n' else 30. for k,*_ in labels])
    for _ in range(6):
        result=lsq_linear(a*robust[:,None],b*robust,bounds=(np.zeros(len(labels)),upper),tol=1e-10,max_iter=200);x=result.x;error=a@x-b
        scale=max(.5*np.median(abs(error))/.6745,1e-4);robust=np.sqrt(np.minimum(1.,2*scale/np.maximum(abs(error),1e-12)))
    p=dict(spring_compression_knots_m=list(knots),spring_force_knots_n=[[0.]*len(knots) for _ in range(2)],
           spring_damping_ns_m=[0. if fit_damping else 50.]*2,motor_dynamic_positive_nm=[0.]*4,motor_dynamic_negative_nm=[0.]*4,damping_nms_rad=[0.]*4)
    for value,(key,i,j) in zip(x,labels):
        if key=='spring_force_increment_n':
            for side in (range(2) if common_spring else [i]):
                for knot in range(j,len(knots)):p['spring_force_knots_n'][side][knot]+=float(value)
        elif key=='spring_damping_scaled_n':p['spring_damping_ns_m'][i]=float(value*10.)
        else:p[('damping_nms_rad' if key=='viscous_nms_rad' else 'motor_'+key)][i]=float(value)
    evaluations={}
    for data in ALL:
        force=np.column_stack([np.interp(data['compression'][:,i],knots,p['spring_force_knots_n'][i]) for i in range(2)])-p['spring_damping_ns_m']*data['qvel'][:,SLIDE_INDEX]
        v=data['velocity'];friction=np.where(v>0,p['motor_dynamic_positive_nm'],-np.asarray(p['motor_dynamic_negative_nm']))+p['damping_nms_rad']*v
        residual=data['target']-np.einsum('tij,tj->ti',data['spring_basis_nm_per_n'],force)
        error=friction-residual;mask=abs(v)>.03;mask[:10]=mask[-10:]=False
        evaluations[data['name']]=dict(moving_rmse_nm=[float(np.sqrt(np.mean(error[mask[:,i],i]**2))) for i in range(4)],
             negative_power_fraction=[float(np.mean((residual[:,i]*v[:,i])[mask[:,i]]<-.01)) for i in range(4)],force_range_n=[force.min(0).tolist(),force.max(0).tolist()])
    return dict(parameters=p,moving_load_fit=evaluations,fit_datasets=[d['name'] for d in datasets],common_spring=common_spring,fit_damping=fit_damping,
                normalized_design_condition=float(np.linalg.cond(a)),scope='conditional monotone curve diagnostic; any steep threshold may include unmodelled engagement/support/history')
