"""State-only empirical joint loads, unconstrained by gas-spring specifications; free GPU replay."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
import warp as wp

from scripts.agent.real2sim import fit_leg_repaired_loop as fit
from scripts.agent.real2sim import fit_leg_inverse_curve as inverse


@wp.kernel
def empirical_force(q:wp.array2d(dtype=wp.float32),v:wp.array2d(dtype=wp.float32),
                    mq:wp.array(dtype=wp.int32),mv:wp.array(dtype=wp.int32),
                    centers:wp.array4d(dtype=wp.float32),coeff:wp.array4d(dtype=wp.float32),
                    means:wp.array3d(dtype=wp.float32),scales:wp.array3d(dtype=wp.float32),
                    settings:wp.array2d(dtype=wp.float32),previous:wp.array2d(dtype=wp.float32),
                    memory:wp.array2d(dtype=wp.float32),travel:wp.array2d(dtype=wp.float32),
                    bias:wp.array2d(dtype=wp.float32),applied:wp.array2d(dtype=wp.float32),count:int):
    world=wp.tid()
    for j in range(4):
        delta=q[world,mq[j]]-previous[world,j]
        direction=wp.sign(delta)
        memory[world,j]=direction+(memory[world,j]-direction)*wp.exp(-wp.abs(delta)/settings[world,0])
        previous[world,j]=q[world,mq[j]]
        travel[world,j]+=wp.abs(delta)
    for side in range(2):
        denominator=float(0.00000001)
        first=coeff[world,side,0,0];second=coeff[world,side,1,0]
        sum_first=float(0.);sum_second=float(0.)
        for k in range(count):
            distance=float(0.)
            for feature in range(6):
                j=2*side+feature%2
                value=q[world,mq[j]]
                if feature>=2 and feature<4:value=v[world,mv[j]]
                elif feature>=4:value=memory[world,j]
                scaled=(value-means[world,side,feature])/scales[world,side,feature]
                difference=scaled-centers[world,side,k,feature]
                distance+=difference*difference
            weight=wp.exp(-distance/(2.*settings[world,1]*settings[world,1]))
            denominator+=weight
            sum_first+=weight*coeff[world,side,0,k+1]
            sum_second+=weight*coeff[world,side,1,k+1]
        first+=sum_first/denominator;second+=sum_second/denominator
        j=side*2
        fade=wp.exp(-(travel[world,j]+travel[world,j+1])/.05)
        applied[world,mv[j]]=-wp.clamp(first+bias[world,j]*fade,-40.,40.)
        applied[world,mv[j+1]]=-wp.clamp(second+bias[world,j+1]*fade,-40.,40.)


def memory_path(q,delta):
    state=np.zeros_like(q)
    for i in range(1,len(q)):
        change=q[i]-q[i-1]
        state[i]=np.sign(change)+(state[i-1]-np.sign(change))*np.exp(-abs(change)/delta)
    return state


def features(data,side,delta):
    sl=slice(2*side,2*side+2)
    q=data['qpos'][:,inverse.MOTOR_INDEX[sl]]
    return np.column_stack((q,data['velocity'][:,sl],memory_path(q,delta)))


def basis(x,centers,width):
    weights=np.exp(-np.sum((x[:,None]-centers[None])**2,axis=-1)/(2*width**2))
    weights/=weights.sum(1,keepdims=True)+1e-8
    return np.column_stack((np.ones(len(x)),weights))


def train(data,delta,width,ridge):
    result=dict(delta_rad=delta,width=width,ridge=ridge,legs=[],train_torque_rmse_nm=[])
    for side in (0,1):
        x=[features(d,side,delta) for d in data]
        joined=np.concatenate(x)
        mean=joined.mean(0);scale=np.maximum(joined.std(0),[.05,.05,.03,.03,.5,.5])
        z=(joined-mean)/scale
        # 最远点选64个状态中心；模型输入没有时间、轨迹ID或未来关节角。
        chosen=[0];dist=np.full(len(z),np.inf)
        for _ in range(63):
            dist=np.minimum(dist,np.sum((z-z[chosen[-1]])**2,axis=1))
            chosen.append(int(np.argmax(dist)))
        centers=z[chosen]
        matrices=[];targets=[]
        for d,a in zip(data,x):
            matrices.append(basis((a-mean)/scale,centers,width)/np.sqrt(len(a)))
            targets.append(d['target'][:,2*side:2*side+2]/np.sqrt(len(a)))
        matrix=np.concatenate(matrices);target=np.concatenate(targets)
        coef=np.linalg.solve(matrix.T@matrix+ridge*np.eye(matrix.shape[1]),matrix.T@target)
        residual=matrix@coef-target
        result['train_torque_rmse_nm'].append(np.sqrt(np.mean(residual**2,axis=0)*sum(len(a) for a in x)/len(x)).tolist())
        result['legs'].append(dict(mean=mean.tolist(),scale=scale.tolist(),centers=centers.tolist(),coefficients=coef.T.tolist()))
    return result


def factory(models,inputs,output):
    def prepare(model,gd,ncase):
        assert ncase==len(models)
        mq=np.array([model.joint(n).qposadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]])
        mv=np.array([model.joint(n).dofadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]])
        assert np.array_equal(mq,inverse.MOTOR_INDEX)
        initial=inputs['saved']['qpos'][0,0];velocity=inputs['saved']['qvel'][0,0]
        data=mujoco.MjData(model);data.qpos[:]=initial;data.qvel[:]=velocity
        target,info=fit.prior.elastic.initial_preload(model,data,inputs['saved']['submitted_model_torque'][0,0,:4],initial_spring_force=np.zeros(2))
        offsets=[]
        for candidate in models:
            predicted=[]
            for side,leg in enumerate(candidate['legs']):
                sl=slice(2*side,2*side+2)
                x=np.r_[initial[mq[sl]],velocity[mv[sl]],[0.,0.]]
                phi=basis(((x-np.array(leg['mean']))/leg['scale'])[None],np.array(leg['centers']),candidate['width'])
                predicted.extend((phi@np.array(leg['coefficients']).T)[0])
            offsets.append((target-predicted).tolist())
        fit.search.write_json(output/'initial-load-offsets.json',dict(initial_rule='first pose and first request only; fixed 0.05rad travel fade',
            offsets_nm=offsets,static_load_info=info))
        def duplicate(values):return np.concatenate((np.array(values,dtype=np.float32),)*2,axis=0)
        centers=wp.array(duplicate([[p['centers'] for p in c['legs']] for c in models]))
        coeff=wp.array(duplicate([[p['coefficients'] for p in c['legs']] for c in models]))
        means=wp.array(duplicate([[p['mean'] for p in c['legs']] for c in models]))
        scales=wp.array(duplicate([[p['scale'] for p in c['legs']] for c in models]))
        settings=wp.array(duplicate([[c['delta_rad'],c['width']] for c in models]))
        previous=wp.array(np.broadcast_to(initial[mq],(ncase*2,4)).copy().astype(np.float32))
        memory=wp.zeros((ncase*2,4));travel=wp.zeros_like(memory);bias=wp.array(duplicate(offsets))
        iq,iv=wp.array(mq,dtype=wp.int32),wp.array(mv,dtype=wp.int32)
        def launch():
            wp.launch(empirical_force,dim=ncase*2,inputs=[gd.qpos,gd.qvel,iq,iv,centers,coeff,means,scales,settings,
                previous,memory,travel,bias,gd.qfrc_applied,64])
        return launch
    return prepare


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    protected_paths=[Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'),Path('deployment/H7_RL_ros/imcalib/config/machine_config.c'),Path(fit.prior.replay.training_spring.__file__)]
    protected={str(p):fit.audit.digest(p) for p in protected_paths}
    paths=dict(original=Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/inverse-seeds/inverse-original'),
               wide=Path('artifacts/real2sim/h7-joint-response-refit-20261104-01/inverse-preload-hypothesis'))
    data=[inverse.load(n,p) for n,p in paths.items()]
    inputs={}
    for name in tuple(fit.REFERENCES):
        path=Path('artifacts/real2sim/h7-static-constrained-joint-refit-20261105-02/inputs')/name
        fit.REFERENCES[name]=(str(path),str(path/'rc-inputs.npz'));inputs[name]=fit.input_data(name)
    models=[train(data,delta,width,ridge) for delta in (.005,.03) for width in (1.,2.) for ridge in (1e-6,1e-3)]
    fit.search.write_json(args.output/'empirical-models.json',models)
    source_model=mujoco.MjModel.from_xml_path(inputs['wide']['report']['source_model'])
    nominal_damping=[float(source_model.dof_damping[source_model.joint(n).dofadr[0]]) for n in fit.prior.replay.mirror.MOTORS[:4]]
    profile=dict(friction_nm=[0.]*4,motor_dynamic_positive_nm=[0.]*4,motor_dynamic_negative_nm=[0.]*4,
        motor_velocity_scale_rad_s=[.03]*4,damping_nms_rad=nominal_damping,spring_reference_force_n=[0.]*2,
        spring_stiffness_n_m=[0.]*2,spring_damping_ns_m=[0.]*2,spring_friction_n=[0.]*2,
        spring_force_limit_n=[600.]*2,armature_kg_m2=[.015795]*4)
    fit.wp.init();reports={};total=0
    for name in ('original','wide'):
        directory=args.output/'selection'/name
        _,report=fit.run([profile]*len(models),inputs[name],directory,solver_profile='ordered',extra_load_factory=factory(models,inputs[name],directory))
        reports[name]=report;total+=report['new_gpu_environment_steps']
    scores=[]
    for index in range(len(models)):
        values=[];allowed=True
        for report in reports.values():
            r=report['results'][index]['recorded_torque']
            allowed &= r['original_assembly_branch'] and r['limit_excess_deg']<=.1
            values.extend(np.square(r['rmse_deg']))
        scores.append(float(np.sqrt(np.mean(values))) if allowed else 1e6)
    best=int(np.argmin(scores));selected=models[best]
    fit.search.write_json(args.output/'selection.json',dict(index=best,recorded_torque_scores_deg=scores,
        selection='Whole candidate, original+wide torque-input joint RMSE only; feedback cannot hide load errors'))
    fit.search.write_json(args.output/'selected-empirical-model.json',selected)
    final={}
    for name in inputs:
        directory=args.output/('final-'+name)
        _,report=fit.run([profile],inputs[name],directory,full=True,solver_profile='ordered',extra_load_factory=factory([selected],inputs[name],directory))
        final[name]=report['results'][0];total+=report['new_gpu_environment_steps']
        fit.search.write_json(args.output/'progress.json',dict(final=final,new_gpu_environment_steps=total))
    assert {str(p):fit.audit.digest(p) for p in protected_paths}==protected
    fit.search.write_json(args.output/'report.json',dict(final=final,selected_index=best,new_gpu_environment_steps=total,
        protected_sources=protected,gas_spring_specifications_constrained=False,shared_model_modified=False,
        empirical_load='64 normalized Gaussian state bases per leg; input=current simulated q,v and displacement-driven direction memory',
        no_time_or_future_position_features=True,development_sessions=['original','wide'],regression_sessions=['slow','fast'],
        source_inverse_hashes={n:fit.audit.digest(p/'inverse-loads.npz') for n,p in paths.items()},
        equivalent_load_can_be_active=True,empirical_torque_clip_nm=40.,physical_parameters_identified=False,
        hardware_access=False,policy_training_updates=0))
    for p in (Path(__file__),Path(fit.__file__)):
        shutil.copy2(p,args.output/p.name)


if __name__=='__main__':main()
