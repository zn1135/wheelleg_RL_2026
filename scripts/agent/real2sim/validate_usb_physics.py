"""Export the frozen log-fit candidate and compare frozen-policy GPU parking runs."""
import argparse
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import RecomputeLevel

from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from scripts.agent.fudan_ppo_walk import evaluate
from scripts.agent.fudan_ppo_walk.diagnose_parking_posture import parking_items
from .probe_video_leg_parameters import items as hold_items
from .replay_usb_pd_targets import sha, write
from .fit_usb_pd_physics import metrics


FIELDS=('dof_armature','dof_damping','dof_frictionloss','body_mass','body_inertia','body_ipos','geom_friction')


def selected_samples():
    selected=parking_items()
    for s in hold_items(profiles=[dict(name='mechanical',armature_scale=2.,wheel_motor_gain=1.)]):
        h=s['case']['segments'][0]['command'][2];s['id']=f'hold_h{round(h*100)}'
        s['case']=dict(id=f'zero_hold_{h:g}',category='zero',duration=30.,
            segments=[dict(start=0.,duration=30.,command=[0.,0.,h])])
        selected.append(s)
    assert len(selected)==12 and not any(s['randomized'] for s in selected)
    return selected


def export(fit,out,comparison,parameters):
    source=Path(json.loads((fit/'plan.json').read_text())['source_baseline'])
    replay=out/'target-replay';replay.mkdir()
    i=comparison['selected_index']
    for name in ('inputs.npz','inputs-report.json'):shutil.copy2(source/name,replay/name)
    with np.load(fit/'candidate-trajectories.npz') as f:
        np.savez_compressed(replay/'gpu-replay.npz',qpos=f['qpos'][:,i],
            motor_velocity=f['motor_velocity'][:,i],torque=f['torque'][:,i],time_s=f['time_s'])
    model=mujoco.MjModel.from_binary_path(str(fit/'template-model.mjb'))
    for name in FIELDS:getattr(model,name)[:]=np.asarray(parameters[name][i])
    mujoco.mj_saveModel(model,str(replay/'model.mjb'))
    execution=json.loads((fit/'execution.json').read_text())
    write(replay/'gpu-report.json',dict(execution,motor_qpos_indices=comparison['motor_qpos_indices'],
        physical_candidate=comparison['selected_profile'],baseline_source=str(source),
        fitted_actual_parameters_sha256=sha(fit/'actual-parameters.json'),
        note='Frozen fit trajectory, not re-integrated during export; spring force parameters are runtime controls'))
    write(replay/'physical-candidate.json',dict(profile=comparison['selected_profile'],
        actual={k:v[i] for k,v in parameters.items()},adopted_as_default=False))


def summarize(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=json.loads((out/'run.json').read_text());fit=Path(run['fit'])
    comparison=json.loads((fit/'comparison.json').read_text());i=run.get('candidate_index',comparison['selected_index'])
    with np.load(out/'target-replay/inputs.npz') as f:data={k:f[k].copy() for k in f.files}
    with np.load(fit/'candidate-trajectories.npz') as f:
        q=f['qpos'][:,[0,i]].copy();v=f['motor_velocity'][:,[0,i]].copy()
    qids=np.asarray(comparison['motor_qpos_indices']);all_times=np.ones(len(q),dtype=bool)
    errors=[metrics(data,q[:,j],v[:,j],qids,all_times) for j in (0,1)]
    def gravity(quat):
        w,x,y,z=quat.T
        return np.stack((2*(w*y-x*z),-2*(w*x+y*z),-(1-2*(x*x+y*y))),axis=-1)
    indices=data['reference_indices'];real_g=gravity(data['reference_qpos'][:,3:7])
    attitude=[]
    for j in (0,1):
        simulated_g=gravity(q[indices,j,3:7]);cos=np.sum(real_g*simulated_g,axis=1)
        angle=np.rad2deg(np.arccos(np.clip(cos,-1,1)))
        attitude.append({label:float(np.sqrt(np.mean(angle[mask]**2))) for label,mask in
            (('full_RMSE_deg',np.ones(len(indices),dtype=bool)),('fit_RMSE_deg',data['time_s'][indices]<16),
             ('internal_validation_RMSE_deg',data['time_s'][indices]>=16))})
    fig,axes=plt.subplots(6,1,figsize=(13,13),sharex=True)
    for ax,motor,name in zip(axes,[0,1,3,4,2,5],['LF front','LR rear','RF front','RR rear','Left wheel','Right wheel']):
        leg=motor not in (2,5)
        real=np.rad2deg(data['current'][:,motor]) if leg else data['velocity'][:,motor]
        ax.plot(data['time_s'],real,label='Real measured',color='black',lw=.8)
        for j,label,color in ((0,'Baseline','#d55e00'),(1,'Candidate','#0072b2')):
            values=np.rad2deg(q[:,j,qids[motor]]) if leg else v[:,j,motor]
            ax.plot(data['time_s'],values,label=label,color=color,lw=.7,alpha=.85)
        ax.axvspan(16,data['time_s'][-1],alpha=.1,color='green');ax.axvline(16,color='gray',ls='--',lw=.8)
        ax.set_ylabel(name+('\n(deg)' if leg else '\n(rad/s)'));ax.grid(alpha=.2)
    axes[0].legend(ncol=3);axes[0].set_title('Same logged PD targets / shaded: internal validation after16s / base state estimated')
    axes[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(out/'gap-comparison.png',dpi=140);plt.close(fig)
    scores={g:json.loads((out/(g+'-acceptance/acceptance.json')).read_text()) for g in ('baseline','candidate')}
    sets={g:{s['sample_id'] for s in score['samples'] if s['passed']} for g,score in scores.items()}
    initial=[]
    physical=set(FIELDS)|{'spring_bias_n','spring_damping'}
    for s in run['samples']:
        name=s['id']+'_initial.npz'
        with np.load(out/'baseline/fixtures'/name) as a,np.load(out/'candidate/fixtures'/name) as b:
            keys=sorted(set(a.files)-physical)
            initial.append(dict(id=s['id'],identical_nonphysical_initial_conditions=all(np.array_equal(a[k],b[k]) for k in keys),fields=keys))
    assert all(x['identical_nonphysical_initial_conditions'] for x in initial)
    complete=all(s['complete'] and s['finite'] for score in scores.values() for s in score['samples'])
    lost=sorted(sets['baseline']-sets['candidate'])
    write(out/'result.json',dict(candidate=run['candidate'],full_record_metrics=dict(baseline=errors[0],candidate=errors[1]),
        published_gravity_direction_error=dict(baseline=attitude[0],candidate=attitude[1],
            note='Compared with published policy gravity at20ms; possible real observation attitude offset remains unknown'),
        fit_metrics=dict(baseline=comparison['rows'][0]['fit'],candidate=comparison['rows'][i]['fit']),
        internal_validation_metrics=dict(baseline=comparison['rows'][0]['validation'],candidate=comparison['rows'][i]['validation']),
        closed_loop={g:dict(passed=scores[g]['technical_passed'],total=scores[g]['total'],passing_ids=sorted(sets[g])) for g in sets},
        closed_loop_complete_and_finite=complete,initial_comparison=initial,lost_successes=lost,
        closed_loop_regression=bool(lost),adopted_as_default=False,training_updates=0,
        limitation='Single previously inspected motion log, internal time split only. Not unique physical identification or independent real-robot validation.'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--candidate-index',type=int,help='Explicit frozen diagnostic candidate, even if the continuous-rollout objective rejects it')
    a=p.parse_args();fit=a.fit.resolve();out=a.output.resolve()
    if a.report_only:summarize(out);return
    out.mkdir(parents=True,exist_ok=False)
    comparison=json.loads((fit/'comparison.json').read_text());params=json.loads((fit/'actual-parameters.json').read_text())
    i=comparison['selected_index'] if a.candidate_index is None else a.candidate_index
    assert 0<i<len(comparison['rows'])
    comparison['selected_index']=i;comparison['selected_profile']=comparison['rows'][i]['profile']
    export(fit,out,comparison,params);shutil.copy2(__file__,out/'validation-source.py')
    write(out/'run.json',dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint),
        fit=str(fit),fit_selection_sha256=sha(fit/'fit-selection.json'),candidate=comparison['selected_profile'],candidate_index=i,
        explicit_candidate_override=a.candidate_index is not None,
        samples=selected_samples(),PD=dict(leg_p=60.,leg_d=2.,wheel_p=.1),training_updates=0,
        scope='12 diagnostic cases per plant, frozen3000 policy; not full987 or hardware acceptance'))
    original=evaluate.configure_samples;torch.set_num_threads(1)
    for group,selected in (('baseline',0),('candidate',i)):
        def configure(wrapper,batch,folder):
            result=original(wrapper,batch,folder);sim=wrapper.env.sim;action=wrapper.action
            # 使用已冻结的实际物理数组；不重新拟合，也不改变任何初始关节状态或PD。
            q=sim.data.qpos.clone();v=sim.data.qvel.clone()
            sim.expand_model_fields(FIELDS)
            for name in FIELDS:
                value=torch.as_tensor(params[name][selected],device=wrapper.device,dtype=torch.float32)
                getattr(sim.model,name).copy_(value.expand_as(getattr(sim.model,name)))
            action._spring_force_bias.copy_(torch.as_tensor(params['spring_bias_n'][selected],device=wrapper.device).expand_as(action._spring_force_bias))
            action.spring_damping.copy_(torch.as_tensor(params['spring_damping'][selected],device=wrapper.device).expand_as(action.spring_damping))
            sim.recompute_constants(RecomputeLevel.set_const)
            sim.data.qpos.copy_(q);sim.data.qvel.copy_(v);sim.forward();wrapper.env.scene.update(dt=0.)
            assert torch.equal(sim.data.qpos,q) and torch.equal(sim.data.qvel,v)
            assert action.cfg.leg_stiffness==60 and action.cfg.leg_damping==2 and action.cfg.wheel_stiffness==.1
            assert action.kp_scale.eq(1).all() and action.kd_scale.eq(1).all() and action.motor_strength.eq(1).all()
            assert not action._wheel_speed_shaping and not action.wheel_motor.enabled
            actual={k:getattr(sim.model,k).cpu().numpy() for k in FIELDS}
            for j,s in enumerate(batch):
                path=folder/(s['id']+'_initial.npz')
                with np.load(path) as f:record={k:f[k].copy() for k in f.files}
                record.update({k:val[j] for k,val in actual.items()})
                record.update(spring_bias_n=np.asarray(params['spring_bias_n'][selected]),
                    spring_damping=np.asarray(params['spring_damping'][selected]))
                np.savez_compressed(path,**record)
            model=sim.mj_model
            for name in FIELDS:getattr(model,name)[:]=actual[name][0]
            mujoco.mj_saveModel(model,str(folder/'model.mjb'))
            write(folder/'actual-physical-profile.json',dict(profile=comparison['rows'][selected]['profile'],
                actual={k:v[selected] for k,v in params.items()},PD=dict(leg_p=60.,leg_d=2.,wheel_p=.1)))
            return result
        request=SimpleNamespace(checkpoint=a.checkpoint,output=out/group,deadline=time.time()+360,
            diagnostic=True,diagnostic_height=None,batch_size=12,selection=None,leg_load_profile='legacy',response_parameters=None)
        with patch.object(FudanWalkCfg.control,'leg_stiffness',60.),patch.object(FudanWalkCfg.control,'leg_damping',2.),\
             patch.object(FudanWalkCfg.control,'wheel_stiffness',.1),patch.object(FudanWalkCfg.sim,'wheel_model','mechanical'),\
             patch.object(FudanWalkCfg.sim,'wheel_motor_gain',1.),patch.object(evaluate,'evaluation_items',return_value=selected_samples()),\
             patch.object(evaluate,'configure_samples',configure):evaluate.gpu(request)
        protocol_path=out/group/'protocol.json';protocol=json.loads(protocol_path.read_text())
        protocol['log_fitted_physics']=dict(profile=comparison['rows'][selected]['profile'],actual_parameters_sha256=sha(fit/'actual-parameters.json'))
        write(protocol_path,protocol)
    write(out/'execution.json',dict(complete=True,groups=['baseline','candidate'],training_updates=0,
        checkpoint_unchanged=sha(a.checkpoint)==json.loads((out/'run.json').read_text())['checkpoint_sha256']))


if __name__=='__main__':main()
