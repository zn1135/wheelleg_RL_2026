"""Verify a response candidate, compare GPU trajectories and export a separate model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.agent.real2sim import replay_leg_test_gpu as replay
from scripts.agent.real2sim import elastic_joint_resistance as elastic


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--stiff',type=Path,required=True)
    parser.add_argument('--aligned',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--validation-inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=[args.baseline,args.stiff,args.aligned]
    reports=[json.loads((p/'report.json').read_text()) for p in paths]
    states=[]
    for path in paths:
        with np.load(path/'replay.npz') as archive:
            states.append({key:archive[key] for key in archive.files})
    baseline,stiff,aligned=reports
    for key in ('stream_sha256','source_model_sha256','config_sha256','spring_model','training_spring_source_sha256','dt_s','duration_s'):
        assert all(r[key]==baseline[key] for r in reports),key
    for state in states[1:]:
        for key in ('time_s','real_geometry','real_motor_state','initial_qpos','initial_qvel'):
            assert np.array_equal(state[key],states[0][key]),key
        assert np.array_equal(state['submitted_model_torque'][:,0],states[0]['submitted_model_torque'][:,0])
    profile=aligned['response_parameters']
    assert profile and aligned['constraint_check']=='stiff' and not stiff.get('response_parameters')
    old=ET.parse(args.stiff/'fixed-base-model.xml').getroot()
    new=ET.parse(args.aligned/'fixed-base-model.xml').getroot()
    allowed=('frictionloss','damping','solreffriction','solimpfriction','armature')
    for name in replay.mirror.MOTORS[:4]:
        first,last=[r.find(f".//joint[@name='{name}']") for r in (old,new)]
        for key in allowed:
            if key in first.attrib:last.set(key,first.get(key))
            else:last.attrib.pop(key,None)
    assert ET.tostring(old)==ET.tostring(new),'Unexpected model difference beyond resistance candidate'
    source=Path(aligned['source_model'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==aligned['source_model_sha256']
    config=Path(json.loads((Path(aligned['session'])/'meta.json').read_text())['machine_config']['path'])
    assert hashlib.sha256(config.read_bytes()).hexdigest()==aligned['config_sha256']
    sign,zero=replay.mirror.load_mapping(config)
    model=mujoco.MjModel.from_xml_path(str(args.aligned/'fixed-base-model.xml'))
    mq=[model.joint(n).qposadr[0] for n in replay.mirror.MOTORS[:4]]
    sq=[model.joint(s+'_spring2_joint').qposadr[0] for s in ('left','right')]
    sv=[model.joint(s+'_spring2_joint').dofadr[0] for s in ('left','right')]
    sample_time=aligned['window_start_board_s']+states[-1]['time_s']
    measured=states[-1]['real_motor_state']
    real=np.column_stack([np.interp(sample_time,measured[:,0],measured[:,1+i]) for i in range(4)])
    errors=[];predicted=[];metrics={}
    for label,state,report in zip(('original','stiff_only','aligned'),states,reports):
        q=np.take(state['qpos'],mq,axis=2)*sign[:4]+zero
        error=q-real[:,None,:]
        predicted.append(q);errors.append(error)
        metrics[label]={}
        for branch,name in enumerate(('recorded_torque','recorded_target_feedback')):
            rmse=np.sqrt(np.mean(error[:,branch]**2,axis=0))
            assert np.allclose(rmse,report['metrics'][name]['motor_position_rmse_rad'],atol=1e-10)
            metrics[label][name]=dict(motor_rmse_deg=np.degrees(rmse).tolist(),
                                      peak_error_deg=np.max(np.abs(np.degrees(error[:,branch])),axis=0).tolist(),
                                      final_error_deg=np.degrees(error[-1,branch]).tolist(),
                                      combined_rmse_deg=float(np.degrees(np.sqrt(np.mean(rmse**2)))),
                                      pin_error_max_mm=(1000*np.array(report['metrics'][name]['max_loop_pin_residual_m'])).tolist())
    assert np.max(np.sqrt(np.mean(errors[-1]**2,axis=0)))<np.deg2rad(3.)
    assert np.max(np.abs(errors[-1][-1]))<np.deg2rad(1.)
    assert np.max(np.abs(errors[-1]))<np.deg2rad(5.)
    assert max(max(m['pin_error_max_mm']) for m in metrics['aligned'].values())<.1
    assert not aligned['feedback_branch_blocked']
    expected=replay.training_spring.gas_spring_force(states[-1]['qpos'][:-1,:,sq].astype(np.float32),
                                                   states[-1]['qvel'][:-1,:,sv].astype(np.float32))
    spring_error=float(np.max(np.abs(expected-states[-1]['spring_force_n'])))
    assert spring_error<2e-5
    elastic_error=0.;minimum_loss=0.;minimum_discrete_loss=0.;total_discrete_loss=0.
    stiffness=np.asarray(profile.get('stiffness_nm_rad',[0.]*4))
    if np.any(stiffness>0):
        limit=np.asarray(profile['friction_nm']);mask=stiffness>0
        holding=np.asarray(aligned['elastic_preload']['holding_torque_nm'])
        force=np.tile(np.where(mask,np.clip(states[-1]['submitted_model_torque'][0,0,:4]-holding,-limit,limit),0.),(2,1))
        stored_force=states[-1]['elastic_resistance_nm']
        for k in range(len(stored_force)):
            elastic_error=max(elastic_error,float(np.max(np.abs(force-stored_force[k]))))
            delta=np.take(states[-1]['qpos'][k+1]-states[-1]['qpos'][k],mq,axis=1)
            delta=np.where(mask,delta,0.)
            previous_force=force.copy()
            force,loss=elastic.advance(force,delta,limit,np.maximum(stiffness,1.))
            minimum_loss=min(minimum_loss,loss)
            discrete=previous_force*delta+np.asarray(profile['damping_nms_rad'])/aligned['dt_s']*delta**2-(force**2-previous_force**2)/(2*np.maximum(stiffness,1.))
            minimum_discrete_loss=min(minimum_discrete_loss,float(discrete.min()))
            total_discrete_loss+=float(discrete.sum())
        assert elastic_error<1e-5 and minimum_loss>=-1e-8
        assert minimum_discrete_loss>=-1e-7 and total_discrete_loss>=0
    validation=json.loads((args.validation/'report.json').read_text())
    v=np.load(args.validation/'trials.npz');vi=np.load(args.validation_inputs/'replay.npz')
    assert validation['best']['parameters']==profile
    assert validation['fit_window_from_start_s']==[0.,8.] and not validation['selection_uses_validation']
    assert validation['objective_branches']==['recorded_torque','recorded_target_feedback']
    assert np.array_equal(v['qpos'][0],states[-1]['qpos'][0])
    prefix_error=float(np.max(np.abs(v['qpos'][:801]-states[-1]['qpos'][::10])))
    assert prefix_error<1e-4,'Batch/controller implementation mismatch'
    extended_q=v['qpos'][:,:,mq]*sign[:4]+zero
    vt=aligned['window_start_board_s']+v['time_s'];vm=vi['real_motor_state']
    extended_real=np.column_stack([np.interp(vt,vm[:,0],vm[:,1+i]) for i in range(4)])
    extended_error=extended_q-extended_real[:,None,:]
    hold=v['time_s']>8.
    hold_rmse=np.degrees(np.sqrt(np.mean(extended_error[hold]**2,axis=0)))
    end=np.degrees(extended_error[-1])
    assert np.max(hold_rmse)<1.5 and np.max(np.abs(end))<1.5
    assert max(validation['best_sampled_pin_error_m'])<.0001
    result=dict(scope='Same-trial response candidate, not independent physical friction calibration',
                metrics=metrics,postfit_15s_motor_rmse_deg=hold_rmse.tolist(),postfit_final_error_deg=end.tolist(),
                sample_criteria=dict(each_8s_motor_rmse_deg_below=3.,each_8s_final_error_deg_below=1.,
                                     each_8s_peak_error_deg_below=5.,postfit_rmse_and_final_error_deg_below=1.5,connection_gap_mm_below=.1),
                criteria_passed=True,parameters=profile,source_model_sha256=aligned['source_model_sha256'],
                source_model_modified=False,firmware_modified=False,independent_trial_tested=False,
                validation_batch_scalar_max_qpos_difference=prefix_error,spring_force_check_max_difference_n=spring_error,
                elastic_force_check_max_difference_nm=elastic_error,minimum_elastic_dissipation=minimum_loss,
                minimum_discrete_resistance_work_margin_j=minimum_discrete_loss,total_discrete_resistance_dissipation_j=total_discrete_loss,
                same_initial_states_and_torque_inputs=True,model_changes_only_documented_resistance=True,
                physics_steps=0,scope_of_this_report='saved-GPU-state checks and static compilation only')
    args.output.mkdir(parents=True,exist_ok=False)
    # 导出实际验证的固定机身模型；有状态阻力须配套JSON和运行时，不伪装为已验收的训练模型。
    candidate=ET.parse(args.aligned/'fixed-base-model.xml').getroot()
    candidate.find('option').attrib.update(timestep=str(aligned['dt_s']),integrator='implicitfast',jacobian='dense')
    xml=ET.tostring(candidate,encoding='unicode')
    exported=mujoco.MjModel.from_xml_string(xml)
    assert exported.nv==model.nv and candidate.find('.//freejoint') is None
    (args.output/'candidate-fixed-base.xml').write_text(xml)
    result['candidate_requires_companion_friction_runtime']=bool(np.any(stiffness>0))
    result['candidate_xml_sha256']=hashlib.sha256(xml.encode()).hexdigest()
    (args.output/'response-candidate.json').write_text(json.dumps(profile,indent=2)+'\n')
    (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_time=sample_time-aligned['first_dial_board_s']
    fig,axes=plt.subplots(4,2,figsize=(13,10),sharex=True)
    for i,name in enumerate(replay.mirror.MOTORS[:4]):
        for branch in range(2):
            ax=axes[i,branch]
            ax.plot(plot_time,np.degrees(real[:,i]),color='black',lw=1.8,label='Recorded robot')
            ax.plot(plot_time,np.degrees(predicted[-1][:,branch,i]),color='tab:orange',label='GPU resistance candidate')
            ax.set_ylabel(name+' (deg)');ax.grid(alpha=.3)
    axes[0,0].set_title('Same recorded torque');axes[0,1].set_title('Same targets + feedback')
    for ax in axes[0]:ax.legend()
    for ax in axes[-1]:ax.set_xlabel('Time from first dial input (s)')
    fig.tight_layout();fig.savefig(args.output/'alignment-8s.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for i,name in enumerate(replay.mirror.MOTORS[:4]):
        ax=axes[i];ax.plot(v['time_s'],np.degrees(extended_real[:,i]),color='black',label='Recorded robot')
        for b,label in enumerate(('GPU: torque','GPU: target feedback')):
            ax.plot(v['time_s'],np.degrees(extended_q[:,b,i]),label=label)
        ax.axvline(8,color='grey',ls='--',label='End of fitting window' if i==0 else None)
        ax.set_ylabel(name+' (deg)');ax.grid(alpha=.3)
    axes[0].legend();axes[-1].set_xlabel('Time from replay start (s); no state resets')
    fig.tight_layout();fig.savefig(args.output/'validation-23s.png',dpi=150);plt.close(fig)
    for path in (Path(__file__),Path(replay.__file__),Path(replay.training_spring.__file__),Path(elastic.__file__)):
        dest=args.output/'source'/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
