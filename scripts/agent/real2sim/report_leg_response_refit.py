"""Audit finite leg-response trials and plot fixed-parameter crosschecks; no integration."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.agent.real2sim import replay_leg_test_gpu as replay


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    before=json.loads((args.experiment/'before-sources.json').read_text())
    protected={p:h for p,h in before['sources'].items() if p.startswith(('resources/','mjlab_training/'))}
    for path,digest in protected.items():assert sha(path)==digest,path
    for path,digest in before['firmware_sources'].items():
        assert sha(Path('deployment/H7_RL_ros')/path)==digest,path
    parameters=json.loads((args.experiment/'rated-fixed-crosscheck-parameters.json').read_text())[0]
    rows=[];plots=[];maximum_force_error=0.;default_force_error=0.
    for name in ('wide','slow','fast','original'):
        directory=args.experiment/('rated-fixed-'+name)
        report=json.loads((directory/'report.json').read_text())
        reference=Path(report['reference']);source=json.loads((reference/'report.json').read_text())
        assert report['physics_backend']=='mujoco_warp_gpu'
        assert report['worlds']==2 and len(report['trials'])==1
        assert report['best']['parameters']==parameters
        assert report['source_model_sha256']==source['source_model_sha256']==sha(source['source_model'])
        config=Path(json.loads((Path(source['session'])/'meta.json').read_text())['machine_config']['path'])
        assert sha(config)==source['config_sha256']
        sign,zero=replay.mirror.load_mapping(config)
        model=mujoco.MjModel.from_xml_path(str(directory/'candidate-model.xml'))
        data=mujoco.MjData(model)
        mq=[model.joint(n).qposadr[0] for n in replay.mirror.MOTORS[:4]]
        sq=[model.joint(s+'_spring2_joint').qposadr[0] for s in ('left','right')]
        sv=[model.joint(s+'_spring2_joint').dofadr[0] for s in ('left','right')]
        springs=[model.actuator(s+'_gas_spring_motor').id for s in ('left','right')]
        assert np.allclose(model.dof_armature[mq],parameters['armature_kg_m2'])
        assert np.all(model.actuator_forcerange[springs,1]>=parameters['spring_force_limit_n'])
        with np.load(directory/'trials.npz') as saved,np.load(reference/'replay.npz') as original:
            q=saved['qpos'];v=saved['qvel'];times=saved['time_s'];real=saved['real_motor_q']
            assert np.allclose(q[0],original['qpos'][0,0],rtol=0,atol=1e-7)
            assert np.allclose(v[0],original['qvel'][0,0],rtol=0,atol=1e-7)
            absolute=times+source['window_start_board_s'];measured=original['real_motor_state']
            expected=np.column_stack([np.interp(absolute,measured[:,0],measured[:,1+i]) for i in range(4)])
            assert np.array_equal(expected,real)
            predicted=q[:,:,mq]*sign[:4]+zero
            error=np.degrees(predicted-real[:,None,:])
            rmse=np.sqrt(np.mean(error**2,axis=0));peak=np.max(np.abs(error),axis=0)
            best=report['best']
            assert np.allclose(rmse[0],best['all_motor_rmse_deg'],atol=1e-6)
            assert np.allclose(rmse[1],best['feedback_motor_rmse_deg'],atol=1e-6)
            assert np.allclose(error[-1,0],best['final_error_deg'],atol=1e-6)
            assert np.allclose(error[-1,1],best['feedback_final_error_deg'],atol=1e-6)
            assert np.allclose(peak[0],best['peak_error_deg'],atol=1e-6)
            assert np.allclose(peak[1],best['feedback_peak_error_deg'],atol=1e-6)
            assert np.isfinite(q).all() and np.isfinite(v).all()
            # 逐个保存状态核对诊断公式，并核对MuJoCo执行器确实没有夹在150N。
            expected_force=np.clip(150.+np.asarray(parameters['spring_stiffness_n_m'])*
                                   (np.maximum(0.,.010-q[:,:,sq])-.005)-50.*v[:,:,sv],
                                   0.,np.asarray(parameters['spring_force_limit_n']))
            actual=replay.spring_state(q[:,:,sq],'linear-damped',v[:,:,sv],parameters)[1]
            maximum_force_error=max(maximum_force_error,float(np.max(np.abs(expected_force-actual))))
            old=replay.spring_state(q[:,:,sq],'linear-damped',v[:,:,sv])[1]
            default_force_error=max(default_force_error,float(np.max(np.abs(old-replay.training_spring.gas_spring_force(q[:,:,sq],v[:,:,sv])))))
            for position,velocity,force in zip(q.reshape(-1,model.nq),v.reshape(-1,model.nv),actual.reshape(-1,2)):
                data.qpos[:]=position;data.qvel[:]=velocity;data.ctrl[:]=0.;data.ctrl[springs]=force
                mujoco.mj_forward(model,data)
                maximum_force_error=max(maximum_force_error,float(np.max(np.abs(data.actuator_force[springs]-force))))
            plots.append((name,times.copy(),np.degrees(real),np.degrees(predicted)))
        passed=bool(np.all(rmse<3.) and np.all(peak<5.) and np.all(np.abs(error[-1])<1.) and
                    max(report['best_sampled_pin_error_m'])<.0001 and not best['feedback_blocked'])
        rows.append(dict(trajectory=name,duration_s=float(times[-1]),recording_sha256=report['stream_sha256'],
                         data_role='development_selection' if name=='wide' else 'fixed_parameter_development_crosscheck',
                         joint_rmse_deg=rmse.tolist(),combined_rmse_deg=np.sqrt(np.mean(rmse**2,axis=1)).tolist(),
                         peak_error_deg=peak.tolist(),final_error_deg=error[-1].tolist(),
                         maximum_pin_error_m=max(report['best_sampled_pin_error_m']),
                         feedback_blocked=best['feedback_blocked'],response_thresholds_passed=passed))
    assert maximum_force_error<1e-4 and default_force_error==0.
    # 已完成的有限批次全部计数；初始化失败且无report的目录不计物理步。
    reports=[]
    for path in args.experiment.rglob('report.json'):
        report=json.loads(path.read_text())
        if report.get('physics_backend')=='mujoco_warp_gpu' and 'worlds' in report:
            reports.append(dict(path=str(path),environment_steps=report['new_environment_steps'],
                                candidate_evaluations=len(report['trials'])))
    result=dict(integrity_checks_passed=True,response_alignment_passed=all(r['response_thresholds_passed'] for r in rows),
                scope='Fixed-base offline diagnostic; no adopted training model or independent hardware acceptance',
                protected_source_hashes=protected,unchanged_firmware_sources=len(before['firmware_sources']),
                motor_order=list(replay.mirror.MOTORS[:4]),branch_order=['recorded_torque','recorded_target_feedback'],
                fixed_diagnostic_parameters=parameters,trajectories=rows,
                maximum_spring_formula_and_actuator_error_n=maximum_force_error,default_shared_force_error_n=default_force_error,
                completed_gpu_batches=reports,total_gpu_environment_steps=sum(r['environment_steps'] for r in reports),
                candidate_evaluations=sum(r['candidate_evaluations'] for r in reports),
                new_environment_steps_in_this_report=0,policy_training_updates=0,
                limitations=['Manufacturer nominal 150N is user-confirmed; actual force curve/reference and maximum remain unmeasured',
                             '5mm reference convention and tested slopes are hypotheses, not specifications for the actual unidentified part',
                             'All existing recordings are development/regression data, not independent acceptance after parameter selection',
                             'Fixed-base support and commanded motor torque are assumptions; actual shaft torque and support motion were not independently measured'])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,times,real,predicted in plots:
        fig,axes=plt.subplots(4,1,figsize=(11,9),sharex=True)
        for i,ax in enumerate(axes):
            ax.plot(times,real[:,i],label='Measured',color='black')
            ax.plot(times,predicted[:,0,i],label='Recorded torque',alpha=.8)
            ax.plot(times,predicted[:,1,i],label='Recorded target feedback',alpha=.8)
            ax.set_ylabel(replay.mirror.MOTORS[i]+' (deg)');ax.grid(alpha=.25)
        axes[0].legend();axes[-1].set_xlabel('Time (s)');fig.tight_layout()
        fig.savefig(args.output/(name+'.png'),dpi=130);plt.close(fig)
    print(json.dumps(dict(integrity_checks_passed=True,response_alignment_passed=result['response_alignment_passed'],
                          total_gpu_environment_steps=result['total_gpu_environment_steps'],trajectories=rows),indent=2))


if __name__=='__main__':main()
