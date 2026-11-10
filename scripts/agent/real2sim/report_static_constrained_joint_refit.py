"""Independently verify/plot the frozen old-recording static-constrained GPU refit; no integration."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.agent.real2sim import fit_leg_repaired_loop as fit


def close(actual, expected, tolerance):
    discrepancy = float(np.max(abs(np.asarray(actual)-np.asarray(expected))))
    assert discrepancy <= tolerance, (discrepancy, tolerance)
    return discrepancy


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.experiment/'report.json').read_text())
    all_results, plot_data = {}, {}
    max_curve_error = max_capacity_error = max_controller_error = 0.
    for name in ('original', 'wide', 'slow', 'fast'):
        reference = args.experiment/'inputs'/name
        fit.REFERENCES[name] = (str(reference), str(reference/'rc-inputs.npz'))
        inputs = fit.input_data(name)
        plot_data[name] = {}
        all_results[name] = {}
        for prefix in ('final', 'baseline'):
            directory = args.experiment/(prefix+'-'+name)
            report = json.loads((directory/'report.json').read_text())
            model = mujoco.MjModel.from_xml_path(str(directory/'model.xml'))
            assert fit.audit.digest(directory/'model.xml') == report['model_sha256']
            for path, digest in report['input_sha256'].items():
                assert fit.audit.digest(Path(path)) == digest
            assert fit.audit.digest(Path(report['rc_path'])) == report['rc_sha256']
            profile = report['results'][0]['parameters']
            mq = [model.joint(n).qposadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
            mv = [model.joint(n).dofadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
            sq = [model.joint(s+'_spring2_joint').qposadr[0] for s in ('left', 'right')]
            sv = [model.joint(s+'_spring2_joint').dofadr[0] for s in ('left', 'right')]
            aid = [model.actuator(n+'_motor').id for n in fit.prior.replay.mirror.MOTORS]
            sid = [model.actuator(s+'_gas_spring_motor').id for s in ('left', 'right')]
            with np.load(directory/'physics.npz') as raw:
                q, v, ctrl, friction = [raw[k] for k in ('qpos','qvel','ctrl','friction_capacity')]
                endq = raw['final_qpos']
            # 提交的滑杆压力和原生阻力容量按每个0.2ms步前仿真状态独立复算。
            x = np.maximum(0., .010-q[:,:,sq])
            pressure = (np.array(profile['spring_reference_force_n'])
                        +np.array(profile['spring_stiffness_n_m'])*(x-.005)
                        -np.array(profile['spring_damping_ns_m'])*v[:,:,sv])
            expected_force = np.clip(pressure, 0., 600.)
            expected_capacity = np.maximum(0., np.array(profile['spring_friction_n'])
                +np.array(profile.get('spring_friction_stiffness_n_m',[0.,0.]))*(x-.005))
            max_curve_error = max(max_curve_error, close(ctrl[:,:,sid], expected_force, 2e-4))
            max_capacity_error = max(max_capacity_error, close(friction[:,:,sv], expected_capacity, 2e-5))
            expected_control = np.repeat(inputs['saved']['submitted_model_torque'][:,0], 5, axis=0)
            close(ctrl[:,0,aid], expected_control, 1e-7)
            close(ctrl[:,:,aid[4:]], 0., 1e-7)
            recovering, blocked = np.array([inputs['recovering']]), np.array([False])
            for k in range(len(expected_control)//5):
                target_control = fit.controller(q[5*k,1:2], v[5*k,1:2], mq, mv, inputs, k, recovering, blocked)
                max_controller_error = max(max_controller_error,
                    close(ctrl[5*k:5*k+5,1,aid[:4]], np.broadcast_to(target_control,(5,4)), 2e-4))
            state = np.concatenate((q[::5],endq[None]),axis=0)
            times = np.arange(len(state))*.001
            measured = inputs['saved']['real_motor_state']
            absolute = inputs['report']['window_start_board_s']+times
            real = np.column_stack([np.interp(absolute,measured[:,0],measured[:,i+1]) for i in range(4)])
            predicted = state[:,:,mq]*inputs['sign'][:4]+inputs['zero']
            error = np.degrees(predicted-real[:,None,:])
            rows = []
            data = mujoco.MjData(model)
            peak_pin = np.zeros(2)
            # 逐1ms保存状态检查销点闭合；不能称所有0.2ms子步最大值。
            for states in state:
                for branch, pose in enumerate(states):
                    data.qpos[:] = pose; mujoco.mj_kinematics(model,data)
                    for pairs in fit.prior.replay.mirror.PIN_PAIRS:
                        for a,b in pairs:
                            peak_pin[branch] = max(peak_pin[branch],np.linalg.norm(data.site(a).xpos-data.site(b).xpos))
            for branch, key in enumerate(('recorded_torque','recorded_target')):
                row = report['results'][0][key]
                computed = dict(rmse_deg=np.sqrt(np.mean(error[:,branch]**2,axis=0)).tolist(),
                                peak_deg=abs(error[:,branch]).max(0).tolist(), terminal_deg=abs(error[-1,branch]).tolist())
                for metric, values in computed.items():
                    close(values,row[metric],1e-6)
                joint_pass = ((np.array(computed['rmse_deg'])<3.) & (np.array(computed['peak_deg'])<5.)
                              & (np.array(computed['terminal_deg'])<1.))
                computed.update(pin_error_max_m=float(peak_pin[branch]), joint_thresholds_passed=joint_pass.tolist(),
                    accepted=bool(joint_pass.all() and peak_pin[branch]<1e-4 and row['original_assembly_branch']
                                  and row['limit_excess_deg']<=.1 and not report['results'][0]['feedback_blocked']))
                rows.append(computed)
            all_results[name][prefix] = dict(zip(('recorded_torque','recorded_target'),rows))
            plot_data[name][prefix] = dict(time=times, real=real, predicted=predicted)
            print(f'Verified {prefix}-{name}',flush=True)
    composition = []
    selected = summary['selected_parameters']
    for name in ('original','wide'):
        model = mujoco.MjModel.from_xml_path(str(args.experiment/('final-'+name)/'model.xml'))
        mq = [model.joint(n).qposadr[0] for n in fit.prior.replay.mirror.MOTORS[:4]]
        with np.load(args.experiment/('final-'+name)/'trials.npz') as saved:
            final_q = saved['qpos'][::10].copy()
        for side in (0,1):
            found = False
            for generation in range(len(summary['history'])):
                directory = args.experiment/f'generation-{generation:02d}'/name
                report = json.loads((directory/'report.json').read_text())
                for index,row in enumerate(report['results']):
                    profile = row['parameters']
                    same = all((profile[key][2*side:2*side+2] if len(values)==4 else profile[key][side:side+1])
                               == (values[2*side:2*side+2] if len(values)==4 else values[side:side+1])
                               for key,values in selected.items())
                    if not same:continue
                    with np.load(directory/'trials.npz') as saved:
                        batch = saved['qpos'][:,[index,index+len(report['results'])]]
                    joint_ids = mq[2*side:2*side+2]
                    delta = abs(np.degrees(final_q[:,:,joint_ids]-batch[:,:,joint_ids])).max((0,1)).tolist()
                    composition.append(dict(trajectory=name,side=side,generation=generation,index=index,
                                            max_joint_difference_deg=delta))
                    found = True;break
                if found:break
            assert found,'Selected side did not match any evaluated candidate'
    expected = json.loads(Path('artifacts/real2sim/h7-static-calibration-restart-20261105-01/usb/meta.json').read_text())['firmware_manifest']['expected_build']['sources']
    assert all(fit.audit.digest(Path(p))==h for p,h in expected.items())
    assert all(fit.audit.digest(Path(p))==h for p,h in summary['protected_sources'].items())
    regression = []
    for name, records in all_results.items():
        for branch in ('recorded_torque','recorded_target'):
            regression.append(dict(trajectory=name,branch=branch,
                baseline_joint_pass=records['baseline'][branch]['joint_thresholds_passed'],
                new_joint_pass=records['final'][branch]['joint_thresholds_passed']))
    result = dict(integrity_passed=True, results=all_results,
        all_response_thresholds_passed=all(r['final'][b]['accepted'] for r in all_results.values() for b in ('recorded_torque','recorded_target')),
        submission_formula_max_errors=dict(spring_force_n=max_curve_error,slider_friction_capacity_n=max_capacity_error,
                                           target_controller_nm=max_controller_error),
        joint_pass_comparison=regression, unchanged_firmware_source_count=len(expected),
        selected_combination_vs_batch_comparisons=composition,
        new_environment_steps=0, fitted_run_gpu_environment_steps=summary['new_gpu_environment_steps'],
        policy_training_updates=0, independent_hardware_validation=False, shared_model_modified=False)
    fit.search.write_json(args.output/'report.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,4,figsize=(16,11),sharex='col')
    for column,name in enumerate(('original','wide','slow','fast')):
        a=plot_data[name]['final'];b=plot_data[name]['baseline']
        for joint in range(4):
            ax=axes[joint,column]
            ax.plot(a['time'],np.degrees(a['real'][:,joint]),color='black',label='Measured')
            ax.plot(a['time'],np.degrees(a['predicted'][:,0,joint]),color='tab:red',label='New static-constrained fit')
            ax.plot(b['time'],np.degrees(b['predicted'][:,0,joint]),color='tab:blue',alpha=.6,ls='--',label='Previous candidate')
            if joint==0:ax.set_title(name)
            if column==0:ax.set_ylabel(('LF','LR','RF','RR')[joint]+' angle (deg)')
            if joint==3:ax.set_xlabel('Time (s)')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Recorded-torque GPU response on the original four dynamic recordings')
    fig.tight_layout();fig.savefig(args.output/'joint-response-comparison.png',dpi=140)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    a,b=plot_data['wide']['final'],plot_data['wide']['baseline']
    for joint,ax in enumerate(axes.flat):
        ax.plot(a['time'],np.degrees(a['real'][:,joint]),color='black',label='Measured')
        ax.plot(a['time'],np.degrees(a['predicted'][:,0,joint]),color='tab:red',label='New fit')
        ax.plot(b['time'],np.degrees(b['predicted'][:,0,joint]),color='tab:blue',ls='--',label='Previous candidate')
        ax.set_title(('Left front','Left rear','Right front','Right rear')[joint])
        ax.set_ylabel('Motor position (deg)');ax.set_xlabel('Time (s)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Wide round trip, identical recorded torque inputs')
    fig.tight_layout();fig.savefig(args.output/'wide-joint-response-comparison.png',dpi=140)
    print(json.dumps({k:{b:round(float(np.sqrt(np.mean(np.array(v['final'][b]['rmse_deg'])**2))),3)
        for b in ('recorded_torque','recorded_target')} for k,v in all_results.items()},indent=2))


if __name__=='__main__':
    main()
