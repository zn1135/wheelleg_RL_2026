"""Score a frozen candidate before lockout without claiming a complete protocol."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.agent.real2sim.validate_leg_response_gpu import sha
from scripts.agent.real2sim import replay_leg_test_gpu as replay


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--trial-report',type=Path,required=True)
    parser.add_argument('--frozen-candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    freeze=json.loads((args.frozen_candidate/'preparation.json').read_text())
    report=json.loads((args.replay/'report.json').read_text())
    trial=json.loads((args.trial_report/'report.json').read_text())
    meta,asset,config=replay.mirror.load_session(Path(report['session']))
    assert report['stream_sha256']==trial['stream_sha256']==meta['stream_sha256']!=freeze['fit_stream_sha256']
    assert datetime.fromisoformat(meta['started_utc'])>datetime.fromisoformat(freeze['frozen_utc'])
    assert report['response_parameters_sha256']==freeze['response_parameters_sha256']
    assert sha(asset)==freeze['model_sha256'] and sha(config)==freeze['config_sha256']
    assert report['training_spring_source_sha256']==freeze['spring_source_sha256']
    assert all(sha(path)==digest for path,digest in freeze['validation_sources'].items())
    assert report['physics_backend']=='mujoco_warp_gpu' and report['reused_replay'] is None
    assert report['replay_window_has_no_output_sequence_gaps']
    model=mujoco.MjModel.from_xml_path(str(args.replay/'fixed-base-model.xml'))
    motors=[model.joint(n).qposadr[0] for n in replay.mirror.MOTORS[:4]]
    sign,zero=replay.mirror.load_mapping(config)
    with np.load(args.replay/'replay.npz') as data:
        times=report['window_start_board_s']+data['time_s'];state=data['real_motor_state']
        measured=np.column_stack([np.interp(times,state[:,0],state[:,1+i]) for i in range(4)])
        prediction=np.take(data['qpos'],motors,axis=2)*sign[:4]+zero
    error=np.degrees(prediction-measured[:,None,:]);criteria=freeze['acceptance'];metrics={}
    for branch,name in enumerate(('recorded_torque','recorded_target_feedback')):
        rmse=np.sqrt(np.mean(error[:,branch]**2,axis=0));peak=np.max(np.abs(error[:,branch]),axis=0)
        final=np.abs(error[-1,branch]);pin=np.asarray(report['metrics'][name]['max_loop_pin_residual_m'])*1000
        blocks=[]
        for block in trial['normal_input_blocks']:
            mask=(times>=block['start_board_s'])&(times<=block['end_board_s']+.2)
            if not np.any(mask):continue
            blocks.append(dict(**block,motor_rmse_deg=np.sqrt(np.mean(error[mask,branch]**2,axis=0)).tolist()))
        metrics[name]=dict(motor_rmse_deg=rmse.tolist(),combined_rmse_deg=float(np.sqrt(np.mean(error[:,branch]**2))),
                           peak_error_deg=peak.tolist(),final_error_deg=final.tolist(),max_pin_error_mm=pin.tolist(),
                           blocks=blocks,angular_and_closure_thresholds_passed=bool(np.max(rmse)<criteria['rmse_deg']
                           and np.max(peak)<criteria['peak_deg'] and np.max(final)<criteria['final_deg']
                           and np.max(pin)<criteria['pin_error_mm']))
    result=dict(scope='Independent pre-lockout response assessment only; no refit or protocol acceptance',
                complete_protocol_passed=False,interrupted_by_physical_lockout=trial['first_lock'],
                candidate_frozen_before_capture=True,parameters_refitted=False,
                response_thresholds_passed=all(m['angular_and_closure_thresholds_passed'] for m in metrics.values()) and not report['feedback_branch_blocked'],
                stream_sha256=meta['stream_sha256'],candidate_sha256=freeze['response_parameters_sha256'],
                window_start_board_s=report['window_start_board_s'],duration_s=report['duration_s'],
                criteria=criteria,metrics=metrics,feedback_branch_blocked=report['feedback_branch_blocked'],
                measured_motor_excursion_deg=np.ptp(np.degrees(measured),axis=0).tolist(),
                physics_backend=report['physics_backend'],total_environment_steps=report['total_environment_steps'],
                limitations=['No full two-speed round-trip protocol completion or terminal hold',
                             'No post-lockout physical response replay','Fixed body only; no free-body/ground acceptance'])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2))
    for name,m in metrics.items():print(name,'RMSEdeg',m['motor_rmse_deg'],'peakdeg',m['peak_error_deg'])


if __name__=='__main__':main()
