"""Validate a frozen response candidate on a new bidirectional USB recording."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

from scripts.agent.real2sim import replay_leg_test_gpu as replay


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def motion_blocks(frames, times, start, end):
    """Only classify actual dial input; never invent new hardware commands."""
    selected=(times>=start)&(times<=end)
    times=times[selected];rates=frames[selected,91]
    directions=np.sign(rates)
    boundaries=np.r_[0,np.flatnonzero(np.diff(directions)!=0)+1,len(times)]
    blocks=[]
    for first,last in zip(boundaries[:-1],boundaries[1:]):
        if directions[first]==0 or times[last-1]-times[first]<.2:
            continue
        blocks.append(dict(start_board_s=float(times[first]),end_board_s=float(times[last-1]),
                           direction=int(directions[first]),
                           median_target_rate_m_s=float(np.median(rates[first:last])),
                           duration_s=float(times[last-1]-times[first])))
    # Each direction must contain both a slow and a fast segment.
    speed_ratios=[]
    for direction in (-1,1):
        speeds=[abs(block['median_target_rate_m_s']) for block in blocks if block['direction']==direction]
        speed_ratios.append(max(speeds)/min(speeds) if len(speeds)>=2 else 0.)
    return blocks,speed_ratios


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--decoded',type=Path,required=True)
    parser.add_argument('--frozen-candidate',type=Path,required=True,help='Prepared directory created before recording')
    parser.add_argument('--window-start-board-s',type=float,required=True)
    parser.add_argument('--duration-s',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    freeze=json.loads((args.frozen_candidate/'preparation.json').read_text())
    parameters=args.frozen_candidate/'frozen-response-parameters.json'
    assert sha(parameters)==freeze['response_parameters_sha256'],'Frozen candidate was changed'
    assert not freeze['parameter_refit_allowed']
    for path,digest in freeze['validation_sources'].items():
        assert sha(path)==digest,f'Validation source changed since freeze: {path}'
    meta,asset,config,input_report,frames,ft,control,geometry,paired,origin,start,end=replay.read_data(
        args.session,args.decoded,args.window_start_board_s,args.duration_s)
    assert meta['stream_sha256']!=freeze['fit_stream_sha256'],'This is the fitting recording'
    assert datetime.fromisoformat(meta['started_utc'])>datetime.fromisoformat(freeze['frozen_utc']), 'Recording predates the candidate freeze'
    assert sha(asset)==freeze['model_sha256'] and sha(config)==freeze['config_sha256']
    assert sha(Path(replay.training_spring.__file__))==freeze['spring_source_sha256']
    assert meta['firmware_manifest']['file']['sha256']==sha(args.frozen_candidate/'firmware-manifest.json')
    # Normal control only: overlong-leg recovery cannot validate extension response.
    assert not np.any(geometry[:,6].astype(int)&32),'Complete overlong-leg recovery before selecting this window'
    assert np.max(np.abs(control[:,6:8]))<1e-8,'Wheels must remain at zero request'
    assert np.max(np.abs(control[:,2:6]))<=40.001
    blocks,ratios=motion_blocks(frames,ft,start,end)
    assert min(ratios)>=1.5,'Need two distinct speeds in each direction in this continuous window'
    assert end-max(block['end_board_s'] for block in blocks)>=2.,'Keep the final dial neutral for at least2s'
    first=max(0,np.searchsorted(paired[:,0],start,side='right')-1)
    quiet=paired[(paired[:,0]>=start)&(paired[:,0]<=start+.2),5:9]
    assert len(quiet)>1 and np.max(np.abs(quiet))<.05,'Start from a stationary hold for friction initialization'
    assert np.max(np.abs(paired[first,5:9]))<.05
    args.output.mkdir(parents=True,exist_ok=False)
    inputs=dict(stream_sha256=meta['stream_sha256'],candidate_sha256=sha(parameters),
                start_board_s=start,end_board_s=end,blocks=blocks,speed_ratios=ratios,
                parameters_refitted=False,sequence_contiguous=True,
                whole_stream_complete=input_report['complete_observed_span'])
    (args.output/'validated-inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
    command=[sys.executable,'-m','scripts.agent.real2sim.replay_leg_test_gpu',
             '--session',str(args.session),'--decoded',str(args.decoded),
             '--output',str(args.output/'gpu'),'--spring-model','linear-damped',
             '--constraint-check','stiff','--response-parameters',str(parameters),
             '--window-start-board-s',str(start),'--duration-s',str(args.duration_s)]
    with (args.output/'gpu-console.log').open('w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=replay.ROOT)
    report=json.loads((args.output/'gpu/report.json').read_text())
    assert report['physics_backend']=='mujoco_warp_gpu' and report['duration_s']==end-start
    assert report['response_parameters_sha256']==sha(parameters)
    model=mujoco.MjModel.from_xml_path(str(args.output/'gpu/fixed-base-model.xml'))
    motors=[model.joint(name).qposadr[0] for name in replay.mirror.MOTORS[:4]]
    sign,zero=replay.mirror.load_mapping(config)
    with np.load(args.output/'gpu/replay.npz') as archive:
        times=start+archive['time_s'];state=archive['real_motor_state']
        measured=np.column_stack([np.interp(times,state[:,0],state[:,1+i]) for i in range(4)])
        predicted=np.take(archive['qpos'],motors,axis=2)*sign[:4]+zero
    error=np.degrees(predicted-measured[:,None,:])
    metrics={};criteria=freeze['acceptance']
    for branch,name in enumerate(('recorded_torque','recorded_target_feedback')):
        rmse=np.sqrt(np.mean(error[:,branch]**2,axis=0))
        peak=np.max(np.abs(error[:,branch]),axis=0);final=np.abs(error[-1,branch])
        closure=np.asarray(report['metrics'][name]['max_loop_pin_residual_m'])*1000
        per_block=[]
        for block in blocks:
            mask=(times>=block['start_board_s'])&(times<=block['end_board_s']+.5)
            per_block.append(dict(**block,motor_rmse_deg=np.sqrt(np.mean(error[mask,branch]**2,axis=0)).tolist()))
        metrics[name]=dict(motor_rmse_deg=rmse.tolist(),peak_error_deg=peak.tolist(),
                           final_error_deg=final.tolist(),max_pin_error_mm=closure.tolist(),blocks=per_block,
                           passed=bool(np.max(rmse)<criteria['rmse_deg'] and np.max(peak)<criteria['peak_deg']
                                       and np.max(final)<criteria['final_deg'] and np.max(closure)<criteria['pin_error_mm']))
    passed=all(m['passed'] for m in metrics.values()) and not report['feedback_branch_blocked']
    result=dict(scope='Independent fixed-candidate bidirectional speed validation; no refit, hardware command or training',
                passed=passed,inputs=inputs,criteria=criteria,metrics=metrics,
                measured_motor_excursion_deg=np.ptp(np.degrees(measured),axis=0).tolist(),
                weakly_excited_motors=[name for name,span in zip(replay.mirror.MOTORS[:4],np.ptp(np.degrees(measured),axis=0)) if span<1.],
                feedback_branch_blocked=report['feedback_branch_blocked'],
                physics_backend=report['physics_backend'],total_environment_steps=report['total_environment_steps'],
                limitations=['Fixed suspended body only; not ground/free-body acceptance',
                             'Effective fitted parameters are not independent physical measurements',
                             'Motors with less than1deg excursion do not establish their response across travel'])
    (args.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
