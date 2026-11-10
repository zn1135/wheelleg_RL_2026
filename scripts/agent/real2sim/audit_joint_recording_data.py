"""Audit actual selected H7 joint-recording windows, independent of simulation fitting."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from scripts.agent.real2sim import fit_leg_repaired_loop as fit


DECODED={
    'original':Path('artifacts/real2sim/h7-leg-capture-20261104/leg-neutral-motion-05-output'),
    'wide':Path('artifacts/real2sim/h7-response-validation-20261104-01/trial-01-output'),
    'slow':Path('artifacts/real2sim/h7-leg-soft-boundary-20261104-01/powered-01-output'),
    'fast':Path('artifacts/real2sim/h7-leg-soft-boundary-20261104-01/powered-01-output')}


def metrics(reference, rows, archive_path):
    start=reference['window_start_board_s'];end=start+reference['duration_s']
    sequence=np.array([int(r['sequence']) for r in rows])
    controls={};states={};geometry={};faults=[];gated=[];can_failed=[]
    for row in rows:
        t=int(row['begin_ns'])/1e9;identifier=int(row['id']);kind=row['kind']
        if kind=='can':
            if row['flags_or_hal']!='0':can_failed.append(t)
            continue
        if row['fault']!='0' or row['strategy']!='6':faults.append([t,row['strategy'],row['fault']])
        if kind=='control':
            if int(row['flags_or_hal'])&62!=60:gated.append(t)
            controls[identifier]=[t,*[float(row[f'request_{j}_firmware_nm']) for j in range(6)]]
        elif kind=='leg_state':
            states[identifier]=[t,*[float(row[f'leg_{j}_position_rad']) for j in range(4)],
                *[float(row[f'leg_{j}_velocity_rad_s']) for j in range(4)],*[float(row[f'leg_{j}_rx_age_us']) for j in range(4)]]
        elif kind=='leg_geometry':
            geometry[identifier]=[t,*[float(row[f'leg_{j}_feedback_torque_nm']) for j in range(4)],int(row['leg_valid_flags'])]
    all_controls=np.array([r for r in controls.values() if start<=r[0]<=end])
    ids=[i for i,r in controls.items() if start<=r[0]<=end and i in states and i in geometry]
    c=np.array([controls[i] for i in ids]);s=np.array([states[i] for i in ids]);g=np.array([geometry[i] for i in ids])
    u=c[:,1:5];q=s[:,1:5];v=s[:,5:9];feedback=g[:,1:5]
    dt=np.diff(all_controls[:,0]);state_dt=np.diff(s[:,0])
    assert len(c)>100 and not faults and not gated and not can_failed
    assert np.all(np.diff(sequence)==1),'Missing output event in selected window including50ms margins'
    assert dt.max()<=.002 and state_dt.min()>0
    assert s[:,9:].max()<5000 and np.max(abs(all_controls[:,5:7]))==0.
    assert max(abs(all_controls[0,0]-start),abs(all_controls[-1,0]-end))<=.002
    # 位置记录节流到最多500Hz；使用窗外原始点确认插值覆盖，而非误当1kHz控制缺帧。
    state_times=np.array([r[0] for r in states.values()])
    assert state_times.min()<=start and state_times.max()>=end
    # 核对真正送入拟合器的1ms零阶保持力矩及未平滑位置来自这些原始记录。
    metadata=json.loads((Path(reference['session'])/'meta.json').read_text())
    config=Path(metadata['machine_config']['path'])
    assert fit.audit.digest(config)==reference['config_sha256']
    sign,_=fit.prior.replay.mirror.load_mapping(config)
    control_source=np.array(list(controls.values()))
    state_source=np.array(list(states.values()))
    with np.load(archive_path) as saved:
        times=start+saved['time_s'][:-1]
        indices=np.searchsorted(control_source[:,0],times,side='right')-1
        assert indices.min()>=0
        torque_input_error=float(abs(control_source[indices,1:7]*sign-saved['submitted_model_torque'][:,0]).max())
        assert torque_input_error<1e-8
        measured=saved['real_motor_state']
        keep=(measured[:,0]>=start+.01)&(measured[:,0]<=end-.01)
        indices=np.searchsorted(state_source[:,0],measured[keep,0])
        assert abs(state_source[indices,0]-measured[keep,0]).max()<1e-8
        pose_input_error=float(abs(state_source[indices,1:9]-measured[keep,1:9]).max())
        assert pose_input_error<1e-10
    rank=[]
    for side in (0,1):
        a=all_controls[:,1+2*side:3+2*side];sv=np.linalg.svd(a,compute_uv=False)
        rank.append(dict(side=('left','right')[side],second_to_first_singular_ratio=float(sv[1]/sv[0]),
                         effective_rank_relative_1e_5=int(np.sum(sv>sv[0]*1e-5)),
                         front_plus_rear_max_abs_nm=float(abs(a.sum(1)).max()),
                         front_minus_rear_rms_nm=float(np.sqrt(np.mean((a[:,0]-a[:,1])**2)))))
    grid=np.arange(s[0,0],s[-1,0],.001)
    position=np.column_stack([np.interp(grid,s[:,0],q[:,j]) for j in range(4)])
    speed=np.column_stack([np.interp(grid,s[:,0],v[:,j]) for j in range(4)])
    derivative=savgol_filter(position,31,3,deriv=1,delta=.001,axis=0)
    velocity=[]
    for j in range(4):
        moving=abs(derivative[:,j])>.05;moving[:100]=moving[-100:]=False
        quiet=(grid<=start+.2)
        if moving.sum()>20:
            corr=float(np.corrcoef(derivative[moving,j],speed[moving,j])[0,1])
            gain=float(derivative[moving,j]@speed[moving,j]/(derivative[moving,j]@derivative[moving,j]))
        else:corr=gain=None
        dq=abs(np.diff(q[:,j]));resolution=float(np.percentile(dq[dq>1e-7],5)) if np.any(dq>1e-7) else None
        velocity.append(dict(joint=('LF','LR','RF','RR')[j],position_span_deg=float(np.degrees(np.ptp(q[:,j]))),
            position_resolution_p5_rad=resolution,max_sample_position_jump_deg=float(np.degrees(dq.max())),
            initial_200ms_position_span_deg=float(np.degrees(np.ptp(position[quiet,j]))),
            initial_200ms_feedback_speed_range_rad_s=[float(speed[quiet,j].min()),float(speed[quiet,j].max())],
            moving_samples=int(moving.sum()),position_derivative_vs_feedback_correlation=corr,
            feedback_over_position_derivative_gain=gain,
            derivative_feedback_rmse_rad_s=float(np.sqrt(np.mean((speed[100:-100,j]-derivative[100:-100,j])**2)))))
    torque=[]
    for j in range(4):
        error=feedback[:,j]-u[:,j]
        # 此处只比较两种板端数值，不把驱动器反馈当作独立实测轴矩。
        quiet=(c[:,0]<=start+.2)
        torque.append(dict(joint=('LF','LR','RF','RR')[j],request_peak_nm=float(abs(u[:,j]).max()),
            feedback_minus_request_rmse_nm=float(np.sqrt(np.mean(error**2))),
            feedback_minus_request_peak_nm=float(abs(error).max()),
            signed_gain_no_intercept=float(u[:,j]@feedback[:,j]/(u[:,j]@u[:,j])),
            correlation=float(np.corrcoef(u[:,j],feedback[:,j])[0,1]),
            initial_mean_request_nm=float(u[quiet,j].mean()),initial_mean_feedback_nm=float(feedback[quiet,j].mean())))
    return dict(window_board_s=[start,end],samples_controls=len(all_controls),samples_paired=len(c),event_sequence_contiguous=True,
        faults=faults,gated_outputs=gated,failed_can_enqueues=can_failed,
        control_period_ms=[float(np.min(dt)*1000),float(np.median(dt)*1000),float(np.max(dt)*1000)],
        state_period_ms=[float(np.min(state_dt)*1000),float(np.median(state_dt)*1000),float(np.max(state_dt)*1000)],
        state_record_rate_hz=float((len(s)-1)/(s[-1,0]-s[0,0])),
        position_interpolation_brackets_cover_window=True,
        fitter_input_torque_max_difference_nm=torque_input_error,
        fitter_measured_state_max_difference=pose_input_error,
        max_motor_feedback_age_us=float(s[:,9:].max()),wheel_requests_zero=True,
        state_minus_control_timestamp_us=[float((s[:,0]-c[:,0]).min()*1e6),float((s[:,0]-c[:,0]).max()*1e6)],
        geometry_minus_control_timestamp_us=[float((g[:,0]-c[:,0]).min()*1e6),float((g[:,0]-c[:,0]).max()*1e6)],
        rank=rank,velocity=velocity,torque=torque,matched_id_count=len(ids))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    references={n:json.loads((Path('artifacts/real2sim/h7-static-constrained-joint-refit-20261105-02/inputs')/n/'report.json').read_text()) for n in DECODED}
    results={};sources=[]
    for decoded in dict.fromkeys(DECODED.values()):
        names=[n for n,p in DECODED.items() if p==decoded]
        source_report=json.loads((decoded/'report.json').read_text())
        first=references[names[0]];session=Path(first['session'])
        metadata=json.loads((session/'meta.json').read_text())
        assert metadata['complete'] and fit.audit.digest(session/'stream.jsonl')==metadata['stream_sha256']==source_report['stream_sha256']
        windows={n:[] for n in names}
        for row in csv.DictReader((decoded/'outputs.csv').open()):
            t=int(row['begin_ns'])/1e9
            for name in names:
                r=references[name];start=r['window_start_board_s'];end=start+r['duration_s']
                if start-.05<=t<=end+.05:windows[name].append(row)
        for n in names:
            archive=Path('artifacts/real2sim/h7-static-constrained-joint-refit-20261105-02/inputs')/n/'replay.npz'
            results[n]=metrics(references[n],windows[n],archive)
            print('Audited '+n,flush=True)
        sources.append(dict(decoded=str(decoded),decoded_report_sha256=fit.audit.digest(decoded/'report.json'),
            decoded_csv_sha256=fit.audit.digest(decoded/'outputs.csv'),
            session=str(session),stream_sha256=metadata['stream_sha256'],whole_stream_complete=source_report['complete_observed_span'],
            whole_stream_event_gaps=source_report['event_sequence_gaps'],whole_stream_drop_increase=source_report['queue_drop_increase']))
    report=dict(scope='Original selected dynamic-data quality and identification coverage; no simulation or calibration changes',
        sources=sources,results=results,new_environment_steps=0,hardware_access=False,
        limitations=['Numeric feedback/request agreement does not establish true shaft torque',
            'Rank-deficient input coverage does not prove that fitting the observed trajectories is impossible',
            '31ms offline derivative smoothing is only a consistency diagnostic, not a measured acceleration',
            'No independently measured fixture translation, contact force or shaft torque'])
    fit.search.write_json(args.output/'report.json',report)
    shutil_source=Path(__file__)
    (args.output/shutil_source.name).write_bytes(shutil_source.read_bytes())


if __name__=='__main__':main()
