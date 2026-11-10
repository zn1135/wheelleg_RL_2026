"""Check measured leg-return episodes and request directions from a completed USB capture."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.agent.real2sim import replay_leg_test_gpu as replay


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--decoded',type=Path,required=True)
    parser.add_argument('--start-board-s',type=float,required=True)
    parser.add_argument('--end-board-s',type=float,required=True,help='Before the operator intentionally exits')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    meta,_,config=replay.mirror.load_session(args.session)
    decoded=json.loads((args.decoded/'report.json').read_text())
    assert decoded['stream_sha256']==meta['stream_sha256']
    controls={};states={};geometry=[];sequences=[];blocked=[]
    with (args.decoded/'outputs.csv').open() as file:
        for row in csv.DictReader(file):
            time=int(row['begin_ns'])/1e9
            # 状态快照略早于同id的输出拍；两端留10ms仅用于完整配对和序号核对。
            if not args.start_board_s-.01<=time<=args.end_board_s+.01:continue
            sequences.append(int(row['sequence']))
            if row['kind']=='can':continue
            assert row['strategy']=='6' and row['fault']=='0','Inactive or faulted control in selected test span'
            identifier=int(row['id'])
            if row['kind']=='control':
                assert int(row['flags_or_hal'])&62==60,'Final output was gated or CAN enqueue failed'
                controls[identifier]=[time,*[float(row[f'request_{i}_firmware_nm']) for i in range(6)]]
            elif row['kind']=='leg_state':
                states[identifier]=[time,*[float(row[f'leg_{i}_position_rad']) for i in range(4)],
                                    *[float(row[f'leg_{i}_velocity_rad_s']) for i in range(4)],
                                    *[float(row[f'leg_{i}_rx_age_us']) for i in range(4)]]
            elif row['kind']=='leg_geometry':
                if not args.start_board_s<=time<=args.end_board_s:continue
                flags=int(row['leg_valid_flags'])
                if flags&16:blocked.append(time)
                assert flags&15==15 and flags&64,'Leg control is locked, invalid, or not the new version'
                geometry.append([time,identifier,*[float(row[f'{side}_leg_length_m']) for side in ('left','right')],
                                 *[float(row[f'{side}_leg_test_target_m']) for side in ('left','right')],flags])
    assert len(geometry)>2 and len(controls)>2
    assert not np.any(np.diff(sequences)!=1),'Missing output records inside selected test span'
    geometry=np.asarray(geometry);commands=np.asarray(list(controls.values()))
    assert commands[0,0]<=args.start_board_s+.002 and commands[-1,0]>=args.end_board_s-.002
    assert np.max(np.diff(commands[:,0]))<=.002,'Control gap exceeds2ms'
    paired=np.asarray([states[int(g[1])] for g in geometry])
    assert max(paired[:,9:].ravel())<5000,'Stale motor feedback'
    assert np.max(np.abs(commands[:,5:7]))==0.,'Nonzero wheel request'
    assert np.max(np.abs(commands[:,1:5]))<=40.001,'Motor request exceeds40Nm'
    kp,kd=replay.gains(config);forces=[];reconstruction_error=[];geometry_error=[]
    for g,state in zip(geometry,paired):
        request=np.asarray(controls[int(g[1])][1:5]);force_row=[]
        for side in range(2):
            length,jac,velocity=replay.leg_state(state[1+2*side:3+2*side],state[5+2*side:7+2*side])
            force=kp[side]*(g[4+side]-length)-kd[side]*velocity
            if int(g[6])&32:force=min(force,0.)
            if length>=.33:force=min(force,0.)
            elif length<=.14:force=max(force,0.)
            expected=np.clip(jac*force,-40,40)
            reconstruction_error.extend(expected-request[2*side:2*side+2])
            geometry_error.append(length-g[2+side])
            # 这是请求的等效轴向力，不能当独立测得的实际腿部受力。
            force_row.append(float(jac@request[2*side:2*side+2]/(jac@jac)))
        forces.append(force_row)
    forces=np.asarray(forces)
    assert np.max(np.abs(reconstruction_error))<.02,'Controller requests do not match the reconstructed new law'
    assert np.max(np.abs(geometry_error))<1e-4,'Recorded leg geometry differs from the firmware-coordinate reconstruction'
    normal=(geometry[:,6].astype(int)&32)==0;events=[];boundary_stats={}
    for side,name in enumerate(('left','right')):
        length=geometry[:,2+side];upper=length>=.33;lower=length<=.14
        high_violation=float(max(0.,forces[upper,side].max())) if np.any(upper) else 0.
        low_violation=float(max(0.,-forces[lower,side].min())) if np.any(lower) else 0.
        assert max(high_violation,low_violation)<.1,'Boundary request points farther out of the working domain'
        boundary_stats[name]=dict(upper_samples=int(upper.sum()),lower_samples=int(lower.sum()),
                                  max_outward_request_force_n=high_violation,max_compressing_request_force_n=low_violation,
                                  initial_overlength_returned=bool(upper[0] and np.any(normal & ~upper)))
        for mask,boundary in ((upper,'upper'),(lower,'lower')):
            active=mask&normal
            starts=np.flatnonzero(active & ~np.r_[False,active[:-1]])
            for first in starts:
                end=first
                while end+1<len(active) and active[end+1]:end+=1
                returned=end+1<len(active) and normal[end+1] and not mask[end+1]
                crossed=first>0 and normal[first-1] and not mask[first-1]
                events.append(dict(side=name,boundary=boundary,start_board_s=float(geometry[first,0]),
                                   duration_s=float(geometry[end,0]-geometry[first,0]),
                                   crossed_from_normal=bool(crossed),returned_without_lockout=bool(returned),
                                   maximum_excursion_m=float(max(length[first:end+1]-.33)) if boundary=='upper' else float(max(.14-length[first:end+1]))))
    result=dict(scope='Real capture: request-direction and measured return check; not independent shaft-force, joint-limit, or general robot acceptance',
                stream_sha256=meta['stream_sha256'],window_board_s=[args.start_board_s,args.end_board_s],
                new_controller_recorded=True,sequence_contiguous=True,locked_samples=blocked,
                controller_max_error_nm=float(np.max(np.abs(reconstruction_error))),
                geometry_max_error_m=float(np.max(np.abs(geometry_error))),max_motor_feedback_age_us=float(paired[:,9:].max()),
                max_requested_torque_nm=float(np.max(np.abs(commands[:,1:5]))),wheel_requests_zero=True,
                initial_return_passed=all(s['initial_overlength_returned'] for s in boundary_stats.values()),
                normal_boundary_return_observed=any(e['crossed_from_normal'] and e['returned_without_lockout'] for e in events),
                boundary_stats=boundary_stats,normal_boundary_events=events,
                limitations=['No observation of a boundary crossing means that boundary remains unverified',
                             'No actual shaft torque or external world-translation measurement',
                             'Startup queue loss outside selected span does not affect this span check'])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
