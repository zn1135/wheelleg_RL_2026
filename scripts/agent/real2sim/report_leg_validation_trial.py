"""Report actual validation inputs and locate the first leg-test lockout."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import mujoco

from scripts.agent.real2sim import replay_leg_test_gpu as replay
from scripts.agent.real2sim.validate_leg_response_gpu import motion_blocks


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--decoded',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--window-start-board-s',type=float,help='Explicit stationary start after preparation; excludes the initial recovery inputs')
    args=parser.parse_args()
    meta,_,_=replay.mirror.load_session(args.session)
    decode=json.loads((args.decoded/'report.json').read_text())
    assert decode['stream_sha256']==meta['stream_sha256']
    frames=[];reader=replay.mirror.FrameParser()
    for event in replay.mirror.iter_session_events(args.session):
        if event['type']=='data':frames.extend(f for f in reader.feed(bytes.fromhex(event['data_hex'])) if int(f[152])==6)
    frames=np.asarray(frames);ft=np.array([replay.mirror.read_u32(f[8:10])/1000 for f in frames])
    geometry=[];control=[];previous=None;first_lock=None;normal_start=None
    with (args.decoded/'outputs.csv').open() as file:
        for row in csv.DictReader(file):
            if row['strategy']!='6':continue
            time=int(row['begin_ns'])/1e9
            if row['kind']=='control':control.append([time,*[float(row[f'request_{i}_firmware_nm']) for i in range(6)]])
            if row['kind']!='leg_geometry':continue
            flags=int(row['leg_valid_flags'])
            current=dict(board_s=time,lengths_m=[float(row[f'{s}_leg_length_m']) for s in ('left','right')],
                         targets_m=[float(row[f'{s}_leg_test_target_m']) for s in ('left','right')] if flags&8 else None,
                         flags=flags,fault=int(row['fault']))
            if flags&15==15 and not flags&(16|32) and normal_start is None:normal_start=time
            if flags&16 and first_lock is None:first_lock=dict(current=current,previous=previous)
            geometry.append([time,*current['lengths_m'],*(current['targets_m'] or [np.nan,np.nan]),flags])
            previous=current
    geometry=np.asarray(geometry);control=np.asarray(control)
    after=normal_start+1 if args.window_start_board_s is None else args.window_start_board_s
    moved=np.flatnonzero((ft>after)&(np.abs(frames[:,91])>1e-6))
    if not len(moved):raise ValueError('No new normal-phase input')
    origin=ft[moved[0]];start=round(origin-.5,3) if args.window_start_board_s is None else args.window_start_board_s
    end=first_lock['current']['board_s']-.12 if first_lock else ft[-1]-.12
    duration=np.floor((end-start)*1000)/1000
    blocks,ratios=motion_blocks(frames,ft,start,end)
    selected=frames[(ft>=start)&(ft<=start+duration)]
    orientations=selected[:,22:26];gravity=[];rotation=np.empty(9)
    for quat in orientations:
        mujoco.mju_quat2Mat(rotation,quat/np.linalg.norm(quat))
        gravity.append(rotation.reshape(3,3)[2].copy())
    gravity=np.asarray(gravity)
    tilt=np.degrees(np.arccos(np.clip(gravity@gravity[0],-1,1)))
    gyro=np.linalg.norm(selected[:,26:29],axis=1)
    imu=dict(max_gravity_direction_change_deg=float(tilt.max()),
             rms_gravity_direction_change_deg=float(np.sqrt(np.mean(tilt**2))),
             peak_gyro_rad_s=float(gyro.max()),rms_gyro_rad_s=float(np.sqrt(np.mean(gyro**2))),
             max_received_age_ms=float(selected[:,144].max()),
             world_translation='not measured; fixed suspended base remains an assumption')
    result=dict(stream_sha256=meta['stream_sha256'],first_lock=first_lock,
                normal_phase_start_board_s=normal_start,first_new_dial_board_s=origin,
                normal_input_blocks=blocks,two_speed_ratios=ratios,
                imu_motion=imu,
                requested_protocol_completed=bool(min(ratios)>=1.5 and first_lock is None),
                whole_stream_complete=decode['complete_observed_span'],
                scope='Actual recording and lockout report only; not GPU acceptance or parameter fitting',
                partial_replay_window=dict(start_board_s=start,duration_s=float(duration),
                                           limitation='ends before lockout; does not satisfy complete independent protocol'))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    axes[0].plot(ft-origin,frames[:,91]*1000);axes[0].set_ylabel('Actual dial rate (mm/s)')
    for side,name in enumerate(('Left','Right')):
        axes[1].plot(geometry[:,0]-origin,geometry[:,1+side]*1000,label=name+' measured')
        axes[1].plot(geometry[:,0]-origin,geometry[:,3+side]*1000,'--',label=name+' target')
    axes[1].axhline(330,color='red',linestyle=':',label='Existing protection')
    axes[1].set_ylabel('Length (mm)');axes[1].legend(ncol=2)
    for i,name in enumerate(replay.mirror.MOTORS[:4]):axes[2].plot(control[:,0]-origin,control[:,1+i],label=name)
    axes[2].set_ylabel('Requested torque (Nm)');axes[2].legend(ncol=4)
    for ax in axes:
        ax.set_xlim(-1,end-origin+2);ax.grid(alpha=.3)
        if first_lock:ax.axvline(first_lock['current']['board_s']-origin,color='red',linestyle=':')
    axes[-1].set_xlabel('Time from new normal-phase input (s)')
    fig.tight_layout();fig.savefig(args.output/'recorded-inputs.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
