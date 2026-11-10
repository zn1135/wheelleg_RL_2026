"""Freeze a longer same-trial torque/feedback window for post-fit free GPU validation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.agent.real2sim import replay_leg_test_gpu as replay


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--decoded', type=Path, required=True)
    parser.add_argument('--duration-s', type=float, default=23.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.reference/'report.json').read_text())
    decoded = json.loads((args.decoded/'report.json').read_text())
    assert decoded['stream_sha256'] == report['stream_sha256']
    start = report['window_start_board_s'];end = start + args.duration_s
    assert args.duration_s > report['duration_s']
    control=[];states={};geometry=[];sequences=[]
    with (args.decoded/'outputs.csv').open() as file:
        for row in csv.DictReader(file):
            t=int(row['begin_ns'])/1e9
            if not start-.1 <= t <= end+.1:
                continue
            sequences.append(int(row['sequence']))
            if row['kind']=='can':
                continue
            assert row['strategy']=='6' and row['fault']=='0'
            key=int(row['id'])
            if row['kind']=='control':
                assert int(row['flags_or_hal'])&62 == 60
                control.append([t,*[float(row[f'request_{i}_firmware_nm']) for i in range(6)]])
            elif row['kind']=='leg_state':
                assert max(float(row[f'leg_{i}_rx_age_us']) for i in range(4)) < 5000
                states[key]=[t,*[float(row[f'leg_{i}_position_rad']) for i in range(4)],
                             *[float(row[f'leg_{i}_velocity_rad_s']) for i in range(4)]]
            elif row['kind']=='leg_geometry':
                assert int(row['leg_valid_flags'])&15 == 15
                geometry.append([t,key,float(row['left_leg_length_m']),float(row['right_leg_length_m']),
                                 float(row['left_leg_test_target_m']),float(row['right_leg_test_target_m']),int(row['leg_valid_flags'])])
    assert not np.any(np.diff(sequences)!=1)
    control=np.asarray(control);geometry=np.asarray(geometry)
    assert np.max(np.diff(control[:,0])) <= .002
    paired=np.asarray([states[int(g[1])] for g in geometry])
    meta=json.loads((Path(report['session'])/'meta.json').read_text())
    config=Path(meta['machine_config']['path'])
    assert hashlib.sha256(config.read_bytes()).hexdigest()==report['config_sha256']
    sign,_=replay.mirror.load_mapping(config)
    steps=round(args.duration_s/report['dt_s'])
    times=np.arange(steps+1)*report['dt_s']
    indices=np.searchsorted(control[:,0],start+times[:-1],side='right')-1
    assert indices.min() >= 0
    torque=control[indices,1:]*sign
    old=np.load(args.reference/'replay.npz')
    assert np.array_equal(torque[:len(old['submitted_model_torque'])].astype(np.float32),
                          old['submitted_model_torque'][:,0].astype(np.float32))
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'replay.npz',time_s=times,qpos=old['qpos'][:1],qvel=old['qvel'][:1],
                        submitted_model_torque=torque[:,None,:],real_motor_state=paired,real_geometry=geometry)
    result={key:report[key] for key in ('spring_model','constraint_check','source_model','source_model_sha256',
                                       'config_sha256','stream_sha256','window_start_board_s','dt_s','session')}
    result.update(input_only=True,physics_steps=0,duration_s=args.duration_s,
                  sequence_contiguous=True,original_prefix_torques_identical=True,
                  max_control_gap_s=float(np.max(np.diff(control[:,0]))),source_reference=str(args.reference))
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(args.reference/'fixed-base-model.xml',args.output/'fixed-base-model.xml')
    shutil.copy2(Path(__file__),args.output/Path(__file__).name)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
