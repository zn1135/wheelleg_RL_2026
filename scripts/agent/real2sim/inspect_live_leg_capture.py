"""Inspect the tail of a running capture; never opens USB or sends commands."""
import argparse
import json
from pathlib import Path
import struct

from scripts.agent.real2sim import replay_leg_test_gpu as replay
from h7_output_capture import CaptureParser


def inspect(session):
    path=session/'stream.jsonl'
    with path.open('rb') as file:
        offset=max(0,path.stat().st_size-524288)
        file.seek(offset)
        if offset:file.readline()
        lines=file.read().splitlines()
    mirrors=replay.mirror.FrameParser();outputs=CaptureParser();frames=[];records=[]
    for line in lines:
        try:event=json.loads(line)
        except json.JSONDecodeError:continue  # 写盘时最后一行可能尚未结束。
        if event['type']=='data':
            raw=bytes.fromhex(event['data_hex']);frames.extend(mirrors.feed(raw))
            for packet in outputs.feed(raw):records.extend(packet['records'])
    if not frames or not records:raise ValueError('No complete recent mirror/output packets')
    frame=frames[-1];geometry=[r for r in records if r[1]==4][-1]
    control=[r for r in records if r[1]==1][-1]
    target_records=[r for r in records if r[1]==5]
    target=target_records[-1] if target_records and control[21]==7 else None
    return dict(board_s=replay.mirror.read_u32(frame[8:10])/1000,
                joint_capture=(dict(target_firmware_rad=list(target[6:10]),kp=target[10],kd=target[11],limit_nm=target[12],
                                    elapsed_ms=target[14],stage=target[22],axis=target[23],flags=target[20],
                                    record_board_s=target[2]/1e9) if target is not None else None),
                rc_switches=frame[89:91].tolist(),dial_rate_m_s=float(frame[91]),arm_state=int(frame[92]),
                strategy=int(frame[152]),fault=int(frame[2]),request_nm=list(control[6:12]),
                motor_positions_rad=frame[10:14].tolist(),motor_velocities_rad_s=frame[14:18].tolist(),
                motor_feedback_torques_nm=list(geometry[6:10]),
                motor_ages_ms=frame[138:142].tolist(),geometry_flags=geometry[20],
                lengths_m=list(geometry[11:13]),
                angles_rad=[geometry[13],struct.unpack('<f',struct.pack('<I',geometry[14]))[0]],
                targets_m=struct.unpack('<2f',struct.pack('<4h',*geometry[16:20])) if geometry[20]&8 else None,
                stream_bytes=path.stat().st_size,
                limitation='Tail inspection only; whole-window continuity and freshness need formal decoding')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=inspect(args.session)
    content=json.dumps(result,indent=2)+'\n'
    if args.output:
        with args.output.open('x') as file:file.write(content)
    print(content)


if __name__=='__main__':main()
