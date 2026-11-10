"""Freeze measured length-only replay inputs and verify the archived firmware controller."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.agent.real2sim.precheck_right_leg_pose import State, mirror


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3]
    audit=json.loads((args.audit/'report.json').read_text())
    meta=json.loads((args.session/'meta.json').read_text())
    manifest=json.loads((args.release/'firmware-manifest.json').read_text())
    assert meta['complete'] and len(audit['runs'])==1
    assert not audit['sequence_gap_count'] and not audit['crc_rejected']
    assert manifest['candidate']['sha256']==meta['firmware_manifest']['expected_build']['candidate']['sha256']
    with np.load(args.audit/'paired-records.npz') as source: rows={k:source[f'kind{k}'] for k in (1,3,4,5)}
    maps={k:{int(row[4]):row for row in values} for k,values in rows.items()}
    ids=sorted(set.intersection(*(set(values) for values in maps.values())),key=lambda key:maps[5][key][14])
    control,state,geometry,target=[np.array([maps[k][key] for key in ids]) for k in (1,3,4,5)]
    assert len(ids)==len(rows[5]) and np.all(np.diff(target[:,14])==2) and target[0,14]==0
    assert np.all((target[:,20].astype(int)&31)==17) and not np.any(control[:,[6,7,10,11]])
    assert not np.any(control[:,5]) and not np.any((target[:,20].astype(int)>>8)&15)
    stream=args.session/'stream.jsonl';digest=hashlib.sha256();frame_parser=mirror.FrameParser();frames=[]
    with stream.open('rb') as file:
        file.seek(audit['stream_window_start_byte'])
        while file.tell()<audit['stream_window_end_byte']:
            line=file.readline();assert file.tell()<=audit['stream_window_end_byte']
            digest.update(line);event=json.loads(line)
            if event['type']=='data': frames.extend(frame_parser.feed(bytes.fromhex(event['data_hex'])))
    assert digest.hexdigest()==audit['stream_window_sha256']
    frame=np.asarray(frames);ft=np.array([mirror.read_u32(row[8:10])/1000 for row in frame])
    assert np.all(np.diff(ft)>0)
    config_relative='deployment/H7_RL_ros/imcalib/config/machine_config.c'
    config=args.release/'source'/config_relative
    assert hashlib.sha256(config.read_bytes()).hexdigest()==manifest['sources'][config_relative]
    recorder_config=root/meta['machine_config']['project_path']
    assert hashlib.sha256(recorder_config.read_bytes()).hexdigest()==meta['machine_config']['sha256']
    recorder_mapping=mirror.load_mapping(recorder_config)
    actual_mapping=mirror.load_mapping(config)
    assert all(np.array_equal(a,b) for a,b in zip(recorder_mapping,actual_mapping))
    assert hashlib.sha256(mirror.MODEL.read_bytes()).hexdigest()==meta['model']['sha256']
    pose=mirror.MirrorPose(machine_config=config)
    reference=root/'artifacts/pace-joint-capture/20261108-lowfreq-load-fit-01'
    model=mujoco.MjModel.from_xml_path(str(reference/'gpu-replay/model.xml'))
    model.geom_contype[:]=model.geom_conaffinity[:]=0
    mujoco.mj_saveLastXML(str(args.output/'model.xml'),model)
    q0=pose.sign[:4]*(state[0,6:10]-pose.zero)
    for side in (0,1): assert pose._solve_leg(side,q0[2*side:2*side+2])
    initial_q=np.array([pose.data.qpos[pose.model.joint(model.joint(j).name).qposadr[0]] for j in range(model.njnt)])
    initial_v=np.zeros(model.nv)  # 首帧静止；原速度±.01099rad/s为12bit半格，不当作真实初速。
    mq=np.array([model.joint(n).qposadr[0] for n in mirror.MOTORS[:4]])
    mv=np.array([model.joint(n).dofadr[0] for n in mirror.MOTORS[:4]])
    sign=pose.sign[:4];zero=pose.zero
    real_q=(state[:,6:10]-zero)*sign;real_v=state[:,10:14]*sign
    desired_q=(target[:,6:10]-zero)*sign;actual_tau=control[:,6:10]*sign
    age=np.column_stack((state[:,14:16],np.array([np.frombuffer(np.array(row[16:20],dtype='<i2').tobytes(),dtype='<u4') for row in state])))
    age+=((control[:,2]-state[:,2])/1000)[:,None]
    assert np.max(age)<5000 and np.min(age)>=0
    bt=target[:,2]/1e9
    quat=frame[:,22:26].copy()
    for i in range(1,len(quat)):
        if quat[i]@quat[i-1]<0: quat[i]*=-1
    qinterp=np.column_stack([np.interp(bt,ft,quat[:,j]) for j in range(4)])
    qinterp/=np.linalg.norm(qinterp,axis=1,keepdims=True)
    body_rotation=Rotation.from_quat(qinterp[:,[1,2,3,0]]).as_matrix()
    model_rotation=Rotation.from_quat(model.body('base_link').quat[[1,2,3,0]]).as_matrix()
    gravity=np.einsum('tji,j->ti',body_rotation,model.opt.gravity)@model_rotation.T
    direction=gravity/np.linalg.norm(gravity,axis=1,keepdims=True)
    gravity_span=float(np.degrees(np.max(np.arccos(np.clip(direction@direction[0],-1,1)))))
    algorithm=args.release/'source/deployment/H7_RL_ros/imcalib/Algorithm'
    for name in ('joint_capture.c','joint_capture.h','joint_chirp.c','joint_chirp.h','joint_pose_load.h'):
        assert hashlib.sha256((algorithm/name).read_bytes()).hexdigest()==manifest['sources']['deployment/H7_RL_ros/imcalib/Algorithm/'+name]
    f4=ctypes.c_float*4;s6=ctypes.c_int8*6
    ff=np.zeros_like(desired_q);desired_v=np.zeros_like(desired_q);request_error=target_error=0.
    with tempfile.TemporaryDirectory() as temp:
        library=Path(temp)/'controller.so'
        subprocess.run([os.environ.get('CC','cc'),'-std=c11','-O2','-shared','-fPIC',str(algorithm/'joint_capture.c'),str(algorithm/'joint_chirp.c'),'-lm','-o',str(library)],check=True)
        lib=ctypes.CDLL(str(library));generate=lib.Joint_Pose_Target;update=lib.Joint_Capture_Update
        generate.argtypes=[ctypes.c_uint32,ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_int8),ctypes.POINTER(ctypes.c_float),ctypes.c_float,ctypes.c_float,*([ctypes.POINTER(ctypes.c_float)]*3),ctypes.POINTER(ctypes.c_uint8)]
        update.argtypes=[ctypes.POINTER(State),ctypes.c_uint32,ctypes.c_uint8,*([ctypes.POINTER(ctypes.c_float)]*3),ctypes.POINTER(ctypes.c_int8),*([ctypes.POINTER(ctypes.c_float)]*3),ctypes.c_float,ctypes.c_float,ctypes.c_float,ctypes.POINTER(ctypes.c_float)]
        cstate=State();previous=control[0,6:10].copy();sgn=s6(*pose.sign.astype(int));z=f4(*zero)
        for i in range(len(bt)):
            tgt,vel,load,stage=f4(),f4(),f4(),ctypes.c_uint8()
            if i and target[i,14]>=25000:
                ff[i]=ff[i-1];desired_v[i]=desired_v[i-1]
            else:
                assert generate(int(target[i,14]),f4(*q0),sgn,z,.21,.25,tgt,vel,load,ctypes.byref(stage))
                ff[i]=load[:];desired_v[i]=vel[:]
                target_error=max(target_error,float(abs(np.array(tgt[:])-desired_q[i]).max()))
            output=f4()
            assert update(ctypes.byref(cstate),int(target[i,14]),1,f4(*state[i,6:10]),f4(*state[i,10:14]),f4(*previous),sgn,z,f4(60,60,60,60),f4(2,2,2,2),.21,.25,40.,output)
            request_error=max(request_error,float(abs(np.array(output[:])-control[i,6:10]).max()))
            previous=control[i,6:10]
    assert target_error<2e-6 and request_error<.01,(target_error,request_error)
    # 导出量化格只引用首帧与已有MIT量程；不更改实机零点、极性或量程。
    inputs=dict(time_s=target[:,14]/1000,board_s=bt,real_q=real_q,real_v=real_v,recorded_tau=actual_tau,
                target_q=desired_q,target_v=desired_v,feedforward=ff,initial_q=initial_q,initial_v=initial_v,
                initial_request=actual_tau[0],initial_sensor_q=real_q[0],initial_sensor_v=real_v[0],
                delay_s=age*1e-6,gravity=gravity,initial_q0=q0,sign=sign,zero=zero,
                position_quantum=np.full(4,2*3.14159/65535),velocity_quantum=np.full(4,2*45/4095),
                torque_quantum=np.full(4,2*54/4095),real_length=geometry[:,11:13],
                real_angle=np.column_stack((geometry[:,13],np.array([np.frombuffer(np.uint32(v).tobytes(),dtype='<f4')[0] for v in geometry[:,14]]))),mq=mq,mv=mv)
    np.savez_compressed(args.output/'inputs.npz',**inputs)
    shutil.copy2(reference/'parameters.json',args.output/'baseline-parameters.json')
    report=dict(scope='Frozen real inputs; C reconstruction only, no physics or hardware',samples=len(bt),duration_s=float(inputs['time_s'][-1]),
                source_window_sha256=audit['stream_window_sha256'],source_session=str(args.session.resolve()),source_audit=str(args.audit.resolve()),
                source_release=str(args.release.resolve()),input_npz_sha256=hashlib.sha256((args.output/'inputs.npz').read_bytes()).hexdigest(),
                controller_request_reconstruction_max_nm=request_error,target_reconstruction_max_rad=target_error,
                recorded_gravity_span_deg=gravity_span,feedback_delay_range_ms=[float(age.min()/1000),float(age.max()/1000)],
                recorder_config_metadata_mismatch=meta['machine_config']['sha256']!=manifest['sources'][config_relative],
                recorder_and_flashed_model_sign_zero_identical=True,actual_config_sha256=manifest['sources'][config_relative],
                initial_velocity_policy='stationary zero; raw velocity half-code retained in fixed feedback quantizer',
                protected_sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (mirror.MODEL,config)},new_physics_steps=0)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copy2(__file__,args.output/Path(__file__).name)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
