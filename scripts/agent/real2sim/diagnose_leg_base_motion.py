"""Quantify recorded support rotation in inverse loads; no integration or leg forcing."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import savgol_filter

from scripts.agent.real2sim import replay_leg_test_gpu as replay


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--inverse',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report=json.loads((args.reference/'report.json').read_text());session=Path(report['session'])
    parser_wire=replay.mirror.FrameParser();samples=[]
    for event in replay.mirror.iter_session_events(session):
        if event['type']!='data':continue
        for frame in parser_wire.feed(bytes.fromhex(event['data_hex'])):
            t=replay.mirror.read_u32(frame[8:10])/1000
            if report['window_start_board_s']-.1<=t<=report['window_start_board_s']+report['duration_s']+.1:
                samples.append([t,*frame[22:29]])
    samples=np.asarray(samples);source=np.load(args.inverse/'inverse-loads.npz')
    t=source['time_s']+report['window_start_board_s'];q=source['qpos'];v=source['qvel'];a=source['qacc']
    quat=np.column_stack([np.interp(t,samples[:,0],samples[:,1+j]) for j in range(4)]);quat/=np.linalg.norm(quat,axis=1,keepdims=True)
    gyro=np.column_stack([np.interp(t,samples[:,0],samples[:,5+j]) for j in range(3)])
    gyro=savgol_filter(gyro,11,3,axis=0);alpha=savgol_filter(gyro,11,3,deriv=1,delta=.01,axis=0)
    quaternion_rate=[]
    for first,second in zip(quat[:-1],quat[1:]):
        angular=np.empty(3);mujoco.mju_subQuat(angular,second,first);quaternion_rate.append(angular/.01)
    quaternion_rate=np.asarray(quaternion_rate)
    model=mujoco.MjModel.from_xml_path(report['source_model']);model.opt.jacobian=mujoco.mjtJacobian.mjJAC_DENSE
    model.geom_contype[:]=model.geom_conaffinity[:]=0
    mv=[model.joint(n).dofadr[0] for n in replay.mirror.MOTORS[:4]]
    model.dof_armature[mv]=.015795;model.dof_damping[mv]=0.;model.dof_frictionloss[mv]=0.
    mujoco.mj_setConst(model,mujoco.MjData(model));data=mujoco.MjData(model)
    fixed=mujoco.MjModel.from_xml_path(str(args.reference/'fixed-base-model.xml'))
    joints=[fixed.joint(j).name for j in range(fixed.njnt)]
    wheel=[model.joint(n).dofadr[0] for n in replay.mirror.MOTORS[4:]]
    passive=[i for i in range(6,model.nv) if i not in mv+wheel]
    delta=[];base_required=[]
    for index in range(len(q)):
        data.qpos[:3]=[0.,0.,.35];data.qpos[3:7]=quat[index];data.qvel[:]=0.
        acceleration=np.zeros(model.nv)
        for name in joints:
            src=fixed.joint(name);dst=model.joint(name)
            data.qpos[dst.qposadr[0]]=q[index,src.qposadr[0]]
            data.qvel[dst.dofadr[0]]=v[index,src.dofadr[0]]
            acceleration[dst.dofadr[0]]=a[index,src.dofadr[0]]
        mujoco.mj_forward(model,data)
        rows=data.efc_type[:data.nefc]==mujoco.mjtConstraint.mjCNSTR_EQUALITY
        jac=data.efc_J.reshape(data.nefc,model.nv)[rows]
        tangent=np.zeros((model.nv,4));tangent[mv]=np.eye(4)
        tangent[passive]=np.linalg.lstsq(jac[:,passive],-jac[:,mv],rcond=None)[0]
        force=np.empty(model.nv);mujoco.mj_mulM(model,data,force,acceleration)
        nominal=tangent.T@(force+data.qfrc_bias-data.qfrc_passive)
        data.qvel[3:6]=gyro[index];acceleration[3:6]=alpha[index]
        mujoco.mj_forward(model,data);mujoco.mj_mulM(model,data,force,acceleration)
        actual=tangent.T@(force+data.qfrc_bias-data.qfrc_passive)
        delta.append(actual-nominal);base_required.append(actual)
    delta=np.asarray(delta)
    result=dict(scope='measured base-rotation inverse-load sensitivity; translation absent; gyro may include vibration/noise',
                environment_steps=0,recording_sha256=report['stream_sha256'],
                source_model_sha256=hashlib.sha256(Path(report['source_model']).read_bytes()).hexdigest(),
                relative_motor_load_rms_nm=np.sqrt(np.mean(delta**2,axis=0)).tolist(),
                relative_motor_load_abs_p95_nm=np.percentile(abs(delta),95,axis=0).tolist(),relative_motor_load_max_nm=abs(delta).max(0).tolist(),
                smoothed_gyro_rms_rad_s=np.sqrt(np.mean(gyro**2,axis=0)).tolist(),
                quaternion_rate_rms_rad_s=np.sqrt(np.mean(quaternion_rate**2,axis=0)).tolist(),
                gyro_minus_quaternion_rate_rms_rad_s=np.sqrt(np.mean((gyro[:-1]-quaternion_rate)**2,axis=0)).tolist(),
                limitations=['Not a measured translation or complete moving-fixture boundary',
                             'No measured leg poses are applied to an autonomous simulation',
                             'Computed inverse-load differences are diagnostic, not an adopted force correction'])
    np.savez_compressed(args.output/'base-motion.npz',time_s=t-t[0],quat=quat,gyro=gyro,alpha=alpha,relative_motor_load_nm=delta,quaternion_rate=quaternion_rate)
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
