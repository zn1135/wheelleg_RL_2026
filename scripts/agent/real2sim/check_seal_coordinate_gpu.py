"""GPU static-friction coordinate check; equal physical load, no real hardware or policy."""
import argparse
import hashlib
import json
from pathlib import Path
PROGRAM_SOURCE=Path(__file__).read_bytes()

import mujoco
import numpy as np

from scripts.agent.real2sim.control_leg_pose_gpu import gas_spring_force,leg_pose
from scripts.agent.real2sim.elastic_joint_resistance import initial_preload


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    a=np.load(args.inputs/'inputs.npz');m=mujoco.MjModel.from_xml_path(str(args.model));d=mujoco.MjData(m)
    mq,mv=a['mq'].astype(int),a['mv'].astype(int)
    slider=m.joint('right_spring2_joint').dofadr[0];span=m.tendon('right_seal_span').id
    m.dof_frictionloss[:]=0;m.dof_frictionloss[slider]=1e-6;m.tendon_frictionloss[:]=1e-6
    m.dof_solref[slider]=m.dof_solref[mv[2]];m.dof_solimp[slider]=m.dof_solimp[mv[2]]
    m.geom_contype[:]=m.geom_conaffinity[:]=0;m.opt.gravity[:]=a['gravity'][0]
    d.qpos[:]=a['initial_q'];d.qvel[:]=0
    residual,info=initial_preload(m,d,np.zeros(4),return_tangent=True);holding=-residual
    tangent=np.array(info['tangent']);mujoco.mj_forward(m,d)
    tendon_jac=np.zeros(m.nv);start=m.ten_J_rowadr[span];stop=start+m.ten_J_rownnz[span]
    tendon_jac[m.ten_J_colind[start:stop]]=d.ten_J[start:stop]
    axial=d.qpos[m.joint('right_spring2_joint').qposadr[0]]+m.body('right_spring2').pos[2]
    force_ratio=d.ten_length[span]/axial
    parity=float(abs(force_ratio*tendon_jac@tangent-tangent[slider]).max())
    motors=[m.actuator(n+'_motor').id for n in ('lf0','lf00','rf0','rf00')]
    spring=[m.actuator(s+'_gas_spring_motor').id for s in ('left','right')]
    sq=[m.joint(s+'_spring2_joint').qposadr[0] for s in ('left','right')]
    sv=[m.joint(s+'_spring2_joint').dofadr[0] for s in ('left','right')]
    sf=np.zeros((2,m.nv),dtype='f4');sf[0,slider]=100
    tf=np.zeros((2,m.ntendon),dtype='f4');tf[1,span]=100*force_ratio
    pressure=np.zeros((2,m.nv),dtype='f4');pressure[:,slider]=-80
    import warp as wp
    import mujoco_warp as mw
    from scripts.agent.real2sim import ordered_leg_constraints as ordered
    wp.init();positions=[];velocities=[]
    with wp.ScopedDevice('cuda:0'):
        gm=mw.put_model(m,batch_sizes={'dof_frictionloss':2,'tendon_frictionloss':2})
        gm.dof_frictionloss.assign(sf);gm.tendon_frictionloss.assign(tf)
        gd=mw.put_data(m,d,nworld=2,nconmax=128,njmax=256)
        gd.qpos.assign(np.tile(a['initial_q'],(2,1)).astype('f4'));gd.qvel.zero_()
        buffers=ordered.buffers(gm,gd)
        with wp.ScopedCapture() as capture:
            for _ in range(10):mw.step1(gm,gd);ordered.apply(gd,buffers);mw.step2(gm,gd)
        for i in range(501):
            q,v=gd.qpos.numpy(),gd.qvel.numpy();positions.append(q.copy());velocities.append(v.copy())
            if i==500:break
            control=np.zeros((2,m.nu),dtype='f4');control[:,motors]=holding
            control[:,spring]=gas_spring_force(q[:,sq],v[:,sv])
            axial=q[1,sq[1]]+m.body('right_spring2').pos[2]
            tf[1,span]=100*np.hypot(axial,m.body('right_spring2').pos[1])/abs(axial)
            gm.tendon_frictionloss.assign(tf)
            gd.ctrl.assign(control);gd.qfrc_applied.assign(pressure);wp.capture_launch(capture.graph)
    q=np.array(positions);v=np.array(velocities);length,angle,_=leg_pose(q[:,:,mq]*a['sign']+a['zero'])
    drift=abs(length[:,:,1]-length[0,:,1]).max(0)*1000
    report=dict(scope='synthetic static-friction coordinate check, not real-response validation',physics_backend='mujoco_warp_gpu',
                seconds=1.,new_gpu_environment_steps=10000,policy_training_updates=0,hardware_access=False,
                applied_constant_slider_load_n=-80.,friction_capacity_n=100.,motor_holding_nm=holding.tolist(),
                virtual_work_jacobian_max_error_m_rad=parity,slider_reference_inverse_weight=float(m.dof_invweight0[slider]),
                spatial_reference_inverse_weight=float(m.tendon_invweight0[span]),
                initial_axial_to_span_force_ratio=float(force_ratio),
                max_leg_drift_mm={'slider':float(drift[0]),'spatial':float(drift[1])},
                final_slider_velocity_m_s=v[-1,:,slider].tolist(),input_model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest())
    report['program_sha256']=hashlib.sha256(PROGRAM_SOURCE).hexdigest()
    report['passed']=bool(parity<1e-5 and drift[1]<.5 and drift[0]>5*max(drift[1],.001))
    np.savez_compressed(args.output/'traces.npz',time_s=np.arange(501)*.002,qpos=q,qvel=v,length_m=length,angle_rad=angle)
    (args.output/Path(__file__).name).write_bytes(PROGRAM_SOURCE)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['passed']


if __name__=='__main__':main()
