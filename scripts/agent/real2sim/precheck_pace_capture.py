"""Check proposed PACE joint chirps on the intact closed-chain model; no hardware access."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror
from ct_mjlab.envs.chuanliantui.gas_spring import gas_spring_force
from scripts.agent.real2sim.elastic_joint_resistance import initial_preload
from scripts.agent.real2sim.fit_leg_length_response import leg_pose


STAGES = (('precheck', .15, .1, 1.5), ('capture', .40, .1, 5.), ('validation', .30, 5., .1))
KP, KD, LIMIT = 60., 2., 40.
PHYSICS_DT, CONTROL_DT = .0002, .002


def trajectory(time_s, stage):
    """5s hold, 20s linear chirp with 2s quintic envelopes, 5s hold."""
    _, amplitude, f0, f1 = stage
    local = np.asarray(time_s) - 5.
    x = np.clip(local, 0., 20.)
    edge = np.clip(np.minimum(x, 20.-x)/2., 0., 1.)
    window = edge**3*(10.-15.*edge+6.*edge**2)
    phase = 2*np.pi*(f0*x+(f1-f0)*x*x/40.)
    return amplitude*window*np.sin(phase)


def starting_pose(length, config):
    pose = mirror.MirrorPose(machine_config=config)
    pose.data.qpos[:7] = [0., 0., .8, 1., 0., 0., 0.]
    def residual(pair):
        lengths, tilts, _ = leg_pose(np.tile(pair, 2)[None])
        return [lengths[0, 0]-length, tilts[0, 0]]
    result = least_squares(residual, [-.8, .8], xtol=1e-12, ftol=1e-12, gtol=1e-12)
    assert abs(np.asarray(residual(result.x))).max() < 1e-8
    firmware = np.tile(result.x, 2)
    model_angles = (firmware-pose.zero)*pose.sign[:4]
    for side in (0, 1):
        assert pose._solve_leg(side, model_angles[2*side:2*side+2])
    return pose, firmware


def model_copy(pose, output):
    root = ET.parse(mirror.MODEL).getroot()
    root.find('compiler').set('meshdir', str((mirror.MODEL.parent/'../meshes').resolve()))
    base = root.find("worldbody/body[@name='base_link']")
    base.remove(base.find('freejoint')); base.set('pos', '0 0 .8'); base.set('quat', '1 0 0 0')
    option = root.find('option')
    if option is None: option = ET.SubElement(root, 'option')
    option.attrib.update(timestep=str(PHYSICS_DT), integrator='implicitfast', jacobian='dense',
                         solver='Newton', cone='elliptic', iterations='100', tolerance='1e-10',
                         ls_iterations='100', ls_tolerance='.0001')
    for side in ('left','right'):
        motor = root.find(f"actuator/motor[@name='{side}_gas_spring_motor']")
        motor.set('ctrlrange','0 600'); motor.set('forcerange','0 600')
    xml = ET.tostring(root, encoding='unicode')
    (output/'model.xml').write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    from wheel_response import prepare_wheel_response
    prepare_wheel_response(model)
    model.geom_contype[:] = model.geom_conaffinity[:] = 0
    return model


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--continuous',action='store_true',help='Run all 12 segments continuously from a synthetic265mm posture')
    parser.add_argument('--profile',choices=('doubled','small-high'),default='doubled')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    stages=(('independent_validation',.05,2.,5.),) if args.profile=='small-high' else STAGES
    config=ROOT/'deployment/H7_RL_ros/imcalib/config/machine_config.c'
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (mirror.MODEL,config)}
    pose, firmware=starting_pose(.265,config)
    model=model_copy(pose,args.output)
    mq=np.array([model.joint(n).qposadr[0] for n in mirror.MOTORS[:4]])
    mv=np.array([model.joint(n).dofadr[0] for n in mirror.MOTORS[:4]])
    sq=np.array([model.joint(s+'_spring2_joint').qposadr[0] for s in ('left','right')])
    sv=np.array([model.joint(s+'_spring2_joint').dofadr[0] for s in ('left','right')])
    aid=np.array([model.actuator(n+'_motor').id for n in mirror.MOTORS])
    sid=np.array([model.actuator(s+'_gas_spring_motor').id for s in ('left','right')])
    knee=np.array([model.joint(n).id for n in ('lf1','rf1')]); kq=model.jnt_qposadr[knee]
    data=mujoco.MjData(model)
    scenarios=[]; initial=[]; centers=[]; stage_ids=[]; axes=[]; spring_ids=[]
    for length in ((.265,) if args.continuous else (.250,.265,.280)):
        solved, fw=starting_pose(length,config)
        for joint in range(model.njnt):
            data.qpos[model.jnt_qposadr[joint]]=solved.data.qpos[solved.model.joint(model.joint(joint).name).qposadr[0]]
        data.qvel[:]=0.
        for spring_id,spring_name in enumerate(('shared_model','constant_150N','static_effective_midline')):
            if spring_id==0: force=gas_spring_force(data.qpos[sq],np.zeros(2))
            elif spring_id==1: force=np.full(2,150.)
            else: force=np.clip(137.343391+1759.576221*(np.maximum(0.,.010-data.qpos[sq])-.005),0,600)
            residual,info=initial_preload(model,data,np.zeros(4),initial_spring_force=force)
            center=data.qpos[mq]-residual/KP
            for stage_id,stage in enumerate(stages):
                for axis in range(4):
                    if args.continuous and (stage_id!=0 or axis!=0):continue
                    scenarios.append(dict(starting_length_m=length,spring_assumption=spring_name,stage=stage[0],
                        axis=('LF','LR','RF','RR')[axis],amplitude_rad=stage[1],frequency_hz=list(stage[2:]),
                        initial_model_angles_rad=data.qpos[mq].tolist(),target_center_rad=center.tolist(),
                        initial_holding_torque_nm=(-residual).tolist()))
                    initial.append(data.qpos.copy());centers.append(center.copy());stage_ids.append(stage_id);axes.append(axis);spring_ids.append(spring_id)
                    if args.continuous:scenarios[-1].update(stage='full_sequence',axis='all',amplitude_rad=None,frequency_hz=None)
    initial=np.asarray(initial);centers=np.asarray(centers);axes=np.asarray(axes);spring_ids=np.asarray(spring_ids)
    steps=60000*len(stages) if args.continuous else 15000
    n=len(scenarios); times=np.arange(steps+1)*CONTROL_DT
    offsets=np.array([trajectory(times,stages[i]) for i in stage_ids]).T
    # 各段独立从静置开始作必要条件筛查；失败时不声称整套连续序列已通过。
    import warp as wp
    import mujoco_warp as mw
    from scripts.agent.real2sim import ordered_leg_constraints as ordered
    wp.init(); start=time.monotonic()
    q_saved=[];v_saved=[];tau_saved=[];targets_saved=[]
    extrema=[dict(length_min_m=[1.,1.],length_max_m=[0.,0.],minimum_knee_margin_rad=10.,
                  max_abs_speed_rad_s=0.,peak_request_nm=0.,saturated_control_steps=0,
                  first_violation=None,first_buffer_entry=None,max_pin_error_m=0.) for _ in scenarios]
    stopped=np.zeros(n,dtype=bool)
    with wp.ScopedDevice('cuda:0'):
        gm=mw.put_model(model);gd=mw.put_data(model,data,nworld=n,nconmax=128,njmax=256)
        gd.qpos.assign(initial.astype(np.float32));gd.qvel.zero_()
        storage=ordered.buffers(gm,gd)
        with wp.ScopedCapture() as capture:
            for _ in range(10):
                mw.step1(gm,gd);ordered.apply(gd,storage);mw.step2(gm,gd)
        assert not np.any(gd.time.numpy())
        qp,qv=gd.qpos.numpy(),gd.qvel.numpy()
        for step,t in enumerate(times):
            q_target=centers.copy()
            if args.continuous:
                segment=min(int(t//30.),len(stages)*4-1)
                q_target[:,segment%4]+=trajectory(t-segment*30.,stages[segment//4])
            else:q_target[np.arange(n),axes]+=offsets[step]
            raw=KP*(q_target-qp[:,mq])-KD*qv[:,mv]
            ctrl=np.zeros((n,model.nu),dtype=np.float32)
            ctrl[:,aid[:4]]=np.clip(raw,-LIMIT,LIMIT)
            lengths=leg_pose(qp[:,mq]*pose.sign[:4]+pose.zero)[0]
            margins=np.minimum(qp[:,kq]-model.jnt_range[knee,0],model.jnt_range[knee,1]-qp[:,kq])
            finite=np.isfinite(qp).all(1)&np.isfinite(qv).all(1)&np.isfinite(raw).all(1)&np.isfinite(lengths).all(1)
            # 最新约定只保留实际边界保护；旧计划的测试余量仅报告，不作停止门控。
            violation=(~finite)|(lengths.min(1)<.14)|(lengths.max(1)>.33)|(margins.min(1)<0.)
            buffer_entry=(lengths.min(1)<.20)|(lengths.max(1)>.31)|(margins.min(1)<.05)
            for i in range(n):
                if stopped[i]:continue
                item=extrema[i]
                item['length_min_m']=np.minimum(item['length_min_m'],lengths[i]).tolist()
                item['length_max_m']=np.maximum(item['length_max_m'],lengths[i]).tolist()
                item['minimum_knee_margin_rad']=min(item['minimum_knee_margin_rad'],float(margins[i].min()))
                item['max_abs_speed_rad_s']=max(item['max_abs_speed_rad_s'],float(abs(qv[i,mv]).max()))
                item['peak_request_nm']=max(item['peak_request_nm'],float(abs(ctrl[i,aid[:4]]).max()))
                item['saturated_control_steps']+=int(np.any(abs(raw[i])>LIMIT))
                if buffer_entry[i] and item['first_buffer_entry'] is None:
                    item['first_buffer_entry']=dict(time_s=float(t),lengths_m=lengths[i].tolist(),knee_margin_rad=margins[i].tolist())
                if violation[i]:item['first_violation']=dict(time_s=float(t),lengths_m=lengths[i].tolist(),knee_margin_rad=margins[i].tolist(),finite=bool(finite[i]))
            stopped |= violation
            ctrl[stopped[:,None]&np.isin(np.arange(model.nu)[None],aid)]=0.
            force=gas_spring_force(qp[:,sq],qv[:,sv])
            force[spring_ids==1]=np.clip(150.-50.*qv[spring_ids==1][:,sv],0,600)
            force[spring_ids==2]=np.clip(137.343391+1759.576221*(np.maximum(0.,.010-qp[spring_ids==2][:,sq])-.005)-50.*qv[spring_ids==2][:,sv],0,600)
            ctrl[:,sid]=force
            if step%5==0:
                q_saved.append(qp[:,mq].copy());v_saved.append(qv[:,mv].copy());tau_saved.append(ctrl[:,aid[:4]].copy());targets_saved.append(q_target.copy())
                for i in range(n):
                    if stopped[i]:continue
                    data.qpos[:]=qp[i];mujoco.mj_kinematics(model,data)
                    pairs=[p for group in mirror.PIN_PAIRS for p in group]+[(s+'_spring_tip',s+'_gas_spring_lower') for s in ('left','right')]
                    gap=max(np.linalg.norm(data.site(a).xpos-data.site(b).xpos) for a,b in pairs)
                    extrema[i]['max_pin_error_m']=max(extrema[i]['max_pin_error_m'],float(gap))
            if step==len(times)-1:break
            gd.ctrl.assign(ctrl);wp.capture_launch(capture.graph)
            qp,qv=gd.qpos.numpy(),gd.qvel.numpy()
            if step%2500==0:print(json.dumps(dict(simulation_s=float(t),worlds=n,stopped=int(stopped.sum()),elapsed_s=time.monotonic()-start)),flush=True)
    for scenario,result in zip(scenarios,extrema):
        result['limits_passed']=result['first_violation'] is None and result['max_pin_error_m']<1e-4
        result['saturation_fraction']=result['saturated_control_steps']/steps
        scenario['result']=result
    assert {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in protected}==protected
    report=dict(scope=f'Continuous{len(stages)*4}-segment GPU precheck from synthetic265mm starts' if args.continuous else 'Necessary isolated-phase GPU precheck; synthetic starts, not real-robot acceptance or a full sequential test',
        continuous_sequence=args.continuous,
        stages=stages,profile=args.profile,scenarios=scenarios,all_limits_passed=all(x['limits_passed'] for x in extrema),
        actual_speed_stop_gate=False,automatic_phase_quality_gate=False,
        preserved_protection=['RC disable/offline and fault/invalid feedback in planned firmware','leg and calibrated knee limits','40Nm request saturation'],
        stopping_limits=dict(leg_length_m=[.14,.33],knee_range_model_rad=model.jnt_range[knee].tolist()),
        report_only_buffers=dict(leg_length_m=[.20,.31],knee_margin_rad=.05),
        kp=KP,kd=KD,torque_limit_nm=LIMIT,physics_dt_s=PHYSICS_DT,control_dt_s=CONTROL_DT,
        new_gpu_environment_steps=steps*10*n,source_sha256=protected,hardware_access=False,firmware_modified=False,
        hardware_ready=False,policy_training_updates=0,
        limitations=['Shared dynamics were not identified; spring scenarios are hypotheses, not measured bounds',
                    'Initial holding torque is computed for each scenario; actual takeover requires recorded holding torque',
                    'Stopped cases continue passive model evolution but are excluded from post-stop extrema',
                    'No collisions/contact-enabled checks; no new real data; do not flash/run based on these results alone'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.output/'traces.npz',time_s=times[::5],q=np.asarray(q_saved),v=np.asarray(v_saved),
                        request_nm=np.asarray(tau_saved),target_rad=np.asarray(targets_saved))
    shutil.copy2(__file__,args.output/Path(__file__).name)
    print(json.dumps(dict(all_limits_passed=report['all_limits_passed'],failed=sum(not x['limits_passed'] for x in extrema),
        completed=len(scenarios),new_gpu_environment_steps=report['new_gpu_environment_steps']),indent=2))


if __name__=='__main__':main()
