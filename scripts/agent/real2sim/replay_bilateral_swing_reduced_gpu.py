"""Validate frozen near-constant-length swing candidates using MuJoCo Warp GPU.

This is a reduced 1-DOF experiment, not full-robot or free-length acceptance.
"""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mw

from scripts.agent.real2sim.identify_bilateral_swing import load_trials, sha

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim2sim'))
import real2sim_huananhu as mirror


@wp.kernel
def drive(q: wp.array2d(dtype=wp.float32), v: wp.array2d(dtype=wp.float32),
          inputs: wp.array2d(dtype=wp.float32), parameters: wp.array2d(dtype=wp.float32),
          index: wp.array(dtype=wp.int32), ctrl: wp.array2d(dtype=wp.float32)):
    world = wp.tid()
    theta = q[world,0]
    load = parameters[world,3]*wp.sin(theta) + parameters[world,4]*wp.cos(theta)
    load += parameters[world,5]*wp.sin(2.0*theta) + parameters[world,6]*wp.cos(2.0*theta)
    # 单位惯量模型：输入/惯量转换为角加速度；粘性和库仑阻力由引擎隐式求解。
    ctrl[world,0] = (inputs[index[0],world] - load) / parameters[world,0]


@wp.kernel
def save(q: wp.array2d(dtype=wp.float32), v: wp.array2d(dtype=wp.float32),
         index: wp.array(dtype=wp.int32), output: wp.array3d(dtype=wp.float32)):
    world = wp.tid()
    output[index[0],world,0] = q[world,0]
    output[index[0],world,1] = v[world,0]


@wp.kernel
def advance(index: wp.array(dtype=wp.int32)):
    index[0] += 1


def nominal_inertia(trials):
    asset = mirror.MODEL
    config = ROOT/'deployment/H7_RL_ros/imcalib/config/machine_config.c'
    pose = mirror.MirrorPose(machine_config=config)
    model, data = pose.model, pose.data
    result = {}
    for side, name in enumerate(('left','right')):
        trial = next(tr for tr in trials if tr['side']==name and tr['level']==1)
        q = trial['q'][0] * pose.sign[side*2:side*2+2]
        if not pose._solve_leg(side,q):
            raise ValueError('Cannot solve measured initial assembly')
        mujoco.mj_forward(model,data)
        matrix = np.empty((model.nv,model.nv))
        mujoco.mj_fullM(model,data,matrix)
        tangent = np.zeros(model.nv)
        for j, motor in enumerate(mirror.DRIVES[side]):
            tangent[model.joint(motor).dofadr[0]] = -pose.sign[side*2+j]
        wheel = model.joint(mirror.WHEELS[side]).dofadr[0]
        fixed = tangent @ matrix @ tangent
        free = fixed - (tangent @ matrix[:,wheel])**2 / matrix[wheel,wheel]
        # 取XZ投影，剔除模型轮轴和髋之间固定的横向偏置。
        delta = data.body(mirror.WHEELS[side]).xpos-data.body(mirror.DRIVES[side][0]).xpos
        ranges = []
        for joint in mirror.PASSIVE[side]:
            j = model.joint(joint)
            value = float(data.qpos[j.qposadr[0]])
            excess = max(j.range[0]-value,value-j.range[1],0.) if j.limited[0] else 0.
            ranges.append(dict(joint=joint,initial_rad=value,range_rad=j.range.tolist(),excess_rad=float(excess)))
        result[name] = dict(fixed_wheel_inertia_kg_m2=float(fixed),free_wheel_inertia_kg_m2=float(free),
                            initial_model_projected_length_m=float(np.linalg.norm(delta[[0,2]])),
                            recorded_initial_length_m=float(trial['length'][0]),
                            initial_passive_joint_ranges=ranges, closure_m=pose.residual_m[side])
    return result, {str(asset):sha(asset),str(config):sha(config)}


def run(trials, parameters, nominal, dt):
    xml = f'''<mujoco model="reduced_swing"><option timestep="{dt}" gravity="0 0 0"
        integrator="implicitfast" solver="Newton" jacobian="dense" iterations="50" tolerance="1e-10"/>
        <worldbody><body><joint name="swing" type="hinge" axis="0 0 1" limited="false" frictionloss="0.000001"
        solreffriction="0.002 1" solimpfriction="0.999 0.9999 0.001 0.5 2"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body></worldbody>
        <actuator><motor joint="swing"/></actuator></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    count = len(trials)*2
    duration = np.ceil(max(tr['t'][-1] for tr in trials)*1000)/1000
    steps = int(round(duration/dt))
    timeline = np.arange(steps+1)*dt
    table = np.zeros((count,7),np.float32)
    inputs = np.zeros((steps+1,count),np.float32)
    q, v = np.zeros((count,1),np.float32), np.zeros((count,1),np.float32)
    for i,tr in enumerate(trials):
        for branch in range(2):
            world = branch*len(trials)+i
            p = parameters[tr['side']]['coefficients'] if branch else [nominal[tr['side']]['free_wheel_inertia_kg_m2'],0,0]
            table[world,:len(p)] = p
            # 按原始H7时标保持提交力矩，绝不把后续实测状态写入仿真。
            index = np.clip(np.searchsorted(tr['t'],timeline,side='right')-1,0,len(tr['t'])-1)
            inputs[:,world] = -tr['tau'][index].sum(1)
            q[world,0],v[world,0] = tr['theta'][0],tr['reported_speed'][0]
    wp.init()
    with wp.ScopedDevice('cuda:0'):
        gm = mw.put_model(model)
        gd = mw.put_data(model,data,nworld=count,nconmax=1,njmax=4)
        gm.dof_frictionloss = wp.array((table[:,2]/table[:,0])[:,None],dtype=wp.float32)
        gm.dof_damping = wp.array((table[:,1]/table[:,0])[:,None],dtype=wp.float32)
        gd.qpos.assign(q);gd.qvel.assign(v)
        u, p = wp.array(inputs),wp.array(table)
        index = wp.zeros(1,dtype=wp.int32)
        output = wp.empty((steps+1,count,2),dtype=wp.float32)
        with wp.ScopedCapture() as capture:
            wp.launch(save,dim=count,inputs=[gd.qpos,gd.qvel,index,output])
            wp.launch(drive,dim=count,inputs=[gd.qpos,gd.qvel,u,p,index,gd.ctrl])
            mw.step(gm,gd)
            wp.launch(advance,dim=1,inputs=[index])
        if index.numpy()[0] != 0:
            raise ValueError('Capture advanced the initial state')
        for _ in range(steps):
            wp.capture_launch(capture.graph)
        wp.launch(save,dim=count,inputs=[gd.qpos,gd.qvel,index,output])
        saved = output.numpy()
    if not np.isfinite(saved).all():
        raise ValueError('Nonfinite reduced-model response')
    return timeline,saved,steps*count,xml


def metrics(trials,timeline,saved):
    rows = []
    for i,tr in enumerate(trials):
        row = dict(side=tr['side'],level=tr['level'],held_out=tr['level'] in (2,4),branches={})
        for branch,label in enumerate(('zero_added_resistance','identified')):
            state = saved[:,branch*len(trials)+i]
            pred = np.column_stack([np.interp(tr['t'],timeline,state[:,axis]) for axis in range(2)])
            e = pred[:,0]-tr['theta']
            ev = pred[:,1]-tr['reported_speed']
            qpred = tr['q'][0] - (pred[:,0]-tr['theta'][0])[:,None]
            eq = np.degrees(qpred-tr['q'])
            release = np.searchsorted(tr['t'],tr['row']['release_s'])
            # 长度固定于初值只是降阶假设，长度误差单列，不能称为自由伸缩预测。
            row['branches'][label] = dict(angle_rmse_deg=float(np.sqrt(np.mean(e**2))*180/np.pi),
                angle_peak_deg=float(np.max(abs(e))*180/np.pi),terminal_angle_error_deg=float(e[-1]*180/np.pi),
                speed_rmse_rad_s=float(np.sqrt(np.mean(ev**2))),
                active_speed_rmse_rad_s=float(np.sqrt(np.mean(ev[tr['active']]**2))),
                release_speed_rmse_rad_s=float(np.sqrt(np.mean(ev[release:]**2))),
                front_rear_angle_rmse_deg=np.sqrt(np.mean(eq**2,axis=0)).tolist(),
                fixed_length_assumption_rmse_mm=float(np.sqrt(np.mean((tr['length']-tr['length'][0])**2))*1000))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--fit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    trials,audit = load_trials(args.input)
    identification = json.loads((args.fit/'identification.json').read_text())
    if audit != identification['input_audit']:
        # JSON object keys are strings, including frame type numbers.
        if json.loads(json.dumps(audit)) != identification['input_audit']:
            raise ValueError('Input identity changed')
    nominal,sources = nominal_inertia(trials)
    timeline,saved,steps,xml = run(trials,identification['fitted'],nominal,.001)
    rows = metrics(trials,timeline,saved)
    # 半步长复核同一冻结模型，不重拟合。
    fine_t,fine,fine_steps,_ = run(trials,identification['fitted'],nominal,.0005)
    discrepancy = abs(saved-fine[::2][:len(saved)])
    refinement = []
    for i,tr in enumerate(trials):
        valid = timeline<=tr['t'][-1]
        for branch,label in enumerate(('zero_added_resistance','identified')):
            d = discrepancy[valid,branch*len(trials)+i]
            refinement.append(dict(side=tr['side'],level=tr['level'],branch=label,
                                   max_angle_difference_deg=float(d[:,0].max()*180/np.pi),
                                   max_speed_difference_rad_s=float(d[:,1].max())))
    report = dict(scope=__doc__,nominal=nominal,trials=rows,gpu_environment_steps=steps+fine_steps,
                  timestep_s=.001,refinement_timestep_s=.0005,
                  refinement_by_trial=refinement,
                  refinement_max_angle_difference_deg=max(r['max_angle_difference_deg'] for r in refinement if r['branch']=='identified'),
                  refinement_max_speed_difference_rad_s=max(r['max_speed_difference_rad_s'] for r in refinement if r['branch']=='identified'),
                  source_sha256={**sources,str(Path(__file__)):sha(Path(__file__)),
                                 str(args.fit/'identification.json'):sha(args.fit/'identification.json')},
                  full_robot_acceptance=False,free_length_dynamics_validated=False,training_updates=0,
                  orientation='user confirms side lying; exact tilt and spring installation unknown')
    np.savez_compressed(args.output/'traces.npz',time_s=timeline,state=saved)
    (args.output/'reduced-model.xml').write_text(xml)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(trials=rows,nominal=nominal,gpu_environment_steps=report['gpu_environment_steps'],
                          refinement_max_angle_difference_deg=report['refinement_max_angle_difference_deg']),indent=2))


if __name__ == '__main__':
    main()
