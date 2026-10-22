#!/usr/bin/env python3
"""Reproduce the historical compensation anomaly offline, separate from calibration.

This uses the previous experimental controller only to audit that old claim.
The calibration itself never uses compensation or policy-generated inputs.
"""
import argparse
import json
from pathlib import Path
import sys
import hashlib
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'sim2sim'))
import real2sim_dynamics_huananhu as d
import mj_sim2sim_huananhu_multimode as sim


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--small',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'artifacts/real2sim-dynamics-20261021/verified-run'
    params=np.array(json.loads((old/'candidate-parameters.json').read_text())['additional_parameters'])
    model=args.out/'historical-candidate.xml';d.export_candidate(sim.MODEL_ASSET,model,params)
    # Verify old absolute mesh path relocation changes no compiled physical values.
    historical_tree=ET.parse(old/'candidate.xml')
    historical_tree.find('compiler').set('meshdir',str(ROOT/'resources/robots/chuanliantui_new_1/meshes'))
    om=mujoco.MjModel.from_xml_string(ET.tostring(historical_tree.getroot(),encoding='unicode'))
    nm=mujoco.MjModel.from_xml_path(str(model))
    for n in ('body_mass','body_inertia','dof_armature','dof_damping','dof_frictionloss','geom_friction'):
        np.testing.assert_array_equal(getattr(om,n),getattr(nm,n))
    coeff=params[:,2].copy();rounded=np.array([.11604,.06147],dtype=np.float32).astype(float)
    ev=ReferenceEvaluator(onnx.load(str(ROOT/'exports/chuanliantui_multimode/model5849-h7-20261017/model_5849_opset13.onnx')))
    def controller(obs):return ev.run(None,{'obs':obs[None]})[0][0].astype(np.float32)
    cases=[('full_a',coeff,{}),('full_b',coeff,{}),('rounded_a',rounded,{}),('rounded_b',rounded,{})]
    if not args.small:
        cases += [('full_L_plus_1e-6',coeff+[1e-6,0],{}),('full_R_plus_1e-6',coeff+[0,1e-6],{}),
                  ('full_R_minus_1e-6',coeff-[0,1e-6],{}),('armature_plus_01pct',coeff,{'armature':1.001}),
                  ('pitch_plus_1e-8rad',coeff,{'pitch':1e-8}),('height_plus_10um',coeff,{'height':1e-5}),
                  ('friction_plus_01pct',coeff,{'friction':1.001}),
                  ('full_dt_1ms',coeff,{'dt':.001}),('rounded_dt_1ms',rounded,{'dt':.001}),
                  ('full_tolerance_1e-12',coeff,{'tolerance':1e-12})]
    report={'scope':'historical experimental-controller anomaly only; no calibration or H7 change',
            'cases':{},'mujoco':mujoco.__version__,'numpy':np.__version__,'onnx':onnx.__version__}
    traces={};initials={}
    count=5 if args.small else 1500
    for name,c,change in cases:
        sim.DT,sim.DECIMATION=.002,10
        start=time.time();env=sim.Playback('normal',(0.,0.,.27),model_path=model,wheel_friction_compensation=c)
        if 'armature' in change:
            q0,v0=env.data.qpos.copy(),env.data.qvel.copy()
            env.model.dof_armature[env.v[sim.WHEELS]]*=change['armature'];mujoco.mj_setConst(env.model,env.data)
            env.data.qpos[:],env.data.qvel[:]=q0,v0
        if 'dt' in change:
            sim.DT,sim.DECIMATION=change['dt'],20;env.model.opt.timestep=change['dt']
            original_controls=env.controls
            held={'calls':0,'torque':None}
            def controls_500hz(action):
                torque,spring=original_controls(action)
                if held['calls']%2==0:held['torque']=torque.copy()
                held['calls']+=1
                return held['torque'].copy(),spring
            # Refine only physics integration; keep H7 torque updates at 500 Hz.
            env.controls=controls_500hz
        if 'tolerance' in change:env.model.opt.tolerance=change['tolerance']
        if 'pitch' in change:mujoco.mju_quatIntegrate(env.data.qpos[3:7],np.array([0.,change['pitch'],0.]),1.)
        env.data.qpos[2]+=change.get('height',0.)
        if 'friction' in change:env.model.geom_friction[:,0]*=change['friction']
        mujoco.mj_forward(env.model,env.data)
        initials[name]=env.data.qpos.copy();rows=[];actions=[];torques=[];states=[]
        for _ in range(count):
            action,torque=env.step(controller);row=env.metrics(action,torque)
            rows.append([row['time_s'],row['x_m'],row['y_m'],row['vx_m_s'],row['yaw_rate_rad_s'],*row['roll_pitch_deg']])
            actions.append(action);torques.append(torque);states.append(env.data.qpos.copy())
        arr=np.array(rows);traces[name]=arr
        np.savez_compressed(args.out/(name+'.npz'),metrics=arr,actions=np.array(actions),last_substep_torque=np.array(torques),
                            qpos=np.array(states),initial_qpos=initials[name],coefficient=c)
        settled=arr[arr[:,0]>=2] if not args.small else arr
        result={'coefficient':c.tolist(),'perturbation':change,'elapsed_s':time.time()-start,
                'mean_abs_vx_m_s':float(np.mean(abs(settled[:,3]))),
                'mean_abs_yaw_rad_s':float(np.mean(abs(settled[:,4]))),
                'net_displacement_full_m':float(np.hypot(arr[-1,1]-arr[0,1],arr[-1,2]-arr[0,2])),
                'pitch_peak_to_peak_deg':float(np.ptp(settled[:,6])),
                'initial_qpos_sha256':hashlib.sha256(initials[name].tobytes()).hexdigest()}
        report['cases'][name]=result;d.save_json(args.out/'progress.json',report);print(name,result,flush=True)
    report['same_baseline_repeats_max_abs']={prefix:float(np.max(abs(traces[prefix+'_a']-traces[prefix+'_b']))) for prefix in ('full','rounded')}
    a,b=traces['full_a'],traces['rounded_a'];report['divergence']=[]
    for name,arr in traces.items():
        delta=np.max(abs(arr-a),axis=1)
        report['divergence'].append({'case':name,'first_metric_difference_over_1e-3_s':next((float(arr[i,0]) for i in np.flatnonzero(delta>1e-3)),None),
                                     'max_metric_difference':float(max(delta))})
    if not args.small:
        for case,rel in [('full_a','artifacts/real2sim-wheel-closed-loop-20261021/friction-compensation-cli.json'),
                         ('rounded_a','artifacts/real2sim-wheel-response-h7-20261021/sim-approved-coefficients.json')]:
            oldrows=json.loads((ROOT/rel).read_text())['trajectory']
            arr=np.array([[r['time_s'],r['x_m'],r['y_m'],r['vx_m_s'],r['yaw_rate_rad_s'],*r['roll_pitch_deg']] for r in oldrows])
            report[case+'_vs_historical_max_abs']=float(np.max(abs(traces[case]-arr)))
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
        for name,arr in traces.items():
            for ax,col in zip(axes,[3,4,6]):ax.plot(arr[:,0],arr[:,col],label=name)
        for ax,label in zip(axes,['vx (m/s)','yaw (rad/s)','pitch (deg)']):ax.set_ylabel(label);ax.grid()
        axes[0].legend(fontsize=7,ncol=3);axes[-1].set_xlabel('s');fig.tight_layout();fig.savefig(args.out/'rounding-sensitivity.png',dpi=140)
    report['complete']=True;report['script_sha256']=d.identity(Path(__file__))['sha256'];d.save_json(args.out/'report.json',report)
if __name__=='__main__':main()
