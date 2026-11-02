"""Offline energy and closed-loop audit of the saved zero-action launch sample."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    # 禁止物理积分；仅对已保存姿态计算惯性、几何和势能。
    def forbidden(*args,**kwargs):raise RuntimeError('Offline audit must not integrate physics')
    for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,forbidden)
    fixture=a.root/'precheck-v2/fixtures'
    model_path=fixture/'model.mjb';initial_path=fixture/'precheck_18_initial.npz'
    model=mujoco.MjModel.from_binary_path(str(model_path));data=mujoco.MjData(model)
    mapping=json.loads((fixture/'mapping.json').read_text())
    with np.load(initial_path) as f:initial={k:f[k].copy() for k in f.files}
    for key in ('body_mass','body_inertia','body_ipos','geom_friction','geom_solref'):
        getattr(model,key)[:]=initial[key]
    mujoco.mj_setConst(model,data)
    legs=np.array([0,1,3,4]);qadr=np.array(mapping['motor_q'])[legs]
    gains=60.*initial['kp'][legs]*initial['strength'][legs]
    target=np.clip(initial['nominal'][legs],np.array(mapping['reference'])[legs]-3.14,
        np.array(mapping['reference'])[legs]+3.14)
    motor_ids=np.array(mapping['motor_ids'])[legs]
    assert np.all(model.actuator_gainprm[motor_ids,0]==1.)
    assert np.all(model.actuator_gear[motor_ids,0]==1.) and not np.any(model.jnt_stiffness)

    def energy_parts():
        gravity=-float(np.sum(model.body_mass[:,None]*data.xipos*model.opt.gravity))
        compression=np.maximum(.01-data.qpos[mapping['spring_q']],0.)
        spring=float(np.sum(np.where(compression<=.09,.5*(150./.09)*compression**2,
            .5*(150./.09)*.09**2+150.*(compression-.09))))
        error=np.abs(target-data.qpos[qadr])
        # 固定目标位置PD的位置项可写成势能；饱和区用线性段接续。
        pd=float(np.sum(np.where(error<=40./gains,.5*gains*error**2,
            40.*error-.5*1600./gains)))
        return dict(gravity_J=gravity,gas_spring_J=spring,pd_position_J=pd,
            potential_lower_bound_J=gravity+spring+pd)

    data.qpos[:]=initial['qpos'];data.qvel[:]=initial['qvel']
    mujoco.mj_forward(model,data);mujoco.mj_energyVel(model,data)
    initial_energy=energy_parts();initial_energy['kinetic_J']=float(data.energy[1])
    initial_energy['total_J']=initial_energy['potential_lower_bound_J']+initial_energy['kinetic_J']
    nominal_error=float(np.abs(data.qpos[qadr]-target).max())
    rows=[];inputs={}
    for engine,folder in [('GPU','precheck-v2'),('CPU','precheck-v2-native-recorded')]:
        path=a.root/folder/'precheck_18.npz';inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as f:
            for time_s,pose in zip(f['qpos_time'],f['qpos'],strict=True):
                data.qpos[:]=pose;data.qvel[:]=0.;mujoco.mj_fwdPosition(model,data)
                ids=np.flatnonzero(np.asarray(data.efc_type)==mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                worst=ids[np.abs(data.efc_pos[ids]).argmax()]
                eq_id=int(data.efc_id[worst]);eq_rows=ids[np.asarray(data.efc_id)[ids]==eq_id]
                penetration=float(min(0.,data.contact.dist[:data.ncon].min())) if data.ncon else 0.
                tick=int(round(float(time_s)/.002));parts=energy_parts()
                rows.append(dict(engine=engine,time_s=float(time_s),root_height_m=float(pose[2]),
                    equality_max_abs_component_m=float(np.abs(data.efc_pos[worst])),
                    equality_name=model.equality(eq_id).name,
                    equality_vector_m=data.efc_pos[eq_rows].tolist(),
                    deepest_contact_distance_m=penetration,
                    recorded_base_contact_N=float(f['base_contact'][tick]),
                    recorded_vertical_velocity_m_s=float(f['linear_world'][tick,2]),**parts))
    selected=[r for r in rows if any(abs(r['time_s']-t)<1e-8 for t in (0.,.94,.95,.96,.97,1.,1.26))]
    collision=np.flatnonzero((model.geom_contype!=0)&(model.geom_bodyid==mapping['base']))[0]
    report=dict(mode='saved_state_offline_causal_evidence_not_integration_comparison',
        mujoco_version=mujoco.__version__,physics_integration_steps=0,network_forwards=0,optimizer_updates=0,
        initial_energy=initial_energy,initial_pd_target_error_max_rad=nominal_error,
        contact_solref=model.geom_solref[collision].tolist(),contact_solimp=model.geom_solimp[collision].tolist(),
        nominal_contact_damping_ratio=float(-model.geom_solref[collision,1]/(2*np.sqrt(-model.geom_solref[collision,0]))),
        integrator=mujoco.mjtIntegrator(model.opt.integrator).name,timestep_s=float(model.opt.timestep),
        selected_states=selected,states=rows,inputs_sha256=inputs,
        interpretation='Large nonphysical energy gain and loop violation occur at CPU impact; contact/constraint/control numerical instability is strongly indicated',
        unresolved='No original per-substep joint velocity, torque, solver iteration or residual logs; cannot isolate integration versus contact/constraint/control contributions',
        limitations=['Potential is a lower bound on the audited physical/P-D storage model, not an exact MuJoCo soft-constraint energy invariant',
            'Contact penetration and constraint residual are CPU static reconstructions of saved 100 Hz poses; impact peak is recorded at 500 Hz',
            'Fixed PD targets, zero policy action, positive damping and fixed spring curve are required for the energy interpretation',
            'No 0.5 ms run or force/solver ablation was performed'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(initial_energy=initial_energy,selected_states=selected),indent=2))


if __name__=='__main__':main()
