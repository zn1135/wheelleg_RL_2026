"""Static robot comparison; distances use each robot's own hip/wheel frames."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from .replay import XML_REL,write


def body_bounds(model,data,body):
    pts=[]
    for g in range(model.ngeom):
        if model.geom_bodyid[g]!=body or not(model.geom_contype[g] or model.geom_conaffinity[g]):continue
        if model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
        mid=model.geom_dataid[g];a=model.mesh_vertadr[mid];b=a+model.mesh_vertnum[mid]
        v=model.mesh_vert[a:b]@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        pts.append((v-data.xpos[body])@data.xmat[body].reshape(3,3))
    v=np.concatenate(pts);return [v.min(0).tolist(),v.max(0).tolist()]


def record(model,base,hip,knee,wheel):
    d=mujoco.MjData(model);mujoco.mj_kinematics(model,d)
    joint=model.joint(knee);lengths=[]
    for q in np.linspace(*joint.range,2001):
        d.qpos[joint.qposadr[0]]=q;mujoco.mj_kinematics(model,d)
        vec=d.body(wheel).xpos-d.xanchor[model.joint(hip).id]
        axis=d.xaxis[model.joint(hip).id]
        lengths.append(np.linalg.norm(vec-axis*(vec@axis)))
    mujoco.mj_resetData(model,d);mujoco.mj_kinematics(model,d)
    bid=model.body(base).id
    return dict(mass_kg=float(model.body_subtreemass[bid]),base_bounds_m=body_bounds(model,d,bid),
        base_com_offset_m=model.body_ipos[bid].tolist(),
        knee_range_rad=joint.range.tolist(),hip_wheel_planar_distance_envelope_m=[float(min(lengths)),float(max(lengths))],
        envelope_note='Front-chain FK under knee limits; closed linkage/gas-spring/motor torque feasibility not proven',
        upper_segment_m=float(np.linalg.norm(model.body(knee.replace('_Joint','_Link') if '_Joint' in knee else knee).pos[[0,2]])) if '_Joint' not in knee else .17472,
        lower_segment_m=.208 if '_Joint' in knee else .25)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    original=mujoco.MjModel.from_xml_path(str(a.output/'source'/XML_REL))
    from ct_mjlab.envs.chuanliantui.model import load_model
    local=load_model()
    o=record(original,'base_Link_del','lf0_Joint','lf1_Joint','l_wheel_Link')
    l=record(local,'base_link','lf0','lf1','lfwheel')
    for r,torque in [(o,5.),(l,3.9)]:
        r.update(wheel_radius_m=.06,step_to_radius=.2/.06,step_to_diameter=.2/.12,
            wheel_torque_cap_Nm=torque,normalized_total_wheel_torque=2*torque/(r['mass_kg']*9.81*.06))
    write(a.output/'analysis/robot-comparison.json',dict(fudan=o,local=l))
    print(json.dumps(dict(fudan=o,local=l),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
