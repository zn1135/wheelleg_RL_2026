"""Read saved precheck traces and inspect mesh clearance; never step physics."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from ct_mjlab.envs.fudan_ppo_walk.reset_geometry import mesh_vertices


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    model=mujoco.MjModel.from_binary_path(str(a.root/'precheck-v2/fixtures/model.mjb'))
    data=mujoco.MjData(model)
    collision=[i for i in range(model.ngeom) if model.body(model.geom_bodyid[i]).name.startswith('robot/') and
        (model.geom_contype[i] or model.geom_conaffinity[i])]
    assert all(model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH for i in collision)
    rows=[]
    for engine,folder in [('GPU','precheck-v2'),('CPU','precheck-v2-native-recorded')]:
        for i in range(20):
            path=a.root/folder/f'precheck_{i}.npz'
            with np.load(path) as f:
                t=f['time'];z=f['position'][:,2];vz=f['linear_world'][:,2];force=f['base_contact']
                def event(index):
                    return dict(time_s=float(t[index]),root_height_m=float(z[index]),
                        world_vertical_velocity_m_s=float(vz[index]),base_contact_N=float(force[index]))
                peak=int(z.argmax());pose_index=int(np.abs(f['qpos_time']-t[peak]).argmin())
                # 仅重建已录制姿态的几何，确认高点是实际离地而非镜头运动。
                data.qpos[:]=f['qpos'][pose_index];mujoco.mj_kinematics(model,data)
                clearance=min(float(mesh_vertices(model,data,j)[:,2].min()) for j in collision)
                rows.append(dict(engine=engine,sample=f'precheck_{i}',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    initial=event(0),peak_height=event(peak),peak_upward_velocity=event(int(vz.argmax())),
                    peak_base_contact=event(int(force.argmax())),base_contact_peak_weight_ratio=float(force.max()/f['weight'][0]),
                    nearest_peak_pose_time_s=float(f['qpos_time'][pose_index]),
                    minimum_robot_collision_mesh_height_at_nearest_peak_m=clearance))
    report=dict(physics_integration_steps=0,network_forwards=0,optimizer_updates=0,samples=rows,
        conclusion='CPU sample18 launches after a large recorded impact; finite-state checks do not establish dynamic validity',
        cause='Unresolved: saved traces lack full joint velocities, torques, and solver internals; no integration-step comparison was run')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([row for row in rows if row['sample']=='precheck_18'],indent=2))


if __name__=='__main__':main()
