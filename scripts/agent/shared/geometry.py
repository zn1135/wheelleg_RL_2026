"""标定及Isaac共用的静态几何；不提供multimode训练入口。"""
import mujoco
import numpy as np
from ct_mjlab.envs.chuanliantui.huananhu_poses import _front_kinematics, _mesh_vertices
from ct_mjlab.envs.chuanliantui.model import solve_standing_joint_positions, closed_chain_residual

def standing_endpoint(model, knee):
    data = mujoco.MjData(model)
    _, phi = _front_kinematics(model, data, knee)
    positions = solve_standing_joint_positions(
        model, front_joint_targets={'left': (phi, knee), 'right': (-phi, -knee)})
    data.qpos[:] = model.qpos0
    for name, value in positions.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    residual = closed_chain_residual(model, data)
    assert residual < 1e-6, residual
    for name, value in positions.items():
        joint = model.joint(name)
        if joint.limited:
            assert joint.range[0] - 1e-6 <= value <= joint.range[1] + 1e-6, name

    # Put the lowest wheel mesh vertex at z=0; this is geometry, not load balance.
    wheels = {model.body(name).id for name in ('lfwheel', 'rfwheel')}
    collision_ids = [gid for gid in range(model.ngeom) if model.geom_group[gid] == 3]
    wheel_bottom = min(float(_mesh_vertices(model, data, gid)[:, 2].min())
                       for gid in collision_ids if model.geom_bodyid[gid] in wheels)
    data.qpos[2] -= wheel_bottom
    mujoco.mj_forward(model, data)
    sides = {}
    for side, hip, wheel in (('left', 'lf0', 'lfwheel'), ('right', 'rf0', 'rfwheel')):
        delta = data.body(wheel).xpos - data.xanchor[model.joint(hip).id]
        sides[side] = {
            'hip_wheel_xz_length_m': float(np.hypot(delta[0], delta[2])),
            'hip_wheel_3d_distance_m': float(np.linalg.norm(delta)),
            'hip_wheel_vertical_projection_m': float(-delta[2]),
        }
    nonwheel = [
        (float(_mesh_vertices(model, data, gid)[:, 2].min()),
         model.body(model.geom_bodyid[gid]).name)
        for gid in collision_ids if model.geom_bodyid[gid] not in wheels]
    lowest_z, lowest_body = min(nonwheel)
    return {
        'left_knee_rad': knee,
        'sides': sides,
        'base_reference_height_on_wheel_plane_m': float(data.qpos[2]),
        'lowest_nonwheel_mesh_clearance_m': lowest_z,
        'lowest_nonwheel_body': lowest_body,
        'closed_chain_residual_m': residual,
    }
