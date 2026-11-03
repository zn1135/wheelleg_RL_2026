"""Correct the original scene's platform-relative height using saved GPU poses only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    args = parser.parse_args()
    root = args.recording
    budget = json.loads((root / 'budget.json').read_text())
    if time.time() >= budget['deadline_unix']:
        raise TimeoutError('Original total budget expired')
    source = root / 'traces/reference_zero.npz'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source) as saved:
        trace = {k: saved[k].copy() for k in saved.files}
    model = mujoco.MjModel.from_binary_path(str(root / 'fixtures/model.mjb'))
    data = mujoco.MjData(model)
    platform = model.geom('landing_box').id
    assert model.geom_type[platform] == mujoco.mjtGeom.mjGEOM_BOX
    assert np.array_equal(model.geom_quat[platform], [1., 0., 0., 0.])
    assert model.geom_bodyid[platform] == 0
    center = model.geom_pos[platform]
    half = model.geom_size[platform]
    ground_z = float(center[2] + half[2])
    wheels = [model.body(n).id for n in ('l_wheel_Link', 'r_wheel_Link')]
    geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] in wheels]
    assert len(geoms) == 2
    clearance = []; margin = []
    # 只重建已保存姿态的几何，不调用mj_step或任何策略。
    for index in np.flatnonzero(trace['qpos_time'] >= 2.):
        if time.time() >= budget['deadline_unix']:
            raise TimeoutError('Original total budget expired')
        data.qpos[:] = trace['qpos'][index]
        mujoco.mj_kinematics(model, data)
        for geom in geoms:
            assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER
            axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
            radius, length = model.geom_size[geom, :2]
            extent = radius * np.sqrt(np.maximum(0., 1. - axis**2)) + length * np.abs(axis)
            position = data.geom_xpos[geom]
            margin.append(float(np.min(half[:2] - np.abs(position[:2] - center[:2]) - extent[:2])))
            clearance.append(float(position[2] - extent[2] - ground_z))
    assert min(margin) > 0., 'Wheel footprint leaves horizontal platform; cannot apply constant support height'
    assert max(np.abs(clearance)) < .005, 'Saved wheel geometry does not establish contact with this platform'
    target = root / 'platform-relative'
    target.mkdir(exist_ok=False)
    (target / 'traces').mkdir(); (target / 'fixtures').mkdir()
    protocol = json.loads((root / 'protocol.json').read_text())
    protocol['position_frame'] = dict(xy='world', z='world minus verified landing_box top', support_height_m=ground_z)
    (target / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    shutil.copy2(root / 'execution.json', target / 'execution.json')
    for path in (root / 'fixtures').iterdir():
        shutil.copy2(path, target / 'fixtures' / path.name)
    trace['position_world'] = trace['position'].copy()
    trace['position'][:, 2] -= ground_z
    np.savez_compressed(target / 'traces/reference_zero.npz', **trace)
    result = dict(source_trace_sha256=source_hash, source_unchanged=hashlib.sha256(source.read_bytes()).hexdigest() == source_hash,
        source_file=str(source), derived_recording=str(target), support_geom='landing_box', support_height_m=ground_z,
        checked_saved_poses=len(clearance)//2, minimum_xy_footprint_margin_m=min(margin),
        wheel_surface_clearance_range_m=[min(clearance), max(clearance)],
        corrected_steady_height_mean_m=float(trace['position'][trace['time']>=2., 2].mean()),
        fields_changed=['position[:,2]'], fields_added=['position_world'], physics_steps=0, model_forwards=0,
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), finished_unix=time.time())
    (root / 'platform-height-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
