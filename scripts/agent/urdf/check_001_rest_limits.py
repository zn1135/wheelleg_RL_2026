"""GPU comparison of the 001 exported hip limits at rest with passive hips.

The candidate only removes the four exported full-turn hip bounds in memory.
It is a diagnosis of the current bench model, not hardware travel calibration.
"""
import json
from pathlib import Path

import mujoco as mj
import mujoco_warp as mw
import numpy as np
import torch
import warp as wp

SOURCE = Path("artifacts/urdf-align/20261126-001-closed/robot_001_closed_mouse_free_passive.xml")
OUTPUT = Path("artifacts/urdf-align/20261126-001-limit-review/hip_rest_comparison.json")
HIPS = ("lf0", "lf00", "rf0", "rf00")
JOINTS = HIPS + ("lf1", "rf1")


def run_case(label, unlimited):
    spec = mj.MjSpec.from_file(str(SOURCE))
    for name in ("lfwheel_motor", "rfwheel_motor"):
        spec.actuator(name).gaintype = mj.mjtGain.mjGAIN_FIXED
    if unlimited:
        for name in HIPS:
            spec.joint(name).limited = False
    model = spec.compile()
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, model.key("manual_center").id)
    mj.mj_forward(model, data)
    gpu_model = mw.put_model(model)
    gpu = mw.put_data(model, data, nworld=1, nconmax=32, njmax=128)
    ctrl = wp.to_torch(gpu.ctrl)
    ctrl[:, 2:] = 400
    for _ in range(1500):
        mw.step(gpu_model, gpu)
    wp.synchronize()
    data.qpos[:] = gpu.qpos.numpy()[0]
    data.qvel[:] = gpu.qvel.numpy()[0]
    data.ctrl[2:] = 400
    mj.mj_forward(model, data)
    records = {}
    for name in JOINTS:
        joint = model.joint(name)
        q = float(data.qpos[joint.qposadr[0]])
        velocity = float(data.qvel[joint.dofadr[0]])
        records[name] = dict(
            angle_rad=q, speed_rad_s=velocity,
            limited=bool(joint.limited[0]), range_rad=joint.range.tolist(),
        )
    active_limits = []
    for row in np.flatnonzero(data.efc_type == mj.mjtConstraint.mjCNSTR_LIMIT_JOINT):
        jid = data.efc_id[row]
        active_limits.append(dict(joint=model.joint(jid).name,
                                  constraint_force=float(data.efc_force[row]),
                                  violation_m_or_rad=float(data.efc_pos[row])))
    wheels = {}
    for name in ("lfwheel", "rfwheel"):
        wheels[name] = (data.body(name).xpos - data.body("base_link").xpos).tolist()
    result = dict(case=label, time_s=float(gpu.time.numpy()[0]),
                  joints=records, active_limits=active_limits, wheel_body_positions_m=wheels)
    assert np.isfinite(data.qpos).all()
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    wp.init()
    torch.empty(1, device="cuda")
    with wp.ScopedDevice("cuda:0"), wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
        results = [run_case("current hip limits", False),
                   run_case("diagnostic hip limits removed", True)]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
