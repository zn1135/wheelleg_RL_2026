"""Check positive spring effort and leg-extension virtual work at observed poses; zero integration."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.agent.real2sim import replay_leg_test_gpu as replay


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    source = Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')
    digest = hashlib.sha256(source.read_bytes()).hexdigest(); root = ET.parse(source).getroot()
    root.find('compiler').set('meshdir', str((source.parent/'../meshes').resolve()))
    base = root.find("worldbody/body[@name='base_link']"); base.remove(base.find('freejoint'))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode')); model.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    from wheel_response import prepare_wheel_response
    prepare_wheel_response(model)
    active = [model.joint(n).dofadr[0] for n in replay.mirror.MOTORS]
    passive = [i for i in range(model.nv) if i not in active]
    sliders = [model.joint(side+'_spring2_joint').dofadr[0] for side in ('left', 'right')]
    springs = [model.actuator(side+'_gas_spring_motor').id for side in ('left', 'right')]
    positions = []; provenance = []
    paths = [Path('artifacts/real2sim/h7-joint-response-refit-20261104-01')/p/'inverse-loads.npz' for p in
             ['inverse-preload-hypothesis', 'inverse-near-slow', 'inverse-near-fast']]
    paths.append(Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/inverse-seeds/inverse-original/inverse-loads.npz'))
    for path in paths:
        with np.load(path) as saved:
            for i in np.unique(np.linspace(0, len(saved['qpos'])-1, 21).astype(int)):
                positions.append((str(path), int(i), saved['qpos'][i].copy()))
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    reference = Path('artifacts/real2sim/h7-joint-response-refit-20261104-01/input-wide'); r = json.loads((reference/'report.json').read_text())
    metadata = json.loads((Path(r['session'])/'meta.json').read_text())
    pose = replay.mirror.MirrorPose(reference/'fixed-base-model.xml', Path(metadata['machine_config']['path']))
    for path in sorted(Path('artifacts/real2sim/h7-joint-limits-20261104-01').glob('*endpoint-*/report.json')):
        endpoint = json.loads(path.read_text())['model_joints_rad']
        for name, value in endpoint.items(): pose.data.qpos[pose.model.joint(name).qposadr[0]] = value
        for side in (0, 1):
            names = replay.mirror.MOTORS[side*2:side*2+2]
            assert pose._solve_leg(side, [endpoint[n] for n in names])
        positions.append((str(path), 0, pose.data.qpos.copy()))
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    rows = []; maximum_tangent_error = 0.; maximum_effort_error = 0.
    for path, index, q in positions:
        data = mujoco.MjData(model); data.qpos[:] = q; data.qvel[:] = 0.; data.ctrl[springs] = 1.
        mujoco.mj_forward(model, data)
        equality = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        jac = data.efc_J.reshape(data.nefc, model.nv)[equality]
        tangent = np.zeros((model.nv, 6)); tangent[active] = np.eye(6)
        tangent[passive], _, rank, _ = np.linalg.lstsq(jac[:, passive], -jac[:, active], rcond=None)
        assert rank == len(passive)
        maximum_tangent_error = max(maximum_tangent_error, float(np.max(abs(jac@tangent))))
        actual = tangent.T@data.qfrc_actuator
        expected = tangent[sliders].sum(0)
        maximum_effort_error = max(maximum_effort_error, float(np.max(abs(actual-expected))))
        jp = np.zeros((3, model.nv)); jr = np.zeros_like(jp)
        for side, prefix in enumerate(('lf', 'rf')):
            name = ['left', 'right'][side]; section = slice(side*2, side*2+2)
            foot = (data.body(prefix+'wheel').xpos-data.body(prefix+'0').xpos)[[0, 2]]
            mujoco.mj_jacBody(model, data, jp, jr, model.body(prefix+'wheel').id)
            length_jacobian = foot/np.linalg.norm(foot)@((jp@tangent)[[0, 2], section])
            spring_jacobian = tangent[sliders[side], section]
            equivalent_leg_force = float(spring_jacobian@length_jacobian/(length_jacobian@length_jacobian))
            jid = model.joint(name+'_spring2_joint').id
            delta = data.site(name+'_gas_spring_lower').xpos-data.site(name+'_gas_spring_upper').xpos
            axial = float(data.xaxis[jid]@delta)
            rows.append(dict(source=path, row=index, side=prefix, leg_length_m=float(np.linalg.norm(foot)),
                axial_eye_distance_m=axial, positive_slider_motion_increases_eye_distance=axial > 0.,
                actuator_gear=float(model.actuator_gear[springs[side], 0]),
                generalized_motor_torque_per_newton_spring_nm=spring_jacobian.tolist(),
                equivalent_outward_leg_force_per_newton_spring=equivalent_leg_force,
                assists_extension=equivalent_leg_force > 0.))
    assert maximum_tangent_error < 1e-6 and maximum_effort_error < 1e-9
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    result = dict(scope='Current model force-direction and virtual-work check at observed poses; no new physical calibration',
        environment_steps=0, source_model_sha256=digest, observed_pose_count=len(positions), evaluated_leg_poses=len(rows),
        positive_force_extends_spring_in_all_poses=all(r['positive_slider_motion_increases_eye_distance'] for r in rows),
        positive_force_assists_leg_extension_in_all_poses=all(r['assists_extension'] for r in rows),
        outward_leg_force_ratio_range=[min(r['equivalent_outward_leg_force_per_newton_spring'] for r in rows),
                                      max(r['equivalent_outward_leg_force_per_newton_spring'] for r in rows)],
        maximum_constraint_tangent_error=maximum_tangent_error, maximum_projected_actuator_effort_error_nm=maximum_effort_error,
        sources=provenance, rows=rows,
        limitations=['Direction correctness does not establish force magnitude, installation points, stroke endpoints or friction',
                     'Only observed poses and marked endpoints are checked, not every mathematically possible mechanism configuration'])
    (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'sources')}, indent=2))


if __name__ == '__main__': main()
