"""Prepare measured-pose reconstruction and same-request GPU baselines for visual comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.agent.real2sim import replay_leg_test_gpu as replay


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    source = Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'); source_hash = sha(source)
    root = ET.parse(source).getroot(); root.find('compiler').set('meshdir', str((source.parent/'../meshes').resolve()))
    base = root.find("worldbody/body[@name='base_link']"); base.remove(base.find('freejoint')); base.set('pos', '0 0 .8')
    option = root.find('option')
    if option is None: option = ET.SubElement(root, 'option')
    option.attrib.update(timestep='.001', integrator='implicitfast', solver='Newton', iterations='100', tolerance='1e-10', impratio='10', jacobian='dense')
    from wheel_response import prepare_wheel_response
    import mujoco_warp as mw
    import warp as wp
    wp.init(); sessions = []; total = 0
    references = [('original', '原始收腿 · 8秒', Path('artifacts/real2sim/h7-response-align-20261104-01/final-elastic-replay')),
                  ('wide', '大幅往返 · 35.92秒', Path('artifacts/real2sim/h7-joint-response-refit-20261104-01/input-wide'))]
    display = ET.parse(source).getroot(); display.find('compiler').set('meshdir', str((source.parent/'../meshes').resolve()))
    for body in display.findall('.//body'):
        for geom in list(body.findall('geom')):
            if geom.get('class') == 'collision': body.remove(geom)
    (args.output/'display-model.xml').write_text(ET.tostring(display, encoding='unicode'))
    for name, label, reference in references:
        directory = args.output/name; directory.mkdir(); report = json.loads((reference/'report.json').read_text())
        assert report['source_model_sha256'] == source_hash and report['dt_s'] == .001
        saved = np.load(reference/'replay.npz'); controls = saved['submitted_model_torque'][:, 0].astype(np.float32)
        steps = len(controls); indices = np.arange(0, steps+1, 10); times = saved['time_s'][indices]
        metadata = json.loads((Path(report['session'])/'meta.json').read_text()); config = Path(metadata['machine_config']['path'])
        assert sha(config) == report['config_sha256']; sign, zero = replay.mirror.load_mapping(config)
        template = ET.fromstring(ET.tostring(root, encoding='unicode'))
        reference_base = ET.parse(reference/'fixed-base-model.xml').getroot().find("worldbody/body[@name='base_link']")
        template.find("worldbody/body[@name='base_link']").set('quat', reference_base.get('quat', '1 0 0 0'))
        (directory/'measured-pose-template.xml').write_text(ET.tostring(template, encoding='unicode'))
        pose = replay.mirror.MirrorPose(directory/'measured-pose-template.xml', config)
        pose.data.qpos[:] = saved['qpos'][0, 0]
        absolute = times+report['window_start_board_s']; raw = saved['real_motor_state']
        measured = np.column_stack([np.interp(absolute, raw[:, 0], raw[:, i+1]) for i in range(4)])
        real_q = []; real_closure = []
        for angles in measured:
            angles = (angles-zero)*sign[:4]
            assert all(pose._solve_leg(side, angles[side*2:side*2+2]) for side in (0, 1)), 'Measured pose cannot close'
            real_q.append(pose.data.qpos.copy()); real_closure.append(max(pose.residual_m))
        real_q = np.asarray(real_q); assert max(real_closure) < 1e-6
        models = []; model_q = []; model_v = []; model_closure = []; spring_controls = []
        for stiffness in ('source', 'stiff'):
            tree = ET.fromstring(ET.tostring(template, encoding='unicode'))
            if stiffness == 'stiff':
                for eq in tree.find('equality'):
                    eq.set('solref', '.002 1'); eq.set('solimp', '.999 .9999 .001 .5 2')
            model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode')); prepare_wheel_response(model)
            model.geom_contype[:] = model.geom_conaffinity[:] = 0
            data = mujoco.MjData(model); data.qpos[:] = saved['qpos'][0, 0]; data.qvel[:] = saved['qvel'][0, 0]
            motor_ids = [model.actuator(n+'_motor').id for n in replay.mirror.MOTORS]
            spring_ids = [model.actuator(s+'_gas_spring_motor').id for s in ('left', 'right')]
            sq = [model.joint(s+'_spring2_joint').qposadr[0] for s in ('left', 'right')]
            sv = [model.joint(s+'_spring2_joint').dofadr[0] for s in ('left', 'right')]
            model_path = directory/(stiffness+'-model.xml'); model_path.write_text(ET.tostring(tree, encoding='unicode'))
            count = len(indices); q = np.empty((count, 2, model.nq), dtype=np.float32); v = np.empty_like(q)
            spring_inputs = np.empty((count-1, 2, 2), dtype=np.float32)
            ctrl = np.zeros((2, model.nu), dtype=np.float32)
            with wp.ScopedDevice('cuda:0'):
                gm = mw.put_model(model); gd = mw.put_data(model, data, nworld=2, nconmax=128, njmax=256)
                with wp.ScopedCapture() as capture: mw.step(gm, gd)
                assert not np.any(gd.time.numpy()); qp = gd.qpos.numpy(); qv = gd.qvel.numpy(); q[0] = qp; v[0] = qv
                for k in range(steps):
                    ctrl[:, motor_ids] = controls[k]
                    ctrl[0, spring_ids] = replay.training_spring.gas_spring_force(qp[0, sq], qv[0, sv])
                    ctrl[1, spring_ids] = 0.
                    assert np.array_equal(ctrl[:, motor_ids], np.broadcast_to(controls[k], (2, 6)))
                    if k%10 == 0: spring_inputs[k//10] = ctrl[:, spring_ids]
                    gd.ctrl.assign(ctrl); wp.capture_launch(capture.graph); qp = gd.qpos.numpy(); qv = gd.qvel.numpy()
                    assert np.isfinite(qp).all() and np.isfinite(qv).all()
                    if (k+1)%10 == 0: q[(k+1)//10] = qp; v[(k+1)//10] = qv
                assert np.max(abs(gd.time.numpy()-steps*.001)) < .01
            closure = np.zeros((count, 2))
            for k, states in enumerate(q):
                for branch, state in enumerate(states):
                    data.qpos[:] = state; mujoco.mj_kinematics(model, data)
                    closure[k, branch] = max(np.linalg.norm(data.site(a).xpos-data.site(b).xpos)
                        for pairs in replay.mirror.PIN_PAIRS for a, b in pairs)
            model_q.append(q); model_v.append(v); model_closure.append(closure); spring_controls.append(spring_inputs)
            models.extend([f'{stiffness}_gas_on', f'{stiffness}_gas_off']); total += steps*2
            print(json.dumps(dict(completed=name+'/'+stiffness, environment_steps=steps*2)), flush=True)
        simulation = np.concatenate(model_q, axis=1); velocity = np.concatenate(model_v, axis=1)
        closure = np.concatenate(model_closure, axis=1); gas = np.concatenate(spring_controls, axis=1)
        display_model = mujoco.MjModel.from_xml_path(str(directory/'measured-pose-template.xml'))
        mq = [display_model.joint(n).qposadr[0] for n in replay.mirror.MOTORS[:4]]
        assert np.allclose(real_q[:, mq]*sign[:4]+zero, measured, atol=1e-9)
        error = np.degrees(simulation[:, :, mq]*sign[:4]+zero-measured[:, None, :])
        states = np.concatenate((real_q[:, None], simulation), axis=1)
        xpos = np.empty((len(times), 5, display_model.nbody, 3), dtype=np.float32)
        xmat = np.empty((len(times), 5, display_model.nbody, 3, 3), dtype=np.float32)
        data = mujoco.MjData(display_model)
        for i, row in enumerate(states):
            for branch, state in enumerate(row):
                data.qpos[:] = state; mujoco.mj_kinematics(display_model, data)
                xpos[i, branch] = data.xpos; xmat[i, branch] = data.xmat.reshape(-1, 3, 3)
        np.savez_compressed(directory/'comparison.npz', time_s=times, qpos=states, simulation_qvel=velocity,
            xpos=xpos, xmat=xmat, measured_firmware_angles_rad=measured, simulation_error_deg=error,
            submitted_model_torque_nm=controls, sampled_spring_controls_n=gas,
            model_closure_error_m=closure, real_reconstruction_closure_m=real_closure,
            motor_q_indices=mq, sign=sign, zero=zero)
        result = dict(name=name, label=label, reference=str(reference), session=report['session'],
            recording_sha256=report['stream_sha256'], source_model_sha256=source_hash, config_sha256=report['config_sha256'],
            mode='Recorded remote leg-length mode with gas springs installed; not the later described independent pure-torque test',
            models=models, models_have_no_fitted_motor_friction=True, simulation_uses_future_measured_positions=False,
            real_branch='Measured motor positions reconstructed by closed-chain kinematics; not physical simulation or camera footage',
            sample_dt_s=.01, physics_dt_s=.001, duration_s=float(times[-1]), new_gpu_environment_steps=steps*4,
            joint_rmse_deg=np.sqrt(np.mean(error**2, axis=0)).tolist(), max_loop_pin_error_m=closure.max(0).tolist(),
            source_reference_hashes={str(reference/f):sha(reference/f) for f in ['report.json', 'replay.npz', 'fixed-base-model.xml']},
            comparison_sha256=sha(directory/'comparison.npz'), real_motor_span_deg=np.degrees(np.ptp(measured, axis=0)).tolist(),
            gas_off_is_diagnostic_only=True, candidate_accepted=False)
        (directory/'report.json').write_text(json.dumps(result, indent=2)+'\n'); sessions.append(result)
    assert sha(source) == source_hash
    manifest = dict(scope='Read-only visual comparison of measured motor reconstruction and same-request GPU predictions',
        source_model_sha256=source_hash, display_model_sha256=sha(args.output/'display-model.xml'), sessions=sessions,
        new_gpu_environment_steps=total, physical_steps_in_real_reconstruction=0, hardware_access=False,
        policy_or_network_updates=0, source_model_modified=False, response_fit_accepted=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n'); shutil.copy2(__file__, args.output/Path(__file__).name)
    print(json.dumps(dict(prepared_sessions=len(sessions), gpu_environment_steps=total)), flush=True)


if __name__ == '__main__': main()
