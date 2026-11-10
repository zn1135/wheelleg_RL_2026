"""Equal-torque mechanism diagnostics with gas force off; finite GPU pulses, no hardware."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from scripts.agent.real2sim import replay_leg_test_gpu as replay


NAMES = ('lf0', 'lf00', 'rf0', 'rf00')
SIGN = np.array([1., 1., -1., -1.])
PATTERNS = dict(zero=[0., 0., 0., 0.], front_only=[2., 0., -2., 0.],
                rear_only=[0., 2., 0., -2.], common=[2., 2., -2., -2.],
                retract=[2., -2., -2., 2.], extend=[-2., 2., 2., -2.])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reduced_dynamics(model, data):
    motors = [model.joint(n).dofadr[0] for n in NAMES]
    wheels = [model.joint(n).dofadr[0] for n in ('lfwheel', 'rfwheel')]
    rows = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    jac = data.efc_J.reshape(data.nefc, model.nv)[rows]
    rank = int(np.linalg.matrix_rank(jac, tol=1e-8)) if len(jac) else 0
    if rank:
        independent = motors+wheels; passive = [i for i in range(model.nv) if i not in independent]
        tangent = np.zeros((model.nv, 6)); tangent[independent] = np.eye(6)
        tangent[passive], _, passive_rank, _ = np.linalg.lstsq(jac[:, passive], -jac[:, independent], rcond=None)
        assert passive_rank == len(passive) and model.nv-rank == 6
        tangent_error = float(np.max(abs(jac@tangent))); assert tangent_error < 1e-6
    else:
        independent = motors+[i for i in range(model.nv) if i not in motors]
        tangent = np.eye(model.nv)[:, independent]; tangent_error = 0.
    mass = np.empty((model.nv, model.nv)); mujoco.mj_fullM(model, data, mass)
    projected = tangent.T@mass@tangent
    eliminate = np.vstack((np.eye(4), -np.linalg.solve(projected[4:, 4:], projected[4:, :4])))
    basis = tangent@eliminate; effective = basis.T@mass@basis
    body_sum = np.zeros((4, 4)); parts = []
    jp = np.zeros((3, model.nv)); jr = np.zeros_like(jp)
    for body in range(1, model.nbody):
        mujoco.mj_jacBodyCom(model, data, jp, jr, body)
        rotation = data.ximat[body].reshape(3, 3)
        inertia = rotation@np.diag(model.body_inertia[body])@rotation.T
        translation = jp@basis; angular = jr@basis
        contribution = model.body_mass[body]*(translation.T@translation)+angular.T@inertia@angular
        body_sum += contribution
        parts.append(dict(body=model.body(body).name, mass_kg=float(model.body_mass[body]),
                          motor_diagonal_kg_m2=np.diag(contribution).tolist()))
    armature = basis.T@np.diag(model.dof_armature)@basis
    reconstruction = float(np.max(abs(body_sum+armature-effective))); assert reconstruction < 1e-10
    foot = []
    for side, prefix in enumerate(('lf', 'rf')):
        mujoco.mj_jacBody(model, data, jp, jr, model.body(prefix+'wheel').id)
        position = (data.body(prefix+'wheel').xpos-data.body(prefix+'0').xpos)[[0, 2]]
        derivative = (jp@basis)[[0, 2], side*2:side*2+2]
        angle = (position[0]*derivative[1]-position[1]*derivative[0])/np.dot(position, position)
        foot.append(dict(side=prefix, length_m=float(np.linalg.norm(position)),
                         angular_jacobian_per_motor=angle.tolist()))
    return dict(equality_rank=rank, independent_dofs=model.nv-rank, motor_dofs=4,
                constraint_tangent_error=tangent_error, effective_motor_inertia_kg_m2=effective.tolist(),
                analytic_retract_acceleration_rad_s2=np.linalg.solve(effective, PATTERNS['retract']).tolist(),
                body_contributions=parts, armature_contribution_kg_m2=armature.tolist(),
                mass_reconstruction_max_error_kg_m2=reconstruction, foot_kinematics=foot)


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    source = Path('resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')
    digest = sha(source); root = ET.parse(source).getroot()
    root.find('compiler').set('meshdir', str((source.parent/'../meshes').resolve()))
    base = root.find("worldbody/body[@name='base_link']"); base.remove(base.find('freejoint')); base.set('pos', '0 0 .8')
    option = root.find('option')
    if option is None: option = ET.SubElement(root, 'option')
    option.attrib.update(timestep='.001', integrator='implicitfast', jacobian='dense', iterations='200', tolerance='1e-12')
    from wheel_response import prepare_wheel_response
    reference_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    pose_files = [('extended', Path('artifacts/real2sim/h7-joint-response-fit-20261104-02/inverse-seeds/inverse-original/inverse-loads.npz'), 0),
                  ('middle', Path('artifacts/real2sim/h7-joint-response-refit-20261104-01/inverse-preload-hypothesis/inverse-loads.npz'), 0),
                  ('folded', Path('artifacts/real2sim/h7-joint-response-refit-20261104-01/inverse-preload-hypothesis/inverse-loads.npz'), 2400)]
    poses = []
    for name, path, index in pose_files:
        with np.load(path) as data: q = data['qpos'][index].copy()
        assert q.shape == (reference_model.nq,)
        poses.append(dict(name=name, q=q, source=str(path), row=index, sha256=sha(path)))
    variants = [('closed_stiff_ideal', True, False, False, False, False),
                ('closed_soft_ideal', False, False, False, False, False),
                ('open_ideal', True, True, False, False, False),
                ('spring_assembly_removed', True, False, True, False, False),
                ('closed_nominal_no_gravity', True, False, False, True, False),
                ('closed_nominal_gravity', True, False, False, True, True)]
    import mujoco_warp as mw
    import warp as wp
    wp.init(); results = []; total_steps = 0
    for name, stiff, opened, remove_spring, nominal, gravity in variants:
        directory = args.output/name; directory.mkdir()
        tree = copy.deepcopy(root)
        if stiff:
            for eq in tree.find('equality'):
                eq.set('solref', '.002 1'); eq.set('solimp', '.999 .9999 .001 .5 2')
        if opened: tree.remove(tree.find('equality'))
        if remove_spring:
            parents = {child: parent for parent in tree.iter() for child in parent}
            for side in ('left', 'right'):
                node = tree.find(f".//body[@name='{side}_spring1']"); parents[node].remove(node)
                eq = tree.find(f"equality/connect[@name='{side}_spring_loop']"); tree.find('equality').remove(eq)
                actuator = tree.find(f"actuator/motor[@name='{side}_gas_spring_motor']"); tree.find('actuator').remove(actuator)
        xml = ET.tostring(tree, encoding='unicode'); (directory/'model.xml').write_text(xml)
        model = mujoco.MjModel.from_xml_string(xml); prepare_wheel_response(model)
        model.geom_contype[:] = model.geom_conaffinity[:] = 0
        if not nominal: model.dof_frictionloss[:] = 0.; model.dof_damping[:] = 0.
        if not gravity: model.opt.gravity[:] = 0.
        # 气弹簧力与轴向阻尼均关闭；保留组件的案例仅保留其真实质量和铰接。
        for side in ('left', 'right'):
            if not remove_spring: model.dof_damping[model.joint(side+'_spring2_joint').dofadr[0]] = 0.
        mujoco.mj_saveLastXML(str(directory/'compiled-model.xml'), model)
        mq = [model.joint(n).qposadr[0] for n in NAMES]; mv = [model.joint(n).dofadr[0] for n in NAMES]
        ids = [model.actuator(n+'_motor').id for n in NAMES]
        source_indices = [reference_model.joint(model.joint(j).name).qposadr[0] for j in range(model.njnt)]
        initial = []; labels = []; controls = []; analytic = []
        for pose in poses:
            data = mujoco.MjData(model); data.qpos[:] = pose['q'][source_indices]; data.qvel[:] = 0.
            mujoco.mj_forward(model, data); analytic.append(dict(pose=pose['name'], **reduced_dynamics(model, data)))
            for pattern, torque in PATTERNS.items():
                initial.append(data.qpos.copy()); labels.append((pose['name'], pattern))
                control = np.zeros(model.nu, dtype=np.float32); control[ids] = torque; controls.append(control)
        nworld = len(initial); q0 = np.asarray(initial, dtype=np.float32); controls = np.asarray(controls)
        data = mujoco.MjData(model)
        with wp.ScopedDevice('cuda:0'):
            gm = mw.put_model(model); gd = mw.put_data(model, data, nworld=nworld, nconmax=128, njmax=256)
            gd.qpos.assign(q0); gd.qvel.assign(np.zeros((nworld, model.nv), dtype=np.float32)); gd.ctrl.assign(controls)
            mw.forward(gm, gd)
            acceleration = gd.qacc.numpy().copy(); initial_constraint = gd.qfrc_constraint.numpy().copy()
            initial_actuator = gd.qfrc_actuator.numpy().copy()
            assert np.max(abs(initial_actuator[:, mv]-controls[:, ids])) < 1e-5
            spring_ids = [i for i in range(model.nu) if 'gas_spring' in model.actuator(i).name]
            if spring_ids: assert not np.any(gd.actuator_force.numpy()[:, spring_ids])
            with wp.ScopedCapture() as capture: mw.step(gm, gd)
            assert not np.any(gd.time.numpy())
            q = [gd.qpos.numpy().copy()]; v = [gd.qvel.numpy().copy()]
            for step in range(120):
                wp.capture_launch(capture.graph); q.append(gd.qpos.numpy().copy()); v.append(gd.qvel.numpy().copy())
            final_time = gd.time.numpy(); assert np.max(abs(final_time-.12)) < 1e-6
        q = np.asarray(q); v = np.asarray(v); assert np.isfinite(q).all() and np.isfinite(v).all()
        closure = np.zeros((len(q), nworld))
        limited = np.flatnonzero(model.jnt_limited); lq = model.jnt_qposadr[limited]
        touched = np.any((q[:, :, lq] < model.jnt_range[limited, 0]+model.jnt_margin[limited]) |
                         (q[:, :, lq] > model.jnt_range[limited, 1]-model.jnt_margin[limited]), axis=2)
        for step, positions in enumerate(q):
            for world, position in enumerate(positions):
                data.qpos[:] = position; mujoco.mj_kinematics(model, data)
                closure[step, world] = max(np.linalg.norm(data.site(a).xpos-data.site(b).xpos)
                    for pairs in replay.mirror.PIN_PAIRS for a, b in pairs)
                if not remove_spring:
                    closure[step, world] = max(closure[step, world], *(np.linalg.norm(data.site(s+'_spring_tip').xpos-data.site(s+'_gas_spring_lower').xpos) for s in ('left', 'right')))
        measurements = []
        for world, (pose, pattern) in enumerate(labels):
            zero = labels.index((pose, 'zero'))
            row = dict(pose=pose, pattern=pattern, model_torque_nm=controls[world, ids].tolist(),
                       initial_gpu_motor_acceleration_rad_s2=acceleration[world, mv].tolist(),
                       initial_gpu_constraint_motor_torque_nm=initial_constraint[world, mv].tolist())
            for index in (40, 120):
                response = ((q[index, world, mq]-q[0, world, mq])-(q[index, zero, mq]-q[0, zero, mq]))*SIGN
                row[f'net_{index}ms_motor_displacement_deg'] = np.degrees(response).tolist()
                row[f'joint_limit_touched_through_{index}ms'] = bool(touched[:index+1, world].any() or touched[:index+1, zero].any())
            row['max_pin_error_m'] = float(closure[:, world].max()); measurements.append(row)
        np.savez_compressed(directory/'trajectories.npz', time_s=np.arange(121)*.001, qpos=q, qvel=v, controls=controls,
                            motor_q_indices=mq, motor_v_indices=mv, labels=np.asarray(labels),
                            initial_qacc=acceleration, initial_qfrc_constraint=initial_constraint,
                            initial_qfrc_actuator=initial_actuator, closure_error_m=closure, joint_limit_touched=touched)
        entry = dict(name=name, stiff=stiff, opened=opened, spring_assembly_removed=remove_spring,
                     nominal_passive_resistance=nominal, gravity=gravity, gas_force_enabled=False,
                     worlds=nworld, steps_per_world=120, new_environment_steps=nworld*120,
                     analytics=analytic, measurements=measurements, model_xml_sha256=sha(directory/'model.xml'))
        (directory/'report.json').write_text(json.dumps(entry, indent=2)+'\n'); results.append(entry); total_steps += nworld*120
        print(json.dumps(dict(completed=name, environment_steps=nworld*120)), flush=True)
    assert sha(source) == digest
    summary = dict(scope='Finite equal-torque mechanism diagnosis only; not a real-hardware or complete trajectory fit acceptance',
                   source_model=str(source.resolve()), source_model_sha256=digest, source_model_unchanged=True,
                   hardware_access=False, policy_training_updates=0, physics_backend='mujoco_warp_gpu',
                   new_environment_steps=total_steps, pulse_duration_s=.12, pulse_magnitude_nm=2.,
                   canonical_position_sign=SIGN.tolist(), patterns_model_torque_nm=PATTERNS,
                   poses=[{k: v for k, v in pose.items() if k != 'q'} for pose in poses], variants=results,
                   limitations=['Open mechanism is an artificial comparison only',
                                'Gas force off retains assembly inertia unless explicitly marked removed',
                                'Angles cannot be compared as free responses after a joint limit becomes active',
                                'Near-equal real-hardware torque responses require the same starting posture and loading to compare quantitatively'])
    (args.output/'report.json').write_text(json.dumps(summary, indent=2)+'\n'); shutil.copy2(__file__, args.output/Path(__file__).name)
    print(json.dumps(dict(completed_variants=len(results), environment_steps=total_steps, source_model_unchanged=True)), flush=True)


if __name__ == '__main__': main()
