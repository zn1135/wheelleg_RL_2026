"""Check transferred load indices/parameters and paired GPU traces without integration."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--branch', default='candidate-both')
    args = parser.parse_args()
    root, branch = args.recording, args.recording / args.branch
    output = branch / 'transfer-verification.json'
    assert not output.exists()
    protocol = json.loads((branch / 'protocol.json').read_text())
    baseline = json.loads((root / 'baseline/protocol.json').read_text())
    assert protocol['samples'] == baseline['samples']
    identity = protocol['response_parameters']
    bundle_path = Path(identity['path'])
    assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == identity['sha256']
    bundle = json.loads(bundle_path.read_text())
    for path, digest in bundle['source'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    p, sides = bundle['candidate'], bundle.get('sides', ['right'])
    model = mujoco.MjModel.from_binary_path(str(branch / 'fixtures/model.mjb'))
    reference = mujoco.MjModel.from_binary_path(str(root / 'baseline/fixtures/model.mjb'))
    assert np.array_equal(model.dof_armature, reference.dof_armature)
    assert np.array_equal(model.actuator_ctrlrange, reference.actuator_ctrlrange)
    qids, vids, sids = [], [], []
    for side in sides:
        prefix = 'lf' if side == 'left' else 'rf'
        joints = [model.joint('robot/' + prefix + suffix) for suffix in ('0', '00')]
        qids.append([j.qposadr[0] for j in joints])
        vids.append([j.dofadr[0] for j in joints])
        sids.append(model.joint('robot/' + side + '_spring2_joint').dofadr[0])
    with np.load(branch / 'response-batch-0/response-audit.npz') as source:
        a = source['audit'].astype(float)
        assert np.array_equal(source['motor_q'], qids)
        assert np.array_equal(source['motor_v'], vids)
        assert source['sides'].tolist() == sides
        friction = source['dof_frictionloss'].copy()
        damping = source['dof_damping'].copy()
        slider_q = source['slider_q'].copy()
    expected_friction = np.tile(model.dof_frictionloss, (len(protocol['samples']), 1))
    expected_damping = np.tile(model.dof_damping, (len(protocol['samples']), 1))
    expected_friction[:, np.asarray(vids)] = p['static_nm']
    expected_damping[:, np.asarray(vids)] = p['damping_nms_rad']
    expected_damping[:, sids] += p['seal_damping_ns_m']
    friction_error = float(abs(friction-expected_friction).max())
    damping_error = float(abs(damping-expected_damping).max())
    signs = np.array([-1. if s == 'left' else 1. for s in sides])
    parity = slider_parity = 0.
    for world, sample in enumerate(protocol['samples']):
        sid = sample['id']
        with np.load(branch / 'fixtures' / (sid + '_initial.npz')) as left, np.load(root / 'baseline/fixtures' / (sid + '_initial.npz')) as right:
            assert set(left.files) == set(right.files)
            for key in left.files:assert np.array_equal(left[key], right[key]), (sid, key)
        with np.load(branch / 'traces' / (sid + '.npz')) as trace:
            n = min(len(trace['qpos'])-1, len(a)//5)
            states = trace['qpos'][:n]
            parity = max(parity, float(abs(a[:n*5:5, world, :, :2] - states[:, qids]*signs[None,:,None]).max()))
            slider_parity = max(slider_parity, float(abs(a[:n*5:5, world, :, 4] - states[:, slider_q]).max()))
    force_check = json.loads((branch / 'response-batch-0/response-verification.json').read_text())
    selected = protocol['selected_policy']
    assert hashlib.sha256(Path(selected['checkpoint']).read_bytes()).hexdigest() == selected['checkpoint_sha256']
    result = dict(passed=bool(force_check['passed'] and parity<1e-7 and slider_parity<1e-7
        and friction_error<1e-6 and damping_error<1e-4), samples=len(protocol['samples']),
        sides=sides, mirrored_left='left' in sides, initial_parameters_identical=True,
        initial_memory_force_n=0., motor_pose_parity_rad=parity, slider_pose_parity_m=slider_parity,
        friction_parameter_error_nm=friction_error, damping_parameter_error=damping_error,
        actuator_ranges_and_armature_unchanged=True, selected_policy=selected,
        physics_steps=0, policy_training_updates=0, hardware_commands=0,
        scope='Transfer/state/force check, not physical identification or behavioral acceptance')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    assert result['passed']


if __name__ == '__main__':main()
