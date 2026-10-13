"""A2500 high-start diagnosis, using poses reconstructed by the original solver.

No learning or imported-source edits. All outputs must be new campaign folders.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shlex
import sys
import traceback
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from ct_mjlab.experiments import recovery_factors as factors
from ct_mjlab.reference_training import huananhu_observations
from check_chuanliantui_xml_support_height import Audit, LEFT, PAIRS
from diagnose_chuanliantui_height_response import numerical_probe

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/high-start-a2500-20261013'
STATIC = ROOT / 'artifacts/wheel-support-xml-height-check-20261012/jobs/20261012-xml-wheel-support-height-v5/xml-height-audit.json'
MODEL = ROOT / 'artifacts/height-response-a2500-20261012/models/model_2499.pt'
SOURCE_HASH = '706c8387cba4815b491be3f5914a5951881c2966edabbd5fbb90d0da9c9d3dbd'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def reconstruct(path):
    source = json.loads(path.read_text())
    audit = Audit()
    assert source['asset_sha256'] == digest(ROOT / 'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml')
    result = dict(source_file=str(path), source_sha256=digest(path),
                  solver_file='scripts/agent/check_chuanliantui_xml_support_height.py',
                  solver_sha256=digest(ROOT / 'scripts/agent/check_chuanliantui_xml_support_height.py'),
                  poses_json_sha256=digest(ROOT / 'mjlab_training/src/ct_mjlab/poses.json'),
                  method='Original Audit.set_pose([joint_positions[n] for n in LEFT] + [radians(root_roll_deg)], target_height_m); nominal all other joints, wheel angles zero',
                  complete_qpos_origin='按原求解器约定还原的', nq=audit.model.nq, cases=[])
    assert sorted(case['target_height_m'] for case in source['cases']) == [.27, .30, .34]
    for case in source['cases']:
        x = np.r_[[case['joint_positions'][name] for name in LEFT], math.radians(case['root_roll_deg'])]
        audit.set_pose(x, case['target_height_m'])
        qpos = audit.data.qpos.copy()
        joints = {audit.model.joint(i).name: dict(type=int(audit.model.jnt_type[i]),
            qpos_address=int(audit.model.jnt_qposadr[i]),
            qpos=qpos[int(audit.model.jnt_qposadr[i]):int(audit.model.jnt_qposadr[i]) +
                       (7 if audit.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE else 1)].tolist())
                  for i in range(audit.model.njnt)}
        error = max(abs(qpos[audit.model.joint(name).qposadr[0]] - value)
                    for name, value in case['joint_positions'].items())
        pin = max(float(np.linalg.norm(audit.data.site_xpos[a] - audit.data.site_xpos[b]))
                  for a, b in audit.pairs)
        bottom = [float(audit.bottom(geom)[2]) for geom in audit.wheel_geoms]
        assert error == 0. and pin < 1e-6 and bottom == case['wheel_bottom_m']
        result['cases'].append(dict(target_height_m=case['target_height_m'], qpos=qpos.tolist(),
            joints=joints, stored_joint_mismatch=error, max_pin_error_m=pin, wheel_bottom_m=bottom,
            force_equilibrium=case['force_equilibrium']))
    return result


def force_snapshot(env):
    for name in ('wheel_ground_probe', 'nonwheel_ground_probe', 'base_contact', 'wheel_contact'):
        env.scene[name]._invalidate_cache()
    wheel = factors.wheel_force(env).clone()
    other = factors.other_force(env).clone()
    found = env.scene['wheel_ground_probe'].data.found.clone() > 0
    weight = factors.weight(env).clone()
    return wheel, other, found, weight


def initialize(env, pose, command, velocity=None):
    env.common_step_counter = 0
    env._sim_step_counter = 0
    env.reset(seed=42)
    ids = torch.arange(64, device=env.device)
    env.scene.reset(ids)
    model = env.sim.mj_model
    mapping = {}
    for name, value in pose['joints'].items():
        joint = model.joint('robot/' + name)
        assert int(model.jnt_type[joint.id]) == value['type']
        address = int(joint.qposadr[0])
        tensor = torch.tensor(value['qpos'], device=env.device, dtype=env.sim.data.qpos.dtype)
        env.sim.data.qpos[:, address:address + len(tensor)] = tensor
        if value['type'] == int(mujoco.mjtJoint.mjJNT_FREE):
            env.sim.data.qpos[:, address:address + 3] += env.scene.env_origins
        mapping[name] = dict(cpu_address=value['qpos_address'], scene_address=address, width=len(tensor))
    env.sim.data.qvel.zero_()
    env.sim.data.ctrl.zero_()
    env.sim.data.qfrc_applied.zero_()
    env.sim.data.xfrc_applied.zero_()
    env.sim.data.qacc_warmstart.zero_()
    if velocity is not None:
        adr = env.scene['robot'].indexing.free_joint_v_adr
        env.sim.data.qvel[:, adr] = velocity
    state = env.command_manager.get_term('standup')
    state.sampled[:, :2] = state.command[:, :2] = 0.
    state.sampled[:, 2] = state.command[:, 2] = command
    state.ground_command.zero_()
    state.ground_start.zero_()
    state.numerical_failure.zero_()
    action = env.action_manager.get_term('pd')
    q = env.sim.data.qpos[:, action._motor_q_adr]
    control = torch.zeros_like(action._control_actions)
    legs = action._leg_action_ids
    assert action.cfg.leg_position_scale == .5
    control[:, legs] = (q[:, legs] - action.nominal_positions[:, legs]) / .5
    policy = torch.zeros_like(control)
    for physical, name in enumerate(('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')):
        policy[:, action.policy_joint_order.index(name)] = control[:, physical]
    action.process_actions(policy)
    action._action_history[:] = control.unsqueeze(0)
    action.apply_actions()
    # Entity effort targets are written to ctrl by the scene before forward.
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    huananhu_observations.reset_observations(env, ids)
    env.observation_manager.reset(ids)
    env.obs_buf = env.observation_manager.compute(update_history=True)
    return mapping, policy


def initial_gate(env, pose, mapping, policy):
    model = env.sim.mj_model
    terrain = model.geom('terrain')
    terrain_height = float(model.geom_pos[terrain.id, 2])
    assert int(model.geom_type[terrain.id]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
    assert terrain_height == 0. and torch.count_nonzero(env.scene.env_origins[:, 2]) == 0
    wheel, other, found, weight = force_snapshot(env)
    sites = env.sim.data.site_xpos
    pin = torch.stack([(sites[:, model.site('robot/' + a).id] -
                        sites[:, model.site('robot/' + b).id]).norm(dim=-1) for a, b in PAIRS], -1).amax(-1)
    ratios = wheel.sum(-1) / weight
    checks = dict(both_wheels_touch=found.all(-1), total_load=(ratios >= .8) & (ratios <= 1.2),
                  nonwheel=other <= .05 * weight, closed_chain=pin < 1e-6)
    # Verify the complete named mapping, including quaternion and environment offsets.
    errors = {}
    for name, value in pose['joints'].items():
        address = mapping[name]['scene_address']
        expected = torch.tensor(value['qpos'], device=env.device, dtype=env.sim.data.qpos.dtype).repeat(64, 1)
        if value['type'] == int(mujoco.mjtJoint.mjJNT_FREE):
            expected[:, :3] += env.scene.env_origins
        errors[name] = float((env.sim.data.qpos[:, address:address + len(value['qpos'])] - expected).abs().max())
    assert max(errors.values()) == 0.
    result = dict(target_height_m=pose['target_height_m'], terrain_height_m=terrain_height,
        mapping=mapping, named_mapping_max_error=errors, qvel_max=float(env.sim.data.qvel.abs().max()),
        wheel_support_n=wheel.cpu().tolist(), nonwheel_support_n=other.cpu().tolist(),
        wheel_found=found.cpu().tolist(), robot_weight_n=weight.cpu().tolist(),
        total_wheel_weight_ratio=ratios.cpu().tolist(), max_pin_error_m=pin.cpu().tolist(),
        action_history_method='all physical history slots=(q_motor-nominal)/0.5 for four legs; wheels=0; raw previous policy action synchronized',
        policy_previous_actions=policy.cpu().tolist(),
        initial_ctrl=env.sim.data.ctrl.cpu().tolist(),
        passed_counts={key:int(value.sum()) for key, value in checks.items()},
        passed=all(bool(value.all()) for value in checks.values()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--static-solutions', type=Path, default=STATIC)
    parser.add_argument('--checkpoint', type=Path, default=MODEL)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gate-only', action='store_true')
    args = parser.parse_args()
    if args.output.resolve() == OUT or not args.output.resolve().is_relative_to(OUT):
        parser.error('Output must be a new subdirectory of artifacts/high-start-a2500-20261013')
    args.output.mkdir(parents=True, exist_ok=False)
    run = dict(started=now(), ended=None, command=shlex.join([sys.executable, *sys.argv]),
        learning=False, optimizer_loaded=False, num_envs=64, device=args.device, exit_code=None,
        initialization=[], versions={name:importlib.metadata.version(name) for name in
            ('torch', 'mjlab', 'mujoco', 'mujoco-warp', 'warp-lang')}, source_hashes={})
    frozen = [path for folder in ('mjlab_training/src/ct_mjlab/experiments',
        'mjlab_training/src/ct_mjlab/reference_training', 'scripts/agent')
        for path in (ROOT / folder).rglob('*.py') if path.resolve() != Path(__file__).resolve()]
    frozen.append(ROOT / 'mjlab_training/src/ct_mjlab/wheel_balance_assessment.py')
    run['source_hashes'] = {str(path.relative_to(ROOT)):digest(path) for path in frozen}
    wrapped = None
    try:
        assert digest(args.checkpoint) == SOURCE_HASH
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        assert factors.validate_checkpoint(saved, 'baseline')['completed_updates'] == 2500
        assert torch.cuda.is_available() and args.device.startswith('cuda')
        torch.set_num_threads(1)
        run['gpu'] = torch.cuda.get_device_name(args.device)
        rebuilt = reconstruct(args.static_solutions)
        qpath = OUT / 'static-qpos.json'
        if qpath.exists():
            assert json.loads(qpath.read_text()) == rebuilt
        else:
            write(qpath, rebuilt)
        cfg = factors.make_env_cfg('baseline', 64, 42, play=True)
        cfg.auto_reset = False
        cfg.commands['standup'].ground_start_probability = 0.
        cfg.commands['standup'].zero_velocity_commands = True
        cfg.commands['standup'].height_range = (.30, .30)
        cfg.commands['standup'].resampling_time_range = (1e6, 1e6)
        cfg.terminations['failure'] = TerminationTermCfg(func=numerical_probe)
        wrapped = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device), clip_actions=None)
        env = wrapped.unwrapped
        for pose in rebuilt['cases']:
            mapping, policy = initialize(env, pose, pose['target_height_m'])
            gate = initial_gate(env, pose, mapping, policy)
            run['initialization'].append(gate)
            write(args.output / 'initialization.json', run['initialization'])
            print('HIGH_START_GATE ' + json.dumps(dict(height=pose['target_height_m'],
                passed=gate['passed'], passed_counts=gate['passed_counts'],
                wheel_ratio_range=[min(gate['total_wheel_weight_ratio']),max(gate['total_wheel_weight_ratio'])],
                max_pin_error_m=max(gate['max_pin_error_m']))), flush=True)
            if not gate['passed']:
                raise RuntimeError('Initialization gate failed; no rollout or threshold relaxation')
        if not args.gate_only:
            raise RuntimeError('Replay stage will only be enabled after initial gates pass')
        run['exit_code'] = 0
    except Exception:
        run['exit_code'] = 1
        run['exception'] = traceback.format_exc()
        raise
    finally:
        if wrapped is not None:
            wrapped.close()
        run['ended'] = now()
        run['frozen_sources_unchanged'] = all(digest(ROOT / path) == value for path, value in run['source_hashes'].items())
        write(args.output / 'run.json', run)


if __name__ == '__main__':
    main()
