"""Verify shared loop/limit integration and compare actual contact-enabled GPU environments."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from ct_mjlab.envs.chuanliantui import model as robot_model
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv, make_env_cfg
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_task_config
from ct_mjlab.fudan_ppo_walk.measurement import gpu_row


PAIRS = [('left_rear_pin1', 'left_front_pin1'), ('left_rear_pin2', 'left_front_pin2'),
         ('right_rear_pin1', 'right_front_pin1'), ('right_rear_pin2', 'right_front_pin2'),
         ('left_spring_tip', 'left_gas_spring_lower'), ('right_spring_tip', 'right_gas_spring_lower')]
RANGES = {'lf1':[-.97, .14], 'rf1':[-.16, 1.01]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def canonical(node):
    return node.tag, sorted(node.attrib.items()), (node.text or '').strip(), [canonical(n) for n in node]


def check_parameters(model, prefix=''):
    assert model.neq == 6
    assert np.array_equal(model.eq_solref, np.tile([.0004, 1.], (6, 1)))
    assert np.array_equal(model.eq_solimp, np.tile([.999, .9999, .001, .5, 2.], (6, 1)))
    for name, limits in RANGES.items():
        joint = model.joint(prefix+name); assert joint.limited
        assert np.array_equal(joint.range, limits)
        assert np.array_equal(model.jnt_solref[joint.id], [.0004, 1.])
        assert np.array_equal(model.jnt_solimp[joint.id], [.999, .9999, .001, .5, 2.])


def static(output):
    before = ET.parse(output/'before-source.xml').getroot()
    current = ET.parse(robot_model.MODEL_XML).getroot()
    reverted = ET.fromstring(ET.tostring(current))
    for key in ('solreflimit', 'solimplimit'): reverted.find('default/joint').attrib.pop(key)
    for a, b in zip(before.find('equality'), reverted.find('equality')):
        for key in ('solref', 'solimp'): b.set(key, a.get(key))
    assert canonical(before) == canonical(reverted), 'Unexpected shared XML changes'
    old = mujoco.MjModel.from_xml_path(str(output/'before-model.xml'))
    new = robot_model.load_model(); check_parameters(new); robot_model.validate_model_contract(new)
    fields = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos', 'body_quat',
              'jnt_axis', 'jnt_pos', 'jnt_range', 'jnt_limited', 'eq_data', 'qpos0', 'site_pos',
              'dof_armature', 'dof_damping', 'dof_frictionloss', 'geom_solref', 'geom_solimp',
              'actuator_gainprm', 'actuator_ctrlrange', 'actuator_forcerange')
    assert all(np.array_equal(getattr(old, f), getattr(new, f)) for f in fields)
    source = Path('artifacts/real2sim/h7-joint-limits-20261104-01/calibration-final/poses-through-endpoints.jsonl')
    count = 0; max_excess = 0.
    with source.open() as handle:
        for line in handle:
            row = json.loads(line)
            for name, q in row['joints_rad'].items():
                joint = new.joint(name)
                if joint.limited: max_excess = max(max_excess, joint.range[0]-q, q-joint.range[1])
            count += 1
    assert count >= 20_000 and max_excess < 1e-8
    # 正式原生sim2sim仍使用共享源；只编译和核参数，不作CPU物理验收。
    sys.path.insert(0, str(Path('sim2sim').resolve()))
    import mj_sim2sim_huananhu_multimode as native
    model = native.build_model(); check_parameters(model)
    cfg = make_env_cfg(play=False)
    assert cfg.sim.mujoco.timestep == .002 and cfg.decimation == 5
    assert not cfg.sim.mujoco.disableflags
    import ct_mjlab.envs
    from ct_mjlab.utils.task_registry import task_registry
    entries = {}
    for task in ('supermimic', 'smp'):
        config, _ = task_registry.get_cfgs(task)
        entity = config.scene.entities['robot']; compiled = entity.spec_fn().compile()
        check_parameters(compiled)
        entries[task] = dict(timestep_s=config.sim.mujoco.timestep, decimation=config.decimation)
    result = dict(passed=True, calibrated_pose_count=count, calibrated_pose_max_limit_excess_rad=max_excess,
        preserved_physical_arrays=list(fields), source_model_sha256=digest(robot_model.MODEL_XML),
        measured_pose_sha256=digest(source), training_configs=entries,
        fudan_physics_dt_s=.002, fudan_policy_dt_s=.01, native_physics_dt_s=model.opt.timestep,
        nominal_timeconstant_s=.0004, refsafe_enabled=True,
        effective_minimum_timeconstant_at_2ms_s=.004, new_cpu_physics_steps=0)
    write(output/'static-check.json', result)


def trajectory_metrics(model, q, v):
    data = mujoco.MjData(model); peak = 0.; peaks = np.zeros(q.shape[1]); areas = []
    pairs = [(model.site('robot/'+a).id, model.site('robot/'+b).id) for a, b in PAIRS]
    for states in q:
        for i, state in enumerate(states):
            data.qpos[:] = state; mujoco.mj_kinematics(model, data)
            gap = max(np.linalg.norm(data.site_xpos[a]-data.site_xpos[b]) for a, b in pairs)
            peak = max(peak, gap); peaks[i] = max(peaks[i], gap)
    for prefix in ('lf', 'rf'):
        joint = model.joint('robot/'+prefix+'02'); position = model.body('robot/'+prefix+'02').pos
        pin = model.site('robot/'+('left' if prefix == 'lf' else 'right')+'_rear_pin1').pos
        angle = (q[:, :, joint.qposadr[0]]-model.qpos0[joint.qposadr[0]])*joint.axis[1]
        rx = np.cos(angle)*pin[0]+np.sin(angle)*pin[2]
        rz = -np.sin(angle)*pin[0]+np.cos(angle)*pin[2]
        areas.append(float(np.min(position[2]*rx-position[0]*rz)))
    joints = np.flatnonzero(model.jnt_limited); value = q[:, :, model.jnt_qposadr[joints]]
    excess = np.maximum(model.jnt_range[joints, 0]-value, value-model.jnt_range[joints, 1]).clip(0.)
    return dict(max_pin_gap_mm=float(peak*1000), per_environment_max_pin_gap_mm=(peaks*1000).tolist(),
        max_joint_limit_excess_deg=float(np.degrees(excess.max())), minimum_branch_area_m2=areas,
        original_assembly_branch=all(a > 0. for a in areas), max_abs_velocity=float(abs(v).max()))


def gpu(output):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.metrics_manager import MetricsTermCfg
    torch.set_num_threads(1); results = {}; initial = {}; randomized_fields = {}; steps = 300
    for branch, path in [('before', output/'before-model.xml'), ('after', robot_model.MODEL_XML)]:
        qlog = []; vlog = []; rows = []
        def probe(env):
            qlog.append(env.sim.data.qpos.detach().clone()); vlog.append(env.sim.data.qvel.detach().clone())
            rows.append(gpu_row(env).detach().clone())
            return env.sim.data.qpos[:, 0]*0.
        task = FudanWalkCfg(); disturbances = from_task_config(task)
        with ExitStack() as stack:
            stack.enter_context(patch.object(robot_model, 'MODEL_XML', path))
            cfg = make_env_cfg(play=False, task_cfg=task, disturbances=disturbances)
            cfg.scene.num_envs = 20; cfg.seed = 20261104
            cfg.metrics['closed_chain_integration_probe'] = MetricsTermCfg(func=probe, per_substep=True, reduce='last')
            physics = ManagerBasedRlEnv(cfg, device='cuda:0')
            env = FudanEnv(20, seed=20261104, randomized=True, physics_env=physics, cfg=task, disturbances=disturbances)
            stack.callback(env.close)
            env.reset_idx(torch.arange(20, device='cuda:0'))
            q0 = physics.sim.data.qpos.cpu().numpy().copy(); v0 = physics.sim.data.qvel.cpu().numpy().copy()
            initial[branch] = (q0, v0)
            randomized_fields[branch] = {f:getattr(physics.sim.model, f).cpu().numpy().copy()
                for f in ('body_mass', 'body_inertia', 'body_ipos', 'geom_friction', 'geom_solref')}
            qlog.clear(); vlog.clear(); rows.clear(); actions = []
            for k in range(steps):
                action = torch.zeros(20, 6, device='cuda:0')
                if k >= 50:
                    phase = torch.arange(20, device='cuda:0')*.3+2*np.pi*.5*k*.01
                    action[:, [0, 1, 3, 4]] = (.05*phase.sin())[:, None]
                    action[10:, [2, 5]] = (.05*phase[10:].sin())[:, None]
                actions.append(action.cpu().numpy().copy())
                env.step(action, auto_reset=False, resample=False)
            q = np.concatenate((q0[None], torch.stack(qlog).cpu().numpy()))
            v = np.concatenate((v0[None], torch.stack(vlog).cpu().numpy()))
            assert len(q) == steps*5+1 and np.isfinite(q).all() and np.isfinite(v).all()
            contact = torch.stack(rows).cpu().numpy(); model = physics.sim.mj_model
            if branch == 'after': check_parameters(model, 'robot/')
            assert model.opt.timestep == .002 and not (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE))
            results[branch] = trajectory_metrics(model, q, v)
            results[branch].update(contact_samples=int((abs(contact[:, :, 14:16]).sum(-1)>1.).sum()),
                max_wheel_contact_n=float(abs(contact[:, :, 14:16]).max()), physics_dt_s=.002, policy_dt_s=.01,
                effective_minimum_timeconstant_s=.004, environment_physics_steps=20*steps*5)
            np.savez_compressed(output/(branch+'-contact.npz'), time_s=np.arange(len(q))*.002,
                qpos=q, qvel=v, rows=contact, action=np.asarray(actions))
            mujoco.mj_saveModel(model, str(output/(branch+'-contact.mjb')))
        print(json.dumps(dict(branch=branch, results=results[branch])), flush=True)
    assert all(np.array_equal(a, b) for a, b in zip(initial['before'], initial['after']))
    assert all(np.array_equal(randomized_fields['before'][f], randomized_fields['after'][f]) for f in randomized_fields['before'])
    result = dict(physics_backend='mujoco_warp_gpu', environments_per_branch=20, duration_s=3.,
        identical_initial_qpos_qvel=True, identical_randomized_physical_parameters=True,
        identical_open_loop_actions=True, results=results,
        new_gpu_environment_steps=sum(row['environment_physics_steps'] for row in results.values()),
        optimizer_updates=0, policy_forwards=0,
        loop_stability_passed=bool(results['after']['max_pin_gap_mm'] < .1
            and results['after']['max_joint_limit_excess_deg'] < .1 and results['after']['original_assembly_branch']),
        limitations='Contact-enabled integration check with scripted actions; not trained locomotion acceptance')
    write(output/'gpu-contact-check.json', result)


def policy_comparison(output):
    from scripts.agent.fudan_ppo_walk import evaluate
    from ct_mjlab.fudan_ppo_walk.selected_policy import selection_record
    torch.set_num_threads(1); selection = selection_record()
    deadline = time.time()+360.
    for branch, path in [('before', output/'before-model.xml'), ('after', robot_model.MODEL_XML)]:
        args = SimpleNamespace(checkpoint=Path(selection['checkpoint']), selection=selection,
            diagnostic=True, diagnostic_height=None, batch_size=20, output=output/('policy-'+branch), deadline=deadline)
        with patch.object(robot_model, 'MODEL_XML', path): evaluate.gpu(args)
        execution = json.loads((args.output/'execution.json').read_text())
        assert execution['complete_traces'] == execution['planned'] == 20
    equal = []
    for path in sorted((output/'policy-before/fixtures').glob('*_initial.npz')):
        with np.load(path) as a, np.load(output/'policy-after/fixtures'/path.name) as b:
            assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
        equal.append(path.name)
    assert len(equal) == 20
    write(output/'policy-initial-consistency.json', dict(identical=True, samples=equal,
        checkpoint=selection, policy_training_updates=0, mode='20-case diagnostic; not full987 acceptance'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy-only', action='store_true')
    args = parser.parse_args(); assert (args.output/'before.json').exists()
    if args.policy_only:
        assert json.loads((args.output/'gpu-contact-check.json').read_text())['loop_stability_passed']
        policy_comparison(args.output)
        return
    static(args.output); gpu(args.output)
    before = json.loads((args.output/'before.json').read_text())
    firmware = Path(before['firmware_root'])
    assert all(digest(firmware/p) == value for p, value in before['firmware_sources'].items())
    assert digest('mjlab_training/src/ct_mjlab/envs/chuanliantui/gas_spring.py') == before['gas_spring_sha256']
    write(args.output/'source-check.json', dict(model_before=before['model_sha256_before'],
        model_after=digest(robot_model.MODEL_XML), unchanged_firmware_sources=len(before['firmware_sources']),
        gas_spring_unchanged=True))


if __name__ == '__main__': main()
