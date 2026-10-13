"""CPU/CUDA contact audit; no learning or changes to imported training sources.

Run with PYTHONPATH=mjlab_training/src in the wheelleg-mjlab Python environment.
The output directory must not exist. JSON preserves both asynchronous .data
comparisons (legacy method) and synchronized raw sensordata comparisons.
"""
import argparse
import contextlib
import copy
from datetime import datetime
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shlex
import sys
import traceback
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True

import mujoco
import mujoco_warp
import numpy as np
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from ct_mjlab.experiments import recovery_factors as factors

ROOT = Path(__file__).resolve().parents[2]
SENSORS = ('undesired_contact', 'nonwheel_ground_probe', 'base_contact',
           'wheel_contact', 'wheel_ground_probe')
STATE_FIELDS = ('qpos', 'qvel', 'ctrl', 'qfrc_applied', 'xfrc_applied', 'qacc_warmstart', 'time')


def state_buffers(env):
    """Capture command/controller/randomization buffers, excluding backreferences."""
    roots = {'action': env.action_manager.get_term('pd'),
             'command': env.command_manager.get_term('standup'),
             'randomization': env._huananhu_randomization}
    for name in ('_huananhu_observations', '_huananhu_rewards'):
        if hasattr(env, name):
            roots[name] = getattr(env, name)
    found = {}
    seen = set()

    def visit(obj, path, depth=0):
        if isinstance(obj, torch.Tensor):
            found[path] = obj
            return
        if id(obj) in seen or depth > 4:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            children = obj.items()
        elif isinstance(obj, (list, tuple)):
            children = enumerate(obj)
        elif hasattr(obj, '__dict__'):
            children = ((k, v) for k, v in vars(obj).items()
                        if k not in ('env', '_env', 'cfg', '_cfg', '_entity', '_robot',
                                     'scene', '_scene', '_sim', 'sim'))
        else:
            return
        for key, value in children:
            visit(value, (*path, key), depth + 1)

    for name, obj in roots.items():
        visit(obj, (name,))
    for name, value in vars(env).items():
        if isinstance(value, torch.Tensor):
            found[('env', name)] = value
    return found


def shared_inputs(env, args, pose):
    """CPU records exact initial inputs; CUDA restores them, not just the seed."""
    path = args.paired_inputs / f'{pose}.json'

    def pack(value):
        array = value.detach().cpu().numpy()
        return dict(dtype=array.dtype.str, shape=list(array.shape), bytes_hex=array.tobytes().hex())

    def unpack(value):
        return torch.from_numpy(np.frombuffer(bytes.fromhex(value['bytes_hex']),
            dtype=np.dtype(value['dtype'])).reshape(value['shape']).copy())
    buffers = state_buffers(env)
    if args.device == 'cpu' and not args.replay_from:
        args.paired_inputs.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        generator = torch.Generator(device='cpu').manual_seed(args.seed)
        actions = torch.rand((args.steps, args.num_envs, 6), generator=generator) * 2 - 1
        if args.actions == 'zero':
            actions.zero_()
        elif args.actions == 'mixed':
            actions[1::2].zero_()
        bundle = dict(parameters=dict(num_envs=args.num_envs, steps=args.steps,
                                     seed=args.seed, actions=args.actions, pose=pose),
            data={f: getattr(env.sim.data, f).detach().cpu().clone() for f in STATE_FIELDS},
            model={f: getattr(env.sim.model, f).detach().cpu().clone()
                   for f in sorted(env.sim.expanded_fields)},
            buffers={p: v.detach().cpu().clone() for p, v in buffers.items()},
            actions=actions)
        write(path, dict(parameters=bundle['parameters'],
            data={f: pack(v) for f, v in bundle['data'].items()},
            model={f: pack(v) for f, v in bundle['model'].items()},
            buffers=[dict(path=list(p), tensor=pack(v)) for p, v in bundle['buffers'].items()],
            actions=pack(bundle['actions'])))
    else:
        packed = json.loads(path.read_text())
        bundle = dict(parameters=packed['parameters'],
            data={f: unpack(v) for f, v in packed['data'].items()},
            model={f: unpack(v) for f, v in packed['model'].items()},
            buffers={tuple(v['path']): unpack(v['tensor']) for v in packed['buffers']},
            actions=unpack(packed['actions']))
        expected = dict(num_envs=args.num_envs, steps=args.steps, seed=args.seed,
                        actions=args.actions, pose=pose)
        assert bundle['parameters'] == expected, 'Paired input parameters differ'
        assert set(buffers) == set(bundle['buffers']), 'Initial buffer structure differs'
        for f, value in bundle['model'].items():
            getattr(env.sim.model, f).copy_(value.to(env.device))
        from mjlab.managers.event_manager import RecomputeLevel
        env.sim.recompute_constants(RecomputeLevel.set_const)
        if args.replay_from:
            for f, value in bundle['model'].items():
                getattr(env.sim.model, f).copy_(value.to(env.device))
        for f, value in bundle['data'].items():
            getattr(env.sim.data, f).copy_(value.to(env.device))
        for p, value in bundle['buffers'].items():
            buffers[p].copy_(value.to(buffers[p].device))
        env.sim.forward()
        env.sim.data.qacc_warmstart.copy_(bundle['data']['qacc_warmstart'].to(env.device))
    errors = {'data': {}, 'model': {}, 'buffers': {}}
    for group, actual in [('data', {f: getattr(env.sim.data, f) for f in STATE_FIELDS}),
                          ('model', {f: getattr(env.sim.model, f) for f in bundle['model']}),
                          ('buffers', buffers)]:
        for key, value in actual.items():
            reference = bundle[group][key]
            host = value.detach().cpu()
            if host.numpy().tobytes() == reference.numpy().tobytes():
                errors[group][str(key)] = 0.
            else:
                delta = (host.to(torch.float64) - reference.to(torch.float64)).abs()
                errors[group][str(key)] = float(delta.max()) if delta.numel() else 0.
    # Recomputed derived model constants may round differently; randomized inputs
    # and physical initial data must match exactly.
    assert not any(errors['data'].values()), errors['data']
    assert not any(errors['buffers'].values()), 'Command/controller inputs differ'
    return bundle['actions'].to(env.device), dict(path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), initial_max_errors=errors)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path, value):
    # Keep large per-frame arrays compact; run metadata remains readable.
    path.write_text(json.dumps(value, ensure_ascii=False,
                               indent=2 if path.name == 'run.json' else None,
                               allow_nan=False) + '\n')


def raw(env, name, field='force'):
    sensor = env.scene[name]
    model = env.sim.mj_model
    chunks = []
    for body in sensor.primary_names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f'{name}_{body}_{field}')
        adr, dim = int(model.sensor_adr[sid]), int(model.sensor_dim[sid])
        chunks.append(env.sim.data.sensordata[:, adr:adr + dim].clone())
    return torch.stack(chunks, 1) if field == 'force' else torch.cat(chunks, 1)


def stats(a, g, threshold=3.):
    vector_error = float(np.abs(a - g).max())
    a, g = np.linalg.norm(a, axis=-1), np.linalg.norm(g, axis=-1)
    on, fired = g > threshold, a > threshold
    return dict(ground=on.sum(0).tolist(), missed=(on & ~fired).sum(0).tolist(),
                extra=(~on & fired).sum(0).tolist(),
                max_abs_norm_difference_n=float(np.abs(a - g).max()),
                max_abs_vector_difference_n=vector_error)


def cpu_state(env, eid, model):
    d = mujoco.MjData(model)
    for field in ('qpos', 'qvel', 'ctrl', 'qfrc_applied', 'xfrc_applied', 'qacc_warmstart'):
        if hasattr(env.sim.data, field):
            getattr(d, field)[:] = getattr(env.sim.data, field)[eid].cpu().numpy()
    mujoco.mj_forward(model, d)
    return d


def cpu_contacts(model, data):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
    terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'terrain')
    ground, others, pairs = np.zeros((model.nbody, 3)), np.zeros((model.nbody, 3)), []
    for i in range(data.ncon):
        c = data.contact[i]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        world = c.frame.reshape(3, 3).T @ force[:3]
        g1, g2 = map(int, c.geom)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        # Sensor convention primary -> secondary: negative of force ON primary.
        target = ground if terrain in (g1, g2) else others
        target[b1] += world
        target[b2] -= world
        pairs.append(dict(geom=[g1, g2], body=[names[b1], names[b2]],
                          terrain=terrain in (g1, g2), efc_address=int(c.efc_address),
                          distance_m=float(c.dist), position_w=c.pos.tolist(),
                          local_force=force.tolist(), world_force_on_geom2=world.tolist()))
    return ground, others, pairs


def cpu_sensor(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    a, n = int(model.sensor_adr[sid]), int(model.sensor_dim[sid])
    return data.sensordata[a:a+n].tolist()


def warp_contacts(env, eid):
    """Use native Warp forces and active constraint addresses, avoiding host remapping."""
    m, d = env.sim.mj_model, env.sim.wp_data
    from mujoco_warp._src.types import ContactType
    count = int(d.nacon.numpy()[0])
    types = d.contact.type.numpy()[:count]
    worlds = d.contact.worldid.numpy()[:count]
    ids = np.flatnonzero((worlds == eid) & ((types & int(ContactType.CONSTRAINT)) != 0))
    ground, other, pairs = np.zeros((m.nbody, 3)), np.zeros((m.nbody, 3)), []
    if len(ids):
        # The helper's wp.launch uses Warp's current default device, which may
        # be CUDA even for CPU arrays. Force its launch scope to CPU as well.
        with wp.ScopedDevice(env.device):
            forces = wp.zeros(len(ids), dtype=wp.spatial_vector, device=env.device)
            mujoco_warp.contact_force(env.sim.wp_model, d, wp.array(ids, dtype=int, device=env.device), True, forces)
        force_values = forces.numpy()
        geoms = d.contact.geom.numpy()
        distances = d.contact.dist.numpy()
        terrain = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'terrain')
        for cid, force in zip(ids, force_values):
            g1, g2 = map(int, geoms[cid])
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            target = ground if terrain in (g1, g2) else other
            target[b1] += force[:3]
            target[b2] -= force[:3]
            pairs.append(dict(geom=[g1, g2], body=[m.body(b1).name, m.body(b2).name],
                terrain=terrain in (g1, g2), distance_m=float(distances[cid]),
                world_force_on_geom2=force[:3].tolist()))
    return ground, other, pairs


def evidence(env, eid, step, kind, cached):
    m = env.sim.mj_model
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot/base_link')
    state = {field: getattr(env.sim.data, field)[eid].cpu().numpy().tolist()
             for field in ('qpos', 'qvel', 'ctrl', 'qfrc_applied', 'xfrc_applied', 'qacc_warmstart')}
    result = dict(step=step, env=eid, kind=kind, state=state,
                  warp_raw={s: raw(env, s)[eid].tolist() for s in SENSORS},
                  warp_found={s: raw(env, s, 'found')[eid].tolist() for s in SENSORS},
                  cached={s: cached[s][eid].tolist() for s in SENSORS},
                  cache_valid={s: env.scene[s]._cache_valid for s in SENSORS})
    ground, other, pairs = warp_contacts(env, eid)
    result['warp_contact_buffer'] = dict(manual_ground_base_force=ground[base].tolist(),
        manual_other_base_force=other[base].tolist(), contacts=pairs,
        overflow=env.sim.wp_data.overflow.numpy().tolist())
    # Include every narrowphase contact, not only active constraint contacts,
    # so absence of a constraint is not mislabeled as absence of geometry.
    contact = env.sim.wp_data.contact
    count = int(env.sim.wp_data.nacon.numpy()[0])
    worldids, types = contact.worldid.numpy()[:count], contact.type.numpy()[:count]
    geoms, distances = contact.geom.numpy()[:count], contact.dist.numpy()[:count]
    result['warp_all_contact_entries'] = []
    for cid in np.flatnonzero(worldids == eid):
        g1, g2 = map(int, geoms[cid])
        result['warp_all_contact_entries'].append(dict(index=int(cid), type=int(types[cid]),
            geom=[g1, g2], distance_m=float(distances[cid]),
            body=[m.body(int(m.geom_bodyid[g])).name if 0 <= g < m.ngeom else None
                  for g in (g1, g2)]))
    # Required nominal env.sim.mj_model comparison, plus randomized per-world model.
    for label, model in [('nominal_mj_model', m), ('matched_world_model', copy.copy(m))]:
        fields = []
        field_values = {}
        if label == 'matched_world_model':
            for field in sorted(env.sim.expanded_fields):
                if hasattr(env.sim.model, field) and hasattr(model, field):
                    getattr(model, field)[:] = getattr(env.sim.model, field)[eid].cpu().numpy()
                    fields.append(field)
                    field_values[field] = getattr(model, field).tolist()
        d = cpu_state(env, eid, model)
        ground, other, pairs = cpu_contacts(model, d)
        result[label] = dict(copied_model_fields=fields, model_field_values=field_values,
            undesired_base_force=cpu_sensor(model, d, 'undesired_contact_base_link_force'),
            ground_probe_base_force=cpu_sensor(model, d, 'nonwheel_ground_probe_base_link_force'),
            manual_ground_base_force=ground[base].tolist(),
            manual_other_base_force=other[base].tolist(), contacts=pairs)
    before = env.sim.data.sensordata.clone()
    env.sim.sense()
    result['sense_max_sensordata_change'] = float((before - env.sim.data.sensordata).abs().max())
    # Refresh cache only in diagnostic; same sensordata, no simulation step/history roll.
    result['refresh'] = {}
    for s in SENSORS:
        sensor = env.scene[s]
        saved_data, saved_valid = sensor._cached_data, sensor._cache_valid
        sensor._invalidate_cache()
        result['refresh'][s] = sensor.data.force[eid].tolist()
        sensor._cached_data, sensor._cache_valid = saved_data, saved_valid
    return result


def replay_pose(args, pose):
    """Recompute the same saved qpos/qvel/ctrl/model cases on each Warp device."""
    cfg = factors.make_env_cfg('keep_contact', args.num_envs, args.seed)
    cfg.commands['standup'].ground_start_probability = float(pose == 'ground')
    for sensor in cfg.scene.sensors:
        if sensor.name in ('nonwheel_ground_probe', 'wheel_ground_probe'):
            sensor.history_length = 8
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env.reset()
        _, pairing = shared_inputs(env, args, pose)
        source = args.replay_from / f'{pose}.json'
        cases = json.loads(source.read_text())['examples']
        outputs = []
        for case in cases:
            # Every case starts with identical full-world data and cleared
            # constraint/solver workspace; cases cannot carry warm caches over.
            env.sim.reset()
            shared_inputs(env, args, pose)
            eid = case['env']
            for field, values in case['matched_world_model']['model_field_values'].items():
                getattr(env.sim.model, field)[eid].copy_(torch.tensor(
                    values, dtype=getattr(env.sim.model, field).dtype, device=env.device))
            from mjlab.managers.event_manager import RecomputeLevel
            env.sim.recompute_constants(RecomputeLevel.set_const)
            # Isolate forward/sensor behavior from CPU/CUDA rounding in set_const:
            # restore the saved derived constants as well as randomized inputs.
            for field, values in case['matched_world_model']['model_field_values'].items():
                getattr(env.sim.model, field)[eid].copy_(torch.tensor(
                    values, dtype=getattr(env.sim.model, field).dtype, device=env.device))
            model_errors = {}
            for field, values in case['matched_world_model']['model_field_values'].items():
                reference = torch.tensor(values, dtype=getattr(env.sim.model, field).dtype,
                                         device=env.device)
                difference = (getattr(env.sim.model, field)[eid].to(torch.float64)
                              - reference.to(torch.float64)).abs()
                model_errors[field] = float(difference.max()) if difference.numel() else 0.
            assert not any(model_errors.values()), 'Replay model inputs differ'
            for field, values in case['state'].items():
                getattr(env.sim.data, field)[eid].copy_(torch.tensor(
                    values, dtype=getattr(env.sim.data, field).dtype, device=env.device))
            env.sim.forward()
            cached = {s: raw(env, s) for s in SENSORS}
            item = evidence(env, eid, case['step'], case['kind'], cached)
            item['input_sha256'] = hashlib.sha256(json.dumps(
                dict(state=case['state'], model=case['matched_world_model']['model_field_values']),
                sort_keys=True).encode()).hexdigest()
            item['input_model_max_error'] = model_errors
            item['input_state_max_error'] = {field: float((getattr(env.sim.data, field)[eid]
                - torch.tensor(values, dtype=getattr(env.sim.data, field).dtype,
                               device=env.device)).abs().max())
                for field, values in case['state'].items() if field != 'qacc_warmstart'}
            # forward updates warmstart; its input, before forward, is saved above.
            outputs.append(item)
        result = dict(device=args.device, pairing=pairing, source=str(source), cases=outputs)
        write(args.output / f'{pose}-replay.json', result)
        return dict(cases=len(outputs), source=str(source))
    finally:
        env.close()


def run_pose(args, pose):
    torch.manual_seed(args.seed)
    cfg = factors.make_env_cfg('keep_contact', args.num_envs, args.seed)
    cfg.commands['standup'].ground_start_probability = float(pose == 'ground')
    # Only the audit probes gain read-only history to compare equal windows.
    for sensor in cfg.scene.sensors:
        if sensor.name in ('nonwheel_ground_probe', 'wheel_ground_probe'):
            sensor.history_length = 8
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env.reset()
        paired_actions, pairing = (shared_inputs(env, args, pose) if args.paired_inputs
                                  else (None, None))
        names = env.scene['undesired_contact'].primary_names
        assert names == env.scene['nonwheel_ground_probe'].primary_names
        base_index = names.index('base_link')
        streams = {phase: {s: [] for s in SENSORS} for phase in ('legacy_cached', 'post_forward_raw', 'reward_raw')}
        histories = {s: [] for s in ('undesired_contact', 'nonwheel_ground_probe')}
        substep_ground, history_windows = [], []
        examples, trace, cpu_rows, trajectory = [], [], [], []
        original_compute = env.reward_manager.compute
        reward_snapshot = {}

        def compute(*a, **kw):
            reward_snapshot['raw'] = {s: raw(env, s) for s in SENSORS}
            reward_snapshot['history'] = {s: env.scene[s]._history_state['force'].clone()
                                          for s in histories}
            return original_compute(*a, **kw)

        env.reward_manager.compute = compute
        original_step = env.sim.step

        def sim_step():
            original_step()
            substep_ground.append(raw(env, 'nonwheel_ground_probe').norm(dim=-1))

        env.sim.step = sim_step
        cpu_models = []
        for eid in range(args.num_envs):
            model = copy.copy(env.sim.mj_model)
            for field in sorted(env.sim.expanded_fields):
                if hasattr(env.sim.model, field) and hasattr(model, field):
                    getattr(model, field)[:] = getattr(env.sim.model, field)[eid].cpu().numpy()
            cpu_models.append(model)
        body_ids = [mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'robot/' + n) for n in names]
        for step in range(args.steps):
            substep_ground.clear()
            random_action = args.actions == 'random' or (args.actions == 'mixed' and step % 2 == 0)
            action = (paired_actions[step] if paired_actions is not None else
                      (torch.rand(args.num_envs, 6, device=args.device) * 2 - 1)
                      if random_action else torch.zeros(args.num_envs, 6, device=args.device))
            observations, rewards, _, _, _ = env.step(action)
            assert torch.isfinite(rewards).all(), f'Nonfinite reward at {step}'
            assert all(torch.isfinite(v).all() for v in observations.values()), f'Nonfinite observation at {step}'
            trajectory.append(dict(qpos=env.sim.data.qpos.cpu().tolist(),
                                   qvel=env.sim.data.qvel.cpu().tolist()))
            norms = torch.stack(substep_ground, -1)
            history_windows.extend(norms.cpu().numpy().tolist())
            cached = {s: env.scene[s].data.force.clone() for s in SENSORS}
            current = {s: raw(env, s) for s in SENSORS}
            for phase, values in [('legacy_cached', cached), ('post_forward_raw', current), ('reward_raw', reward_snapshot['raw'])]:
                for s in SENSORS:
                    streams[phase][s].extend(values[s].cpu().numpy().tolist())
            for s in histories:
                histories[s].extend(reward_snapshot['history'][s].norm(dim=-1).amax(-1).cpu().numpy().tolist())
            a = cached['undesired_contact'].norm(dim=-1)
            g = cached['nonwheel_ground_probe'].norm(dim=-1)
            r = current['undesired_contact'].norm(dim=-1)
            rg = current['nonwheel_ground_probe'].norm(dim=-1)
            masks = dict(cached_miss=(g[:, base_index] > 3.) & (a[:, base_index] <= 3.),
                         cached_extra=(g[:, base_index] <= 3.) & (a[:, base_index] > 3.),
                         raw_miss=(rg[:, base_index] > 3.) & (r[:, base_index] <= 3.),
                         raw_extra=(rg[:, base_index] <= 3.) & (r[:, base_index] > 3.))
            for kind, mask in masks.items():
                if mask.any() and sum(x['kind'] == kind for x in examples) < 2:
                    eid = int(mask.nonzero()[0])
                    item = evidence(env, eid, step, kind, cached)
                    item['reward_raw'] = {s: reward_snapshot['raw'][s][eid].tolist() for s in SENSORS}
                    examples.append(item)
            # Independent contactForce truth at the exact post-forward qpos/qvel,
            # with copied per-world randomized model parameters and applied forces.
            frame = []
            for eid, model in enumerate(cpu_models):
                d = cpu_state(env, eid, model)
                ground, other, pairs = cpu_contacts(model, d)
                frame.append(dict(ground=ground[body_ids].tolist(), other=other[body_ids].tolist(),
                    u=[cpu_sensor(model, d, f'undesired_contact_{n}_force') for n in names],
                    g=[cpu_sensor(model, d, f'nonwheel_ground_probe_{n}_force') for n in names]))
                cpu_norm = np.linalg.norm(ground[body_ids], axis=-1)
                warp_norm = current['undesired_contact'][eid].norm(dim=-1).cpu().numpy()
                misses = (cpu_norm > 3.) & (warp_norm <= 3.)
                kind = 'cpu_positive_warp_base_zero' if misses[base_index] else 'cpu_positive_warp_other_zero'
                if misses.any() and sum(x['kind'] == kind for x in examples) < 2:
                    item = evidence(env, eid, step, kind, cached)
                    item['disagreeing_bodies'] = [n for n, yes in zip(names, misses) if yes]
                    examples.append(item)
            cpu_rows.extend(frame)
            trace.append(dict(step=step, action='uniform[-1,1]' if random_action else 'zero',
                              resets=env.reset_buf.tolist()))
            if (step + 1) % 25 == 0:
                print(f'AUDIT {pose} {step+1}/{args.steps}', flush=True)
        summary = dict(body_names=names, env_steps=args.num_envs * args.steps, phases={},
                       pairing=pairing, finite_observations_and_rewards=True,
                       reset_count=sum(sum(row['resets']) for row in trace))
        m = env.sim.mj_model
        summary['sensor_definitions'] = {}
        for s in SENSORS:
            for body in env.scene[s].primary_names:
                name = f'{s}_{body}_force'
                sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
                summary['sensor_definitions'][name] = dict(objtype=int(m.sensor_objtype[sid]),
                    objid=int(m.sensor_objid[sid]), reftype=int(m.sensor_reftype[sid]),
                    refid=int(m.sensor_refid[sid]), intprm=m.sensor_intprm[sid].tolist())
        for phase, values in streams.items():
            values = {k: np.asarray(v) for k, v in values.items()}
            table = stats(values['undesired_contact'], values['nonwheel_ground_probe'])
            base = stats(values['base_contact'], values['nonwheel_ground_probe'][:, base_index:base_index+1], 1.)
            wheel = stats(values['wheel_contact'], values['wheel_ground_probe'])
            summary['phases'][phase] = dict(undesired=table, base=base, wheel=wheel)
        h = {k: np.asarray(v) for k, v in histories.items()}
        ha, hg = h['undesired_contact'] > 3., h['nonwheel_ground_probe'] > 3.
        summary['reward_history'] = dict(ground=hg.sum(0).tolist(), missed=(hg & ~ha).sum(0).tolist(),
            extra=(~hg & ha).sum(0).tolist(), env_ground=int(hg.any(-1).sum()),
            env_missed=int((hg.any(-1) & ~ha.any(-1)).sum()),
            env_extra=int((~hg.any(-1) & ha.any(-1)).sum()))
        all10 = np.asarray(history_windows).max(-1) > 3.
        summary['history_8_vs_decimation_10'] = dict(ground_10=all10.sum(0).tolist(),
            omitted_first2_only=(all10 & ~hg).sum(0).tolist(),
            env_ground_10=int(all10.any(-1).sum()),
            env_omitted_first2_only=int((all10.any(-1) & ~hg.any(-1)).sum()))
        cpu = {k: np.asarray([v[k] for v in cpu_rows]) for k in ('ground', 'other', 'u', 'g')}
        summary['cpu_manual_truth'] = dict(sensor_vs_manual_ground=stats(cpu['u'], cpu['ground']),
            probe_vs_manual_ground=stats(cpu['g'], cpu['ground']),
            max_ground_probe_vector_error_n=float(np.abs(cpu['g']-cpu['ground']).max()),
            max_all_contact_vector_error_n=float(np.abs(cpu['u']-cpu['ground']-cpu['other']).max()),
            nonwheel_self_contact_env_steps=int((np.linalg.norm(cpu['other'], axis=-1).max(-1) > 3.).sum()))
        summary['raw_warp_vs_cpu_manual_ground'] = stats(np.asarray(streams['post_forward_raw']['undesired_contact']), cpu['ground'])
        summary['raw_warp_probe_vs_cpu_manual_ground'] = stats(np.asarray(streams['post_forward_raw']['nonwheel_ground_probe']), cpu['ground'])
        write(args.output / f'{pose}.json', dict(summary=summary, examples=examples, trace=trace,
              streams=streams, reward_history_norms=histories, substep_ground_norms=history_windows,
              cpu_contact_force=cpu_rows, trajectory=trajectory))
        print('SUMMARY ' + json.dumps(summary), flush=True)
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--paired-inputs', type=Path,
                        help='Shared exact initial buffers/model/data and action tensors; CPU writes, CUDA reads')
    parser.add_argument('--replay-from', type=Path,
                        help='Replay saved examples at identical states, without stepping a new trajectory')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--actions', choices=('mixed', 'zero', 'random'), default='mixed')
    parser.add_argument('--poses', choices=('ground', 'normal'), nargs='+', default=['ground', 'normal'])
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.threads) < 1:
        parser.error('num-envs, steps and threads must be positive')
    if args.device != 'cpu' and args.num_envs > 256:
        parser.error('Local CUDA runs are limited to at most 256 environments')
    if args.device != 'cpu' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; do not change environment dependencies')
    if args.replay_from and not args.paired_inputs:
        parser.error('--replay-from requires --paired-inputs')
    allowed = ROOT / 'artifacts/contact-sensor-audit-20261012'
    if not args.output.resolve().is_relative_to(allowed.resolve()):
        parser.error('output must be a NEW directory under artifacts/contact-sensor-audit-20261012')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    metadata = dict(started=now(), command=shlex.join([sys.executable, *sys.argv]),
        cwd=str(Path.cwd()), argv=sys.argv, python=sys.version, platform=platform.platform(),
        versions={p: importlib.metadata.version(p) for p in ('mjlab', 'mujoco', 'mujoco-warp', 'torch', 'warp-lang')},
        torch_cuda_available=torch.cuda.is_available(), device=args.device,
        evidence_scope='仅 CPU 证据' if args.device == 'cpu' else '本机 GPU 证据',
        parameters=vars(args) | {'output': str(args.output),
                                'paired_inputs': str(args.paired_inputs) if args.paired_inputs else None,
                                'replay_from': str(args.replay_from) if args.replay_from else None}, learning=False,
        env_vars={k: os.environ.get(k) for k in ('PYTHONPATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        source_sha256={}, installed_sources={}, exit_code=None,
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for folder in ('mjlab_training/src/ct_mjlab/experiments', 'mjlab_training/src/ct_mjlab/reference_training'):
        for path in (ROOT / folder).rglob('*.py'):
            metadata['source_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    import mjlab.sensor.contact_sensor as contact_module
    import mjlab.sensor.sensor as sensor_module
    import mjlab.envs.manager_based_rl_env as env_module
    import mujoco_warp._src.sensor as warp_module
    for module in (contact_module, sensor_module, env_module, warp_module):
        path = Path(module.__file__)
        metadata['installed_sources'][module.__name__] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source=path.read_text())
    write(args.output / 'run.json', metadata)
    console = io.StringIO()

    class Tee:
        def isatty(self):
            return False

        def write(self, value):
            sys.__stdout__.write(value)
            console.write(value)

        def flush(self):
            sys.__stdout__.flush()

    try:
        with contextlib.redirect_stdout(Tee()):
            runner = replay_pose if args.replay_from else run_pose
            metadata['summary'] = {pose: runner(args, pose) for pose in dict.fromkeys(args.poses)}
        metadata['exit_code'] = 0
    except Exception:
        metadata['exit_code'] = 1
        metadata['exception'] = traceback.format_exc()
        raise
    finally:
        metadata['ended'] = now()
        metadata['stdout'] = console.getvalue()
        metadata['source_unchanged'] = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in metadata['source_sha256'].items())
        write(args.output / 'run.json', metadata)


if __name__ == '__main__':
    main()
