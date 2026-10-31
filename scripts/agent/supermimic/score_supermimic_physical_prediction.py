#!/usr/bin/env python3
"""CPU-only frozen V10 recorded-action prediction, not policy or behavior scoring.

The complete 24-step predictions are sealed before their errors are summarized.
True first-terminal physical samples are retained; reset/post-terminal samples
are never counted. No actor, critic, optimizer or normalization update is used.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'mjlab_training/src'))
from ct_mjlab.supermimic.analytic_reward import WEIGHTS, differentiable_reward
from ct_mjlab.supermimic.buffer import ReplayBuffer
from ct_mjlab.supermimic.learner import ALGORITHM_CONTRACT, asset_digest
from ct_mjlab.supermimic.networks import RunningNorm, WorldModel
from ct_mjlab.supermimic.state import (MASSES, NOISE, PHYSICAL, STATE_DIM,
    PARAM_DIM, PHYS_DIM, SUBSTEPS, physical_delta)
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from ct_mjlab.envs.chuanliantui_split.runner import contracts

STARTS = (.5, 1., 1.5, 2., 2.1, 2.3, 6., 10., 14., 16., 16.1, 16.3, 20., 24., 28.)
HORIZON, ENDPOINTS, DT = 24, (1, 8, 24), .01
TERMS = tuple(WEIGHTS)
V9 = 'chuanliantui-supermimic-canonical-physics-path-derived-observation-reward-termination-v9'
V10 = 'chuanliantui-supermimic-canonical-physics-contact-kb-fixed-scale-nominal-mix-v10'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def state_digest(module):
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(name.encode()); result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode()); result.update(array.tobytes())
    return result.hexdigest()


def current_training_sources():
    source = ROOT/'mjlab_training/src/ct_mjlab'
    result = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*.py'))}
    if len(result) != 99:
        raise ValueError('This frozen V10 diagnostic requires the reviewed99-source tree')
    return result


def verified_training_sources(checkpoint, current):
    manifest = checkpoint.parent/'source_sha256.json'
    hashes = json.loads(manifest.read_text())
    if hashes != current:
        raise ValueError(f'Training/current99-source mismatch: {checkpoint}')
    for name, sha in hashes.items():
        if digest(checkpoint.parent/'snapshot/ct_mjlab'/name) != sha:
            raise ValueError(f'Training snapshot mismatch: {name}')
    return {'path': str(manifest), 'sha256': digest(manifest), 'files': hashes}


def receipt_hash(receipt, names):
    values = [receipt[name] for name in names if name in receipt]
    if not values or len(set(values)) != 1:
        raise ValueError(f'Missing/ambiguous external exit hash: {names}')
    return values[0]


def load_recording(directory, current, expected_infos):
    directory = directory.resolve()
    paths = {name: directory/name for name in
             ('record.json', 'transitions.pt', 'source-sha256.json', 'initial.npz', 'execution-exit.json')}
    hashes = {name: digest(path) for name, path in paths.items()}
    meta = json.loads(paths['record.json'].read_text())
    external = json.loads(paths['execution-exit.json'].read_text())
    if (external.get('returncode') != 0
            or receipt_hash(external, ('record.json_sha256', 'record_sha256')) != hashes['record.json']
            or receipt_hash(external, ('transitions.pt_sha256', 'transitions_sha256')) != hashes['transitions.pt']):
        raise ValueError('Recording lacks matching external actualexit0 evidence')
    if (meta.get('completed') is not True or meta.get('algorithm_contract') != V10
            or meta.get('capture_algorithm_contract') != V10
            or meta.get('controller_algorithm_contract') != V9
            or meta.get('controller_training_source_verification',{}).get('verified') is not True
            or meta.get('world_or_critic_used') is not False
            or meta.get('asset_sha256') != asset_digest()
            or meta.get('infos') != json.loads(json.dumps(expected_infos))
            or meta.get('seed') != 55 or meta.get('num_envs') != 24 or meta.get('seconds') != 30.
            or meta.get('height_m') != .26
            or meta.get('command_schedule') != dict(initial_zero_s=[0., 2.], travel_s=[2., 16.], parking_from_s=16.)
            or meta.get('policy_dt_s') != DT or meta.get('physics_dt_s') != .002
            or meta.get('source_snapshot_sha256') != hashes['source-sha256.json']
            or meta.get('transitions_sha256') != hashes['transitions.pt']
            or meta.get('initial_sha256') != hashes['initial.npz']
            or meta.get('planned_window_starts_s') != list(STARTS)
            or meta.get('planned_windows') != 360
            or meta.get('parameter_updates') != 0 or meta.get('normalization_updates') != 0
            or meta.get('exploration_std') != 0.):
        raise ValueError('Recording identity/protocol/hash mismatch')
    actor_checkpoint = Path(meta['checkpoint'])
    if not actor_checkpoint.is_absolute() or digest(actor_checkpoint) != meta['checkpoint_sha256']:
        raise ValueError('Recording actor checkpoint identity changed')
    controller = torch.load(actor_checkpoint,map_location='cpu',weights_only=False)
    if (controller.get('algorithm_contract') != meta['controller_algorithm_contract']
            or controller.get('asset_sha256') != asset_digest()
            or controller.get('infos') != expected_infos
            or controller.get('config',{}).get('task') != 'chuanliantui_walk'
            or controller.get('config',{}).get('mode') != 'train'):
        raise ValueError('Recording controller must be the declared original V9 actor checkpoint')
    del controller
    sources = json.loads(paths['source-sha256.json'].read_text())
    prefix = 'mjlab_training/src/ct_mjlab/'
    if {name[len(prefix):]: sha for name, sha in sources.items() if name.startswith(prefix)} != current:
        raise ValueError('Recording does not cover the same99 training sources')
    xml = 'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
    if sources.get(xml) != asset_digest():
        raise ValueError('Recording source manifest does not cover the unchanged XML')
    for name, sha in sources.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Source manifest must use repository-relative paths')
        if digest(ROOT/relative) != sha or digest(directory/'snapshot'/relative) != sha:
            raise ValueError(f'Recording source/snapshot mismatch: {name}')
    payload = torch.load(paths['transitions.pt'], map_location='cpu', weights_only=False)
    data, live = payload['data'], payload['live_before']
    if set(data) != set(ReplayBuffer.FIELDS):
        raise ValueError('Recording fields differ from the native ReplayBuffer schema')
    for name, dim in ReplayBuffer.FIELDS.items():
        expected = (3000, 24)+((dim,) if dim else ())
        value = data[name]
        dtype = torch.bool if name in ('done', 'failure') else torch.float32
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.shape != expected or value.dtype != dtype:
            raise ValueError(f'Invalid recorded tensor: {name}')
    if (live.dtype != torch.bool or live.shape != (3000, 24) or live.device.type != 'cpu'
            or not live[0].all() or not torch.equal(live[1:], live[:-1] & ~data['done'][:-1])):
        raise ValueError('live_before is not a monotonic first-episode mask')
    if meta['fields'] != {name: list(value.shape) for name, value in data.items()}:
        raise ValueError('Recorded tensor shapes do not match record.json')
    requested = payload['requested_vx']
    if requested.shape != (3000, 24) or requested.dtype != torch.float32:
        raise ValueError('Missing original per-tick command requests')
    expected_requests = torch.zeros_like(requested)
    for case in meta['cases']:
        expected_requests[200:1600, case['env_ids']] = case['speed']
    if not torch.equal(requested, expected_requests) or (data['failure'] & ~data['done']).any():
        raise ValueError('Command request or original failure/done labels differ')
    proof = dict(directory=str(directory), files={name: {'path': str(paths[name]), 'sha256': sha}
                 for name, sha in hashes.items()}, metadata=meta, external_exit=external,
                 actor_checkpoint_sha256=meta['checkpoint_sha256'],
                 controller_algorithm_contract=meta['controller_algorithm_contract'],
                 capture_algorithm_contract=meta['capture_algorithm_contract'],
                 controller_training_sources=meta['controller_training_source_verification'], sources=sources)
    return {'directory': directory, 'data': data, 'live': live, 'proof': proof}


def load_world(label, checkpoint, current, expected_infos):
    checkpoint = checkpoint.resolve()
    if checkpoint.suffix != '.pt':
        raise ValueError('Only an atomically published.pt checkpoint is accepted')
    sha = digest(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (saved.get('algorithm_contract') != V10 or saved.get('asset_sha256') != asset_digest()
            or saved.get('infos') != expected_infos or saved.get('interrupted', False)
            or saved.get('config', {}).get('task') != 'chuanliantui_walk'
            or saved.get('config', {}).get('mode') != 'train'
            or saved['config'].get('world_steps_per_update') != 4):
        raise ValueError('World checkpoint must be a matching complete V10 walk model')
    source_proof = verified_training_sources(checkpoint, current)
    world = WorldModel().cpu().eval().requires_grad_(False)
    world.load_state_dict(saved['world'], strict=True)
    for name, value in world.state_dict().items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite world checkpoint tensor: {name}')
    return {'label': label, 'world': world, 'proof': dict(checkpoint=str(checkpoint),
        checkpoint_sha256=sha, algorithm_contract=V10, asset_sha256=asset_digest(),
        iteration=saved['iteration'], world_updates=saved['world_updates'],
        training_sources=source_proof, world_state_sha256_before=state_digest(world))}


def make_plan(meta):
    mapping = {}
    for case in meta['cases']:
        for env in case['env_ids']:
            if env in mapping:
                raise ValueError('Case environment IDs overlap')
            mapping[env] = case['name']
    if set(mapping) != set(range(24)) or len(meta['cases']) != 6:
        raise ValueError('Expected the six original cases with four environments each')
    expected = [(speed, acceleration) for acceleration in (.5, 2.) for speed in (0., .5, -.5)]
    for i, (case, (speed, acceleration)) in enumerate(zip(meta['cases'], expected)):
        if (case['speed'] != speed or case['acceleration'] != acceleration or case.get('yaw_rate') != 0.
                or case['name'] != f'v{speed:g}-a{acceleration:g}'
                or case['env_ids'] != list(range(i*4, (i+1)*4))):
            raise ValueError('Recording case requests differ from the fixed diagnostic plan')
    result = []
    for env in range(24):
        for start in STARTS:
            phase = ('initial' if start < 2 else 'travel-switch' if start < 6 else
                     'travel-hold' if start < 16 else 'stop-switch' if start < 20 else 'parking')
            result.append(dict(window=len(result), env=env, start_s=start,
                start_tick=round(start/DT), case=mapping[env], phase=phase))
    return result


def weighted_terms(values, context):
    # Only source weights/dt are applied here; caps, masks and formulas are native.
    return torch.stack([torch.nan_to_num(values[name]*WEIGHTS[name]*context[..., 20].to(values[name].dtype),
        nan=0., posinf=0., neginf=0.) for name in TERMS], -1)


def make_windows(record):
    plan = make_plan(record['proof']['metadata'])
    start = torch.tensor([p['start_tick'] for p in plan])
    env = torch.tensor([p['env'] for p in plan])
    times = start[:, None]+torch.arange(HORIZON)
    data = {name: value[times, env[:, None]] for name, value in record['data'].items()}
    live = record['live'][times, env[:, None]]
    prefix = torch.cumprod(torch.cat((torch.ones(len(plan), 1, dtype=torch.bool),
        ~data['done'][:, :-1]), 1).long(), 1).bool()
    mask = live & prefix
    values, total = differentiable_reward(data['features'], data['action'], data['previous'],
                                         data['previous_previous'], data['context'])
    torch.testing.assert_close(total[mask], data['reward'][mask], atol=2e-7, rtol=2e-5)
    data['weighted_terms'] = weighted_terms(values, data['context'])
    data['path'] = data['path'].reshape(len(plan), HORIZON, SUBSTEPS, PHYS_DIM)
    same = (data['context'][..., 3:5] == data['context'][:, :1, 3:5]).all(-1)
    constant = (same | ~mask).all(-1)
    for i, item in enumerate(plan):
        item['valid_positions'] = int(mask[i].sum())
        item['context_group'] = 'unavailable' if not mask[i].any() else 'constant' if constant[i] else 'changed'
    return dict(plan=plan, truth=data, mask=mask, context_constant=constant)


def oracle_check(adapter, record):
    data, live = record['data'], record['live']
    selected = set()
    for window in make_plan(record['proof']['metadata']):
        for h in ENDPOINTS:
            tick, env = window['start_tick']+h-1, window['env']
            if live[tick, env]:
                selected.add((tick, env))
    terminal = (live & data['done']).nonzero()
    selected.update(tuple(item) for item in terminal.tolist())
    if not selected and live.any():
        selected.add(tuple(live.nonzero()[0].tolist()))
    if not selected:
        raise ValueError('No true physical samples available for an oracle check')
    selected = sorted(selected)
    times = torch.tensor([p[0] for p in selected]); envs = torch.tensor([p[1] for p in selected])
    get = lambda name: data[name][times, envs]
    derived = adapter.advance(get('state'), get('parameters'),
        get('path').reshape(-1, SUBSTEPS, PHYS_DIM), get('next_frame')[:, 6:9],
        get('action'), get('previous'), get('previous_previous'), get('context'), get('next_state')[:, NOISE])
    state, frame, reward, failure, features = derived
    errors = {}
    for name, actual, truth in (('features', features, get('features')), ('reward', reward, get('reward')),
                               ('pre_reset_state', state, get('next_state'))):
        torch.testing.assert_close(actual, truth, atol=2e-5, rtol=2e-5)
        errors[name] = float((actual-truth).abs().max())
    if not torch.equal(failure, get('failure')):
        raise AssertionError('True-path hard failure differs from original capture')
    valid = ~get('done')
    torch.testing.assert_close(frame[valid], get('next_frame')[valid], atol=2e-5, rtol=2e-5)
    errors['nonterminal_frame'] = float((frame[valid]-get('next_frame')[valid]).abs().max()) if valid.any() else None
    return dict(selection='All available planned H1/H8/H24 real endpoints, deduplicated; all first terminal steps also included',
        selected_samples=len(selected), selected_tick_env=[list(p) for p in selected], maximum_errors=errors,
        nonterminal_frame_samples=int(valid.sum()),
        failure_exact=True, terminal_sample_checked=bool(len(terminal)),
        terminal_physical_state_included=True, returned_terminal_frame_excluded=True)


def predict(world, windows, batch):
    truth, mask = windows['truth'], windows['mask']
    outputs = {name: [] for name in ('path', 'state', 'features', 'reward', 'failure', 'weighted_terms', 'effective_context')}
    for first in range(0, len(windows['plan']), batch):
        last = min(first+batch, len(windows['plan'])); state = truth['state'][first:last, 0].clone()
        command = truth['history'][first:last, 0, -19:-16].clone()
        previous = truth['previous'][first:last, 0]; previous_previous = truth['previous_previous'][first:last, 0]
        parameters, context = truth['parameters'][first:last, 0], truth['context'][first:last, 0]
        steps = {name: [] for name in outputs}
        for t in range(HORIZON):
            valid = mask[first:last, t, None]
            action = torch.where(valid, truth['action'][first:last, t], previous)
            parameters = torch.where(valid, truth['parameters'][first:last, t], parameters)
            context = torch.where(valid, truth['context'][first:last, t], context)
            previous = torch.where(valid, truth['previous'][first:last, t], previous)
            previous_previous = torch.where(valid, truth['previous_previous'][first:last, t], previous_previous)
            next_command = torch.where(valid, truth['next_frame'][first:last, t, 6:9], command)
            noise = torch.where(valid, truth['next_state'][first:last, t, NOISE], state[..., NOISE])
            observed = []
            def reward_observer(features, a, prev, prevprev, effective_context):
                values, total = differentiable_reward(features, a, prev, prevprev, effective_context)
                observed.append((weighted_terms(values, effective_context), effective_context.clone()))
                return values, total
            # Observe the native effective masks; do not duplicate their task-state logic.
            with patch('ct_mjlab.supermimic.state.differentiable_reward', side_effect=reward_observer):
                state, frame, reward, failure, features, path = world.step(state, parameters,
                    action, next_command, context, previous, previous_previous, noise, return_path=True)
            if len(observed) != 1:
                raise AssertionError('Expected exactly one native analytic reward call')
            terms, effective_context = observed[0]
            for name, value in zip(steps, (path, state, features, reward, failure, terms, effective_context)):
                steps[name].append(value.cpu())
            previous_previous, previous, command = previous, action, next_command
        for name in outputs:
            outputs[name].append(torch.stack(steps[name], 1))
    return {name: torch.cat(values) for name, values in outputs.items()}


def stats(predicted, actual, mask, prediction_finite=None, truth_finite=None):
    pf = torch.isfinite(predicted) if prediction_finite is None else prediction_finite
    tf = torch.isfinite(actual) if truth_finite is None else truth_finite
    finite_computation = torch.isfinite(predicted) & torch.isfinite(actual)
    valid = mask & pf & tf & finite_computation
    error = (predicted.double()-actual.double())[valid]
    return dict(planned_samples=mask.numel(), available_samples=int(mask.sum()), finite_pairs=len(error),
        nonfinite_prediction=int((mask & ~pf).sum()), nonfinite_truth=int((mask & ~tf).sum()),
        nonfinite_derived_computation=int((mask & pf & tf & ~finite_computation).sum()),
        prediction_mean=None if not len(error) else float(predicted.double()[valid].mean()),
        truth_mean=None if not len(error) else float(actual.double()[valid].mean()),
        bias=None if not len(error) else float(error.mean()),
        mae=None if not len(error) else float(error.abs().mean()),
        rmse=None if not len(error) else float(error.square().mean().sqrt()),
        p95_abs=None if not len(error) else float(torch.quantile(error.abs(), .95)),
        max_abs=None if not len(error) else float(error.abs().max()))


def support_conditions(path, parameters):
    force = -path[..., -1, (60, 63)]
    weight = parameters[..., MASSES].sum(-1)*9.81
    finite = torch.isfinite(force).all(-1) & torch.isfinite(weight) & (weight > 0.)
    each = force >= (.1*weight).clamp_min(5.)[..., None]
    total = force.sum(-1)
    conditions = dict(left_min=each[..., 0], right_min=each[..., 1],
        total_lower=total >= .8*weight, total_upper=total <= 1.2*weight,
        total_band=(total >= .8*weight) & (total <= 1.2*weight),
        training_loaded=each.all(-1) & (total >= .8*weight),
        formal_support=each.all(-1) & (total >= .8*weight) & (total <= 1.2*weight))
    return force, weight, conditions, finite


def confusion(predicted, actual, mask, finite):
    valid = mask & finite
    return dict(planned_samples=mask.numel(), available_samples=int(mask.sum()), valid_pairs=int(valid.sum()),
        nonfinite_inputs=int((mask & ~finite).sum()),
        tp=int((valid & predicted & actual).sum()), fp=int((valid & predicted & ~actual).sum()),
        tn=int((valid & ~predicted & ~actual).sum()), fn=int((valid & ~predicted & actual).sum()))


def metric_arrays(pred, truth):
    pp, tp = pred['path'][..., -1, :], truth['path'][..., -1, :]
    rotation = physical_delta(tp, pp)[..., 3:6]
    pforce, weight, pc, pfinite = support_conditions(pred['path'], truth['parameters'])
    tforce, _, tc, tfinite = support_conditions(truth['path'], truth['parameters'])
    result = {}
    pquat_finite, tquat_finite = torch.isfinite(pp[..., 3:7]).all(-1), torch.isfinite(tp[..., 3:7]).all(-1)
    for i, axis in enumerate('xyz'):
        result['root_world_v'+axis+'_m_s'] = (pp[..., 25+i], tp[..., 25+i])
        result['reward_link_v'+axis+'_m_s'] = (pred['features'][..., i], truth['features'][..., i])
        result['rotation_body_'+axis+'_rad'] = (rotation[..., i], torch.zeros_like(rotation[..., i]), pquat_finite, tquat_finite)
    result['rotation_geodesic_deg'] = (rotation.norm(dim=-1)*180./math.pi,
        torch.zeros_like(rotation[..., 0]), pquat_finite, tquat_finite)
    result['reward_yaw_rad_s'] = (pred['features'][..., 5], truth['features'][..., 5])
    result['reward_height_m'] = (pred['features'][..., 9], truth['features'][..., 9])
    result['post_state_height_m'] = (pp[..., 2], tp[..., 2])
    for i, side in enumerate(('left', 'right')):
        result['support_'+side+'_N'] = (pforce[..., i], tforce[..., i])
        result['support_'+side+'_bodyweight'] = (pforce[..., i]/weight, tforce[..., i]/weight)
    result['support_total_N'] = (pforce.sum(-1), tforce.sum(-1))
    result['support_total_bodyweight'] = (pforce.sum(-1)/weight, tforce.sum(-1)/weight)
    result['reward_weighted_dt'] = (pred['reward'], truth['reward'])
    for i, name in enumerate(TERMS):
        result['reward_term_'+name] = (pred['weighted_terms'][..., i], truth['weighted_terms'][..., i])
    return result, pc, tc, pfinite & tfinite


def summarize(pred, windows):
    truth, mask, plan = windows['truth'], windows['mask'], windows['plan']
    arrays, pc, tc, support_finite = metric_arrays(pred, truth)
    raw_nonfinite, expected_bound_infs = {}, {}
    for side, tensors in (('prediction', pred), ('truth', dict(path=truth['path'],
            state=truth['next_state'], features=truth['features'], effective_context=truth['context']))):
        for name in ('path', 'state', 'features'):
            values = tensors[name].flatten(2)
            raw_nonfinite[side+'/'+name] = (~torch.isfinite(values)).sum(-1)
        context = tensors['effective_context']
        unlimited = torch.zeros_like(context, dtype=torch.bool)
        unlimited[..., 7:19] = torch.isinf(context[..., 7:19])
        expected_bound_infs[side] = unlimited.sum(-1)
        raw_nonfinite[side+'/context_nan_or_other_nonfinite'] = (~torch.isfinite(context) & ~unlimited).sum(-1)
    lengths = mask.sum(-1)
    cumulative = {'total': (torch.where(mask, pred['reward'], 0.).cumsum(1),
                            torch.where(mask, truth['reward'], 0.).cumsum(1))}
    for i, name in enumerate(TERMS):
        cumulative[name] = (torch.where(mask, pred['weighted_terms'][..., i], 0.).cumsum(1),
                            torch.where(mask, truth['weighted_terms'][..., i], 0.).cumsum(1))
    groups = {'all': torch.ones(len(plan), dtype=torch.bool)}
    for field in ('case', 'phase', 'context_group'):
        for value in sorted({item[field] for item in plan}):
            groups[field+'/'+value] = torch.tensor([item[field] == value for item in plan])
    def at_horizon(h, selected):
        m = mask[selected, h-1]
        n = lengths[selected]
        def physical_stats(values):
            p, t, *finite = values
            return stats(p[selected, h-1], t[selected, h-1], m,
                         *(value[selected, h-1] for value in finite))
        return dict(planned_windows=int(selected.sum()), complete_windows=int((n >= h).sum()),
            partial_windows=int(((n > 0) & (n < h)).sum()), missing_windows=int((n == 0).sum()),
            planned_positions=int(selected.sum())*h, valid_positions=int(mask[selected, :h].sum()),
            endpoint_seconds=h*DT, reward_feature_seconds=h*DT-.002,
            raw_nonfinite={name: dict(
                nonfinite_available_scalar_values=int(torch.where(mask[selected, :h], counts[selected, :h], 0).sum()),
                nonfinite_available_positions=int(((counts[selected, :h] > 0) & mask[selected, :h]).sum()),
                all_planned_nonfinite_scalar_values=int(counts[selected, :h].sum()))
                for name, counts in raw_nonfinite.items()},
            expected_context_unlimited_bound_infs={side: int(torch.where(mask[selected, :h], counts[selected, :h], 0).sum())
                for side, counts in expected_bound_infs.items()},
            metrics={name: physical_stats(values) for name, values in arrays.items()},
            support_confusion={name: confusion(pc[name][selected, h-1], tc[name][selected, h-1],
                m, support_finite[selected, h-1]) for name in pc},
            hard_failure_confusion=confusion(pred['failure'][selected, h-1], truth['failure'][selected, h-1],
                m, torch.ones_like(m)),
            cumulative_full_window_reward={name: stats(p[selected, h-1], t[selected, h-1], m)
                for name, (p, t) in cumulative.items()},
            cumulative_partial_observed_reward={name: stats(p[selected, h-1], t[selected, h-1], (n > 0) & (n < h))
                for name, (p, t) in cumulative.items()})
    summaries = {name: {str(h): at_horizon(h, selected) for h in ENDPOINTS} for name, selected in groups.items()}
    curve = [at_horizon(h, groups['all']) for h in range(1, HORIZON+1)]
    per_window = []
    def number(value):
        return float(value) if torch.isfinite(value) else None
    for i, item in enumerate(plan):
        endpoints = {}
        for h in ENDPOINTS:
            endpoints[str(h)] = dict(available=bool(mask[i, h-1]), valid_positions=int(mask[i, :h].sum()),
                nonfinite_available_scalar_values={name: int(torch.where(mask[i, :h], counts[i, :h], 0).sum())
                    for name, counts in raw_nonfinite.items()},
                expected_context_unlimited_bound_infs={side: int(torch.where(mask[i, :h], counts[i, :h], 0).sum())
                    for side, counts in expected_bound_infs.items()},
                complete=bool(mask[i, h-1]), error={name: number(p[i, h-1]-t[i, h-1])
                    if mask[i, h-1] else None for name, (p, t, *_) in arrays.items()},
                cumulative_observed_weighted_dt_error={name: number(p[i, h-1]-t[i, h-1])
                    if lengths[i] else None for name, (p, t) in cumulative.items()},
                predicted_failure=bool(pred['failure'][i, h-1]) if mask[i, h-1] else None,
                true_failure=bool(truth['failure'][i, h-1]) if mask[i, h-1] else None)
        per_window.append(dict(**item, endpoints=endpoints))
    return dict(groups=summaries, curve_24_policy_steps=curve, per_window=per_window)


def markdown(report):
    lines = ['V10冻结世界模型的记录动作预测诊断；不执行策略闭环，不代表行为验收。', '',
        '每数据分支保留全部360个预声明窗口；H1/H8/H24分别是10/80/240ms，奖励特征时点为8/78/238ms。真实首terminal步的有限pre-reset物理标签保留，其后屏蔽，不按预测failure删误差。', '',
        '| world / actor数据 | H | 完整/部分/缺失 | world vx MAE | reward LINK vx MAE | 左/右承重 MAE N | 累计奖励bias/MAE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def f(value):
        return '—' if value is None else f'{value:.6g}'
    for result in report['results']:
        for h in ENDPOINTS:
            row = result['groups']['all'][str(h)]; metric = row['metrics']; reward = row['cumulative_full_window_reward']['total']
            lines.append(f"| {result['world_label']} / {result['recording']} | {h} | "
                f"{row['complete_windows']}/{row['partial_windows']}/{row['missing_windows']} | "
                f"{f(metric['root_world_vx_m_s']['mae'])} | {f(metric['reward_link_vx_m_s']['mae'])} | "
                f"{f(metric['support_left_N']['mae'])}/{f(metric['support_right_N']['mae'])} | "
                f"{f(reward['bias'])}/{f(reward['mae'])} |")
    lines += ['', 'bias为预测减真实；奖励使用原18项权重和dt。累计主表只比较完整H窗口，部分/缺失原分母和截断累计值另列JSON，非有限预测计数单列，不计作成功。', '',
        '各case、计划阶段、恒定/变化context标志分组，24步曲线、逐窗H1/H8/H24误差、承重与hardfailure混淆矩阵见JSON；全部预测和真值先保存于predictions.pt，便于独立重算。', '',
        '两world在同一记录动作上交叉评分，不能直接归因真实策略动作梯度或行为改善；seed55/24环境数据不是原seed43/96例评估重放。']
    return '\n'.join(lines)+'\n'


@torch.no_grad()
def run(args):
    if ALGORITHM_CONTRACT != V10:
        raise ValueError('Expected the reviewed V10 implementation')
    torch.set_num_threads(2)
    current = current_training_sources(); expected_infos = contracts(command_cfg('walk'))
    own_sha = digest(Path(__file__))
    records = [load_recording(path, current, expected_infos) for path in args.recording]
    if records[0]['directory'] == records[1]['directory']:
        raise ValueError('Two distinct actor recording directories are required')
    models = [load_world(label, Path(path), current, expected_infos) for label, path in args.checkpoint]
    if len({model['label'] for model in models}) != 2:
        raise ValueError('World labels must be unique')
    world_shas = {model['proof']['checkpoint_sha256'] for model in models}
    actor_shas = {record['proof']['actor_checkpoint_sha256'] for record in records}
    if len(world_shas) != 2 or len(actor_shas) != 2:
        raise ValueError('Require two distinct V10 worlds and two distinct recorded V9 controller checkpoints')
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    sources = {str(Path(__file__).resolve().relative_to(ROOT)): own_sha}
    for record in records:
        for name, sha in record['proof']['sources'].items():
            if name in sources and sources[name] != sha:
                raise ValueError('Recording source manifests disagree')
            sources[name] = sha
    for name, sha in sources.items():
        destination = args.output/'snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, destination)
        if digest(destination) != sha:
            raise ValueError('Score source snapshot differs')
    write(args.output/'source-sha256.json', sources)
    initial = []
    for record in records:
        with np.load(record['directory']/'initial.npz', allow_pickle=False) as archive:
            initial.append({name: archive[name].copy() for name in archive.files})
    initial_equal = {name: name in initial[0] and name in initial[1]
        and initial[0][name].shape == initial[1][name].shape and initial[0][name].dtype == initial[1][name].dtype
        and initial[0][name].tobytes() == initial[1][name].tobytes() for name in sorted(set(initial[0]) | set(initial[1]))}
    windows = [make_windows(record) for record in records]
    oracle = [oracle_check(models[0]['world'].adapter, record) for record in records]
    predictions = []
    with ExitStack() as guards:
        guards.enter_context(patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No optimizer in scoring')))
        guards.enter_context(patch.object(RunningNorm, 'update', side_effect=AssertionError('No norm fitting in scoring')))
        for model in models:
            for index, data in enumerate(windows):
                predictions.append(dict(world_label=model['label'], recording_index=index,
                    tensors=predict(model['world'], data, args.batch)))
    artifact = dict(schema='v10-fixed-physical-prediction-v1', starts_s=STARTS, horizon=HORIZON,
        term_names=TERMS, recordings=[r['proof'] for r in records], worlds=[m['proof'] for m in models],
        windows=windows, predictions=predictions, oracle=oracle,
        score_script_sha256=own_sha, mask_semantics='first terminal included; no later/reset sample; no predicted-failure selection')
    temporary = args.output/'predictions.tmp'; sealed = args.output/'predictions.pt'
    torch.save(artifact, temporary); temporary.replace(sealed)
    sealed_sha = digest(sealed)
    # Reload the sealed file: all reported errors derive from the saved predictions.
    del artifact, predictions, windows
    artifact = torch.load(sealed, map_location='cpu', weights_only=False)
    results = []
    for prediction in artifact['predictions']:
        index = prediction['recording_index']
        result = summarize(prediction['tensors'], artifact['windows'][index])
        results.append(dict(world_label=prediction['world_label'], recording=str(records[index]['directory']), **result))
    for model in models:
        model['proof']['world_state_sha256_after'] = state_digest(model['world'])
        if model['proof']['world_state_sha256_before'] != model['proof']['world_state_sha256_after']:
            raise AssertionError('World parameter/normalizer state changed')
        if digest(Path(model['proof']['checkpoint'])) != model['proof']['checkpoint_sha256']:
            raise AssertionError('Checkpoint changed during scoring')
        verified_training_sources(Path(model['proof']['checkpoint']), current)
    for record in records:
        for name, info in record['proof']['files'].items():
            if digest(Path(info['path'])) != info['sha256']:
                raise AssertionError(f'Recording changed during score: {name}')
        if digest(Path(record['proof']['metadata']['checkpoint'])) != record['proof']['actor_checkpoint_sha256']:
            raise AssertionError('Recording actor checkpoint changed')
        for name, sha in record['proof']['sources'].items():
            if digest(ROOT/name) != sha or digest(record['directory']/'snapshot'/name) != sha:
                raise AssertionError('Recording/current source changed')
    if digest(sealed) != sealed_sha or digest(Path(__file__)) != own_sha or current_training_sources() != current:
        raise AssertionError('Sealed predictions or scoring sources changed')
    report = dict(schema=artifact['schema'], cpu_only=True, optimizer_steps=0, normalization_updates=0,
        actor_forward_calls=0, critic_forward_calls=0, simulator_steps=0,
        score_script_sha256=own_sha, predictions_file=str(sealed), predictions_sha256=sealed_sha,
        sources_sha256=digest(args.output/'source-sha256.json'), worlds=[m['proof'] for m in models],
        recordings=[r['proof'] for r in records], initial_arrays_bitwise_equal=initial_equal,
        all_initial_arrays_bitwise_equal=all(initial_equal.values()), oracle=oracle, results=results,
        planned_windows_per_recording=360, physical_endpoints_s=[.01, .08, .24],
        reward_feature_times_s=[.008, .078, .238],
        scope='Recorded-action open-loop physics prediction; not actual training batches, counterfactual actor feedback, gradient accuracy or behavior acceptance',
        metric_rules='Physical units; bias=prediction-truth; true first-terminal finite physical sample included; post-terminal/reset excluded. Nonfinite predictions counted, not erased. Cumulative partial windows reported separately.')
    write(args.output/'score.json', report)
    (args.output/'score.md').write_text(markdown(report))
    print(json.dumps(dict(output=str(args.output), predictions_sha256=sealed_sha,
        model_data_pairs=len(results), planned_windows_per_recording=360, optimizer_steps=0), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, action='append', required=True)
    parser.add_argument('--checkpoint', nargs=2, action='append', metavar=('LABEL', 'PT'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    if len(args.recording) != 2 or len(args.checkpoint) != 2 or args.batch < 1:
        parser.error('Exactly two recordings, two labeled worlds, and a positive batch are required')
    run(args)


if __name__ == '__main__':
    main()
