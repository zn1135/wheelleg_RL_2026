#!/usr/bin/env python3
"""Record complete frozen current transitions for fixed-action prediction checks.

This is a new finite diagnostic recording, not a replay of the original96-case
assessment. The existing collect() observes every original2ms step without an
extra forward/integration; no model fitting or normalization update is allowed.
"""
import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'mjlab_training/src'))
from scripts.agent.supermimic.assess_chuanliantui_supermimic import (
    actor_digest, digest, make_cases, snapshot_sources, verify_training_sources, MODEL_FIELDS)
from ct_mjlab.envs import task_registry
from ct_mjlab.supermimic.__main__ import collect, validate_contact_encoding
from ct_mjlab.supermimic.analytic_reward import capture, WEIGHTS
from ct_mjlab.supermimic.buffer import ReplayBuffer
from ct_mjlab.supermimic.learner import Learner, ALGORITHM_CONTRACT, asset_digest
from ct_mjlab.supermimic.networks import RunningNorm
from ct_mjlab.supermimic.state import capture_state, capture_parameters
from ct_mjlab.envs.chuanliantui_split.rewards import WalkRewardState

SECONDS, HEIGHT, NUM_ENVS, SEED = 30., .26, 24, 55
STARTS = (.5, 1., 1.5, 2., 2.1, 2.3, 6., 10., 14., 16., 16.1, 16.3, 20., 24., 28.)
V9 = 'chuanliantui-supermimic-canonical-physics-path-derived-observation-reward-termination-v9'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def verify_controller_sources(checkpoint, output):
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    contract = saved.get('algorithm_contract')
    if saved.get('config',{}).get('mode') != 'train' or saved['config'].get('task') != 'chuanliantui_walk':
        raise ValueError('The diagnostic controller must be a saved walk training checkpoint')
    if contract == ALGORITHM_CONTRACT:
        return contract,verify_training_sources(checkpoint,output,'supermimic')
    if contract != V9:
        raise ValueError('This recorder supports current V10 or explicitly verified V9 actor-only controllers')
    current = ROOT/'mjlab_training/src/ct_mjlab'
    snapshot = checkpoint.parent/'snapshot/ct_mjlab'
    manifest = checkpoint.parent/'source_sha256.json'
    hashes = json.loads(manifest.read_text())
    if len(hashes) != 99 or set(hashes) != {str(p.relative_to(current)) for p in current.rglob('*.py')}:
        raise ValueError('V9 controller source coverage differs from the canonical99 files')
    changes = {}
    permitted = {'supermimic/__main__.py','supermimic/networks.py','supermimic/learner.py'}
    for name,sha in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or digest(snapshot/relative) != sha:
            raise ValueError(f'Invalid V9 controller snapshot: {name}')
        live_sha = digest(current/relative)
        if live_sha != sha:
            if name not in permitted:
                raise ValueError(f'Controller environment/actor dependency changed: {name}')
            changes[name] = dict(controller_sha256=sha,capture_sha256=live_sha)
    # Only inactive learning/world code may differ. The actual actor function stays identical.
    def definition(path,names):
        nodes = ast.parse(path.read_text()).body
        for name in names:
            node = next(x for x in nodes if getattr(x,'name',None) == name)
            nodes = getattr(node,'body',[])
        return ast.dump(node,include_attributes=False)
    for module,names in (('networks.py',('make_actor',)),('networks.py',('act',)),
                         ('__main__.py',('collect',)),('learner.py',('Learner','load'))):
        if definition(snapshot/'supermimic'/module,names) != definition(current/'supermimic'/module,names):
            raise ValueError(f'V9 controller/capture executable path changed: {module}:{names}')
    shutil.copy2(manifest,output/'controller-training-source-sha256.json')
    return contract,dict(verified=True,controller_algorithm_contract=contract,
        manifest=str(manifest),manifest_sha256=digest(manifest),snapshot_root=str(snapshot),
        source_files=99,shared_training_sources_all_equal=True,full_source_tree_equal=not changes,
        source_differences=changes,actor_and_capture_functions_ast_equal=True,
        method='All99 original snapshot hashes verified; only three inactive SM learning modules may differ; actor execution and collect ASTs are equal')


class ObservedWalkRewardState(WalkRewardState):
    def compute(self, raw):
        if self.step != raw.common_step_counter:
            self.record = capture(raw, self)
            self.record['state'] = capture_state(raw)
        return super().compute(raw)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = snapshot_sources(args.output)
    # Include this diagnostic entry alongside the unchanged training sources.
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    target = args.output/'snapshot'/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, target)
    hashes[relative] = digest(Path(__file__))
    write(args.output/'source-sha256.json', hashes)
    controller_contract,verified = verify_controller_sources(args.checkpoint,args.output)
    assert verified['verified']
    checkpoint_sha = digest(args.checkpoint)
    cfg, train = task_registry.get_cfgs('supermimic', play=True)
    cfg.auto_reset, cfg.seed = False, SEED
    env, _ = task_registry.make_env('supermimic', env_cfg=cfg, train_cfg=train,
        num_envs=NUM_ENVS, seed=SEED, device=args.device)
    raw = env.unwrapped
    try:
        from mjlab.utils.torch import configure_torch_backends
        from mjlab.utils.os import dump_yaml
        configure_torch_backends(allow_tf32=False)
        assert raw.physics_dt == .002 and raw.step_dt == .01 and env.clip_actions is None
        contact_encoding = validate_contact_encoding(raw)
        raw._split_walk_rewards = ObservedWalkRewardState(raw)
        learner = Learner(args.device)
        command = raw.command_manager.get_term('standup')
        saved = learner.load(args.checkpoint, command, inference=True)
        assert saved['algorithm_contract'] == controller_contract
        assert saved['asset_sha256'] == asset_digest()
        assert not learner.actor.obs_normalization
        for name in ('actor', 'world', 'critic'):
            getattr(learner, name).eval().requires_grad_(False)
        original = {name:actor_digest(getattr(learner, name)) for name in ('actor','world','critic')}
        command.speed_curriculum_frozen = True
        raw.reset(seed=SEED)
        cases = make_cases(NUM_ENVS)
        targets = torch.empty(NUM_ENVS, device=args.device)
        accelerations = torch.empty_like(targets)
        for case in cases:
            targets[case['env_ids']] = case['speed']
            accelerations[case['env_ids']] = case['acceleration']

        def commands(time):
            command.tracking_episode.fill_(True)
            command.zero_hold.zero_()
            command.tracking_velocity.copy_(targets if 2. <= time < 16. else torch.zeros_like(targets))
            command.tracking_yaw_rate.zero_()
            command.tracking_interval_steps.fill_(1000000)
            command.tracking_acceleration.copy_(accelerations)
            command.episode_height.fill_(HEIGHT)
            command._apply_height(None)

        commands(0.)
        raw.observation_manager.reset()
        raw.obs_buf = raw.observation_manager.compute(update_history=True)
        obs = raw.get_observations()
        fixed = dict(qpos=raw.sim.data.qpos.cpu().numpy().copy(),
            qvel=raw.sim.data.qvel.cpu().numpy().copy(),
            parameters=capture_parameters(raw).cpu().numpy(),
            env_origins=raw.scene.env_origins.cpu().numpy().copy())
        for name in MODEL_FIELDS:
            fixed['nominal_model_'+name] = np.asarray(getattr(raw.sim.mj_model, name)).copy()
            fixed['actual_model_'+name] = getattr(raw.sim.model, name).cpu().numpy().copy()
        np.savez(args.output/'initial.npz', **fixed)
        dump_yaml(args.output/'env.yaml', asdict(cfg))
        ticks = round(SECONDS/raw.step_dt)
        buffer = ReplayBuffer(ticks, NUM_ENVS, args.device)
        live = torch.ones(NUM_ENVS, device=args.device, dtype=torch.bool)
        live_before = torch.empty(ticks, NUM_ENVS, device=args.device, dtype=torch.bool)
        requests = torch.empty(ticks, NUM_ENVS, device=args.device)
        ends = torch.full((NUM_ENVS,), SECONDS, device=args.device)
        # Fail immediately if this diagnostic accidentally enters a fitting path.
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Diagnostic optimizer call')), \
             patch.object(RunningNorm, 'update', side_effect=AssertionError('Diagnostic normalization update')):
            for tick in range(ticks):
                commands(tick*raw.step_dt)
                requests[tick].copy_(command.tracking_velocity)
                live_before[tick].copy_(live)
                obs, _, done = collect(env, learner, buffer, obs, exploration=0.)
                first = live & done.bool()
                ends[first] = (tick+1)*raw.step_dt
                live &= ~done.bool()
                if done.any() and tick+1 < ticks:
                    raw.reset(env_ids=done.nonzero().flatten())
                    obs = raw.get_observations()
        assert buffer.ptr == 0 and buffer.size == ticks
        assert original == {name:actor_digest(getattr(learner, name)) for name in original}
        assert all(not getattr(learner, name).state for name in ('opt_actor','opt_encoder','opt_world','opt_critic'))
        # Keep the actual pre-reset canonical labels, not an estimated reconstruction.
        data = {name:value.cpu() for name, value in buffer.data.items()}
        record = dict(data=data, live_before=live_before.cpu(), requested_vx=requests.cpu())
        torch.save(record, args.output/'transitions.pt')
        assert digest(args.checkpoint) == checkpoint_sha
        assert all(digest(ROOT/name) == sha for name, sha in hashes.items())
        assert all(digest(args.output/'snapshot'/name) == sha for name, sha in hashes.items())
        metadata = dict(completed=True, diagnostic_kind='new_frozen_canonical_recording',
            checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_sha,
            controller_load_method='Learner.load(inference=True): actor only; unused world/critic remain cold',
            recording_actor_iteration=saved['iteration'], world_or_critic_used=False,
            algorithm_contract=ALGORITHM_CONTRACT, asset_sha256=asset_digest(), infos=saved['infos'],
            capture_algorithm_contract=ALGORITHM_CONTRACT,controller_algorithm_contract=controller_contract,
            controller_training_source_verification=verified,
            contact_encoding_assumptions=contact_encoding,
            seed=SEED, num_envs=NUM_ENVS, seconds=SECONDS, height_m=HEIGHT, cases=cases,
            policy_dt_s=raw.step_dt, physics_dt_s=raw.physics_dt,
            command_schedule=dict(initial_zero_s=[0.,2.],travel_s=[2.,16.],parking_from_s=16.),
            planned_window_starts_s=list(STARTS), planned_windows=NUM_ENVS*len(STARTS),
            fields={name:list(value.shape) for name, value in data.items()},
            transitions=ticks*NUM_ENVS, physics_samples=ticks*NUM_ENVS*5,
            first_episode_end_s=ends.cpu().tolist(), survived_first_episodes=int(live.sum()),
            source_snapshot_sha256=digest(args.output/'source-sha256.json'),
            initial_sha256=digest(args.output/'initial.npz'),
            transitions_sha256=digest(args.output/'transitions.pt'),
            training_source_verification=verified, model_state_sha256_before=original,
            model_state_sha256_after=original, parameter_updates=0, normalization_updates=0,
            exploration_std=0., physical_step_source='Original collect(): five existing post-integration substeps, no extra forward',
            reward_label_checks=f'collect verifies all{len(WEIGHTS)} current terms and total at original reward timing every tick',
            scope='New24-env seed55 diagnostic, not the original96-env seed43 behavior assessment; no training or acceptance claim')
        write(args.output/'record.json', metadata)
        print(json.dumps({key:metadata[key] for key in ('completed','checkpoint_sha256','transitions',
            'physics_samples','survived_first_episodes','planned_windows','parameter_updates','normalization_updates')}, ensure_ascii=False))
    finally:
        env.close()


if __name__ == '__main__':
    main()
