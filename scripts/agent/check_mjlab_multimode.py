#!/usr/bin/env python3
"""Check the single-line mjlab multimode package against the frozen model_3023 snapshot.

Scope (no learn(), no TensorBoard, nothing written under the repo root logs/):
  source   every env module equals the frozen snapshot text after the recorded source
           deviations are applied to the snapshot (byte-exact, so any unrecorded edit
           fails), and the module set is exactly the snapshot's 23 recorded modules
  assets   training XML / pose bank hashes and pose-bank cache signature
  config   our env/PPO configs equal the snapshot's own factories after the recorded
           config deviations are applied to the snapshot dump; factories return fresh
           objects
  refs     no tracked code imports deleted ct_mjlab modules
  --gpu    registry env + runner, full resume load of model_3023, save/reload round trip,
           finite policy steps, ONNX-equivalent actor vs runner policy on one observation set

The two deviation tables live in ``multimode_snapshot_deviations.json`` next to this
script. They are the only sanctioned differences from the frozen source, and they exist
because the mid-episode recovery work (2026-10-15/16) and the flat velocity/yaw-rate
tracking phase (2026-10-17) intentionally changed behaviour. Any other difference is a
failure: a new behaviour change must be added to the tables deliberately, it never
becomes a new baseline by itself.

The snapshot that matches the recorded hashes is outside this repository: the copy this repo
used to keep under ``artifacts/multimode-v1-replay-20261014/source_snapshot`` is gone, so
``--snapshot`` defaults to the first candidate in ``SNAPSHOT_CANDIDATES`` whose 23 module hashes
match the recorded table.

Run from the repo root:
  PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_mjlab_multimode.py \
      --gpu --out /tmp/mjlab-multimode-check.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / 'scripts' / 'agent'
PKG = ROOT / 'mjlab_training/src/ct_mjlab'
ENV_DIR = PKG / 'envs/chuanliantui'
ASSET_DIR = ROOT / 'resources/robots/chuanliantui_new_1/mjlab'
CHECKPOINT = ROOT / 'artifacts/multimode-v1-replay-20261014/models/model_3023.pt'
CHECKPOINT_SHA256 = '624ded630635f0fb51a7378cdd04a7350eee98ffb305acbe1104c3cc7705a88a'
DEVIATIONS_PATH = SCRIPTS_DIR / 'multimode_snapshot_deviations.json'
# Copies of the frozen model_3023 snapshot that still hold the 23 chuanliantui modules. Each was
# verified against the recorded hashes below; resolve_snapshot() picks the first one that matches.
SNAPSHOT_CANDIDATES = (
    Path('/home/zn/文档/mjlab/wheelleg_RL/logs/validation/multimode_v1/formal_3000_20261006-213223/source_snapshot'),
    Path('/home/zn/文档/mjlab/wheelleg_RL/logs/validation/multimode_recovery_deploy_20261008-023839/source_snapshot'),
    Path('/home/zn/文档/mjlab/wheelleg_RL/logs/validation/crossing_easy_v2_20261007-235422/source_snapshot'),
    Path('/home/zn/.codex/worktrees/9e9a/Wheel-Legged-Gym/artifacts/multimode-v1-replay-20261014/source_snapshot'),
)
ABSENT = '__absent__'
# From the snapshot's IMPORT_MANIFEST.json (src/wheelleg_rl/chuanliantui/<name>.py).
SNAPSHOT_SHA256 = {
    'gas_spring': 'e5e104066a12e511ea145dbe74b04357541ec03f9e5919063dfd6ae7f16d787a',
    'huananhu_action': '03e7858f90f4183f6daa648d08d5dc12e88a0f13323f489ad6f1c1bf7d1b1038',
    'huananhu_env_cfg': '0a26fd64ed4a10498274f55819dc08cfef46559c7a3b80e878e5bfd2e131a76e',
    'huananhu_mdp': '9ee425d7ac0772eedd04f6ce506fa00c58a709a6cd88b2248b072703cd4166ab',
    'huananhu_observations': 'cec4d05bd089a5139faade265c2858e2865d23ce54ea28187f20a9a8665e806b',
    'huananhu_poses': 'be5ee05224ab52038d19df4c53706f59a55de5582ebf23c2c24f95d9111293fa',
    'huananhu_randomization': 'd365216ca031847a2d1c13afbd93f11050e3e1d5de12c6901e1dac03d14b6d43',
    'huananhu_rewards': '8ed3b05dc8c47607168a86a01408a9c94711f813c3aa71ef581231fb764c4add',
    'huananhu_runner': 'da8796450f4638bcfd0e8df3e33786ce4d84790082660a8b87551fed70e8b5f4',
    'model': '73b7ebc49b07a4f422a99b0a6e623ad468d592285c0c14bacb21bc393675ed05',
    'multimode_env_cfg': 'a1a22b032236fc6aa83006d36f06c75d2502ec7b393775e74e59a2594a8ff966',
    'multimode_mdp': 'd821bca1cd32245513d9c950a14cea1db070c52868665a72daeb31c5ecd141f5',
    'multimode_observations': '6e3a1ca33067eb845b78753208253724bd829b7c58112c5a582fa154d730f911',
    'multimode_rewards': '718e5b0b07ce9dd5ff8d2a0b8e1ef036e56f17afde122f84ce91e26bd3fb0ad1',
    'multimode_runner': 'ef580d18a76d9e6f8f12ca4475ed0878388220116073fb2e651988b6e1674de2',
    'multimode_terrains': '156fefda351ecc24e3fcca3e63112aad80e02bcb66227c48bf0f15ecda936bb3',
    'pd_action': '8e87a7dd5c2d2245e2b1b962ca585dd1a9096e3016f48f22b71b2f171692506d',
    'runner': '9fa68365b2139778fb2ea90f20501141631931002d1d0be7783fdcc4cbd4b3e9',
    'standup_mdp': '43a86e00e398f99d9dc5a5a5085e0c167452ecdfc6eb5b56ea4d5e558367a045',
    'standup_randomization': '0c7af188444d115fab38e554194b130decc4c37ad29a2c46348646ed59dc1d79',
    'v14_observations': '02925f120d15ebd06f2531b1e19d654f47bdeb7f61b6573873a57616421bd58a',
    'velocity_env_cfg': '5a380a7fcc6e93e7eaff3d93a8df09292ea7dcf036930000d67dc00b1aaca5e6',
    'velocity_mdp': '992dd2767c1d675805e0616e206b4d1c9de4b66c6164cb83feb2f99b9b12c8dc',
}
ASSET_SHA256 = {
    'chuanliantui.xml': 'dfe01afb17402ae3318a0430a58cbfc91cd0f755be4c48be702bb61fa2339f03',
    'huananhu_pose_bank.npz': 'ca82470ace721aac630f8376c06b56a6c182c2cda4931051fdb7e003d2a93db1',
}
DELETED = re.compile(r'ct_mjlab\.(experiments|reference_training|policy|cpu|config|asset|environment|'
                     r'registry|recipe|reset|modes|training_viewer|wheel_balance_assessment)\b|'
                     r'ct_mjlab\.envs\.chuanliantui\.(profiles|commands|chuanliantui_config)\b')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(value):
    """JSON-able view of a config dump; objects become typed reprs without addresses."""
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        # str-mixin enums (e.g. TransmissionType) must become plain strings on both sides;
        # otherwise the snapshot dump's JSON round trip compares str with the enum class.
        return str(value)
    # Functions and classes print with their defining module, which differs between the snapshot
    # package and ours. Canonicalise the package prefix so the comparison sees a real class or
    # callable change instead of a naming artefact.
    text = f'{type(value).__name__}:{value!r}'
    for prefix in ('wheelleg_rl.chuanliantui.', 'ct_mjlab.envs.chuanliantui.'):
        text = text.replace(prefix, 'pkg.')
    return re.sub(r' at 0x[0-9a-f]+', '', text)


def load_deviations():
    if not DEVIATIONS_PATH.is_file():
        raise SystemExit(
            f'Missing deviation table {DEVIATIONS_PATH}. It records the only sanctioned '
            f'differences from the frozen snapshot and must be regenerated deliberately.')
    return json.loads(DEVIATIONS_PATH.read_text())


def snapshot_modules(snapshot):
    modules = Path(snapshot) / 'src' / 'wheelleg_rl' / 'chuanliantui'
    if not modules.is_dir():
        raise SystemExit(f'Snapshot modules not found under {modules}')
    return modules


def resolve_snapshot(path=None):
    """Return the snapshot directory whose 23 module hashes match the recorded table."""
    tried = []
    for candidate in ([Path(path)] if path is not None else list(SNAPSHOT_CANDIDATES)):
        modules = candidate / 'src' / 'wheelleg_rl' / 'chuanliantui'
        if not modules.is_dir():
            tried.append(f'{candidate} (no src/wheelleg_rl/chuanliantui)')
            continue
        mismatched = [name for name, digest in SNAPSHOT_SHA256.items()
                      if not (modules / f'{name}.py').is_file()
                      or sha256(modules / f'{name}.py') != digest]
        if not mismatched:
            return candidate
        tried.append(f'{candidate} ({len(mismatched)} hash mismatch, e.g. {mismatched[:3]})')
    raise SystemExit('No snapshot matches the recorded 23 module hashes. Tried:\n  '
                     + '\n  '.join(tried))


def check_source(snapshot):
    """Byte-compare every module with the snapshot, allowing only recorded deviations."""
    deviations = load_deviations()['source']
    modules = snapshot_modules(snapshot)
    present = sorted(path.stem for path in ENV_DIR.glob('*.py') if path.stem != '__init__')
    if present != sorted(SNAPSHOT_SHA256):
        raise AssertionError(
            f'env module set differs from the recorded 23: '
            f'extra={sorted(set(present) - set(SNAPSHOT_SHA256))} '
            f'missing={sorted(set(SNAPSHOT_SHA256) - set(present))}')
    identical, deviating = 0, {}
    for name, digest in SNAPSHOT_SHA256.items():
        frozen = modules / f'{name}.py'
        frozen_text = frozen.read_text()
        if hashlib.sha256(frozen_text.encode()).hexdigest() != digest:
            raise AssertionError(
                f'{name}.py snapshot hash {sha256(frozen)[:12]} != recorded {digest[:12]}: '
                f'wrong snapshot directory?')
        ours = (ENV_DIR / f'{name}.py').read_text()
        if hashlib.sha256(ours.encode()).hexdigest() == digest:
            identical += 1
            continue
        blocks = deviations.get(name, [])
        if not blocks:
            raise AssertionError(f'{name}.py differs from the snapshot with no recorded deviation')
        text = frozen_text
        for theirs, ours_block in blocks:
            if text.count(theirs) != 1:
                raise AssertionError(f'{name}.py: recorded deviation block is not unique in the snapshot')
            text = text.replace(theirs, ours_block)
        if text != ours:
            raise AssertionError(
                f'{name}.py differs from the snapshot outside the {len(blocks)} recorded deviation block(s)')
        deviating[name] = len(blocks)
    stale = sorted(set(deviations) - set(deviating))
    if stale:
        raise AssertionError(f'deviation table lists modules that now match the snapshot: {stale}')
    return dict(modules=len(SNAPSHOT_SHA256), identical=identical, deviating=deviating)


def check_assets():
    import numpy as np
    from ct_mjlab.envs.chuanliantui import huananhu_poses, model
    for name, digest in ASSET_SHA256.items():
        if sha256(ASSET_DIR / name) != digest:
            raise AssertionError(f'{name} hash differs from the snapshot')
    if model.MODEL_XML.resolve() != (ASSET_DIR / 'chuanliantui.xml').resolve():
        raise AssertionError(f'MODEL_XML points to {model.MODEL_XML}')
    with np.load(ASSET_DIR / 'huananhu_pose_bank.npz', allow_pickle=False) as bank:
        if str(bank['cache_signature']) != huananhu_poses._cache_signature():
            raise AssertionError('pose-bank cache signature is stale; training would regenerate it')
    model.load_model()
    return dict(xml=ASSET_SHA256['chuanliantui.xml'], pose_bank=ASSET_SHA256['huananhu_pose_bank.npz'],
                cache_signature_valid=True, xml_compiles=True)


def snapshot_config_dump(snapshot):
    """Run the snapshot's own config factories in a subprocess and return normalised dumps.

    The subprocess imports this module for ``normalize`` so both dumps are produced by one
    implementation; it imports nothing from ``ct_mjlab``.
    """
    code = (
        'import sys, json\n'
        f'sys.path.insert(0, {str(SCRIPTS_DIR)!r})\n'
        'import check_mjlab_multimode as check\n'
        f'sys.path.insert(0, {str(Path(snapshot) / "src")!r})\n'
        'from dataclasses import asdict\n'
        'from wheelleg_rl.chuanliantui.multimode_env_cfg import ('
        'make_multimode_env_cfg as make_env, make_multimode_ppo_cfg as make_train)\n'
        'dump = {"train": check.normalize(asdict(make_train()))}\n'
        'for play in (False, True):\n'
        '    dump[f"env_play_{play}"] = check.normalize(asdict(make_env(play=play)))\n'
        'print(json.dumps(dump))\n')
    out = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def set_path(dump, path, value):
    keys = path.split('.')
    target = dump
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    if value == ABSENT:
        target.pop(keys[-1], None)
    else:
        target[keys[-1]] = value


def first_difference(expected, observed, prefix=''):
    if type(expected) is not type(observed):
        return f'{prefix}: {type(expected).__name__} != {type(observed).__name__}'
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(observed)):
            if key not in expected:
                return f'{prefix}.{key}: ours only'
            if key not in observed:
                return f'{prefix}.{key}: snapshot only'
            found = first_difference(expected[key], observed[key], f'{prefix}.{key}')
            if found:
                return found
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f'{prefix}: length {len(expected)} != {len(observed)}'
        for index, (a, b) in enumerate(zip(expected, observed)):
            found = first_difference(a, b, f'{prefix}[{index}]')
            if found:
                return found
        return None
    return None if expected == observed else f'{prefix}: {expected!r} != {observed!r}'


def check_config(snapshot):
    """Compare our configs with the snapshot factories, allowing recorded config deviations."""
    deviations = load_deviations()['config']
    theirs = snapshot_config_dump(snapshot)
    from dataclasses import asdict
    from ct_mjlab.envs import task_registry
    ours = {}
    for play in (False, True):
        env_cfg, train_cfg = task_registry.get_cfgs('chuanliantui', play=play)
        again, _ = task_registry.get_cfgs('chuanliantui', play=play)
        if env_cfg is again or env_cfg.scene is again.scene:
            raise AssertionError('get_cfgs returned shared config objects')
        ours[f'env_play_{play}'] = normalize(asdict(env_cfg))
        if play is False:
            ours['train'] = normalize(asdict(train_cfg))
    result = {}
    for key, dump in theirs.items():
        blocks = deviations.get(key, [])
        for path, _snapshot_value, ours_value in blocks:
            set_path(dump, path, ours_value)
        difference = first_difference(dump, ours[key])
        if difference:
            raise AssertionError(
                f'{key} config differs from the snapshot outside the recorded deviations: {difference}')
        result[key] = f'{len(blocks)} recorded deviations, otherwise equal'
    return result


def check_refs():
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard',
                                     '*.py', '*.sh', '*.toml'], cwd=ROOT, text=True).split()
    hits = [f for f in files if not f.startswith(('wheel_legged_gym/', 'deployment/'))
            and f != 'scripts/agent/check_mjlab_multimode.py'
            and (ROOT / f).is_file() and DELETED.search((ROOT / f).read_text(errors='ignore'))]
    if hits:
        raise AssertionError(f'files still import deleted ct_mjlab modules: {hits}')
    return dict(scanned=len(files), stale_imports=0)


def check_gpu(num_envs, steps, device):
    import torch
    from ct_mjlab.envs import task_registry
    from ct_mjlab.scripts.export import load_actor

    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise AssertionError('model_3023 checkpoint hash differs')
    source_infos = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)['infos']
    env_cfg, train_cfg = task_registry.get_cfgs('chuanliantui')
    env, _ = task_registry.make_env('chuanliantui', env_cfg=env_cfg, train_cfg=train_cfg,
                                    num_envs=num_envs, seed=train_cfg.seed, device=device)
    try:
        # Same call train.py makes with --resume, but log_dir=None: nothing is written to logs/.
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train_cfg,
                                                  resume=CHECKPOINT, device=device)
        effective = runner._effective_training_iteration()
        if effective != int(source_infos['huananhu_training_iteration']):
            raise AssertionError(f'curriculum iteration {effective} != {source_infos["huananhu_training_iteration"]}')
        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp) / 'model_roundtrip.pt'
            runner.save(str(saved))
            reloaded = torch.load(saved, map_location='cpu', weights_only=False)
            original = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
            for key in ('actor_state_dict', 'critic_state_dict'):
                for name, value in original[key].items():
                    if not torch.equal(value, reloaded[key][name]):
                        raise AssertionError(f'{key}.{name} changed in save/load round trip')
            for key in ('observation_contract', 'experiment_profile_contract', 'control_contract',
                        'huananhu_training_iteration'):
                if reloaded['infos'][key] != source_infos[key]:
                    raise AssertionError(f'infos.{key} changed in save/load round trip')
            runner.load(str(saved))  # contract checks accept our own save
        policy = runner.get_inference_policy(device=device)
        actor = load_actor(CHECKPOINT).to(device)
        obs = env.get_observations()
        max_diff = 0.
        with torch.inference_mode():
            for _ in range(steps):
                action = policy(obs)
                max_diff = max(max_diff, float((actor(obs['actor']) - action).abs().max()))
                obs, reward, _, _ = env.step(action)
                for name in ('actor', 'critic'):
                    if not torch.isfinite(obs[name]).all():
                        raise AssertionError(f'non-finite {name} observation')
                if not torch.isfinite(reward).all():
                    raise AssertionError('non-finite reward')
        if max_diff > 1e-5:
            raise AssertionError(f'export actor differs from runner policy by {max_diff}')
        shapes = {name: list(obs[name].shape) for name in ('actor', 'critic')}
        if shapes != {'actor': [num_envs, 39], 'critic': [num_envs, 82]} or env.num_actions != 6:
            raise AssertionError(f'unexpected shapes {shapes}, actions {env.num_actions}')
        return dict(num_envs=num_envs, steps=steps, device=device, resume_curriculum_iteration=effective,
                    roundtrip='actor/critic/infos identical', export_actor_vs_policy_max_abs=max_diff,
                    observation_shapes=shapes, learn_called=False)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--snapshot', type=Path, default=None,
                        help='frozen model_3023 source snapshot holding src/wheelleg_rl/chuanliantui; '
                             'default: the first candidate whose 23 module hashes match')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(ROOT / 'logs'):
        parser.error('--out must not be under the repo logs/')
    snapshot = resolve_snapshot(args.snapshot)
    result = dict(snapshot=dict(path=str(snapshot), resolved_from_candidates=args.snapshot is None),
                  source=check_source(snapshot), assets=check_assets(),
                  config=check_config(snapshot), refs=check_refs())
    if args.gpu:
        result['gpu'] = check_gpu(args.num_envs, args.steps, args.device)
    result['scope'] = ('structure/config/finite-step check against the frozen snapshot with recorded '
                       'deviations; not behavior acceptance; no learn()')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
