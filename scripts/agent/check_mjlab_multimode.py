#!/usr/bin/env python3
"""Check the single-line mjlab multimode package against the frozen model_3023 snapshot.

Scope (no learn(), no TensorBoard, nothing written under the repo root logs/):
  source   23 env modules byte-identical to the snapshot (model.py: only its two path lines differ)
  assets   training XML / pose bank hashes and pose-bank cache signature
  config   env/PPO configs equal to the snapshot's factories; factories return fresh objects
  refs     no tracked code imports deleted ct_mjlab modules
  --gpu    registry env + runner, full resume load of model_3023, save/reload round trip,
           finite policy steps, ONNX-equivalent actor vs runner policy on the same observations

Run from the repo root:
  PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_mjlab_multimode.py \
      --snapshot <.../source_snapshot> --gpu --out /tmp/mjlab-multimode-check.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / 'mjlab_training/src/ct_mjlab'
ENV_DIR = PKG / 'envs/chuanliantui'
ASSET_DIR = ROOT / 'resources/robots/chuanliantui_new_1/mjlab'
CHECKPOINT = ROOT / 'artifacts/multimode-v1-replay-20261014/models/model_3023.pt'
CHECKPOINT_SHA256 = '624ded630635f0fb51a7378cdd04a7350eee98ffb305acbe1104c3cc7705a88a'
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
MODEL_PATH_LINES = (
    ('_PROJECT_ROOT = Path(__file__).resolve().parents[5]',
     '_PROJECT_ROOT = Path(__file__).resolve().parents[3]'),
    ('MODEL_XML = _PROJECT_ROOT / "resources" / "robots" / "chuanliantui_new_1" / "mjlab" / "chuanliantui.xml"',
     'MODEL_XML = _PROJECT_ROOT / "assets" / "chuanliantui" / "chuanliantui.xml"'),
)
DELETED = re.compile(r'ct_mjlab\.(experiments|reference_training|policy|cpu|config|asset|environment|'
                     r'registry|recipe|reset|modes|training_viewer|wheel_balance_assessment)\b|'
                     r'ct_mjlab\.envs\.chuanliantui\.(profiles|commands|chuanliantui_config)\b')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_source():
    present = sorted(p.stem for p in ENV_DIR.glob('*.py') if p.stem != '__init__')
    if present != sorted(SNAPSHOT_SHA256):
        raise AssertionError(f'env modules differ: extra={set(present)-set(SNAPSHOT_SHA256)} '
                             f'missing={set(SNAPSHOT_SHA256)-set(present)}')
    for name, digest in SNAPSHOT_SHA256.items():
        data = (ENV_DIR / f'{name}.py').read_text()
        if name == 'model':
            for ours, theirs in MODEL_PATH_LINES:
                if data.count(ours) != 1:
                    raise AssertionError(f'model.py path line not found: {ours}')
                data = data.replace(ours, theirs)
        if hashlib.sha256(data.encode()).hexdigest() != digest:
            raise AssertionError(f'{name}.py differs from the snapshot')
    return dict(modules=len(SNAPSHOT_SHA256), identical=True, model_py_changed_lines=len(MODEL_PATH_LINES))


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


def check_config(snapshot):
    """Compare our registry configs with the snapshot's own factories, loaded in a subprocess."""
    from dataclasses import asdict
    from ct_mjlab.envs import task_registry
    ours = {}
    for play in (False, True):
        env_cfg, train_cfg = task_registry.get_cfgs('chuanliantui', play=play)
        again, _ = task_registry.get_cfgs('chuanliantui', play=play)
        if env_cfg is again or env_cfg.scene is again.scene:
            raise AssertionError('get_cfgs returned shared config objects')
        ours[str(play)] = dict(env=repr(asdict(env_cfg)), train=repr(asdict(train_cfg)))
    code = (
        'import sys,json;from dataclasses import asdict;sys.path.insert(0,sys.argv[1]+"/src")\n'
        'from wheelleg_rl.chuanliantui.multimode_env_cfg import make_multimode_env_cfg as e, make_multimode_ppo_cfg as t\n'
        'print(json.dumps({str(p):dict(env=repr(asdict(e(play=p))),train=repr(asdict(t()))) for p in (False,True)}))')
    out = subprocess.run([sys.executable, '-I', '-c', code, str(snapshot)], capture_output=True, text=True, check=True)
    theirs = json.loads(out.stdout.strip().splitlines()[-1])
    # Function objects print with their module path; normalise the package prefix only.
    norm = lambda s: s.replace('wheelleg_rl.chuanliantui.', 'ct_mjlab.envs.chuanliantui.')  # noqa: E731
    result = {}
    for play in ('False', 'True'):
        for part in ('env', 'train'):
            ours_s = re.sub(r' at 0x[0-9a-f]+', '', ours[play][part])
            theirs_s = re.sub(r' at 0x[0-9a-f]+', '', norm(theirs[play][part]))
            if ours_s != theirs_s:
                raise AssertionError(f'{part} config (play={play}) differs from the snapshot')
            result[f'{part}_play_{play}'] = 'equal'
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
    parser.add_argument('--snapshot', type=Path, required=True, help='frozen model_3023 source_snapshot directory')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(ROOT / 'logs'):
        parser.error('--out must not be under the repo logs/')
    result = dict(source=check_source(), assets=check_assets(), config=check_config(args.snapshot),
                  refs=check_refs())
    if args.gpu:
        result['gpu'] = check_gpu(args.num_envs, args.steps, args.device)
    result['scope'] = 'structure/config/finite-step check; not behavior acceptance; no learn()'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
