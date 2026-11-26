"""本次授权：6838续训662轮到7500，仅将gyroX观测噪声增至±0.6rad/s。"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

from . import inspect_randomization_4000 as frozen
from . import resume_regions3000 as regional
from .resume_pd200_random_from3000 import ROOT, PYTHON, XML, sha, write
from .run_torque_ablation import run_command

MODULE = 'scripts.agent.fudan_ppo_walk.resume_gyrox6838_to7500'
SOURCE_RUN = ROOT / 'logs/mjlab/fudan_ppo_walk/20261116_scan_slope5to20_mu08_resume5838_to6838_01'
SOURCE = SOURCE_RUN / 'model_final.pt'
SOURCE_SHA = '8a42776d8acb0bfdbbc922375d19b594964432f4fb065b23fef3a1384c3d96ad'
RECIPE = ROOT / 'artifacts/fudan_ppo_walk/20261118-gyrox-noise06-config-01/target-effective-config.json'
LOG = ROOT / 'logs/mjlab/fudan_ppo_walk/20261118_gyrox_noise06_resume6838_to7500_01'


def verify_source():
    assert sha(SOURCE) == SOURCE_SHA
    manifest = json.loads((SOURCE_RUN / 'source_sha256.json').read_text())
    for name, digest in manifest.items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            path = SOURCE_RUN / 'source/ct_mjlab' / name.removeprefix('mjlab_training/src/ct_mjlab/')
            assert sha(path) == digest, name
        elif name.startswith('resources/'):
            assert sha(ROOT / name) == digest, name
    return manifest


def prepare(out):
    import torch
    assert not out.exists() and not LOG.exists()
    manifest = verify_source()
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert source['iterations'] == 6838 and not source['partial_update'] and source['num_envs'] == 4096
    assert source['primary_updates'] == source['encoder_updates'] == 136760
    assert source['transitions'] == 6838 * 4096 * 48 + 184320
    assert all(torch.isfinite(v).all() for v in source['model_state_dict'].values())
    for key in ('optimizer_state_dict', 'encoder_optimizer_state_dict'):
        assert {int(v['step']) for v in source[key]['state'].values()} == {136760}
        assert all(torch.isfinite(v).all() for state in source[key]['state'].values()
                   for v in state.values() if isinstance(v, torch.Tensor))
    before = json.loads((SOURCE_RUN / 'effective-training-config.json').read_text())
    target = json.loads(RECIPE.read_text()); restored = json.loads(json.dumps(target))
    assert before['normalization']['noise_scales'][0] == .075 and target['normalization']['noise_scales'][0] == .15
    restored['normalization']['noise_scales'][0] = .075
    assert restored == before
    assert target['terrain']['mode'] == 'regions' and target['observations']['critic_height_scan']
    out.mkdir(parents=True)
    helpers = [Path(__file__), Path(regional.__file__), Path(frozen.__file__),
               ROOT / 'scripts/agent/fudan_ppo_walk/check_region_terrain.py',
               ROOT / 'scripts/agent/fudan_ppo_walk/check_critic_scan.py',
               ROOT / 'scripts/agent/fudan_ppo_walk/resume_pd200_random_from3000.py',
               ROOT / 'scripts/agent/fudan_ppo_walk/run_torque_ablation.py']
    budget = dict(authorization='User requested resume6838 to7500; total662 additional iterations, gyroX noise±0.6rad/s',
        checkpoint=str(SOURCE), checkpoint_sha256=SOURCE_SHA, source_run=str(SOURCE_RUN),
        start_iteration=6838, target_iteration=7500, additional_iterations=662,
        num_envs=4096, rollout_steps=48, seed=1, log_dir=str(LOG), actual_config=target,
        terrain=source['terrain'], critic_observation=source['critic_observation'],
        runtime_capacity=dict(nconmax=64, njmax=256),
        expected_added_transitions=662 * 4096 * 48, expected_added_optimizer_steps=662 * 20,
        expected_final_optimizer_steps_each=150000, inherited_abandoned_rollout_transitions=184320,
        only_recipe_changes={'normalization.noise_scales[0]': [.075, .15]},
        execution='Frozen6838ct_mjlab; current helpers only; no current step implementation imported',
        approved_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in helpers},
        xml_sha256=sha(XML), created_unix=time.time())
    for key in ('control_timing', 'disturbances', 'command_sampling', 'height_reward', 'physical_wheel_model'):
        budget[key] = source[key]
    write(out / 'budget.json', budget); write(out / 'target-effective-config.json', target)
    write(out / 'source-checkpoint-audit.json', dict(passed=True, complete_rollout_boundary=True,
        checkpoint_sha256=SOURCE_SHA, iteration=6838, optimizer_steps_each=136760,
        all_learning_tensors_finite=True, only_recipe_changes=budget['only_recipe_changes'], optimizer_updates=0))
    write(out / 'frozen-source-sha256.json', manifest)
    shutil.copy2(__file__, out / 'launcher.py'); shutil.copy2(regional.__file__, out / 'regional-helper.py')
    write(out / 'status.json', dict(status='prepared', training_started=False, target_iteration=7500))


def frozen_snapshot(output):
    output = Path(output)
    shutil.copytree(SOURCE_RUN / 'source/ct_mjlab', output / 'source/ct_mjlab', ignore=shutil.ignore_patterns('__pycache__'))
    manifest = verify_source()
    hashes = {n: h for n, h in manifest.items() if n.startswith('mjlab_training/src/ct_mjlab/')}
    for name, digest in manifest.items():
        if name.startswith('resources/'):
            target = output / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target); hashes[name] = digest
    doc = ROOT / 'docs/mjlab-fudan-ppo-walk.md'
    hashes[str(doc.relative_to(ROOT))] = sha(doc)
    write(output / 'source_sha256.json', hashes)
    write(output / 'frozen-runtime-origin.json', dict(source_run=str(SOURCE_RUN),
        checkpoint_sha256=SOURCE_SHA, config_override='effective-training-config.json',
        note='Config source defaults remain original6838; actual runtime overrides are saved separately'))


def worker(out, stage):
    verify_source()
    assert not any(n == 'ct_mjlab' or n.startswith('ct_mjlab.') for n in sys.modules)
    frozen.RUN = SOURCE_RUN; sys.meta_path.insert(0, frozen.FrozenTrainingSource())
    import torch
    from ct_mjlab.envs.chuanliantui.fudan_ppo_walk import fudan_ppo_walk as physical
    from ct_mjlab.rsl_rl.runners import fudan_on_policy_runner as runner
    b = json.loads((out / 'budget.json').read_text())
    old_init = physical.FudanEnv.__init__

    def initialized(env, *args, **kwargs):
        old_init(env, *args, **kwargs)
        expected = b['actual_config']['normalization']['noise_scales']
        assert env.cfg.normalization.noise_scales == expected and expected[0] == .15
        assert float(env.cfg.normalization.angular_velocity) == .25
        friction = env.env._fudan_parameters['friction']
        assert torch.allclose(friction, torch.full_like(friction, .8), atol=1e-7, rtol=0.)
        write(out / (stage + '-noise-config.json'), dict(passed=True, noise_scales=expected,
            physical_gyroX_noise_rad_s=.6, fixed_friction_minmax=[float(friction.min()), float(friction.max())]))
        if stage == 'precheck':
            rng = torch.cuda.get_rng_state_all()
            frame = env._history_frame.clone() if hasattr(env, '_history_frame') else None
            had_mask = hasattr(env, 'noise_mask'); old_mask = getattr(env, 'noise_mask', None)
            scales = env.cfg.normalization.noise_scales
            try:
                env.noise_mask = torch.zeros(env.num_envs, device=env.device)
                clean, _ = env.observations()
                env.noise_mask.fill_(1.)
                torch.cuda.set_rng_state_all(rng)
                prior = list(scales); prior[0] = .075; env.cfg.normalization.noise_scales = prior
                low, critic_low = env.observations()
                torch.cuda.set_rng_state_all(rng); env.cfg.normalization.noise_scales = scales
                high, critic_high = env.observations()
                error = float(((high[:, 0] - clean[:, 0]) - 2 * (low[:, 0] - clean[:, 0])).abs().max())
                assert error < 1e-6
                assert torch.equal(high[:, 1:], low[:, 1:]) and torch.equal(critic_low, critic_high)
                assert torch.equal(env._history_frame[:, 0], high[:, 0])
                assert float((high[:, 0] - clean[:, 0]).abs().max()) <= .150001
                write(out / 'noise-injection-audit.json', dict(passed=True, same_rng_only_gyroX_changed=True,
                    gyroX_noise_exactly_doubled_error=error, history_receives_noisy_current=True,
                    critic_unchanged=True, RNG_and_history_restored=True, optimizer_updates=0))
            finally:
                env.cfg.normalization.noise_scales = scales
                if frame is None:del env._history_frame
                else:env._history_frame.copy_(frame)
                torch.cuda.set_rng_state_all(rng)
                if had_mask:env.noise_mask = old_mask
                else:del env.noise_mask
        write(out / (stage + '-frozen-imports.json'), frozen.LOADED)

    with patch.object(regional, 'SOURCE', SOURCE), patch.object(physical.FudanEnv, '__init__', initialized), \
         patch.object(runner, 'snapshot', frozen_snapshot):
        regional.worker(out, stage)
    if stage == 'precheck':
        path = out / 'precheck/report.json'; report = json.loads(path.read_text())
        report['workspace_source_sha256'] = report['source_sha256'].copy()
        saved = verify_source()
        for name in report['source_sha256']:
            if name.startswith('mjlab_training/src/ct_mjlab/'):
                report['source_sha256'][name] = saved[name]
        report['executed_source'] = 'Verified frozen6838; runtime overrides in actual_config'
        write(path, report)
    if stage == 'precheck':
        path = out / 'precheck/report.json'; proof = json.loads(path.read_text())
        proof['workspace_source_sha256'] = dict(proof.get('source_sha256', {}))
        manifest = verify_source()
        for name in proof.get('source_sha256', {}):
            if name.startswith('mjlab_training/src/ct_mjlab/'):
                proof['source_sha256'][name] = manifest[name]
        proof['executed_source'] = 'Frozen6838ct_mjlab plus target-effective-config.json runtime overrides'
        write(path, proof)


def audit(out):
    with patch.object(regional, 'SOURCE', SOURCE):regional.audit(out)
    b = json.loads((out / 'budget.json').read_text())
    assert json.loads((LOG / 'effective-training-config.json').read_text()) == b['actual_config']
    write(out / 'completion.json', dict(passed=True, final_iteration=7500, iterations_added=662,
        optimizer_steps_each=150000, automatic_extra_training=False, behavior_accepted=False))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=('prepare', 'precheck', 'capacity', 'train', 'run', 'audit'))
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); out = a.output.resolve()
    if a.mode == 'prepare':return prepare(out)
    if a.mode == 'audit':return audit(out)
    if a.mode != 'run':return worker(out, a.mode)
    try:
        assert not LOG.exists()
        for stage in ('precheck', 'capacity', 'train'):
            write(out / 'status.json', dict(status=stage, training_started=stage == 'train', target_iteration=7500))
            run_command(out, stage, [str(PYTHON), '-u', '-m', MODULE, stage, '--output', str(out)])
            proof = 'precheck-runtime-audit.json' if stage == 'precheck' else 'capacity-audit.json'
            if stage != 'train':assert json.loads((out / proof).read_text())['passed']
        audit(out)
        write(out / 'status.json', dict(status='training_complete', final_iteration=7500,
            iterations_added=662, behavior_accepted=False))
    except BaseException as exc:
        write(out / 'status.json', dict(status='failed', error=repr(exc), automatic_retry=False))
        raise


if __name__ == '__main__':main()
