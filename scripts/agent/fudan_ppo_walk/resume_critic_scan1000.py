"""用户授权：真实77点Critic地形，从5000完整学习状态新增1000轮，到6000即停。"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

from . import resume_regions3000 as regional
from .resume_pd200_random_from3000 import ROOT, PYTHON, XML, sha, write
from .run_torque_ablation import run_command

MODULE = 'scripts.agent.fudan_ppo_walk.resume_critic_scan1000'
LOG = ROOT/'logs/mjlab/fudan_ppo_walk/20261116_critic_scan_regions_resume5000_to6000_01'
PREVIOUS = ROOT/'artifacts/fudan_ppo_walk/20261116-regions-resume5000-8000-01'


def prepare(out):
    import torch
    from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg, section_dict
    from ct_mjlab.envs.fudan_ppo_walk.terrain import critic_record, record
    previous = json.loads((PREVIOUS/'budget.json').read_text())
    assert sha(regional.SOURCE) == regional.SOURCE_SHA == previous['checkpoint_sha256']
    source = torch.load(regional.SOURCE, map_location='cpu', weights_only=False)
    assert source['iterations'] == 5000 and not source['partial_update']
    assert source['primary_updates'] == source['encoder_updates'] == 100000
    cfg = json.loads(json.dumps(previous['actual_config']))
    assert 'observations' not in cfg
    cfg['observations'] = dict(critic_height_scan=True)
    with ExitStack() as stack:
        for section, values in cfg.items():
            for key, value in values.items():
                stack.enter_context(patch.object(getattr(FudanWalkCfg, section), key, value))
        task = FudanWalkCfg()
        assert record(task) == previous['terrain']
        assert {name: section_dict(getattr(task, name)) for name in cfg} == cfg
        profile = critic_record(task)
    assert not LOG.exists()
    out.mkdir(parents=True, exist_ok=False)
    b = dict(previous)
    b.update(authorization='User requested adding real77point terrain heights to Critic, resume5000 for1000 additional iterations to6000; supersedes previous5000to8000 run',
        log_dir=str(LOG), target_iteration=6000, additional_iterations=1000,
        expected_added_transitions=1000*4096*48, expected_added_optimizer_steps=20000,
        actual_config=cfg, critic_observation=profile,
        certified_terrain_precheck=None, superseded_experiment=str(PREVIOUS),
        same_recipe_comparison_to=str(PREVIOUS/'budget.json'),
        only_change_from_previous_recipe={'observations.critic_height_scan': [False, True]},
        initialization='all_networks_both_optimizers_rng_counters_restored_exact_fresh_physics_history',
        critic_migration='same141slots; replace46:123 contents; retain critic weights and Adam state',
        created_unix=time.time())
    paths = list((ROOT/'mjlab_training/src/ct_mjlab').rglob('*.py'))
    paths += [Path(__file__), Path(regional.__file__), ROOT/'scripts/agent/fudan_ppo_walk/check_region_terrain.py',
              ROOT/'scripts/agent/fudan_ppo_walk/check_critic_scan.py']
    b['approved_source_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    b['xml_sha256'] = sha(XML)
    write(out/'budget.json', b)
    write(out/'target-effective-config.json', cfg)
    write(out/'recipe-diff.json', b['only_change_from_previous_recipe'])
    shutil.copy2(__file__, out/'launcher.py')
    shutil.copy2(regional.__file__, out/'regional-launcher.py')
    write(out/'source-identity.json', dict(branch=subprocess.check_output(['git','branch','--show-current'], cwd=ROOT,text=True).strip(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(), dirty=True))
    write(out/'status.json', dict(status='prepared', training_started=False))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('mode', choices=('prepare', 'precheck', 'capacity', 'train', 'run', 'audit'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if a.mode == 'prepare':
        return prepare(out)
    if a.mode == 'audit':
        return regional.audit(out)
    if a.mode != 'run':
        return regional.worker(out, a.mode)
    try:
        assert not LOG.exists()
        for stage in ('precheck', 'capacity', 'train'):
            run_command(out, stage, [str(PYTHON), '-u', '-m', MODULE, stage, '--output', str(out)])
            if stage == 'precheck':
                assert json.loads((out/'precheck/report.json').read_text())['critic_scan']['passed']
                assert json.loads((out/'precheck-runtime-audit.json').read_text())['passed']
            elif stage == 'capacity':
                assert json.loads((out/'capacity-audit.json').read_text())['passed']
        regional.audit(out)
        write(out/'status.json', dict(status='training_complete', final_iteration=6000, iterations_added=1000, behavior_accepted=False))
    except BaseException as exc:
        write(out/'status.json', dict(status='failed', error=repr(exc), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
