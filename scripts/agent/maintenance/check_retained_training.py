"""核对仅保留⑥⑦⑧后的task、依赖及MuJoCo保护；不训练、不推进物理。"""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import shutil
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'artifacts/retain-678-cleanup-20261101'
OLD_TASKS = ('chuanliantui', 'chuanliantui_stand', 'chuanliantui_stand_v2',
             'chuanliantui_getup', 'chuanliantui_walk')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from ct_mjlab.envs import task_registry
    assert task_registry.names() == ('smp', 'supermimic')
    for task in OLD_TASKS:
        try:
            task_registry.get_cfgs(task)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Retired task remains registered: {task}')
    from ct_mjlab.scripts.train import train
    for task in (*OLD_TASKS, 'supermimic', 'smp'):
        try:
            train(SimpleNamespace(task=task, smp_prior=None))
        except ValueError:
            pass
        else:
            raise AssertionError('Plain PPO training must be rejected before constructing an environment')
    before = json.loads((OUT/'before-sha256.json').read_text())
    shared = [name for name in before if name.startswith('mjlab_training/src/ct_mjlab/envs/')
              and name != 'mjlab_training/src/ct_mjlab/envs/__init__.py']
    for name in shared:
        if name == 'mjlab_training/src/ct_mjlab/envs/chuanliantui_split/env_cfg.py':
            with tarfile.open(OUT/'before-source.tar.gz') as archive:
                old = ast.parse(archive.extractfile(name).read())
            old.body = [n for n in old.body if not (isinstance(n,ast.FunctionDef)
                        and n.name in ('make_getup_env_cfg','make_getup_ppo_cfg'))]
            assert ast.dump(old,include_attributes=False) == ast.dump(ast.parse((ROOT/name).read_text()),include_attributes=False)
        else:
            assert digest(ROOT/name) == before[name], f'Shared physical/model source changed: {name}'
    protection = json.loads((ROOT/'docs/maintenance/20261101-sim2sim-protection.json').read_text())
    for name, expected in protection['protected_files'].items():
        assert digest(ROOT/name) == expected, f'MuJoCo source changed: {name}'
    # 原脚本的同级导入要求将sim2sim目录放在sys.path；不修改其代码。
    sys.path.insert(0, str(ROOT/'sim2sim'))
    importlib.import_module('mj_sim2sim_chuanliantui_two_policy')
    importlib.import_module('mj_sim2sim_chuanliantui_stand_v2')
    for name in ('ct_mjlab.supermimic.__main__', 'ct_mjlab.smp.ppo', 'ct_mjlab.bc',
                 'ct_mjlab.scripts.train_bc', 'ct_mjlab.scripts.train_prior',
                 'scripts.agent.shared.recording', 'scripts.agent.shared.rendering',
                 'scripts.agent.shared.geometry'):
        importlib.import_module(name)
    with tarfile.open(OUT/'before-source.tar.gz') as archive:
        for old, new, names in [
            ('stand/render_chuanliantui_stand.py', 'shared/rendering.py', ('renderer_for',)),
            ('split/diagnose_walk_v8_regression.py', 'shared/recording.py', ('digest','write','sample_physics','Recorder')),
            ('multimode/inspect_multimode_leg_length.py', 'shared/geometry.py', ('standing_endpoint',))]:
            original = ast.parse(archive.extractfile('scripts/agent/'+old).read())
            current = ast.parse((ROOT/'scripts/agent'/new).read_text())
            for name in names:
                a = next(n for n in original.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name == name)
                b = next(n for n in current.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name == name)
                assert ast.dump(a,include_attributes=False) == ast.dump(b,include_attributes=False), name
    plan = json.loads((OUT/'deletion-plan.json').read_text())
    for name in (*plan['script_delete'], *plan['doc_delete'], *plan['asset_delete']):
        assert not (ROOT/name).exists() and not (ROOT/name).is_symlink(), f'Retired file still exists: {name}'
    for name in plan['retained_dependencies']:
        assert (ROOT/name).exists(), f'Retained data/model dependency missing: {name}'
    for p in (ROOT/'scripts/agent').rglob('*.py'):
        tree = ast.parse(p.read_text())
        compile(tree,str(p),'exec')
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('scripts.agent.'):
                path = ROOT/node.module.replace('.','/')
                assert path.is_dir() or path.with_suffix('.py').is_file(), (p,node.module)
                if path.is_dir():
                    assert all((path/(alias.name+'.py')).is_file() for alias in node.names), (p,node.module)
    from ct_mjlab.utils.source_migration import verify_training_tree
    run = ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261101_v13_continuation_to700_r1'
    hashes = json.loads((run/'source_sha256.json').read_text())
    current = ROOT/'mjlab_training/src/ct_mjlab'
    source_proof = verify_training_tree(hashes, run/'snapshot/ct_mjlab', current)
    with tempfile.TemporaryDirectory() as directory:
        clone = Path(directory)/'ct_mjlab'
        shutil.copytree(current, clone, ignore=shutil.ignore_patterns('__pycache__'))
        changed = clone/'envs/__init__.py'
        changed.write_text(changed.read_text()+'\n# unapproved change\n')
        try:
            verify_training_tree(hashes, run/'snapshot/ct_mjlab', clone)
        except ValueError:
            pass
        else:
            raise AssertionError('Unapproved source change accepted')
    import torch
    from ct_mjlab.supermimic.learner import Learner
    from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
    learner = Learner('cpu')
    received_curriculum = []
    command = SimpleNamespace(cfg=command_cfg('walk'), load_checkpoint_state=received_curriculum.append)
    saved = learner.load(run/'final.pt', command, inference=True)
    assert received_curriculum == [saved['curriculum']]
    assert all(torch.equal(value, saved['actor'][key]) for key,value in learner.actor.state_dict().items())
    report = dict(tasks=list(task_registry.names()), retired_tasks_rejected=list(OLD_TASKS),
        plain_ppo_rejected=True, shared_env_sources_verified=len(shared),
        shared_env_files_byte_identical=len(shared)-1, unused_getup_wrappers_only_removed=True,
        sim2sim_files_unchanged=len(protection['protected_files']), shared_helper_ast_equal=True,
        removed_scripts=len(plan['script_delete']), removed_document_files=len(plan['doc_delete']),
        removed_asset_roots=len(plan['asset_delete']), retained_dependencies=len(plan['retained_dependencies']),
        imports_verified=True, historical_supermimic_actor_loaded=True,
        source_migration=source_proof, unapproved_source_rejected=True, training_updates=0, physics_steps=0)
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
