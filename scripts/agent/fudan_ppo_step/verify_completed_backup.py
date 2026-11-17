"""Safely restore a shutdown backup and run the existing CPU training audit."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    directory = parser.parse_args().directory.resolve()
    ready = json.loads((directory / 'backup-ready.json').read_text())
    archive = directory / 'training-backup.tar.gz'
    assert ready['verified'] is True
    assert archive.stat().st_size == ready['bytes']
    assert sha(archive) == ready['archive_sha256']
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            # 只接受相对普通文件，全部检查后才写入全新的目标路径。
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert path.parts and str(path) == member.name and member.name not in names
            target = directory / member.name
            assert target.is_relative_to(directory) and not target.exists()
            names.add(member.name)
        manifest = json.load(tar.extractfile('backup-manifest.json'))
        assert names == set(manifest['files']) | {'backup-manifest.json'}
        assert manifest['result'] == ready['result']
        assert len(manifest['files']) == ready['files']
        for member in members:
            target = directory / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('xb') as stream:
                shutil.copyfileobj(source, stream)
    for name, expected in manifest['files'].items():
        path = directory / name
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
    assert sha(directory / 'run/budget.json') == manifest['budget_sha256']
    verification = directory / 'verification'
    verification.mkdir()
    result = None
    if ready['result']['outcome'] == 'completed':
        from scripts.agent.fudan_ppo_step import launch_training
        assert sha(Path(launch_training.__file__)) == sha(directory / 'run/launch-source.py')
        budget = json.loads((directory / 'run/budget.json').read_text())
        assert (budget['iterations'], budget['expected_transitions'],
                budget['expected_each_adam_updates']) == (3200, 629145600, 64000)
        # 仅替换本地数据路径，训练预算和配方原样交给正式审计。
        budget['log_dir'] = str(directory / 'training')
        budget['resume_checkpoint'] = str(directory / 'inputs/resume.pt')
        (verification / 'local-budget.json').write_text(json.dumps(budget, ensure_ascii=False, indent=2))
        result = launch_training.audit(verification, budget)
        assert result['checkpoint_sha256'] == ready['result']['checkpoint_sha256']
    receipt = dict(verified=True, archive_sha256=ready['archive_sha256'],
                   local_path=str(directory), verified_unix=time.time(),
                   files_verified=len(manifest['files']), outcome=ready['result']['outcome'])
    with (directory / 'local-backup-receipt.json').open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(receipt=receipt, training_audit=result), ensure_ascii=False))


if __name__ == '__main__':
    main()
