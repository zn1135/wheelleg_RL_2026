"""Strict, explicit old/new source mapping for the 2026-11-03 structural migration."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RECORD = ROOT / 'docs/maintenance/20261103-fudan-source-migration.json'
BEFORE = ROOT / 'artifacts/fudan_ppo_walk/fudan-structure-alignment-20261103/before/mjlab_training/src/ct_mjlab/fudan_ppo_walk'


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matches_source(path, expected):
    path = Path(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == expected:
        return True
    # 只允许已归类的检查脚本改变产物目录，其他源码差异仍严格拒绝。
    scripts = ROOT / 'scripts/agent/fudan_ppo_walk'
    if path.parent != scripts or path.name not in {
        'audit_reference.py', 'check.py', 'check_parking_integral.py',
        'check_reset_v2.py', 'check_structure_alignment.py', 'check_structure_suite.py', 'run_parking_integral.py',
    }:
        return False
    original = re.sub(rb'artifacts/fudan_ppo_walk/(fudan-[A-Za-z0-9_-]+)',
                      rb'artifacts/\1', data)
    return original != data and hashlib.sha256(original).hexdigest() == expected


def verify_migration():
    """Accept only enumerated hashes; unknown edits or modified old snapshots still fail."""
    record = json.loads(RECORD.read_text())
    for item in record['files']:
        if source_hash(ROOT / item['before_snapshot']) != item['old_sha256']:
            raise ValueError('Original migration source changed: ' + item['old_path'])
        if not _matches_source(ROOT / item['new_path'], item['new_sha256']):
            raise ValueError('Unknown production source change: ' + item['new_path'])
    for item in record.get('new_files', []):
        if not _matches_source(ROOT / item['path'], item['sha256']):
            raise ValueError('Unknown new source change: ' + item['path'])
    for item in record['files']:
        if 'compatibility_sha256' in item and source_hash(ROOT / item['old_path']) != item['compatibility_sha256']:
            raise ValueError('Compatibility module changed: ' + item['old_path'])
    return record


def legacy_source_directory():
    verify_migration()
    return BEFORE


def historical_path_matches(path, digest):
    """Check a sealed earlier source through explicit preexisting-change and migration edges."""
    record = verify_migration()
    path = str(path)
    if _matches_source(ROOT / path, digest):
        return True
    entry = next((item for item in record['files'] if item['old_path'] == path), None)
    if entry is None:
        # 自身更新前的源码可用已封存的模型交付快照核对。
        if path == 'mjlab_training/src/ct_mjlab/fudan_ppo_walk/source_migration.py':
            bundle = ROOT / 'exports/fudan_ppo_walk/20261103_model8000_v2'
            snapshot = bundle / 'source' / path
            manifest = bundle / 'source-sha256.json'
            if snapshot.is_file() and manifest.is_file():
                return json.loads(manifest.read_text()).get(path) == digest and source_hash(snapshot) == digest
        return False
    expected = entry.get('compatibility_sha256', entry['new_sha256'])
    if not _matches_source(ROOT / path, expected):
        return False
    if digest == entry['old_sha256']:
        return True
    for previous in record.get('preexisting_changes', []):
        if (previous['path'] == path and previous['old_sha256'] == digest
                and previous['pre_migration_sha256'] == entry['old_sha256']):
            return source_hash(ROOT / previous['old_snapshot']) == digest
    return False
