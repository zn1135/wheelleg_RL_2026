"""Strict, explicit old/new source mapping for the 2026-11-03 structural migration."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RECORD = ROOT / 'docs/maintenance/20261103-fudan-source-migration.json'
BEFORE = ROOT / 'artifacts/fudan-structure-alignment-20261103/before/mjlab_training/src/ct_mjlab/fudan_ppo_walk'


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_migration():
    """Accept only enumerated hashes; unknown edits or modified old snapshots still fail."""
    record = json.loads(RECORD.read_text())
    for item in record['files']:
        if source_hash(ROOT / item['before_snapshot']) != item['old_sha256']:
            raise ValueError('Original migration source changed: ' + item['old_path'])
        if source_hash(ROOT / item['new_path']) != item['new_sha256']:
            raise ValueError('Unknown production source change: ' + item['new_path'])
    for item in record.get('new_files', []):
        if source_hash(ROOT / item['path']) != item['sha256']:
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
    actual = source_hash(ROOT / path)
    if actual == digest:
        return True
    entry = next((item for item in record['files'] if item['old_path'] == path), None)
    if entry is None:
        return False
    expected = entry.get('compatibility_sha256', entry['new_sha256'])
    if actual != expected:
        return False
    if digest == entry['old_sha256']:
        return True
    for previous in record.get('preexisting_changes', []):
        if (previous['path'] == path and previous['old_sha256'] == digest
                and previous['pre_migration_sha256'] == entry['old_sha256']):
            return source_hash(ROOT / previous['old_snapshot']) == digest
    return False
