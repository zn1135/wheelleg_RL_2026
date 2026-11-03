"""校验本次入口清理前后的明确源码映射，不跳过历史源码哈希。"""
import hashlib
import json
from pathlib import Path


def verify_training_tree(hashes, snapshot, current, *, allowed_added_prefixes=()):
    root = Path(__file__).resolve().parents[4]
    proof = json.loads((root/'docs/maintenance/20261101-source-migration.json').read_text())
    current_files = {str(p.relative_to(current)) for p in current.rglob('*.py')}
    allowed_added = set(proof['added']) | {name for name in proof['known_sources']
                     if any(name.startswith(prefix) for prefix in allowed_added_prefixes)}
    if set(hashes)-current_files or current_files-set(hashes)-allowed_added:
        raise ValueError('Training/current source coverage differs outside approved cleanup')
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    migrated = []
    for name, expected in hashes.items():
        if digest(snapshot/name) != expected:
            raise ValueError(f'Training snapshot changed: {name}')
        actual = digest(current/name)
        if actual == expected:
            continue
        mapping = proof['changed'].get(name)
        if not mapping or mapping['before'] != expected or mapping['after'] != actual:
            raise ValueError(f'Training/current source differs outside approved cleanup: {name}')
        migrated.append(name)
    for name in current_files-set(hashes):
        expected = proof['added'].get(name, proof['known_sources'].get(name))
        if expected != digest(current/name):
            raise ValueError(f'New source differs from approved cleanup: {name}')
    return dict(verified=True, full_source_tree_equal=not migrated and current_files == set(hashes),
                migrated_sources=migrated, added_sources=sorted(current_files-set(hashes)),
                method='Historical snapshot exact SHA plus explicit before/after cleanup SHA mapping')
