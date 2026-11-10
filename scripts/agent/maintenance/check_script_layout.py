"""核对脚本迁移、源码归档和内部引用；不导入训练环境，不运行实验。"""
import ast
import hashlib
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[3]


def static_path(node, names):
    """只解析常量路径表达式，绝不执行被检查脚本。"""
    try:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return static_path(node.left, names) / static_path(node.right, names)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'Path' and len(node.args) == 1:
                return Path(static_path(node.args[0], names))
            if isinstance(node.func, ast.Attribute):
                value = static_path(node.func.value, names)
                if isinstance(value, Path) and node.func.attr in ('resolve', 'with_name'):
                    return getattr(value, node.func.attr)(*(static_path(a, names) for a in node.args))
        if isinstance(node, ast.Attribute) and node.attr in ('parent', 'parents'):
            return getattr(static_path(node.value, names), node.attr)
        if isinstance(node, ast.Subscript):
            return static_path(node.value, names)[static_path(node.slice, names)]
    except (KeyError, TypeError, AttributeError, ValueError, IndexError):
        pass
    return None


def main():
    manifest = json.loads((ROOT / 'scripts/agent/migration.json').read_text())
    entries = manifest['entries']
    migration_rows = (ROOT / 'scripts/agent/MIGRATION.md').read_text().splitlines()
    archive = ROOT / manifest['archive']
    hashes = json.loads((archive / 'source-sha256.json').read_text())
    with tarfile.open(archive / 'source.tar.gz') as bundle:
        members = {m.name: m for m in bundle.getmembers()}
        assert set(members) == set(hashes), '归档清单与实际文件不一致'
        for name, digest in hashes.items():
            assert hashlib.sha256(bundle.extractfile(members[name]).read()).hexdigest() == digest, name
    original_scripts = {p for p in hashes if p.startswith('scripts/agent/')
                        and len(Path(p).parts) == 3 and p.endswith(('.py', '.sh', '.c', '.json'))}
    assert original_scripts == {e['old_path'] for e in entries}, '迁移表遗漏或重复'
    assert len(entries) == len(original_scripts)
    old_names = {Path(e['old_path']).stem for e in entries if e['old_path'].endswith('.py')}
    by_name = {Path(e['old_path']).stem: e for e in entries}
    for entry in entries:
        row = next(line for line in migration_rows if line.startswith('| `' + Path(entry['old_path']).name + '`'))
        if entry['module']:
            assert entry['module'] in row, ('迁移文档入口不同步', entry['old_path'])
        if entry['status'] in ('archived', 'retired'):
            assert ('已归档' if entry['status']=='archived' else '已退役') in row, ('处置状态不同步', entry['old_path'])
        assert entry['sha256'] == hashes[entry['old_path']]
        original = ROOT / entry['old_path']
        if entry.get('retained_original'):
            assert original.is_file()
            content = original.read_bytes()
            if entry['old_path'] == 'scripts/agent/assess_chuanliantui_supermimic.py':
                content = content.replace(
                    b'artifacts/legacy_mjlab/walk25-v8-resume2000-20261030-211315',
                    b'artifacts/walk25-v8-resume2000-20261030-211315')
            assert hashlib.sha256(content).hexdigest() == entry['live_sha256']
        elif entry.get('runtime_protected'):
            assert original.is_file() and hashlib.sha256(original.read_bytes()).hexdigest() == entry['sha256'], original
            for name in entry['dependencies']:
                assert by_name[name].get('runtime_protected'), ('运行中依赖未保护', entry['old_path'], name)
        else:
            assert not original.exists(), entry['old_path']
        if entry['status'] in ('archived', 'retired'):
            assert entry['new_path'] is None and entry['module'] is None
        else:
            assert (ROOT / entry['new_path']).is_file(), entry
            if entry['module']:
                assert entry['module'].replace('.', '/') + '.py' == entry['new_path']
    count = 0
    for path in sorted((ROOT / 'scripts/agent').glob('*/*.py')):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), 'exec')
        names = {'__file__': str(path)}
        historical_suffixes = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                value = static_path(node.value, names)
                if value is not None:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            names[target.id] = value
                if any(isinstance(t, ast.Name) and t.id == 'PROTOCOL_SOURCE_SUFFIXES' for t in node.targets):
                    historical_suffixes.update(ast.literal_eval(node.value))
        for node in ast.walk(tree):
            value = static_path(node, names)
            if isinstance(value, Path) and ROOT / 'scripts/agent' in value.parents:
                assert value.exists(), (path, node.lineno, str(value))
            if isinstance(node, ast.Import):
                assert not any(n.name in old_names for n in node.names), (path, node.lineno)
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in old_names, (path, node.lineno)
                if node.module.startswith('scripts.agent.'):
                    target = ROOT / node.module.replace('.', '/')
                    assert target.is_dir() or target.with_suffix('.py').is_file(), (path, node.module)
                    if target.is_dir():
                        for name in node.names:
                            assert (target / (name.name + '.py')).is_file(), (path, name.name)
        for match in re.finditer(r'scripts/agent/[\w/.-]+\.(?:py|sh|json|c)', source):
            relative = match.group()
            prefix = source[max(0, match.start() - 20):match.start()]
            if 'snapshot/' in prefix or 'extractfile(' in prefix or relative in historical_suffixes:
                assert relative in hashes, (path, relative)
                continue
            if path.name == 'check_script_merges.py' and relative in hashes:
                continue  # 此检查显式构造旧报告路径，随后验证来源匹配。
            assert (ROOT / relative).is_file() or relative in hashes, (path, relative)
        count += 1
    result = dict(archive_files=len(hashes), original_python_scripts=len(old_names),
                  checked_python_files=count,
                  archived=sum(e['status'] == 'archived' for e in entries),
                  retired=sum(e['status'] == 'retired' for e in entries),
                  merged=sum(e['status'] == 'merged' for e in entries),
                  runtime_protected=sum(bool(e.get('runtime_protected')) for e in entries))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
