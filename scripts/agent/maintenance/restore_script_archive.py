"""将迁移前源码恢复到仓库外的新目录；不运行旧脚本，不修改历史数据。"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / 'scripts/archive/20261101-script-cleanup'


def restore(output):
    output = output.resolve()
    if output == ROOT or ROOT in output.parents or output.exists():
        raise ValueError('目标必须是仓库之外尚不存在的新目录')
    hashes = json.loads((ARCHIVE / 'source-sha256.json').read_text())
    with tarfile.open(ARCHIVE / 'source.tar.gz') as bundle:
        members = bundle.getmembers()
        if len(members) != len(hashes) or {m.name for m in members} != set(hashes):
            raise ValueError('归档文件清单不一致')
        # 先核对整个压缩包，再创建目标；不接受符号链接或越界路径。
        for member in members:
            relative = Path(member.name)
            if not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                raise ValueError('不安全的归档路径: ' + member.name)
            if hashlib.sha256(bundle.extractfile(member).read()).hexdigest() != hashes[member.name]:
                raise ValueError('源码校验失败: ' + member.name)
        output.mkdir(parents=True, exist_ok=False)
        for member in members:
            path = output / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bundle.extractfile(member).read())
            path.chmod(member.mode)
    links = {}
    # 原模型/数据只引用、不复制；旧命令的输出必须另选新目录。
    for relative in ('logs', 'artifacts', 'resources', 'mjlab_training/.venv'):
        source = ROOT / relative
        if source.exists():
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=True)
            links[relative] = str(source)
    receipt = dict(source_root=str(ROOT), restored_root=str(output),
                   verified_files=len(hashes), external_links=links,
                   scope='源码快照已验证；外部数据仍由原实验哈希检查，未启动训练或复核。')
    (output / 'archive-restore.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    restore(parser.parse_args().output)


if __name__ == '__main__':
    main()
