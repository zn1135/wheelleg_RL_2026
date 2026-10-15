"""Save source provenance into a newly created training run directory."""
import json

from ..config import ROOT


def snapshot_source(path):
    # Keep the dirty source state with this new run, including new assets.
    import subprocess
    import hashlib
    source_roots=('mjlab_training/','resources/robots/chuanliantui_new_1/','scripts/')
    if (ROOT/'.git').exists():
        changed=subprocess.check_output(['git','diff','HEAD','--name-only','-z'],cwd=ROOT).decode().split('\0')[:-1]
        added=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')[:-1]
        relevant=[name for name in sorted(set(changed+added)) if name.startswith(source_roots)]
        provenance=dict(
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
            branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT).decode().strip(),
            source_kind='git-workspace')
        source_diff=subprocess.check_output(['git','diff','HEAD','--','mjlab_training','resources/robots/chuanliantui_new_1','scripts'],cwd=ROOT)
    else:
        # SSH releases carry an immutable version record rather than .git.
        version=(ROOT/'source-version.txt').read_text()
        fields=dict(line.split('：',1) for line in version.splitlines() if '：' in line)
        head=fields.get('提交','')
        if len(head)!=40 or any(c not in '0123456789abcdef' for c in head):
            raise ValueError('Remote source snapshot is missing a valid source commit')
        provenance=dict(head=head,branch=None,source_kind='remote-release',release=fields.get('快照'))
        (path/'source-version.txt').write_text(version)
        relevant=sorted(str(p.relative_to(ROOT)) for name in source_roots for p in (ROOT/name).rglob('*')
            if p.is_file() and not p.is_symlink() and not set(p.relative_to(ROOT).parts)&
               {'.venv','venv','__pycache__','logs','.git','.remote-training'}
            and p.suffix not in {'.pyc','.pyo'})
        source_diff=None
    hashes={}
    for name in relevant:
        source=ROOT/name
        if source.is_file() and not source.is_symlink():
            data=source.read_bytes();target=path/'source-snapshot'/name
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            hashes[name]=hashlib.sha256(data).hexdigest()
    (path/'source-state.json').write_text(json.dumps(dict(
        **provenance,source_diff_available=source_diff is not None,
        changed_files_sha256=hashes,command=__import__('sys').argv),indent=2)+'\n')
    if source_diff is not None:(path/'source.diff').write_bytes(source_diff)
