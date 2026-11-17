"""One-run remote completion backup and platform shutdown; never stops a live worker."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

NATIVE = Path('/usr/local/sbin/shutdown')


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def read(path):
    for attempt in range(3):
        try:return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt==2:raise
            time.sleep(.1)


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def process_start(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (FileNotFoundError,ProcessLookupError):return None


def terminal(config):
    run=Path(config['run']);budget=read(run/'budget.json')
    if sha(run/'budget.json')!=config['budget_sha256']:raise ValueError('Training budget changed; shutdown disarmed')
    if process_start(config['worker_pid'])==config['worker_start']:
        return None
    status=read(run/'status.json')
    if status['status']=='completed':
        audit=read(run/'training-audit.json')
        assert status['passed'] and audit['passed']
        for key,value in [('iterations',budget['iterations']),('transitions',budget['expected_transitions']),
                          ('each_adam_updates',budget['expected_each_adam_updates'])]:
            assert status[key]==audit[key]==value,key
        final=Path(budget['log_dir'])/'model_final.pt'
        assert sha(final)==status['checkpoint_sha256']==audit['checkpoint_sha256']
        return dict(outcome='completed',checkpoint_sha256=audit['checkpoint_sha256'],iterations=audit['iterations'])
    # 异常退出也先保存所有现有检查点/日志；不把失败写成完成。
    return dict(outcome='failed_or_interrupted',checkpoint_sha256=None,status=status)


def gpu_idle():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                          capture_output=True,text=True,timeout=15,check=True)
    return not result.stdout.strip()


def backup(config,output,result):
    run=Path(config['run']);budget=read(run/'budget.json');log=Path(budget['log_dir'])
    entries=[]
    for directory,label in [(log,'training'),(run,'run')]:
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():raise ValueError('Unexpected link in training backup')
            if path.is_file():entries.append((path,str(Path(label)/path.relative_to(directory))))
    resume=budget.get('resume_checkpoint')
    if resume:entries.append((Path(resume),'inputs/resume.pt'))
    manifest_path=run.parents[2]/'remote-input-manifest.json'
    if manifest_path.exists():entries.append((manifest_path,'metadata/remote-input-manifest.json'))
    entries.append((Path(__file__),'metadata/completion_shutdown.py'))
    manifest=dict(created_unix=time.time(),result=result,budget_sha256=config['budget_sha256'],
        files={name:dict(sha256=sha(path),bytes=path.stat().st_size) for path,name in entries})
    write(output/'backup-manifest.json',manifest)
    archive=output/'training-backup.tar.gz';temporary=output/'training-backup.tar.gz.partial'
    if archive.exists() or temporary.exists():raise FileExistsError('Preserve existing backup; do not overwrite')
    with temporary.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz',compresslevel=3) as tar:
        for path,name in entries:tar.add(path,arcname=name,recursive=False)
        data=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
        info=tarfile.TarInfo('backup-manifest.json');info.size=len(data);info.mode=0o644
        tar.addfile(info,io.BytesIO(data))
    temporary.rename(archive)
    # 解压流式复核，确认备份中每个文件均与原文件SHA一致。
    with tarfile.open(archive,'r:gz') as tar:
        names=set()
        for member in tar:
            if member.name=='backup-manifest.json':continue
            assert member.isfile() and member.name in manifest['files']
            value=hashlib.sha256();source=tar.extractfile(member)
            for chunk in iter(lambda:source.read(1024*1024),b''):value.update(chunk)
            assert value.hexdigest()==manifest['files'][member.name]['sha256']
            names.add(member.name)
        assert names==set(manifest['files'])
    ready=dict(archive=str(archive),archive_sha256=sha(archive),bytes=archive.stat().st_size,
        files=len(entries),result=result,verified=True,ready_unix=time.time(),local_receipt_wait_s=600)
    write(output/'backup-ready.json',ready)
    return ready


def receipt_valid(output,ready):
    path=output/'local-backup-receipt.json'
    if not path.exists():return False
    value=read(path)
    return value.get('verified') is True and value.get('archive_sha256')==ready['archive_sha256']


def watch(args):
    output=args.output;config=read(output/'config.json');ready=None
    try:
        while True:
            if (output/'cancel.json').exists():
                write(output/'status.json',dict(status='cancelled',pid=os.getpid()));return
            result=terminal(config)
            if result is None:
                if not (output/'status.json').exists():
                    write(output/'status.json',dict(status='watching_training',pid=os.getpid(),worker_pid=config['worker_pid']))
                time.sleep(30);continue
            if not gpu_idle():
                write(output/'status.json',dict(status='waiting_for_gpu_idle',pid=os.getpid()))
                time.sleep(30);continue
            if ready is None:
                write(output/'status.json',dict(status='backing_up',pid=os.getpid(),result=result))
                ready=backup(config,output,result)
            acknowledged=receipt_valid(output,ready)
            if not acknowledged and time.time()<ready['ready_unix']+ready['local_receipt_wait_s']:
                write(output/'status.json',dict(status='waiting_for_local_backup',pid=os.getpid(),backup=ready))
                time.sleep(30);continue
            assert sha(NATIVE)==config['native_sha256'],'Platform shutdown executable changed'
            assert terminal(config) is not None and gpu_idle()
            request=dict(requested_unix=time.time(),command=[str(NATIVE)],local_backup_verified=acknowledged,
                remote_backup_verified=True,archive_sha256=ready['archive_sha256'],result=result)
            write(output/'shutdown-request.json',request)
            write(output/'status.json',dict(status='requesting_shutdown',pid=os.getpid(),request=request))
            os.sync()
            with (output/'shutdown-command.log').open('xb') as stream:
                action=subprocess.run([str(NATIVE)],stdin=subprocess.DEVNULL,stdout=stream,
                                      stderr=subprocess.STDOUT,timeout=60)
            assert action.returncode==0,('Platform shutdown returned failure',action.returncode)
            write(output/'status.json',dict(status='shutdown_requested',pid=os.getpid(),request=request))
            return
    except BaseException as exc:
        write(output/'status.json',dict(status='failed',pid=os.getpid(),error=repr(exc),automatic_restart=False))
        raise


def arm(args):
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    if (output/'config.json').exists():raise FileExistsError('Shutdown watcher already armed')
    run=args.run.resolve();budget=read(run/'budget.json');process=read(run/'process.json')
    assert budget['iterations']==3200 and budget['additional_iterations']==3000 and budget['initial_iterations']==200
    assert budget['expected_transitions']==629145600 and budget['expected_each_adam_updates']==64000
    start=process_start(process['pid']);assert start is not None,'Expected active training worker'
    command=Path(f"/proc/{process['pid']}/cmdline").read_bytes()
    assert b'launch-source.py' in command and str(run).encode() in command
    data=NATIVE.read_bytes()
    assert data[:4]==b'\x7fELF' and b'https://www.gpufree.cn/api/v1/inferring-api/webide/shutdown' in data
    config=dict(run=str(run),budget_sha256=sha(run/'budget.json'),worker_pid=process['pid'],worker_start=start,
        native_sha256=sha(NATIVE),armed_unix=time.time(),authorization='User: 训练完自动关远程',
        policy='After target exits, verify completion audit or preserve failure, verify persistent-disk backup; wait up to600s for local copy then platform shutdown.')
    write(output/'config.json',config)
    with (output/'watcher.log').open('xb') as stream:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'watch','--output',str(output.resolve())],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(output/'process.json',dict(pid=child.pid,script_sha256=sha(Path(__file__)),started_unix=time.time()))
    print(json.dumps(dict(armed=True,pid=child.pid,worker_pid=process['pid'],target_iterations=3200)))


def selfcheck(args):
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    checked=[]
    with TemporaryDirectory() as temp:
        root=Path(temp);run=root/'run';run.mkdir();logs=root/'logs';logs.mkdir()
        budget=dict(iterations=3200,expected_transitions=629145600,expected_each_adam_updates=64000,log_dir=str(logs))
        write(run/'budget.json',budget);write(run/'status.json',dict(status='completed',passed=True,
            iterations=3200,transitions=629145600,each_adam_updates=64000,checkpoint_sha256='pending'))
        config=dict(run=str(run),budget_sha256=sha(run/'budget.json'),worker_pid=os.getpid(),worker_start=process_start(os.getpid()))
        assert terminal(config) is None;checked.append('live_worker_never_shutdown')
        config['worker_pid']=999999999;config['worker_start']='not_running'
        (logs/'model_final.pt').write_bytes(b'fake-model-for-guard-check-only')
        digest=sha(logs/'model_final.pt');status=read(run/'status.json');status['checkpoint_sha256']=digest
        write(run/'status.json',status);write(run/'training-audit.json',dict(passed=True,iterations=3200,
            transitions=629145600,each_adam_updates=64000,checkpoint_sha256=digest))
        result=terminal(config);assert result['outcome']=='completed';checked.append('completed_audit_and_hash_required')
        altered=dict(status,iterations=3199);write(run/'status.json',altered)
        try:terminal(config)
        except AssertionError:checked.append('wrong_budget_rejected')
        else:raise AssertionError('Invalid terminal budget accepted')
        write(run/'status.json',status)
        (logs/'model_final.pt').write_bytes(b'changed-model')
        try:terminal(config)
        except AssertionError:checked.append('changed_checkpoint_rejected')
        else:raise AssertionError('Changed checkpoint accepted')
        (logs/'model_final.pt').write_bytes(b'fake-model-for-guard-check-only')
        write(run/'status.json',dict(status,status='failed',error='synthetic failure'))
        assert terminal(config)['outcome']=='failed_or_interrupted';checked.append('failure_preserved_without_success_claim')
        write(run/'status.json',status)
        backup_output=root/'backup';backup_output.mkdir();ready=backup(config,backup_output,result)
        assert ready['verified'];checked.append('full_archive_roundtrip_sha_verified')
        assert not receipt_valid(backup_output,ready)
        write(backup_output/'local-backup-receipt.json',dict(verified=True,archive_sha256='wrong'))
        assert not receipt_valid(backup_output,ready);checked.append('incorrect_local_receipt_rejected')
        write(backup_output/'local-backup-receipt.json',dict(verified=True,archive_sha256=ready['archive_sha256']))
        assert receipt_valid(backup_output,ready);checked.append('correct_local_receipt_accepted')
        with patch('subprocess.run') as execute:
            assert terminal(dict(config,worker_pid=os.getpid(),worker_start=process_start(os.getpid()))) is None
            execute.assert_not_called()
        fallback=root/'fallback';fallback.mkdir();native=root/'fake-platform-command';native.write_bytes(b'never executed')
        write(fallback/'config.json',dict(config,native_sha256=sha(native)))
        old_ready=dict(ready,ready_unix=time.time()-601)
        with patch(__name__+'.NATIVE',native),patch(__name__+'.gpu_idle',return_value=True), \
             patch(__name__+'.backup',return_value=old_ready),patch('os.sync'), \
             patch('subprocess.run',return_value=subprocess.CompletedProcess([str(native)],0)) as execute:
            watch(argparse.Namespace(output=fallback))
            execute.assert_called_once()
        assert read(fallback/'status.json')['status']=='shutdown_requested'
        assert not read(fallback/'shutdown-request.json')['local_backup_verified']
        checked.append('offline_fallback_only_after_verified_backup')
        checked.append('selfcheck_executes_no_shutdown_or_training')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write(args.output,dict(passed=True,checks=checked,real_shutdown_calls=0,optimizer_updates=0))
    print(json.dumps(dict(passed=True,checks=checked)))


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='mode',required=True)
    for name in ('watch','arm','selfcheck','cancel'):
        command=sub.add_parser(name);command.add_argument('--output',type=Path,required=True)
        if name=='arm':command.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    if args.mode=='cancel':write(args.output/'cancel.json',dict(cancelled_unix=time.time()));return
    {'watch':watch,'arm':arm,'selfcheck':selfcheck}[args.mode](args)


if __name__=='__main__':main()
