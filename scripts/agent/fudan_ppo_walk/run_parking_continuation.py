"""Run one authorized continuation, then one read-only completion audit; never retry training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args();root=args.experiment;budget=json.loads((root/'budget.json').read_text())
    proof=json.loads(Path(budget['precheck']).read_text())
    assert proof['passed'] and proof['reset_contract']==budget['expected_reset_contract']
    assert proof['command_sampling']==budget['command_sampling'] and proof['disturbances']==budget['disturbances']
    assert budget['additional_iterations']==1000 and budget['num_envs']==4096 and budget['target_iteration']==8000
    assert sha(budget['resume_checkpoint'])==budget['resume_checkpoint_sha256']
    assert not Path(budget['log_dir']).exists()
    for path,digest in json.loads((root/'protected.json').read_text()).items():assert sha(path)==digest,path
    command=budget['command'];started=time.time()
    with (root/'train.log').open('x') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        (root/'process-start.json').write_text(json.dumps(dict(pid=child.pid,wrapper_pid=os.getpid(),
            started_unix=started,command=command,log_dir=budget['log_dir']),indent=2)+'\n')
        code=child.wait()
    process=dict(returncode=code,pid=child.pid,started_unix=started,finished_unix=time.time(),
        elapsed_s=time.time()-started,command=command)
    (root/'process-result.json').write_text(json.dumps(process,indent=2)+'\n')
    if code:
        print(json.dumps(dict(training_exit_code=code,status='training_failed_no_retry')),flush=True)
        return code
    audit=Path('scripts/agent/fudan_ppo_walk/audit_resume_run.py')
    if sha(audit)!=budget['audit_script_sha256']:
        (root/'audit-failure.json').write_text(json.dumps(dict(error='Completion auditor changed after launch; manual audit required'))+'\n')
        return 1
    audit_command=[sys.executable,'-m','scripts.agent.fudan_ppo_walk.audit_resume_run',
                   '--experiment',str(root),'--output',str(root/'completion-audit.json')]
    with (root/'completion-audit.log').open('x') as stream:
        result=subprocess.run(audit_command,stdout=stream,stderr=subprocess.STDOUT)
    (root/'audit-process.json').write_text(json.dumps(dict(returncode=result.returncode,
        command=audit_command,finished_unix=time.time()),indent=2)+'\n')
    if result.returncode:
        print(json.dumps(dict(training_exit_code=0,audit_exit_code=result.returncode,status='completed_requires_audit_repair')),flush=True)
        return result.returncode
    completed=json.loads((root/'completion-audit.json').read_text())
    (root/'COMPLETED.md').write_text(f"# 停车采样续训完成\n\n恰新增1000完整轮，7000→8000，4096环境；v2→v2、无历史迁移。\n\n模型：{budget['log_dir']}/model_final.pt\n\nSHA256：`{completed['model_sha256']}`\n\n累计转移{completed['final_transitions']}，两Adam各{completed['final_updates_each']}次更新。训练退出0，一次只读完成审计通过；停车行为尚未评估。\n\n[完整审计](completion-audit.json)、[训练日志](train.log)、[运行退出](process-result.json)。\n")
    print(json.dumps(dict(training_exit_code=0,audit_exit_code=0,iterations=8000,
        checkpoint_sha256=completed['model_sha256'],status='complete_behavior_not_tested')),flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
