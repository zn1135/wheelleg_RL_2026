"""Exactly two parallel +500 learners, followed by serial early/final height scans."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

CASES=('curriculum','curriculum_knee')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def worker(root,release):
    launch=json.loads((root/'launch.json').read_text())
    env=dict(os.environ,PYTHONPATH=str(release/'mjlab_training/src'),MUJOCO_GL='egl',TZ='Asia/Shanghai',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('WHEELLEG_HUANANHU_INIT_CHECKPOINT',None)
    processes={};logs={}
    status=dict(state='training',started=now(),worker_pid=os.getpid(),groups={},commands=[],failures={})
    def update():
        status['updated']=now();atomic(root/'status.json',status)
    def execute(command,log):
        status['commands'].append(dict(command=command,started=now(),log=str(log)))
        update()
        with log.open('x') as output:
            child=subprocess.Popen(command,cwd=release,env=env,stdout=output,stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
            status['evaluation_pid']=child.pid;update();code=child.wait()
        status['commands'][-1].update(ended=now(),exit_code=code)
        log.with_suffix('.exit-code').write_text(str(code)+'\n');update()
        if code:raise RuntimeError(f'exit{code}: {log}')
    try:
        for case in CASES:
            folder=root/case;folder.mkdir(exist_ok=False)
            command=[sys.executable,'-u',str(release/'scripts/agent/run_chuanliantui_height_curriculum.py'),
                '--case',case,'--resume',launch['source'],'--run-dir',launch['runs'][case],
                '--iterations','500','--num-envs','4096','--seed','42']
            log=(folder/'training.log').open('x');logs[case]=log
            child=subprocess.Popen(command,cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
            processes[case]=child
            status['groups'][case]=dict(state='training',pid=child.pid,started=now(),command=command)
            update()
        remaining=set(CASES)
        while remaining:
            for case in list(remaining):
                code=processes[case].poll()
                if code is None:continue
                logs[case].close();remaining.remove(case)
                group=status['groups'][case];group.update(ended=now(),exit_code=code,
                    state='training_complete' if code==0 else 'failed')
                (root/case/'training.exit-code').write_text(str(code)+'\n')
                if code:status['failures'][case]=f'Training exit{code}; no restart or extra updates'
                else:
                    from ct_mjlab.experiments import height_curriculum as h
                    import torch
                    saved=torch.load(Path(launch['runs'][case])/'model_2999.pt',map_location='cpu',weights_only=False)
                    infos=h.validate_checkpoint(saved,case)
                    assert infos['completed_updates']==3000
                    group['height_curriculum']=infos['height_curriculum']
                    group['height_level_records']=[json.loads(x.split('HEIGHT_LEVEL ',1)[1])
                        for x in (root/case/'training.log').read_text().splitlines() if x.startswith('HEIGHT_LEVEL ')]
                    atomic(root/case/'training-completion.json',group)
                update()
            if remaining:time.sleep(5)
        status['state']='evaluation';update()
        for case in CASES:
            if case in status['failures']:continue
            try:
                for point,checkpoint in (('early101','model_2600.pt'),('final500','model_2999.pt')):
                    output=root/case/point
                    command=[sys.executable,'-u',str(release/'scripts/agent/eval_chuanliantui_height_curriculum.py'),
                        '--checkpoint',str(Path(launch['runs'][case])/checkpoint),'--output',str(output),
                        '--initial-normal',launch['initial_normal'],'--initial-ground',launch['initial_ground'],
                        '--reference',launch['reference']]
                    status['groups'][case]['state']='evaluating_'+point;update()
                    execute(command,root/case/(point+'.log'))
                    result=json.loads((output/'evaluation.json').read_text())
                    assert result['completed_updates']==(2601 if point=='early101' else 3000)
                    status['groups'][case][point]=dict(acceptance=result['acceptance'],fits=result['fits'],
                        poses=result['poses'],height_curriculum=result['height_curriculum'])
                    command=[sys.executable,str(release/'scripts/agent/render_chuanliantui_ablation_replay.py'),
                             '--input',str(output),'--groups',case,'--width','640']
                    execute(command,root/case/(point+'-render.log'))
                status['groups'][case]['state']='completed'
            except Exception as error:
                status['failures'][case]=str(error);status['groups'][case]['state']='failed_evaluation'
            update()
        status.update(state='failed' if status['failures'] else 'completed',ended=now(),evaluation_pid=None)
        update();(root/'worker.exit-code').write_text(str(int(bool(status['failures'])))+'\n')
    except Exception as error:
        status.update(state='coordinator_failed',error=str(error),ended=now());update()
        (root/'worker.exit-code').write_text('1\n')
        raise
    finally:
        for log in logs.values():log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('launch','worker'))
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--initial-normal',type=Path)
    p.add_argument('--initial-ground',type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--preflight',type=Path)
    p.add_argument('--source-replay',type=Path)
    args=p.parse_args();release=Path(__file__).resolve().parents[2];root=args.campaign
    if args.command=='worker':worker(root,release);return
    if not all((args.source,args.initial_normal,args.initial_ground,args.reference,args.preflight,args.source_replay)):
        p.error('All passed preflight/reference/canonical input paths required')
    from ct_mjlab.experiments import height_curriculum as h
    h.validate_source(args.source)
    check=json.loads((args.preflight/'preflight.json').read_text())
    assert check['passed'] and check['exit_code']==0 and check['frozen_sources_unchanged']
    assert set(check['cases'])=={f'{case}:{n}' for case in CASES for n in (16,4096)}
    assert all(all(check['cases'][f'{c}:4096']['complete_state_equal'].values()) for c in CASES)
    replay=json.loads((args.source_replay/'evaluation.json').read_text())
    assert replay['completed_updates']==2500 and replay['exit_code']==0
    assert replay['poses']['normal']['sustained_last5s']['wheel_balance']==64
    assert replay['poses']['ground']['sustained_last5s']['wheel_balance']==60
    assert replay['acceptance']['balance_retention_pass']
    if (root/'launch.json').exists() or (root/'status.json').exists():
        raise FileExistsError('Campaign already launched; never duplicate or restart automatically')
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<9000:raise RuntimeError('Insufficient GPU memory for two4096 learners')
    prefix=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    remote_root=release.parents[1]
    runs_root=remote_root/'logs/mjlab/chuanliantui'/f'{prefix}_height_curriculum'
    assert not runs_root.exists()
    record=dict(created=now(),release=str(release),campaign=str(root),source=str(args.source.resolve()),
        source_sha256=h.SOURCE_SHA256,groups=list(CASES),num_envs=4096,seed=42,
        updates_per_group=500,initial_updates=2500,target_updates=3000,
        initial_normal=str(args.initial_normal.resolve()),initial_ground=str(args.initial_ground.resolve()),
        reference=str(args.reference.resolve()),preflight=str(args.preflight.resolve()),source_replay=str(args.source_replay.resolve()),
        runs={c:str(runs_root/c) for c in CASES},execution='two parallel learners; evaluations serial; no extra budget',
        tensorboard_command=f'{Path(sys.executable).with_name("tensorboard")} --logdir={runs_root} --port=8080')
    with (root/'launch.json').open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    env=dict(os.environ,PYTHONPATH=str(release/'mjlab_training/src'),MUJOCO_GL='egl',TZ='Asia/Shanghai',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (root/'coordinator.log').open('xb') as log:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--campaign',str(root)],
            cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    record['coordinator_pid']=child.pid;atomic(root/'launch.json',record)
    print('HEIGHT_CAMPAIGN '+json.dumps(record),flush=True)


if __name__=='__main__':
    main()
