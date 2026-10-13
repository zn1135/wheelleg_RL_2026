"""Two parallel A2500 +1000 learners, then fixed-budget deterministic assessments."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

CASES = ('knee_gate', 'knee_only')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def training_curve(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(run), size_guidance={'scalars': 0})
    acc.Reload()
    tags = acc.Tags()['scalars']
    selected = {}
    for name in ('upright_fraction', 'height_error', 'knee_limit'):
        matches = [k for k in tags if k.endswith('/'+name)]
        if len(matches) != 1:
            raise ValueError(f'Missing/ambiguous {name}: {matches}')
        selected[name] = {e.step: e.value for e in acc.Scalars(matches[0])}
    rows = []
    for updates in range(2600, 3501, 100):
        step = updates-1
        rows.append(dict(completed_updates=updates, checkpoint=f'model_{step}.pt',
            **{name: values[step] for name, values in selected.items()}))
    lowest = min(rows, key=lambda x: (x['upright_fraction'], x['completed_updates']))
    return dict(checkpoint_metrics=rows, lowest_saved_checkpoint=lowest,
        selection='minimum original Metrics/standup/upright_fraction at each saved100-update checkpoint; earlier wins ties',
        interpretation='original episode metrics normalize by20s; training height_error is not evaluation height MAE',
        tags=tags)


def worker(root, release):
    launch = json.loads((root/'launch.json').read_text())
    env = dict(os.environ, PYTHONPATH=str(release/'mjlab_training/src'), MUJOCO_GL='egl', TZ='Asia/Shanghai',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('WHEELLEG_HUANANHU_INIT_CHECKPOINT', None)
    processes, logs = {}, {}
    status = dict(state='training', started=now(), worker_pid=os.getpid(), groups={}, commands=[], failures={})
    def update():
        status['updated'] = now()
        atomic(root/'status.json', status)
    def execute(command, log):
        status['commands'].append(dict(command=command, started=now(), log=str(log)))
        update()
        with log.open('x') as output:
            child = subprocess.Popen(command, cwd=release, env=env, stdout=output,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            status['evaluation_pid'] = child.pid
            update()
            code = child.wait()
        status['commands'][-1].update(ended=now(), exit_code=code)
        log.with_suffix('.exit-code').write_text(str(code)+'\n')
        update()
        if code:
            raise RuntimeError(f'exit{code}: {log}')
    try:
        for case in CASES:
            folder = root/case
            folder.mkdir(exist_ok=False)
            command = [sys.executable, '-u', str(release/'scripts/agent/run_chuanliantui_height_knee.py'),
                '--case', case, '--resume', launch['source'], '--run-dir', launch['runs'][case],
                '--iterations', '1000', '--num-envs', '4096', '--seed', '42']
            logs[case] = (folder/'training.log').open('x')
            child = subprocess.Popen(command, cwd=release, env=env, stdout=logs[case],
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            processes[case] = child
            status['groups'][case] = dict(state='training', pid=child.pid, started=now(), command=command)
            update()
        remaining = set(CASES)
        while remaining:
            for case in list(remaining):
                code = processes[case].poll()
                if code is None:
                    continue
                logs[case].close()
                remaining.remove(case)
                group = status['groups'][case]
                group.update(ended=now(), exit_code=code, state='training_complete' if code == 0 else 'failed')
                (root/case/'training.exit-code').write_text(str(code)+'\n')
                if code:
                    status['failures'][case] = f'Training exit{code}; no restart or extra updates'
                else:
                    from ct_mjlab.experiments import height_knee as h
                    import torch
                    run = Path(launch['runs'][case])
                    for updates in range(2600, 3501, 100):
                        saved = torch.load(run/f'model_{updates-1}.pt', map_location='cpu', weights_only=False)
                        assert h.validate_checkpoint(saved, case)['completed_updates'] == updates
                    curve = training_curve(run)
                    atomic(root/case/'training-curve.json', curve)
                    group['training_curve'] = curve
                    atomic(root/case/'training-completion.json', group)
                update()
            if remaining:
                time.sleep(5)
        status['state'] = 'evaluation'
        update()
        for case in CASES:
            if case in status['failures']:
                continue
            try:
                low = status['groups'][case]['training_curve']['lowest_saved_checkpoint']['completed_updates']
                points = sorted({3000, 3500, low})
                status['groups'][case]['evaluated_updates'] = points
                for updates in points:
                    output = root/case/f'eval-{updates}'
                    checkpoint = str(Path(launch['runs'][case])/f'model_{updates-1}.pt')
                    command = [sys.executable, '-u', str(release/'scripts/agent/eval_chuanliantui_height_knee.py'),
                        '--checkpoint', checkpoint, '--output', str(output), '--initial-normal', launch['initial_normal'],
                        '--initial-ground', launch['initial_ground'], '--reference', launch['reference']]
                    status['groups'][case]['state'] = f'evaluating_{updates}'
                    update()
                    execute(command, root/case/f'eval-{updates}.log')
                    result = json.loads((output/'evaluation.json').read_text())
                    assert result['completed_updates'] == updates
                    status['groups'][case][f'eval-{updates}'] = dict(acceptance=result['acceptance'],
                        fits=result['fits'], poses=result['poses'])
                    execute([sys.executable, str(release/'scripts/agent/render_chuanliantui_ablation_replay.py'),
                        '--input', str(output), '--groups', case, '--width', '640'], root/case/f'eval-{updates}-render.log')
                    if updates == 3500:
                        counts = {pose: v['sustained_last5s']['height_tracking'] for pose, v in result['poses'].items()}
                        if max(counts.values()) >= 32:
                            execute([sys.executable, '-u', str(release/'scripts/agent/eval_chuanliantui_height_knee_robustness.py'),
                                '--checkpoint', checkpoint, '--evaluation', str(output/'evaluation.json'),
                                '--output', str(root/case/'robustness')], root/case/'robustness.log')
                            status['groups'][case]['robustness'] = json.loads((root/case/'robustness/robustness.json').read_text())
                        else:
                            status['groups'][case]['robustness'] = dict(executed=False,
                                reason='neither canonical pose reaches32/64 height-tracking successes at0.30m', counts=counts)
                    update()
                status['groups'][case]['state'] = 'completed'
            except Exception as error:
                status['failures'][case] = str(error)
                status['groups'][case]['state'] = 'failed_evaluation'
            update()
        status.update(state='failed' if status['failures'] else 'completed', ended=now(), evaluation_pid=None)
        update()
        (root/'worker.exit-code').write_text(str(int(bool(status['failures'])))+'\n')
    except Exception as error:
        status.update(state='coordinator_failed', error=str(error), ended=now())
        update()
        (root/'worker.exit-code').write_text('1\n')
        raise
    finally:
        for log in logs.values():
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('launch', 'worker'))
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--initial-normal', type=Path)
    p.add_argument('--initial-ground', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--preflight', type=Path)
    p.add_argument('--source-replay-root', type=Path)
    args = p.parse_args()
    release = Path(__file__).resolve().parents[2]
    root = args.campaign
    if args.command == 'worker':
        worker(root, release)
        return
    assert all((args.source, args.initial_normal, args.initial_ground, args.reference, args.preflight, args.source_replay_root))
    from ct_mjlab.experiments import height_knee as h
    h.validate_source(args.source)
    manifest = json.loads((release/'release-files.json').read_text())
    assert all(hashlib.sha256((release/k).read_bytes()).hexdigest() == v for k, v in manifest['files'].items())
    check = json.loads((args.preflight/'preflight.json').read_text())
    assert check['passed'] and check['exit_code'] == 0 and check['frozen_sources_unchanged']
    assert check['module_sha256'] == manifest['files']['mjlab_training/src/ct_mjlab/experiments/height_knee.py']
    assert set(check['cases']) == {f'{case}:{n}' for case in CASES for n in (16, 4096)}
    for case in CASES:
        assert all(check['cases'][f'{case}:4096']['complete_state_equal'].values())
        replay = json.loads((args.source_replay_root/case/'evaluation.json').read_text())
        assert replay['completed_updates'] == 2500 and replay['exit_code'] == 0
        assert replay['checkpoint_sha256'] == check['cases'][f'{case}:4096']['checkpoint_sha256']
        assert replay['poses']['normal']['sustained_last5s']['wheel_balance'] == 64
        assert replay['poses']['ground']['sustained_last5s']['wheel_balance'] == 60
        assert replay['acceptance']['balance_retention_pass']
    assert not (root/'launch.json').exists() and not (root/'status.json').exists()
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    assert free >= 9000
    prefix = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    runs_root = release.parents[1]/'logs/mjlab/chuanliantui'/f'{prefix}_height_knee'
    assert not runs_root.exists()
    record = dict(created=now(), release=str(release), campaign=str(root), source=str(args.source.resolve()),
        source_sha256=h.SOURCE_SHA256, groups=list(CASES), num_envs=4096, seed=42,
        updates_per_group=1000, initial_updates=2500, target_updates=3500,
        initial_normal=str(args.initial_normal.resolve()), initial_ground=str(args.initial_ground.resolve()),
        reference=str(args.reference.resolve()), preflight=str(args.preflight.resolve()), source_replay_root=str(args.source_replay_root.resolve()),
        runs={c: str(runs_root/c) for c in CASES}, execution='two parallel learners; assessments serial; no extra budget',
        tensorboard_command=f'{Path(sys.executable).with_name("tensorboard")} --logdir={runs_root} --port=8080')
    with (root/'launch.json').open('x') as f:
        f.write(json.dumps(record, indent=2)+'\n')
    env = dict(os.environ, PYTHONPATH=str(release/'mjlab_training/src'), MUJOCO_GL='egl', TZ='Asia/Shanghai',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (root/'coordinator.log').open('xb') as log:
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--campaign', str(root)],
            cwd=release, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    record['coordinator_pid'] = child.pid
    atomic(root/'launch.json', record)
    print('HEIGHT_KNEE_CAMPAIGN '+json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
