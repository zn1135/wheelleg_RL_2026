"""Adopt the existing baseline and run the other authorized factors concurrently."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from launch_chuanliantui_recovery_factors import CASES, analysis, now


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def alive(pid):
    try:
        # A finished child can remain a zombie until the old parent is reaped.
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def validate_final(path, case):
    import torch
    from ct_mjlab.experiments.recovery_factors import validate_checkpoint
    saved = torch.load(path, map_location='cpu', weights_only=False)
    infos = validate_checkpoint(saved, case)
    if infos['completed_updates'] != 2500:
        raise ValueError('Expected exactly500 additional updates')
    if not all(torch.isfinite(v).all() for key in ('actor_state_dict', 'critic_state_dict')
               for v in saved[key].values()):
        raise ValueError('Nonfinite final model')


def worker(root, release):
    record = read(root / 'parallel-launch.json')
    original = read(root / 'launch.json')
    source = read(root / 'source2000/evaluation.json')
    env = dict(os.environ, PYTHONPATH=str(release / 'mjlab_training/src'), PYTHONUNBUFFERED='1',
               MUJOCO_GL='egl', TZ='Asia/Shanghai', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('WHEELLEG_HUANANHU_INIT_CHECKPOINT', None)
    os.environ.update(env)  # The coordinator's own final-model validator uses the same frozen code.
    sys.path.insert(0, str(release / 'mjlab_training/src'))
    processes = {}
    outputs = {}
    evaluated = set()
    failed = {}
    stages = {case: 'training' for case in CASES}
    pids = {'baseline': record['adopted_baseline_pid']}
    progress = dict(execution='parallel', worker_pid=os.getpid(), state='starting',
                    groups=stages, train_pids=pids, started_at_utc=now(), evaluations={})

    def update():
        progress['updated_at_utc'] = now()
        atomic(root / 'parallel-status.json', progress)
        atomic(root / 'status.json', progress)

    def execute(command, log, phase, case):
        with log.open('x') as output:
            child = subprocess.Popen(command, cwd=release, env=env, stdout=output,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            progress.update(evaluation_case=case, evaluation_pid=child.pid, evaluation_phase=phase)
            update()
            code = child.wait()
        log.with_suffix('.exit-code').write_text(str(code) + '\n')
        if code:
            raise RuntimeError(f'{phase} exit{code}: {log}')

    def evaluate(case):
        folder = root / case
        run = Path(original['runs'][case])
        for label, checkpoint, count in (('early101', 'model_2100.pt', 2101), ('final500', 'model_2499.pt', 2500)):
            output = folder / label
            command = [sys.executable, '-u', str(release / 'scripts/agent/eval_chuanliantui_recovery_factors.py'),
                '--checkpoint', str(run / checkpoint), '--output', str(output),
                '--initial-ground', original['initial_ground'],
                '--initial-normal', str(root / 'source2000/initial-normal.npz')]
            execute(command, folder / (label + '.log'), 'evaluating', case)
            result = read(output / 'evaluation.json')
            if result['completed_updates'] != count:
                raise ValueError('Evaluation update count mismatch')
            command = [sys.executable, str(release / 'scripts/agent/render_chuanliantui_ablation_replay.py'),
                       '--input', str(output), '--groups', case, '--width', '640']
            execute(command, folder / (label + '-render.log'), 'rendering', case)
            analysis(root, case, label, result, source)
            if case == 'recovery_reference':
                for other in CASES:
                    previous = root / other / label / 'evaluation.json'
                    if other != case and previous.exists():
                        analysis(root, other, label, read(previous), source)
            progress['evaluations'][case] = label
            update()

    # No baseline restart: its original child/output/run remain in place.
    try:
        for case in CASES[1:]:
            folder = root / case
            folder.mkdir(exist_ok=False)
            command = [sys.executable, '-u', str(release / 'scripts/agent/run_chuanliantui_recovery_factors.py'),
                '--case', case, '--resume', original['source'], '--run-dir', original['runs'][case],
                '--iterations', '500', '--num-envs', '4096', '--seed', '42']
            output = (folder / 'training.log').open('x')
            outputs[case] = output
            child = subprocess.Popen(command, cwd=release, env=env, stdout=output,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            processes[case] = child
            pids[case] = child.pid
            update()
        progress['state'] = 'training_and_evaluation'
        update()
        while len(evaluated | set(failed)) < len(CASES):
            ready = []
            for case in CASES:
                if case in evaluated or case in failed:
                    continue
                if case == 'baseline':
                    finished = not alive(pids[case])
                    code = None  # Exit status cannot be waitpid-observed after adoption.
                else:
                    code = processes[case].poll()
                    finished = code is not None
                if not finished:
                    continue
                try:
                    if code not in (None, 0):
                        raise RuntimeError(f'Training exit{code}')
                    final = Path(original['runs'][case]) / 'model_2499.pt'
                    validate_final(final, case)
                    stages[case] = 'training_complete'
                    completion = dict(completed_updates=2500, final_checkpoint=str(final),
                        exit_code=code, adopted=case == 'baseline', verified_at_utc=now(),
                        evidence='final full checkpoint/interface/count/finite tensors verified')
                    (root / case / 'training-completion.json').write_text(json.dumps(completion, indent=2) + '\n')
                    if case != 'baseline':
                        (root / case / 'training.exit-code').write_text(str(code) + '\n')
                        outputs[case].close()
                    ready.append(case)
                except Exception as error:
                    failed[case] = str(error)
                    stages[case] = 'failed'
            update()
            for case in ready:
                try:
                    stages[case] = 'evaluating'
                    update()
                    evaluate(case)
                    evaluated.add(case)
                    stages[case] = 'completed'
                except Exception as error:
                    failed[case] = str(error)
                    stages[case] = 'failed'
                update()
            if len(evaluated | set(failed)) < len(CASES):
                time.sleep(5)
        progress.update(state='failed' if failed else 'completed', failures=failed,
                        finished_at_utc=now(), evaluation_pid=None)
        update()
        (root / 'parallel.exit-code').write_text(str(int(bool(failed))) + '\n')
    except Exception as error:
        progress.update(state='coordinator_failed', error=str(error), failures=failed)
        update()
        raise
    finally:
        for output in outputs.values():
            output.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('launch', 'worker'))
    p.add_argument('--campaign', type=Path, required=True)
    args = p.parse_args()
    root = args.campaign
    release = Path(__file__).resolve().parents[2]
    if args.command == 'worker':
        worker(root, release)
        return
    original = read(root / 'launch.json')
    handoff = read(root / 'parallel-handoff.json')
    assert original['iterations_per_group'] == 500 and original['num_envs'] == 4096
    assert tuple(original['groups']) == CASES
    assert handoff['old_status']['current_case'] == 'baseline'
    assert handoff['old_status']['state'] == 'training'
    for case in CASES[1:]:
        if (root / case).exists() or Path(original['runs'][case]).exists():
            raise ValueError('A queued factor already started; do not launch a duplicate')
    memory = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                         '--format=csv,noheader,nounits'], text=True).strip())
    if memory < 5 * 3300 + 2000:
        raise ValueError('Insufficient memory for five additional4096-env learners and one replay')
    old_pid = handoff['old_coordinator_pid']
    proc = Path(f'/proc/{old_pid}')
    if proc.exists():
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        if 'launch_chuanliantui_recovery_factors.py worker' not in command or str(root) not in command:
            raise ValueError('Old coordinator identity changed')
        if (proc / 'stat').read_text().split(') ', 1)[1][0] != 'T':
            raise ValueError('Old coordinator must be paused before taking over')
    record = dict(release=str(release), campaign=str(root), execution='parallel',
        original_launch='launch.json', source=original['source'], groups=list(CASES),
        iterations_per_group=500, num_envs=4096, seed=42,
        adopted_baseline_pid=handoff['active_child_pid'], old_coordinator_pid=old_pid,
        created_at_utc=now(), baseline_restarted=False, tensorboard_command=original['tensorboard_command'])
    with (root / 'parallel-launch.json').open('x') as output:
        output.write(json.dumps(record, indent=2) + '\n')
    # Terminate only the paused scheduler, preserving its learner child.
    if proc.exists():
        os.kill(old_pid, signal.SIGTERM)
        os.kill(old_pid, signal.SIGCONT)
    with (root / 'parallel-coordinator.log').open('xb') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker', '--campaign', str(root)],
            cwd=release, env=dict(os.environ, TZ='Asia/Shanghai'), stdout=log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    record['coordinator_pid'] = child.pid
    atomic(root / 'parallel-launch.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
