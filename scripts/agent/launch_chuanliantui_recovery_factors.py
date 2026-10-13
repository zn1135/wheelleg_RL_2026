"""Run the authorized six500-update continuations sequentially, then paired replays."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from ct_mjlab.wheel_balance_assessment import paired_acceptance

CASES = ('baseline', 'recovery_reference', 'skip_righting', 'keep_height', 'keep_contact', 'bearing_reward')


def now():
    return datetime.now(timezone.utc).isoformat()


def status(root, value):
    value['updated_at_utc'] = now()
    tmp = root / 'status.tmp'
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(root / 'status.json')


def analysis(root, case, point, result, source):
    row = dict(case=case, completed_updates=result['completed_updates'],
               num_envs=64, learning=False, poses={}, acceptance=paired_acceptance(result, source))
    reference_path = root / 'recovery_reference' / point / 'evaluation.json'
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
    for pose, value in result['poses'].items():
        original = source['poses'][pose]
        assert value['initial_state_sha256'] == original['initial_state_sha256']
        old, new = set(original['balanced_samples']), set(value['balanced_samples'])
        row['poses'][pose] = dict(balance=value['sustained_last5s']['wheel_balance'],
            height_tracking=value['sustained_last5s']['height_tracking'],
            lost_source_samples=sorted(old-new), gained_source_samples=sorted(new-old),
            mean_height_m=value['mean_last5s_height_m'], nonwheel_n=value['mean_last5s_nonwheel_n'],
            numerical=value['numerical_failures'], sample0=value['sample0'])
        if reference:
            other = reference['poses'][pose]
            assert reference['completed_updates'] == result['completed_updates']
            assert value['initial_state_sha256'] == other['initial_state_sha256']
            row['poses'][pose]['delta_vs_recovery_reference'] = value['sustained_last5s']['wheel_balance'] - other['sustained_last5s']['wheel_balance']
    (root / case / point / 'analysis.json').write_text(json.dumps(row, indent=2) + '\n')
    destination = root / 'analysis.md'
    lines = ['# 恢复单项对照训练结果', '',
        '各组独立从2000轮完整模型起步，4096环境、seed42、追加500轮；101与500次更新后分别配对回放。',
        '回放普通/地面各64份、20秒、零速度、目标0.30m、无学习。相同初态，双轮承重与近水平判据不限制腿角。', '',
        '原成功样本有任何一个失效即判退化、平衡保留验收不通过；新增成功不能抵消丢失。保留原能力不等于所有样本或高度达标。', '',
        '| 组 | 累计轮数 | 普通平衡 | 地面平衡 | 地面高度达标 | 丢失原地面样本 | 与同轮数recovery参照之差 | 判定 |',
        '|---|---:|---:|---:|---:|---|---:|---|']
    for name in CASES:
        for label in ('early101', 'final500'):
            path = root / name / label / 'analysis.json'
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            a, b = record['poses']['normal'], record['poses']['ground']
            delta = b.get('delta_vs_recovery_reference', '待同轮数参照')
            acceptance = record.get('acceptance')
            verdict = ({'regressed_rejected': '退化，验收不通过', 'retained': '保留原平衡能力',
                        'numerical_failure_rejected': '数值失败，验收不通过'}[acceptance['verdict']]
                       if acceptance else '历史汇总，未按新标准验收')
            lines.append(f"| {name} | {record['completed_updates']} | {a['balance']}/64 | {b['balance']}/64 | {b['height_tracking']}/64 | {b['lost_source_samples']} | {delta} | {verdict} |")
    lines += ['', '训练reward因目标不同不能直接排名。单种seed与有限配对初态只提供本批单项影响证据；500轮趋势不等同长期稳定或部署验收。',
              'recovery_reference复现新课程；skip_righting只改变初始阶段；keep_height只恢复前两阶段高度奖励；keep_contact只恢复扶正阶段接触惩罚；bearing_reward只改变轮支撑奖励公式。']
    destination.write_text('\n'.join(lines) + '\n')


def worker(args, release):
    root = args.campaign
    record = json.loads((root / 'launch.json').read_text())
    env = dict(os.environ, PYTHONPATH=str(release / 'mjlab_training/src'),
               PYTHONUNBUFFERED='1', MUJOCO_GL='egl', TZ='Asia/Shanghai', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('WHEELLEG_HUANANHU_INIT_CHECKPOINT', None)
    progress = dict(state='source_evaluation', worker_pid=os.getpid(), current_case=None,
                    groups={case: 'queued' for case in CASES}, started_at_utc=now())
    status(root, progress)

    def execute(command, log, phase):
        with log.open('x') as output:
            child = subprocess.Popen(command, cwd=release, env=env, stdout=output, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
            progress.update(state=phase, child_pid=child.pid, command=command)
            status(root, progress)
            code = child.wait()
        log.with_suffix('.exit-code').write_text(str(code) + '\n')
        if code:
            raise RuntimeError(f'{phase} failed with exit{code}; see {log}')

    def evaluate(checkpoint, output):
        command = [sys.executable, '-u', str(release / 'scripts/agent/eval_chuanliantui_recovery_factors.py'),
            '--checkpoint', str(checkpoint), '--output', str(output), '--initial-ground', record['initial_ground']]
        normal = root / 'source2000' / 'initial-normal.npz'
        if normal.exists():
            command += ['--initial-normal', str(normal)]
        execute(command, output.parent / (output.name + '.log'), 'evaluating')
        result = json.loads((output / 'evaluation.json').read_text())
        command = [sys.executable, str(release / 'scripts/agent/render_chuanliantui_ablation_replay.py'),
                   '--input', str(output), '--groups', result['case'], '--width', '640']
        execute(command, output.parent / (output.name + '-render.log'), 'rendering')
        return result

    try:
        source = evaluate(Path(record['source']), root / 'source2000')
        if (source['poses']['ground']['sustained_last5s']['wheel_balance'] != 58
                or source['poses']['normal']['sustained_last5s']['wheel_balance'] != 64):
            raise RuntimeError('Canonical2000 source behavior did not reproduce; no PPO started')
        for case in CASES:
            progress['current_case'] = case
            progress['groups'][case] = 'training'
            folder = root / case
            folder.mkdir(exist_ok=False)
            run = Path(record['runs'][case])
            command = [sys.executable, '-u', str(release / 'scripts/agent/run_chuanliantui_recovery_factors.py'),
                '--case', case, '--resume', record['source'], '--run-dir', str(run),
                '--iterations', '500', '--num-envs', '4096', '--seed', '42']
            execute(command, folder / 'training.log', 'training')
            for label, checkpoint, expected in (('early101', 'model_2100.pt', 2101), ('final500', 'model_2499.pt', 2500)):
                result = evaluate(run / checkpoint, folder / label)
                if result['completed_updates'] != expected:
                    raise RuntimeError('Unexpected update count in factor comparison')
                analysis(root, case, label, result, source)
                if case == 'recovery_reference':
                    old = root / 'baseline' / label / 'evaluation.json'
                    analysis(root, 'baseline', label, json.loads(old.read_text()), source)
            progress['groups'][case] = 'completed'
            status(root, progress)
        progress.update(state='completed', child_pid=None, finished_at_utc=now())
        status(root, progress)
        (root / 'exit-code.txt').write_text('0\n')
    except Exception as error:
        progress.update(state='failed', error=str(error), finished_at_utc=now())
        if progress['current_case']:
            progress['groups'][progress['current_case']] = 'failed'
        status(root, progress)
        (root / 'exit-code.txt').write_text('1\n')
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('launch', 'worker'))
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--initial-ground', type=Path)
    p.add_argument('--preflight', type=Path)
    args = p.parse_args()
    release = Path(__file__).resolve().parents[2]
    if args.command == 'worker':
        worker(args, release)
        return
    if not all((args.source, args.initial_ground, args.preflight)):
        p.error('launch requires source, canonical ground states and passed preflight')
    for name in ('run_chuanliantui_recovery_factors.py', 'eval_chuanliantui_recovery_factors.py',
                 'render_chuanliantui_ablation_replay.py'):
        if not (release / 'scripts/agent' / name).is_file():
            raise ValueError(f'Incomplete campaign release: missing {name}')
    check = json.loads((args.preflight / 'preflight.json').read_text())
    if not check.get('passed') or tuple(check['cases']) != CASES:
        raise ValueError('All six native preflights must pass before launch')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    memory, utilization = map(int, gpu.strip().split(','))
    if memory > 1000 or utilization > 10:
        raise RuntimeError('GPU is occupied; defer this sequential campaign')
    args.campaign.mkdir(parents=True, exist_ok=False)
    prefix = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_root = (release / 'logs/mjlab/chuanliantui' / f'{prefix}_recovery_factors').resolve()
    runs = {case: str(run_root / case) for case in CASES}
    record = dict(release=str(release), source=str(args.source.resolve()),
        initial_ground=str(args.initial_ground.resolve()), preflight=str(args.preflight.resolve()),
        num_envs=4096, seed=42, iterations_per_group=500, initial_updates=2000, target_updates=2500,
        groups=list(CASES), runs=runs, created_at_utc=now(), execution='sequential; no extra training after six groups',
        tensorboard_command=f'{Path(sys.executable).with_name("tensorboard")} --logdir={run_root} --port=8080')
    pending = args.campaign / 'launch.tmp'
    pending.write_text(json.dumps(record, indent=2) + '\n')
    pending.replace(args.campaign / 'launch.json')
    with (args.campaign / 'coordinator.log').open('xb') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
            '--campaign', str(args.campaign)], cwd=release, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, env=dict(os.environ, TZ='Asia/Shanghai'))
    record['coordinator_pid'] = child.pid
    pending.write_text(json.dumps(record, indent=2) + '\n')
    pending.replace(args.campaign / 'launch.json')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
