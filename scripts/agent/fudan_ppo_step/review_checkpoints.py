"""冻结百轮存档，评估 200 mm GPU 九例与原生平地切模六例。"""
import argparse
import json
import os
import re
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .launch_training import sha, write
from .review_restart import REGIONAL_CHECKPOINT, native_trial, stairs


def read(path):
    return json.loads(path.read_text())


def native_cases(checkpoint, output, surface='flat'):
    tick = int(checkpoint.stem.split('_')[-1])
    folder = output/f'native-{surface}-{tick}'
    protocol = dict(checkpoint_sha256=sha(checkpoint), walk_sha256=sha(REGIONAL_CHECKPOINT),
                    surface=surface, waits=[0., 1.], seeds=[1, 2, 3],
                    stop_check_from_s=.2, stop_timeout_s=3., optimizer_updates=0)
    if not folder.exists():
        # 六例与索引一起发布。中途退出时下次从该存档重做。
        with TemporaryDirectory(dir=output) as work:
            recording = Path(work)/'recording'
            recording.mkdir()
            rows = []
            for wait in protocol['waits']:
                for seed in protocol['seeds']:
                    rows.append(native_trial(checkpoint, surface, wait, seed,
                        recording/f'wait-{wait:g}-seed-{seed}.npz',
                        stop_check_from_s=.2, stop_timeout_s=3.))
            write(recording/'cases.json', dict(protocol=protocol, cases=rows))
            recording.rename(folder)
    cache = read(folder/'cases.json')
    if cache['protocol'] != protocol:
        raise ValueError(f'原生缓存协议或模型不同，请使用新输出目录 {folder}')
    rows = cache['cases']
    cases = [(r['t_to_up_s'], r['seed']) for r in rows
             if r['iterations'] == tick and r['surface'] == surface]
    if len(rows) != 6 or sorted(cases) != [(w, s) for w in protocol['waits'] for s in protocol['seeds']]:
        raise ValueError(f'原生六例不完整 {folder}')
    for wait, seed in cases:
        with np.load(folder/f'wait-{wait:g}-seed-{seed}.npz') as trace:
            if not np.isfinite(trace['data']).all():
                raise ValueError(f'原生轨迹存在非有限值 {folder}, wait={wait}, seed={seed}')
    return rows


def stair_scores(folder, digest):
    protocol = read(folder/'protocol.json')
    audit = read(folder/'audit/audit.json')
    expected = dict(backend='mujoco_warp_gpu', source_sha256=digest, seconds=20.,
                    command_height_m=.35, high_speed=True, settle_s=0., ramp_s=0.,
                    optimizer_updates=0, friction=[.4, .8, 1.2], max_approach_speed=3.,
                    action_delay_steps=[15]*48, sensor_delay_steps=[15]*48)
    if (any(protocol[k] != v for k, v in expected.items()) or not audit['passed']
            or not audit['trace_hashes_and_metrics_recomputed'] or audit['cases'] != 48):
        raise ValueError(f'GPU 存档、协议或轨迹审计不匹配 {folder}')
    metrics = read(folder/'metrics.json')
    cases = protocol['cases']
    if len(cases) != 48 or len(metrics) != 48 or any(
            any(row[k] != v for k, v in case.items()) for case, row in zip(cases, metrics)):
        raise ValueError(f'GPU 工况与评分不匹配 {folder}')
    target = [r for r in metrics if r['height'] == .2 and r['speed'] == 2.]
    expected_cases = {(a, m) for a in (0., -10., 10.) for m in (.4, .8, 1.2)}
    if (len(target) != 9 or {r['id'] for r in target} !=
            {f'forward_h0.200_v2.00_a{a:+.0f}_mu{m:.2f}' for a, m in expected_cases}):
        raise ValueError(f'200 mm 固定速度九例不完整 {folder}')
    durations = []
    peaks = []
    for row in target:
        if row['task_score']['passed']:
            durations.append(row['valid_duration_s'])
        case = row['id']
        path = folder/'traces'/f'{case}.npz'
        if sha(path) != protocol['trace_sha256'][case]:
            raise ValueError(f'轨迹指纹不符 {path}')
        with np.load(path) as tr:
            # 沿用 v11 审计，避开刚松开前进键时仍以约 2 m/s 行驶的制动瞬间。
            zero = tr['valid'] & (tr['command'][:, :2] == 0).all(1)
            ticks = np.flatnonzero(zero)
            if ticks.size:
                window = slice(ticks[0]+50, ticks[0]+200)
                if zero[window].sum() == 150:
                    peaks.append(float(np.linalg.norm(tr['linear_world'][window, :2], axis=1).max()))
    return dict(passed=len(durations), durations_s=durations, peak_cases=len(peaks),
                median_peak_m_s=float(np.median(peaks)) if peaks else None)


def score_native(rows):
    results = []
    for r in rows:
        first = r['first_start']['sustained_1_9_onset_s'] if r['walk_settled'] else None
        stop = r['stop_observation_s'] if first is not None and r['stopped_before_restart'] else None
        restart = r['restart']['sustained_1_9_onset_s'] if stop is not None else None
        results.append(dict(wait_s=r['t_to_up_s'], seed=r['seed'], walk_settled=r['walk_settled'],
            first_s=first, stop_s=stop, restart_s=restart, peak_vx_m_s=r['first_start']['peak_vx_m_s']))
    return results


def summary(tick, gpu, native):
    cases = score_native(native)
    return dict(iterations=tick, stairs=gpu, flat=cases,
                all_passed=gpu['passed'] == 9 and all(r['restart_s'] is not None for r in cases))


def timing(values, total):
    if not values:
        return f'0/{total}'
    if min(values) == max(values):
        return f'{len(values)}/{total} · {min(values):.2f}'
    return f'{len(values)}/{total} · {min(values):.2f}–{max(values):.2f}'


def table(rows):
    lines = ['用时单位为秒。登台从场景开始计至成功回合结束。平地起步、再起步计至首个连续 0.10 s 达速窗口的起点；'
             '停稳计至首个连续 0.20 s 达标窗口的终点。六例均留在分母中。',
             '', '| 存档 | 登台停车 | 首次起步 | 停稳 | 再起步 | 零速峰值中位数 m/s | 三项全过 |',
             '|---|---|---|---|---|---|---|']
    for row in rows:
        stair, flat = row['stairs'], row['flat']
        cells = [row['iterations'], timing(stair['durations_s'], 9)]
        for key, prereq in [('first_s', None), ('stop_s', 'first_s'), ('restart_s', 'stop_s')]:
            value = timing([r[key] for r in flat if r[key] is not None], 6)
            missing = sum(r[prereq] is None for r in flat) if prereq else 0
            cells.append(value + (f'（{missing}例前提不成立）' if missing else ''))
        peak = stair['median_peak_m_s']
        cells.extend([f'{peak:.3f}（{stair["peak_cases"]}/9）' if peak is not None else '—',
                      '是' if row['all_passed'] else '—'])
        lines.append('| ' + ' | '.join(map(str, cells)) + ' |')
    failed = [(row['iterations'], r) for row in rows for r in row['flat'] if r['first_s'] is None]
    if failed:
        lines.extend(['', '首次起步未通过的工况，峰值取首次 2.50 秒行驶段。', '',
                      '| 存档 | 等待 s | 种子 | 行驶峰值 m/s | 停稳 | 再起步 |',
                      '|---|---|---|---|---|---|'])
        for tick, r in failed:
            peak = r['peak_vx_m_s']
            speed = '—' if peak is None else f'{peak:.4f}'
            lines.append(f"| {tick} | {r['wait_s']:g} | {r['seed']} | {speed} | 前提不成立 | 前提不成立 |")
    return '\n'.join(lines)+'\n'


def run(log_dir, output, reuse_stairs=None):
    checkpoints = sorted((p for p in log_dir.glob('model_*.pt')
                          if re.fullmatch(r'model_\d+', p.stem) and int(p.stem[6:]) % 100 == 0),
                         key=lambda p: int(p.stem[6:]))
    if not checkpoints:
        raise ValueError(f'未找到百轮 model_<轮数>.pt 存档 {log_dir}')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    results = []
    for checkpoint in checkpoints:
        tick = int(checkpoint.stem[6:])
        digest = sha(checkpoint)
        native = native_cases(checkpoint, output)
        folder = output/f'stair-{tick}'
        if not folder.exists():
            prior = reuse_stairs/folder.name if reuse_stairs else None
            if prior and prior.is_dir():
                gpu = stair_scores(prior, digest)
                folder.symlink_to(os.path.relpath(prior.resolve(), output.resolve()), target_is_directory=True)
            else:
                with TemporaryDirectory(dir=output) as work:
                    recording = Path(work)/folder.name
                    stairs(checkpoint, recording, time.time()+1800)
                    gpu = stair_scores(recording, digest)
                    recording.rename(folder)
        else:
            gpu = stair_scores(folder, digest)
        results.append(dict(checkpoint_sha256=digest, **summary(tick, gpu, native)))
    # 仅在全目录完成后发布汇总；中断后可复用已完成的单项。
    with TemporaryDirectory(dir=output) as work:
        scratch = Path(work)
        write(scratch/'checkpoints.json', dict(checkpoints=results, optimizer_updates=0))
        (scratch/'checkpoints.md').write_text(table(results))
        for name in ('checkpoints.json', 'checkpoints.md'):
            (scratch/name).replace(output/name)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-stairs', type=Path, help='已有 stair-<轮数> GPU 审计目录的父目录')
    args = parser.parse_args()
    print(table(run(args.log_dir, args.output, args.reuse_stairs)))
