#!/usr/bin/env python3
"""Offline H7 timing distributions from hashed mirror-v2/H7OUT001 recordings.

Never opens a device or runs a policy/physics. All durations use board clocks;
host receive times are retained only as evidence locations. See report.md for
the distinction between data age, software duration and unmeasured motor delay.
"""
from __future__ import annotations

import argparse
import binascii
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'sim2sim'))
import real2sim_huananhu as mirror
import h7_output_capture as capture

U32 = 2**32
DEFAULT_SESSIONS = (
    'artifacts/model6848-startup-review-20261022-122136/session',
    'artifacts/model6848-real-rl-20261021/session-01',
    'artifacts/model6848-real-rl-20261021/session-02',
)
PHASES = ('idle', 'hold', 'recover', 'normal', 'failed')
SOURCE_FILES = (
    'imcalib/task/vofa_debug.c', 'imcalib/task/task_policy.c',
    'imcalib/task/task_actuation.c', 'imcalib/Algorithm/rl_policy.c',
    'imcalib/bsp/mono_ns.c', 'imcalib/bsp/output_capture.h',
)
LIMITS = [
    '毫秒tick差值量化误差小于±1ms；单个毫秒数据龄是向下取整值，可能饱和于60000ms。',
    '缓存到发布时间差为0ms只表示小于1ms；毫秒策略周期的标准差为0不能证明亚毫秒抖动为0。',
    'ns字段是板端软件时间戳，精度由DWT时基和插桩位置决定，不代表纳秒测量准确度。',
    '缓存到发布包含样本等待、观测构造、推理与发布，不能分别拆出调度/观测/PD耗时。',
    '100Hz镜像仅保存最新策略/输出；未观察到的推理、输出拍和失败推理耗时无法补齐。',
    '动作发布到输出是该已观察输出使用动作时的数据龄，不是动作首次应用延迟。',
    'CONTROL覆盖门控/驱动下发/镜像发布，起点在PD计算之后，不是整个执行任务耗时。',
    'CAN记录只覆盖DJI的HAL提交调用，HAL成功不证明电气发送、送达或电机生效。',
    'CAN唯一包含区间仅说明时间嵌套，不证明调用来源；dispatch_to_hal_begin_us仅为区间内起点偏移。',
    '恢复阶段和遥控投入来自100Hz镜像；高频记录只在前后镜像状态一致时借用该上下文。',
    'startup_*是投入后首1秒的分析分组，非实际测得的固定启动等待时间；跨状态的镜像区间另列。',
    '不将主机接收时间与H7时间相减；未由板端时间戳配对的源头年龄不相加。',
    '当前区间按首末镜像的毫秒板端时间裁剪，边界精度为1ms；全文件记录保存在full-output中。',
    '固件manifest只是预期构建清单，不是板端启动证明；无manifest会话保留该身份缺口。',
    '绳索/外力使本记录不适合确定电机物理响应延迟；本报告不作行为验收。',
]
MISSING = [
    ('传感器内部采样与滤波', '传感器实际采样时刻、滤波输入输出或同步硬件边沿'),
    ('传感器链路到H7解析', '源头采样时间与硬件接收/解析起止的同钟配对'),
    ('任务唤醒、观测构造、PD各自耗时', '每个环节独立起止测点及逐拍关联序号'),
    ('四腿逐路CAN提交', '每路DM发送起止和该拍控制/动作序号'),
    ('CAN排队、仲裁、电气发送和送达', 'CAN Tx完成事件或带共同时间基准的总线分析记录'),
    ('电机收命令到电流/力矩生效', '电机内部时间戳或同步独立电流/轴端力矩测量'),
    ('完整闭环端到端及其抖动', '源头采样→对应动作→实际力矩的贯穿编号和同步测点'),
]


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest()}


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w') as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def lift_ms(tick, reference):
    """Lift a uint32 ms tick near an already anchored board time (natural wrap)."""
    return reference + ((int(tick) - reference + U32 // 2) % U32 - U32 // 2)


def state_key(frame):
    return tuple(int(frame[i]) for i in (152, 153, 154, 156, 157, 2))


def group(row):
    return f"strategy={row['strategy']};engaged={row['engaged']};fault={row['fault']};stage={row['stage']}"


def read_recording(session):
    parser, clock, decoder = mirror.FrameParser(), mirror.FrameClock(), capture.CaptureParser()
    frames, records, anomalies = [], [], []
    epoch = -1
    mirror_span = record_span = 0
    last_record = last_packet = last_frame = None
    previous_kind = {}
    for event_index, event in enumerate(mirror.iter_session_events(session)):
        kind = event['type']
        if kind != 'data':
            parser.buffer.clear()
            decoder.buffer.clear()
            if kind == 'connected':
                epoch += 1
                mirror_span += 1
                record_span += 1
                clock = mirror.FrameClock()
                last_record = last_packet = last_frame = None
                previous_kind = {}
            continue
        chunk = bytes.fromhex(event['data_hex'])
        old_rejected = parser.rejected
        for frame in parser.feed(chunk):
            restarts = clock.restarts
            if not clock.accept(frame):
                continue
            if clock.restarts != restarts:
                raise ValueError('board reset inside connection: split the recording at the reset before analysis')
            tick = mirror.read_u32(frame[8:10])
            sequence = mirror.read_u32(frame[6:8])
            if last_frame is not None:
                delta = (sequence - last_frame['sequence']) % U32
                if delta != 1 or parser.rejected != old_rejected:
                    mirror_span += 1
                lifted = last_frame['ms'] + ((tick - last_frame['tick']) % U32)
            else:
                lifted = tick
            item = dict(epoch=epoch, span=mirror_span, ms=lifted, tick=tick,
                        sequence=sequence, event=event_index, receive_ns=event['t_ns'], frame=frame)
            frames.append(item)
            last_frame = item
        rejected = decoder.rejected
        for packet in decoder.feed(chunk):
            broken = decoder.rejected != rejected
            if last_packet is not None:
                broken |= (packet['sequence'] - last_packet['sequence']) % U32 != 1
                broken |= packet['dropped'] != last_packet['dropped']
            if broken:
                record_span += 1
                previous_kind = {}
            for row in packet['records']:
                if row[3] < row[2]:
                    raise ValueError('record end precedes start')
                delta = (row[0] - last_record[0]) % U32 if last_record is not None else 1
                backwards = row[1] in previous_kind and row[2] < previous_kind[row[1]]
                if delta != 1 or backwards:
                    anomalies.append(dict(epoch=epoch, after=last_record[0] if last_record else None,
                                          before=row[0], sequence_delta=delta, same_kind_clock_backwards=backwards))
                    record_span += 1
                    previous_kind = {}
                records.append(dict(epoch=epoch, span=record_span, event=event_index,
                                    receive_ns=event['t_ns'], raw=row))
                previous_kind[row[1]] = row[2]
                last_record = row
            last_packet = packet
    if not frames or any(len(item['frame']) != 160 for item in frames):
        raise ValueError('requires mirror-v2 frames')
    # Most raw records belong to the current uptime; stale startup queue is a minority.
    for e in sorted({f['epoch'] for f in frames}):
        fs = [f for f in frames if f['epoch'] == e]
        rs = [r['raw'][2] // 1000000 for r in records if r['epoch'] == e]
        offset = int(np.median([(t - fs[0]['ms'] + U32 // 2) // U32 for t in rs])) * U32 if rs else 0
        for f in fs:
            f['ms'] += offset
    return frames, records, anomalies, parser.rejected


def summarize(values, evidence):
    a = np.asarray(values, dtype=float)
    maximum = int(np.argmax(a))
    return dict(count=len(a), min_us=float(a.min()), mean_us=float(a.mean()),
                median_us=float(np.median(a)), std_us=float(a.std(ddof=0)),
                p95_us=float(np.percentile(a, 95)), p99_us=float(np.percentile(a, 99)),
                max_us=float(a[maximum]), peak_to_peak_us=float(np.ptp(a)),
                max_sample=evidence[maximum])


def add_periods(rows, stamp, sequence=None, partition=(), nominal=None):
    previous = {}
    for row in rows:
        key = (row['epoch'], row['span'], *(row.get(k) for k in partition))
        prev = previous.get(key)
        row['period_us'] = None
        row['previous_sequence'] = None
        row['previous_board_ns'] = None
        if prev is not None and row['group'] == prev['group']:
            continuous = sequence is None or (row[sequence] - prev[sequence]) % U32 == 1
            elapsed = row[stamp] - prev[stamp]
            if continuous and elapsed > 0:
                row['period_us'] = elapsed / 1000
                row['previous_sequence'] = prev['sequence']
                row['previous_board_ns'] = prev[stamp]
        row['nominal_period_us'] = nominal(row) if nominal else None
        row['period_error_us'] = (row['period_us'] - row['nominal_period_us']
                                  if row['period_us'] is not None and row['nominal_period_us'] is not None else None)
        previous[key] = row


def analyze(session, out):
    out.mkdir()
    meta, xml, config = mirror.load_session(session)
    # Preserve the standard whole-file completeness result, including historical gaps.
    quality = capture.analyze_session(session, out / 'full-output')
    frames, records, anomalies, rejected = read_recording(session)
    frame_by_epoch = defaultdict(list)
    for f in frames:
        frame_by_epoch[f['epoch']].append(f)
    times = {e: [f['ms'] for f in fs] for e, fs in frame_by_epoch.items()}
    first_ms = frames[0]['ms']
    engagement_starts = defaultdict(list)
    previous_engaged = {}
    for f in frames:
        a = f['frame']
        engaged = int(a[152]) == 0 and bool(int(a[153]) & 4)
        if engaged and not previous_engaged.get(f['epoch'], False):
            engagement_starts[f['epoch']].append(f['ms'])
        previous_engaged[f['epoch']] = engaged

    def context(e, ms):
        fs, ts = frame_by_epoch[e], times[e]
        before = bisect_right(ts, ms) - 1
        after = bisect_left(ts, ms)
        if before < 0 or after >= len(fs):
            return dict(strategy=-1, engaged=-1, fault=-1, stage='unobserved')
        a, b = fs[before], fs[after]
        f = a['frame']
        same = a['span'] == b['span'] and state_key(f) == state_key(b['frame']) and b['ms'] - a['ms'] <= 20
        stage = PHASES[int(f[154])] if int(f[154]) < len(PHASES) else 'unknown'
        if not same:
            stage = 'transition_or_mirror_gap'
        else:
            starts = engagement_starts[e]
            j = bisect_right(starts, ms) - 1
            if j >= 0 and 0 <= ms - starts[j] < 1000 and int(f[152]) == 0 and int(f[153]) & 4:
                stage = 'startup_' + stage
        return dict(strategy=int(f[152]), engaged=int(int(f[152]) == 0 and bool(int(f[153]) & 4)),
                    fault=int(f[2]), stage=stage)

    def base(e, span, stamp, receive, event, sequence):
        row = dict(epoch=e, span=span, board_ns=stamp, board_s=(stamp / 1000000 - first_ms) / 1000,
                   receive_s=receive / 1e9, event=event, sequence=sequence)
        row.update(context(e, stamp // 1000000))
        row['group'] = group(row)
        return row

    feedback, policies, outputs, controls, cans = [], [], [], [], []
    policy_map, output_map = {}, {}
    duplicate_policy = duplicate_output = 0
    skipped = Counter()
    for f in frames:
        a, e, ms = f['frame'], f['epoch'], f['ms']
        common = base(e, f['span'], ms * 1000000, f['receive_ns'], f['event'], f['sequence'])
        for name, index in [('imu', 144)] + [(n, 138 + i) for i, n in enumerate(mirror.MOTORS)]:
            age = float(a[index])
            common[f'{name}_age_us'] = age * 1000 if np.isfinite(age) and age >= 0 else None
            common[f'{name}_age_saturated'] = int(age >= 60000)
        feedback.append(common)
        if int(a[47]) & 1:
            seq = mirror.read_u32(a[48:50])
            key = (e, seq)
            payload = a[48:103].tobytes()
            if key in policy_map:
                if policy_map[key]['payload'] != payload:
                    raise ValueError(f'policy sequence reused with different content: {key}')
                duplicate_policy += 1
            else:
                publish = lift_ms(mirror.read_u32(a[52:54]), ms)
                sample = lift_ms(mirror.read_u32(a[50:52]), publish)
                row = base(e, f['span'], publish * 1000000, f['receive_ns'], f['event'], seq)
                row.update(sample_ms=sample, publish_ms=publish, snapshot_ms=ms,
                           cache_to_publish_us=(publish - sample) * 1000, inference_us=float(a[102]),
                           reset_count=mirror.read_u32(a[156:158]))
                row['in_observed_window'] = times[e][0] <= sample <= publish <= times[e][-1] + 1
                policy_map[key] = dict(row=row, payload=payload, action=a[96:102].copy())
                if row['in_observed_window']:
                    policies.append(row)
                else:
                    skipped['policy_outside_observed_window'] += 1
        if int(a[47]) & 2:
            seq = mirror.read_u32(a[103:105])
            key = (e, seq)
            payload = a[103:138].tobytes()
            if key in output_map:
                if output_map[key]['payload'] != payload:
                    raise ValueError(f'output sequence reused with different content: {key}')
                duplicate_output += 1
                continue
            stamp = lift_ms(mirror.read_u32(a[105:107]), ms)
            row = base(e, f['span'], stamp * 1000000, f['receive_ns'], f['event'], seq)
            row.update(output_flags=int(a[109]), used_policy_sequence=mirror.read_u32(a[107:109]),
                       action_age_us=None, pairing='not_valid_pd', paired_publish_ms=None)
            output_map[key] = dict(row=row, payload=payload, action=a[116:122].copy())
            if times[e][0] <= stamp <= times[e][-1] + 1:
                outputs.append(row)
            else:
                skipped['output_outside_observed_window'] += 1
    for row in outputs:
        if not row['output_flags'] & 1:
            continue
        pair = policy_map.get((row['epoch'], row['used_policy_sequence']))
        if pair is None or not pair['row']['in_observed_window']:
            row['pairing'] = 'policy_not_observed'
        elif not np.array_equal(output_map[(row['epoch'], row['sequence'])]['action'], pair['action']):
            row['pairing'] = 'action_mismatch'
        elif pair['row']['publish_ms'] > row['board_ns'] // 1000000:
            row['pairing'] = 'time_inconsistent'
        else:
            row['pairing'] = 'matched'
            row['paired_publish_ms'] = pair['row']['publish_ms']
            row['action_age_us'] = row['board_ns'] / 1000 - pair['row']['publish_ms'] * 1000
    current_record_spans = set()
    for record in records:
        r, e = record['raw'], record['epoch']
        if e not in times or not times[e][0] * 1000000 <= r[2] < (times[e][-1] + 1) * 1000000:
            skipped['raw_outside_observed_window'] += 1
            continue
        row = base(e, record['span'], r[2], record['receive_ns'], record['event'], r[0])
        row.update(end_ns=r[3], duration_us=(r[3] - r[2]) / 1000, identifier=r[4])
        current_record_spans.add((e, record['span']))
        if r[1] == 1:
            row.update(strategy=r[21], fault=r[5], output_flags=r[20], output_sequence=r[4],
                       pd_valid=int(bool(r[20] & 1)), drive=int(bool(r[20] & 32)),
                       nonzero=int(any(r[6:12])),
                       left_wheel_age_us=r[14] if r[14] != U32 - 1 else None,
                       right_wheel_age_us=r[15] if r[15] != U32 - 1 else None)
            row['group'] = group(row) + f";pd={row['pd_valid']};drive={row['drive']}"
            controls.append(row)
        else:
            row.update(hal_status=r[20], containing_control=None, containment='unmatched')
            cans.append(row)
    # Pair HAL calls only to one enclosing recorded control; overlaps remain ambiguous.
    control_index = defaultdict(list)
    for row in controls:
        control_index[(row['epoch'], row['span'])].append(row)
    for values in control_index.values():
        values.sort(key=lambda r: r['board_ns'])
    starts = {k: [r['board_ns'] for r in v] for k, v in control_index.items()}
    max_duration = {k: max(r['end_ns'] - r['board_ns'] for r in v) for k, v in control_index.items()}
    for row in cans:
        key = row['epoch'], row['span']
        candidates = control_index.get(key, [])
        stamps = starts.get(key, [])
        lo = bisect_left(stamps, row['board_ns'] - max_duration.get(key, 0))
        hi = bisect_right(stamps, row['board_ns'])
        matches = [c for c in candidates[lo:hi] if row['end_ns'] <= c['end_ns']]
        if len(matches) == 1:
            c = matches[0]
            row.update(containing_control=c['sequence'], containment='unique',
                       control_output_sequence=c['output_sequence'],
                       dispatch_to_hal_begin_us=(row['board_ns'] - c['board_ns']) / 1000)
            for name in ('strategy', 'fault', 'engaged', 'stage', 'group'):
                row[name] = c[name]
        elif matches:
            row['containment'] = 'ambiguous'
        if row['containment'] != 'unique':
            row.update(strategy=-1, engaged=-1, fault=-1, stage='can_without_unique_control',
                       group='can_without_unique_control', dispatch_to_hal_begin_us=None)
    policies.sort(key=lambda r: (r['epoch'], r['board_ns']))
    outputs.sort(key=lambda r: (r['epoch'], r['board_ns']))
    add_periods(policies, 'board_ns', 'sequence', nominal=lambda r: 20000 if r['strategy'] == 0 else None)
    add_periods(controls, 'board_ns', 'output_sequence',
                nominal=lambda r: 2000 if r['strategy'] == 0 and r['pd_valid'] and r['drive'] else None)
    add_periods(cans, 'board_ns', partition=('identifier',), nominal=None)
    nonzero_previous = {}
    for row in controls:
        row['nonzero_request_interval_us'] = None
        row['previous_nonzero_sequence'] = None
        key = row['epoch'], row['span']
        if row['nonzero']:
            previous = nonzero_previous.get(key)
            if previous:
                row['nonzero_request_interval_us'] = (row['board_ns'] - previous['board_ns']) / 1000
                row['previous_nonzero_sequence'] = previous['sequence']
            nonzero_previous[key] = row

    tables = dict(feedback=feedback, policy=policies, output=outputs, control=controls, can=cans)
    metrics = dict(feedback=[k for k in feedback[0] if k.endswith('_age_us')],
                   policy=['cache_to_publish_us', 'inference_us', 'period_us', 'period_error_us'],
                   output=['action_age_us'],
                   control=['duration_us', 'period_us', 'period_error_us', 'left_wheel_age_us',
                            'right_wheel_age_us', 'nonzero_request_interval_us'],
                   can=['duration_us', 'period_us', 'dispatch_to_hal_begin_us'])
    stats, by_segment = [], []
    segment_rows, reconciliation = [], {}
    for name, rows in tables.items():
        buckets, segment_buckets = defaultdict(list), defaultdict(list)
        segment = 0
        last_key = None
        for index, row in enumerate(rows):
            key = row['epoch'], row['span'], row['group']
            if key != last_key:
                segment += 1
                segment_rows.append(dict(table=name, segment=segment, epoch=row['epoch'], span=row['span'],
                                         group=row['group'], first_sequence=row['sequence'],
                                         first_board_ns=row['board_ns'], last_sequence=row['sequence'],
                                         last_board_ns=row['board_ns'], count=0))
            segment_rows[-1].update(last_sequence=row['sequence'], last_board_ns=row['board_ns'],
                                     count=segment_rows[-1]['count'] + 1)
            last_key = key
            row['segment'] = segment
            for metric in metrics[name]:
                value = row.get(metric)
                if value is not None:
                    if not np.isfinite(value):
                        raise ValueError(f'nonfinite timing value: {name}/{metric}')
                    buckets[(row['group'], metric)].append((value, index))
                    segment_buckets[(segment, row['group'], metric)].append((value, index))
        for (label, metric), items in buckets.items():
            stats.append(dict(table=name, group=label, metric=metric,
                              **summarize([v for v, _ in items],
                                          [dict(csv=f'{name}.csv', line=i + 2, sequence=rows[i]['sequence'],
                                                board_ns=rows[i]['board_ns'], event=rows[i]['event']) for _, i in items])))
        for (seg, label, metric), items in segment_buckets.items():
            by_segment.append(dict(table=name, segment=seg, group=label, metric=metric,
                                   **summarize([v for v, _ in items], [i + 2 for _, i in items])))
        reconciliation[name] = {metric: dict(total_rows=len(rows), measured=sum(row.get(metric) is not None for row in rows),
                                             unavailable=sum(row.get(metric) is None for row in rows)) for metric in metrics[name]}
        write_csv(out / f'{name}.csv', rows)
    write_csv(out / 'statistics.csv', stats)
    write_csv(out / 'segments.csv', segment_rows)
    save(out / 'segment-statistics.json', by_segment)
    save(out / 'statistics.json', stats)
    plot_tables(out, tables, metrics)
    verification = verify_extremes(stats, tables, frames, records)
    expected_sources = (meta.get('firmware_manifest') or {}).get('expected_build', {}).get('input_sources_sha256', {})
    source_comparison = []
    for relative in SOURCE_FILES:
        source = identity(ROOT / 'deployment/H7_RL_ros' / relative)
        expected = expected_sources.get('deployment/H7_RL_ros/' + relative)
        source.update(expected_sha256=expected, matches_expected=source['sha256'] == expected if expected else None)
        source_comparison.append(source)
    report = dict(session=identity(session / 'meta.json'), stream=identity(session / 'stream.jsonl'),
                  recorded_started_utc=meta['started_utc'], recorded_model=meta['model'],
                  recorded_machine_config=meta['machine_config'], validated_model=identity(xml),
                  validated_machine_config=identity(config), firmware_expectation=meta.get('firmware_manifest'),
                  standard_full_file_complete=quality['complete_observed_span'],
                  standard_full_file_report='full-output/report.json', current_record_spans=len(current_record_spans),
                  raw_discontinuities=anomalies, mirror_scan_rejected_candidates=rejected,
                  note='scan candidates are not a count of definitely corrupted full frames',
                  table_counts={k: len(v) for k, v in tables.items()},
                  duplicate_policy_snapshots=duplicate_policy, duplicate_output_snapshots=duplicate_output,
                  skipped=dict(skipped), policy_pairing=dict(Counter(r['pairing'] for r in outputs)),
                  can_containment=dict(Counter(r['containment'] for r in cans)),
                  can_hal_status=dict(Counter(str(r['hal_status']) for r in cans)), reconciliation=reconciliation,
                  statistics=stats, verification=verification, source_comparison=source_comparison,
                  limitations=LIMITS, missing_measurements=MISSING)
    save(out / 'report.json', report)
    return report


def verify_extremes(stats, tables, frames, records):
    """Cross-check reported maxima against decoded source fields, not plot data."""
    raw = {(r['epoch'], r['raw'][0]): r['raw'] for r in records}
    frame_lookup = {f['event']: [] for f in frames}
    for f in frames:
        frame_lookup[f['event']].append(f)
    checked = 0
    for stat in stats:
        name, metric = stat['table'], stat['metric']
        row = tables[name][stat['max_sample']['line'] - 2]
        value = row[metric]
        if metric == 'period_us':
            expected = (row['board_ns'] - row['previous_board_ns']) / 1000
        elif metric == 'period_error_us':
            expected = (row['board_ns'] - row['previous_board_ns']) / 1000 - row['nominal_period_us']
        elif name in ('control', 'can') and metric == 'duration_us':
            r = raw[(row['epoch'], row['sequence'])]
            expected = (r[3] - r[2]) / 1000
        elif name == 'control' and metric in ('left_wheel_age_us', 'right_wheel_age_us'):
            r = raw[(row['epoch'], row['sequence'])]
            expected = r[14 if metric.startswith('left') else 15]
        elif name == 'policy' and metric in ('inference_us', 'cache_to_publish_us'):
            f = next(f['frame'] for f in frame_lookup[row['event']]
                     if mirror.read_u32(f['frame'][48:50]) == row['sequence'])
            expected = float(f[102]) if metric == 'inference_us' else (
                (mirror.read_u32(f[52:54]) - mirror.read_u32(f[50:52])) % U32) * 1000
        elif name == 'feedback':
            f = next(f['frame'] for f in frame_lookup[row['event']] if f['sequence'] == row['sequence'])
            index = 144 if metric == 'imu_age_us' else 138 + mirror.MOTORS.index(metric.removesuffix('_age_us'))
            expected = float(f[index]) * 1000
        elif name == 'output' and metric == 'action_age_us':
            f = next(f['frame'] for f in frame_lookup[row['event']]
                     if mirror.read_u32(f['frame'][103:105]) == row['sequence'])
            source_policy = next(p for p in tables['policy'] if p['epoch'] == row['epoch']
                                 and p['sequence'] == row['used_policy_sequence'])
            expected = ((mirror.read_u32(f[105:107]) - source_policy['publish_ms']) % U32) * 1000
        elif metric == 'dispatch_to_hal_begin_us':
            c = raw[(row['epoch'], row['containing_control'])]
            expected = (raw[(row['epoch'], row['sequence'])][2] - c[2]) / 1000
        elif metric == 'nonzero_request_interval_us':
            previous = raw[(row['epoch'], row['previous_nonzero_sequence'])]
            expected = (raw[(row['epoch'], row['sequence'])][2] - previous[2]) / 1000
        else:
            continue
        if not np.isclose(value, expected, rtol=0, atol=1e-6):
            raise AssertionError(f'extreme differs from source: {name}/{metric}')
        checked += 1
    for name, rows in tables.items():
        for metric in {s['metric'] for s in stats if s['table'] == name}:
            observed = sum(r.get(metric) is not None for r in rows)
            grouped = sum(s['count'] for s in stats if s['table'] == name and s['metric'] == metric)
            if observed != grouped:
                raise AssertionError('grouped sample counts differ')
    return dict(group_counts_reconciled=True, extrema_checked_against_source=checked,
                scope='all grouped maxima: source ages/durations, policy/output sequence pairing, period endpoints')


def plot_tables(out, tables, metrics):
    selections = [('policy', 'inference_us', 'Network call'), ('policy', 'cache_to_publish_us', 'Cache to publish'),
                  ('output', 'action_age_us', 'Observed action age'), ('control', 'duration_us', 'Control dispatch'),
                  ('control', 'period_us', 'Control period'), ('can', 'duration_us', 'CAN HAL call')]
    fig, axes = plt.subplots(6, 2, figsize=(14, 17), constrained_layout=True)
    for (name, metric, title), (trace, distribution) in zip(selections, axes):
        rows = [r for r in tables[name] if r.get(metric) is not None]
        labels = sorted({r['group'] for r in rows})
        for j, label in enumerate(labels):
            selected = [r for r in rows if r['group'] == label]
            x = np.asarray([r['board_s'] for r in selected])
            y = np.asarray([r[metric] for r in selected])
            color = plt.get_cmap('tab20')(j % 20)
            trace.scatter(x, y, s=2, alpha=.45, color=color)
            distribution.hist(y, bins=50, histtype='step', color=color, label=label)
        trace.set(xlabel='H7 time from first mirror (s)', ylabel='us', title=title)
        distribution.set(xlabel='us', ylabel='sample count', title='Distribution by state')
        trace.grid(alpha=.2)
        if rows and max(r[metric] for r in rows) > 10000:
            trace.set_yscale('symlog', linthresh=1000)
            distribution.set_xscale('symlog', linthresh=1000)
        distribution.set_yscale('log')
    fig.savefig(out / 'timing.png', dpi=130)
    plt.close(fig)
    # One trace and distribution for every metric, including periods and signed jitter.
    for name, keys in metrics.items():
        if name == 'feedback':
            continue
        fig, axes = plt.subplots(len(keys), 2, figsize=(13, 3 * len(keys)), squeeze=False, constrained_layout=True)
        for key, (trace, hist) in zip(keys, axes):
            plotted = []
            for active in (False, True):
                rows = [r for r in tables[name] if r.get(key) is not None
                        and (r['strategy'] == 0 and r['engaged'] == 1 and r['fault'] == 0) == active]
                values = [r[key] for r in rows]
                plotted.extend(values)
                label = 'RL engaged / fault=0' if active else 'other / transition / fault'
                trace.scatter([r['board_s'] for r in rows], values, s=1, label=label)
                hist.hist(values, bins=80, histtype='step', label=label)
            trace.set(xlabel='H7 time from first mirror (s)', ylabel='us', title=key)
            hist.set(xlabel='us', ylabel='sample count')
            # Keep narrow microsecond distributions on readable linear ticks.
            typical = max(float(np.median(np.abs(plotted))), 1) if plotted else 1
            if plotted and max(np.abs(plotted)) > 50 * typical:
                trace.set_yscale('symlog', linthresh=max(typical, 10))
                hist.set_xscale('symlog', linthresh=max(typical, 10))
            hist.set_yscale('log')
            trace.legend(fontsize=6)
        fig.savefig(out / f'{name}-all-metrics.png', dpi=110)
        plt.close(fig)
    # Feedback ages and wheel ages have different sampling sites; show them separately.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for name, keys, pair in (
        ('feedback', ['imu_age_us'] + [f'{n}_age_us' for n in mirror.MOTORS], axes[0]),
        ('control', ['left_wheel_age_us', 'right_wheel_age_us'], axes[1]),
    ):
        for key in keys:
            rows = [r for r in tables[name] if r.get(key) is not None]
            pair[0].scatter([r['board_s'] for r in rows], [r[key] for r in rows], s=1, label=key)
            pair[1].hist([r[key] for r in rows], bins=50, histtype='step', label=key)
        pair[0].set(xlabel='H7 time from first mirror (s)', ylabel='age (us)', title=f'{name} data age')
        pair[1].set(xlabel='age (us)', ylabel='sample count')
        pair[0].set_yscale('symlog', linthresh=1000)
        pair[1].set_xscale('symlog', linthresh=1000)
        pair[1].set_yscale('log')
        pair[0].legend(fontsize=7)
    fig.savefig(out / 'feedback-age.png', dpi=130)
    plt.close(fig)


def selfcheck():
    # Finite boundary checks; synthetic clocks are not hardware measurements.
    assert lift_ms(2, U32 - 2) == U32 + 2
    assert lift_ms(U32 - 2, U32 + 2) == U32 - 2
    rows = [dict(epoch=0, span=s, group=g, sequence=q, board_ns=t)
            for s, g, q, t in [(0, 'a', U32 - 1, 0), (0, 'a', 0, 2000000),
                                (0, 'a', 2, 6000000), (1, 'a', 3, 8000000),
                                (1, 'b', 4, 10000000), (1, 'b', 5, 12000000)]]
    add_periods(rows, 'board_ns', 'sequence')
    assert [r['period_us'] for r in rows] == [None, 2000, None, None, None, 2000]
    assert summarize([1, 2, 3], [0, 1, 2])['peak_to_peak_us'] == 2
    with tempfile.TemporaryDirectory(prefix='h7-latency-check-') as temporary:
        session = Path(temporary)
        events = [dict(type='connected', t_ns=0)]
        for i, seq in enumerate((U32 - 1, 0, 2)):
            frame = np.zeros(160, dtype='<f4')
            frame[3:5] = [7101, 2]
            frame[6:8] = [i & 65535, i >> 16]
            tick = (U32 - 2 + 2 * i) % U32
            frame[8:10] = [tick & 65535, tick >> 16]
            frame[159] = binascii.crc_hqx(frame[:159].tobytes(), 0xffff)
            raw_frame = frame.tobytes() + mirror.TAIL
            values = [seq, 1, (U32 - 2 + 2 * i) * 1000000, (U32 - 2 + 2 * i) * 1000000 + 1000,
                      100 + i, 0] + [0.] * 8 + [U32 - 1, 1234] + [0] * 11
            record = capture.RECORD.pack(*values)
            header = capture.HEADER.pack(capture.MAGIC, 1, 88, 1, 0, i, i + 1, 0, 1, 1, 0)
            packet = header + record
            packet += binascii.crc_hqx(packet, 0xffff).to_bytes(2, 'little')
            events.append(dict(type='data', t_ns=i + 1, data_hex=(raw_frame + packet).hex()))
        events.append(dict(type='stop', t_ns=4))
        (session / 'stream.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
        fs, rs, gaps, _ = read_recording(session)
        assert [f['ms'] for f in fs] == [U32 - 2, U32, U32 + 2]
        assert rs[0]['span'] == rs[1]['span'] and rs[2]['span'] != rs[1]['span']
        assert len(gaps) == 1 and gaps[0]['sequence_delta'] == 2
        assert rs[0]['raw'][14] == U32 - 1
        damaged = [dict(e) for e in events]
        data = bytearray.fromhex(damaged[2]['data_hex'])
        data[-1] ^= 1
        damaged[2]['data_hex'] = data.hex()
        (session / 'stream.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in damaged))
        _, damaged_rs, _, _ = read_recording(session)
        assert len(damaged_rs) == 2 and damaged_rs[0]['span'] != damaged_rs[1]['span']
        dropped = [dict(e) for e in events]
        data = bytearray.fromhex(dropped[2]['data_hex'])
        header = list(capture.HEADER.unpack_from(data, 644))
        header[7] = 1
        data[644:644 + capture.HEADER.size] = capture.HEADER.pack(*header)
        data[-2:] = binascii.crc_hqx(data[644:-2], 0xffff).to_bytes(2, 'little')
        dropped[2]['data_hex'] = data.hex()
        (session / 'stream.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in dropped))
        _, dropped_rs, _, _ = read_recording(session)
        assert dropped_rs[0]['span'] != dropped_rs[1]['span']
    return dict(passed=True, cases=['uint32 timestamp wrap', 'uint32 sequence wrap', 'sequence gap',
                                   'span boundary', 'state boundary', 'source parser roundtrip',
                                   'unknown wheel-age sentinel retained', 'CRC rejection splits span',
                                   'queue drop counter splits span', 'summary arithmetic'],
                hardware_test=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, action='append', help='repeatable; default: the three agreed recordings')
    parser.add_argument('--selfcheck', action='store_true', help='finite synthetic boundary checks, no recording or hardware')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts' / ('h7-latency-' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')))
    args = parser.parse_args()
    if args.selfcheck:
        print(json.dumps(selfcheck(), ensure_ascii=False, indent=2))
        return
    sessions = args.session or [ROOT / p for p in DEFAULT_SESSIONS]
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / 'deployment/H7_RL_ros' / p for p in SOURCE_FILES]
    reports = []
    for i, session in enumerate(sessions):
        print(f'Analyzing {session}', flush=True)
        reports.append(analyze(session, args.out / f'session-{i + 1:02d}'))
    save(args.out / 'manifest.json', dict(script=identity(__file__),
         finite_selfcheck=selfcheck(),
         decoders=[identity(mirror.__file__), identity(capture.__file__)],
         source_semantics_checked=[identity(p) for p in sources],
         source_identity_scope='local source supports field semantics; actual flashed source identity is limited by each recording manifest',
         sessions=[r['stream'] for r in reports], python=sys.version, numpy=np.__version__,
         matplotlib=matplotlib.__version__, generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))
    lines = ['# H7 已录链路耗时与抖动', '', '全部数值单位为μs，标准差使用总体标准差。分位数使用NumPy线性插值。', '',
             '每会话统计见 statistics.csv；连续区间见 segments.csv / segment-statistics.json；逐样本CSV保留原始事件行号及板端时间。', '',
             '同一状态的不同连续区间可以合并统计，但相邻间隔绝不跨缺口、状态变化或策略序号缺失计算。', '']
    for i, report in enumerate(reports):
        lines += [f'## 会话 {i + 1}', '', f"输入：`{report['stream']['path']}`", '',
                  f"全文件标准完整性：{report['standard_full_file_complete']}；表样本数：`{report['table_counts']}`。", '',
                  f"CAN HAL状态：`{report['can_hal_status']}`；动作配对：`{report['policy_pairing']}`。", '',
                  f"全文件CAN HAL失败：{json.loads((args.out / f'session-{i + 1:02d}/full-output/report.json').read_text())['failed_CAN_attempts']}；"
                  f"极值原始记录复核：{report['verification']['extrema_checked_against_source']}项。", '',
                  f'![软件耗时](session-{i + 1:02d}/timing.png)', '',
                  f'![反馈数据龄](session-{i + 1:02d}/feedback-age.png)', '']
        normal = 'strategy=0;engaged=1;fault=0;stage=normal'
        selected = [s for s in report['statistics'] if s['group'] == normal or
                    s['group'] == normal + ';pd=1;drive=1']
        if selected:
            lines += ['### 稳定NORMAL、投入且无故障的已观察样本', '',
                      '| 表/指标 | n | 均值μs | 标准差μs | P95μs | P99μs | 最大μs | 峰峰值μs |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|']
            for s in selected:
                lines.append(f"| {s['table']}/{s['metric']} | {s['count']} | " + ' | '.join(
                    f"{s[k]:.3f}" for k in ('mean_us', 'std_us', 'p95_us', 'p99_us', 'max_us', 'peak_to_peak_us')) + ' |')
            lines += ['']
        lines += [f"所有指标的时间曲线及分布：{', '.join(f'[{name}]' + f'(session-{i + 1:02d}/{name}-all-metrics.png)' for name in ('policy', 'output', 'control', 'can'))}。", '']
    lines += ['## 测量边界', ''] + [f'- {limit}' for limit in LIMITS]
    lines += ['', '## 缺失测点', '', '| 环节 | 所需测点 |', '|---|---|']
    lines += [f'| {name} | {needed} |' for name, needed in MISSING]
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    print(args.out.resolve(), flush=True)


if __name__ == '__main__':
    main()
