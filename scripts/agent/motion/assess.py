"""统一运动指标、保存轨迹复算与离线审计；不运行策略或物理。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STANDARD = 'chuanliantui-motion-acceptance-v1'
THRESHOLDS = dict(vx_mae_m_s=.10, yaw_mae_rad_s=.10,
                  parking_horizontal_peak_m_s=.05, steady_pitch_peak_deg=5.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics_verdict(*, vx_mae=None, yaw_mae=None, horizontal_peak=None,
                    pitch_peak=None, complete=False, parking=False):
    """统一门槛的唯一实现；缺失或非法指标不能通过。"""
    values = dict(tracked_vx=vx_mae, tracked_yaw=yaw_mae,
                  steady_pitch=pitch_peak, **(dict(stopped=horizontal_peak) if parking else {}))
    limits = dict(tracked_vx=THRESHOLDS['vx_mae_m_s'], tracked_yaw=THRESHOLDS['yaw_mae_rad_s'],
                  steady_pitch=THRESHOLDS['steady_pitch_peak_deg'],
                  stopped=THRESHOLDS['parking_horizontal_peak_m_s'])
    checks = {name: bool(value is not None and np.isfinite(value) and 0. <= value <= limits[name])
              for name, value in values.items()}
    checks['complete'] = bool(complete)
    return dict(standard=STANDARD, thresholds=THRESHOLDS.copy(), checks=checks,
                missing_metrics=[name for name, value in values.items() if value is None],
                failed_checks=[name for name, passed in checks.items() if not passed],
                passed=bool(all(checks.values())))


def phase_verdict(*, vx, yaw, horizontal_speed, pitch_deg, vx_target, yaw_target,
                  selected, valid, complete, parking=False, command_vx=None, command_yaw=None):
    """对预先确定的阶段评分；统计窗口不依赖实际跟踪效果。"""
    selected, valid = np.asarray(selected, bool), np.asarray(valid, bool)
    if selected.ndim != 1 or selected.shape != valid.shape:
        raise ValueError('selected and valid must be matching one-dimensional masks')
    def vector(value):
        return None if value is None else np.broadcast_to(np.asarray(value), selected.shape)
    vx, yaw, speed, pitch = map(vector, (vx, yaw, horizontal_speed, pitch_deg))
    target_v, target_y = map(vector, (vx_target, yaw_target))
    finite = np.ones_like(selected)
    for value in (vx, yaw, speed, pitch, target_v, target_y):
        if value is not None:
            finite &= np.isfinite(value)
    delivered = np.ones_like(selected)
    for command, target in ((command_vx, target_v), (command_yaw, target_y)):
        if command is not None:
            delivered &= np.isclose(vector(command), target, rtol=0., atol=1e-5)
    mask = selected & valid & finite
    def mean_error(actual, target):
        return None if actual is None or not mask.any() else float(np.abs(actual[mask]-target[mask]).mean())
    def peak(value):
        return None if value is None or not mask.any() else float(np.abs(value[mask]).max())
    metrics = dict(vx_mae=mean_error(vx, target_v), yaw_mae=mean_error(yaw, target_y),
                   horizontal_peak=peak(speed), pitch_peak=peak(pitch))
    complete = bool(complete and selected.any() and (valid & finite & delivered)[selected].all())
    return dict(**metrics_verdict(**metrics, complete=complete, parking=parking),
                metrics=metrics, expected_samples=int(selected.sum()), observed_samples=int(mask.sum()),
                command_delivered=bool(selected.any() and delivered[selected].all()))


def settled_window(reached, dt, wait_s=.5):
    """只按指令到位确定稳态，沿用原协议的0.5秒等待。"""
    measured = np.zeros(len(reached), dtype=bool)
    count = 0
    for index, value in enumerate(reached):
        count = count+1 if value else 0
        measured[index] = count > round(wait_s/dt)
    return measured


def supermimic_phase(values, layout, env, start, end, reference, dt, planned_end, yaw_reference=0.):
    get = lambda name: values[:, env, slice(*layout[name])]
    times = (np.arange(len(values))+1)*dt
    selected = (times > start+1e-9) & (times <= min(end, planned_end)+1e-9)
    expected = max(0, round((end-start)/dt))
    command = get('applied_command') if 'applied_command' in layout else None
    delivery_available = bool((reference == 0. and yaw_reference == 0.) or
                              (command is not None and command.shape[1] >= 2))
    result = phase_verdict(vx=get('true_velocity')[:, 0],
        yaw=get('yaw_rate_body_z')[:, 0] if 'yaw_rate_body_z' in layout else None,
        horizontal_speed=np.linalg.norm(get('root_linear_velocity_w')[:, :2], axis=-1),
        pitch_deg=get('roll_pitch')[:, 1], vx_target=reference, yaw_target=yaw_reference,
        selected=selected, valid=(get('valid')[:, 0] > .5) & np.isfinite(values[:, env]).all(-1),
        complete=bool(end <= planned_end+1e-8 and selected.sum() == expected and delivery_available),
        parking=bool(reference == 0. and yaw_reference == 0.),
        command_vx=command[:, 0] if reference != 0. and delivery_available else None,
        command_yaw=command[:, 1] if yaw_reference != 0. and delivery_available else None)
    if not delivery_available:
        result['command_delivered'] = False
    return result


def stand_phase(p, c, *, selected, vx_target, yaw_target=0., complete=True, yaw_columns=False):
    """稳站轨迹共用物理列；旧直线记录无偏航信号，明确报告缺项。"""
    return phase_verdict(vx=c[:, 4] if yaw_columns else c[:, 2],
        yaw=c[:, 5] if yaw_columns else (c[:, 4] if c.shape[1] >= 5 else None),
        horizontal_speed=np.linalg.norm(p[:, 7:9], axis=-1), pitch_deg=np.rad2deg(p[:, 4]),
        vx_target=vx_target, yaw_target=yaw_target, selected=selected,
        valid=(p[:, 12] > .5) & (p[:, 13] > .5) & np.isfinite(p).all(-1) & np.isfinite(c).all(-1), complete=complete,
        parking=bool(vx_target == 0. and yaw_target == 0.),
        command_vx=(c[:, 2] if yaw_columns else c[:, 0]) if vx_target != 0. else None,
        command_yaw=c[:, 3] if yaw_columns and yaw_target != 0. else None)


def tracking_rows(meta, v, physical, active, times):
    rows, windows = [], []
    # 同一100Hz周期中的500Hz峰值不能被降采样漏掉。
    ratio = physical.shape[0]//v.shape[0]
    assert ratio*v.shape[0] == physical.shape[0]
    grouped = physical.reshape(len(v), ratio, v.shape[1], physical.shape[-1])
    pitch = np.rad2deg(np.abs(grouped[..., 4])).max(1)
    speed = np.linalg.norm(grouped[..., 7:9], axis=-1).max(1)
    live = active & (grouped[..., 12:14] > .5).all((1, 3))
    for old in meta['rows']:
        env = old['env']; phases = {}
        for name, start, end, target in [('cruise', *meta['measured_window_s'], old['requested_vx_m_s']),
                ('parking', meta['stop_window_s'][1]-1., meta['stop_window_s'][1], 0.)]:
            selected = (times > start+1e-8) & (times <= end+1e-8)
            phases[name] = phase_verdict(vx=v[:, env, 1], yaw=v[:, env, 3],
                horizontal_speed=speed[:, env], pitch_deg=pitch[:, env], vx_target=target,
                yaw_target=0., selected=selected, valid=live[:, env],
                complete=bool(old['ready_at_8s'] and selected.sum() == round((end-start)/meta['policy_dt_s'])),
                parking=target == 0., command_vx=v[:, env, 0])
            windows.append(dict(env=env, phase=name, start_s=start, end_s=end))
        rows.append(dict(env=env, phases=phases, passed=bool(old['survived_22s'] and all(p['passed'] for p in phases.values()))))
    return rows, windows


def stand_segments(meta, physical, command):
    """按保存报告的指令分段复算，历史缺失偏航仍保留失败分母。"""
    yaw_columns = command.shape[-1] >= 15
    episodes = meta['episodes'] if yaw_columns else meta['moving']['episodes']
    dt = meta.get('physics_dt_s', meta['duration_s']/len(physical))
    rows, windows = [], []
    for episode in episodes:
        env, phases = episode['env'], {}
        segments = episode['segments']
        for index, segment in enumerate(segments):
            start = segment['start_s']
            end = segment.get('end_s', segments[index+1]['start_s'] if index+1 < len(segments) else meta['duration_s'])
            section = slice(round(start/dt), round(end/dt))
            p, c = physical[section, env], command[section, env]
            vx = segment['requested_vx_m_s'] if yaw_columns else segment['target_vx_m_s']
            yaw = segment.get('requested_yaw_rad_s', 0.)
            live = (p[:, 12] > .5) & (p[:, 13] > .5) & np.isfinite(p).all(-1) & np.isfinite(c).all(-1)
            gated = live & ((c[:, 9] > .5) & (c[:, 10] > .5) if yaw_columns else c[:, 3] > .5)
            reached = gated & (np.abs(c[:, 2 if yaw_columns else 0]-vx) <= 1e-5)
            if yaw_columns and yaw != 0.:
                reached &= np.abs(c[:, 3]-yaw) <= 1e-5
            if yaw_columns and 'position_hold' in meta and vx == 0.:
                reached = gated.copy()
                if yaw != 0.:
                    reached &= np.abs(c[:, 3]-yaw) <= 1e-5
            measured = settled_window(reached, dt)
            name = str(index)+'-'+segment['kind']
            ended = episode.get('first_end_s')
            full = bool(len(p) == round((end-start)/dt) and gated.all()
                        and measured.sum()*dt >= .5 and (ended is None or ended > end))
            phases[name] = stand_phase(p, c, selected=measured, vx_target=vx, yaw_target=yaw,
                yaw_columns=yaw_columns, complete=full)
            windows.append(dict(env=env, phase=name, start_s=start, end_s=end,
                selection='actual command reaches request, then wait0.5s; no actual-speed selection'))
        rows.append(dict(env=env, phases=phases, passed=bool(phases and all(p['passed'] for p in phases.values()))))
    return rows, windows


def reevaluate(directory):
    """使用原报告已经声明的阶段；结果写新文件，不修改输入。"""
    directory = Path(directory)
    meta_path = directory / ('summary.json' if (directory/'summary.json').exists() else 'result.json')
    meta = json.loads(meta_path.read_text())
    assert meta['parameter_updates'] == 0
    if 'source_snapshot_sha256' in meta:
        sources = directory/'source-sha256.json'
        assert digest(sources) == meta['source_snapshot_sha256']
        verify_sources(dict(checkpoint=meta['checkpoint'], sources=json.loads(sources.read_text())), directory)
    elif 'sources' in meta:
        verify_sources(meta, directory)
    else:
        raise ValueError('Recorded source hashes are required; cannot certify this historical run')
    rows, windows = [], []
    if 'layout' in meta and 'summary' in meta:
        trace = directory/'trajectory.npz'
        assert digest(trace) == meta['trajectory_sha256']
        with np.load(trace, allow_pickle=False) as saved:
            values = saved['values']
            assert len(values) == round(meta['seconds']/meta['physics_dt_s'])
            for original in meta['summary']['per_environment']:
                phases = {}
                for name, old in original['phases'].items():
                    if name not in ('cruise', 'parking'):
                        continue
                    phases[name] = supermimic_phase(values, meta['layout'], original['env'],
                        old['start_s'], old['end_s'], old['reference_vx_m_s'], meta['physics_dt_s'],
                        meta['seconds'], old.get('reference_yaw_rad_s', 0.))
                    windows.append(dict(env=original['env'], phase=name, start_s=old['start_s'], end_s=old['end_s']))
                rows.append(dict(env=original['env'], phases=phases,
                    passed=bool(original['survived'] and phases and all(p['passed'] for p in phases.values()))))
    elif (directory/'trace.npz').exists() and 'measured_window_s' in meta:
        trace = directory/'trace.npz'
        assert digest(trace) == meta['trace_sha256']
        with np.load(trace, allow_pickle=False) as saved:
            v, physical, active = saved['values'], saved['physical'], saved['active']
            rows, windows = tracking_rows(meta, v, physical, active, saved['time_s'])
    elif (directory/'trajectories.npz').exists():
        trace = directory/'trajectories.npz'
        assert digest(trace) == meta['trajectories_sha256']
        with np.load(trace, allow_pickle=False) as saved:
            rows, windows = stand_segments(meta, saved['physical'], saved['command'])
    else:
        raise ValueError('Supported saved runs: SuperMimic or stand_v2 tracking/speed/yaw')
    return dict(standard=STANDARD, thresholds=THRESHOLDS, input_files={str(p.resolve()): digest(p) for p in (meta_path, trace)},
        windows=windows, total=len(rows), passed=sum(row['passed'] for row in rows), rows=rows,
        source_verified=True, evaluator_sha256=digest(__file__),
        full_task_accepted=False, scope='Four numeric criteria on original declared windows; get-up excluded; no simulation or training')


def verify_sources(report, directory):
    """历史来源须逐字节匹配；可从已校验的保存源码中查找。"""
    import tarfile
    root = Path(__file__).resolve().parents[3]
    checkpoint = Path(report['checkpoint'])
    archives = []
    for archive in sorted((root/'scripts/archive').glob('*/source.tar.gz')):
        archives.append(tarfile.open(archive))
    try:
        for source, expected in report['sources'].items():
            relative = Path(source)
            if relative.is_absolute():
                relative = relative.relative_to(root)
            candidates = [root/relative, directory/'snapshot'/relative,
                          directory.parent/'source'/relative, directory.parent/'evaluation-source'/relative,
                          directory/'source.py', checkpoint.parent/'snapshot'/relative]
            prefix = Path('mjlab_training/src')
            if relative.is_relative_to(prefix):
                candidates.append(checkpoint.parent/'snapshot'/relative.relative_to(prefix))
            if any(p.is_file() and digest(p) == expected for p in candidates):
                continue
            for archive in archives:
                try:
                    data = archive.extractfile(str(relative)).read()
                except KeyError:
                    continue
                if hashlib.sha256(data).hexdigest() == expected:
                    break
            else:
                raise ValueError(f'Missing matching historical source: {source}')
    finally:
        for archive in archives:
            archive.close()


def audit_tracking(measurement, initial_reference=None):
    from types import SimpleNamespace
    args = SimpleNamespace(measurement=Path(measurement), initial_reference=initial_reference)
    r = json.loads((args.measurement/"result.json").read_text())
    assert r["parameter_updates"] == 0 and r["actor_and_encoder_unchanged"]
    assert digest(r["checkpoint"]) == r["checkpoint_sha256"]
    verify_sources(r, args.measurement)
    assert digest(args.measurement/"trace.npz") == r["trace_sha256"]
    assert digest(args.measurement/"qpos.npz") == r["qpos_sha256"]
    with np.load(args.measurement/"trace.npz") as f:
        v, active, physical, ready = f["values"], f["active"], f["physical"], f["ready"]
        weight = f["weight_N"]
        assert v.shape == (2200, 64, 8) and physical.shape == (11000, 64, 15)
        assert np.allclose(f["time_s"], (np.arange(2200)+1)*.01)
        assert np.isfinite(v[active]).all()
        assert np.isfinite(physical[physical[:, :, 13] > .5]).all()
        if args.initial_reference:
            with np.load(args.initial_reference) as old:
                for key in ("initial_qpos", "initial_qvel", "weight_N"):
                    assert np.array_equal(old[key], f[key]), key
    assert not (np.diff(active.astype(int), axis=0) > 0).any()
    assert np.array_equal(ready, (v[799, :, 5] > .5) & (v[799, :, 6] < .5) & active[799])
    requested = np.array([row["requested_vx_m_s"] for row in r["rows"]], dtype=np.float32)
    expected = np.zeros((2200,64), dtype=np.float32)
    expected[800:1800] = np.where(ready, requested, 0.)
    assert np.array_equal(v[:, :, 0], expected), "Nonzero command did not reach the measurement window"
    assert (v[800:1800, :, 0] != 0).any(), "No nonzero command was delivered"
    tail = physical[5000:9000]
    balanced = ((tail[:, :, :2] >= np.maximum(.1*weight, 5.)[None, :, None]).all(-1)
        & (tail[:, :, :2].sum(-1) >= weight*.8) & (tail[:, :, :2].sum(-1) <= weight*1.2)
        & (tail[:, :, 2] <= 1.) & (tail[:, :, 3:5].max(-1) <= np.deg2rad(35.))
        & (tail[:, :, 5] <= .5) & (tail[:, :, 12] > .5) & (tail[:, :, 13] > .5))
    for i,row in enumerate(r["rows"]):
        if not row["ready_at_8s"]:
            assert row["actual_mean_vx_m_s"] is None and row["measured_s"] == 0.
            continue
        data = v[1000:1800, i][active[1000:1800, i]]
        assert np.isclose(row["actual_mean_vx_m_s"], data[:, 1].mean())
        assert np.isclose(row["vx_mean_abs_error_m_s"], np.abs(data[:, 1]-requested[i]).mean())
        assert np.isclose(row["steady_balance_fraction"], balanced[:, i].mean())
    for group in r["groups"]:
        rows = [row for row in r["rows"] if np.isclose(row["requested_vx_m_s"], group["command_vx_m_s"])]
        complete = [row for row in rows if row["complete_tracking_window"]]
        assert group["total"] == len(rows) and group["completed"] == len(complete)
        if complete:
            assert np.isclose(group["actual_mean_vx_m_s"], np.mean([row["actual_mean_vx_m_s"] for row in complete]))
            assert np.isclose(group["mean_abs_error_m_s"], np.mean([row["vx_mean_abs_error_m_s"] for row in complete]))
    result = dict(verified=True, checkpoint_sha256=r["checkpoint_sha256"],
        recorded_policy_steps=2200, recorded_physics_steps=11000,
        nonzero_command_delivered=True, original_64_initial_states_match=bool(args.initial_reference),
        ready_envs=int(ready.sum()), not_ready_envs=np.where(~ready)[0].tolist(),
        complete_tracking_envs=sum(row["complete_tracking_window"] for row in r["rows"]),
        final_drive_balanced_envs=sum(row["final_drive_balance_1s"] for row in r["rows"]),
        parameter_updates=0, source_sha256=digest(__file__))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--measurement', type=Path, help='Recompute four motion criteria from saved trajectories')
    mode.add_argument('--audit-tracking', type=Path, help='Original stand_v2 source/hash/command/balance audit')
    mode.add_argument('--selfcheck', action='store_true', help='Finite arithmetic checks only')
    parser.add_argument('--initial-reference', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.selfcheck:
        from scripts.agent.maintenance.check_script_merges import motion
        result = motion()
    elif args.audit_tracking:
        result = audit_tracking(args.audit_tracking, args.initial_reference)
    else:
        if args.output is None:
            parser.error('--measurement requires --output pointing to a new JSON file')
        result = reevaluate(args.measurement)
    if args.output:
        with args.output.open('x') as out:
            out.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(result if args.selfcheck else {k:v for k,v in result.items() if k not in ('rows', 'windows')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
