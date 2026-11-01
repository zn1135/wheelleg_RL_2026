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
    else:
        raise ValueError('Expected a retained SuperMimic summary/trajectory run')
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




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--measurement', type=Path, help='Recompute four motion criteria from saved trajectories')
    mode.add_argument('--selfcheck', action='store_true', help='Finite arithmetic checks only')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.selfcheck:
        from scripts.agent.maintenance.check_script_merges import motion
        result = motion()
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
