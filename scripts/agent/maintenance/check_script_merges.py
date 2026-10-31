"""用已有测量和预测复核脚本合并；不训练、不仿真、不改写原产物。"""
import ast
import copy
import importlib
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / 'scripts/archive/20261101-script-cleanup/source.tar.gz'


def original_functions(name, functions, namespace):
    with tarfile.open(ARCHIVE) as bundle:
        source = bundle.extractfile('scripts/agent/' + name + '.py').read().decode()
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in functions]
    assert len(nodes) == len(functions)
    scope = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), scope)
    return scope


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        np.testing.assert_array_equal(left.cpu().numpy(), right.cpu().numpy())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    elif isinstance(left, float) and np.isnan(left):
        assert np.isnan(right)
    else:
        assert left == right, (left, right)


def contacts():
    from scripts.agent.stand_v2 import _contact_diagnostics as shared
    measurement = ROOT / 'artifacts/stand-v2-model499-diagnostic-20261027-final/measurements.npz'
    with np.load(measurement) as data:
        assert tuple(data['fields']) == shared.FIELDS
        values, active = data['values'], data['active']
    result = json.loads(measurement.with_name('result.json').read_text())
    late = active.copy()
    late[:max(0, round((result['duration_s'] - 5.) / result['policy_dt_s']))] = False
    for name in ('diagnose_chuanliantui_stand_v2', 'diagnose_stand_v2_rewards_velocity'):
        old = original_functions(name, ['contact_summary'], vars(shared))['contact_summary']
        for mask in (active, np.zeros_like(active)):
            same(old(values, mask, late & mask, result['policy_dt_s']),
                 shared.contact_summary(values, mask, late & mask, result['policy_dt_s']))
    return dict(source=str(measurement.relative_to(ROOT)), samples=int(values.shape[0]),
                environments=int(values.shape[1]), original_variants=2, empty_mask_checked=True)


def rounding():
    from scripts.agent.supermimic.check_supermimic_motion_retrain import norm_count_rounding_proof as check
    old = original_functions('check_supermimic_motion_retrain_counts', ['norm_count_rounding_proof'], {'torch': torch})['norm_count_rounding_proof']
    cases = [(100., 500., 10, 64), (7154886., 35774412., 6992, 1024)]
    for count, delta, steps, samples in cases:
        args = (torch.tensor(count), torch.tensor(delta), steps, samples)
        same(old(*args), check(*args))
    rejected = 0
    for count, delta, steps, samples in [(100., 501., 10, 64), (7154886., 35874412., 6992, 1024),
                                         (100., float('nan'), 10, 64), (100., 500., -1, 64)]:
        for function in (old, check):
            try:
                function(torch.tensor(count), torch.tensor(delta), steps, samples)
            except AssertionError:
                rejected += 1
            else:
                raise AssertionError('坏计数被接受')
    assert rejected == 8
    return dict(exact_and_rounded_cases=2, rejected_cases=4, original_implementation_equal=True)


def scoring():
    from scripts.agent.supermimic import _motion_scoring as shared
    saved_path = ROOT / 'artifacts/supermimic-v12-retrain-20261031-r1/score/predictions.pt'
    saved = torch.load(saved_path, map_location='cpu', weights_only=False, mmap=True)
    reports = []
    for name in ('score_supermimic_motion', 'score_supermimic_motion_retrain'):
        module = importlib.import_module('scripts.agent.supermimic.' + name)
        old = original_functions(name, ['predict', 'summarize'], vars(module))
        # 使用实际保存的预测和窗口；保留两个版本不同的指标集合。
        selected = [p for p in saved['predictions'] if p['world_label'] == saved['predictions'][0]['world_label']]
        for entry in selected:
            w = saved['windows'][entry['recording_index']]
            same(old['summarize'](entry['tensors'], w), module.summarize(entry['tensors'], w))
        # 空掩码、非有限预测必须保持原有拒计／计数方式。
        entry = selected[0]; w = copy.deepcopy(saved['windows'][entry['recording_index']])
        w['mask'].zero_(); pred = dict(entry['tensors']); pred['path'] = pred['path'].clone()
        pred['path'][0, 0, 0, 0] = float('nan')
        same(old['summarize'](pred, w), module.summarize(pred, w))
        reports.append(dict(variant=name, saved_prediction_groups=len(selected)))
    # 以保存的真实窗口驱动原/新预测循环；回放适配器不学习、不积分物理。
    window = saved['windows'][0]
    small = dict(plan=window['plan'][:3], mask=window['mask'][:3].clone(),
                 truth={k: v[:3].clone() for k, v in window['truth'].items()})
    small['mask'][1, 8:] = False
    class RecordedWorld:
        adapter = SimpleNamespace(continuation=lambda path: torch.isfinite(path).all(dim=(-2, -1)).float())

        def step(self, state, parameters, action, command, context, previous, previous_previous, noise, return_path):
            # 输出依赖调用传入量，能发现历史动作、命令、噪声传递顺序变化。
            updated = state + action.sum(-1, keepdim=True) * .001 + command.sum(-1, keepdim=True) * .002
            updated = updated + previous.sum(-1, keepdim=True) * .003 + previous_previous.sum(-1, keepdim=True) * .004
            updated = updated + noise.sum(-1, keepdim=True) * .005
            path = updated[..., :64].unsqueeze(1).expand(-1, 5, -1)
            return updated, updated[..., :25], updated.sum(-1), updated[..., 0] > 1., updated[..., :31], path
    for teacher in (False, True):
        for batch in (1, 3):
            same(old['predict'](RecordedWorld(), small, teacher, batch),
                 shared.predict(RecordedWorld(), small, teacher, batch))
    return dict(source=str(saved_path.relative_to(ROOT)), variants=reports,
                prediction_loop_cases=4, empty_mask_and_nonfinite_checked=True,
                scope='保存的预测张量与预测循环等价；未重新评分模型或运行物理。')


def provenance():
    from scripts.agent.stand import analyze_chuanliantui_stand as stand
    from scripts.agent.split import analyze_walk_velocity_compensation as velocity
    from scripts.agent.imitation.prepare_bc_wheel_dataset import snapshot_text
    old_path = 'scripts/agent/assess_chuanliantui_stand.py'
    sources = [dict(path='/historical/' + old_path, sha256='original-assessor')]
    sources += [dict(path='/historical/' + name, sha256=name) for name in stand.PROTOCOL_SOURCE_SUFFIXES[1:]]
    legacy = stand.protocol_sources(dict(sources=sources))
    assert all(legacy.values()), '旧报告丢失协议来源'
    relocated = copy.deepcopy(sources)
    relocated[0]['path'] = '/current/scripts/agent/stand/assess_chuanliantui_stand.py'
    same(legacy, stand.protocol_sources(dict(sources=relocated)))
    relocated[0]['sha256'] = 'changed-assessor'
    assert stand.protocol_sources(dict(sources=relocated)) != legacy
    proof = velocity.verify_candidate_source(ROOT / 'scripts/agent/split/walk_sensor_velocity.py')
    assert proof['original_sha256'] == velocity.EXPECTED_CANDIDATE_SHA256
    with tempfile.TemporaryDirectory() as directory:
        changed = Path(directory) / 'changed.py'
        changed.write_text((ROOT / 'scripts/agent/split/walk_sensor_velocity.py').read_text() + '\n# changed\n')
        try:
            velocity.verify_candidate_source(changed)
        except AssertionError:
            pass
        else:
            raise AssertionError('额外源码修改没有被拒绝')
        with tarfile.open(ARCHIVE) as bundle:
            for filename in ('collect_bc_braking.py', 'check_lqr_student_closed_loop.py'):
                original = bundle.extractfile('scripts/agent/' + filename).read()
                for prefix in ('scripts/agent/', 'scripts/agent/imitation/'):
                    key = prefix + filename
                    snapshot = Path(directory) / 'snapshot' / key
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    snapshot.write_bytes(original)
                    report = dict(source_sha256={key: hashlib.sha256(original).hexdigest()})
                    assert snapshot_text(Path(directory), report, filename) == original.decode()
                    snapshot.write_bytes(original + b'\n# changed\n')
                    try:
                        snapshot_text(Path(directory), report, filename)
                    except AssertionError:
                        pass
                    else:
                        raise AssertionError('损坏的历史快照未被拒绝')
        # 替身解释器只核对工作目录和调度目标；不执行任何校准。
        fake = Path(directory) / 'python-path-check'
        fake.write_text('#!/usr/bin/env bash\nset -eu\n'
                        '[[ "$PWD" == "$EXPECTED_REPO" ]]\n'
                        r'if [[ "$1" == "-m" ]]; then test -f "${2//./\/}.py"; else test -f "$1"; fi' + '\n')
        fake.chmod(0o700)
        subprocess.run(['bash', str(ROOT / 'scripts/agent/real2sim/reproduce_fixed_h7_calibration.sh'),
                        str(Path(directory) / 'unused-output')], cwd=ROOT,
                       env=dict(os.environ, TASK_PYTHON=str(fake), EXPECTED_REPO=str(ROOT)), check=True)
    return dict(old_and_new_report_paths=True, changed_report_hash_not_equal=True,
                root_only_source_migration=proof, tampered_source_rejected=True,
                historical_snapshot_paths_and_hashes=True, shell_dispatch_without_calibration=True)


def motion():
    """四项统一标准的边界、缺项、完整性与横向漂移检查。"""
    from scripts.agent.motion.assess import metrics_verdict, phase_verdict
    base = dict(vx_mae=.1, yaw_mae=.1, horizontal_peak=.05, pitch_peak=5.,
                complete=True, parking=True)
    assert metrics_verdict(**base)['passed']
    checked = ['inclusive_boundaries']
    for key in ('vx_mae', 'yaw_mae', 'horizontal_peak', 'pitch_peak'):
        assert not metrics_verdict(**{**base, key: np.nextafter(base[key], np.inf)})['passed']
        assert not metrics_verdict(**{**base, key: None})['passed']
        assert not metrics_verdict(**{**base, key: np.nan})['passed']
        assert not metrics_verdict(**{**base, key: -1.})['passed']
        checked.append(key + '_outside_missing_nan_negative')
    assert not metrics_verdict(**{**base, 'complete': False})['passed']
    assert metrics_verdict(**{**base, 'parking': False, 'horizontal_peak': None})['passed']
    data = dict(vx=np.array([-.11, .11]), yaw=np.zeros(2), horizontal_speed=np.zeros(2),
                pitch_deg=np.zeros(2), vx_target=0., yaw_target=0.,
                selected=np.ones(2, dtype=bool), valid=np.ones(2, dtype=bool),
                complete=True, parking=True)
    assert not phase_verdict(**data)['checks']['tracked_vx'], '正负误差不能抵消'
    zero = {**data, 'vx': np.zeros(2)}
    assert phase_verdict(**zero)['passed']
    assert not phase_verdict(**{**zero, 'horizontal_speed': np.array([0., .051])})['passed']
    assert not phase_verdict(**{**zero, 'valid': np.array([True, False])})['passed']
    assert not phase_verdict(**{**zero, 'selected': np.zeros(2, dtype=bool)})['passed']
    assert not phase_verdict(**{**zero, 'yaw': None})['passed']
    assert not phase_verdict(**{**zero, 'pitch_deg': np.array([0., np.nan])})['passed']
    assert not phase_verdict(**{**zero, 'command_vx': np.ones(2)})['passed']
    checked += ['absolute_not_signed_mean', 'lateral_peak', 'early_termination', 'empty_window',
                'missing_yaw', 'nonfinite_sample', 'command_delivery']
    # 新旧multimode的原始统计与专属诊断必须相同，门槛变化单独验证。
    from scripts.agent.motion.assess import THRESHOLDS
    path = ROOT/'scripts/agent/multimode/assess_multimode_tracking.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'episode_verdict']
    scope = dict(metrics_verdict=metrics_verdict, THRESHOLDS=THRESHOLDS, HEIGHT_TOL=.03, TILT_TOL_DEG=25.)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    old = original_functions('assess_multimode_tracking', ['episode_verdict'],
        dict(VX_TOL=.30, YAW_TOL=.30, HEIGHT_TOL=.03, TILT_TOL_DEG=25., BASE_CONTACT_N=1.))['episode_verdict']
    kwargs = dict(observed_s=10., vx_err_abs_sum=2., vx_err_signed_sum=-1., vy_abs_sum=3.,
        yaw_err_abs_sum=0., yaw_err_signed_sum=0., height_err_abs_sum=0., tilt_max_deg=0.,
        base_contact_s=0., numerical_failure=False, survived=True)
    previous = old(**kwargs)
    current = scope['episode_verdict'](**kwargs, pitch_max_deg=0.)
    assert previous['means'] == current['means']
    assert previous['passed'] and not current['passed'], '旧0.30通过应在统一0.10下拒绝'
    for key in ('tracked_height', 'upright', 'no_base_contact', 'survived', 'numerical_ok'):
        assert previous['checks'][key] == current['diagnostic_checks'][key]
    try:
        scope['episode_verdict'](**{**kwargs, 'observed_s': 0.})
    except ValueError:
        pass
    else:
        raise AssertionError('空累积窗口未拒绝')
    checked += ['multimode_statistics_unchanged', 'legacy_030_now_rejected', 'diagnostics_retained', 'zero_accumulator_guard']
    from scripts.agent.motion.assess import stand_segments, supermimic_phase, settled_window
    dt = .01
    p = np.zeros((1000, 1, 15)); p[:, :, 12:14] = 1.
    c = np.zeros_like(p); c[:, :, 9:11] = 1.
    c[:, 0, 2] = np.minimum((np.arange(1000)+1)*dt*.1, .3)
    c[:, 0, 3:6] = .3
    c[:, 0, 4] = c[:, 0, 2]
    meta = dict(physics_dt_s=dt, duration_s=10., position_hold={}, episodes=[dict(env=0, first_end_s=None,
        segments=[dict(start_s=0., end_s=10., kind='rate', requested_vx_m_s=.3, requested_yaw_rad_s=.3)])])
    expected = settled_window(np.abs(c[:, 0, 2]-.3) <= 1e-5, dt)
    rows, _ = stand_segments(meta, p, c)
    assert rows[0]['phases']['0-rate']['expected_samples'] == int(expected.sum()), '位置保持不能提前统计非零指令'
    damaged = p.copy(); damaged[0, 0, 0] = np.nan
    damaged_rows, _ = stand_segments(meta, damaged, c)
    assert not damaged_rows[0]['passed'], '瞬态数值异常不能因稳态选窗被丢弃'
    meta['episodes'][0]['first_end_s'] = 10.
    rows, _ = stand_segments(meta, p, c)
    assert not rows[0]['passed'], '恰在段末终止不能通过'
    layout = dict(valid=[0,1], true_velocity=[1,4], root_linear_velocity_w=[4,7],
                  roll_pitch=[7,9], yaw_rate_body_z=[9,10], applied_command=[10,13])
    values = np.zeros((1000, 1, 13)); values[:, :, 0] = 1.
    assert not supermimic_phase(values, layout, 0, 0., 10., 0., dt, 10., .05)['passed'], '小转向指令未送达不能通过'
    checked += ['position_hold_nonzero_window', 'nonfinite_transient', 'termination_at_segment_end', 'nonzero_yaw_delivery']
    cpu_path = ROOT/'scripts/agent/multimode/assess_multimode_cpu_envelope.py'
    cpu_tree = ast.parse(cpu_path.read_text())
    functions = [n for n in cpu_tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_first_failure', '_verdict')]
    cpu_scope = dict(np=np, assess=SimpleNamespace(episode_verdict=scope['episode_verdict'],
        VX_TOL=.1, YAW_TOL=.1, HEIGHT_TOL=.03, TILT_TOL_DEG=25., BASE_CONTACT_N=1.),
        DOWN_TILT_DEG=35., RECOVERY_WINDOW_S=2.)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(cpu_path), 'exec'), cpu_scope)
    rollout = dict(t=np.arange(1,1001)*.01, vx=np.zeros(1000), yaw_rate=np.zeros(1000),
        height=np.full(1000,.27), tilt=np.zeros(1000), contact=np.zeros(1000,bool),
        pitch=np.zeros(1000), horizontal_speed=np.zeros(1000))
    assert cpu_scope['_verdict'](rollout, [0.,0.,.27], 2.)['passed'], 'CPU调用方必须补齐四项输入'
    checked += ['cpu_envelope_shared_verdict']
    return dict(cases=checked, training_updates=0, physics_steps=0)


def motion_saved():
    """保存轨迹与独立旧统计逐项比较；不加载策略或运行仿真。"""
    import math
    from scripts.agent.motion.assess import supermimic_phase, reevaluate, stand_phase, settled_window
    directory = ROOT/'artifacts/supermimic-v13-final-seed43-20261031'
    meta = json.loads((directory/'summary.json').read_text())
    old = original_functions('assess_chuanliantui_supermimic', ['phase_metrics'],
                             dict(np=np, math=math, HEIGHT=.26))['phase_metrics']
    compared, maximum_error = 0, 0.
    with np.load(directory/'trajectory.npz', allow_pickle=False) as saved:
        values = saved['values']
        for row in meta['summary']['per_environment']:
            for name, phase in row['phases'].items():
                if name not in ('cruise', 'parking'):
                    continue
                args = (values, meta['layout'], row['env'], phase['start_s'], phase['end_s'],
                        phase['reference_vx_m_s'], meta['physics_dt_s'], meta['seconds'])
                previous = old(*args, yaw_reference=phase['reference_yaw_rad_s'])
                current = supermimic_phase(*args, yaw_reference=phase['reference_yaw_rad_s'])
                for new_key, old_key, stat in [('vx_mae', 'vx_error_m_s', 'mae'),
                        ('yaw_mae', 'yaw_error_rad_s', 'mae'),
                        ('horizontal_peak', 'horizontal_speed_m_s', 'max_abs'), ('pitch_peak', 'pitch_deg', 'max_abs')]:
                    before = None if previous[old_key] is None else previous[old_key][stat]
                    after = current['metrics'][new_key]
                    assert (before is None) == (after is None)
                    if before is not None:
                        maximum_error = max(maximum_error, abs(before-after))
                        assert np.isclose(before, after, rtol=1e-6, atol=1e-7), (name, new_key, before, after)
                assert current['checks']['complete'] == previous['complete']
                compared += 1
    tracking = reevaluate(ROOT/'artifacts/stand-v2-model1999-tracking-20261027/measurement-v2')
    tracking_meta = json.loads((ROOT/'artifacts/stand-v2-model1999-tracking-20261027/measurement-v2/result.json').read_text())
    with np.load(ROOT/'artifacts/stand-v2-model1999-tracking-20261027/measurement-v2/trace.npz') as saved:
        v, p, active = saved['values'], saved['physical'], saved['active']
        times = saved['time_s']; selected = (times > 10.+1e-8) & (times <= 18.+1e-8)
        for row in tracking['rows']:
            env = row['env']; result = row['phases']['cruise']
            mask = selected & active[:, env]
            if mask.any():
                target = tracking_meta['rows'][env]['requested_vx_m_s']
                error = float(np.abs(v[mask, env, 1]-target).mean())
                assert np.isclose(error, result['metrics']['vx_mae'], atol=1e-7)
                start, end = 5000, 9000
                valid = (p[start:end, env, 12:14] > .5).all(-1)
                if valid.all():
                    assert np.isclose(np.rad2deg(p[start:end, env, 4]).max(), result['metrics']['pitch_peak'], atol=1e-6)
                    assert np.isclose(np.linalg.norm(p[start:end, env, 7:9],axis=-1).max(), result['metrics']['horizontal_peak'], atol=1e-7)
    # 旧直线记录缺失偏航，新记录增加该信号后可通过，不补造旧信号。
    p = np.zeros((1000, 15)); p[:, 12:14] = 1.
    c = np.zeros((1000, 4)); c[:, 3] = 1.
    selected = settled_window(np.ones(1000, bool), .002)
    incomplete = stand_phase(p, c, selected=selected, vx_target=0.)
    assert incomplete['missing_metrics'] == ['tracked_yaw'] and not incomplete['passed']
    complete = stand_phase(p, np.column_stack((c,np.zeros(1000))), selected=selected, vx_target=0.)
    assert complete['passed']
    return dict(supermimic_phases_compared=compared, maximum_metric_error=maximum_error,
        stand_tracking_samples=len(tracking['rows']), stand_tracking_passed=tracking['passed'],
        missing_old_yaw_not_fabricated=True, training_updates=0, physics_steps=0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-only', action='store_true')
    parser.add_argument('--saved-motion', action='store_true')
    args = parser.parse_args()
    if args.motion_only:
        print(json.dumps(motion(), ensure_ascii=False))
        return
    if args.saved_motion:
        print(json.dumps(dict(motion=motion(), saved=motion_saved()), ensure_ascii=False))
        return
    torch.set_num_threads(2)
    print(json.dumps(dict(motion=motion(), motion_saved=motion_saved(), provenance=provenance(), contacts=contacts(), rounding=rounding(), scoring=scoring(),
                          training_updates=0, physics_steps=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
