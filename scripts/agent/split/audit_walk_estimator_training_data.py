#!/usr/bin/env python3
"""Inventory existing complete2999 runs without fitting, scoring or simulation."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.agent.split.analyze_walk_velocity_compensation import phase_specs


MODEL_SHA = 'c409bb9a4c28308c5919541d2bb4c5082f43fd9d904e40c8b5fcfc1a33dd3708'
RUNS = (
    ('compensation44','artifacts/walk25-sensor-compensation-20261031-r1/discovery'),
    ('compensation45','artifacts/walk25-sensor-compensation-20261031-r1/validation'),
    ('shadow46','artifacts/walk25-shadow-delay-20261031-r1/captures-r2/study'),
    ('shadow47','artifacts/walk25-shadow-delay-20261031-r1/captures-r2/validation'),
    ('goal43','artifacts/walk25-goal-wheel-vx-probe-20261030-r1'),
)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def ranges(array):
    return [[float(array[...,axis].min()),float(array[...,axis].max())] for axis in range(array.shape[-1])]


def segments(case):
    raw = case.get('segments')
    if raw is None:
        raw = [dict(start=0.,end=2.,speed=0.),dict(start=2.,end=16.,speed=case['speed']),
               dict(start=16.,end=30.,speed=0.)]
    result = []
    for row in raw:
        value = dict(row,acceleration=case['acceleration'])
        if result and result[-1]['speed'] == value['speed']:
            result[-1]['end'] = value['end']
        else:
            result.append(value)
    return result


def main(args):
    root = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'data-audit.json').exists() and not (output/'data-audit.md').exists()
    runs = []
    for label, relative in RUNS:
        folder = root/relative
        meta_path, trace_path = folder/'encoder.json',folder/'encoder.npz'
        meta = json.loads(meta_path.read_text())
        receipt = json.loads((folder/'result.json').read_text())
        assert receipt['verified'] and meta['checkpoint_sha256'] == MODEL_SHA
        assert digest(Path(meta['checkpoint'])) == MODEL_SHA
        assert meta['feedback_mode'] == meta['velocity_source'] == 'encoder'
        assert meta['parameter_updates'] == 0 and meta['actor_unchanged']
        assert meta['actor_state_sha256'] == meta['actor_state_sha256_after']
        assert meta['survived'] == 96 and not any(meta['episode_done'])
        with np.load(trace_path,allow_pickle=False) as saved:
            keys = saved.files
            native_history_present = 'native_history' in keys
            native_observation_present = 'policy_observation' in keys
            native_target_present = 'target_com_m_s' in keys
            reset_present = 'reset_after_policy' in keys
            if native_observation_present:
                observation = saved['policy_observation']
                truth, valid, times = saved['target_com_m_s'],saved['policy_valid'],saved['policy_time_s']
                command = observation[...,6:9]/np.array((2.,.25,5.),dtype=np.float32)
            else:
                # 旧seed43仅用于覆盖统计；不由这些物理字段推测网络历史。
                sampled = saved['values'][::5].copy()
                layout = meta['layout']
                truth = sampled[...,slice(*layout['true_velocity_before'])]
                valid = sampled[...,layout['valid'][0]] > .5
                command = sampled[...,slice(*layout['applied_command'])]
                times = np.arange(len(sampled))*.01
                observation = None
                del sampled
            live = saved['qpos_live']
            assert truth.shape == (3000,96,3) and valid.shape == (3000,96)
            assert np.isfinite(truth).all() and np.isfinite(command).all()
            assert valid.all() and live.all()
            assert np.array_equal(times,np.arange(3000)*.01)
            history_residual = None
            if native_history_present:
                history = saved['native_history']
                assert history.shape == (3000,96,125)
                history_residual = 0.
                for slot in range(5):
                    indices = np.maximum(0,np.arange(3000)+slot-4)
                    history_residual = max(history_residual,float(abs(history[...,slot*25:(slot+1)*25]-observation[indices]).max()))
                assert history_residual == 0.
                del history
            if reset_present:
                assert not saved['reset_after_policy'].any()
            stored_shapes = {}
            for key in ('policy_observation','native_history','target_com_m_s','policy_valid','reset_after_policy',
                        'raw_sensors','policy_previous_action','policy_command_scaled'):
                if key in keys:
                    stored_shapes[key] = list(saved[key].shape)
        phase_masks = {name:np.zeros((3000,96),dtype=bool) for name in
            ('initial','parking','cruise_forward','cruise_backward','start','brake')}
        interval_rows = {name:[] for name in phase_masks}
        for case in meta['cases']:
            for env in case['env_ids']:
                for phase in phase_specs(segments(case),times,command[:,env,0],valid[:,env]):
                    name = phase['kind']
                    if name == 'cruise':
                        name += '_forward' if phase['requested_speed'] > 0 else '_backward'
                    mask = (times >= phase['start']-1e-9) & (times < phase['end']-1e-9)
                    phase_masks[name][:,env] |= mask
                    interval_rows[name].append(dict(env=env,start_s=phase['start'],end_s=phase['end'],
                        valid_frames=int((mask & valid[:,env]).sum()),duration_s=float(mask.sum()*.01)))
        assert np.array_equal(sum(mask.astype(int) for mask in phase_masks.values()),np.ones((3000,96),dtype=int))
        phases = {}
        for name, mask in phase_masks.items():
            effective = mask & valid
            durations = [row['duration_s'] for row in interval_rows[name] if row['valid_frames']]
            phases[name] = dict(valid_policy_frames=int(effective.sum()),
                aggregate_environment_seconds=float(effective.sum()*.01),
                episodes_with_phase=int(effective.any(0).sum()),contiguous_intervals=len(durations),
                contiguous_duration_s=dict(minimum=min(durations),median=float(np.median(durations)),maximum=max(durations)),
                true_COM_xyz_m_s_ranges=ranges(truth[effective]),command_vx_yaw_height_ranges=ranges(command[effective]))
        availability = dict(native_history125=native_history_present,exact_native_Actor25=native_observation_present,
            same_tick_three_axis_COM=native_target_present,native_policy_valid='policy_valid' in keys,
            explicit_reset_after_policy=reset_present,first_episode_completeness=True,
            original_model_identity=True,native_history_reconstruction_residual=history_residual)
        usability = ('direct_native125_and_native_target' if native_history_present else
            'deterministic_history_rebuild_from_exact25_required' if native_observation_present else
            'reference_only_not_recommended_for_estimator_training')
        runs.append(dict(id=label,seed=meta['seed'],folder=str(folder),
            checkpoint_sha256=MODEL_SHA,checkpoint_file_sha256_verified=True,actor_state_sha256=meta['actor_state_sha256'],
            files_sha256={str(path):digest(path) for path in (meta_path,trace_path,folder/'result.json',folder/'source.py')},
            stored_shapes=stored_shapes,availability=availability,usability=usability,
            native_history_rule=('Direct native125 verified slot-by-slot against exact25, reset backfill and old-to-new order.' if native_history_present else
                'Exact25 is available for every policy tick; source-certified old-to-new five-frame history with initial backfill is deterministic. Native125 bytes were not saved, so no direct byte comparison or new encoder replay was performed.' if native_observation_present else
                'Neither exact25 nor125 was saved; do not infer them from the physical trace for this plan.'),
            observation_conditions='2999 original encoder controls; nominal play sensors, no noise/delay; no resets in retained full first episodes. Requested yaw is zero, but native heading-hold produces nonzero yaw command corrections.',
            full_run_wall_duration_s=30.,episodes=96,episode_duration_s=30.,
            valid_policy_frames=int(valid.sum()),aggregate_environment_seconds=float(valid.sum()*.01),
            true_COM_xyz_m_s_ranges=ranges(truth[valid]),command_vx_yaw_height_ranges=ranges(command[valid]),phases=phases))
        print(label,'complete-run inventory',int(valid.sum()),'policy frames',flush=True)
    assert len({run['actor_state_sha256'] for run in runs}) == 1
    audit = dict(scope='Read-only data inventory and split proposal; no fitting, model-output scoring, network replay, physics or collection',
        parent_model_iteration=2999,parent_model_sha256=MODEL_SHA,policy_dt_s=.01,
        target_contract='Pre-action root-body inertial COM velocity in root link frame, m/s. Original encoder supervision uses this same three-axis value multiplied by2; not root-link velocity or whole-robot weighted COM.',
        phase_rule='Existing fixed phase_specs: initial0-1.5s; transition ends max(nominal ramp end,native command first reaches target)+1.5s; then parking or cruise. These are descriptive data bins, not a newly approved transient tolerance.',
        count_limit='Each run contains96 complete environment episodes, but only one vectorized run/seed. Neighboring100Hz frames and overlapping5-frame histories are correlated; no claim that frames are independent samples. Episodes within a shared six-case nominal run are not96 independent protocol validations.',
        runs=runs,
        proposed_split=dict(status='Proposal only; no training authorization or budget',
            primary_train_whole_runs=['shadow46'],primary_development_whole_runs=['shadow47'],
            primary_reason='Both contain original native125, same-tick3-axis targets and explicit resets; whole-run separation changes seed, acceleration and schedule without any frame-level leakage.',
            optional_training_extension_whole_runs=['compensation44'],optional_development_extension_whole_runs=['compensation45'],
            extension_condition='Use only after deterministic exact25-to125 construction is separately frozen and parity-checked against the original frozen encoder outputs; native125 was not recorded. Such a parity check is future verification, not performed in this inventory.',
            regression_reference_only=['goal43'],
            prohibited='Never random-split adjacent frames or overlapping histories; never distribute episodes from one vectorized run across training/development; no history spanning run/episode/reset boundaries.',
            final_test='None of these are a final test: every run has already been repeatedly inspected during compensation/delay diagnosis. Seed45/47 names validation do not make them untouched final tests.'),
        future_final_test_proposal=dict(status='Not collected; do not launch as part of this plan',
            whole_run_seeds=[48,49],duration_s=30,environments_per_run=96,
            requests='Predeclare distinct held-out acceleration/timing sequences spanning initial stand, zero, forward+.5, brake, backward-.5, brake; preserve requested0/±.5 andheight.26 for same-condition parent/candidate comparisons.',
            freeze_before_collection='Exact schedules, phase definitions, scoring gates and eventual single checkpoint-selection rule; collect once after method choice and keep targets out of training/development.',
            limits='All current runs use fixedheight.26, zero requested yaw with native heading-hold corrections and nominal play conditions; estimator data do not establish robustness to noise, randomization, slopes, commanded turning or other heights. A new raw-data holdout also cannot by itself establish closed-loop behavior.'),
        height='Backward low height remains unresolved; no height reward/control changes proposed.',
        execution=dict(training=False,fitting=False,network_inference=False,model_output_scoring=False,simulation=False,collection=False))
    (output/'data-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines = ['# 2999估速器现有数据盘点与整运行划分建议','',
        '本次仅盘点已有数据，不训练、不重新评分、不跑网络或仿真。父模型2999 SHA为`'+MODEL_SHA+'`；五组原encoder运行身份一致、均96/96完成30秒。','',
        '| 完整运行 | 原生history125 | 精确Actor25 / 同拍3轴COM | reset事件数组 | 使用建议 |',
        '|---|---|---|---|---|']
    for run in runs:
        a=run['availability']
        lines.append(f"| {run['id']} seed{run['seed']} | {'有' if a['native_history125'] else '无'} | {'均有' if a['exact_native_Actor25'] and a['same_tick_three_axis_COM'] else '均无直接字段，COM仅在trace'} | {'有' if a['explicit_reset_after_policy'] else '无；valid/qpos_live证实首回合连续'} | {run['usability']} |")
    lines += ['', 'seed46/47的五帧原生历史逐槽与实际Actor25核对，最大差0，可直接用于离线估速训练。seed44/45每拍精确25槽和同拍3轴COM完整，按源码确认的首帧填充/旧到新规则可以确定性组历史；但没有保存原生125，本轮不进行新的网络复现验证，因此先列为可选扩充。旧seed43缺精确25/125，不从物理trace猜测重建。','',
        '每个运行288,000个100 Hz有效frame、96个30秒首回合，累计2,880环境秒。每run实际上只有30秒并行采集；相邻帧及重叠历史相关，不能把288,000称作独立样本。','',
        '| 运行 | 阶段 | 有效frame | 环境秒 | 含该阶段episode | 连续段时长最小/中位/最大(s) |',
        '|---|---|---:|---:|---:|---|']
    for run in runs:
        for name,p in run['phases'].items():
            d=p['contiguous_duration_s']
            lines.append(f"| {run['id']} | {name} | {p['valid_policy_frames']} | {p['aggregate_environment_seconds']:.2f} | {p['episodes_with_phase']} | {d['minimum']:.2f}/{d['median']:.2f}/{d['maximum']:.2f} |")
    lines += ['', '各阶段真实3轴COM和实际命令的最小/最大范围见[data-audit.json](data-audit.json)，没有计算估速误差或拟合参数。阶段按已有phase_specs划分：初始化0–1.5秒，实际命令到位与名义斜坡结束取较晚者后再等1.5秒划入停车/匀速；这只是描述数据，不代表用户已允许1.5秒超限。','',
        '最推荐划分：整个seed46用于训练、整个seed47用于开发；二者均直接记录125，seed、加速度和命令时序不同。可选扩充是整个seed44加训练、整个seed45加开发，但先单独冻结确定性组历史过程并做原模型复现核验。禁止按相邻帧随机切分，也不把同一次并行运行的episode分到两侧。','',
        'seed45/47虽然目录叫validation，均已反复查看，必须标为开发数据，不能作为最终测试。建议方法和模型选择规则确定后再事前固定seed48/49的新完整运行，使用不同起停/反向时序；本次不采集、不指定训练预算。后续父模型与候选闭环仍须同条件对比，数据集验证不能代替96/96存活、跟踪、pitch、摆腿及漂移检查。','',
        '当前数据均固定高度0.26米、外部偏航请求为零、无噪声/延迟的nominal play；原生航向保持仍会产生非零偏航修正命令（实测约−0.147～+0.053 rad/s），不能把它写成实际命令始终零偏航。这些数据不能证明对其他高度、主动转弯、扰动或传感噪声的泛化。后退欠高继续独立未解决，不改高度奖励。','']
    (output/'data-audit.md').write_text('\n'.join(lines))
    print(json.dumps(dict(runs=len(runs),total_policy_frames=sum(r['valid_policy_frames'] for r in runs),
        native125_direct_runs=['shadow46','shadow47'],training=False,scoring=False)))


def verify_legacy_history(args):
    """The sole extra check needed to choose whole-run train/development sets."""
    import torch
    from rsl_rl.modules import MLP
    torch.set_num_threads(1)
    output = args.output.resolve()
    audit_path = output/'data-audit.json'
    audit = json.loads(audit_path.read_text())
    assert 'legacy_history_parity' not in audit
    root = Path(__file__).resolve().parents[3]
    meta = json.loads((root/RUNS[0][1]/'encoder.json').read_text())
    checkpoint = Path(meta['checkpoint'])
    assert digest(checkpoint) == MODEL_SHA
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['iter'] == 2999
    state = {key.removeprefix('encoder.'):value for key,value in saved['actor_state_dict'].items() if key.startswith('encoder.')}
    encoder = MLP(125,3,(128,64),'elu')
    encoder.load_state_dict(state,strict=True)
    encoder.eval().requires_grad_(False)
    before = {key:value.detach().clone() for key,value in encoder.state_dict().items()}
    source_sha = digest(Path(__file__))
    source_copy = output/'history-replay-source.py'
    assert not source_copy.exists()
    shutil.copy2(__file__,source_copy)
    rows = []
    with torch.inference_mode():
        for label,relative in RUNS[:2]:
            trace = root/relative/'encoder.npz'
            # 不打开trueCOM标签，只对照当时保存的原encoder输出。
            with np.load(trace,allow_pickle=False) as data:
                frames, original, valid = data['policy_observation'],data['encoder_m_s'],data['policy_valid']
                live = data['qpos_live']
            assert frames.shape == (3000,96,25) and original.shape == (3000,96,3)
            assert valid.all() and live.all()
            indices = np.maximum(0,np.arange(3000)[:,None]+np.arange(-4,1))
            history = frames[indices].transpose(0,2,1,3).reshape(3000,96,125)
            flattened = history.reshape(-1,125)
            predicted = np.empty_like(original.reshape(-1,3))
            for begin in range(0,len(flattened),8192):
                end = min(begin+8192,len(flattened))
                predicted[begin:end] = (encoder(torch.from_numpy(flattened[begin:end]))/2.).numpy()
            residual = abs(predicted.reshape(original.shape)-original)
            maximum = float(residual.max())
            passed = bool(np.isfinite(residual).all() and maximum <= 1e-5)
            rows.append(dict(id=label,trace=str(trace),trace_sha256=digest(trace),
                trace_keys_opened=['policy_observation','encoder_m_s','policy_valid','qpos_live'],
                frames=288000,max_abs_output_residual_m_s=maximum,threshold_m_s=1e-5,passed=passed,
                per_axis_max_abs_residual_m_s=residual.max(axis=(0,1)).tolist()))
            print(label,'frozen encoder replay max residual',maximum,'passed',passed,flush=True)
    assert all(torch.equal(before[key],value) for key,value in encoder.state_dict().items())
    assert digest(Path(__file__)) == digest(source_copy) == source_sha
    passed = all(row['passed'] for row in rows)
    audit['legacy_history_parity'] = dict(scope='Original encoder output parity only; no truth labels or model-error scoring',
        source=str(source_copy),source_sha256=source_sha,checkpoint_sha256=MODEL_SHA,
        history_rule='A[max(0,k-4)]...A[k], first-frame backfill, oldest-to-newest, no reset crossing',
        encoder_eval=True,requires_grad=False,encoder_unchanged=True,training_updates=0,
        truth_labels_opened=False,model_error_scoring=False,simulation=False,
        passed=passed,runs=rows,execution_receipt=None)
    if passed:
        audit['proposed_split'].update(primary_train_whole_runs=['compensation44','shadow46'],
            primary_development_whole_runs=['compensation45','shadow47'],
            primary_reason='44/45 deterministic125 reconstructs original frozen2999 encoder outputs within1e-5;46/47 directly stores native125. Whole-run separation retains seed, acceleration and schedule differences without frame leakage.',
            optional_training_extension_whole_runs=[],optional_development_extension_whole_runs=[],
            extension_condition='No further data selection check needed; both older exact25 runs passed the frozen encoder parity gate.')
        for run in audit['runs']:
            if run['id'] in ('compensation44','compensation45'):
                run['usability'] = 'usable_after_verified_deterministic25_to125_rebuild'
                run['history_encoder_parity_verified'] = True
    else:
        audit['proposed_split']['extension_condition'] = '44/45 failed the sole frozen encoder parity gate; exclude both. Use only native46 for training and47 fordevelopment; stop this parity round without further patching.'
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    md_path = output/'data-audit.md'
    text = md_path.read_text()
    text += '\n## 仅影响数据选择的旧历史复现核对\n\n'
    text += '不读取真速度，只把44/45精确25槽按既定规则组125，输入2999冻结原估速网络，与保存的原encoder输出比较。门槛在运行前固定为最大绝对差≤1e-5 m/s，无训练、无误差评分或仿真。\n\n'
    for row in rows:
        text += f"- {row['id']}：最大输出差{row['max_abs_output_residual_m_s']:.9g} m/s，{'通过' if row['passed'] else '未通过'}。\n"
    if passed:
        text += '\n最终推荐整个seed44+46训练、整个seed45+47开发。两侧各576,000个frame/192个30秒episode；不打散相邻帧。它们均已查看过，仍不是最终测试。前述44/45待核验条件现已满足；此处更新数据选择，不启动训练。\n'
    else:
        text += '\n复现未通过，明确排除44/45，只推荐原生46训练、47开发，结束核对，不继续修补。\n'
    md_path.write_text(text)
    print(json.dumps(dict(history_parity_passed=passed,training=False,truth_opened=False,scoring=False)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-legacy-history',action='store_true')
    args=parser.parse_args()
    verify_legacy_history(args) if args.verify_legacy_history else main(args)
