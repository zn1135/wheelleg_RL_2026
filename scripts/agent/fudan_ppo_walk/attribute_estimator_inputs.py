"""Frozen 7000 encoder input attribution on recorded GPU/CPU histories only."""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import time

import numpy as np


GROUPS = {
    'gyro': [0, 1, 2],
    'gravity': [3, 4, 5],
    'leg_position': [9, 10, 11, 12],
    'leg_motor_velocity': [13, 14, 16, 17],
    'wheel_motor_velocity': [15, 18],
    'previous_leg_action': [19, 20, 22, 23],
    'previous_wheel_action': [21, 24],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attribution(values):
    """Exact Shapley value: average each group's effect over every swap order."""
    n = len(GROUPS)
    result = np.zeros(values.shape[:-1] + (n,), dtype=np.float64)
    for i in range(n):
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            k = mask.bit_count()
            weight = math.factorial(k) * math.factorial(n-k-1) / math.factorial(n)
            result[..., i] += weight * (values[..., mask | (1 << i)] - values[..., mask])
    return result


def hybrids(cpu, gpu):
    shape = (len(cpu), 1 << len(GROUPS), 5, 25)
    result = np.broadcast_to(cpu[:, None].reshape(len(cpu), 1, 5, 25), shape).copy()
    for mask in range(shape[1]):
        for i, columns in enumerate(GROUPS.values()):
            if mask & (1 << i):
                result[:, mask, :, columns] = gpu.reshape(-1, 5, 25)[:, :, columns].transpose(2, 0, 1)
    return result.reshape(len(cpu), shape[1], 125)


def check():
    columns = [c for group in GROUPS.values() for c in group]
    assert len(set(columns)) == len(columns) == 22
    assert sorted(columns + [6, 7, 8]) == list(range(25))
    cpu = np.arange(250, dtype=np.float32).reshape(2, 125)
    gpu = cpu + 1000
    gpu.reshape(2, 5, 25)[:, :, 6:9] = cpu.reshape(2, 5, 25)[:, :, 6:9]
    mixed = hybrids(cpu, gpu)
    assert np.array_equal(mixed[:, 0], cpu)
    assert np.array_equal(mixed[:, -1], gpu)
    for i, cols in enumerate(GROUPS.values()):
        expected = cpu.reshape(2, 5, 25).copy()
        expected[:, :, cols] = gpu.reshape(2, 5, 25)[:, :, cols]
        assert np.array_equal(mixed[:, 1 << i], expected.reshape(2, 125))
    weights = np.arange(1, 8, dtype=np.float64)
    values = np.array([sum(weights[i] for i in range(7) if mask & (1 << i))
                       + (8 if mask & 1 and mask & 2 else 0) for mask in range(128)])
    expected = weights.copy()
    expected[:2] += 4
    assert np.allclose(attribution(values), expected, atol=1e-12)
    assert np.allclose(attribution(np.ones(128)), 0)
    return {'passed': True, 'checks': 12, 'physics_steps': 0,
            'encoder_forward_calls': 0, 'optimizer_updates': 0}


def write_report(root):
    """Read saved inference results; no model loading or new forwards."""
    assert not (root/'README.md').exists() and not (root/'input-attribution.png').exists()
    result = json.loads((root/'results/report.json').read_text())
    budget = json.loads((root/'budget.json').read_text())
    source = Path(budget['recording'])
    prior = json.loads((source/'estimator-analysis-r2/report.json').read_text())
    pitch = {}
    for name in ['cpu_pd_closed', 'gpu_closed']:
        with np.load(source/'runs'/f'{name}.npz') as data:
            pitch[name] = float(data['states_row'][150, 13])  # Recorded units already degrees.
    at = result['at030']
    with np.load(root/'results/inference.npz') as data:
        values = data['outputs_m_s'][300:2988, 0].reshape(21, 128)
        attitude_and_gyro = float(values[-1, 3])
    names = list(GROUPS)
    labels = ['Gyro', 'Gravity direction', 'Leg position', 'Leg motor velocity',
              'Wheel motor velocity', 'Previous leg action', 'Previous wheel action']
    contribution = np.asarray(result['cpu_minus_gpu_contributions_m_s'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for name, color, label in [('gpu_closed', '#178457', 'GPU'), ('cpu_pd_closed', '#c44736', 'CPU')]:
        row = prior['runs'][name]
        t = np.arange(100)*.01
        axes[0].plot(t, np.asarray(row['estimated_m_s'])[:, 0], color=color, label=label+' estimate')
        axes[0].plot(t, np.asarray(row['training_definition_labels_m_s'])[:, 0], color=color,
                     ls='--', label=label+' target')
    axes[0].set(xlim=(.1, .31), xlabel='Time (s)', ylabel='Body vx (m/s)',
                title='Same-definition target versus estimator')
    axes[0].legend(fontsize=8)
    colors = ['#547eaa', '#c44736', '#d89d36', '#7d64a4', '#359d91', '#788142', '#888888']
    axes[1].barh(labels, contribution[-1], color=colors)
    axes[1].axvline(0, color='black', lw=.7)
    axes[1].invert_yaxis()
    axes[1].set(xlabel='Contribution to CPU minus GPU estimate (m/s)',
                title='At 0.30s: exact 7-group attribution; sum = +0.554')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(root/'input-attribution.png', dpi=160)
    plt.close(fig)
    lines = ['# 7000轮CPU运动后估速失准：输入原因', '',
        '直接定位到冻结估速网络对姿态输入的敏感性，不能归因于CPU浮点精度；更早的物理轨迹分叉原因及估速对倾角的独立贡献尚未证明。', '',
        '## 同一时刻对照', '',
        '| 0.30s | CPU | GPU |', '|---|---:|---:|',
        f"| 训练同口径vx（10ms位移差分，m/s） | {at['cpu_label_m_s']:.6f} | {at['gpu_label_m_s']:.6f} |",
        f"| 估速vx（m/s） | {at['cpu_estimate_m_s']:.6f} | {at['gpu_estimate_m_s']:.6f} |",
        f"| 俯仰角（°） | {pitch['cpu_pd_closed']:.4f} | {pitch['gpu_closed']:.4f} |", '',
        'CPU/GPU真实后退速度只差0.0495m/s，但估速相差0.5539m/s且方向相反；CPU误差+0.4532，GPU误差−0.1501m/s。', '',
        '## 冻结网络直接替换输入', '',
        '在CPU的125维历史中，只把一类信号的5帧替换成GPU同时间信号，其余输入保持CPU原值。这里“姿态”是重力在机体坐标中的方向，来自真实姿态，不是新增传感器。', '',
        '| 替换的输入 | 0.30s估速vx（m/s） |', '|---|---:|',
        f"| 不替换 | {at['cpu_estimate_m_s']:.6f} |",
        *[f"| {label} | {result['groups'][name]['replace_this_group_only_at030_estimate_m_s']:.6f} |" for name, label in zip(names, labels)],
        f'| 重力方向＋陀螺仪一起替换 | {attitude_and_gyro:.6f} |', '',
        '重力方向单独替换即从−0.955变为−1.601m/s；仅替换重力x分量变为−1.598，轮速单独替换几乎不变。逐步将姿态从CPU插值到GPU，估速单调变为−.955/−1.072/−1.231/−1.415/−1.601。归因指网络依赖哪些输入；不表示把GPU姿态喂给CPU是可部署的修复。', '',
        '## 考虑信号之间的相互影响', '',
        '计算7类信号的全部128种替换组合，把每类信号在所有替换顺序中的影响取平均（精确Shapley归因）。正值把CPU估速推向更慢后退，负值抵消该偏差；不是错误百分比或物理因果分摊。', '',
        '| 输入 | .30s CPU−GPU输出差贡献（m/s） | .15～.30s平均贡献（m/s） |', '|---|---:|---:|',
        *[f"| {label} | {result['groups'][name]['cpu_minus_gpu_contribution_at030_m_s']:+.6f} | {result['groups'][name]['mean_contribution_015_to030_m_s']:+.6f} |" for name, label in zip(names, labels)], '',
        '姿态贡献+.7156m/s，其他类别合计−.1618，最终输出差+.5539。整个.15～.30s区间，姿态平均贡献+.2346，净输出差+.2507；并非仅挑单点。', '',
        '## 排除项与解释边界', '',
        f"- 同一冻结网络在CPU重算三条真实输入，最大预测误差分别为{result['reproduction_max_abs_m_s']} m/s；比0.554差异小五个数量级，排除本例模型执行精度为主因。",
        '- 上轮已核对姿态/腿角/电机速度/动作打包，最大约1e−7，倍率与标签同拍正确。姿态差异反映机器人真的更倾斜，不能当作IMU坐标错误。',
        '- 共同的复位缺速度信息、共同的电机曲线不能独立解释CPU运动后新增误差；刷新时序A/B此前已否定。',
        '- 直接证据支持“网络在CPU更倾斜的状态下，把后退速度估得偏小”。可能涉及训练中姿态与速度关联、这类组合状态覆盖或学习不足，但没有原训练逐状态数据，不能证明具体哪一种。',
        '- 混合历史不保证物理可实现，不能证明估速是CPU先倾斜的起因，也不能证明用姿态替换能修复闭环。下一步应同初态仅将估速输出换为同口径真值，比较峰值倾角及轮目标触边时长；需要新的物理实验预算，本次未执行。', '',
        '## 本次计算与复现', '',
        f"原7000-v2 checkpoint SHA `{result['checkpoint_sha256']}`；完全从归档network/contract加载，未导入当前扰动开关实现。17:40:00～17:41:00推理窗口内，实际起止{result['started_unix']:.6f}～{result['completed_unix']:.6f}，6次批量encoder前向/3045条输入、actor前向0、物理步进0、优化器更新0；权重及所有输入文件前后SHA/张量核对不变。无新训练或仿真行为验收。", '',
        '无物理、无网络前向检查：', '```bash',
        'mjlab_training/.venv/bin/python scripts/agent/fudan_ppo_walk/attribute_estimator_inputs.py --check-only',
        '```', '已有结果报告生成：', '```bash',
        f'mjlab_training/.venv/bin/python scripts/agent/fudan_ppo_walk/attribute_estimator_inputs.py --report-only {root}',
        '```', '推理复现必须另设新的60秒budget.json，不能复用本次已过期窗口。', '',
        '[数值与逐时贡献](results/report.json)、[全部混合输入及预测](results/inference.npz)、[运行输出](process.log)、[预算](budget.json)。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report': str(root/'README.md'), 'pitch_deg': pitch,
                      'attitude_and_gyro_swap_m_s': attitude_and_gyro,
                      'new_encoder_forwards': 0, 'new_physics_steps': 0}))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--report-only', type=Path)
    parser.add_argument('--budget', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    checks = check()
    if args.report_only:
        write_report(args.report_only)
        return
    if args.check_only:
        print(json.dumps(checks))
        return
    assert args.budget and args.output
    budget = json.loads(args.budget.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def deadline():
        assert budget['start_unix'] <= time.time() < budget['deadline_unix'], 'Outside inference budget'

    deadline()
    checkpoint = Path(budget['checkpoint'])
    assert sha(checkpoint) == budget['checkpoint_sha256']
    frozen = checkpoint.parent / 'source/ct_mjlab/fudan_ppo_walk'
    source = Path(budget['recording'])
    paths = {'gpu_closed': source / 'runs/gpu_closed.npz',
             'gpu_repeat': source / 'repeat-audit/gpu_repeat.npz',
             'cpu_pd_closed': source / 'runs/cpu_pd_closed.npz'}
    inputs_sha = {str(p): sha(p) for p in [checkpoint, frozen/'network.py', frozen/'contract.py', *paths.values()]}
    recorded = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            recorded[name] = {key: data[key].copy() for key in data.files}
    cpu = recorded['cpu_pd_closed']['policy_history']
    gpu = recorded['gpu_closed']['policy_history']
    assert np.array_equal(cpu.reshape(-1, 5, 25)[:, :, 6:9], gpu.reshape(-1, 5, 25)[:, :, 6:9])
    ks = np.arange(10, 31)
    mixed = hybrids(cpu[ks], gpu[ks])
    interpolation = []
    single_features = []
    for columns in GROUPS.values():
        for alpha in np.linspace(0, 1, 5):
            row = cpu[30].reshape(5, 25).copy()
            row[:, columns] += alpha * (gpu[30].reshape(5, 25)[:, columns] - row[:, columns])
            interpolation.append(row.reshape(125))
        for column in columns:
            row = cpu[30].reshape(5, 25).copy()
            row[:, column] = gpu[30].reshape(5, 25)[:, column]
            single_features.append(row.reshape(125))
    all_inputs = np.concatenate([*[d['policy_history'] for d in recorded.values()],
                                 mixed.reshape(-1, 125), interpolation, single_features]).astype(np.float32)
    assert len(all_inputs) <= budget['maximum_encoder_rows']
    assert np.isfinite(all_inputs).all()

    import torch
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('frozen_fudan_network', frozen/'network.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = next(n for n in ast.parse((frozen/'contract.py').read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'make_network')
    call = next(n.value for n in factory.body if isinstance(n, ast.Return))
    with redirect_stdout(io.StringIO()):
        model = module.ActorCriticSequence(*[ast.literal_eval(a) for a in call.args],
                                           **{k.arg: ast.literal_eval(k.value) for k in call.keywords})
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    outputs = []
    with torch.inference_mode():
        for i in range(0, len(all_inputs), 512):
            deadline()
            # Same frozen encoder for both factual and counterfactual histories.
            outputs.append(model.encoder(torch.from_numpy(all_inputs[i:i+512])).numpy() / 2)
    outputs = np.concatenate(outputs)
    deadline()
    calls = len(range(0, len(all_inputs), 512))
    reproduction = {}
    for i, (name, data) in enumerate(recorded.items()):
        error = float(np.abs(outputs[i*100:(i+1)*100] - data['policy_estimated_velocity_scaled']/2).max())
        assert error < 2e-5, (name, error)
        reproduction[name] = error
    values = outputs[300:300+mixed.shape[0]*mixed.shape[1], 0].reshape(len(ks), 128).astype(np.float64)
    contribution = -attribution(values)  # Positive means CPU input shifts vx towards slower backward motion.
    assert np.max(np.abs(contribution.sum(axis=1) - (values[:, 0]-values[:, -1]))) < 1e-12
    offset = 300 + mixed.shape[0]*mixed.shape[1]
    interp = outputs[offset:offset+35, 0].reshape(7, 5)
    feature_outputs = outputs[offset+35:, 0]
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    for path, digest in inputs_sha.items():
        assert sha(Path(path)) == digest
    prior = json.loads((source/'estimator-analysis-r2/report.json').read_text())
    group_report = {}
    columns_order = [c for cols in GROUPS.values() for c in cols]
    for i, (name, columns) in enumerate(GROUPS.items()):
        group_report[name] = {
            'columns_in_each_frame': columns,
            'cpu_minus_gpu_contribution_at030_m_s': float(contribution[-1, i]),
            'mean_contribution_015_to030_m_s': float(contribution[5:, i].mean()),
            'replace_this_group_only_at030_estimate_m_s': float(values[-1, 1 << i]),
            'replace_this_group_last_at030_estimate_m_s': float(values[-1, 127 ^ (1 << i)]),
            'interpolate_cpu_to_gpu_at030_estimate_m_s': interp[i].tolist(),
            'cpu_history_at030_unscaled': (cpu[30].reshape(5, 25)[:, columns] /
                (.25 if name == 'gyro' else .05 if 'motor_velocity' in name else 1)).tolist(),
            'gpu_history_at030_unscaled': (gpu[30].reshape(5, 25)[:, columns] /
                (.25 if name == 'gyro' else .05 if 'motor_velocity' in name else 1)).tolist(),
        }
    result = {
        'scope': 'Original7000 v2 frozen encoder; recorded diagnostic2 backward2 height030; offline only',
        'checkpoint_sha256': budget['checkpoint_sha256'], 'inputs_sha256': inputs_sha,
        'synthetic_checks': checks, 'reproduction_max_abs_m_s': reproduction,
        'method': 'Exact Shapley over all128 subsets of7 groups across5 frames; CPU baseline, GPU replacement; contribution sign is CPU minus GPU',
        'times_s': (ks*.01).tolist(), 'cpu_minus_gpu_contributions_m_s': contribution.tolist(),
        'groups': group_report,
        'at030': {'cpu_estimate_m_s': float(values[-1, 0]), 'gpu_estimate_m_s': float(values[-1, -1]),
                  'output_gap_cpu_minus_gpu_m_s': float(values[-1, 0]-values[-1, -1]),
                  'cpu_label_m_s': prior['runs']['cpu_pd_closed']['at030s']['position_difference_label_m_s'][0],
                  'gpu_label_m_s': prior['runs']['gpu_closed']['at030s']['position_difference_label_m_s'][0]},
        'single_feature_replacements_at030': {str(c): float(v) for c, v in zip(columns_order, feature_outputs)},
        'physics_steps': 0, 'encoder_forward_calls': calls, 'encoder_rows': len(all_inputs),
        'actor_forward_calls': 0, 'optimizer_updates': 0, 'parameters_unchanged': True,
        'started_unix': started, 'completed_unix': time.time(),
        'limits': ['Hybrid histories are not guaranteed physically realizable.',
                   'Attribution explains frozen encoder output difference, not causal contribution to pitch.',
                   'No original training rollout archive; cannot prove training distribution coverage.',
                   'Both recorded factual histories are re-evaluated on CPU; this does not benchmark every inference backend.']}
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(args.output/'inference.npz', inputs=all_inputs, outputs_m_s=outputs,
                        times_s=ks*.01, contributions_m_s=contribution)
    print(json.dumps({k: result[k] for k in ['at030', 'reproduction_max_abs_m_s', 'groups',
                     'physics_steps', 'encoder_forward_calls', 'encoder_rows', 'optimizer_updates', 'completed_unix']}, indent=2))


if __name__ == '__main__':
    main()
