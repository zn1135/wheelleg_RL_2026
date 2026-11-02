"""Locate early engine divergence in existing probes; no stepping or inference."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    budget = json.loads((args.experiment/'budget.json').read_text())
    fixture = Path(budget['source_recording'])/'fixtures'
    paths = {
        'gpu': args.experiment/'runs/gpu_closed.npz',
        'cpu': args.experiment/'runs/cpu_pd_closed.npz',
        'gpu_repeat': args.experiment/'repeat-audit/gpu_repeat.npz',
        'gpu_own_actions': args.experiment/'repeat-audit/gpu_self_actions.npz',
        'cpu_gpu_actions': args.experiment/'runs/cpu_pd_gpu_actions.npz',
    }
    source = Path(budget['checkpoint']).parent/'source/ct_mjlab/envs/chuanliantui/gas_spring.py'
    spec = importlib.util.spec_from_file_location('frozen_gas_spring', source)
    spring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spring)
    tracked = [*paths.values(), source, fixture/'mapping.json', fixture/'model.mjb']
    hashes = {str(p): sha(p) for p in tracked}
    data = {}
    for name, path in paths.items():
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['complete'] and meta['states'] == 501
        with np.load(path, allow_pickle=False) as saved:
            data[name] = {k: saved[k].copy() for k in saved.files}
        assert all(np.isfinite(v).all() for v in data[name].values())
    g, c = data['gpu'], data['cpu']
    assert np.array_equal(g['states_qpos'][0], c['states_qpos'][0])
    assert np.array_equal(g['states_qvel'][0], c['states_qvel'][0])
    assert np.array_equal(g['policy_action_raw'], data['cpu_gpu_actions']['policy_action_raw'])
    assert np.array_equal(g['policy_action_raw'], data['gpu_own_actions']['policy_action_raw'])
    mapping = json.loads((fixture/'mapping.json').read_text())
    # Loading model metadata does not create simulation data or solve dynamics.
    model = mujoco.MjModel.from_binary_path(str(fixture/'model.mjb'))
    angular, labels = [], {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = int(model.jnt_dofadr[j])
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            for i in range(6):
                labels[adr+i] = f'{name}:{i}'
            angular.extend(range(adr+3, adr+6))
        else:
            assert model.jnt_type[j] in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]
            labels[adr] = name
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                angular.append(adr)
    spring_forces = {name: spring.gas_spring_force(d['pd_qpos_pre'][:, mapping['spring_q']],
                     d['pd_qvel_pre'][:, mapping['spring_v']]) for name, d in data.items()}
    timeline = []
    for k in [0, 15, 16, 17, 18, 20, 25, 50, 75, 100, 125, 150]:
        row = {'time_s': k*.002, 'cpu_pitch_deg': float(c['states_row'][k, 13]),
               'gpu_pitch_deg': float(g['states_row'][k, 13]),
               'cpu_estimate_m_s': float(c['policy_estimated_velocity_scaled'][k//5, 0]/2),
               'gpu_estimate_m_s': float(g['policy_estimated_velocity_scaled'][k//5, 0]/2)}
        for name, d in data.items():
            diff = np.abs(d['states_qvel'][k, angular] - g['states_qvel'][k, angular])
            row[name+'_angular_velocity_max_diff_rad_s'] = float(diff.max())
        row['cpu_minus_gpu_vx_m_s'] = float(c['states_row'][k, 3]-g['states_row'][k, 3])
        if k:
            # Control index k-1 produced state k; do not compare with the next control.
            row['preceding_motor_force_max_diff_Nm'] = float(np.abs(
                c['pd_actuator_force_Nm'][k-1]-g['pd_actuator_force_Nm'][k-1]).max())
            row['preceding_spring_force_max_diff_N'] = float(np.abs(
                spring_forces['cpu'][k-1]-spring_forces['gpu'][k-1]).max())
        timeline.append(row)
    k = 18
    signed = c['states_qvel'][k]-g['states_qvel'][k]
    order = sorted(angular, key=lambda i: abs(signed[i]), reverse=True)
    transition = {
        'interval_s': [.034, .036],
        'pre_qpos_max_diff': float(np.abs(c['pd_qpos_pre'][17]-g['pd_qpos_pre'][17]).max()),
        'pre_motor_force_max_diff_Nm': float(np.abs(c['pd_actuator_force_Nm'][17]-g['pd_actuator_force_Nm'][17]).max()),
        'pre_spring_force_cpu_N': spring_forces['cpu'][17].tolist(),
        'pre_spring_force_gpu_N': spring_forces['gpu'][17].tolist(),
        'largest_angular_velocity_differences': [dict(dof=labels[i], cpu_minus_gpu_rad_s=float(signed[i])) for i in order[:4]],
        'cpu_gpu_actions_vs_cpu_closed_velocity_max_diff': float(np.abs(
            data['cpu_gpu_actions']['states_qvel'][k]-c['states_qvel'][k]).max()),
    }
    result = dict(scope='Recorded7000-v2 diagnostic2; early state/control divergence, no new dynamics',
        timeline=timeline, transition_034_to036=transition,
        state_dtypes={n: str(d['states_qvel'].dtype) for n, d in data.items()},
        native_fixture_options={n: float(getattr(model.opt, n)) for n in
            ['timestep', 'solver', 'integrator', 'iterations', 'ls_iterations', 'tolerance', 'ls_tolerance', 'cone']},
        gpu_actions_exactly_replayed_by_cpu=True,
        inputs_sha256=hashes, physics_steps=0, model_forwards=0, optimizer_updates=0,
        limitations=['Recorded trajectories are not a same-state single-step intervention.',
            'Only motor forces were directly recorded; spring efforts are reconstructed from frozen source.',
            'Contact force phase differs between engines; do not infer touchdown timing from unaligned forces.',
            'No saved contact Jacobian, constraint forces, solver residual or per-contact features.',
            'Cannot separate float precision, contact implementation and closed-chain constraint solver from these records.',
            'Long fixed-action replay is unstable even within GPU; only early divergence is used here.'])
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# CPU/GPU状态差异的先后关系', '',
        '已有记录支持：起步接触附近先出现动力学响应差异，之后输入/动作分叉与估速偏差逐步扩大。不能把“估速失准”直接当作最早原因，也未证明接触求解的具体哪一项导致差异。', '',
        '## 早期差异不是由不同神经动作首先触发', '',
        '初始qpos/qvel逐位一致，0.034s前最大角速度差约0.0002rad/s。0.034→0.036s时，六电机实际力矩最大只差0.0001605Nm，弹簧按原7000源复算见JSON，位置仍很接近；随后左被动膝速度差约0.01059rad/s、机体roll角速度差约0.00634rad/s。', '',
        '同一GPU闭环重复在0.036s的最大速度差仅5.96e−8；GPU同动作回放约1.19e−7。CPU完整使用GPU保存的相同动作，0.036s与GPU差仍约0.01060，和CPU闭环几乎相同。这排除了最初神经输出微小浮点差为该拍分叉主因；不是用后续1s开环发散峰值作因果分摊。', '',
        '## 后续扩大', '',
        '| 时刻(s) | CPU pitch(°) | GPU pitch(°) | CPU估速(m/s) | GPU估速(m/s) |', '|---|---:|---:|---:|---:|']
    for row in timeline:
        if row['time_s'] in [0.1, .15, .2, .25, .3]:
            lines.append(f"| {row['time_s']:.2f} | {row['cpu_pitch_deg']:.4f} | {row['gpu_pitch_deg']:.4f} | {row['cpu_estimate_m_s']:.4f} | {row['gpu_estimate_m_s']:.4f} |")
    lines.extend(['',
        '此前离线输入替换已证明：0.30s估速差主要来自网络对姿态的依赖。结合本次顺序，合理的放大链条是：接触附近运动响应分叉→传感输入/策略动作分叉→CPU更倾斜时网络估速偏慢→可能进一步放大控制差异。最后一段的独立贡献尚未用真值估速干预验证，不能说每条箭头已单独证实。', '',
        '## 为什么同XML仍会不同', '',
        'GPU是MuJoCo Warp、CPU是原生MuJoCo，记录中的状态分别float32/float64。共享XML、步长、PD和主要求解参数不等于两种实现逐位相同；接触/摩擦与闭链约束同时求解时，对微小状态/数值差可能敏感。但本记录缺每接触特征、约束力及残差，没有同状态单步干预，不能定为“就是float32”或“某个摩擦参数错了”。当前只能把最早明显差异定位到这一段动力学响应。', '',
        '接触力GPU来自积分前派生量、CPU含积分后forward，两边原始接触力时间戳不能直接用于判断谁先触地。刷新频率此前A/B姿态/速度/动作逐位相同，已经排除为本例原因。', '',
        '## 优先验证', '',
        '取0.034s同一qpos/qvel、同一完整ctrl（含两弹簧）、同一warmstart，分别推进2ms，记录接触点/法线/摩擦、闭链约束误差、广义约束力和求解残差。若这一拍速度响应仍明显不同，再分离接触几何与求解误差；这一验证比先改奖励或加训更直接。本次没有新动态预算，未推进新仿真或网络。', '',
        '[全部先后关系与原输入SHA](report.json)。复现只读分析：', '', '```bash',
        f'mjlab_training/.venv/bin/python scripts/agent/fudan_ppo_walk/diagnose_engine_divergence.py --experiment {args.experiment} --output <新目录>',
        '```'])
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    for path, digest in hashes.items():
        assert sha(Path(path)) == digest
    print(json.dumps(dict(transition=transition, physics_steps=0, model_forwards=0, optimizer_updates=0), indent=2))


if __name__ == '__main__':
    main()
