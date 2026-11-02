"""Read completed same-state single-step probes; no new physics or NN calls."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=args.experiment;report=json.loads((root/'run-r2/report.json').read_text())
    budget=json.loads((root/'budget.json').read_text());failed=json.loads((root/'run/failure.json').read_text())
    checks=[]
    def verify(name,condition):
        assert condition,name
        checks.append(name)
    verify('nine_cases_complete',report['complete'] and len(report['cases'])==9)
    verify('inside_new_window',budget['start_unix']<=report['started_unix']<report['completed_unix']<budget['deadline_unix'])
    verify('failed_attempt_no_steps',failed['counts']['cpu_steps']==failed['counts']['gpu_environment_steps']==0)
    verify('144_total_environment_steps',report['counts']['cpu_steps']==9 and report['counts']['gpu_environment_steps']==135)
    verify('no_network_or_optimizer',report['counts']['model_forwards']==report['counts']['optimizer_updates']==0)
    verify('all_input_identity_checks_passed',all(r['identical_input_state_ctrl_warmstart'] for r in report['cases']))
    for path,digest in report['inputs_sha256'].items():verify('input_sha:'+path,sha(Path(path))==digest)
    protected=json.loads((root/'protected-source.json').read_text())
    for path,digest in protected.items():verify('protected_source:'+path,sha(Path(path))==digest)
    rows=[]
    for case in report['cases']:
        row=dict(name=case['name'],velocity_gap=case['max_velocity_difference'],engines={})
        for engine in ['cpu','gpu']:
            d=case[engine]['start_forward']
            f=np.array([c['force_world_xyz'] for c in d['contacts']]);p=np.array([c['position'] for c in d['contacts']])
            normal=f[:,2].sum();cop=(p*f[:,2,None]).sum(axis=0)/normal
            torque=np.cross(p-np.array(case['common_inputs']['qpos'][:3]),f).sum(axis=0)
            row['engines'][engine]=dict(ncon=d['ncon'],normal_force_N=float(normal),
                center_of_normal_force_world_m=cop.tolist(),contact_torque_about_root_Nm=torque.tolist(),
                equality_position_error=d['equality_position_max_abs'],solver_iterations=d['solver_iterations'],
                equality_acceleration_residual=d['equality_acceleration_residual_max_abs'],
                dynamics_residual=d['dynamics_residual_max_abs'])
        rows.append(row)
    first=report['cases'][0]
    primary=rows[0];c=primary['engines']['cpu'];g=primary['engines']['gpu']
    verify('primary_contact_count_difference',c['ncon']==1 and g['ncon']==2)
    verify('same_contact_count_pattern_in_all9',all(r['engines']['cpu']['ncon']==1 and r['engines']['gpu']['ncon']==2 for r in rows))
    repeat=max(abs(np.array(first['gpu']['after_step']['qvel'])-
                   np.array(report['cases'][-1]['gpu']['after_step']['qvel'])))
    verify('gpu_single_step_repeat_much_smaller_than_gap',repeat<primary['velocity_gap']*.001)
    normal_shift=np.array(g['center_of_normal_force_world_m'])-np.array(c['center_of_normal_force_world_m'])
    versions={k:importlib.metadata.version(k) for k in ['mujoco','mujoco-warp','warp-lang']}
    result=dict(primary=primary,cases=rows,normal_force_center_shift_m=normal_shift.tolist(),
        gpu_single_step_repeat_max_velocity_difference=float(repeat),versions=versions,
        counts_including_failed_attempt=dict(cpu_steps=9,gpu_environment_steps=135,cpu_forwards=19,
            gpu_environment_forwards=270,cpu_set_const=2,gpu_set_const=2,model_forwards=0,optimizer_updates=0),
        checks=checks,passed=True,additional_physics_steps=0,additional_model_forwards=0,
        limitations=report['limitations']+['Contact manifold differs directly, but all extra11deg closed-loop pitch is not causally apportioned.',
            'No contact-point removal/matching intervention was performed.'])
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    cp=first['cpu']['start_forward']['contacts'];gp=first['gpu']['start_forward']['contacts']
    origin=cp[0]['position'][1]
    for engine,contacts,delta,color in [('CPU',cp,-1.5,'#c44736'),('GPU',gp,1.5,'#178457')]:
        axes[0].bar([(v['position'][1]-origin)*1000+delta for v in contacts],
                    [v['force_world_xyz'][2] for v in contacts],width=3,color=color,label=engine)
    axes[0].set(xlabel='Contact position along wheel axle (mm; CPU point = 0)',
                ylabel='Normal force (N)',title='Same state: CPU 1 contact, GPU 2 contacts')
    axes[0].legend();axes[0].grid(axis='y',alpha=.2)
    names=['GPU/local/saved','GPU/local/zero','GPU/grid/saved','GPU/grid/zero',
           'CPU/local/saved','CPU/local/zero','CPU/grid/saved','CPU/grid/zero','GPU/local/repeat']
    axes[1].barh(names,[r['velocity_gap'] for r in rows],color='#547eaa')
    axes[1].invert_yaxis();axes[1].set(xlabel='Max joint velocity gap after 2ms (rad/s)',
        title='Gap persists across all 9 pairs')
    axes[1].grid(axis='x',alpha=.2)
    fig.savefig(args.output/'contact-comparison.png',dpi=160);plt.close(fig)
    source_dir=args.output/'source';source_dir.mkdir()
    for file in ['probe_single_step.py','report_single_step.py']:
        source=(root/'report-final/source'/file if file=='probe_single_step.py' else Path(__file__))
        shutil.copy2(source,source_dir/file)
    packages=Path(importlib.metadata.distribution('mujoco-warp').locate_file('mujoco_warp/_src'))
    for file in ['collision_primitive.py','collision_driver_test.py']:
        shutil.copy2(packages/file,source_dir/('installed-'+file))
    (args.output/'source-sha.json').write_text(json.dumps({p.name:sha(p) for p in source_dir.iterdir()},indent=2)+'\n')
    lines=['# 0.034s同状态、同8路输入的单2ms响应对照','',
        '**已复现并定位到明确的接触生成差异：CPU原生MuJoCo给左轮1个接触点，GPU MuJoCo Warp给2个；在逐位相同的姿态、速度、8路ctrl和热启动下，仅推进2ms仍产生约0.01056rad/s的被动膝速度差。**','',
        '热启动是求解器用上一拍结果作为本拍起点。本次同时对照保留/清零热启动、GPU/CPU来源状态和原点/训练网格位置；9组均有1对2个接触点，不支持将这一拍差异归于热启动或网格位置。','',
        '## 相同状态下的接触','',
        '| 起始状态的左轮 | CPU | GPU |','|---|---:|---:|',
        f"| 接触点数 | {c['ncon']} | {g['ncon']} |",
        f"| 法向力合计(N) | {c['normal_force_N']:.6f} | {g['normal_force_N']:.6f} |",
        '| 各点法向力(N) | 53.6603 | 32.0487＋22.6156 |',
        '| 接触深度(mm) | 0.198160 | 0.198157、0.029419 |',
        f"| 约束求解迭代次数 | {c['solver_iterations']} | {g['solver_iterations']} |",
        f"| 闭链位置误差最大值(m) | {c['equality_position_error']:.9f} | {g['equality_position_error']:.9f} |",'',
        f"GPU新增点在轮轴方向距原点约25mm，法向合力作用位置比CPU偏移{normal_shift[1]*1000:.4f}mm；总法向力只差{g['normal_force_N']-c['normal_force_N']:.4f}N，但作用位置改变了转动效果。关于共同根原点的接触力矩CPU={c['contact_torque_about_root_Nm']}、GPU={g['contact_torque_about_root_Nm']}Nm。法线均世界+z，滑动摩擦均0.5。两边8路实际执行器输出最大差约1.17e−7（6电机Nm、2弹簧N），支持驱动输入本次已对齐。",'',
        '## 为什么会生成不同接触点','',
        '机器人轮子的碰撞形状是STL网格，与平面接触时，两引擎采用不同的多接触点选择方法。安装版本的`collision_primitive.py::plane_convex`搜索多个网格顶点，`plane_convex_wrapper`最多写4点；`collision_driver_test.py`约604行明确说明原生MuJoCo和Warp对plane/mesh生成多接触点的heuristics不同，并允许Warp点数不少于原生。此次实际出现1对2，与该实现差别一致，不是仅凭float32/float64猜测。两边包版本均3.11.0。', '',
        '源码保存为[source/installed-collision_primitive.py](source/installed-collision_primitive.py)和[source/installed-collision_driver_test.py](source/installed-collision_driver_test.py)；本仓轮碰撞定义见`resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml`。本次未修改这些文件。', '',
        '## 9组结果','',
        '| 输入来源/位置/热启动 | 单拍后最大速度差 |','|---|---:|']
    lines.extend([f"| {r['name']} | {r['velocity_gap']:.9f} |" for r in rows])
    lines.extend(['',
        f"主例差值出现在左被动膝，单位rad/s；GPU相同输入重复单步差{repeat:.3g}，比CPU/GPU差小约五个数量级。旧闭环同一拍约0.010594，本次同状态单步约0.010557，量级吻合。用CPU来源状态也复现；把热启动清零或移至网格(6,3)m都未消除。",'',
        '## 检查、计算范围与限制','',
        f"已完成{len(checks)}项输入/预算/产物/生产源SHA核对。首次运行在CPU一次forward后的数据整理中因枚举比较TypeError退出，积分0；修正仅诊断脚本，原失败目录/日志保留。相同预算内r2退出0，起止{report['started_unix']:.6f}～{report['completed_unix']:.6f}，成功计算{report['completed_unix']-report['started_unix']:.3f}s，所有真实物理调用均在17:57～18:02新窗口内。合计CPU9＋GPU135=144环境积分步，额外显式CPU forward19（含失败1次）、GPU forward270环境次，不含step内部自带的forward；模型常数重算CPU/GPU各2次。神经模型前向与优化器更新均0，没有继续消耗剩余时间追加场景。",'',
        '共享单拍输入统一为float32可精确表达值，CPU双精度数组恢复相同值，GPU batch15为同一状态的15份，指定world2比较；不是重新运行原异质15环境全过程。8路ctrl包含6电机与2弹簧；使用XML增益前ctrl，不把实际输出重复乘增益。每拍开始和结束分别forward，使接触力与该状态同拍；after_step的qpos/qvel在终点、派生力仍对应起点，报告明确分相。', '',
        '证明范围：相同输入下接触点集合确实不同，且同一2ms运动差异可重现。尚未实施“GPU删除新增点”的干预，因此不声称额外接触点独立解释全部误差，更不把起步峰值24°对10°全部归给这一拍。未改正式模型、XML、PD、奖励、扰动配置或H7，没有训练/新策略/完整行为验收。', '',
        '下一步优先只在独立诊断中，让GPU保留与CPU一致的接触点，再重做相同2ms。若0.01056rad/s差降到数值重复误差附近，就能进一步确认接触点选择的独立贡献；不应先改奖励或加训。', '',
        '[接触力分布图](contact-comparison.png)、[摘要与校验](report.json)、[完整原始结果](../run-r2/report.json)、[原始张量](../run-r2/raw.npz)、[预算](../budget.json)。', '',
        '只读报告复现：', '', '```bash',
        f'mjlab_training/.venv/bin/python scripts/agent/fudan_ppo_walk/report_single_step.py --experiment {root} --output <新目录>',
        '```','新的动态复现必须另建时间窗口和新输出目录，不能复用已过期预算。'])
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks),normal_force_center_shift_m=normal_shift.tolist(),
        velocity_gap=primary['velocity_gap'],additional_physics_steps=0,additional_model_forwards=0)))


if __name__=='__main__':main()
