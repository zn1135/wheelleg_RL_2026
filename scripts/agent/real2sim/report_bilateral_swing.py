"""Make figures and a local report from frozen bilateral identification/replay files."""
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from scripts.agent.real2sim.identify_bilateral_swing import load_trials, sha


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    trials,_ = load_trials(args.input)
    fit = json.loads((args.fit/'identification.json').read_text())
    replay = json.loads((args.replay/'report.json').read_text())
    for source,digest in replay['source_sha256'].items():
        if sha(Path(source)) != digest:
            raise ValueError('Frozen replay source has changed: '+source)
    with np.load(args.replay/'traces.npz') as a:
        timeline,state = a['time_s'],a['state']
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family'] = font.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    labels = {'left':'左腿','right':'右腿'}
    rows = []
    for row in replay['trials']:
        fitted = row['branches']['identified']
        rows.append(dict(side=row['side'],level=row['level'],held_out=row['held_out'],**fitted))
    with (args.output/'metrics.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    fig,axes = plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for axis,(side,level) in zip(axes.flat,[('left',2),('left',4),('right',2),('right',4)]):
        i,tr = next((i,tr) for i,tr in enumerate(trials) if tr['side']==side and tr['level']==level)
        valid = timeline<=tr['t'][-1]
        axis.plot(tr['t'],tr['reported_speed'],color='#222222',alpha=.75,lw=.8,label='实测摆速')
        axis.plot(timeline[valid],state[valid,len(trials)+i,1],color='#c45524',lw=1.2,label='冻结模型预测')
        axis.axvline(tr['row']['release_s'],color='#4d72a3',ls='--',lw=.8,label='撤力')
        axis.set(title=f'{labels[side]} · 留出 {level} rad/s',xlabel='试次时间 / s',ylabel='摆速 / (rad/s)')
        axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.savefig(args.output/'heldout-speed.png',dpi=160);plt.close(fig)
    for side in ('left','right'):
        fig,axes = plt.subplots(5,3,figsize=(14,14),layout='constrained')
        for r,level in enumerate(range(1,6)):
            i,tr = next((i,tr) for i,tr in enumerate(trials) if tr['side']==side and tr['level']==level)
            valid = timeline<=tr['t'][-1]
            pred = state[valid,len(trials)+i]
            axes[r,0].plot(tr['t'],tr['reported_speed'],color='#222222',lw=.7,label='实测')
            axes[r,0].plot(timeline[valid],pred[:,1],color='#c45524',lw=1,label='预测')
            axes[r,1].plot(tr['t'],np.degrees(tr['theta']-tr['theta'][0]),color='#222222',lw=.8,label='实测')
            axes[r,1].plot(timeline[valid],np.degrees(pred[:,0]-tr['theta'][0]),color='#c45524',lw=1,label='预测')
            axes[r,2].plot(tr['t'],tr['length']*1000,color='#222222',lw=.8,label='实测')
            axes[r,2].axhline(tr['length'][0]*1000,color='#c45524',ls='--',label='固定初始长度假设')
            axes[r,0].set_ylabel(f'{level} rad/s · '+('留出' if level in (2,4) else '拟合'))
            for c,ax in enumerate(axes[r]):
                ax.grid(alpha=.2);ax.axvline(tr['row']['release_s'],color='#4d72a3',ls='--',lw=.7)
                if r==0:ax.set_title(('摆速 / (rad/s)','连续摆角 / °','腿长 / mm')[c]);ax.legend(fontsize=8)
                if r==4:ax.set_xlabel('试次时间 / s')
        fig.suptitle(labels[side]+'：连续同输入降阶回放；腿长自由动力学未辨识',fontsize=14)
        fig.savefig(args.output/(side+'-all-levels.png'),dpi=130);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    phase = np.linspace(-np.pi,np.pi,400)
    for ax,side in zip(axes,('left','right')):
        p = fit['fitted'][side]['coefficients']
        periodic = p[3]*np.sin(phase)+p[4]*np.cos(phase)+p[5]*np.sin(2*phase)+p[6]*np.cos(2*phase)
        for speed in (1,3,5):
            ax.plot(np.degrees(phase),p[2]+p[1]*speed+periodic,label=f'{speed} rad/s')
        ax.set(title=labels[side]+' · 等效稳态负载',xlabel='固件摆角 / °',ylabel='命令等效负载 / N·m')
        ax.legend();ax.grid(alpha=.2)
    fig.savefig(args.output/'effective-load.png',dpi=160);plt.close(fig)
    table = ''.join('<tr><td>'+labels[r['side']]+f" {r['level']}</td><td>"+('留出' if r['held_out'] else '拟合')+
                    f"</td><td>{r['speed_rmse_rad_s']:.3f}</td><td>{r['angle_rmse_deg']:.2f}</td><td>{r['terminal_angle_error_deg']:.2f}</td></tr>" for r in rows)
    parameter_rows = ''
    for side,p in fit['fitted'].items():
        parameter_rows += f"<tr><td>{labels[side]}</td><td>{p['effective_inertia_kg_m2']:.4f}</td><td>{p['positive_direction_offset_nm']:.3f}</td><td>{p['effective_damping_nm_s_rad']:.4g}</td></tr>"
    seed_rows = ''
    for side,p in fit.get('inverse_regression_seed',fit['fitted']).items():
        seed_rows += f"<tr><td>{labels[side]}</td><td>{p['effective_inertia_kg_m2']:.4f}</td><td>{p['positive_direction_offset_nm']:.3f}</td><td>{p['effective_damping_nm_s_rad']:.4g}</td></tr>"
    findings = dict(heldout=[r for r in rows if r['held_out']],
                    heldout_mean_speed_rmse_rad_s=float(np.mean([r['speed_rmse_rad_s'] for r in rows if r['held_out']])),
                    heldout_max_angle_rmse_deg=float(max(r['angle_rmse_deg'] for r in rows if r['held_out'])),
                    result_status='FAILED continuous prediction; neither effective candidate accepted for training or deployment',
                    report_source_sha256=sha(Path(__file__)))
    force_balance = []
    for i,tr in enumerate(trials):
        n = np.searchsorted(timeline,tr['t'][-1],side='right')
        pred = state[:n,len(trials)+i].astype(float)
        p = np.asarray(fit['fitted'][tr['side']]['coefficients'])
        ix = np.clip(np.searchsorted(tr['t'],timeline[:n-1],side='right')-1,0,len(tr['t'])-1)
        u = -tr['tau'][ix].sum(1)
        angle,velocity = pred[:-1,0],pred[:-1,1]
        load = p[3]*np.sin(angle)+p[4]*np.cos(angle)+p[5]*np.sin(2*angle)+p[6]*np.cos(2*angle)
        error = p[0]*np.diff(pred[:,1])/.001+p[1]*pred[1:,1]+p[2]*np.sign(velocity)+load-u
        error = error[abs(velocity)>.5]
        row = dict(side=tr['side'],level=tr['level'],samples=len(error),
                   p99_nm=float(np.percentile(abs(error),99)) if len(error) else None,
                   max_nm=float(max(abs(error))) if len(error) else None)
        if row['p99_nm'] is not None and row['p99_nm']>.05:
            raise ValueError('GPU force balance does not match the proposed reduced equation')
        force_balance.append(row)
    findings['gpu_moving_force_balance'] = force_balance
    (args.output/'summary.json').write_text(json.dumps(findings,indent=2,ensure_ascii=False)+'\n')
    content = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>左右腿系统辨识 · 2026-11-17</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1120px;margin:36px auto;padding:0 24px;color:#222}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:32px}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;text-align:left;padding:8px}}img{{width:100%;height:auto}}.notice{{background:#fff2e4;padding:16px;border-left:4px solid #c45524}}code{{background:#f3f3f3;padding:2px 5px}}</style>
<h1>左右腿系统辨识 · 2026-11-17</h1>
<p class="notice">本次辨识没有得到可采用的参数：积分回归和整段输出误差拟合都没有通过留出档连续预测。近恒定腿长的一自由度模型不足以复现本批运动；不把候选写成真实电机、弹簧或完整并联腿标定。未写入训练、共享XML或H7。</p>
<h2>数据与工况</h2><p>左右共104,355个样本、十个完整试次；原始BIN按Jetson分块解码，再校验JRL2 CRC，与CSV逐值一致。CRC错误、试次内序号缺口、故障和板端丢样增量均为0。两份BIN各留128字节未完成串口帧，不影响已导出的完整试次。用户确认侧躺；是否翻转机身、气弹簧安装和台架精确倾角未知。</p>
<p>每侧每速度只有一次独立起转，三圈不作为三个独立试次。全部反馈年龄字段为0，不能证明反馈年龄真实为零。左腿中位长度约352mm，右腿约340mm；日志长度和摆角由同一编码器几何计算，不是额外独立传感器。</p>
<h2>模型与拟合</h2><p><code>I·θ̈ + B·θ̇ + C·sign(θ̇) + a₁sinθ + b₁cosθ + a₂sin2θ + b₂cos2θ = −(τfront + τrear)</code>。输入是最终提交的两路力矩命令之和；并非轴端测力。I、B、C全部依赖命令到实际力矩的比例。C在单向数据中也可能含恒定负载，周期项可能含倾斜重力、线束或姿态阻力，均不拆成已确定的零件参数。</p>
<p>两种方法都只用1、3、5rad/s拟合，各档等权；谐波阶数只在这三档内交叉验证选择。2、4rad/s完整留出。方法一对连续电机位置做51ms局部多项式求导，再用200ms积分方程拟合，另核100/300ms窗口与原协议速度。局部力矩残差虽小，连续回放仍失效。方法二从方法一种子出发，用整段连续摆角和速度误差直接优化同一七参数模型，不按每圈重置。CPU简化求解只用于找种子，最终另做MuJoCo Warp GPU验证。</p>
<p>方法一候选（未通过）：</p><table><tr><th>侧别</th><th>I / kg·m²</th><th>C / N·m</th><th>B / N·m·s/rad</th></tr>{seed_rows}</table>
<p>方法二候选（同样未通过，后续曲线对应这一组）：</p><table><tr><th>侧别</th><th>I / kg·m²</th><th>C / N·m</th><th>B / N·m·s/rad</th></tr>{parameter_rows}</table>
<p>两种拟合准则给出的阻尼和恒定负载明显不同，不能挑其中一组当已标定物理量。方法一右腿B落在非负下界，只表示该回归没有支持额外线性阻尼，不表示真实阻尼为零。单向运动不能分离正负向摩擦、恒定偏载、两电机各自参数、弹簧刚度/预载和执行器力矩增益。输出参数JSON中继承的积分残差、矩阵秩、条件数和窗口敏感性字段属于原始回归种子，不是输出拟合的新验证结果。</p>
<img src="effective-load.png" alt="等效负载随摆角变化">
<h2>留出档连续预测</h2><p>MuJoCo Warp GPU只注入各试次初始摆角和速度，按H7时间戳保持最终提交力矩；不重置每圈状态、不在后续喂入实测状态、不重新拟合留出档。库仑阻力由引擎约束求解。腿长固定为各试次初始值，前后关节预测共用该固定开角，完整轴向/自由伸缩动力学未验证。</p>
<img src="heldout-speed.png" alt="留出档实测和预测摆速">
<table><tr><th>侧别/档位</th><th>用途</th><th>全段速度RMSE / rad/s</th><th>连续摆角RMSE / °</th><th>末端摆角误差 / °</th></tr>{table}</table>
<p>角度按连续角计分，不按360°折返掩盖圈数漂移；包含施力前和撤力后。RMSE是整段误差的均方根。逐关节角度、起转/撤力速度和固定腿长假设误差见metrics.csv。基线只是当前XML在实测初态投影的摆动惯量、零附加阻力近似，不冒充原完整机器人自由回放。</p>
<p>本组两步长1/0.5ms共{replay['gpu_environment_steps']:,}个GPU环境积分步；候选在各试次有效时段内最大摆角差{replay['refinement_max_angle_difference_deg']:.3f}°、最大速度差{replay['refinement_max_speed_difference_rad_s']:.4f}rad/s。步长敏感性也未通过，逐档数值见replay report.json。对已保存轨迹反算运动方程，速度绝对值大于0.5rad/s的各档99%力矩平衡残差均小于0.05N·m，未发现输入或阻力公式写错造成的量级差异；详情见summary.json。这里检查的是降阶模型，不替代统一整机GPU行为验收。</p>
<h2>当前XML不覆盖左腿起点</h2><p>按现有H7符号/零点映射求闭链，左腿起点lf1约{replay['nominal']['left']['initial_passive_joint_ranges'][0]['initial_rad']:.5f}rad，XML上限0.14rad；几何能闭合，但已在限位外。右腿初态在现有限位内。正式参数未改，不能用当前左腿硬限位下的回放直接拟合摩擦。H7映射来源是当前仓库配置；采集包没有固件版本，所以与采集当天一致性仍待确认。</p>
<h2>适用边界与下一步</h2><p>这批数据可约束正向、约340–352mm腿长、侧躺架空时的命令等效摆动负载；没有证明整机接地、负重、短腿、反向运动或气弹簧参数正确。建议优先补同侧同腿长的正反向1/3/5rad/s及重复起转，并记录台架姿态、弹簧安装、固件哈希；同时核对左腿实际开度与模型限位。冻结这次候选后再用新记录验证：各档前后关节和摆速均改善、撤力响应正确、跨步长稳定，才讨论接入完整模型；不能仅以力矩回归残差小决定采用。</p>
<h2>完整曲线</h2><img src="left-all-levels.png" alt="左腿五档"><img src="right-all-levels.png" alt="右腿五档">
<p>方法依据：<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html">SciPy lsq_linear文档</a>（有界最小二乘）；<a href="https://mujoco.readthedocs.io/en/stable/computation/index.html">MuJoCo Computation</a>（动力学与摩擦约束）。项目数据、指标与结论均来自本次本地计算。</p>
<p>输入、参数和源码SHA见identification.json / replay report.json；所有产物在本任务新目录，未启动策略训练或访问硬件。</p></html>'''
    (args.output/'report.html').write_text(content)
    print(json.dumps(findings,indent=2))


if __name__ == '__main__':
    main()
