"""Read-only parking diagnostic report and plots; no simulator or policy imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.experiment;out=args.output;out.mkdir(parents=True,exist_ok=False)
    data=json.loads((root/'analysis.json').read_text());budget=json.loads((root/'budget.json').read_text())
    names={'diagnostic_0':'原地保持','diagnostic_13':'后退后停车','diagnostic_14':'前进后停车'}
    branches=['baseline','true_vx_at_parking','true_xyz_at_parking']
    labels=['Original policy','True forward velocity','True 3-axis velocity'];colors=['#bb4430','#2675b5','#16835d']
    rows=[]
    for branch in branches:
        for name,values in data['results'][branch].items():
            rows.append(dict(branch=branch,case=name,**{k:v for k,v in values.items() if not isinstance(v,list)},
                left_wheel_target_rad_s=values['wheel_target_rad_s'][0],right_wheel_target_rad_s=values['wheel_target_rad_s'][1]))
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(11,9),constrained_layout=True)
    inputs={}
    for branch,label,color in zip(branches,labels,colors):
        path=root/branch/'policy.npz';inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as d:
            for row,(name,(lo,hi)) in enumerate(data['windows'].items()):
                i=int(name.split('_')[-1]);mask=(d['time']>=lo)&(d['time']<hi);t=d['time'][mask]-lo
                axes[row,0].plot(t,d['label_m_s'][mask,i,0],color=color,label=label)
                if branch=='baseline':axes[row,0].plot(t,d['estimate_m_s'][mask,i,0],color=color,ls='--',label='Original estimated vx')
                position=d['position'][mask,i,0];axes[row,1].plot(t,(position-position[0])*100,color=color,label=label)
    titles=['Stand still','Stop after backward','Stop after forward']
    for i,title in enumerate(titles):
        axes[i,0].set(title=title+' - velocity',ylabel='Body vx (m/s)',xlabel='Seconds since intervention/window start')
        axes[i,1].set(title=title+' - displacement',ylabel='World x displacement (cm)',xlabel='Seconds since intervention/window start')
        for ax in axes[i]:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'parking.png',dpi=150);plt.close(fig)
    run=data['execution'];zone=ZoneInfo('Asia/Shanghai')
    dates=[datetime.fromtimestamp(run[k],zone).isoformat() for k in ('started_unix','finished_unix')]
    lines=['# 7000轮停车慢退：GPU估速替换诊断','',
        '结论：原模型低估后退速度；输入真实速度能显著减轻漂移，但不能停车。仅在这组三个名义0.30m场景成立，不把改善比例解释为根因贡献百分比。','',
        '| 稳定段5秒 | 原模型后退cm | 仅前向真值cm | 三轴真值cm | 原估速vx m/s | 原真实vx m/s |',
        '|---|---:|---:|---:|---:|---:|']
    for name,label in names.items():
        original=data['results']['baseline'][name]
        distances=[abs(data['results'][branch][name]['net_dx_m'])*100 for branch in branches]
        lines.append(f"| {label} | {distances[0]:.2f} | {distances[1]:.2f} | {distances[2]:.2f} | {original['estimate_vx_m_s']:.5f} | {original['label_vx_m_s']:.5f} |")
    maximum=max(v['pre_intervention_max_position_difference_m'] for row in data['comparisons'].values() for v in row.values())
    lines.extend(['','## 比较条件','',
        '原地窗口2～7秒；前后停车在7秒给零指令，窗口固定10～15秒。替换仅从对应窗口开始，改变传给Actor的估速latent，不改观测、模型权重、PD或物理。真实值按原训练10ms根位置差分转机体系，再乘2；输出估速报告除2还原m/s。原7000-v2历史保持，额外动作/传感延迟与推扰全关。','',
        '原15例完整跑三组，只有0/13/14在稳定停车段介入；各组初始qpos/qvel、PD随机参数、质量/摩擦等与源回放逐项相同。各组均1500个策略批步，15秒。',
        f'GPU重复并非逐位相同：介入前位置最大差{maximum*1000:.3f}mm；本次漂移减少约12cm。保留这一误差，不声称完全相同介入状态或统计置信区间。',
        '上一停车报告的两例测量起点是8.358/8.684秒，本次统一10秒，不能直接把小量差异解释成改进；基线仍复现约0.051m/s持续后退。','',
        '## 解释与限制','',
        '- 原模型将约−0.051m/s估成−0.023m/s，前向估速偏差约+0.0277m/s。不是完全把运动估成零。',
        '- 仅替换前向速度后仍约−0.026m/s；三轴真值后仍约−0.0255m/s。因此只修估速并不足以消除零速残差。',
        '- 三组测量窗口机身接触峰值均0，俯仰峰值约0.22°；不是靠机身撑地换取低速。替换瞬间速度峰值仍约0.052m/s，且仍漂移约13cm，不能宣称停车验收通过。',
        '- 原左右实体轮目标均值约−2.74/+2.10rad/s，实际轮速约−0.87/+0.84rad/s；符号属于各关节物理轴，不能直接解释为机器人左右轮朝相反方向。目标含维持平衡与克服阻力的需要，不建议直接清零轮动作。',
        '- 真值替换属于仿真特权输入，不可当作部署修复；没有模型更新、没有最终用户视频验收。','',
        '## 执行与复现','',
        f"实际动态窗口：{dates[0]} ～ {dates[1]}，{run['wall_s']:.3f}秒，低于300秒上限。Encoder批调用{run['encoder_batch_calls']}、Actor批调用{run['actor_batch_calls']}；显式环境积分子步{run['explicit_environment_physics_substeps']}，后端准备/图捕获计入墙钟但未计入该显式子步计数；优化器更新0。",
        f"模型SHA：`{budget['checkpoint_sha256']}`。动态结束时权重张量及受保护文件哈希未变。实际诊断源码已保存到`../source/`；后续新增停车采样代码不属于本次运行源码。",'',
        '只读复算命令（输出换新目录，不重新仿真）：','```bash',
        f'PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.report_parking --experiment {root} --output <新报告目录>',
        '```','',
        '新增动态复现需另获预算，不能复用本次已结束窗口。完整数据见各分组`policy.npz`、`traces/`、`fixtures/`；`policy.npz`保存实际Actor输入/历史、原估速与使用值、同口径标签、瞬时速度、命令、原/实际动作、轮目标/轮速及步后最后一拍PD力矩。','',
        '下一步：先补停车命令样本，保持16奖励/网络/物理。若从7000-v2做受控续训，需要保持v2历史或明确把历史迁移作为独立实验；新训练须另定预算，当前仅代码准备。','',
        '[全部数值](../analysis.json)、[逐例CSV](summary.csv)、[曲线](parking.png)、[动态执行](../execution.json)、[预算及来源](../budget.json)。'])
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    (out/'inputs_sha256.json').write_text(json.dumps(inputs,indent=2)+'\n')
    print(json.dumps(dict(report=str(out/'README.md'),physics_steps=0,model_forwards=0,optimizer_updates=0)))


if __name__=='__main__':main()
