"""Frozen step-training review: log evidence and the existing 47-case walking regression."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def source_checks(run):
    manifest=json.loads((run/'source_sha256.json').read_text())
    checks={name:Path(name).exists() and sha(Path(name))==digest for name,digest in manifest.items()}
    assert all(checks.values()),[name for name,ok in checks.items() if not ok]
    return checks


def training(args):
    import numpy as np
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from ct_mjlab.envs.chuanliantui.fudan_ppo_step.config import FudanStepCfg, recipe_record
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True,exist_ok=False)
    checks=source_checks(args.run)
    model=torch.load(args.run/'model_final.pt',map_location='cpu',weights_only=False)
    assert model['training_recipe']==recipe_record(FudanStepCfg())
    rows=[]
    for run in args.episode_runs or [args.run]:
        path=run/'episodes.jsonl'
        if path.exists():
            rows.extend(dict(run=str(run),**json.loads(line)) for line in path.open())
    groups=[]
    for window,minimum in [('all',0),('last50',max(1,model['iterations']-49))]:
        selected=[row for row in rows if minimum<=row['iteration']<=model['iterations']]
        for mode in (0,1,2):
            for level in sorted({row['level'] for row in selected if row['mode']==mode}):
                batch=[r for r in selected if r['mode']==mode and r['level']==level]
                groups.append(dict(window=window,mode=('flat','up','down')[mode],level=level,
                    height_m=0. if mode==0 else model['training_recipe']['step']['heights'][level],episodes=len(batch),
                    success=sum(r['success'] for r in batch),crossed=sum(r['crossed'] for r in batch),
                    failed=sum(r['failure'] for r in batch),ray_missed=sum(r.get('ray_missed',False) for r in batch),
                    body_contact_rate_mean=float(np.mean([r['contact_rate'][0] for r in batch])),
                    leg_contact_rate_mean=float(np.mean([r['contact_rate'][1] for r in batch]))))
    event=EventAccumulator(str(args.run),size_guidance={'scalars':0}).Reload()
    tags=['Train/reward','Step/terrain_level','Step/crossed_rate','Step/success_rate',
          'Contact/body/contact_rate','Contact/leg/contact_rate']
    scalars={tag:[dict(step=r.step,value=r.value) for r in event.Scalars(tag)] for tag in tags}
    fig,axes=plt.subplots(3,2,figsize=(11,8))
    for ax,tag in zip(axes.flat,tags):
        data=scalars[tag];ax.plot([x['step'] for x in data],[x['value'] for x in data],lw=.8)
        ax.set(title=tag,xlabel='Iteration');ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(args.output/'training-curves.png',dpi=130);plt.close(fig)
    result=dict(checkpoint=str((args.run/'model_final.pt').resolve()),checkpoint_sha256=sha(args.run/'model_final.pt'),
        iterations=model['iterations'],transitions=model['transitions'],optimizer_updates_during_review=0,
        source_checks=checks,recipe_matches=True,std=model['model_state_dict']['std'].tolist(),
        final_terrain_level_counts=dict(Counter(model['terrain_levels'].tolist())),
        episodes=len(rows),groups=groups,scalars=scalars,
        limitation='Training episodes mix stochastic actions, up/down, heights and randomization; not acceptance rates.')
    write(args.output/'training-evidence.json',result)
    print(json.dumps(dict(iterations=model['iterations'],episodes=len(rows),groups=groups),ensure_ascii=False))


def flat(args):
    from scripts.agent.fudan_ppo_walk import inspect_pd200_training as inspect
    checks=source_checks(args.run)
    # 直接复用原47例及其初态、延迟、控制审计；回放期间禁止优化器更新。
    inspect.MODEL=args.checkpoint.resolve()
    inspect.RUN=args.run.resolve()
    with patch('torch.optim.Adam.step',side_effect=AssertionError('Review cannot train')):
        inspect.run(args.output,checkpoint_audit=lambda _:checks)


def report(args):
    import numpy as np
    root=args.output
    evidence=json.loads((root/'training/training-evidence.json').read_text())
    pairs=[]
    for baseline,candidate,height in [('baseline-h30','candidate-h30',.30),('baseline','candidate',.35)]:
        protocols=[json.loads((root/name/'protocol.json').read_text()) for name in (baseline,candidate)]
        assert protocols[0]['cases']==protocols[1]['cases']
        for protocol in protocols:
            assert protocol['all_networks_unchanged'] and protocol['all_finite']
            assert protocol['all_substep_controls_within_caps'] and protocol['optimizer_updates']==0
            assert sha(Path(protocol['source_checkpoint']))==protocol['source_sha256']
            assert protocol['seconds']==20. and protocol['friction']==.8
            assert set(protocol['action_delay_steps'])==set(protocol['sensor_delay_steps'])=={15}
        initial_difference=0.
        for case in protocols[0]['cases']:
            traces=[]
            for name,protocol in zip((baseline,candidate),protocols):
                path=root/name/'traces'/(case['id']+'.npz')
                assert sha(path)==protocol['trace_sha256'][case['id']]
                with np.load(path) as data:traces.append({k:data[k] for k in data.files})
            difference=float(abs(traces[0]['initial']-traces[1]['initial']).max())
            initial_difference=max(initial_difference,difference)
            assert difference<2e-6
            assert all(t['terrain_valid'][t['valid']].all() for t in traces)
        metrics=[json.loads((root/name/'metrics.json').read_text()) for name in (baseline,candidate)]
        groups=[]
        for step_height in sorted({r['height'] for r in metrics[0] if r['height']>0}):
            selected=[[r for r in rows if r['height']==step_height] for rows in metrics]
            groups.append(dict(height_m=step_height,cases=len(selected[0]),
                landed_before=sum(r['landed'] for r in selected[0]),landed_after=sum(r['landed'] for r in selected[1]),
                task_passed_before=sum(r['task_score']['passed'] for r in selected[0]),
                task_passed_after=sum(r['task_score']['passed'] for r in selected[1])))
        pairs.append(dict(command_height_m=height,initial_max_abs_difference=initial_difference,
            initial_comparison='Same reset protocol; GPU reset float rounding within2e-6, not bitwise equality',groups=groups))
    flat=json.loads((root/'flat47/acceptance/acceptance.json').read_text())
    assert flat['comparison']['regression_rejected']
    result=dict(training_iterations=evidence['iterations'],pairs=pairs,
        flat=dict(before_passed=47,after_passed=flat['technical_passed'],total=flat['total'],comparison=flat['comparison']),
        optimizer_updates=0,behavior_accepted=False,full_step_matrix_run=False,
        limitation='Nominal diagnostic: heights20/50/80/120/160/200mm, speeds.3/.5, angles0/±10, both directions. '
                   'Not the full540-case robustness matrix or native RMUC field. Legacy stop_reason fall_or_boundary also includes timeouts; '
                   '20s records must not be described as falls based on that string.')
    write(root/'comparison.json',result)
    text='''# 台阶500轮效果：未学会上台阶，原平地能力退化

500轮正常完成，预算/训练源码审计通过；最终PT SHA256为`%s`。本次只回放和评分，优化器更新0次，没有追加训练。

| 同条件GPU检查 | 训练前3000平地模型 | 台阶500轮模型 |
|---|---:|---:|
| 20mm双轮登台，指令高度.30m | 12/12 | 0/12 |
| 20mm双轮登台，指令高度.35m | 7/12 | 0/12 |
| 200mm双轮登台，两种高度合计 | 0/24 | 0/24 |
| 原平地47例统一运动标准 | 47/47 | %d/47 |

“双轮登台”要求台面内两轮持续承重至少.1秒，不等于登台后停车成功；500轮模型全部144个台阶诊断工况严格任务评分均失败。训练日志的完整登台/稳定停车2秒成功也一直为0。最高课程到210mm含下台和跨回合课程状态，不代表能登上210mm。

观察到的主要问题：低速跟踪严重变慢，没有形成主动抬轮。正对200mm倒车、指令−.30m/s、高度.30m时，训练前实际接近.299m/s，500轮约.077m/s；后者双轮最高中心约.0612m，基本停留在地面轮高，机身碰沿峰约87.7N、持续.201s，仍失败。高度.35m的同方向.50m/s例则腿杆碰沿峰118.7N/连续.208s。两种姿态的失败分别保留，不能把接触减少写成登台改善，也没有证据支持加大电机限矩。

训练前50轮/末50轮的平均奖励约.02557/.02492；末50轮平均课程等级1.79。末50轮上台20mm有51/489次过沿、50mm为0/445，所有上台最终停车成功0；这属于带探索噪声的训练回合统计，不能当确定策略验收。机身接触率下降伴随腿杆接触率增加，也受课程和128→4096变化影响，不能据此推断因果。

## 检查范围与可复现性

每个模型152个工况：两组高度指令.30/.35，各72台阶＋4平地对照，每例最长20秒；.3/.5m/s、0/±10°、正向/倒车、摩擦.8、固定动作/传感延迟15ms、名义质量/增益，无外载、无侧障碍。两模型同复位协议，GPU复位浮点差小于2e−6，未声称逐位相同；原始初态、逐1ms力矩/接触、逐10ms状态/终止掩码和77射线有效性有留档。全部记录有限、实际控制不超40/3.9N·m、无有效回合射线缺测、网络逐值未变。当前源码与训练快照逐文件一致。

原平地47例另运行30秒最长窗口，1ms积分/200Hz PD/100Hz策略、源模型10～20ms双延迟，无外载；实际PD公式误差0、外力0、全部47例完整记录。统一评分通过初态cohort门禁，丢失46个原成功样本，唯一保留是低高度.25m/s前进。含高速/大偏航的旧样本超出台阶低速训练分布，但低速、原地和停车同样退化，仍按约定判退化。未跑完整540项台阶矩阵、987平地随机总体、原版场地或实机，不作这些通过声明。

旧诊断脚本把超时也写入`fall_or_boundary`；20秒到期例不能仅按此字符串称为摔倒，本报告依据有效轨迹和独立评分。

## 视频与证据

- [200mm倒车碰沿失败，1倍速](video-200mm-reverse/replay-1x.mp4)（高度指令.35m，速度指令−.5m/s）。
- [20mm：训练前3000](video-20mm-walk3000/replay-1x.mp4)、[台阶500轮](video-20mm-step500/replay-1x.mp4)（相同−.3m/s、.30m高度；保留各自终止时间）。
- [训练曲线](training/training-curves.png)、[课程分高度统计](training/training-evidence.json)、[同条件比较](comparison.json)。
- [原平地统一评分](flat47/acceptance/acceptance.json)、[实际PD/时序检查](flat47/runtime-audit.json)。
- `candidate[-h30]/`、`baseline[-h30]/`保留协议、模型、原始轨迹、全部指标；`audit-*`复算哈希及指标。视频目录的media.json保存来源/视频SHA与帧数。

## 建议下一步

保留3000模型为基线，暂不续训此500轮模型。优先查清低速跟踪为何下降，再从3000做单一20～50mm主动抬轮课程对照，调整过沿引导及平地保持；先要求20mm恢复通过、原47例不丢失，再升50mm。奖励/课程失配是待验证解释，本次没有实施新配方或训练。

脚本：`scripts/agent/fudan_ppo_step/inspect_training.py`（训练证据/47例复用/比较）、`diagnose_reverse.py`（新增显式checkpoint及高度指令、记录并独立评分，原默认源模型保持）。所有命令使用本仓mjlab虚拟环境，现场训练源码、原模型和历史产物保持。
'''%(evidence['checkpoint_sha256'],flat['technical_passed'])
    (root/'RESULTS.md').write_text(text)
    print(json.dumps(dict(flat_passed=flat['technical_passed'],pairs=pairs),ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=('training','flat','report'))
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--episode-runs',type=Path,nargs='+',help='Explicit additional run segments; default only the selected run')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.mode=='flat' and args.checkpoint is None:p.error('flat requires --checkpoint')
    {'training':training,'flat':flat,'report':report}[args.mode](args)


if __name__=='__main__':
    main()
