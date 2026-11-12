"""Read paired torque experiment results; optionally wait for its existing runner."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--wait',action='store_true');args=parser.parse_args();root=args.experiment.resolve()
    while True:
        status=json.loads((root/'status.json').read_text())
        if status['status'] in ('complete','failed'):break
        if not args.wait:raise RuntimeError('Experiment is still running')
        time.sleep(15)
    if status['status']=='failed':
        (root/'RESULTS.md').write_text('# 试验未完成\n\n阶段：'+status['stage']+'\n\n错误：'+status['error']+'\n\n已保留现场，不自动重试或追加训练；未完成配对，不能判断外矩影响。\n')
        return
    plan=json.loads((root/'plan.json').read_text())
    folders={'baseline10000':Path(plan['baseline_recording']),
        'no_torque10500':root/'no_torque-evaluation','torque10_10500':root/'torque10-evaluation'}
    leg=json.loads((root/'leg-motion/leg-motion.json').read_text())['rows']
    result=dict(scope='One-seed500-iteration-per-arm exploration; no real-robot verification',groups={},pairing=[])
    for arm in ('no_torque','torque10'):
        audit=json.loads((root/arm/'resume-audit.json').read_text())
        assert audit['passed'] and audit['additional_iterations']==500 and audit['final_iteration']==10500
    for group in ('no_external','configured_external'):
        table={}
        baseline=json.loads((folders['baseline10000']/(group+'-acceptance/acceptance.json')).read_text())
        original={s['sample_id']:s for s in baseline['samples']}
        for name,folder in folders.items():
            run=json.loads((folder/'run.json').read_text())
            model=Path(run['checkpoint']).stem+f' P{run["leg_p"]:g}/D{run["leg_d"]:g}'
            values=[r for r in leg if r['model']==model and r['group']==group and r['case'].startswith('stop_')]
            assert len(values)==6
            score=json.loads((folder/(group+'-acceptance/acceptance.json')).read_text())
            rows=score['samples'];stops=[r for r in rows if r['case']['category'].startswith('stop_')]
            table[name]=dict(median_last_second_joint_pp_deg=float(np.median([r['last_second_max_peak_to_peak_deg'] for r in values])),
                maximum_last_second_joint_pp_deg=max(r['last_second_max_peak_to_peak_deg'] for r in values),
                median_last_five_seconds_joint_pp_deg=float(np.median([r['max_peak_to_peak_deg'] for r in values])),
                stop_passed=sum(r['passed'] for r in stops),stop_total=6,total_passed=score['technical_passed'],total=9,
                lost_baseline_samples=[r['sample_id'] for r in rows if original[r['sample_id']]['passed'] and not r['passed']],
                samples=values)
            for sample in rows:
                filename=sample['sample_id']+'_initial.npz'
                with np.load(folder/group/'fixtures'/filename) as a,np.load(folders['baseline10000']/group/'fixtures'/filename) as b:
                    result['pairing'].append(dict(model=name,group=group,sample_id=sample['sample_id'],
                        equal=all(np.array_equal(a[k],b[k]) for k in a.files)))
        result['groups'][group]=table
    assert all(v['equal'] for v in result['pairing'])
    a=result['groups']['no_external']['no_torque10500'];b=result['groups']['no_external']['torque10_10500']
    result['clean_comparison']=dict(no_torque_median_lower=a['median_last_second_joint_pp_deg']<b['median_last_second_joint_pp_deg'],
        no_torque_pass_count_not_lower=a['stop_passed']>=b['stop_passed'],
        no_torque_retains_original_success=not a['lost_baseline_samples'])
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 持续外力矩配对试验结果','',
        '两组均从10000/P60D2新增500完整轮到10500，恢复与日志审计通过。原16项奖励和原随机化相同，仅训练持续外矩0或±10Nm不同；持续外力/推扰均关闭。',
        '', '## 无外力停车：腿摆与停稳','',
        '摆幅为六个行驶后停车样本末1秒、四个实际腿关节最大峰峰值的中位数；不是机身倾角。正式停车评分仍按原窗口/速度≤0.05m/s等条件。','',
        '| 模型 | 摆幅中位数 | 最差摆幅 | 行驶后停车通过 | 丢失10000原通过样本 |',
        '|---|---:|---:|---:|---|']
    for name,row in result['groups']['no_external'].items():
        lines.append(f'| {name} | {row["median_last_second_joint_pp_deg"]:.3f}° | {row["maximum_last_second_joint_pp_deg"]:.3f}° | {row["stop_passed"]}/6 | {", ".join(row["lost_baseline_samples"]) or "无"} |')
    lines+=['','## 解读','']
    if a['median_last_second_joint_pp_deg']<b['median_last_second_joint_pp_deg']:
        lines.append('无外矩组的摆幅中位数较低，支持外矩可能加重本次训练结果的摆动；是否值得采用仍须逐例核对停稳和原成功保持。')
    else:
        lines.append('无外矩组的摆幅中位数没有更低，本次对照不支持“去掉持续外矩就能减轻摆腿”；不能把假设写成已证原因。')
    if a['lost_baseline_samples']:
        lines.append('无外矩候选丢失原通过样本，不因摆幅变化直接采用。')
    lines+=['','只有一个种子、各500轮；训练奖励/动作曲线不能替代真机结果。未做完整987或新真机验证，无自动追加训练。',
        '', '[全部对照数据](comparison.json) · [腿部曲线](leg-motion/leg-oscillation.png)',
        '', '[无外矩组正常速度停车视频](no_torque-evaluation/video/no_external_parking_5_side.mp4)',
        '', '[外矩±10组正常速度停车视频](torque10-evaluation/video/no_external_parking_5_side.mp4)',
        '', '两视频均完整15秒，第7秒零指令，跟随相机；看速度/世界位置读数判断漂移。正式验收仍待用户观看。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
