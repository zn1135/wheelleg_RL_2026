"""Read-only delay/randomization coverage and complete-window prediction audit."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
from ct_mjlab.supermimic.learner import Learner
from ct_mjlab.supermimic.networks import RunningNorm
from scripts.agent.supermimic.diagnose_v13_v5_response import sha


def bounds(x):return [float(x.min()),float(x.max())]


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('delay','training','checkpoint','candidate','output'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    parent=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    candidate=torch.load(args.candidate,map_location='cpu',weights_only=False)
    assert candidate['parent_sha256']==sha(args.checkpoint)
    reports={};scales=[]
    for mode,directory in [('delay_only',args.delay),('training',args.training)]:
        report=json.loads((directory/'report.json').read_text())
        assert report['recording_sha256']==sha(directory/'recording.pt')
        assert report['checkpoint_sha256']==sha(args.checkpoint) and report['candidate_sha256']==sha(args.candidate)
        assert report['optimizer_updates']==report['normalization_updates']==0 and report['actor_frozen'] and report['worlds_frozen']
        data=torch.load(directory/'recording.pt',map_location='cpu',weights_only=False);params=data['initial']['parameters']
        assert torch.equal(params[:,24],params[:,25])
        unique,counts=params[:,25].unique(return_counts=True)
        coverage=dict(num_envs=len(params),action_delay_ms_counts={str(int(x)*2):int(n) for x,n in zip(unique,counts)},
            sensor_delay_ms=bounds(params[:,26:30]*2),motor_strength=bounds(params[:,18:24]),
            nominal_pose_spread_per_action=(params[:,:6].max(0).values-params[:,:6].min(0).values).tolist(),
            total_mass_kg=bounds(params[:,30:50].sum(-1)),wheel_friction=bounds(params[:,[53,56]]),
            solref_first_component_counts={str(float(v)):int(n) for v,n in zip(*params[:,67].unique(return_counts=True))},
            motor_randomization=data['initial']['action_randomization'],observation_settings=data['initial']['observations'])
        strict=[];push_positions=0;eligible_positions=0
        for s in data['samples']:
            if s['anchor_windows'] is None:continue
            w=s['anchor_windows'];mask=w['mask'].bool()
            push_positions+=int(((w['parameters'][...,69:75].abs().amax(-1)>0)&mask).sum());eligible_positions+=int(mask.sum())
        coverage.update(stored_valid_positions_with_active_wrench=push_positions,stored_valid_positions=eligible_positions)
        for label,weights in [('parent',parent['world']),('candidate',candidate['world'])]:
            learner=Learner('cpu');learner.world.load_state_dict(weights);learner.world.requires_grad_(False)
            scales.append(learner.world.delta_norm.std().clone())
            for length in (1,8,24):
                total=0.;positions=windows=0
                for s in data['samples']:
                    if s['anchor_windows'] is None:continue
                    original=s['anchor_windows'];complete=original['mask'][:,:length].bool().all(1)
                    if not complete.any():continue
                    w={k:v[complete,:length] for k,v in original.items()}
                    loss,m=learner.world_loss(w);n=m['valid_world_positions']
                    assert n==len(w['state'])*length
                    total+=float(loss)*n;positions+=n;windows+=len(w['state'])
                strict.append(dict(model=label,horizon_ms=length*10,complete_windows=windows,valid_positions=positions,
                    normalized_mse=total/positions if positions else None))
        assert all(torch.equal(scales[0],x) for x in scales)
        # The two models must be compared on identical eligible cases and complete windows.
        for ms in (10,80,240):
            x,y=[next(r for r in strict if r['model']==label and r['horizon_ms']==ms) for label in ('parent','candidate')]
            assert x['complete_windows']==y['complete_windows'] and x['valid_positions']==y['valid_positions']
        for delay in range(0,11,2):
            x,y=[next(r for r in report['rows'] if r['model']==label and r['delay_ms']==delay) for label in ('parent','candidate')]
            assert all(x[k]==y[k] for k in ('possible_cases','valid_cases','eligible','true_near_zero'))
        reports[mode]=dict(source=report,coverage=coverage,complete_window_horizons=strict)
    result=dict(reports=reports,optimizer_updates=0,normalization_updates=0,parent_sha256=sha(args.checkpoint),candidate_sha256=sha(args.candidate),
        nominal_acceptance_does_not_imply_training_acceptance=True,script_sha256=sha(Path(__file__)))
    (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# V13 v5：非零延迟与训练随机化检查','',
        '**不能直接按原训练配方续训。** 候选保留名义工况通过记录，但此次训练工况未通过。全部网络、优化器和归一化冻结；没有训练。','',
        '|工况|原模型240ms误差|候选240ms误差|共同完整窗口数|','|---|---:|---:|---:|']
    for name,r in reports.items():
        v=[next(x for x in r['complete_window_horizons'] if x['model']==m and x['horizon_ms']==240) for m in ('parent','candidate')]
        lines.append(f"|{name}|{v[0]['normalized_mse']:.6f}|{v[1]['normalized_mse']:.6f}|{v[0]['complete_windows']}|")
    lines+=['','每行使用同一轨迹、同一有效mask、同一父模型46维归一化尺度；上表只用完整窗口，失败后复位数据排除。',
        '', '|仅增加延迟|方向正确/可判定|反向|预测过弱|真实近零但候选预测非零|','|---|---:|---:|---:|---:|']
    for r in reports['delay_only']['source']['rows']:
        if r['model']=='candidate':lines.append(f"|{r['delay_ms']}ms|{r['correct']}/{r['eligible']}|{r['opposite']}|{r['weak']}|{r['spurious_nonzero']}/{r['true_near_zero']}|")
    for name,r in reports.items():
        s=r['source'];c=r['coverage']
        lines+=['',f"## {name}覆盖",'',f"seed{s['seed']}，初始{s['initial_envs']}例，20.01秒内原回合失败{s['failed_first_episodes']}例。各采样时刻有效例数："+str([(x['time_s'],x['live']) for x in s['samples']]),
            f"动作延迟分布(ms:环境数)：{c['action_delay_ms_counts']}；腿轮同延迟。传感延迟范围{c['sensor_delay_ms']}ms，强度倍率{c['motor_strength']}，总质量{c['total_mass_kg']}kg，轮摩擦{c['wheel_friction']}。",
            f"封存有效回放位置{c['stored_valid_positions']}个，其中外力非零{c['stored_valid_positions_with_active_wrench']}个（回放窗可重叠，不代表独立推力次数）。"]
    lines+=['','随机化使用真实make_env(play=False)配方，包含原质量/COM/惯量/摩擦/材料、PD/标称偏置、观测噪声/延迟及外力调度，接触材料保持训练中每4环境1个名义值的混合。仅延迟工况使用名义物理，0/2/4/6/8/10ms每组24例。',
        '0～4ms响应方向改善；6/8ms仍有反向；10ms当前命令通常尚未进入这10ms真实执行窗口，但候选仍常预测有影响。真实近零响应单列，不从诊断中隐藏。绝对噪声门槛1e-5m/s，并要求真实效应大于5倍恢复/重跑误差；少量不可判定样本和±3边界样本排除但计数保留。',
        '本次真实控制始终为相同原A300策略，模型候选不参与实际动作计算；随机工况的失败不能归因于候选改变了控制。预测比较是在相同原始失败前片段上完成，不是用幸存者填补完整行为指标。',
        '下一步优先在真实延迟/随机化分布校准预测响应，冻结策略并同时保住名义、延迟和随机工况误差后，再考虑控制更新。不能仅因本次相关性就确定唯一根因，也不自动恢复用户已删除的历史输入。',
        '首轮延迟检查r1因包装器返回整型done而中止，未产生诊断数据；修正脚本中的布尔处理后r2完成，旧目录/日志保留。生产算法、物理、XML/H7未修改，未提交推送。']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete_window_horizons={k:v['complete_window_horizons'] for k,v in reports.items()},coverage={k:v['coverage'] for k,v in reports.items()}),ensure_ascii=False))


if __name__=='__main__':
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No fitting')),patch.object(RunningNorm,'update',side_effect=AssertionError('Frozen normalization')):
        main()
