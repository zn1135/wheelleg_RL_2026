"""Summarize initial/final 96-case measurements without survivor-only averages."""
import argparse
import json
from pathlib import Path
import numpy as np


def measurement(directory):
    s=json.loads((directory/'summary.json').read_text());rows=s['summary']['per_environment']
    assert s['seed']==43 and s['num_envs']==96 and s['seconds']==30
    cases={}
    for case in sorted({r['case'] for r in rows}):
        group=[r for r in rows if r['case']==case];phases={}
        for phase in ('cruise','parking'):
            entries=[r['phases'].get(phase) for r in group]
            if not all(x and x['complete'] for x in entries):
                phases[phase]=None;continue
            phases[phase]=dict(vx_mean=float(np.mean([x['reference_vx_m_s']+x['vx_error_m_s']['mean'] for x in entries])),
                vx_mae=float(np.mean([x['vx_error_m_s']['mae'] for x in entries])),
                horizontal_speed_mean=float(np.mean([x['horizontal_speed_m_s']['mean'] for x in entries])),
                horizontal_speed_peak=max(x['horizontal_speed_m_s']['max_abs'] for x in entries),
                height_mean=s['height_m']+float(np.mean([x['height_error_m']['mean'] for x in entries])))
        cases[case]=dict(request_speed=group[0]['speed_m_s'],num_envs=len(group),
            cruise_defined=any('cruise' in r['phases'] for r in group),**phases)
    with np.load(directory/'trajectory.npz') as z:
        trace=z['values'];v=trace[::5];layout=s['layout'];valid=v[:,:,layout['valid'][0]]>.5
        a=v[:,:,slice(*layout['raw_action'])][valid]
        saturated={name:float((np.abs(a[:,ids])>=3-1e-6).mean()*100) for name,ids in [('legs',[0,1,3,4]),('wheels',[2,5])]}
        max_abs=float(np.abs(a).max());assert max_abs<=3+1e-6
        support=trace[-500:,:,slice(*layout['support'])].sum(-1)/z['weight_N'][None,:]
        support_stats=dict(min=float(support.min()),max=float(support.max()),mean=float(support.mean()),
            below_0p8_percent=float((support<.8).mean()*100),above_1p2_percent=float((support>1.2).mean()*100))
    failures=[r['episode_end_s'] for r in rows if r['episode_end_reason']=='failure']
    return dict(survived=s['summary']['survived'],balanced=s['summary']['final_balanced'],numerical_failures=s['summary']['numerical_failures'],
        failure_range_s=[min(failures),max(failures)] if failures else None,cases=cases,action_at_bound_percent=saturated,
        max_abs_action=max_abs,checkpoint_sha256=s['checkpoint_sha256'],trajectory_sha256=s['trajectory_sha256'],
        final1s_total_support_ratio=support_stats,
        balance_failed_conditions={k:sum(not r['final1s_balance_conditions'][k] for r in rows) for k in rows[0]['final1s_balance_conditions']},
        surviving_ids=[r['env'] for r in rows if r['survived']],balanced_ids=[r['env'] for r in rows if r['final_balanced']])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);args=p.parse_args();d=args.experiment
    initial=measurement(d/'initial-eval');final=measurement(d/'final-eval')
    audit=json.loads((d/'training-audit.json').read_text())
    report=dict(initial=initial,final=final,training=audit,
        lost_survival_ids=sorted(set(initial['surviving_ids'])-set(final['surviving_ids'])),
        lost_balance_ids=sorted(set(initial['balanced_ids'])-set(final['balanced_ids'])),
        scope='Fixed seed43 nominal 96-case 30s evaluation; no multi-seed or hardware validation')
    (d/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# V13 v5：SM615初始化后300次策略更新','',
        f"实际更新：{audit['counts']}；学习耗时{audit['elapsed_learning_s']/60:.2f}分钟；真实转移{audit['transitions']}。世界模型4000次预热另计，actor从0到300。",'',
        '|模型|存活|双轮平衡|轮/腿动作达到±3边界比例|','|---|---:|---:|---|']
    for name,r in [('初始SM615＋限幅',initial),('训练300次',final)]:
        lines.append(f"|{name}|{r['survived']}/96|{r['balanced']}/96|{r['action_at_bound_percent']['wheels']:.3f}% / {r['action_at_bound_percent']['legs']:.3f}%|")
    lines+=['','|模型/条件|请求速度|巡航实际速度|停车均速|停车峰速|停车高度|','|---|---:|---:|---:|---:|---:|']
    for name,r in [('初始',initial),('300次',final)]:
        for case,c in r['cases'].items():
            cruise,park=c['cruise'],c['parking']
            val=lambda obj,key:'N/A' if obj is None else f"{obj[key]:.4f}"
            cruise_text=val(cruise,'vx_mean') if c['cruise_defined'] else '—'
            lines.append(f"|{name}/{case}|{c['request_speed']}|{cruise_text}|{val(park,'horizontal_speed_mean')}|{val(park,'horizontal_speed_peak')}|{val(park,'height_mean')}|")
    lines+=['',f"原初始成功样本丢失：存活{len(report['lost_survival_ids'])}，双轮平衡{len(report['lost_balance_ids'])}。",
        f"最终末1秒平衡条件失败例数：{final['balance_failed_conditions']}。总轮承重/体重范围{final['final1s_total_support_ratio']['min']:.3f}～{final['final1s_total_support_ratio']['max']:.3f}，均值{final['final1s_total_support_ratio']['mean']:.3f}；是否满足连续区间按正式逐例条件判断，不能用平均承重代替。",
        '速度单位m/s，高度m，目标高度0.26m。—表示该条件未设置巡航；N/A表示该条件存在未完成原测量窗口的样本，没有使用幸存者均值。',
        '当前配方与历史V13同时存在世界模型结构、输入坐标、夹紧和初始化差异，不能把效果归因于动作±3单项。在线世界损失随数据和归一化变化，不作为固定轨迹预测进步证据。',
        '仅固定seed43名义工况，未验证多种子或真机；H7未改。全部历史产物保留，未提交或推送。',
        '[训练命令及TensorBoard](RUNNING.md) · [恢复与计数审计](training-audit.json) · [逐例评估](final-eval/summary.json)']
    (d/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
