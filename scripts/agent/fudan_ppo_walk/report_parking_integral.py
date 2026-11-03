"""Original-command unified scoring and fixed-window parking acceptance; no physics/inference."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.agent.motion.acceptance import canonical,compare_baseline
from ct_mjlab.envs.fudan_ppo_walk.parking_integral import REASONS
BRANCHES=('baseline','truth_integral','encoder_integral')
HEIGHTS=('h022','h026','h030')


def parking_metrics(trace,lo,hi):
    i,j=round(lo/.002),round(hi/.002)
    if len(trace['time'])<=j:return dict(complete=False,passed=False)
    pos=trace['position'][i:j+1,:2];speed=np.linalg.norm(trace['linear_world'][i:j,:2],axis=-1)
    delta=pos[-1]-pos[0];maximum=float(np.linalg.norm(pos-pos[0],axis=-1).max())
    mean=float(speed.mean());peak=float(speed.max())
    return dict(complete=True,window=[lo,hi],net_x_m=float(delta[0]),net_y_m=float(delta[1]),net_xy_m=float(np.linalg.norm(delta)),
        maximum_horizontal_offset_m=maximum,mean_horizontal_speed_m_s=mean,horizontal_peak_m_s=peak,
        passed=maximum<=.05 and mean<=.01 and peak<=.05)



def missing_report(experiment,recording):
    """Bookkeeping only when initialization stopped before any replay trajectory."""
    def record(name,default):
        path=experiment/name;return json.loads(path.read_text()) if path.exists() else default
    pipeline=record('pipeline-result.json',{})
    initial=record('initial-consistency.json',[]);initialization=record('initialization-result.json',{})
    budget=record('budget.json',{});checks=record('checks.json',{})
    error=initialization.get('error')
    if error is None:
        log=experiment/'replay.log'
        error=log.read_text().strip().splitlines()[-1] if log.exists() and log.read_text().strip() else 'No replay trajectories; inspect replay/pipeline logs'
    passed=sum(bool(r.get('all_arrays_identical')) for r in initial);failed=len(initial)-passed
    evidence=[str(p.relative_to(experiment)) for p in experiment.glob('h*/*/initial-differences/*') if p.is_file()]
    inventory=[];rows=[]
    for h in HEIGHTS:
        items=json.loads((recording/h/'gpu15/protocol.json').read_text())['samples']
        for b in BRANCHES:
            for item in items:
                inventory.append(dict(height=h,branch=b,id=item['id'],status='not_measured',reason='No replay trajectory; inspect initialization-result.json and replay.log',trajectory=None))
                if item['case']['category'] in ('zero','stop_backward','stop_forward'):
                    rows.append(dict(height=h,branch=b,id=item['id'],category=item['case']['category'],complete=False,passed=False,
                        metrics=None,window=[2,7] if item['case']['category']=='zero' else [10,15]))
    result=dict(conclusion='未产生回放轨迹，不能判断积分改善或估速限制',baseline_reproduced=None,
        baseline_reproduction_status='not_evaluated_no_trajectories',initial_checks_passed=passed,initial_checks_failed=failed,
        initial_checks_unexamined=135-len(initial),initial_check_failure=error,initial_difference_files=evidence,
        parking_passed={b:None for b in BRANCHES},unified={b:dict(passed=None,total=45,measured=0,regressed=None) for b in BRANCHES},
        nonregression={b:None for b in BRANCHES},rows=rows,inventory=inventory,expected_trajectories=135,complete_trajectories=0,
        integral_saturation=None,exit_residual=None,exit_conditions_dynamically_verified=False,final_acceptance='not_ready_no_dynamic_results',
        exceptions=[error],
        model_sha256='39fa104d176f82a33db8317a75b4db28749362daf6bc4d2e06cb83d2c5ae32c9',
        original_model_source_input_unchanged=all(pipeline[k] for k in ('checkpoint_unchanged','source_unchanged','input_unchanged')) if pipeline else None,
        actual_pipeline_wall_s=pipeline.get('wall_s'),optimizer_updates=0,no_dynamic_retry=True,
        exact_failed_values_available=bool(evidence),diagnosis_limit='Only saved differences can support a magnitude/cause diagnosis; missing state checks are unverified')
    (experiment/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (experiment/'video-index.json').write_text(json.dumps([dict(height=h,path=None,complete=False,reason='No replay trajectories; no replacement video generated') for h in HEIGHTS],indent=2)+'\n')
    for b in BRANCHES:
        folder=experiment/'acceptance'/b;folder.mkdir(parents=True,exist_ok=True)
        data=dict(task='fudan_ppo_walk',status='not_measured',total=45,measured=0,technical_passed=None,regression=None,
            samples=[r for r in inventory if r['branch']==b],final_acceptance='not_ready_no_dynamic_results')
        (folder/'acceptance.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    lines=['# 8000轮停车积分试验：无回放结果','',result['conclusion'],'',
        f"无物理检查条目：{checks.get('count','未知')}；原预算起止：{budget.get('start_local','未知')}～{budget.get('deadline_local','未知')}。实际执行与保护哈希见pipeline记录，不追加回放。",
        '', '失败记录：'+error,
        '', f'初态已核对{len(initial)}/135，一致{passed}，不一致{failed}，未核对{135-len(initial)}。差值文件：{evidence}。没有保存的差值不能推断幅度或原因。',
        '', '策略回放完整轨迹0/135。GPU初始化是否完成及内部准备步数以原日志为准；数值和视频均未验收。',
        '', '| 高度/工况 | 原模型 | 真值积分 | Encoder积分 |','|---|---|---|---|']
    for h in HEIGHTS:
        for name in ('原地保持2～7s','后退后停车10～15s','前进后停车10～15s'):
            lines.append('| '+h+'/'+name+' | 未测 | 未测 | 未测 |')
    lines+=['', '三组各45例技术评分、是否退化、停车偏移/均速/峰速、饱和比例、退出残留与动态退出覆盖均未验证，不能按0通过或不退化解释。三份视频均缺失，不使用旧视频冒充本轮结果。',
        '', '积分默认关闭，保护范围及模型/源码/输入哈希以config、source-sha256及pipeline记录为准。',
        '', '没有轨迹时仅整理缺失项，不进行行为评分或生成替代视频；本报告不启动模型、仿真或重试。',
        '', '[比较及135项缺失清单](comparison.json) · [无物理检查](checks.json) · [初态核对](initial-consistency.json) · [实际执行结果](pipeline-result.json) · [回放异常](replay.log) · [视频缺失](video-index.json) · [实际执行源码哈希](source-sha256.json)',
        '', '推荐下一步：根据initialization-result、差值数组和原日志定位失败；修正和无物理检查通过后，另获预算再验证。不得放宽初态检查掩盖差异。']
    (experiment/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(conclusion=result['conclusion'],complete_trajectories=0,no_dynamic_retry=True),ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--recording',type=Path,required=True);a=p.parse_args()
    if not list(a.experiment.glob('h*/*/traces/*.npz')):
        missing_report(a.experiment,a.recording);return
    rows=[];unified={b:dict(passed=0,total=45,regressed=[]) for b in BRANCHES};all_initial=json.loads((a.experiment/'initial-consistency.json').read_text())
    for height in HEIGHTS:
        for branch in BRANCHES:
            folder=a.experiment/height/branch
            if not (folder/'protocol.json').exists():continue
            report=canonical(folder,'fudan_ppo_walk')
            baseline=json.loads((a.recording/height/'acceptance/acceptance.json').read_text())
            report=compare_baseline(report,baseline)
            (folder/'acceptance').mkdir(exist_ok=True)
            (folder/'acceptance/acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
            unified[branch]['passed']+=report['technical_passed']
            unified[branch]['regressed'] += [height+'/'+i for i in report['comparison']['regressed']]
            for r in report['samples']:
                if r['case']['category'] not in ('zero','stop_backward','stop_forward'):continue
                sid=r['sample_id'];lo,hi=(2.,7.) if r['case']['category']=='zero' else (10.,15.)
                path=folder/'traces'/(sid+'.npz')
                if not path.exists():rows.append(dict(height=height,branch=branch,id=sid,complete=False,passed=False));continue
                with np.load(path) as f:trace={k:f[k] for k in f.files}
                metrics=parking_metrics(trace,lo,hi)
                with np.load(a.recording/height/'gpu15/traces'/(sid+'.npz')) as old:previous=parking_metrics(old,lo,hi)
                reproduced=metrics.get('complete',False) and abs(metrics['net_xy_m']-previous['net_xy_m'])<=.02
                # Also check the signed XY displacement vector, so direction changes cannot hide behind equal lengths.
                vector_difference=float(np.linalg.norm(np.array([metrics.get('net_x_m',float('inf')),metrics.get('net_y_m',0)])-np.array([previous['net_x_m'],previous['net_y_m']])))
                after=trace['integral_after'][:,0];comp=trace['compensation'][:,0];active=trace['active'][:,0].astype(bool)
                pre=trace['reason'][:,0].astype(int);post=trace['reason_after'][:,0].astype(int)
                exits=np.flatnonzero(active[:-1]&~active[1:])+1
                residual=[]
                for tick in exits:
                    residual.append(dict(time_s=tick*.01,reason=int(pre[tick]),integral_m=float(after[tick]),compensation_m_s=float(comp[tick]),velocity_m_s=trace['linear_world'][tick*5,:2].tolist()))
                coverage={name:bool(((pre==code)|(post==code)).any()) for name,code in REASONS.items()}
                rows.append(dict(height=height,branch=branch,id=sid,category=r['case']['category'],**metrics,
                    baseline_net_difference_m=vector_difference,baseline_reproduced=bool(reproduced and vector_difference<=.02),
                    integral_saturation_fraction=float((np.abs(after)>=.1-1e-7).mean()),compensation_saturation_fraction=float((np.abs(comp)>=.1-1e-7).mean()),
                    active_fraction=float(active.mean()),window_saturation_fraction=float((np.abs(after[round(lo/.01):round(hi/.01)])>=.1-1e-7).mean()),first_active_s=float(np.flatnonzero(active)[0]*.01) if active.any() else None,
                    exit_events=residual,gate_reason_observed=coverage,exit_condition_coverage={name:any(x['reason']==code for x in residual) for name,code in REASONS.items()},exit_effect_verified=bool(residual and all(x['integral_m']==0 and x['compensation_m_s']==0 for x in residual)),
                    note='Exit reasons absent in trajectories are NOT dynamically verified'))
    baseline_rows=[r for r in rows if r['branch']=='baseline']
    reproduced=len(baseline_rows)==9 and all(r.get('baseline_reproduced',False) for r in baseline_rows) and len(all_initial)==135 and all(r['all_arrays_identical'] for r in all_initial)
    passes={b:sum(r.get('passed',False) for r in rows if r['branch']==b) for b in BRANCHES}
    nonregression={b:unified[b]['passed']==45 and not unified[b]['regressed'] for b in BRANCHES}
    truth_ok=passes['truth_integral']==9 and nonregression['truth_integral']
    encoder_ok=passes['encoder_integral']==9 and nonregression['encoder_integral']
    if not reproduced:conclusion='基线复现不满足，不下改善结论'
    elif truth_ok and not encoder_ok:conclusion='当前积分方案受估速反馈限制'
    elif encoder_ok:conclusion='估速积分候选数值有效，待用户完整视频验收'
    else:conclusion='当前固定参数积分候选未达到停车要求'
    result=dict(conclusion=conclusion,baseline_reproduced=reproduced,parking_passed=passes,unified=unified,nonregression=nonregression,
        rows=rows,expected_trajectories=135,final_acceptance='pending_user_video_review',optimizer_updates=0,
        fixed_windows={'zero':[2,7],'stop':[10,15]},speed_mean_definition='mean of world XY speed magnitude; no signed cancellation')
    (a.experiment/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 8000轮停车积分试验', '',conclusion,'','基线复现：'+str(reproduced),'',
        '| 高度 / 工况 | 原模型：偏移cm / 均速cm/s | 真值积分 | 估速积分 |','|---|---:|---:|---:|']
    for h in HEIGHTS:
        for sid in ('diagnostic_0','diagnostic_13','diagnostic_14'):
            values=[]
            for b in BRANCHES:
                r=next((r for r in rows if r['height']==h and r['id']==sid and r['branch']==b),{})
                values.append(f"{100*r['maximum_horizontal_offset_m']:.2f} / {100*r['mean_horizontal_speed_m_s']:.3f}" if r.get('complete') else '缺失')
            lines.append('| '+h+' / '+sid+' | '+' | '.join(values)+' |')
    lines+=['',f'停车9例通过数：{passes}；原45例统一技术结果：{unified}。',
        '', '固定窗口：原地2～7秒，前后停车10～15秒。偏移包含窗口终点，速度为左闭右开500Hz速度模长统计。阈值：最大水平偏移≤5cm、均速≤0.01m/s、峰值≤0.05m/s。',
        '', '积分默认关闭；只改变策略命令vx，原用户命令用于评分。Actor始终使用原Encoder输出。Actor使用t时刻的原Encoder输出；环境完成t→t+.01后，用新状态真值和新历史的额外Encoder输出采样反馈并提交积分，下拍使用。真实标签是训练同口径10ms位置差分，Encoder除以2；每拍Actor一次、Encoder两次，第二次仅取反馈，不额外推进历史。',
        '', '初态无额外热步；t=0仅一个真实接触样本，凑满10个2ms样本后开始门控；首个合格检查计时0，连续50个10ms间隔后激活，首次激活拍补偿仍为0。正常历史由原env.step只推进一次。',
        '', '动态退出覆盖及饱和见comparison.json。没有发生的移动/转向/失稳/支撑丢失/复位退出不得当成通过；无推扰、无受扰返回原点功能，未修改H7/模型/ONNX/奖励/PD/物理，不训练。',
        '', '[完整比较与覆盖](comparison.json) · [预算](budget.json) · [执行结果](pipeline-result.json) · [视频索引](video-index.json)',
        '', '三份正常速度并排视频：[0.22m](videos/h022.mp4)、[0.26m](videos/h026.mp4)、[0.30m](videos/h030.mp4)。完整起步与刹停均保留，技术评分不代表最终通过，待用户观看确认。']
    (a.experiment/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(conclusion=conclusion,baseline_reproduced=reproduced,parking_passed=passes,unified=unified),ensure_ascii=False))

if __name__=='__main__':main()
