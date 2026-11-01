"""Audit a300 world-only candidate and separate response amplitude from direction."""
import argparse
import json
from pathlib import Path
import torch
from ct_mjlab.supermimic.motion import WorldModel
from ct_mjlab.supermimic.state import NOISE
from scripts.agent.supermimic.diagnose_v13_v5_response import outputs,sha
from scripts.agent.supermimic.check_v13_world_repair import digest_state
from ct_mjlab.supermimic.learner import Learner


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--validation',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);run=args.run
    assert not (run/'audit.json').exists()
    result=json.loads((run/'result.json').read_text());recipe=json.loads((run/'recipe.json').read_text())
    candidate=torch.load(run/'candidate.pt',map_location='cpu',weights_only=False)
    parent_path=Path(candidate['parent_checkpoint']);assert sha(parent_path)==candidate['parent_sha256']==recipe['parent_sha256']
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    assert candidate['additional_world_updates']==result['actual_world_updates']==recipe['requested_updates']==300
    counts=Learner._saved_optimizer_counts(dict(parent,opt_world=candidate['opt_world']))
    assert counts==candidate['optimizer_update_counts']==dict(world=5500,encoder=300,actor=300,critic=300)
    for name,h in candidate['frozen_component_hashes'].items():assert digest_state(parent[name])==h
    for name,value in candidate['world'].items():
        assert torch.isfinite(value).all()
        if not name.startswith('net.'):assert torch.equal(value,parent['world'][name])
    assert sha(Path(args.validation))==recipe['validation_sha256']
    assert sha(run/'candidate.pt')==result['candidate_sha256']
    steps=[json.loads(row)['step'] for row in (run/'fit.jsonl').read_text().splitlines()]
    assert steps==list(range(1,301))
    assert result['accepted']==all(result['gates'].values())
    validation=torch.load(args.validation,map_location='cpu',weights_only=False)
    assert validation['seed']==recipe['validation_seed']!=recipe['training_seed']
    worlds={}
    for label,state in [('parent',parent['world']),('candidate',candidate['world'])]:
        world=WorldModel().eval().requires_grad_(False);world.load_state_dict(state);worlds[label]=world
    rows=[]
    for s in validation['samples']:
        predictions={}
        for label,w in worlds.items():
            predictions[label]={}
            for wheel in (2,5):
                for sign in ('-','+'):
                    key=f'{wheel}_{sign}0.10'
                    result_step=w.step(s['state'],s['parameters'],s['actions'][key],s['context'][:,:3],s['context'],
                        s['previous'],s['previous_previous'],s['state'][:,NOISE],return_path=True)
                    predictions[label][key]=outputs(w,result_step[-1])[:,0]
        for wheel in (2,5):
            plus,minus=f'{wheel}_+0.10',f'{wheel}_-0.10'
            truth=s['truth'][plus][:,0]-s['truth'][minus][:,0]
            eligible=(truth.abs()>torch.maximum(5*s['noise'][:,0],torch.tensor(1e-5)))&(s['action'][:,wheel].abs()+.1<3)
            row=dict(time_s=s['time_s'],wheel='left' if wheel==2 else 'right',eligible=int(eligible.sum()),models={})
            for label in worlds:
                estimate=predictions[label][plus]-predictions[label][minus]
                mask=eligible&(estimate.abs()>1e-5)
                row['models'][label]=dict(gain=float((estimate[eligible]*truth[eligible]).sum()/truth[eligible].square().sum()),
                    response_slope_mse=float(((estimate[eligible]-truth[eligible])/.2).square().mean()),
                    correct=int((mask&(estimate*truth>0)).sum()),opposite=int((mask&(estimate*truth<0)).sum()),
                    weak=int((eligible&~mask).sum()))
            rows.append(row)
    audit=dict(passed=True,world_updates=300,counts=counts,frozen_states_verified=True,norms_and_adapter_unchanged=True,
        exact_step_log=True,independent_seed_verified=True,candidate_accepted=result['accepted'],rows=rows,
        candidate_sha256=sha(run/'candidate.pt'),parent_sha256=sha(parent_path),scope='CPU read-only audit; no optimization or simulator calls')
    (run/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    verdict=('**通过预设预测验收，可保留为修正候选；控制行为尚未验证改善。**' if result['accepted'] else '**未通过全部验收，不采纳为正式修正模型。**')
    lines=['# V13 v5：300次预测模型响应修正结果','',
        verdict+' 原父模型保留，控制/估速/价值与归一化完全冻结。',
        f"实际新增世界模型300次，累计W5500/EAC各300；耗时{result['elapsed_s']:.2f}秒。",'',
        f"|独立seed{recipe['validation_seed']}指标|原模型|候选|",'|---|---:|---:|']
    b,a=result['before']['independent'],result['after']['independent']
    for title,key in [('方向正确/可判定','correct'),('反向','opposite'),('预测过弱','weak'),('速度响应幅度斜率MSE','vx_response_slope_mse'),('46维单步归一化MSE','endpoint_normalized_mse'),('8步回放归一化MSE','rollout8_normalized_mse')]:
        lines.append(f"|{title}|{b[key]:.8g}|{a[key]:.8g}|")
    change=(a['vx_response_slope_mse']/b['vx_response_slope_mse']-1)*100
    lines+=['',f"可判定分母均为{b['eligible']}；各项使用相同数据、相同父模型归一化尺度。响应幅度MSE变化{change:+.2f}%；预设门槛保持，没有根据结果放宽。",'',
        '|时刻s|轮|原响应增益|候选响应增益|原幅度MSE|候选幅度MSE|','|---|---|---:|---:|---:|---:|']
    for r in rows:
        b,a=r['models']['parent'],r['models']['candidate']
        lines.append(f"|{r['time_s']}|{r['wheel']}|{b['gain']:.3f}|{a['gain']:.3f}|{b['response_slope_mse']:.8g}|{a['response_slope_mse']:.8g}|")
    loss_note=('本次已直接约束LINK前进速度正负动作响应，目标尺度仅由训练数据计算。' if 'real_LINK_vx_action_slope_MSE' in recipe['losses'] else '本次仅约束46维平均响应，没有直接约束最终LINK速度响应；方向改善不保证幅度改善。')
    next_step=('下一步先在固定策略下扩展未见工况验证，再决定控制训练预算。' if result['accepted'] else '下一步先分析未过门槛的具体轮/时刻；当前候选不采纳，不自动追加训练。')
    lines+=['','增益1代表实际响应幅度；这里只是局部10ms指标，不是完整控制任务通过。',loss_note,next_step,
        '世界模型修正不改变冻结策略的真实动作，不能声称存活/平衡或停车已恢复。候选是特殊world-only包，不直接部署或完整续训。',
        '[实际配方](recipe.json) · [计数与冻结审计](audit.json) · [完整结果及验收门槛](result.json)']
    (run/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(passed=True,accepted=audit['candidate_accepted'],rows=rows),ensure_ascii=False))


if __name__=='__main__':main()
