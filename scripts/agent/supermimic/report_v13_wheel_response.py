"""Noise-qualified native/model wheel response and model-gradient comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from ct_mjlab.supermimic.motion import WorldModel
from ct_mjlab.supermimic.state import NOISE

ROOT=Path(__file__).resolve().parents[3]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--recording',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(2);out=args.recording/'report';out.mkdir(exist_ok=False)
    saved=torch.load(args.recording/'response.pt',map_location='cpu',weights_only=False)
    assert saved.get('intervention','current_wheel_action')=='current_wheel_action', 'Use the history-control report for history-only perturbations'
    cp=ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261101_v13_continuation_to5200_r3/final.pt'
    assert sha(cp)==saved['checkpoint_sha256']
    world=WorldModel().eval().requires_grad_(False)
    world.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False,mmap=True)['world'],strict=True)
    floors=torch.tensor([1e-5,1e-6,1e-6,1e-7]);rows=[];gradient_checks=[];noise_rows=[]
    for s in saved['samples']:
        real=s['true_outputs'];pred=s['predicted_outputs'];noise=torch.maximum(abs(real['base']-real['repeat']),abs(s['native_outputs']-real['base']))
        noise_rows.append(dict(time_s=s['time_s'],max_absolute_noise=noise.max(0).values.tolist(),base_contact_max_N=s['base_contact_max_N']))
        action=s['action'].clone().requires_grad_(True)
        p=world.step(s['state'],s['parameters'],action,s['context'][:,:3],s['context'],s['previous'],s['previous_previous'],s['state'][:,NOISE],return_path=True)
        values=torch.stack((p[4][:,0],p[-1][:,-1,2],p[4][:,9],p[2]),-1)
        grad=torch.stack([torch.autograd.grad(values[:,j].sum(),action,retain_graph=True)[0][:,[2,5]] for j in range(4)],-1).detach()
        for wheel,k in ((2,0),(5,1)):
            for eps in (.05,.1):
                dr=real[f'{wheel}_{eps:+.2f}']-real[f'{wheel}_{-eps:+.2f}']
                dp=pred[f'{wheel}_{eps:+.2f}']-pred[f'{wheel}_{-eps:+.2f}']
                eligible=abs(dr)>torch.maximum(5*noise,floors)
                pred_detectable=abs(dp)>floors
                valid=eligible&pred_detectable
                for j,name in enumerate(saved['metrics']):
                    same=(dr[:,j]*dp[:,j]>0)&valid[:,j]
                    opp=(dr[:,j]*dp[:,j]<0)&valid[:,j]
                    gv=eligible[:,j]&(abs(grad[:,k,j])*2*eps>floors[j])
                    rows.append(dict(time_s=s['time_s'],wheel='left' if wheel==2 else 'right',raw_action_perturbation=eps,
                        wheel_target_perturbation_rad_s=10*eps,metric=name,total_cases=96,
                        native_effect_detectable=int(eligible[:,j].sum()),both_effects_detectable=int(valid[:,j].sum()),
                        same_direction=int(same.sum()),opposite_direction=int(opp.sum()),
                        true_plus_minus_mean=float(dr[:,j].mean()),predicted_plus_minus_mean=float(dp[:,j].mean()),
                        gradient_eligible=int(gv.sum()),gradient_opposite=int(((dr[:,j]*grad[:,k,j]<0)&gv).sum()),
                        truth_delta=dr[:,j].tolist(),prediction_delta=dp[:,j].tolist(),eligible_ids=valid[:,j].nonzero().flatten().tolist()))
                fd=dp/(2*eps)
                gradient_checks.append(dict(time_s=s['time_s'],wheel=wheel,epsilon=eps,
                    finite_difference_vs_autograd_mae=abs(fd-grad[:,k]).mean(0).tolist(),
                    gradients_mean=grad[:,k].mean(0).tolist()))
    result=dict(rows=rows,noise=noise_rows,gradient_checks=gradient_checks,metric_floors=floors.tolist(),
        noise_multiplier=5,scope='One policy step 10ms; branch state and PD history restored. Motion/reward quantities at original substep timing; fixed context. Not the full 240ms actor objective or critic gradient.',
        reward_scope='True reward uses recorded native torques with zero base contact (verified); model reward uses its existing analytic torque approximation. Reward disagreement includes this approximation.',
        checkpoint_sha256=saved['checkpoint_sha256'],source_data_sha256=sha(args.recording/'response.pt'),parameter_updates=0,
        note='No estimate for cases below detectability threshold; excluded cases remain explicit in original96 denominator.')
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# A5200轮动作局部预测方向', '',
        '每个时刻96个状态；从相同原生GPU物理状态和PD历史分支，左右轮分别±0.05/±0.10 raw动作（目标轮速±0.5/±1rad/s），运行5×2ms。没有训练。',
        '下表用±0.10：差值为“加扰动结果−减扰动结果”；两者间目标轮速相差2rad/s。只统计真实效应超过5倍恢复/重跑误差且双方均超过绝对噪声门槛的样本。', '',
        '| 时刻 | 轮 | 指标 | 真实平均差值 | 模型平均差值 | 方向相反/可判定/全部 |', '|---|---|---|---:|---:|---|']
    for row in rows:
        if row['raw_action_perturbation']!=.1 or row['metric']=='reward_height_m':continue
        lines.append(f"| {row['time_s']} | {row['wheel']} | {row['metric']} | {row['true_plus_minus_mean']:.8g} | {row['predicted_plus_minus_mean']:.8g} | {row['opposite_direction']}/{row['both_effects_detectable']}/96 |")
    lines += ['', '直接自动微分梯度与有限差分方向结果见report.json。右轮速度响应是否反向在两种扰动幅度下分别核对，不把低于噪声的高度效应记为反向。',
        '奖励比较包含既有模型解析力矩近似与真实力矩差异，不能把全部奖励误差归因于运动预测。',
        '只覆盖早期0.2/0.4/0.6秒、A5200和一步10ms；不能直接证明整个策略损失的梯度方向或唯一训练根因。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(out),right_vx=[{k:v for k,v in r.items() if k not in ('truth_delta','prediction_delta','eligible_ids')} for r in rows if r['wheel']=='right' and r['metric']=='LINK_vx_m_s']),ensure_ascii=False))


if __name__=='__main__':main()
