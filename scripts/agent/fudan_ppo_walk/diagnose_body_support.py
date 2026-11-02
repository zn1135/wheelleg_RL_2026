"""Read-only replay arithmetic over existing NPZ files; no simulator or optimizer imports."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

REPRESENTATIVES = (
    ('zero_0_0_0.22__50001', 2., 7., '零速'),
    ('forward_2_0_0.26__50001', 2., 7., '前进'),
    ('backward_-2_0_0.26__50001', 2., 7., '后退'),
    ('stop_1_0.26__50001', 10., 15., '停车后诊断'),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inverse_rotate(q, vector):
    xyz, w = q[:, 1:4], q[:, 0:1]
    cross = np.cross(xyz, vector)
    return vector - 2*w*cross + 2*np.cross(xyz, cross)


def statistics(value):
    return dict(mean=float(np.mean(value)), minimum=float(np.min(value)), maximum=float(np.max(value)))


def commands(case, policy_start_times):
    result = np.zeros((len(policy_start_times), 3))
    for segment in case['segments']:
        result[policy_start_times >= segment['start']-1e-8] = segment['command']
    return result


def diagnose(trace, case, lo, hi):
    t = trace['time']
    mask = (t >= lo-1e-8) & (t < hi-1e-8)
    weight = trace['weight'][mask]
    force = trace['base_contact'][mask]
    ratio = trace['wheel_support'][mask].sum(-1)/weight
    target = commands(case, t[mask])
    state = dict(window_s=[lo,hi], body_contact_N=statistics(force),
        body_contact_over_1N_fraction=float(np.mean(force > 1.)),
        wheels_weight_ratio=statistics(ratio), wheels_ratio_below_80pct_fraction=float(np.mean(ratio < .8)),
        height_m=statistics(trace['position'][mask,2]), target_height_m=statistics(target[:,2]),
        height_mae_m=float(np.mean(np.abs(trace['position'][mask,2]-target[:,2]))),
        pitch_abs_max_deg=float(np.abs(trace['roll_pitch_deg'][mask,1]).max()),
        instantaneous_body_vx_m_s=statistics(trace['linear_body'][mask,0]),
        instantaneous_vx_mae_m_s=float(np.abs(trace['linear_body'][mask,0]-target[:,0]).mean()))

    # The trained reward uses a 10ms position difference, whereas acceptance uses instantaneous velocity.
    qt = trace['qpos_time'][1:]
    q = trace['qpos'][1:,3:7].astype(np.float64)
    velocity = inverse_rotate(q, np.diff(trace['qpos'][:,:3].astype(np.float64),axis=0)/.01)
    cmd = commands(case, np.maximum(0., qt-.01))
    indices = np.rint(qt/.002).astype(int)
    assert indices.max() < len(t)
    assert np.allclose(t[indices],qt,atol=1e-7,rtol=0)
    qmask = (qt > lo+1e-8) & (qt <= hi+1e-8)
    qw,qx,qy,qz = q.T
    gravity = np.stack((2*(qw*qy-qx*qz),-2*(qy*qz+qw*qx),2*(qx*qx+qy*qy)-1),axis=-1)
    angular = trace['angular_body'][indices]
    vx_error = (cmd[:,0]-velocity[:,0])**2
    yaw_error = (cmd[:,1]-angular[:,2])**2
    orientation_raw = np.sum(gravity[:,:2]**2,axis=1)
    raw = dict(tracking_lin_vel=np.exp(-vx_error/.25), tracking_lin_vel_enhance=np.exp(-vx_error/2.5)-1,
        tracking_ang_vel=np.exp(-yaw_error/.25), tracking_ang_vel_enhance=np.exp(-yaw_error/2.5)-1,
        base_height=np.exp(-(trace['qpos'][1:,2]-cmd[:,2])**2/.001),
        orientation=orientation_raw,lin_vel_z=velocity[:,2]**2,
        ang_vel_xy=np.sum(angular[:,:2]**2,axis=1),collision=np.zeros_like(vx_error))
    weights = dict(tracking_lin_vel=1.,tracking_lin_vel_enhance=1.,tracking_ang_vel=1.,
        tracking_ang_vel_enhance=1.,base_height=1.,orientation=-100.,lin_vel_z=-1.,ang_vel_xy=-.2,collision=-1.)
    weighted = {name:np.clip(values*weights[name]*.01,-.01,.01) for name,values in raw.items()}
    subtotal = np.sum(np.stack(list(weighted.values())),axis=0)
    sampled_force = trace['base_contact'][indices]
    reconstructed_orientation = 1.-np.prod(np.cos(np.deg2rad(trace['roll_pitch_deg'][indices])),axis=1)**2
    result = dict(state=state,reward=dict(window_s=[lo,hi],sample_convention='100Hz action endpoints (lo,hi]',
        terms_mean_per_10ms={name:float(value[qmask].mean()) for name,value in weighted.items()},
        orientation_before_clamp_mean=float((-orientation_raw*.01*100)[qmask].mean()),
        orientation_saturated_fraction=float(np.mean(orientation_raw[qmask] >= .01)),
        known_nine_term_subtotal_mean=float(subtotal[qmask].mean()),
        contact_and_positive_known_subtotal_fraction=float(np.mean((sampled_force[qmask]>1.) & (subtotal[qmask]>0.))),
        full_reward_available=False,
        unavailable_terms=['nominal_state','dof_vel','dof_acc','torques','action_rate','action_smooth','dof_pos_limits'],
        orientation_formula_crosscheck_max_abs_error=float(np.abs(orientation_raw-reconstructed_orientation).max())))
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3]
    # Read frozen protocol JSON and original training source; do not construct an environment.
    protocol=json.loads((a.experiment/'final-gpu/protocol.json').read_text())
    lookup={s['id']:s for s in protocol['samples']}
    original=root/'logs/mjlab/fudan_ppo_walk/20261101_trial01/source/ct_mjlab/fudan_ppo_walk/env.py'
    text=original.read_text();assert 'collision=torch.zeros_like(error)' in text
    assert 'terms={k:(v*WEIGHTS[k]*POLICY_DT).clamp(-POLICY_DT,POLICY_DT)' in text
    rows=[];hashes={str(original):digest(original)}
    for engine in ('gpu','cpu'):
        for sid,lo,hi,label in REPRESENTATIVES:
            path=a.experiment/f'final-{engine}/traces'/(sid+'.npz')
            before=digest(path);hashes[str(path)]=before
            with np.load(path) as trace:row=diagnose(trace,lookup[sid]['case'],lo,hi)
            row.update(engine=engine,sample_id=sid,label=label,trajectory_sha256=before)
            assert digest(path)==before
            rows.append(row)
    threshold=float(np.rad2deg(np.arcsin(.1)))
    report=dict(scope='Existing trajectories and frozen source only; no new physics/optimizer updates',
        physics_steps=0,optimizer_updates=0,collision_term_always_zero=True,
        orientation_saturation_tilt_deg=threshold,rows=rows,input_sha256=hashes,
        reward_reconstruction='Nine state-derived terms only; 10ms finite-difference velocity. Not total return.',
        full_training_reward_or_causal_attribution_available=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines=['# 机身撑地：既有轨迹与源码诊断','',
        '**用户视频验收：不通过。** 用户反馈“机身撑地啊”。两引擎速度跟踪不能代表合格轮上行走。','',
        '本次只读已有NPZ及原训练源码，物理步/优化器更新均为0；输入哈希记录在report.json。机身力是接触合力的模，不能把它直接当竖直承重；双轮占比使用向上支撑力/真实体重。','',
        '| 引擎/工况 | 样本 | 窗口s | 机身力均值/峰值 N | 轮承重均值 | 实际/目标高度 m | 最大俯仰° | vx MAE m/s |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        s=row['state'];lines.append(f'| {row["engine"]}/{row["label"]} | {row["sample_id"]} | {s["window_s"]} | {s["body_contact_N"]["mean"]:.2f}/{s["body_contact_N"]["maximum"]:.2f} | {100*s["wheels_weight_ratio"]["mean"]:.2f}% | {s["height_m"]["mean"]:.4f}/{s["target_height_m"]["mean"]:.2f} | {s["pitch_abs_max_deg"]:.2f} | {s["instantaneous_vx_mae_m_s"]:.4f} |')
    lines += ['', '停车10～15秒为发零命令后3～8秒的固定诊断窗口，不替代原先按停稳时刻定义的正式评分窗口。其他工况为响应2秒后的5秒。','',
        '## 可由原轨迹复算的奖励子项','',
        '| 引擎/工况 | 线速跟踪 | yaw跟踪 | 高度 | 姿态裁剪前→后 | 姿态饱和比例 | collision | 已知9项合计 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        r=row['reward'];v=r['terms_mean_per_10ms'];lines.append(f'| {row["engine"]}/{row["label"]} | {v["tracking_lin_vel"]:.6f} | {v["tracking_ang_vel"]:.6f} | {v["base_height"]:.8f} | {r["orientation_before_clamp_mean"]:.6f}→{v["orientation"]:.6f} | {100*r["orientation_saturated_fraction"]:.1f}% | {v["collision"]:.0f} | {r["known_nine_term_subtotal_mean"]:.6f} |')
    lines += ['',
        '奖励表单位为每10ms策略步：先乘系数和dt，再逐项裁剪±0.01。原奖励线速度来自根位置10ms差分并转到当前机体系；上表状态vx MAE来自原500Hz瞬时速度，二者分开。四个跟踪项、height、orientation、lin_vel_z、ang_vel_xy、collision共9项可重建；缺少完整动作/力矩/2ms实体关节采样等，不能准确重放全部16项或总return。重建也不是原训练优化日志。','',
        '## 源码事实与证据边界','',
        '- 固定参考WheelLeggedCfg的penalize_contacts_on=[]、terminate_after_contacts_on=[]。collision函数对空集合求和为0；本机原训练实现直接返回zeros。权重−1并不产生任何碰撞罚，机身触地也没有直接训练终止条件。正式验收的接触失败不反向改变训练奖励或终止。',
        f'- orientation原式是重力投影水平分量平方和，权重−100。逐项裁剪后，当总倾角达到约{threshold:.3f}°就已为−0.01/步；继续倾斜不增加该项惩罚。姿态饱和并不意味着“增大−100”就有更强的大角度区分，且不能把饱和判定等同俯仰验收。',
        '- 机身撑地的状态仍能获得较高的速度/yaw跟踪正奖励；高度低于目标时，exp(−高度误差²/.001)可能已接近0，只是少拿高度奖励。这只证明本次失败轨迹仍获得跟踪子项正奖励；不能证明原机也如此、全16项总奖励为正或优化器因此收敛。',
        '- 首轮回合相位遗漏与首帧噪声不一致已修，修复版尚未重训。已有数据没有相同预算的修复前后对照，因此不能量化这两项遗漏对撑地的贡献。适配后的几何/控制、估速误差、探索及优化过程等因素未在本次只读分析中排除。','',
        '## 当前调查方向','',
        '用户要求只读核对固定复旦8204e853源码/模型来源与本次移植差异；不参考本仓旧经验提出奖励补丁。新增机身碰撞罚提案已撤回、未实施。',
        '按固定参考的注册/覆盖、采样与奖励时序、reset/历史、PD/噪声/随机化/PPO流程分类核对；详见docs/diagnostics/20261101-fudan-fixed-source-audit.md。',
        '本次没有物理步进、模型前向或优化器更新。']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(physics_steps=0,optimizer_updates=0,representative_rows=len(rows),orientation_saturation_tilt_deg=threshold),ensure_ascii=False))

if __name__=='__main__':main()
