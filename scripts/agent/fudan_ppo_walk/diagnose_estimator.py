"""Audit estimator labels/packing and existing predictions; no physics or forwards."""
import argparse
import ast
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ct_mjlab.fudan_ppo_walk.env import FudanEnv
from ct_mjlab.fudan_ppo_walk.contract import RESET_V2_CONTRACT
from ct_mjlab.fudan_ppo_walk.storage import RolloutStorage


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def body_rotation(q):
    q=np.asarray(q,dtype=np.float64);q=q/np.linalg.norm(q)
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def label_check(initial,mapping):
    checks=[]
    def verify(name,value):
        assert value,name
        checks.append(name)
    def forbidden(*a,**kw):raise AssertionError('Physics, policy forward or update forbidden')
    with ExitStack() as stack:
        for name in ('mujoco.mj_step','mujoco.mj_step1','mujoco.mj_step2','torch.nn.Linear.forward','torch.optim.Adam.step'):
            stack.enter_context(patch(name,side_effect=forbidden))
        env=object.__new__(FudanEnv);env.num_envs=1;env.device='cpu';env.randomized=False;env.reset_contract=RESET_V2_CONTRACT
        q=torch.tensor(initial['qpos'][None],dtype=torch.float32);v=torch.tensor(initial['qvel'][None],dtype=torch.float32)
        env.env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=q,qvel=v)),scene=SimpleNamespace(env_origins=torch.zeros(1,3)))
        env.robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=q[:,:3],root_link_quat_w=q[:,3:7],
            root_link_ang_vel_b=v[:,3:6],projected_gravity_b=torch.tensor([[0.,0.,-1.]])))
        env.action=SimpleNamespace(_motor_q_adr=torch.tensor(mapping['motor_q']),nominal_positions=torch.tensor(initial['nominal'][None],dtype=torch.float32),
            raw_action=torch.zeros(1,6),torques=torch.zeros(1,6),nominal_offset=torch.zeros(1,6),sample_velocity=torch.zeros(1,6))
        env.commands=torch.tensor([[-2.,0.,.30]]);env.previous=torch.zeros(1,6,2);env.previous_vel=torch.zeros(1,6)
        env.previous_position=q[:,:3].clone();env._update_state();frame_a=env.frame().clone();_,critic=env.observations()
        verify('reset_velocity_label_zero_with_nonzero_true_velocity',v[:,:3].abs().max()>0 and torch.equal(critic[:,:3],torch.zeros(1,3)))
        v[:,:3]*=-1.;env._update_state();frame_b=env.frame().clone()
        verify('different_root_linear_velocity_same_actor_frame',torch.equal(frame_a,frame_b))
        verify('repeated_initial_history_cannot_disambiguate_velocity',torch.equal(frame_a.repeat(1,5),frame_b.repeat(1,5)))
        q[:,0]+=.002;env._update_state();_,critic=env.observations()
        verify('label_uses10ms_position_difference_not_instantaneous_qvel',torch.allclose(env.linear[:,0],torch.tensor([.2]),atol=1e-6))
        verify('critic_label_scales_velocity_by2_once',torch.allclose(critic[:,0],torch.tensor([.4]),atol=1e-6))

        storage=RolloutStorage(2,3,[25],[144],[125],[6],device='cpu')
        for k in range(3):
            t=RolloutStorage.Transition();token=torch.tensor([k*100.,k*100.+10.])
            t.observations=torch.zeros(2,25);t.observation_history=torch.zeros(2,125)
            t.observation_history[:,:3]=torch.stack([token,token+1,token+2],dim=1)
            t.critic_observations=torch.zeros(2,144);t.critic_observations[:,:3]=t.observation_history[:,:3]
            t.next_observations=torch.zeros(2,25);t.next_observations[:,0]=token+10000
            t.actions=torch.zeros(2,6);t.rewards=torch.zeros(2);t.dones=torch.zeros(2,dtype=torch.bool)
            t.values=torch.zeros(2,1);t.actions_log_prob=torch.zeros(2);t.action_mean=torch.zeros(2,6);t.action_sigma=torch.ones(2,6)
            storage.add_transitions(t)
        seen=[]
        for future,critic,history in storage.encoder_mini_batch_generator(2,1):
            verify('shuffled_batch_history_matches_current_label',torch.equal(critic[:,:3],history[:,:3]))
            verify('future_observation_is_not_current_velocity_label',torch.equal(future[:,0]-10000,critic[:,0]))
            seen.extend(critic[:,0].tolist())
        verify('all_six_synthetic_transitions_retained',sorted(seen)==[0.,10.,100.,110.,200.,210.])
    return checks


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False);root=args.experiment
    b=json.loads((root/'budget.json').read_text());source=Path(b['source_recording']);checkpoint=Path(b['checkpoint'])
    assert sha(checkpoint)==b['checkpoint_sha256']
    frozen=checkpoint.parent/'source/ct_mjlab/fudan_ppo_walk';current=Path('mjlab_training/src/ct_mjlab/fudan_ppo_walk')
    invariants={}
    for name in ('ppo.py','storage.py','network.py'):
        assert (frozen/name).read_bytes()==(current/name).read_bytes(),name
        invariants[name]='byte_identical_to7000'
    def fn(path,name):return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
    for name in ('frame','_update_state'):
        assert ast.dump(fn(frozen/'env.py',name),include_attributes=False)==ast.dump(fn(current/'env.py',name),include_attributes=False),name
        invariants[name]='ast_identical_to7000'
    old_obs=fn(frozen/'env.py','observations');new_obs=fn(current/'env.py','observations')
    critic=lambda node:next(n for n in node.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='critic' for t in n.targets))
    assert ast.dump(critic(old_obs),include_attributes=False)==ast.dump(critic(new_obs),include_attributes=False)
    mapping=json.loads((source/'fixtures/mapping.json').read_text())
    with np.load(source/'fixtures/diagnostic_2_initial.npz',allow_pickle=False) as data:initial={k:data[k].copy() for k in data.files}
    checks=label_check(initial,mapping);rows={};inputs={}
    for name,folder in [('gpu_closed','runs'),('gpu_repeat','repeat-audit'),('cpu_pd_closed','runs')]:
        path=root/folder/(name+'.npz');inputs[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as data:d={k:data[k].copy() for k in data.files}
        pose=d['states_qpos'][::5][:100];vel=d['states_qvel'][::5][:100]
        rotations=np.asarray([body_rotation(q) for q in pose[:,3:7]])
        gravity=np.einsum('nji,j->ni',rotations,np.array([0.,0.,-1.]))
        previous=np.concatenate([np.zeros((1,6)),d['policy_action_applied'][:-1]])
        frame=np.concatenate([vel[:,3:6]*.25,gravity,np.tile([-4.,0.,1.5],(100,1)),
            (pose[:,mapping['motor_q']]-initial['nominal'])[:,[0,1,3,4]],d['states_sample_velocity'][::5][:100]*.05,previous],axis=1)
        pack_error=float(np.abs(np.clip(frame,-100.,100.)-d['policy_obs']).max());assert pack_error<1e-5
        hist=np.tile(d['policy_obs'][0],5);history_errors=[]
        for k in range(100):
            if k:hist=np.concatenate([hist[25:],d['policy_obs'][k]])
            history_errors.append(float(np.abs(hist-d['policy_history'][k]).max()))
        assert max(history_errors)==0.
        labels=np.zeros((100,3))
        for k in range(1,100):labels[k]=rotations[k].T@((pose[k,:3]-pose[k-1,:3])/.01)
        estimate=d['policy_estimated_velocity_scaled']/2.
        error=estimate-labels
        intervals={}
        for lo,hi in [(1,10),(10,30),(15,31),(50,100)]:
            intervals[f'{lo*.01:.2f}_{hi*.01:.2f}']=dict(vx_mae_m_s=float(np.abs(error[lo:hi,0]).mean()),
                vx_bias_m_s=float(error[lo:hi,0].mean()),axis_mae_m_s=np.abs(error[lo:hi]).mean(axis=0).tolist())
        lag=[];ks=np.arange(15,31)
        for shift in range(-10,11):
            lag.append(dict(shift_s=shift*.01,mae_m_s=float(np.abs(estimate[ks,0]-labels[ks-shift,0]).mean())))
        rows[name]=dict(frame_reconstruction_max_abs_error=pack_error,history_rebuild_max_abs_error=max(history_errors),
            history_abs_max=float(np.abs(d['policy_history']).max()),
            rollback_v3_history_exactly_identical_for_recorded_nominal_inputs=True,intervals=intervals,
            initial_estimate_m_s=estimate[0].tolist(),initial_instantaneous_velocity_m_s=d['policy_true_velocity_body'][0].tolist(),
            initial_training_label_m_s=labels[0].tolist(),at030s=dict(estimate_m_s=estimate[30].tolist(),
                position_difference_label_m_s=labels[30].tolist(),instantaneous_velocity_m_s=d['policy_true_velocity_body'][30].tolist()),
            lag_fit_015_to030=lag,estimated_m_s=estimate.tolist(),training_definition_labels_m_s=labels.tolist(),
            instantaneous_velocity_m_s=d['policy_true_velocity_body'].tolist())
    result=dict(scope='7000-v2 estimator labels and three existing1s nominal prediction traces; not new inference or training',
        checkpoint_sha256=b['checkpoint_sha256'],source_invariants=invariants,synthetic_checks=checks,
        current_history_and_current_label_alignment_verified=True,velocity_scale_verified=True,
        root_velocity_not_observable_in_repeated_initial_frame_verified=True,nominal_rollback_history_identity_verified=True,
        runs=rows,inputs_sha256=inputs,physics_steps=0,model_forwards=0,optimizer_updates=0,
        limits=['No saved original training rollout inputs/labels; cannot measure training-support coverage',
            'Observed CPU error does not establish independent causal contribution to pitch',
            'Historyv3 identity applies only to these recorded noiseless,unclipped inputs; not all randomized training data'])
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'check.json').write_text(json.dumps(dict(passed=True,synthetic_checks=len(checks),
        frame_pack_checks=3,history_identity_checks=3,physics_steps=0,model_forwards=0,optimizer_updates=0),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    colors={'gpu_closed':'#178457','cpu_pd_closed':'#c44736'}
    for name,color in colors.items():
        row=rows[name];est=np.asarray(row['estimated_m_s']);target=np.asarray(row['training_definition_labels_m_s']);t=np.arange(100)*.01
        axes[0].plot(t,est[:,0],color=color,label=name+' estimate')
        axes[0].plot(t,target[:,0],color=color,ls='--',label=name+' position-difference target')
        axes[1].plot(t[1:],(est-target)[1:,0],color=color,label=name)
    axes[0].set(title='Estimator output and same-definition training target',ylabel='Body vx (m/s)')
    axes[1].set(title='Positive error: backward speed is underestimated',ylabel='Estimate minus target (m/s)')
    for ax in axes:ax.set_xlabel('Time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(args.output/'estimator-error.png',dpi=160);plt.close(fig)
    lines=['# 估速原因核查：7000轮v2','',
        '已排除本例的倍率/字段/历史拼接错误和训练标签错一拍。已确认两类问题：复位首帧的信息不足及标签置零；CPU接触后这段状态下估速持续偏慢。没有用新的模型前向或物理试验作推断。','',
        '## 源码与程序检查','',
        '7000归档的ppo/storage/network与现源码逐字相同，frame及位置差分标签公式AST相同；PPO.act在env.step前同时保存当前history和当前critic[:3]速度标签，打乱小批量仍用同一索引。next_obs虽被生成器返回，但速度MSE用当前critic标签。真实RolloutStorage的6条带时间哨兵数据经打乱后仍逐条配对，未发现未来帧或错拍；×2在训练目标中一次，报告按÷2还原。','',
        '## 起步误差：先分清标签与真实运动','',
        '复位把previous_position设为当前位置，所以差分速度标签是0；真实根初速仍独立随机±.5m/s。本例真实vx−.445、估速+.031，而训练同口径复位标签0。因此不能仅凭第一帧就断言估速网络没学好。用真实FudanEnv.frame构造两组不同根线速度，其余传感状态相同，得到逐位相同的25维输入/125维重复历史，证明网络没有足够信息区分这两种初速。输入包括gyro/重力/命令/腿角/六电机速度/上次动作，不直接包含根线速度。','',
        '## 后续CPU偏慢是真实的同口径误差','',
        '| 运行 | .01～.10s vx MAE | .15～.31s vx MAE | .50～1s vx MAE | .30s估速／差分标签 |',
        '|---|---:|---:|---:|---:|']
    for name,row in rows.items():
        i=row['intervals'];a=row['at030s']
        lines.append(f"| {name} | {i['0.01_0.10']['vx_mae_m_s']:.3f} | {i['0.15_0.31']['vx_mae_m_s']:.3f} | {i['0.50_1.00']['vx_mae_m_s']:.3f} | {a['estimate_m_s'][0]:.3f} / {a['position_difference_label_m_s'][0]:.3f} |")
    lines.extend(['','误差单位m/s。GPU/CPU最初约.1s误差几乎相同，之后CPU明显变差；CPU.30s估速−.955、10ms差分标签−1.408，误差+.453，不能由瞬时/平均速度定义差别解释。输入/历史开始分叉、估速误差扩大与动作变化互相影响；目前能定位到当前网络在这段CPU反馈状态下失准，但没有原训练逐环境数据，不能证明它从未训练过这些状态，也不能仅凭相关性确定估速是倾角的独立起因。','',
        '## 不是历史格式或固定时间延迟的直接证据','',
        '从每个保存状态独立重建25维输入，最大误差约1e−7，支持坐标/顺序/缩放一致。三条记录各100帧的最新帧、FIFO推进均核对通过；输入历史最大幅度不足100，名义无噪声，首帧五份相同。用history-v3的“首帧重复5次、之后clip帧FIFO”重建，125维数据与这些v2记录逐位相同。所以仅切换这段回放的历史格式不会改变输入；重训对其他噪声/极端状态的影响不能由此判断。当前帧已在历史末尾，不存在必须等满50ms才输出的结构性延迟；移位拟合只能说明曲线相似，不能把学习误差认定为固定通信/采样延迟。','',
        '下一步优先在同初态、同物理/驱动曲线下，仅在独立诊断中用同口径真实速度替换encoder三维输出，与原输出配对比较倾角、轮速触边和实际力矩；这能检验估速误差究竟贡献多少，不改正式输入接口或训练配方。本次未执行该新动态分支，也未训练。','',
        '[误差曲线](estimator-error.png)、[全部时序/标签/检查JSON](report.json)。'])
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    for path,digest in inputs.items():assert sha(Path(path))==digest
    print(json.dumps(dict(passed=True,synthetic_checks=len(checks),runs={name:{k:v for k,v in row.items() if k in ('frame_reconstruction_max_abs_error','history_rebuild_max_abs_error','history_abs_max','intervals','at030s')} for name,row in rows.items()},physics_steps=0,model_forwards=0,optimizer_updates=0),indent=2))


if __name__=='__main__':main()
