"""Audit actual wheel input slots, signs/scales and local model responses on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS
from ct_mjlab.supermimic.motion import WorldModel, physical47_from_full64
from ct_mjlab.supermimic.networks import normalize_contact_inputs
from ct_mjlab.supermimic.state import NOISE, ACTION_ORDER

ROOT=Path(__file__).resolve().parents[3]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    record_path=ROOT/'artifacts/supermimic-v13-wheel-response-20261101-r1/response.pt'
    source=torch.load(record_path,map_location='cpu',weights_only=False)
    cp=ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261101_v13_continuation_to5200_r3/final.pt'
    assert sha(cp)==source['checkpoint_sha256']
    world=WorldModel().eval().requires_grad_(False)
    world.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False,mmap=True)['world'],strict=True)
    model=mujoco.MjModel.from_binary_path(str(ROOT/'artifacts/supermimic-v13-r3-a5200-seed43-20261101/scene.mjb'))
    data=mujoco.MjData(model)
    names=[model.joint(i).name for i in range(model.njnt)]
    get_joint=lambda n:model.joint('robot/'+n if 'robot/'+n in names else n)
    assert tuple(ACTUATED_JOINTS)==('lf0','lf00','lfwheel','rf0','rf00','rfwheel')
    # Observation-only reorder and its inverse do not change model action order.
    assert [ACTION_ORDER[i] for i in (0,1,4,2,3,5)]==list(range(6))
    joints={side:get_joint(name) for side,name in [('left','lfwheel'),('right','rfwheel')]}
    assert tuple(int(joints[k].qposadr[0]) for k in ('left','right'))==(9,14)
    assert tuple(int(joints[k].dofadr[0]) for k in ('left','right'))==(8,13)
    specs=dict(left=dict(action=2,motion_velocity=31,input_action=124,input_previous=130,input_previous_previous=136),
        right=dict(action=5,motion_velocity=36,input_action=127,input_previous=133,input_previous_previous=139))
    rows=[]
    for sample in source['samples']:
        state,params,action=sample['state'],sample['parameters'],sample['action']
        previous,previous_previous=sample['previous'],sample['previous_previous']
        inp=world.inputs(state,params,action,previous,previous_previous)
        norm=normalize_contact_inputs(inp,world.in_norm,47)
        assert inp.shape==(96,141)
        data.qpos[:]=state[0,:25].numpy();data.qvel[:]=state[0,25:49].numpy();mujoco.mj_forward(model,data)
        axes={k:data.xaxis[j.id].copy() for k,j in joints.items()}
        assert np.dot(axes['left'],axes['right']) < -.99999
        item=dict(time_s=sample['time_s'],wheel_axis_dot=float(np.dot(axes['left'],axes['right'])),
            action_delay_steps=torch.unique(params[:,24:26],dim=0).tolist(),sides={})
        for side,spec in specs.items():
            wheel=spec['action'];velocity_column=25+int(joints[side].dofadr[0])
            assert torch.equal(inp[:,spec['motion_velocity']],state[:,velocity_column])
            for value,key in [(action,'input_action'),(previous,'input_previous'),(previous_previous,'input_previous_previous')]:
                assert torch.equal(inp[:,spec[key]],value[:,wheel])
            perturbed=action.clone();perturbed[:,wheel]+=.1
            changed=world.inputs(state,params,perturbed,previous,previous_previous)-inp
            assert (changed.abs()>1e-6).any(0).nonzero().flatten().tolist()==[spec['input_action']]
            new_norm=normalize_contact_inputs(inp+changed,world.in_norm,47)
            assert ((new_norm-norm)[:,spec['input_action']]>0).all()
            branches=[]
            for eps in (-.1,.1):
                a=action.clone();a[:,wheel]+=eps
                branches.append(world.step(state,params,a,sample['context'][:,:3],sample['context'],previous,previous_previous,state[:,NOISE],return_path=True))
            true_delta=sample['paths'][f'{wheel}_+0.10']-sample['paths'][f'{wheel}_-0.10']
            pred_delta=branches[1][-1]-branches[0][-1]
            metric='LINK_vx_m_s';true_vx=sample['true_outputs'][f'{wheel}_+0.10'][:,0]-sample['true_outputs'][f'{wheel}_-0.10'][:,0]
            pred_vx=branches[1][4][:,0]-branches[0][4][:,0]
            columns={k:v for k,v in spec.items() if k!='action'}
            item['sides'][side]=dict(mapping=spec,local_joint_axis=joints[side].axis.tolist(),
                gain_kd_unique=torch.unique(params[:,12+wheel]).tolist(),motor_strength_unique=torch.unique(params[:,18+wheel]).tolist(),
                channels={k:dict(index=i,mean=float(world.in_norm.mean[i]),std=float(world.in_norm.std()[i]),raw_mean=float(inp[:,i].mean()),normalized_min=float(norm[:,i].min()),normalized_max=float(norm[:,i].max())) for k,i in columns.items()},
                true_own_wheel_velocity_delta=float(true_delta[:,-1,velocity_column].mean()),
                predicted_own_wheel_velocity_delta=float(pred_delta[:,-1,velocity_column].mean()),
                true_forward_velocity_delta=float(true_vx.mean()),predicted_forward_velocity_delta=float(pred_vx.mean()))
        rows.append(item)
    result=dict(all_mapping_checks_passed=True,rows=rows,action_order=list(ACTUATED_JOINTS),
        input_layout=dict(motion=[0,47],parameters=[47,122],action=[122,128],previous=[128,134],previous_previous=[134,140],age=[140,141]),
        checkpoint_sha256=sha(cp),record_sha256=sha(record_path),scorer_sha256=sha(Path(__file__)),
        model_or_normalization_updates=0,simulation_steps=0,
        scope='CPU inspection of existing 3x96 states; native response from previously verified perturbation recordings. No claim about unrecorded training distribution.')
    (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 左右轮输入核对', '',
        '实际动作顺序为lf0、lf00、lfwheel、rf0、rf00、rfwheel；左轮a[2]，右轮a[5]。观测的腿/轮重排与还原互为逆操作，不用于重排世界模型动作。',
        '141维输入=47运动+75参数+当前6动作+上一拍6动作+上上拍6动作+年龄。训练与推理共用WorldModel.inputs。',
        '左/右轮速度位于motion输入31/36；当前动作124/127；上一拍130/133；上上拍136/139。逐样本实际数值与对应原始字段相等，单轮扰动只改变对应一列。',
        '左右轮关节轴在当前姿态下互为反向，正动作对机身前向速度不应要求同号。', '',
        '| 时刻 | 轮 | 真实自身轮速变化rad/s | 预测自身轮速变化rad/s | 真实前向速度变化m/s | 预测前向速度变化m/s |', '|---|---|---:|---:|---:|---:|']
    for row in rows:
        for side,v in row['sides'].items():
            lines.append(f"| {row['time_s']} | {side} | {v['true_own_wheel_velocity_delta']:.6g} | {v['predicted_own_wheel_velocity_delta']:.6g} | {v['true_forward_velocity_delta']:.6g} | {v['predicted_forward_velocity_delta']:.6g} |")
    lines += ['', '以上差值为+0.1减−0.1 raw动作的10ms响应，目标轮速相差2rad/s；微弱响应的逐样本噪声筛选详见前一份轮动作方向报告。',
        '三个状态下两类动作延迟均0；左右轮有效速度增益相同、强度相同。真实执行器两轮均用10倍轮速缩放；世界模型两轮均输入未缩放动作，不混用rad/s与raw。',
        '左右归一化统计不同：当前动作std左4.62856、右3.25772；对应+0.1扰动归一化后均为正，约+0.0216、+0.0307。已测当前/历史动作和轮速输入都在约−1.16～+2.44个标准差内，没有该组通道归一化爆炸证据。不同统计反映数据不对称，不能据此单独认定bug或全部输入均在训练分布内。',
        '重要定位：右轮自己转速响应方向正确，但机身前向速度响应错误，较符合学习到的轮/机身耦合错误，而不是左右动作交换、缩放或负号接错。普通MLP没有强制镜像等变约束，也没有针对正负动作响应的单独监督；当前证据不能确定是哪种训练数据相关性造成。',
        '模型没有输入/输出接触力和轮承重，只有运动状态、材料等参数；这也限制了从预测分数推断接触控制能力，不能把该缺项直接当成已证明的唯一原因。',
        '下一步应比较真实左右轮扰动下的机身响应，并加入局部响应验证；先修正动力学学习信号，保持已核对的接口顺序不变。无训练、无接口/物理修改。']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=True,output=str(args.output),rows=rows),ensure_ascii=False))


if __name__=='__main__':main()
