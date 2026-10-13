#!/usr/bin/env python3
"""离线移植已求解的华南虎式 CAD 初态；核对几何、转换电机坐标并验证闭合。"""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from ct_mjlab.config import ASSET

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',type=Path,default=Path('/home/zn/文档/mjlab/wheelleg_RL'))
a=p.parse_args()
base=a.reference/'assets/chuanliantui'
bank_path=base/'huananhu_pose_bank.npz'
reference=mujoco.MjModel.from_xml_path(str(base/'chuanliantui.xml'))
model=mujoco.MjModel.from_xml_path(str(ASSET));data=mujoco.MjData(model);other=mujoco.MjData(reference)
left=('lf0','lf1','lfwheel','left_spring1_joint','left_spring2_joint','lf00','lf01','lf02','lf03')
right=('rf0','rf1','rfwheel','right_spring1_joint','right_spring2_joint','rf00','rf01','rf02','rf03')
with np.load(bank_path,allow_pickle=False) as source:
    payload={k:source[k].copy() for k in source.files}
    # Same CAD frame transforms are required; never blindly reuse another robot's bank.
    for bid in range(1,model.nbody):
        name=model.body(bid).name;rb=reference.body(name)
        assert np.allclose(model.body_pos[bid],reference.body_pos[rb.id],atol=2e-9),name
        assert np.allclose(model.body_quat[bid],reference.body_quat[rb.id],atol=1e-10),name
    max_error=0.;max_reference_delta=0.
    for mode in ('normal','positive','negative'):
        for side,names in (('left',left),('right',right)):
            values=payload[f'{mode}_{side}_pos'].astype(np.float64)
            for col,name in enumerate(names):
                dst=model.joint(name);src=reference.joint(name)
                assert np.allclose(dst.axis,src.axis),name
                values[:,col]+=model.qpos0[dst.qposadr[0]]-reference.qpos0[src.qposadr[0]]
            payload[f'{mode}_{side}_pos']=values.astype(np.float32)
        for row in range(int(payload['count'])):
            mujoco.mj_resetData(model,data);mujoco.mj_resetData(reference,other)
            data.qpos[2]=other.qpos[2]=0.
            for side,names in (('left',left),('right',right)):
                for col,name in enumerate(names):
                    joint=model.joint(name);rjoint=reference.joint(name)
                    value=payload[f'{mode}_{side}_pos'][row,col]
                    assert not joint.limited or joint.range[0]-1e-6<=value<=joint.range[1]+1e-6,(mode,name,value)
                    data.qpos[joint.qposadr[0]]=value
                    other.qpos[rjoint.qposadr[0]]=source[f'{mode}_{side}_pos'][row,col]
            mujoco.mj_forward(model,data);mujoco.mj_forward(reference,other)
            for side in ('left','right'):
                pairs=[(side+'_rear_pin'+i,side+'_front_pin'+i) for i in ('1','2')]
                pairs.append((side+'_spring_tip',side+'_gas_spring_lower'))
                for first,second in pairs:
                    max_error=max(max_error,float(np.linalg.norm(data.site(first).xpos-data.site(second).xpos)))
                    max_reference_delta=max(max_reference_delta,float(np.linalg.norm(data.site(first).xpos-other.site(first).xpos)))
    assert max_error<1e-6 and max_reference_delta<1e-6,(max_error,max_reference_delta)
    payload.update(model_hash=np.array(hashlib.sha256(ASSET.read_bytes()).hexdigest()),joint_names=np.array(left+right))
    out=Path(__file__).resolve().parents[2]/'mjlab_training/src/ct_mjlab/reset_poses.npz'
    np.savez_compressed(out,**payload)
    record=dict(reference_bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        asset_sha256=str(payload['model_hash']),rows_per_mode=int(payload['count']),
        max_closed_error_m=max_error,max_reference_site_delta_m=max_reference_delta,
        source='SCUT V14 flat v0 sampling; wheelleg_RL CAD retargeted pose bank',
        mapping='原生关节按qpos0差值转换；运行时无外部路径依赖。',
        source_normal_length_range_m=[.13,.32],source_ground_length_range_m=[.14,.36],
        robot_length_range_m=payload['length_range'].tolist(),
        clearance='沿用已核对同一CAD几何的保守碰撞高度界；轮胎界覆盖任意轮角。')
    out.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))
