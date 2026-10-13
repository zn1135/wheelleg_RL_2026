#!/usr/bin/env python3
"""离线求解 CAD 闭链装配初值；输出实体关节缓存，训练不调用反算器。"""
from pathlib import Path
import json
import hashlib
import mujoco
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
m=mujoco.MjModel.from_xml_path(str(ASSET));d=mujoco.MjData(m)
def pose(fronts,knees):
    values={}
    # Inputs describe physical CAD rotations, independent of motor coordinates.
    reference=lambda name:float(m.qpos0[m.jnt_qposadr[m.joint(name).id]])
    for prefix,front,knee in zip(('lf','rf'),fronts,knees):
        rear=[prefix+s for s in ('00','01','02','03')]
        front+=reference(prefix+'0');knee+=reference(prefix+'1')
        rear_ref=np.array([reference(name) for name in rear])
        adr=lambda n:m.jnt_qposadr[m.joint(n).id]
        side='left' if prefix=='lf' else 'right'
        def residual(x):
            mujoco.mj_resetData(m,d);d.qpos[adr(prefix+'0')]=front;d.qpos[adr(prefix+'1')]=knee
            for name,q in zip(rear,x):d.qpos[adr(name)]=q+reference(name)
            mujoco.mj_forward(m,d)
            return np.concatenate([d.site(side+'_rear_pin'+i).xpos-d.site(side+'_front_pin'+i).xpos for i in ('1','2')])
        sol=least_squares(residual,np.zeros(4),bounds=(-np.pi,np.pi),xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=2000)
        assert sol.success and np.max(np.abs(residual(sol.x)))<1e-8
        values.update({prefix+'0':front,prefix+'1':knee,prefix+'wheel':0})
        values.update(zip(rear,(sol.x+rear_ref).tolist()))
    mujoco.mj_resetData(m,d)
    for name,q in values.items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=q
    mujoco.mj_forward(m,d)
    for side,parent in (('left','lf0'),('right','rf0')):
        upper=d.site(side+'_gas_spring_upper').xpos
        lower=d.site(side+'_gas_spring_lower').xpos
        delta=d.body(parent).xmat.reshape(3,3).T@(lower-upper)
        quat=m.body(side+'_spring1').quat
        angle=2*np.arctan2(quat[2],quat[0])
        length=m.body(side+'_spring2').pos[2]
        values[side+'_spring1_joint']=float(np.arctan2(delta[0],delta[2])-angle)
        values[side+'_spring2_joint']=float(np.hypot(delta[0],delta[2])-length)
    for name,q in values.items():d.qpos[m.jnt_qposadr[m.joint(name).id]]=q
    mujoco.mj_forward(m,d)
    for side in ('left','right'):
        assert np.linalg.norm(d.site(side+'_spring_tip').xpos-d.site(side+'_gas_spring_lower').xpos)<1e-8
    return values
out=ROOT/'mjlab_training/src/ct_mjlab/poses.json'
out.write_text(json.dumps(dict(asset_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
    root_height=.15,reset=pose((1.566,-1.566),(0,0)),nominal=pose((.06,-.06),(-.1,.1))),indent=2)+'\n')
print(out)
