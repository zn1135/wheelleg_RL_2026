"""Inspect spring extension sign and compare zero/400 N on the GPU."""
from pathlib import Path
import json

import mujoco as mj
import mujoco_warp as mw
import numpy as np
import torch
import warp as wp

source=Path("artifacts/urdf-align/20261126-001-closed/robot_001_closed_mouse_drag.xml").resolve()
spec=mj.MjSpec.from_file(str(source))
for name in ("lfwheel_motor","rfwheel_motor"):
    spec.actuator(name).gaintype=mj.mjtGain.mjGAIN_FIXED
model=spec.compile()
data=mj.MjData(model)
mj.mj_resetDataKeyframe(model,data,model.key("manual_center").id)
mj.mj_forward(model,data)
report={"model":str(source),"sides":{}}
spring_ids=[model.actuator(f"{side}_gas_spring_motor").id for side in ("left","right")]
for side,prefix in (("left","lf"),("right","rf")):
    upper=data.site(f"{side}_gas_spring_upper").xpos.copy()
    lower=data.site(f"{side}_gas_spring_lower").xpos.copy()
    joint=model.joint(f"{side}_spring2_joint")
    world_axis=data.xaxis[joint.id].copy()
    delta=lower-upper
    axis_projection=float(np.dot(delta/np.linalg.norm(delta),world_axis))
    assert axis_projection>0.98
    qadr=joint.qposadr[0]
    initial_length=np.linalg.norm(delta)
    data.qpos[qadr]+=1e-5
    mj.mj_kinematics(model,data)
    increased_length=np.linalg.norm(data.site(f"{side}_spring_tip").xpos-upper)
    assert increased_length>initial_length
    data.qpos[qadr]-=1e-5
    mj.mj_forward(model,data)
    data.ctrl[spring_ids]=400
    mj.mj_forward(model,data)
    assert data.actuator_force[model.actuator(f"{side}_gas_spring_motor").id]==400
    report["sides"][side]=dict(
        upper_world_m=upper.tolist(),lower_world_m=lower.tolist(),
        positive_slide_axis=world_axis.tolist(),length_change_for_positive_10um_slide_m=float(increased_length-initial_length),
        cosine_axis_to_upper_lower=axis_projection,
        actuator_force_N=400.)
    print(side,report["sides"][side],flush=True)

wp.init()
torch.empty(1,device="cuda")
with wp.ScopedDevice("cuda:0"),wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
    m=mw.put_model(model)
    d=mw.put_data(model,data,nworld=2,nconmax=32,njmax=128)
    ctrl=wp.to_torch(d.ctrl)
    ctrl[0,spring_ids]=0
    ctrl[1,spring_ids]=400
    sites=wp.to_torch(d.site_xpos)
    q=wp.to_torch(d.qpos)
    upper_ids=[model.site(f"{s}_gas_spring_upper").id for s in ("left","right")]
    lower_ids=[model.site(f"{s}_gas_spring_lower").id for s in ("left","right")]
    slider_pos=[int(model.joint(f"{s}_spring2_joint").qposadr[0]) for s in ("left","right")]
    qinitial=data.qpos.copy()
    for step in range(25):
        mw.step(m,d)
        if step in (0,4,24):
            lengths=(sites[:,lower_ids]-sites[:,upper_ids]).norm(dim=-1).cpu().numpy()*1000
            travel=(q[:,slider_pos].cpu().numpy()-qinitial[slider_pos])*1000
            print("time",(step+1)*model.opt.timestep,"length_mm zero/400N",lengths.tolist(),"travel_mm zero/400N",travel.tolist(),flush=True)
            if step==24:
                report["gpu_50ms"]=dict(lengths_mm_zero_400N=lengths.tolist(),travel_mm_zero_400N=travel.tolist())
                assert (lengths[1]>lengths[0]).all()
    assert torch.isfinite(q).all()
Path("artifacts/urdf-align/20261126-001-spring-direction/result.json").write_text(json.dumps(report,indent=2)+"\n")
