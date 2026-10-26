#!/usr/bin/env python3
"""Check shared wheel/LQR position resets and unchanged default controller; no training."""
import argparse,importlib.util,json,math
from pathlib import Path
import numpy as np
import torch
from ct_mjlab.teacher.controller import LqrVmcController
from check_lqr_vmc import hip_angles


@torch.inference_mode()
def check(args):
    spec=importlib.util.spec_from_file_location('ct_mjlab.teacher._before_position_alignment',args.before)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    count=20;device='cpu';q=torch.tensor(hip_angles(.30,0.)).repeat(count,2,1)
    dq=torch.zeros_like(q);zero=torch.zeros(count);torque=torch.zeros_like(q)
    command=torch.tensor([0.,0.,.30]).repeat(count,1)
    common=dict(velocity_integral=True,dt=.01)
    original=module.LqrVmcController(count,device,**common);current=LqrVmcController(count,device,**common)
    for c in (original,current):c.reset(slice(None),standing=True)
    maximum=0.
    for tick in range(300):
        wheel=torch.full((count,2),.6+.3*math.sin(tick*.03));pitch=torch.full((count,),.01*math.sin(tick*.02))
        inputs=(q,dq,wheel,pitch,zero,zero,zero,zero,torch.ones(count),zero,zero,torque,command)
        a=original.step(*inputs);b=current.step(*inputs)
        maximum=max(maximum,float((a-b).abs().max()));torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(original.pos,current.pos,atol=0,rtol=0)
    shared=LqrVmcController(count,device,**common);shared.reset(slice(None),standing=True)
    latched=torch.zeros(count,dtype=torch.bool);first=latched.clone();total=torch.zeros(count,dtype=torch.long)
    for tick in range(40):
        if tick==10:latched[::2]=True
        if tick==20:latched[1::2]=True
        inputs=(q,dq,torch.ones(count,2),zero,zero,zero,zero,zero,torch.ones(count),zero,zero,torque,command)
        shared.step(*inputs,position_reference=latched)
        expected=latched & ~first
        torch.testing.assert_close(shared.position_reference_event,expected,atol=0,rtol=0)
        assert not shared.x[~latched,0].any()
        assert not shared.x[expected,0].any() and not shared.contributions[expected,:,0].any()
        total+=shared.position_reference_event.long();first=latched.clone()
    assert (total==1).all()
    # 只复位一个环境，其他环境的共享原点与积分历史不能跟着清空。
    saved=shared.pos[1:].clone();shared.reset(torch.tensor([0]),standing=True)
    assert not shared.position_reference_latched[0]
    torch.testing.assert_close(shared.pos[1:],saved,atol=0,rtol=0)
    shared.step(*inputs,position_reference=latched)
    assert shared.position_reference_event[0] and not shared.position_reference_event[1:].any()
    result=dict(passed=True,optimizer_updates=0,num_envs=count,default_samples=6000,
        max_default_torque_difference=maximum,default_position_trajectory_identical=True,
        before_anchor_position_feedback_zero=True,shared_first_event_zero_in_actual_feedback=True,
        each_environment_event_once=True,partial_reset_isolated=True,
        scope='Frozen-before Python default and direct shared-reference equations. Actual teacher records checked separately.')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);check(p.parse_args())
