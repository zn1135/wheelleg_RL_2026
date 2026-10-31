#!/usr/bin/env python3
"""Run the getup curriculum checks independently of concurrent walking changes."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.agent.split.check_chuanliantui_two_policy import role_check, rejected, digest
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from ct_mjlab.envs.chuanliantui_split.runner import validate
from ct_mjlab.scripts.play import play


def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    calls=[]

    class FiniteViewer:
        def __init__(self,env,policy,*a,**kw):self.env,self.policy=env,policy
        def run(self):
            state=self.env.unwrapped.command_manager.get_term('standup')
            height=.22 if state.unlocked else .30
            obs=self.env.get_observations()
            with torch.inference_mode():
                for _ in range(32):
                    obs,reward,_,_=self.env.step(self.policy(obs))
                    assert (state.command[:,2]==height).all()
                    assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
            calls.append(dict(height_m=height,steps=32))

    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Check cannot train')):
        row=role_check('getup',args)
        old=torch.load(args.old_getup_checkpoint,map_location='cpu',weights_only=False)
        rejected(lambda:validate(old['infos'],command_cfg('getup')))
        for checkpoint in ('interface-only-getup-locked.pt','interface-only-getup.pt'):
            ns=SimpleNamespace(task='chuanliantui_getup',checkpoint=args.output/checkpoint,
                seed=43,num_envs=2,device=args.device,smp_prior=None,knockdown_every=None,
                viewer='native',viser_port=None)
            with patch('mjlab.viewer.NativeMujocoViewer',FiniteViewer):play(ns)
        assert calls==[dict(height_m=.30,steps=32),dict(height_m=.22,steps=32)]
        from mj_sim2sim_chuanliantui_two_policy import OnnxPolicy
        # CPU尚未实现新滤波器，不能把未经滤波的执行冒充同一控制配方。
        rejected(lambda:OnnxPolicy(args.output/'interface-only-getup.pt',
            args.output/'interface-only-getup.onnx','getup'))
    report=dict(verified=True,parameter_updates=0,getup=row,play=calls,
        actual_old_getup_v1_rejected=True,cpu_unadapted_smoothed_getup_rejected=True,
        source_sha256=digest(__file__),shared_checker_sha256=digest(Path(__file__).with_name('check_chuanliantui_two_policy.py')),
        scope='Getup-only finite physics/curriculum thresholds/optimizer resume/ONNX/play and CPU load guards; no walking or behavior acceptance')
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-checkpoint',type=Path,required=True)
    parser.add_argument('--old-getup-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=20)
    parser.add_argument('--steps',type=int,default=256)
    parser.add_argument('--device',default='cuda:0')
    main(parser.parse_args())
