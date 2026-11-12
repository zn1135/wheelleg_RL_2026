"""Frozen-policy zero-command replay of the actual Fudan mechanical/legacy wheel plant."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk_config import FudanWalkCfg
from ct_mjlab.envs.fudan_ppo_walk.disturbances import from_sim2sim_config
from scripts.agent.fudan_ppo_walk import evaluate
from scripts.agent.real2sim.probe_video_leg_parameters import items,report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--profile',choices=('mechanical','legacy_curve'),required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    gain=1. if a.profile=='mechanical' else .45
    selected=items(profiles=[dict(name=a.profile,armature_scale=2.,wheel_motor_gain=gain)])
    for s in selected:
        h=s['case']['segments'][0]['command'][2];s['id']=f'hold_h{round(h*100)}'
        s['case']=dict(id=f'zero_hold_{h:g}',category='zero',duration=30.,segments=[dict(start=0.,duration=30.,command=[0.,0.,h])])
    clean=replace(from_sim2sim_config(),pushes=False,external_force_range=(0.,0.),external_torque_range=(0.,0.))
    original=evaluate.configure_samples;checks=[]
    def configure(env,batch,folder):
        obs=original(env,batch,folder);action=env.action;sim=env.env.sim;w=action._wheel_action_ids
        ids=[sim.mj_model.actuator('robot/'+name+'_motor').id for name in ('lfwheel','rfwheel')]
        old_write=env.env.scene.write_data_to_sim;error=torch.zeros((),device=env.device)
        def write_data():
            nonlocal error
            old_write();request=action._torques[:,w]
            expected=request
            if a.profile=='legacy_curve':
                parameters=action._wheel_response;speed=sim.data.qvel[:,action._motor_v_adr[w]]
                factor=(1-torch.clamp(torch.sign(request)*speed,min=0.)/parameters[:,1]).clamp(min=0.)**parameters[:,2]
                expected=request*gain*factor
            error=torch.maximum(error,abs(sim.data.ctrl[:,ids]-expected).max())
        env.env.scene.write_data_to_sim=write_data
        checks.append(lambda:float(error))
        (folder/'actual-wheel-model.json').write_text(json.dumps(dict(profile=env.wheel_physics_profile,
            leg_p=action.cfg.leg_stiffness,leg_d=action.cfg.leg_damping,wheel_p=action.cfg.wheel_stiffness,
            wheel_motor_gain=action.wheel_motor.gain.cpu().tolist(),speed_shaping=action._wheel_speed_shaping),indent=2)+'\n')
        return obs
    request=SimpleNamespace(checkpoint=a.checkpoint,output=a.output/'no_external',deadline=time.time()+300,diagnostic=True,
        diagnostic_height=None,batch_size=3,selection=None,leg_load_profile='legacy',response_parameters=None)
    with patch.object(FudanWalkCfg.control,'leg_stiffness',60.),patch.object(FudanWalkCfg.control,'leg_damping',2.),patch.object(FudanWalkCfg.control,'wheel_stiffness',.045),patch.object(FudanWalkCfg.sim,'wheel_model',a.profile),patch.object(FudanWalkCfg.sim,'wheel_motor_gain',gain),patch.object(evaluate,'evaluation_items',return_value=selected),patch.object(evaluate,'from_sim2sim_config',return_value=clean),patch.object(evaluate,'configure_samples',configure):evaluate.gpu(request)
    error=max(f() for f in checks);assert error<1e-5
    (a.output/'run.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint.resolve()),sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        leg_p=60.,leg_d=2.,wheel_p=.045,wheel_motor_gain=gain,wheel_model=a.profile,
        display_label='17679 / '+a.profile,actual_ctrl_max_error=error,policy_updates=0),indent=2)+'\n')
    report(a.output/'no_external')


if __name__=='__main__':main()
