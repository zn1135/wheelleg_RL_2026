"""Verify the four-joint x2 default, GPU load, provenance and finite short steps."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from ct_mjlab.envs.chuanliantui.model import MODEL_XML,load_model
from ct_mjlab.envs.fudan_ppo_walk.fudan_ppo_walk import FudanEnv
from ct_mjlab.fudan_ppo_walk.contract import RESET_V2_CONTRACT
from ct_mjlab.fudan_ppo_walk.measurement import configure_samples
from scripts.agent.fudan_ppo_walk.evaluate import evaluation_items

NAMES=('lf0','lf00','rf0','rf00')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--before-xml',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    before=ET.fromstring(args.before_xml.read_text());after=ET.fromstring(MODEL_XML.read_text());changed=[]
    for name in NAMES:
        a=before.find(f'.//joint[@name="{name}"]');b=after.find(f'.//joint[@name="{name}"]')
        assert float(a.attrib['armature'])==.015795 and float(b.attrib['armature'])==.03159
        changed.append(dict(joint=name,before=.015795,after=.03159));b.attrib['armature']=a.attrib['armature']
    assert ET.tostring(before)==ET.tostring(after),'Unrequested XML semantic change'
    native=mujoco.MjModel.from_xml_path(str(MODEL_XML));compiled=load_model()
    for model in (native,compiled):
        for name in NAMES:assert abs(float(model.dof_armature[int(model.joint(name).dofadr[0])])-.03159)<1e-12
    env=FudanEnv(20,seed=30001,randomized=True,reset_contract=RESET_V2_CONTRACT)
    try:
        items=evaluation_items(True,None);configure_samples(env,items,args.output/'fixtures')
        indices=env.action._motor_v_adr[[0,1,3,4]]
        actual=env.env.sim.model.dof_armature[:,indices]
        assert torch.allclose(actual,torch.full_like(actual,.03159),atol=1e-8,rtol=0)
        for item in items:
            with np.load(args.output/'fixtures'/(item['id']+'_initial.npz')) as fixture:
                assert 'dof_armature' in fixture.files
                assert np.allclose(fixture['dof_armature'][indices.cpu().numpy()],.03159,atol=1e-8,rtol=0)
        # 短程有限性检查不替代策略行为评分。
        for _ in range(100):
            env.step(torch.zeros(20,6,device=env.device),auto_reset=True,resample=False)
            assert torch.isfinite(env.env.sim.data.qpos).all() and torch.isfinite(env.env.sim.data.qvel).all()
        report=dict(passed=True,xml_only_four_armatures_changed=True,joints=changed,
            xml_sha256=hashlib.sha256(MODEL_XML.read_bytes()).hexdigest(),native_and_training_loader_verified=True,
            gpu_actual_armature=actual.cpu().tolist(),num_envs=20,physics_steps=500,model_forwards=0,optimizer_updates=0,
            fixtures_record_armature=True,behavior_accepted=False)
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    finally:env.close()


if __name__=='__main__':main()
