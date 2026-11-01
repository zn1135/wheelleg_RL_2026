"""Verify imagined-state guards without clipping world-fit labels or simulator state."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from mjlab.utils.lab_api.math import quat_apply
from ct_mjlab.supermimic.learner import Learner,ALGORITHM_CONTRACT,ALTERNATING_RECIPE
from ct_mjlab.supermimic.motion_adapter import IMAGINATION_CLAMP_RECIPE
from ct_mjlab.supermimic.state import NOISE
from scripts.agent.supermimic.check_supermimic_motion import fixture,synthetic_buffer,equal
from scripts.agent.supermimic.check_supermimic_heading_inputs import transform
from ct_mjlab.supermimic.state import rotation_quat

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(41)
    learner=Learner('cpu');world=learner.world;adapter=world.adapter;s,params,context=fixture(2)
    physical=s[:,:64].clone();snapshot=physical.clone()
    assert torch.equal(adapter.sanitize_endpoint(physical),physical)
    bad=physical.clone();bad[:,2]=torch.tensor([-1.,3.]);bad[:,25:28]=torch.tensor([100.,-100.,30.]);bad[:,28:31]=torch.tensor([100.,-100.,30.])
    bad[:,7:25]=1000.;bad[:,31:49]=1000.
    original=bad.clone();bounded=adapter.sanitize_endpoint(bad)
    assert torch.equal(bad,original) and torch.equal(physical,snapshot)
    torch.testing.assert_close(bounded[:,2],torch.tensor([.02,2.]),atol=0,rtol=0)
    assert bounded[:,25:28].abs().max()<=20 and bounded[:,28:31].abs().max()<=40
    mask=adapter.imagination_q_limited
    assert torch.equal(bounded[:,7:25][:,mask],adapter.imagination_q_upper[mask].expand(2,-1))
    assert torch.equal(bounded[:,7:25][:,~mask],bad[:,7:25][:,~mask])
    speeds=adapter.imagination_joint_speed_limit>0
    assert speeds.sum()==2 and (bounded[:,31:49][:,speeds]==150).all()
    assert torch.equal(bounded[:,31:49][:,~speeds],bad[:,31:49][:,~speeds])
    inside=physical.clone().requires_grad_(True);grad=torch.autograd.grad(adapter.sanitize_endpoint(inside).sum(),inside)[0]
    torch.testing.assert_close(grad,torch.ones_like(grad),atol=0,rtol=0)
    outside=bad.clone().requires_grad_(True);gradient=torch.autograd.grad(adapter.sanitize_endpoint(outside).sum(),outside)[0]
    assert torch.isfinite(gradient).all() and (gradient[:,2]==0).all()
    assert (gradient[:,7:25][:,mask]==0).all() and (gradient[:,31:49][:,speeds]==0).all()
    invalid=physical.clone();invalid[0,2]=float('nan');invalid[1,33]=float('inf')
    preserved=adapter.sanitize_endpoint(invalid)
    assert torch.isnan(preserved[0,2]) and torch.isinf(preserved[1,33])
    assert adapter.hard_failure(preserved).all()
    yaw=rotation_quat(torch.tensor([[0.,0.,1.2],[0.,0.,-2.1]]));offset=torch.tensor([[2.,3.,0.],[-1.,2.,0.]])
    shifted=transform(bad,yaw,offset)
    expected=transform(bounded,yaw,offset)
    torch.testing.assert_close(adapter.sanitize_endpoint(shifted),expected,atol=1e-5,rtol=1e-5)
    with torch.no_grad():world.net[-1].weight.zero_();world.net[-1].bias.zero_();world.net[-1].bias[2]=3.
    action=torch.zeros(2,6)
    raw=world.step(s,params,action,torch.zeros(2,3),context,action,action,s[:,NOISE],return_path=True)
    imagined=world.step(s,params,action,torch.zeros(2,3),context,action,action,s[:,NOISE],return_path=True,imagination=True)
    assert (raw[-1][:,0,2]>2).all() and (imagined[-1][:,0,2]==2).all()
    manual=adapter.advance(s,params,adapter.sanitize_endpoint(raw[-1]),torch.zeros(2,3),action,action,action,context,s[:,NOISE])
    assert all(equal(a,b) for a,b in zip(manual,imagined[:-1]))
    sample=synthetic_buffer().sample(4,4);labels=copy.deepcopy(sample)
    with patch.object(adapter,'sanitize_endpoint',side_effect=AssertionError('World fitting must not sanitize')):
        world.observe(sample);loss,_=learner.world_loss(sample);loss.backward()
    assert equal(sample,labels) and torch.isfinite(loss)
    assert world.net[-1].bias.grad[2].abs()>0
    calls=[];method=world.step
    def observe_call(*a,**kw):
        calls.append(kw.get('imagination',False));return method(*a,**kw)
    with patch.object(world,'step',side_effect=observe_call):
        rollout=learner.imagination(sample)
    assert calls and all(calls)
    assert torch.isfinite(rollout[0])
    assert learner.optimizer_update_counts()==dict(world=0,encoder=0,actor=0,critic=0)
    report=dict(passed=True,algorithm_contract=ALGORITHM_CONTRACT,recipe=IMAGINATION_CLAMP_RECIPE,
        valid_states_bitwise_unchanged=True,height_and_body_velocity_bounds=True,XML_limited_joint_bounds=True,
        unlimited_joint_positions_preserved=True,wheel_speed_bound150=True,unverified_joint_speed_bounds_not_invented=True,
        in_range_gradient_identity=True,clipped_gradient_zero=True,nonfinite_not_hidden=True,
        planar_transform_consistent=True,world_default_raw_prediction_unclipped=True,
        imagined_observation_reward_and_survival_use_clipped_endpoint=True,
        world_fit_and_norm_never_call_sanitize=True,real_replay_tensors_unchanged=True,
        learner_imagination_enables_clamp_each_step=True,optimizer_updates=0,simulator_steps=0,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'mjlab_training/src/ct_mjlab/supermimic'/n for n in ('motion.py','motion_adapter.py','learner.py')]+[Path(__file__)]},
        scope='CPU bound/gradient/call-path checks only; no training or behavior acceptance')
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No training')):main()
