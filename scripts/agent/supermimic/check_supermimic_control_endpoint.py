"""Verify one10ms prediction/target and direct endpoint observations; no training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from ct_mjlab.supermimic.learner import Learner,ALGORITHM_CONTRACT,WORLD_LOSS_RECIPE,ALTERNATING_RECIPE
from ct_mjlab.supermimic.motion import WorldModel,delta46,integrate47,physical47_from_full64,expand_motion47_to_full64
from ct_mjlab.supermimic.networks import MotionCritic,normalize_contact_inputs
from ct_mjlab.supermimic.state import SENSORS,SENSOR_LAGS,NOISE
from scripts.agent.supermimic.check_supermimic_motion import fixture,synthetic_buffer,equal

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(301)
    learner=Learner('cpu');world=learner.world
    assert world.net[-1].out_features==46 and world.in_norm.mean.numel()==129
    with torch.no_grad():world.net[-1].weight.normal_(std=1e-4)
    s,p,c=fixture(4);a=torch.randn(4,6)*.05
    p[:,SENSOR_LAGS]=8;s[:,SENSORS]=torch.randn_like(s[:,SENSORS])*10
    result=world.step(s,p,a,torch.zeros(4,3),c,a*.5,a*.2,s[:,NOISE],return_path=True)
    ns,frame,reward,failure,features,path=result
    assert path.shape==(4,1,64)
    raw=world.net(normalize_contact_inputs(world.inputs(s,p,a),world.in_norm,47))
    expected=expand_motion47_to_full64(integrate47(physical47_from_full64(s[:,:64]),raw*world.delta_norm.std()+world.delta_norm.mean))
    torch.testing.assert_close(path[:,0],expected,atol=0,rtol=0)
    torch.testing.assert_close(ns[:,:64],expected,atol=0,rtol=0)
    assert torch.count_nonzero(ns[:,SENSORS])==0
    torch.testing.assert_close(features[:,9],expected[:,2],atol=0,rtol=0)
    other_state=ns.clone();other_state[:,SENSORS]=float('nan')
    zero_lag=p.clone();zero_lag[:,SENSOR_LAGS]=0
    assert equal(frame,world.adapter.actor_frame(other_state,zero_lag,torch.zeros(4,3),a))
    assert MotionCritic.STATE_WIDTH==66 and learner.critic.norm.mean.numel()==162
    assert equal(learner.critic.inputs(ns,p,c),learner.critic.inputs(other_state,p,c))
    torch.testing.assert_close(learner.critic(ns,p,c),learner.critic(other_state,p,c),atol=0,rtol=0)
    b=synthetic_buffer();batch=b.sample(8,8);altered=copy.deepcopy(batch)
    real_path=altered['path'].reshape(8,8,5,64);real_path[:,:,:4]=float('nan')
    altered['path']=real_path.flatten(-2)
    fresh=WorldModel();fresh.observe(batch);norm=copy.deepcopy(fresh.state_dict())
    second=WorldModel();second.observe(altered)
    for k in ('in_norm.mean','in_norm.var','in_norm.count','delta_norm.mean','delta_norm.var','delta_norm.count'):
        assert equal(norm[k],second.state_dict()[k])
    count=int(batch['mask'].bool().sum());assert int(fresh.in_norm.count)==int(fresh.delta_norm.count)==count
    endpoint=physical47_from_full64(batch['path'].reshape(8,8,5,64)[:,:,-1])
    before=physical47_from_full64(batch['state'][...,:64])
    target=delta46(before,endpoint)[batch['mask'].bool()]
    torch.testing.assert_close(fresh.delta_norm.mean,target.mean(0),atol=1e-7,rtol=1e-6)
    first_loss,first_info=learner.world_loss(batch);other_loss,other_info=learner.world_loss(altered)
    torch.testing.assert_close(first_loss,other_loss,atol=0,rtol=0)
    assert first_info['valid_world_positions']==other_info['valid_world_positions']
    damaged=copy.deepcopy(batch);damaged['path']=damaged['path'].reshape(8,8,5,64)
    damaged['path'][:,:,-1,2]+=.1;damaged['path']=damaged['path'].flatten(-2)
    damaged_loss,_=learner.world_loss(damaged);assert not torch.equal(damaged_loss,first_loss)
    # Check endpoint-only backpropagation through multiple imagined policy steps too.
    first_loss.backward();assert all(x.grad is None or torch.isfinite(x.grad).all() for x in world.parameters())
    for module in (learner.world,learner.actor,learner.critic):module.zero_grad(set_to_none=True)
    imagined=learner.imagination(batch);imagined[0].backward()
    assert torch.isfinite(imagined[0]) and all(x.grad is None or torch.isfinite(x.grad).all() for x in learner.actor.parameters())
    assert learner.optimizer_update_counts()==dict(world=0,encoder=0,actor=0,critic=0)
    report=dict(passed=True,algorithm_contract=ALGORITHM_CONTRACT,world_loss_recipe=WORLD_LOSS_RECIPE,alternating_recipe=ALTERNATING_RECIPE,
        world_input_dim=129,world_output_dim=46,predicted_endpoint_count=1,physics_recorded_substeps=5,
        one_increment_endpoint_exact=True,no_intermediate_interpolation_or_repeat=True,
        normalization_uses_control_interval_delta=True,norm_samples_per_transition=1,
        intermediate_truth_nan_does_not_affect_loss_or_norm=True,endpoint_truth_change_changes_loss=True,
        observations_and_reward_features_from_endpoint=True,imagined_sensor_queue_zero_and_unused=True,
        sensor_lag_parameters_do_not_delay_imagined_observation=True,critic_input_dim=162,
        world_and_imagination_backward_finite=True,optimizer_updates=0,simulator_steps=0,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'mjlab_training/src/ct_mjlab/supermimic'/n for n in ('motion.py','motion_adapter.py','learner.py','networks.py')]+[Path(__file__)]},
        scope='CPU synthetic endpoint/target/gradient checks only; does not validate real delayed observation matching or behavior')
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No training')):main()
