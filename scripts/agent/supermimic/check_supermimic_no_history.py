"""Zero-optimizer checks for the129-input V13 dynamics and checkpoint boundary."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from ct_mjlab.supermimic.learner import Learner,ALGORITHM_CONTRACT,WORLD_LOSS_RECIPE,ALTERNATING_RECIPE
from ct_mjlab.supermimic.motion import WORLD_INPUT_DIM
from ct_mjlab.supermimic.state import NOISE,ACTION_LAGS
from scripts.agent.supermimic.check_supermimic_motion import fixture,synthetic_buffer,equal

ROOT=Path(__file__).resolve().parents[3]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261101_v13_continuation_to5200_r3/final.pt')
    parser.add_argument('--integration',action='store_true',help='Also collect16 frozen GPU steps in12 environments, without optimizer updates')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(123)
    source_sha=sha(args.parent);old=torch.load(args.parent,map_location='cpu',weights_only=False,mmap=True)
    curriculum=copy.deepcopy(old['curriculum'])
    command=SimpleNamespace(cfg=command_cfg('walk'),checkpoint_state=lambda:copy.deepcopy(curriculum),load_checkpoint_state=lambda x:curriculum.update(copy.deepcopy(x)))
    learner=Learner('cpu',actor_lr=3e-5)
    transfer=learner.init_actor(args.parent,command,task='chuanliantui_walk')
    assert equal(learner.actor.state_dict(),old['actor'])
    world=learner.world
    assert WORLD_INPUT_DIM==129 and world.net[0].in_features==129 and world.in_norm.mean.shape==(129,)
    assert world.delta_norm.mean.shape==(46,) and world.net[-1].out_features==46
    # Nonzero head prevents a zero-initialized network from trivially passing invariance.
    with torch.no_grad():world.net[-1].weight.normal_(std=1e-4)
    s,p,c=fixture(4);a=torch.randn(4,6)*.1
    previous=torch.randn(4,6,requires_grad=True);previous2=torch.randn(4,6,requires_grad=True)
    action=a.clone().requires_grad_(True)
    first=world.step(s,p,action,torch.zeros(4,3),c,previous,previous2,s[:,NOISE],return_path=True)
    second=world.step(s,p,action,torch.zeros(4,3),c,previous+10,previous2-20,s[:,NOISE],return_path=True)
    assert equal(first[-1],second[-1])
    gradients=torch.autograd.grad(first[-1].square().sum(),(action,previous,previous2),allow_unused=True)
    assert gradients[0] is not None and torch.isfinite(gradients[0]).all() and gradients[0].abs().max()>0
    assert gradients[1] is gradients[2] is None
    p[:,ACTION_LAGS]=5
    delayed=world.step(s,p,a,torch.zeros(4,3),c,previous,previous2,s[:,NOISE],return_path=True)
    delayed_other=world.step(s,p,a,torch.zeros(4,3),c,previous+.1,previous2-.1,s[:,NOISE],return_path=True)
    assert equal(delayed[-1],delayed_other[-1])
    # Endpoint reward uses the current action, without reconstructing delayed execution.
    assert torch.equal(delayed[4][...,22:28],delayed_other[4][...,22:28])
    buffer=synthetic_buffer();batch=buffer.sample(8,8);modified=copy.deepcopy(batch)
    modified['previous'].fill_(float('nan'));modified['previous_previous'].fill_(float('nan'))
    world.observe(batch);normal=copy.deepcopy(world.state_dict())
    world.load_state_dict(normal);world.observe(batch);expected=copy.deepcopy(world.state_dict())
    world.load_state_dict(normal);world.observe(modified)
    assert equal(world.state_dict(),expected)
    world.load_state_dict(normal)
    altered=copy.deepcopy(batch);altered['previous']+=3;altered['previous_previous']-=7
    loss,_=learner.world_loss(batch);other,_=learner.world_loss(altered)
    torch.testing.assert_close(loss,other,atol=0,rtol=0)
    loss.backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in world.parameters())
    rejected=[]
    def rejects(path):
        before={n:copy.deepcopy(getattr(learner,n).state_dict()) for n in ('actor','world','critic','opt_actor','opt_encoder','opt_world','opt_critic')}
        rng=torch.get_rng_state().clone()
        try:learner.load(path,command)
        except ValueError as e:rejected.append(str(e))
        else:raise AssertionError('Incompatible world checkpoint accepted')
        assert all(equal(before[n],getattr(learner,n).state_dict()) for n in before)
        assert torch.equal(rng,torch.get_rng_state())
    rejects(args.parent)
    spoof=copy.deepcopy(old);spoof['algorithm_contract']=ALGORITHM_CONTRACT
    spoof['config']['world_loss_recipe']=copy.deepcopy(WORLD_LOSS_RECIPE)
    spoof['config']['alternating_recipe']=copy.deepcopy(ALTERNATING_RECIPE)
    spoof_path=args.output/'incompatible-shape-fixture.pt';torch.save(spoof,spoof_path);rejects(spoof_path)
    config=copy.deepcopy(old['config']);config.update(world_loss_recipe=copy.deepcopy(WORLD_LOSS_RECIPE),alternating_recipe=copy.deepcopy(ALTERNATING_RECIPE),initialization=transfer)
    new_path=args.output/'zero-update-roundtrip.pt'
    learner.save(new_path,command,config=config,iteration=0,transitions=0,world_updates=0)
    restored=Learner('cpu',actor_lr=3e-5);restored.load(new_path,command,expected_config=config)
    assert all(equal(getattr(learner,n).state_dict(),getattr(restored,n).state_dict()) for n in ('actor','world','critic','opt_actor','opt_encoder','opt_world','opt_critic'))
    frozen=copy.deepcopy(learner.world.state_dict())
    learner.load(args.parent,command,inference=True)
    assert equal(learner.actor.state_dict(),old['actor']) and equal(learner.world.state_dict(),frozen)
    assert learner.optimizer_update_counts()==dict(world=0,encoder=0,actor=0,critic=0)
    assert sha(args.parent)==source_sha
    report=dict(passed=True,algorithm_contract=ALGORITHM_CONTRACT,world_input_dim=129,output_dim=46,
        history_inputs_removed=12,path_history_invariant_zero_and_nonzero_delay=True,
        path_history_gradient_absent=True,current_action_gradient_finite_nonzero=True,
        normalization_ignores_nan_history=True,world_loss_history_invariant=True,
        analytic_endpoint_PD_current_action_only=True,legacy_full_resume_rejected_before_mutation=True,
        spoofed_contract_wrong_shape_rejected_before_mutation=True,rejection_messages=rejected,
        legacy_actor_transfer_and_inference_bitwise_equal=True,new_checkpoint_roundtrip_equal=True,
        optimizer_updates=0,simulator_steps=0,parent_sha256=source_sha,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'mjlab_training/src/ct_mjlab/supermimic/motion.py',ROOT/'mjlab_training/src/ct_mjlab/supermimic/learner.py',Path(__file__)]},
        scope='CPU tensors/backward/save-restore checks only; no training, physical behavior or hardware acceptance')
    if args.integration:
        from ct_mjlab.envs import task_registry
        from ct_mjlab.supermimic.__main__ import collect
        from ct_mjlab.supermimic.buffer import ReplayBuffer
        from scripts.agent.supermimic.record_supermimic_physical_prediction import ObservedWalkRewardState
        cfg,train=task_registry.get_cfgs('supermimic',play=True)
        env,_=task_registry.make_env('supermimic',env_cfg=cfg,train_cfg=train,num_envs=12,seed=43,device='cuda:0')
        try:
            raw=env.unwrapped;raw._split_walk_rewards=ObservedWalkRewardState(raw)
            l=Learner('cuda:0',actor_lr=3e-5)
            l.init_actor(args.parent,raw.command_manager.get_term('standup'),task='chuanliantui_walk')
            actor_before=copy.deepcopy(l.actor.state_dict());b=ReplayBuffer(24,12,'cuda:0');obs=env.get_observations()
            for _ in range(16):obs,_,_=collect(env,l,b,obs,exploration=0.)
            sample=b.sample(8,8);l.world.observe(sample)
            assert l.world.inputs(sample['state'],sample['parameters'],sample['action']).shape==(8,8,129)
            loss,_=l.world_loss(sample);loss.backward()
            assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in l.world.parameters())
            assert equal(actor_before,l.actor.state_dict())
            assert l.optimizer_update_counts()==dict(world=0,encoder=0,actor=0,critic=0)
            report['integration']=dict(transitions=192,physics_substeps=960,world_input_dim=129,world_loss_finite=True,backward_finite=True,actor_frozen=True,optimizer_updates=0,scope='Finite interface/data/backward check; not behavior acceptance')
            report['simulator_steps']=960
        finally:env.close()
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No training allowed')):
        main()
