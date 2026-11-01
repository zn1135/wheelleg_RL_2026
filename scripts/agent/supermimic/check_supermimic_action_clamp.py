"""Check action bounds, noisy collection, imagination and ONNX without training."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from ct_mjlab.supermimic.learner import Learner,ALGORITHM_CONTRACT
from ct_mjlab.supermimic.networks import act,clip_action,ACTION_CLIP
from ct_mjlab.supermimic.export import export
from scripts.agent.supermimic.check_supermimic_motion import synthetic_buffer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--integration',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(43)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    curriculum=copy.deepcopy(saved['curriculum'])
    command=SimpleNamespace(cfg=command_cfg('walk'),checkpoint_state=lambda:copy.deepcopy(curriculum),
        load_checkpoint_state=lambda x:curriculum.update(copy.deepcopy(x)))
    learner=Learner('cpu');learner.load(args.checkpoint,command)
    raw=torch.tensor([5.,-5.,10.,.5,-.5,-10.])
    with torch.no_grad():
        learner.actor.mlp[-1].weight.zero_();learner.actor.mlp[-1].bias.copy_(raw)
    h=torch.zeros(4,125);expected=raw.clamp(-3,3).expand(4,-1)
    assert torch.equal(act(learner.actor,h),expected)
    assert torch.equal(act(learner.actor,h,unclipped=True),raw.expand(4,-1))
    x=raw.clone().requires_grad_(True)
    grad=torch.autograd.grad(clip_action(learner.actor,x).sum(),x)[0]
    assert torch.equal(grad,torch.tensor([0.,0.,0.,1.,1.,0.]))
    invalid=torch.tensor([float('nan'),float('inf'),-float('inf')])
    assert not clip_action(learner.actor,invalid).isfinite().any()
    calls=[];step=learner.world.step
    def observe_step(*a,**kw):
        assert torch.equal(a[2],raw.clamp(-3,3).expand_as(a[2]))
        result=step(*a,**kw)
        # Last six observation entries carry exactly the action used by dynamics.
        assert torch.equal(result[1][...,-6:],a[2]);calls.append(True)
        return result
    with patch.object(learner.world,'step',side_effect=observe_step):
        rollout=learner.imagination(synthetic_buffer().sample(4,4))
    assert calls and torch.isfinite(rollout[0])
    candidate=args.output/'saturated-fixture.pt'
    learner.save(candidate,command,config=saved['config'],iteration=0,transitions=0,world_updates=0)
    bounded_export=export(candidate,args.output/'bounded.onnx')
    assert bounded_export['action_clip']==ACTION_CLIP
    import numpy as np
    vectors=np.load(args.output/'bounded_reference_vectors.npz')
    assert np.array_equal(vectors['actions'],raw.clamp(-3,3).numpy()[None].repeat(32,0))
    # Historical inference/export must reproduce historical unbounded actions.
    learner.load(args.legacy,command,inference=True)
    assert learner.actor.supermimic_action_clip is None
    assert torch.equal(act(learner.actor,h),act(learner.actor,h,unclipped=True))
    legacy_export=export(args.legacy,args.output/'legacy.onnx')
    assert legacy_export['action_clip'] is None
    try:learner.load(args.legacy,command)
    except ValueError:pass
    else:raise AssertionError('Legacy full resume accepted')
    learner.load(candidate,command)
    assert learner.actor.supermimic_action_clip==ACTION_CLIP
    assert learner.optimizer_update_counts()==dict(world=0,encoder=0,actor=0,critic=0)
    report=dict(passed=True,algorithm_contract=ALGORITHM_CONTRACT,action_clip=ACTION_CLIP,
        clipped_action_and_gradient=True,nonfinite_preserved=True,imagination_action_and_history_equal=True,
        new_save_restore=True,legacy_inference_preserved=True,legacy_full_resume_rejected=True,
        bounded_onnx_error=bounded_export['numeric_max_error'],legacy_onnx_error=legacy_export['numeric_max_error'],
        optimizer_updates=0,scope='Finite interface checks only; no behavior or hardware acceptance')
    if args.integration:
        from ct_mjlab.envs import task_registry
        from ct_mjlab.supermimic.__main__ import collect
        from ct_mjlab.supermimic.buffer import ReplayBuffer
        from scripts.agent.supermimic.record_supermimic_physical_prediction import ObservedWalkRewardState
        cfg,train=task_registry.get_cfgs('supermimic',play=True)
        env,_=task_registry.make_env('supermimic',env_cfg=cfg,train_cfg=train,num_envs=12,seed=43,device='cuda:0')
        try:
            env.unwrapped._split_walk_rewards=ObservedWalkRewardState(env.unwrapped)
            l=Learner('cuda:0');l.load(candidate,env.unwrapped.command_manager.get_term('standup'))
            b=ReplayBuffer(4,12,'cuda:0');obs=env.get_observations()
            noise=torch.tensor([-3.,3.,-20.,.1,-.1,20.],device='cuda:0').expand(12,-1)
            expected=(raw.to('cuda:0')+noise).clamp(-3,3)
            random=torch.randn_like;real_step=env.step;sent=[]
            def fixed_noise(x,*a,**kw):
                return noise.clone() if x.shape==noise.shape else random(x,*a,**kw)
            def observe_real(action):
                assert torch.equal(action,expected);sent.append(action.clone());return real_step(action)
            with patch.object(torch,'randn_like',side_effect=fixed_noise),patch.object(env,'step',side_effect=observe_real):
                obs,_,done=collect(env,l,b,obs,exploration=1.)
            assert len(sent)==1 and not done.any()
            assert torch.equal(b.data['action'][0],expected)
            assert torch.equal(obs['actor'][...,-6:],expected)
            assert torch.equal(b.data['next_frame'][0][...,-6:],expected)
            assert torch.equal(env.unwrapped._split_walk_rewards.record['action'],expected)
            # Nonfinite actions must stop before any simulator step, including Inf.
            with torch.no_grad():l.actor.mlp[-1].bias[0]=float('inf')
            with patch.object(env,'step',side_effect=AssertionError('Invalid action reached simulation')):
                try:collect(env,l,None,obs,exploration=0.)
                except FloatingPointError:pass
                else:raise AssertionError('Nonfinite action accepted')
            assert l.optimizer_update_counts()==dict(world=0,encoder=0,actor=0,critic=0)
            report['integration']=dict(real_transitions=12,physical_substeps=60,
                noise_before_clamp=True,simulation_replay_reward_history_equal=True,
                nonfinite_stopped_before_simulation=True,optimizer_updates=0)
        finally:env.close()
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No training allowed')):
        main()
