"""Budgeted world-only response repair; produces a candidate, never changes its parent."""
import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from ct_mjlab.supermimic.learner import Learner,apply_gradient
from ct_mjlab.supermimic.motion import physical47_from_full64,delta46,integrate47,expand_motion47_to_full64
from ct_mjlab.supermimic.networks import RunningNorm,normalize_contact_inputs
from ct_mjlab.supermimic.state import NOISE
from scripts.agent.supermimic.diagnose_v13_v5_response import outputs,sha
from scripts.agent.supermimic.check_v13_world_repair import digest_state


def prediction(world,state,parameters,action):
    x=world.inputs(state,parameters,action)
    return world.net(normalize_contact_inputs(x,world.in_norm,47))*world.delta_norm.std()+world.delta_norm.mean


def link_vx(world,state,increment):
    # Use the same body-frame velocity projection as the measured base_link.
    endpoint=expand_motion47_to_full64(integrate47(physical47_from_full64(state[:,:64]),increment))
    return world.adapter.reward_features(endpoint.unsqueeze(-2))[:,:1]


def dataset(recording,device):
    points=[];pairs=[];anchors=[]
    for sample in recording['samples']:
        state=sample['state'];p=sample['parameters'];before=physical47_from_full64(state[:,:64])
        ids=torch.arange(len(state))%10!=0
        for key,action in sample['actions'].items():
            if key=='repeat':continue
            target=delta46(before,physical47_from_full64(sample['paths'][key][:,-1]))
            points.append((state[ids],p[ids],action[ids],target[ids]))
        for wheel in (2,5):
            for eps in (.05,.1):
                plus,minus=f'{wheel}_{eps:+.2f}',f'{wheel}_{-eps:+.2f}'
                good=ids & (sample['action'][:,wheel].abs()+eps<3)
                dp=delta46(before,physical47_from_full64(sample['paths'][plus][:,-1]))
                dm=delta46(before,physical47_from_full64(sample['paths'][minus][:,-1]))
                real_vx=sample['truth'][plus][:,0]-sample['truth'][minus][:,0]
                reliable=real_vx.abs()>torch.maximum(5*sample['noise'][:,0],torch.tensor(1e-5))
                pairs.append((state[good],p[good],sample['actions'][plus][good],sample['actions'][minus][good],
                    (dp-dm)[good]/(2*eps),torch.full((int(good.sum()),1),2*eps),
                    real_vx[good,None]/(2*eps),reliable[good,None]))
        if sample['anchor_windows'] is not None:
            anchors.append({k:v[:,:8].to(device) for k,v in sample['anchor_windows'].items()})
    return ([torch.cat([x[i] for x in points]).to(device) for i in range(4)],
        [torch.cat([x[i] for x in pairs]).to(device) for i in range(8)],anchors)


@torch.no_grad()
def score(learner,recording,*,heldout_only):
    scale=learner.world.delta_norm.std();errors=[];response_errors=[];rollout_errors=[];correct=opposite=eligible_count=weak=0
    world=learner.world
    for s in recording['samples']:
        ids=(torch.arange(len(s['state']))%10==0) if heldout_only else torch.ones(len(s['state']),dtype=torch.bool)
        state=s['state'][ids].to(scale.device);params=s['parameters'][ids].to(scale.device)
        before=physical47_from_full64(state[:,:64]);pred={}
        for key in ('base','2_-0.10','2_+0.10','5_-0.10','5_+0.10'):
            a=s['actions'][key][ids].to(scale.device)
            delta=prediction(world,state,params,a)
            target=delta46(before,physical47_from_full64(s['paths'][key][ids,-1].to(scale.device)))
            errors.append(((delta-target)/scale).square().mean())
            # Use the public step for the measured LINK velocity rather than replacing it by root vx.
            result=world.step(state,params,a,s['context'][ids,:3].to(scale.device),s['context'][ids].to(scale.device),
                s['previous'][ids].to(scale.device),s['previous_previous'][ids].to(scale.device),state[:,NOISE],return_path=True)
            pred[key]=outputs(world,result[-1]).cpu()
        for wheel in (2,5):
            plus,minus=f'{wheel}_+0.10',f'{wheel}_-0.10'
            real=s['truth'][plus][ids,0]-s['truth'][minus][ids,0]
            estimate=pred[plus][:,0]-pred[minus][:,0]
            eligible=(real.abs()>torch.maximum(5*s['noise'][ids,0],torch.tensor(1e-5)))&(s['action'][ids,wheel].abs()+.1<3)
            observable=estimate.abs()>1e-5
            eligible_count+=int(eligible.sum());correct+=int((eligible&observable&(real*estimate>0)).sum())
            opposite+=int((eligible&observable&(real*estimate<0)).sum());weak+=int((eligible&~observable).sum())
            response_errors.append(((estimate[eligible]-real[eligible])/.2).square())
        if not heldout_only and s['anchor_windows'] is not None:
            w={k:v[:,:8].to(scale.device) for k,v in s['anchor_windows'].items()}
            rollout_errors.append(learner.world_loss(w)[0])
    response=torch.cat(response_errors)
    return dict(endpoint_normalized_mse=float(torch.stack(errors).mean()),
        vx_response_slope_mse=float(response.mean()),eligible=eligible_count,correct=correct,opposite=opposite,weak=weak,
        rollout8_normalized_mse=float(torch.stack(rollout_errors).mean()) if rollout_errors else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','training','validation','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--updates',type=int,required=True);p.add_argument('--seconds',type=float,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.updates<1 or not math.isfinite(args.seconds) or args.seconds<=0:p.error('Explicit positive update/time budgets required')
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic();torch.set_num_threads(2)
    original=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    training=torch.load(args.training,map_location='cpu',weights_only=False)
    validation=torch.load(args.validation,map_location='cpu',weights_only=False)
    parent_sha=sha(args.checkpoint)
    assert training['seed']!=validation['seed'] and training['checkpoint_hashes']['final']==validation['checkpoint_hashes']['final']==parent_sha
    learner=Learner(args.device,actor_lr=original['config']['actor_lr'])
    learner.load(args.checkpoint,SimpleNamespace(cfg=command_cfg('walk'),load_checkpoint_state=lambda x:None))
    learner.actor.requires_grad_(False);learner.critic.requires_grad_(False)
    frozen={name:digest_state(getattr(learner,name).state_dict()) for name in ('actor','critic','opt_actor','opt_encoder','opt_critic')}
    buffers={k:v.clone() for k,v in learner.world.state_dict().items() if not k.startswith('net.')}
    points,pairs,anchors=dataset(training,args.device);assert anchors
    vx_scale=pairs[6][pairs[7]].square().mean().sqrt().clamp_min(1e-4)
    before={'heldout':score(learner,training,heldout_only=True),'independent':score(learner,validation,heldout_only=False)}
    initial_counts=learner.optimizer_update_counts();steps=0
    recipe=dict(parent_checkpoint=str(args.checkpoint.resolve()),parent_sha256=parent_sha,
        training_sha256=sha(args.training),validation_sha256=sha(args.validation),training_seed=training['seed'],validation_seed=validation['seed'],
        training_envs='env_id % 10 != 0',heldout_envs='env_id % 10 == 0',validation_used_for_fitting=False,
        version='direct_LINK_vx_response_v2',
        losses=['real_endpoint_delta46_MSE','real_plus_minus_delta46_action_slope_MSE','original_real_replay_window8_loss','real_LINK_vx_action_slope_MSE'],
        weights=[1,1,1,1],link_vx_slope_training_rms=float(vx_scale),normalization='parent_fixed',requested_updates=args.updates,seconds=args.seconds,
        frozen_components=list(frozen),no_synthetic_history_labels=True,script_sha256=sha(Path(__file__)))
    (args.output/'recipe.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2)+'\n')
    writer=None
    if not args.check_only:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(args.output/'tensorboard'))
    failure=None
    try:
        with patch.object(RunningNorm,'update',side_effect=AssertionError('Norms must remain frozen')),(args.output/'fit.jsonl').open('x') as log:
            for step in range(1 if args.check_only else args.updates):
                if time.monotonic()-started>=args.seconds:break
                ix=torch.randint(len(points[0]),(128,),device=args.device)
                state,params,a,target=[x[ix] for x in points]
                point=((prediction(learner.world,state,params,a)-target)/learner.world.delta_norm.std()).square().mean()
                ix=torch.randint(len(pairs[0]),(128,),device=args.device)
                state,params,plus,minus,target,width,target_vx,reliable=[x[ix] for x in pairs]
                predicted_plus=prediction(learner.world,state,params,plus)
                predicted_minus=prediction(learner.world,state,params,minus)
                slope=(predicted_plus-predicted_minus)/width
                response=((slope-target)/learner.world.delta_norm.std()).square().mean()
                measured_slope=(link_vx(learner.world,state,predicted_plus)-link_vx(learner.world,state,predicted_minus))/width
                vx_response=(((measured_slope-target_vx)/vx_scale).square()*reliable).sum()/reliable.sum().clamp_min(1)
                anchor,_=learner.world_loss(anchors[step%len(anchors)])
                loss=point+response+anchor+vx_response
                if args.check_only:
                    vx_gradients=torch.autograd.grad(vx_response,list(learner.world.parameters()),retain_graph=True)
                    assert all(torch.isfinite(g).all() for g in vx_gradients)
                    assert any(g.abs().max()>0 for g in vx_gradients)
                    gradients=torch.autograd.grad(loss,list(learner.world.parameters()))
                    assert all(torch.isfinite(g).all() for g in gradients)
                    assert any(g.abs().max()>0 for g in gradients)
                    (args.output/'gradient-check.json').write_text(json.dumps(dict(finite=True,nonzero=True,direct_LINK_vx_gradient_nonzero=True,optimizer_updates=0))+'\n')
                    break
                apply_gradient(loss,learner.opt_world,learner.world.parameters());steps+=1
                log.write(json.dumps(dict(step=steps,loss=float(loss.detach()),point=float(point.detach()),response=float(response.detach()),anchor=float(anchor.detach()),vx_response=float(vx_response.detach())))+'\n');log.flush()
                for name,value in (('loss',loss),('point',point),('response',response),('anchor',anchor),('vx_response',vx_response)):
                    writer.add_scalar(name,float(value.detach()),steps)
    except Exception as error:
        failure=repr(error)
    finally:
        if writer is not None:writer.close()
        assert all(digest_state(getattr(learner,n).state_dict())==h for n,h in frozen.items())
        assert all(torch.equal(v,learner.world.state_dict()[k]) for k,v in buffers.items())
        assert sha(args.checkpoint)==parent_sha
        counts=learner.optimizer_update_counts();assert counts['world']==initial_counts['world']+steps
        assert all(counts[k]==initial_counts[k] for k in ('actor','encoder','critic'))
        candidate=dict(schema='v13-v5-world-response-repair-candidate-v1',parent_checkpoint=str(args.checkpoint.resolve()),
            parent_sha256=parent_sha,world=learner.world.state_dict(),opt_world=learner.opt_world.state_dict(),
            optimizer_update_counts=counts,additional_world_updates=steps,frozen_component_hashes=frozen,recipe=recipe)
        torch.save(candidate,args.output/'candidate.pt')
    after={'heldout':score(learner,training,heldout_only=True),'independent':score(learner,validation,heldout_only=False)}
    gates={}
    for split in before:
        gates[split+'_response_improved']=after[split]['vx_response_slope_mse']<before[split]['vx_response_slope_mse']
        gates[split+'_endpoint_nonregressed']=after[split]['endpoint_normalized_mse']<=before[split]['endpoint_normalized_mse']
        gates[split+'_direction_nonregressed']=after[split]['correct']>=before[split]['correct'] and after[split]['opposite']<=before[split]['opposite']
        if before[split]['rollout8_normalized_mse'] is not None:
            gates[split+'_rollout8_nonregressed']=after[split]['rollout8_normalized_mse']<=before[split]['rollout8_normalized_mse']
    result=dict(before=before,after=after,gates=gates,accepted=all(gates.values()) and failure is None and steps==args.updates,
        actual_world_updates=steps,optimizer_update_counts=counts,error=failure,elapsed_s=time.monotonic()-started,
        candidate_sha256=sha(args.output/'candidate.pt'),actor_unchanged=True,norms_unchanged=True,
        scope='World-only candidate; frozen actor means real control behavior is unchanged. Not a deployable/full-resume checkpoint.')
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
    if failure:raise RuntimeError(failure)


if __name__=='__main__':main()
