"""Native body-velocity and finite-difference checks of the direct response loss; zero updates."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from torch.func import functional_call

from ct_mjlab.envs.chuanliantui.model import load_model
from ct_mjlab.supermimic.motion import WorldModel,physical47_from_full64,delta46
from ct_mjlab.supermimic.networks import normalize_contact_inputs,RunningNorm
from ct_mjlab.supermimic.state import NOISE
from scripts.agent.supermimic.repair_v13_v5_response import prediction,link_vx,dataset
from scripts.agent.supermimic.diagnose_v13_v5_response import sha
from scripts.agent.supermimic.check_v13_world_repair import digest_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('checkpoint','recording','output'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    parent_sha=sha(args.checkpoint)
    parent=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    recording=torch.load(args.recording,map_location='cpu',weights_only=False)
    assert recording['checkpoint_hashes']['final']==parent_sha
    world=WorldModel().double();world.load_state_dict(parent['world']);before=digest_state(world.state_dict())
    model=load_model();data=mujoco.MjData(model);jac=np.zeros((3,model.nv));native_max=roundtrip_max=public_max=0.;native_count=0
    for s in recording['samples']:
        state=s['state'].double();parameters=s['parameters'].double()
        for key in ('2_-0.10','2_+0.10','5_-0.10','5_+0.10'):
            path=s['paths'][key].double();endpoint=path[:,-1]
            derived=world.adapter.reward_features(path)[:,0].detach().numpy()
            for env in range(len(endpoint)):
                data.qpos[:]=endpoint[env,:25].numpy();data.qvel[:]=endpoint[env,25:49].numpy()
                mujoco.mj_normalizeQuat(model,data.qpos);mujoco.mj_forward(model,data)
                mujoco.mj_jacBody(model,data,jac,None,1)
                measured=(data.xmat[1].reshape(3,3).T@(jac@data.qvel))[0]
                native_max=max(native_max,abs(float(measured)-derived[env]));native_count+=1
            increment=delta46(physical47_from_full64(state[:,:64]),physical47_from_full64(endpoint))
            with torch.no_grad():
                reconstructed=link_vx(world,state,increment)[:,0]
                roundtrip_max=max(roundtrip_max,float((reconstructed-torch.from_numpy(derived)).abs().max()))
                a=s['actions'][key].double();delta=prediction(world,state,parameters,a)
                direct=link_vx(world,state,delta)[:,0]
                stepped=world.step(state,parameters,a,s['context'][:,:3].double(),s['context'].double(),s['previous'].double(),s['previous_previous'].double(),state[:,NOISE],return_path=True)
                public=world.adapter.reward_features(stepped[-1])[:,0]
                public_max=max(public_max,float((direct-public).abs().max()))
    assert native_max<2e-6 and roundtrip_max<2e-6 and public_max<1e-10
    _,pairs,_=dataset(recording,'cpu')
    ix=torch.linspace(0,len(pairs[0])-1,32).long()
    s,parameters,plus,minus,_,width,target,reliable=[x[ix].double() if x.dtype!=torch.bool else x[ix] for x in pairs]
    scale=pairs[6][pairs[7]].double().square().mean().sqrt().clamp_min(1e-4)
    xplus=normalize_contact_inputs(world.inputs(s,parameters,plus),world.in_norm,47)
    xminus=normalize_contact_inputs(world.inputs(s,parameters,minus),world.in_norm,47)
    named=dict(world.net.named_parameters())
    def loss(overrides=None):
        tensors=named if overrides is None else dict(named,**overrides)
        dp=functional_call(world.net,tensors,(xplus,))*world.delta_norm.std()+world.delta_norm.mean
        dm=functional_call(world.net,tensors,(xminus,))*world.delta_norm.std()+world.delta_norm.mean
        effect=(link_vx(world,s,dp)-link_vx(world,s,dm))/width
        return (((effect-target)/scale).square()*reliable).sum()/reliable.sum()
    initial=loss();names=('0.weight','6.weight')
    grads=torch.autograd.grad(initial,[named[n] for n in names]);checks=[]
    for name,g in zip(names,grads):
        assert torch.isfinite(g).all() and g.abs().max()>0
        for index in g.abs().flatten().topk(3).indices.tolist():
            eps=1e-5
            positive=named[name].detach().clone();negative=positive.clone()
            positive.view(-1)[index]+=eps;negative.view(-1)[index]-=eps
            with torch.no_grad():fd=(loss({name:positive})-loss({name:negative}))/(2*eps)
            exact=g.flatten()[index]
            error=float((fd-exact).abs());relative=error/max(float(exact.abs()),1e-8)
            assert error<=1e-6+1e-3*float(exact.abs())
            checks.append(dict(parameter=name,index=index,autograd=float(exact),finite_difference=float(fd),absolute_error=error,relative_error=relative))
    assert before==digest_state(world.state_dict()) and sha(args.checkpoint)==parent_sha
    report=dict(passed=True,native_body=model.body(1).name,native_endpoint_checks=native_count,
        native_LINK_vx_max_error_m_s=native_max,true_endpoint_roundtrip_max_error_m_s=roundtrip_max,
        direct_vs_public_step_max_error_m_s=public_max,gradient_checks=checks,precision='float64 for finite differences',
        optimizer_updates=0,model_unchanged=True,checkpoint_sha256=parent_sha,recording_sha256=sha(args.recording),script_sha256=sha(Path(__file__)),
        scope='Formula/native kinematics/derivative checks only; does not establish training improvement or control recovery')
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Zero updates only')),patch.object(RunningNorm,'update',side_effect=AssertionError('Frozen normalization')):
        main()
