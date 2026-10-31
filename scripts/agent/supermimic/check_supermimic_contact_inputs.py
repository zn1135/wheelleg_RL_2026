#!/usr/bin/env python3
"""V11 CPU contact encoding, fixed inputs and real V9/V10 actor compatibility; zero updates."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.agent.supermimic.check_supermimic_bptt import fixture
from ct_mjlab.supermimic import __main__ as entry
from ct_mjlab.supermimic.learner import ALGORITHM_CONTRACT, Learner, asset_digest
from ct_mjlab.supermimic.networks import (WorldModel, Critic, act, contact_parameters,
    normalize_contact_inputs)
from ct_mjlab.supermimic.state import PHYS_DIM, STATE_DIM, PARAM_DIM, SOLREF, NOISE
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg

V9='chuanliantui-supermimic-canonical-physics-path-derived-observation-reward-termination-v9'
V10='chuanliantui-supermimic-canonical-physics-contact-kb-fixed-scale-nominal-mix-v10'
V11='chuanliantui-supermimic-canonical-physics-fixed-support-scale-robust-wheel-angle-v11'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(module):
    result=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        result.update(name.encode());result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode());result.update(value.cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def coefficients(pair):
    # Independent scalar reference: Warp constraint.py's k/b before common dmax factors.
    tau,zeta=map(float,pair);safe_tau=max(tau,2*.002)
    stiffness=1/(safe_tau*safe_tau*zeta*zeta) if tau>0 else -tau
    damping=2/safe_tau if zeta>0 else -zeta
    return stiffness,damping


def encoding_check():
    pairs=[(.02,1.),(-2500.,-100.),(.001,1.),(.004,1.),(.008,.5),
           (.04,2.),(-10000.,-18.238),(-10000.,-37.389),(-5000.,-100.),(-2500.,-50.)]
    p=torch.arange(len(pairs)*PARAM_DIM,dtype=torch.float64).reshape(len(pairs),PARAM_DIM)*.001
    p[:,SOLREF]=torch.tensor(pairs,dtype=p.dtype);before=p.clone()
    encoded=contact_parameters(p)
    expected=torch.tensor([coefficients(pair) for pair in pairs],dtype=p.dtype)/torch.tensor([10000.,100.])
    torch.testing.assert_close(encoded[:,SOLREF],expected,atol=1e-12,rtol=1e-12)
    assert torch.equal(encoded[:,:SOLREF.start],p[:,:SOLREF.start])
    assert torch.equal(encoded[:,SOLREF.stop:],p[:,SOLREF.stop:]) and torch.equal(p,before)
    torch.testing.assert_close(encoded[0,SOLREF],encoded[1,SOLREF],atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(encoded[2,SOLREF],encoded[3,SOLREF],atol=0,rtol=0)
    assert encoded[0,SOLREF][0]!=encoded[8,SOLREF][0] and encoded[0,SOLREF][1]!=encoded[9,SOLREF][1]
    # Supported formats away from the REFSAFE kink must retain differentiable inputs.
    for pair in ((.02,.7),(-2500.,-100.),(.001,.8)):
        raw=torch.tensor(pair,dtype=torch.float64,requires_grad=True)
        def encode(value):
            params=torch.cat((torch.zeros(SOLREF.start,dtype=value.dtype),value,
                              torch.zeros(PARAM_DIM-SOLREF.stop,dtype=value.dtype)))
            return contact_parameters(params)[SOLREF]
        assert torch.autograd.gradcheck(encode,(raw,),eps=1e-6,atol=1e-5,rtol=1e-4)
    return dict(raw_solref_pairs=pairs,encoded_K_over_10000_B_over_100=encoded[:,SOLREF].tolist(),
        positive_direct_equivalent=True,refsafe_2ms_min_tau_004=True,different_materials_distinct=True,
        finite_difference_gradient_checked=True,raw_parameters_unchanged=True,
        interpretation='Reference material coefficients before common dmax factors; not contact force.')


def network_input_check():
    _,state,params,context=fixture(3);action=torch.zeros(3,6)
    reports=[]
    for kind,model,offset in (('world',WorldModel(),PHYS_DIM),('critic',Critic(),STATE_DIM)):
        norm=model.in_norm if kind=='world' else model.norm
        def values(p):
            return model.inputs(state,p,action,action,action) if kind=='world' else model.inputs(state,p,context)
        # Actual update statistics from a constant direct-format material exercise the variance floor.
        training=params.clone();training[:,SOLREF]=torch.tensor([-10000.,-28.])
        norm.update(values(training));baseline=state_digest(model)
        seen=[]
        hook=model.net[0].register_forward_pre_hook(lambda _module,args:seen.append(args[0]))
        raw=params.clone().requires_grad_()
        try:
            if kind=='world':
                output=model.step(state,raw,action,torch.zeros(3,3),context,action,action,state[:,NOISE])
                assert all(torch.isfinite(value).all() for value in output if value.dtype!=torch.bool)
            else:assert torch.isfinite(model(state,raw,context)).all()
        finally:hook.remove()
        assert len(seen)==1
        inputs=values(raw);lo,hi=offset+SOLREF.start,offset+SOLREF.stop
        expected=inputs[:,lo:hi]
        torch.testing.assert_close(seen[0][:,lo:hi],expected,atol=0,rtol=0)
        ordinary=norm(inputs)
        torch.testing.assert_close(seen[0][:,:lo],ordinary[:,:lo],atol=0,rtol=0)
        torch.testing.assert_close(seen[0][:,hi:],ordinary[:,hi:],atol=0,rtol=0)
        contact_grad=torch.autograd.grad(seen[0][:,lo:hi].sum(),raw)[0][:,SOLREF]
        assert torch.isfinite(contact_grad).all() and (contact_grad.abs()>0).all()
        assert state_digest(model)==baseline
        # Even deliberately extreme synthetic running statistics cannot change these input columns.
        with torch.no_grad():norm.mean[lo:hi]=torch.tensor([-10000.,10000.]);norm.var[lo:hi]=0.
        changed=normalize_contact_inputs(inputs,norm,offset)
        torch.testing.assert_close(changed[:,lo:hi],expected,atol=0,rtol=0)
        assert changed[:,lo:hi].abs().max()<=1. and ordinary[:,lo:hi].abs().max()>1.
        reports.append(dict(network=kind,input_columns=[lo,hi],actual_first_linear_contact=expected.detach().tolist(),
            contact_gradient=contact_grad.tolist(),norm_updates_for_synthetic_check=1,
            constant_material_variance_floor_and_extreme_mean_bypassed=True,
            other_input_columns_retain_running_normalization=True,model_state_frozen_during_forward=True))
    return reports


def play_route_check():
    class StopBeforeConstruction(Exception):pass
    observed=[]
    def constructor(*args,**kwargs):
        observed.append(kwargs['env_cfg']);raise StopBeforeConstruction
    args=SimpleNamespace(task='chuanliantui_walk',num_envs=4,seed=55,device='cpu')
    with patch.object(entry.task_registry,'make_env',side_effect=constructor), \
            patch.object(entry,'supermimic_nominal_contact',side_effect=AssertionError('Coverage called in play')) as coverage:
        try:entry.make_env(args,play=True)
        except StopBeforeConstruction:pass
        else:raise AssertionError('Unexpected environment construction')
    assert len(observed)==1 and 'supermimic_nominal_contact' not in observed[0].events and coverage.call_count==0
    return dict(real_play_config_has_no_training_contact_event=True,coverage_calls=0,simulator_constructed=False)


def compatibility_check(checkpoint,output,expected_contract):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert expected_contract in (V9,V10)
    assert saved['algorithm_contract']==expected_contract and saved['asset_sha256']==asset_digest()
    version='v9' if expected_contract==V9 else 'v10'
    command=SimpleNamespace(cfg=command_cfg('walk'),load_checkpoint_state=lambda _state:None)
    learner=Learner('cpu');before={n:state_digest(getattr(learner,n)) for n in ('actor','world','critic')}
    for opt in ('opt_actor','opt_encoder','opt_world','opt_critic'):assert not getattr(learner,opt).state
    with patch.object(command,'load_checkpoint_state') as load_command:
        try:learner.load(checkpoint,command)
        except ValueError:pass
        else:raise AssertionError(f'{version} full resume accepted')
        assert load_command.call_count==0
    assert before=={n:state_digest(getattr(learner,n)) for n in before}
    learner.load(checkpoint,command,inference=True)
    assert all(torch.equal(v,saved['actor'][k]) for k,v in learner.actor.state_dict().items())
    for n in ('world','critic'):assert before[n]==state_digest(getattr(learner,n))
    transferred=Learner('cpu');cold={n:state_digest(getattr(transferred,n)) for n in ('world','critic')}
    initialization=transferred.init_actor(checkpoint,command,task='chuanliantui_walk')
    assert initialization['fresh_from_zero'] is False and initialization['parent_sha256']==digest(checkpoint)
    assert all(torch.equal(v,saved['actor'][k]) for k,v in transferred.actor.state_dict().items())
    assert cold=={n:state_digest(getattr(transferred,n)) for n in cold}
    for n in ('opt_actor','opt_encoder','opt_world','opt_critic'):assert not getattr(transferred,n).state
    assert transferred.world.in_norm.count==0 and transferred.world.delta_norm.count==0 and transferred.critic.norm.count==0
    # Real old encoder+MLP actions remain identical, including exported historical metadata.
    history=torch.randn(8,125)
    torch.testing.assert_close(act(learner.actor,history),act(transferred.actor,history),atol=0,rtol=0)
    from ct_mjlab.supermimic.export import export
    metadata=export(checkpoint,output/f'{version}_actor_only.onnx')
    assert metadata['algorithm_contract']==expected_contract
    return dict(real_checkpoint=str(checkpoint),checkpoint_algorithm_contract=expected_contract,
        checkpoint_sha256=digest(checkpoint),
        fullresume_rejected_before_network_or_curriculum_mutation=True,full_actor_inference_bitwise=True,
        full_actor_transfer_bitwise=True,new_world_value_and_four_optimizers_cold=True,
        initialization=initialization,old_actor_export_contract=metadata['algorithm_contract'],
        onnx_max_error=metadata['numeric_max_error'])


def unchanged_task_sources(checkpoint):
    source=Path(__file__).resolve().parents[3]/'mjlab_training/src/ct_mjlab'
    manifest=checkpoint.parent/'source_sha256.json';saved=json.loads(manifest.read_text())
    protected={key:value for key,value in saved.items() if not key.startswith('supermimic/') or
        Path(key).name in ('state.py','analytic_reward.py','kinematics.py','buffer.py','export.py')}
    assert len([k for k in protected if not k.startswith('supermimic/')])==90
    for name,expected in protected.items():assert digest(source/name)==expected,name
    return dict(v9_source_manifest=str(manifest),manifest_sha256=digest(manifest),
        unchanged_source_sha256=protected,xml_sha256=asset_digest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v9-checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--v10-checkpoint',type=Path)
    args=parser.parse_args();assert ALGORITHM_CONTRACT==V11 and not torch.cuda.is_initialized()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);torch.set_num_threads(1);torch.manual_seed(55)
    sources=unchanged_task_sources(args.v9_checkpoint)
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Optimizer step forbidden')) as steps:
        report=dict(algorithm_contract=ALGORITHM_CONTRACT,encoding=encoding_check(),network_inputs=network_input_check(),
            play_route=play_route_check(),compatibility=compatibility_check(args.v9_checkpoint,args.output,V9),
            compatibility_v10=(compatibility_check(args.v10_checkpoint,args.output,V10) if args.v10_checkpoint else None),
            unchanged_task=sources,actual_optimizer_steps=steps.call_count,device='cpu',physics_steps=0,
            scope='CPU mathematical/input/compatibility checks; real 25% contact coverage is checked separately by integration; no training or behavior evidence.')
    assert steps.call_count==0 and not torch.cuda.is_initialized()
    from ct_mjlab.supermimic import networks,learner,state
    report['source_sha256']={str(Path(m.__file__)):digest(m.__file__) for m in (networks,learner,state,entry)}
    report['checker_sha256']=digest(__file__)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
