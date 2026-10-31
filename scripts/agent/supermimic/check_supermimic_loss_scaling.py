#!/usr/bin/env python3
"""V11 four-coordinate loss, gradients and recipe guards; zero optimizer updates."""
import argparse
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.agent.supermimic.check_supermimic_bptt import fixture, synthetic_buffer
from ct_mjlab.supermimic.learner import Learner, ALGORITHM_CONTRACT, WORLD_LOSS_RECIPE
from ct_mjlab.supermimic.networks import WorldModel
from ct_mjlab.supermimic.state import PHYS_DIM, NOISE, physical_delta, integrate_physical
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg

V11='chuanliantui-supermimic-canonical-physics-fixed-support-scale-robust-wheel-angle-v11'


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def equal(first,second):
    if isinstance(first,torch.Tensor):
        return isinstance(second,torch.Tensor) and first.dtype==second.dtype and first.shape==second.shape and torch.equal(first,second)
    if isinstance(first,dict):return first.keys()==second.keys() and all(equal(first[k],second[k]) for k in first)
    if isinstance(first,(tuple,list)):return len(first)==len(second) and all(equal(a,b) for a,b in zip(first,second))
    return first==second


def module_digest(module):
    result=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        result.update(name.encode());result.update(str(value.dtype).encode());result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def formula_checks():
    learner=Learner('cpu');before={n:copy.deepcopy(getattr(learner,n).state_dict()) for n in ('actor','world','critic')}
    _,state,_,_=fixture(8)
    truth=state[:,None,:PHYS_DIM].double().expand(-1,5,-1).clone()
    norm=learner.world.delta_norm
    norm.var.copy_(torch.linspace(.01,36.,63))
    std=norm.std().clone();wheel=learner.world.adapter.motor_q[[2,5]]-1
    assert wheel.tolist()==[8,13]
    delta=torch.randn(8,5,63,dtype=torch.float64)*.02;delta[...,3:6]=0.
    z=torch.tensor([-100.,-2.,-1.,-.25,.25,1.,2.,100.],dtype=delta.dtype)
    delta[...,wheel]=z[:,None,None]*std[wheel]
    delta[...,59]=torch.tensor([-20.,-5.,0.,5.,20.],dtype=delta.dtype)
    delta[...,62]=-delta[...,59]
    path=integrate_physical(truth,delta).detach().requires_grad_()
    terms=learner.world_loss_terms(truth,path)
    raw=physical_delta(truth,path);old=(raw/std).square()
    untouched=[i for i in range(63) if i not in (8,13,59,62)]
    assert torch.equal(terms[...,untouched],old[...,untouched])
    expected=old.clone()
    normalized=raw[...,wheel]/std[wheel]
    expected[...,wheel]=torch.where(normalized.abs()<=1.,normalized.square(),2.*normalized.abs()-1.)
    expected[...,59]=raw[...,59].square()/25.;expected[...,62]=raw[...,62].square()/25.
    torch.testing.assert_close(terms,expected,atol=2e-12,rtol=2e-12)
    small=(normalized.abs()<=1.)
    assert torch.equal(terms[...,wheel][small],old[...,wheel][small])
    grad=torch.autograd.grad(terms.sum(),path)[0]
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(grad[...,wheel+1]*std[wheel],2.*normalized.clamp(-1.,1.),atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(grad[...,60],2.*raw[...,59]/25.,atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(grad[...,63],2.*raw[...,62]/25.,atol=1e-11,rtol=1e-11)
    # A noisy training force distribution cannot dilute the fixed support penalty.
    norm.var[59]=1e10;norm.var[62]=1e12
    changed=learner.world_loss_terms(truth,path.detach())
    assert torch.equal(changed[...,59:63:3],terms.detach()[...,59:63:3])
    # No wrapping or clipping of actual wheel position targets: large errors still cost more.
    assert (terms[-1,...,wheel]>terms[5,...,wheel]).all()
    learner.world.load_state_dict(before['world'])
    assert all(equal(getattr(learner,n).state_dict(),before[n]) for n in before)
    assert all(not getattr(learner,n).state for n in ('opt_actor','opt_encoder','opt_world','opt_critic'))
    w=synthetic_buffer().sample(4,3);w['mask'].zero_()
    zero,_=learner.world_loss(w);zero.backward()
    assert zero.item()==0. and all(p.grad is None or not torch.count_nonzero(p.grad) for p in learner.world.parameters())
    return dict(untouched_coordinates_bitwise=untouched,wheel_position_delta_indices=wheel.tolist(),
        independently_computed_piecewise_loss=True,small_wheel_errors_keep_exact_square=True,
        normalized_wheel_residual_derivative_bounded_by_2=True,physical_force_derivative_checked=True,
        huge_force_variance_does_not_dilute_support_loss=True,unwrapped_targets_not_clipped=True,
        all_masked_world_loss_and_gradients_zero=True,model_norm_optimizer_state_unchanged=True,
        synthetic_variance_fixture_restored=True)


def recipe_checks(output):
    command=SimpleNamespace(cfg=command_cfg('walk'),checkpoint_state=lambda:dict(check_only=True),
                            load_checkpoint_state=lambda _state:None)
    first=Learner('cpu');path=output/'v11_zero_update_fixture.pt'
    config=dict(task='chuanliantui_walk',mode='train',world_steps_per_update=4,
        world_loss_recipe=WORLD_LOSS_RECIPE.copy(),initialization=dict(kind='zero_update_loss_recipe_fixture'))
    first.save(path,command,config=config,iteration=0,transitions=0,world_updates=0)
    second=Learner('cpu');second.load(path,command)
    names=('actor','world','critic','opt_actor','opt_encoder','opt_world','opt_critic')
    assert all(equal(getattr(first,n).state_dict(),getattr(second,n).state_dict()) for n in names)
    rejected=[]
    for kind in ('missing','force_scale','huber_delta'):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if kind=='missing':saved['config'].pop('world_loss_recipe')
        elif kind=='force_scale':saved['config']['world_loss_recipe']['wheel_vertical_force_scale_N']=10.
        else:saved['config']['world_loss_recipe']['wheel_position_huber_delta']=2.
        wrong=output/f'invalid_recipe_{kind}.pt';torch.save(saved,wrong)
        before={n:copy.deepcopy(getattr(second,n).state_dict()) for n in names}
        rng=torch.get_rng_state().clone()
        with ExitStack() as guards:
            for n in names:
                guards.enter_context(patch.object(getattr(second,n),'load_state_dict',side_effect=AssertionError('State load before rejection')))
            load=guards.enter_context(patch.object(command,'load_checkpoint_state',side_effect=AssertionError('Curriculum before rejection')))
            guards.enter_context(patch.object(torch,'set_rng_state',side_effect=AssertionError('RNG before rejection')))
            try:second.load(wrong,command)
            except ValueError:pass
            else:raise AssertionError('Wrong loss recipe resumed')
            assert load.call_count==0
        assert all(equal(before[n],getattr(second,n).state_dict()) for n in names) and torch.equal(rng,torch.get_rng_state())
        rejected.append(kind)
    bad=copy.deepcopy(config);bad.pop('world_loss_recipe');unsaved=output/'must_not_save.pt'
    try:first.save(unsaved,command,config=bad,iteration=0,transitions=0,world_updates=0)
    except ValueError:pass
    else:raise AssertionError('Missing recipe was saved')
    assert not unsaved.exists()
    return dict(v11_checkpoint_all_networks_and_optimizers_roundtrip=True,
        invalid_recipes_rejected_before_any_network_optimizer_curriculum_rng_load=rejected,
        save_without_recipe_rejected_before_file_write=True,fixture_checkpoint=str(path),actual_optimizer_updates=0)


@torch.no_grad()
def decoder_checks(trial):
    receipt=json.loads((trial/'scoring-exit.json').read_text())
    old_path=trial/'score/predictions.pt'
    assert digest(old_path)==receipt['predictions_sha256']
    sealed=torch.load(old_path,map_location='cpu',weights_only=False)
    checkpoint=next(Path(p['checkpoint']) for p in sealed['worlds']
        if p['checkpoint_sha256']==receipt['final_checkpoint_sha256'])
    assert digest(checkpoint)==receipt['final_checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    world=WorldModel().cpu().eval().requires_grad_(False);world.load_state_dict(saved['world'])
    before=module_digest(world);checked=[]
    for index,w in enumerate(sealed['windows']):
        tr=w['truth'];get=lambda name:tr[name][:64,0]
        outputs=world.step(get('state'),get('parameters'),get('action'),get('next_frame')[:,6:9],
            get('context'),get('previous'),get('previous_previous'),get('next_state')[:,NOISE],return_path=True)
        reference=next(p['tensors'] for p in sealed['predictions'] if p['world_label']=='final' and p['recording_index']==index)
        for key,value in zip(('state','frame','reward','failure','features','path'),outputs):
            if key!='frame':assert torch.equal(value,reference[key][:64,0]),key
        checked.append(dict(recording=index,rows=64,state_reward_failure_features_path_bitwise_original=True))
    assert module_digest(world)==before
    assert digest(old_path)==receipt['predictions_sha256'] and digest(checkpoint)==receipt['final_checkpoint_sha256']
    return dict(real_v10_checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),checks=checked,
        decoder_and_all_normalization_bitwise_unchanged=True,world_step_calls=2,
        scope='Direct frozen WorldModel read only, not V10 Learner full resume or optimizer restore.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    assert ALGORITHM_CONTRACT==V11 and not torch.cuda.is_initialized()
    torch.set_num_threads(2);torch.manual_seed(55)
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No optimizer update')) as optimizer:
        report=dict(algorithm_contract=ALGORITHM_CONTRACT,world_loss_recipe=WORLD_LOSS_RECIPE.copy(),
            formula=formula_checks(),recipe=recipe_checks(args.output),decoder=decoder_checks(args.trial),
            optimizer_steps=optimizer.call_count,simulator_steps=0,
            scope='Loss algebra/gradient, recipe guards and unchanged frozen predictions; no training or behavior improvement evidence.')
    assert optimizer.call_count==0 and not torch.cuda.is_initialized()
    from ct_mjlab.supermimic import learner,networks,state,__main__ as entry
    report['source_sha256']={str(Path(m.__file__)):digest(m.__file__) for m in (learner,networks,state,entry)}
    report['checker_sha256']=digest(__file__)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),optimizer_steps=optimizer.call_count,
        decoder_bitwise_unchanged=True,recipe_checks_passed=True),ensure_ascii=False))


if __name__=='__main__':main()
