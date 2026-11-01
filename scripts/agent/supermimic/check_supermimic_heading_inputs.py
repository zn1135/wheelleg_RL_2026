"""Reference-coordinate and planar symmetry checks; zero optimizer updates."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from mjlab.utils.lab_api.math import quat_apply,quat_mul,matrix_from_quat
from ct_mjlab.supermimic.motion import WorldModel,invariant_features47,physical47_from_full64
from ct_mjlab.supermimic.state import rotation_quat,unit_quat,NOISE,WRENCH
from ct_mjlab.supermimic.learner import Learner,ALGORITHM_CONTRACT,WORLD_LOSS_RECIPE
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from scripts.agent.supermimic.check_supermimic_motion import fixture,equal

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def transform(physical,yaw,translation):
    out=physical.clone()
    out[...,:3]=quat_apply(yaw,physical[...,:3])+translation
    out[...,3:7]=quat_mul(yaw,physical[...,3:7])
    out[...,25:28]=quat_apply(yaw,physical[...,25:28])
    for start in (55,58,61):out[...,start:start+3]=quat_apply(yaw,physical[...,start:start+3])
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=Path('/home/zn/文档/supermimic_wheelbipe/supermimic_wheelbipe'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(20261101)
    n=64;s,p,c=fixture(n);s[:,:3]=torch.randn(n,3);s[:,2]=.3
    euler=(torch.rand(n,3)-.5)*1.4;euler[:,2]=(torch.rand(n)-.5)*2*math.pi
    qx=rotation_quat(torch.stack((euler[:,0],torch.zeros(n),torch.zeros(n)),-1))
    qy=rotation_quat(torch.stack((torch.zeros(n),euler[:,1],torch.zeros(n)),-1))
    qz=rotation_quat(torch.stack((torch.zeros(n),torch.zeros(n),euler[:,2]),-1))
    s[:,3:7]=quat_mul(qz,quat_mul(qy,qx));s[:,25:49]=torch.randn(n,24)*.3
    p[:,WRENCH]=torch.randn(n,6)*2;a=torch.randn(n,6)*.1
    yaw=rotation_quat(torch.stack((torch.zeros(n),torch.zeros(n),(torch.rand(n)-.5)*2*math.pi),-1))
    offset=torch.randn(n,3);offset[:,2]=0
    rotated=s.clone();rotated[:,:64]=transform(s[:,:64],yaw,offset)
    rp=p.clone();rp[:,69:72]=quat_apply(yaw,p[:,69:72]);rp[:,72:75]=quat_apply(yaw,p[:,72:75])
    world=WorldModel()
    with torch.no_grad():world.net[-1].weight.normal_(std=1e-4)
    inp=world.inputs(s,p,a);changed=world.inputs(rotated,rp,a)
    torch.testing.assert_close(inp,changed,atol=3e-6,rtol=3e-6)
    negated=s.clone();negated[:,3:7]*=-1
    torch.testing.assert_close(inp,world.inputs(negated,p,a),atol=0,rtol=0)
    # Reference functions execute standalone; no simulator imports or reference edits.
    spec=importlib.util.spec_from_file_location('_reference_rotations',args.reference/'rotations.py')
    rotations=importlib.util.module_from_spec(spec);spec.loader.exec_module(rotations)
    node=next(n for n in ast.parse((args.reference/'mdp.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='invariant_features')
    namespace=dict(torch=torch,yaw_mat=rotations.yaw_mat,mat_to_6d=rotations.mat_to_6d)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(args.reference/'mdp.py'),'exec'),namespace)
    compact=physical47_from_full64(s[:,:64]);q=unit_quat(s[:,3:7])
    reference=namespace['invariant_features']((s[:,:3],rotations.quat_to_mat(q),s[:,25:28],quat_apply(q,s[:,28:31]),compact[:,7:23],compact[:,29:47]))
    local=invariant_features47(compact)
    torch.testing.assert_close(local,reference,atol=2e-6,rtol=2e-6)
    previous=torch.randn_like(a);previous2=torch.randn_like(a)
    one=world.step(s,p,a,torch.zeros(n,3),c,previous,previous2,s[:,NOISE],return_path=True)
    two=world.step(rotated,rp,a,torch.zeros(n,3),c,previous,previous2,rotated[:,NOISE],return_path=True)
    samples=one[-1].shape[-2]
    expected=transform(one[-1],yaw[:,None].expand(-1,samples,-1),offset[:,None].expand(-1,samples,-1))
    torch.testing.assert_close(expected,two[-1],atol=5e-6,rtol=5e-6)
    ds=s.clone().requires_grad_(True);dp=p.clone().requires_grad_(True);da=a.clone().requires_grad_(True)
    inputs=world.inputs(ds,dp,da);g=torch.autograd.grad(inputs.square().sum(),(ds,dp,da))
    assert all(torch.isfinite(x).all() for x in g)
    assert all(x.abs().max()>0 for x in g)
    vertical=s[:2].clone();vertical[:,3:7]=rotation_quat(torch.tensor([[0.,math.pi/2,0.],[0.,-math.pi/2,0.]]));vertical.requires_grad_(True)
    pole=world.inputs(vertical,p[:2],a[:2]);vg=torch.autograd.grad(pole.square().sum(),vertical)[0]
    assert torch.isfinite(pole).all() and torch.isfinite(vg).all()
    # The old129-wide model must not slip through just because tensor shapes match.
    old_path=ROOT/'artifacts/supermimic-v13-no-history-check-20261101-r2/zero-update-roundtrip.pt'
    old=torch.load(old_path,map_location='cpu',weights_only=False)
    assert old['world']['in_norm.mean'].numel()==129 and old['algorithm_contract']!=ALGORITHM_CONTRACT
    command=SimpleNamespace(cfg=command_cfg('walk'),load_checkpoint_state=lambda _:None)
    learner=Learner('cpu');before={k:copy.deepcopy(getattr(learner,k).state_dict()) for k in ('actor','world','critic')};rng=torch.get_rng_state().clone()
    try:learner.load(old_path,command)
    except ValueError as error:rejection=str(error)
    else:raise AssertionError('Old129 coordinate contract was silently accepted')
    assert all(equal(before[k],getattr(learner,k).state_dict()) for k in before) and torch.equal(rng,torch.get_rng_state())
    report=dict(passed=True,algorithm_contract=ALGORITHM_CONTRACT,world_loss_recipe=WORLD_LOSS_RECIPE,
        input_dim=129,state_feature_dim=47,input_yaw_translation_max_abs=float((inp-changed).abs().max()),
        reference_feature_max_abs=float((local-reference).abs().max()),
        predicted_path_planar_transform_max_abs=float((expected-two[-1]).abs().max()),
        quaternion_sign_invariant=True,nonzero_external_wrench_transformed=True,
        body_angular_velocity_not_rotated_twice=True,input_gradients_finite=True,vertical_heading_features_and_gradients_finite=True,
        old_same_width_coordinate_contract_rejected=True,rejection=rejection,
        reference_sha256={name:sha(args.reference/name) for name in ('mdp.py','rotations.py')},
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in [ROOT/'mjlab_training/src/ct_mjlab/supermimic/motion.py',ROOT/'mjlab_training/src/ct_mjlab/supermimic/learner.py',Path(__file__)]},
        optimizer_updates=0,simulator_steps=0,scope='Tensor/reference/gradient/checkpoint checks; no fit or behavior claim. Heading at exactly vertical forward axis is degenerate; only finiteness checked there.')
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No training')):main()
