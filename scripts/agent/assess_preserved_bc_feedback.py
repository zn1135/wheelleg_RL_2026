#!/usr/bin/env python3
"""Measure retained local feedback on untouched heldout standing anchors."""
import argparse,json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ct_mjlab.bc import load_actor
from ct_mjlab.envs import task_registry
from check_bc_local_feedback import pd_request
from check_lqr_student_closed_loop import digest


@torch.inference_mode()
def assess(args):
    torch.set_num_threads(1)
    actor,saved=load_actor(args.checkpoint,'cuda:0');reference,original=load_actor(args.reference,'cuda:0')
    assert saved['initialization']['sha256']==digest(args.reference) and saved['preservation']['heldout_used'] is False
    before={k:v.clone() for k,v in actor.state_dict().items()}
    cfg,_=task_registry.get_cfgs('chuanliantui',play=True)
    source=json.loads((args.anchors/'check.json').read_text());assert source['probes_sha256']==digest(args.anchors/'probes.npz')
    from mjlab.utils.torch import configure_torch_backends
    summaries={}
    for precision in ('ieee_fp32','tf32_runtime'):
        configure_torch_backends(allow_tf32=precision=='tf32_runtime')
        rows=[]
        for variable,amplitudes in (('pitch_rad',[math.radians(x) for x in (.25,.5,1.)]),('wheel_forward_rad_s',[.1,.25,.5])):
            for amplitude in amplitudes:
                derivatives={name:[] for name in ('stable','preserved')}
                for tick in (2500,5000,7500):
                    with np.load(args.anchors/f'controller-{tick}.npz') as d:
                        obs=torch.from_numpy(d['observation'].copy()).cuda();q=torch.from_numpy(d['physical_q'].copy()).cuda()
                        dq=torch.from_numpy(d['physical_dq'].copy()).cuda();valid=d['valid'].copy()
                        action=SimpleNamespace(cfg=cfg.actions['pd'],_leg_action_ids=torch.tensor([0,1,3,4],device='cuda'),
                            _wheel_action_ids=torch.tensor([2,5],device='cuda'),_effort_limits=torch.tensor([40.,40.,3.9,40.,40.,3.9],device='cuda'),
                            nominal_positions=torch.from_numpy(d['nominal_positions'].copy()).cuda(),
                            reference_positions=torch.from_numpy(d['reference_positions'].copy()).cuda())
                    responses={name:[] for name in derivatives}
                    for sign in (-1,1):
                        x=obs.clone();velocity=dq.clone();delta=sign*amplitude
                        if variable=='pitch_rad':
                            pitch=x[:,7].asin()+delta;roll=torch.atan2(-x[:,8],-x[:,9])
                            x[:,7]=pitch.sin();x[:,8]=-roll.sin()*pitch.cos();x[:,9]=-roll.cos()*pitch.cos()
                        else:
                            x[:,20]+=delta*.1;x[:,21]-=delta*.1;velocity[:,2]+=delta;velocity[:,5]-=delta
                        legacy=x.clone();legacy[:,14:16]=0.
                        for name,target in (('stable',reference.mlp(legacy)),('preserved',actor.mlp(x))):
                            responses[name].append(pd_request(action,target,q,velocity).cpu().numpy())
                    for name in derivatives:derivatives[name].append(((responses[name][1]-responses[name][0])/(2*amplitude))[valid])
                derivatives={k:np.concatenate(v) for k,v in derivatives.items()}
                wheel={k:(v[:,2]-v[:,5])*.5 for k,v in derivatives.items()}
                cosine=(derivatives['stable']*derivatives['preserved']).sum(-1)/(
                    np.linalg.norm(derivatives['stable'],axis=-1)*np.linalg.norm(derivatives['preserved'],axis=-1)).clip(1e-9)
                rows.append(dict(variable=variable,amplitude=amplitude,anchors=len(cosine),
                    stable_wheel_derivative_median=float(np.median(wheel['stable'])),
                    preserved_wheel_derivative_median=float(np.median(wheel['preserved'])),
                    wheel_gain_ratio_median=float(np.median(wheel['preserved']/wheel['stable'])),
                    wheel_direction_agreement=float((wheel['stable']*wheel['preserved']>0).mean()),
                    six_motor_direction_cosine_median=float(np.median(cosine))))
        summaries[precision]=rows
    for name,value in before.items():torch.testing.assert_close(value,actor.state_dict()[name],atol=0,rtol=0)
    result=dict(completed=True,optimizer_updates=0,checkpoint_sha256=digest(args.checkpoint),reference_sha256=digest(args.reference),
        source_anchor_sha256=source['probes_sha256'],heldout_anchors_used_for_training=False,precision_comparison=summaries,
        checker_sha256=digest(Path(__file__)),scope='120 untouched heldout teacher states, fixed-history nominal requests only; does not replace independent physical40 acceptance.')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','reference','anchors','output'):p.add_argument('--'+name,type=Path,required=True)
    assess(p.parse_args())
