#!/usr/bin/env python3
"""Replay saved local feedback probes from frozen controller histories; no physics."""
import argparse,copy,json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ct_mjlab.bc import load_actor,ORDER
from ct_mjlab.teacher.controller import LqrVmcController
from ct_mjlab.envs import task_registry
from check_bc_local_feedback import pd_request,teacher_target
from check_lqr_student_closed_loop import digest


@torch.inference_mode()
def assess(args):
    torch.set_num_threads(1)
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    report=json.loads((args.run/'check.json').read_text())
    assert report['completed'] and report['optimizer_updates']==0 and report['models_unchanged']
    assert report['probes_sha256']==digest(args.run/'probes.npz')
    models={}
    for name,path in (('stable_legacy39',args.stable),('wheel39_5000',args.first),('wheel39_10000',args.latest)):
        assert report['checkpoint_sha256'][name]==digest(path)
        models[name]=load_actor(path,'cuda:0')[0]
    cfg,_=task_registry.get_cfgs('chuanliantui',play=True)
    errors=[];velocity_errors=[];gravity_errors=[]
    with np.load(args.run/'probes.npz') as probes:
        for index,tick in enumerate((2500,5000,7500)):
            with np.load(args.run/f'controller-{tick}.npz') as saved:
                controller=LqrVmcController(40,'cuda:0',velocity_integral=True,dt=.01)
                for name in saved.files:
                    if name.startswith('input.') or name in ('observation','physical_q','physical_dq','valid','nominal_positions','reference_positions'):continue
                    target=controller;parts=name.split('.')
                    for part in parts[:-1]:target=getattr(target,part)
                    current=getattr(target,parts[-1]);value=saved[name]
                    if isinstance(current,torch.Tensor):current.copy_(torch.from_numpy(value.copy()).to('cuda:0'))
                    else:setattr(target,parts[-1],value.item())
                inputs={name[6:]:torch.from_numpy(saved[name].copy()).cuda() for name in saved.files if name.startswith('input.')}
                obs=torch.from_numpy(saved['observation'].copy()).cuda()
                q=torch.from_numpy(saved['physical_q'].copy()).cuda();dq=torch.from_numpy(saved['physical_dq'].copy()).cuda()
                action=SimpleNamespace(cfg=cfg.actions['pd'],_leg_action_ids=torch.tensor([0,1,3,4],device='cuda'),
                    _wheel_action_ids=torch.tensor([2,5],device='cuda'),raw_action=torch.zeros(40,6,device='cuda'),
                    _effort_limits=torch.tensor([40.,40.,3.9,40.,40.,3.9],device='cuda'),
                    nominal_positions=torch.from_numpy(saved['nominal_positions'].copy()).cuda(),
                    reference_positions=torch.from_numpy(saved['reference_positions'].copy()).cuda(),
                    teacher=SimpleNamespace(order=ORDER,sign=torch.tensor([1.,1.,-1.,-1.,1.,-1.],device='cuda'),
                        limits=torch.tensor([40.,40.,3.9,40.,40.,3.9],device='cuda')))
            np.testing.assert_array_equal(probes[f'{index}.observation'],obs.cpu().numpy())
            velocity_errors.append(float(abs(obs[:,16:22]-dq[:,ORDER]*.1).max()))
            g=torch.stack((inputs['pitch'].sin(),-inputs['roll'].sin()*inputs['pitch'].cos(),
                -inputs['roll'].cos()*inputs['pitch'].cos()),-1)
            gravity_errors.append(float(abs(obs[:,7:10]-g).max()))
            for variable,amplitudes in (('pitch_rad',[math.radians(x) for x in (.25,.5,1.)]),('wheel_forward_rad_s',[.1,.25,.5])):
                for amplitude in amplitudes:
                    for sign in (-1,1):
                        measurement={k:v.clone() for k,v in inputs.items()};altered=obs.clone();physical_dq=dq.clone()
                        delta=sign*amplitude
                        if variable=='pitch_rad':
                            measurement['pitch']+=delta;measurement['upright']=measurement['roll'].cos()*measurement['pitch'].cos()
                            angle=torch.asin(obs[:,7].clamp(-1,1))+delta;roll=torch.atan2(-obs[:,8],-obs[:,9])
                            altered[:,7]=angle.sin();altered[:,8]=-roll.sin()*angle.cos();altered[:,9]=-roll.cos()*angle.cos()
                        else:
                            measurement['wheel_speed']+=delta;physical_dq[:,2]+=delta;physical_dq[:,5]-=delta
                            altered[:,20]+=delta*.1;altered[:,21]-=delta*.1
                        target,_=teacher_target(action,copy.deepcopy(controller),measurement,q,physical_dq)
                        targets={'teacher':target}
                        for name,actor in models.items():
                            current=altered.clone()
                            if name=='stable_legacy39':current[:,14:16]=0.
                            targets[name]=actor.mlp(current)
                        key=f'{index}.{variable}:{amplitude}:{sign}'
                        for name,target in targets.items():
                            actual=pd_request(action,target,q,physical_dq).cpu().numpy()
                            expected=probes[f'{key}.{name}.torque']
                            np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-6)
                            np.testing.assert_allclose(target.cpu().numpy(),probes[f'{key}.{name}.target'],atol=2e-6,rtol=2e-6)
                            errors.append(float(np.max(abs(actual-expected))))
    assert max(velocity_errors)<=2e-6 and max(gravity_errors)<=2e-6
    result=dict(passed=True,optimizer_updates=0,branches_recomputed=len(errors),max_nominal_torque_difference_nm=max(errors),
        max_actor_velocity_vs_physical_input_difference=max(velocity_errors),
        max_actor_gravity_vs_teacher_orientation_difference=max(gravity_errors),
        source_probe_sha256=report['probes_sha256'],assessor_sha256=digest(Path(__file__)),
        scope='Frozen teacher histories and all plus/minus targets/nominal motor requests replayed on original GPU precision; no statement about perturbed physical trajectories or unique cause')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','stable','first','latest','output'):p.add_argument('--'+name,type=Path,required=True)
    assess(p.parse_args())
