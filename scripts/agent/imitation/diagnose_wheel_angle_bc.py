#!/usr/bin/env python3
"""Zero-update diagnosis of teacher fitting versus independent student state coverage."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from ct_mjlab.bc import ActionDataset,load_actor


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def diagnose(args):
    torch.set_num_threads(1)
    data=ActionDataset([args.dataset],require_reference=True)
    actor,saved=load_actor(args.checkpoint,args.device)
    assert saved['dataset']['fingerprint']==data.description['fingerprint']
    assert saved['dataset'].get('wheel_angles')
    before={k:v.clone() for k,v in actor.state_dict().items()}
    scale=saved['target_scale'].to(args.device)
    fits={}
    for split in ('train','validation'):
        fits[split]={}
        for name,pool in data.classes[split].items():
            x,y=(a[pool].to(args.device) for a in data.frames[split])
            prediction=torch.cat([actor.mlp(chunk) for chunk in x.split(4096)])
            error=prediction-y
            altered=x.clone();altered[:,14:16]=0.
            ablated=torch.cat([actor.mlp(chunk) for chunk in altered.split(4096)])
            fits[split][name]=dict(frames=len(x),scaled_mse=float((error/scale).square().mean()),
                action_mae=error.abs().mean(0).tolist(),action_rmse=error.square().mean(0).sqrt().tolist(),
                wheel_slots_zeroed_scaled_mse=float(((ablated-y)/scale).square().mean()))
    train=data.frames['train'][0].to(args.device)
    # 固定每十帧取一个训练状态；距离仅衡量覆盖，不视为这些状态上的教师动作。
    std=train.std(0,correction=0).clamp_min(.05)
    variable=(train.std(0,correction=0)>.001).nonzero().flatten()
    bank=train[::10,variable]/std[variable]
    def coverage(x):
        x=x[::max(1,len(x)//4096)][:4096].to(args.device)
        query=x[:,variable]/std[variable]
        distance=torch.cat([torch.cdist(chunk,bank).amin(-1) for chunk in query.split(128)])/len(variable)**.5
        return dict(sampled_frames=len(x),nearest_standardized_rms_quantiles=torch.quantile(distance,torch.tensor([.5,.9,.99],device=args.device)).tolist(),
            outside_training_range_fraction=((x<train.amin(0)-.001)|(x>train.amax(0)+.001)).float().mean(0).tolist(),
            wheel_angle_range_rad=torch.stack((x[:,14:16].amin(0),x[:,14:16].amax(0))).tolist())
    state_coverage={'teacher_validation_rise':coverage(data.frames['validation'][0][data.classes['validation']['rise']])}
    summaries={}
    for name,run in (('training_initial_states',args.train_run),('heldout_initial_states',args.heldout_run)):
        report=json.loads((run/'result.json').read_text())
        assert report['checkpoint_sha256']==digest(args.checkpoint)
        assert report['trace_sha256']==digest(run/'trace.npz')
        with np.load(run/'trace.npz') as d:
            time=d['time']-.01;alive=d['alive']
            bins={}
            for start,end in ((.2,.5),(.5,1.),(1.,2.),(2.,5.)):
                mask=alive&((time>=start)&(time<end))[:,None]
                if mask.any():bins[f'{start}-{end}s']=coverage(torch.from_numpy(d['observation'][mask].copy()))
            lengths=alive.sum(0)*.01
        state_coverage[name]=bins
        summaries[name]=dict(num_envs=report['num_envs'],strict_rise=report['strict_rise'],
            final_balance=report['final_balance'],qualified=report['qualified'],
            full_20s=sum(r['full_20s'] for r in report['rows']),duration_range_s=[float(lengths.min()),float(lengths.max())])
    for k,v in before.items():torch.testing.assert_close(v,actor.state_dict()[k],atol=0,rtol=0)
    result=dict(completed=True,optimizer_updates=0,checkpoint_sha256=digest(args.checkpoint),
        dataset_fingerprint=data.description['fingerprint'],fit=fits,independent_control=summaries,
        state_coverage=state_coverage,distance_training_bank_frames=len(bank),variable_observation_indices=variable.tolist(),
        no_weights_changed=True,checker_sha256=digest(Path(__file__)),
        scope='Offline prediction and descriptive state coverage; zeroed wheel inputs are an offline sensitivity check, not a trained ablation or causal proof')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(fit=fits,independent_control=summaries,optimizer_updates=0),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','checkpoint','train-run','heldout-run','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');diagnose(p.parse_args())
