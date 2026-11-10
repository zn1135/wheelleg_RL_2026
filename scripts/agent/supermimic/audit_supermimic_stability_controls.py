"""Read-only comparison of state clamps, action bounds and learning-rate schedules."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from ct_mjlab.envs.chuanliantui.model import load_model

ROOT=Path(__file__).resolve().parents[3]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    ref=Path('/home/zn/文档/supermimic_wheelbipe');reference_sources={str(p):sha(p) for p in [ref/'supermimic_wheelbipe'/n for n in ('train.py','mdp.py','constants.py','env.py')]}
    reference_checkpoint=torch.load(ref/'logs/wb_final/pi.pt',map_location='cpu',weights_only=False,mmap=True)
    ref_args=reference_checkpoint['args'];ref_args=vars(ref_args) if hasattr(ref_args,'__dict__') else ref_args
    reference_decay=ref_args['lr_decay_iters'];base_lr=ref_args['lr']
    schedule={str(i):dict(factor=.03+.97*.5*(1+math.cos(math.pi*min(i,reference_decay)/reference_decay)),
        lr=base_lr*(.03+.97*.5*(1+math.cos(math.pi*min(i,reference_decay)/reference_decay)))) for i in (0,5200,8000,24000,120000)}
    paths=[('SM615',ROOT/'artifacts/supermimic/supermimic-v13-baseline615-current13-seed43-20261031'),('A2200',ROOT/'artifacts/supermimic/supermimic-v13-r2-a2200-seed43-20261101')]
    paths += [(f'A{n}',ROOT/f'artifacts/supermimic/supermimic-v13-r3-a{n}-seed43-20261101') for n in (3200,4200)]
    paths.append(('A5200',ROOT/'artifacts/supermimic/supermimic-v13-r3-a5200-seed43-20261101'))
    actions=[];hashes={}
    for label,directory in paths:
        s=json.loads((directory/'summary.json').read_text());p=directory/'trajectory.npz'
        assert sha(p)==s['trajectory_sha256'];hashes[str(p)]=s['trajectory_sha256']
        with np.load(p) as z:
            # One recorded action per policy tick; first-episode mask excludes all resets.
            values=z['values'][::5];layout=s['layout'];valid=values[:,:,layout['valid'][0]]>.5
            a=values[:,:,slice(*layout['raw_action'])][valid]
        row=dict(model=label,valid_policy_environment_pairs=len(a),scope='First episodes only; A5200 stops at1.09s, others30s. Not equal-duration trend statistics.',groups={})
        for name,ids in [('legs',[0,1,3,4]),('wheels',[2,5])]:
            x=abs(a[:,ids]);row['groups'][name]=dict(component_samples=x.size,above3=int((x>3).sum()),fraction_above3=float((x>3).mean()),max_abs=float(x.max()),p95_abs=float(np.quantile(x,.95)))
        actions.append(row)
    run=ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261101_v13_continuation_to5200_r3';rates=[]
    for n,name in [(2200,'initial.pt'),(3200,'actor_003200.pt'),(4200,'actor_004200.pt'),(5200,'final.pt')]:
        s=torch.load(run/name,map_location='cpu',weights_only=False,mmap=True)
        rates.append(dict(actor_updates=n,collection_iteration=s['iteration'],optimizer_lrs={key:s['opt_'+key]['param_groups'][0]['lr'] for key in ('world','actor','encoder','critic')}))
    prediction_path=ROOT/'artifacts/supermimic/supermimic-v13-continuation-20261101-r3/prediction/predictions.pt'
    pred=torch.load(prediction_path,map_location='cpu',weights_only=False,mmap=True)
    model=load_model();joint_indices=[i for i in range(1,model.njnt) if model.jnt_limited[i] and model.jnt_qposadr[i] not in (9,14)]
    limits=[dict(name=model.joint(i).name,qpos_index=int(model.jnt_qposadr[i]),low=float(model.jnt_range[i,0]),high=float(model.jnt_range[i,1])) for i in joint_indices]
    sealed_path=ROOT/'artifacts/supermimic/supermimic-v13-a2200-prediction-20261101-r1/historical/predictions.pt'
    sealed=torch.load(sealed_path,map_location='cpu',weights_only=False,mmap=True)
    truth_bounds=[]
    for index,window in enumerate(sealed['windows']):
        for entry in limits:
            x=window['truth']['path'][...,entry['qpos_index']]
            over=torch.maximum((entry['low']-x).clamp_min(0),(x-entry['high']).clamp_min(0))
            if (over>0).any():truth_bounds.append(dict(dataset=f'historical_{index}',joint=entry['name'],outside=int((over>0).sum()),max_excess=float(over.max()),unit='rad' if model.jnt_type[joint_indices[limits.index(entry)]]==3 else 'm'))
    clamps=[]
    for item in pred['predictions']:
        path=item['path'];assert torch.isfinite(path).all()
        z=path[...,2];v=path[...,25:28];w=path[...,28:31]
        body_bad=(z<.02)|(z>2)|(v.abs()>20).any(-1)|(w.abs()>40).any(-1)
        joint_counts={entry['name']:int(((path[...,entry['qpos_index']]<entry['low'])|(path[...,entry['qpos_index']]>entry['high'])).sum()) for entry in limits}
        joint_excess={entry['name']:float(torch.maximum((entry['low']-path[...,entry['qpos_index']]).clamp_min(0),(path[...,entry['qpos_index']]-entry['high']).clamp_min(0)).max()) for entry in limits}
        clamps.append(dict(model=item['model'],dataset=item['dataset'],mode=item['mode'],states=z.numel(),
            root_height_min=float(z.min()),root_height_max=float(z.max()),linear_component_abs_max=float(v.abs().max()),angular_component_abs_max=float(w.abs().max()),
            reference_body_bound_violations=int(body_bad.sum()),current_XML_joint_bound_violations=joint_counts,current_XML_joint_max_excess=joint_excess))
    current_sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'mjlab_training/src/ct_mjlab/supermimic'/n for n in ('motion.py','motion_adapter.py','learner.py','__main__.py','networks.py')]}
    report=dict(action_statistics=actions,checkpoint_learning_rates=rates,reference_schedule=schedule,
        reference_export_iteration=reference_checkpoint['iteration'],reference_export_args={k:ref_args[k] for k in ('lr','lr_decay_iters','horizon','wm_window','buffer_steps')},
        reference_schedule_count='Reference outer iteration, not V13 actual actor optimizer count; formula values are illustrations, not matched training progress.',
        frozen_prediction_clamp_audit=clamps,current_XML_limits=limits,recorded_truth_limit_excess=truth_bounds,
        clamp_scope='Reference broad height/linear/angular bounds audited; joint-position bounds use current XML, not reference USD. Joint-speed limits require robot-specific definition, not blindly copied.',
        source_sha256=current_sources,reference_source_sha256=reference_sources,trajectory_sha256=hashes,
        prediction_sha256=sha(prediction_path),sealed_truth_sha256=sha(sealed_path),reference_export_sha256=sha(ref/'logs/wb_final/pi.pt'),
        optimizer_updates=0,simulator_steps=0,production_code_changes=False,
        scope='Source/config/old frozen trajectories audit. Current endpoint model is untrained; no new performance claim or clipping intervention run.')
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(actions=actions,rates=rates,schedule=schedule,body_clamp_violations=sum(r['reference_body_bound_violations'] for r in clamps),joint_violations=[(r['model'],r['dataset'],{k:v for k,v in r['current_XML_joint_bound_violations'].items() if v}) for r in clamps]),ensure_ascii=False))


if __name__=='__main__':main()
