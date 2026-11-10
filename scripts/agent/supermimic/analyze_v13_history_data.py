"""Read-only audit of original V13 delay coverage and recorded history alignment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261101_v13_continuation_to5200_r3'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def corr(x,y):
    x=np.asarray(x,dtype=np.float64).reshape(-1);y=np.asarray(y,dtype=np.float64).reshape(-1)
    return float(np.corrcoef(x,y)[0,1])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    initial_path=RUN/'initial_physics.pt';initial=torch.load(initial_path,map_location='cpu',weights_only=False)
    cp=torch.load(RUN/'final.pt',map_location='cpu',weights_only=False,mmap=True)
    assert cp['optimizer_update_counts']['actor']==5200
    assert initial['policy_dt_s']==.01 and initial['physics_dt_s']==.002
    delays=initial['parameters'][:,24:26]
    assert torch.equal(delays[:,0],delays[:,1])
    coverage={}
    for name,ids in [('train',initial['train_env_ids']),('validation',initial['validation_env_ids'])]:
        lag=delays[ids,1];v,c=torch.unique(lag,return_counts=True)
        coverage[name]=dict(environments=len(ids),lag_substep_counts={str(int(a)):int(b) for a,b in zip(v,c)},
            zero_delay_ids=ids[lag==0].tolist(),zero_delay_fraction=float((lag==0).double().mean()),
            mean_previous_action_fraction_of_5_substeps=float((lag/5).double().mean()))
    sources=json.loads((RUN/'source_sha256.json').read_text())
    for name in ['supermimic/buffer.py','supermimic/__main__.py','supermimic/learner.py','envs/chuanliantui_stand_v2/action.py']:
        assert sha(RUN/'snapshot/ct_mjlab'/name)==sources[name]
    recorded=[];hashes={str(initial_path):sha(initial_path)}
    for n,directory in [(4200,ROOT/'artifacts/supermimic/supermimic-v13-r3-a4200-seed43-20261101'),
                        (5200,ROOT/'artifacts/supermimic/supermimic-v13-r3-a5200-seed43-20261101')]:
        s=json.loads((directory/'summary.json').read_text());path=directory/'trajectory.npz';assert sha(path)==s['trajectory_sha256']
        with np.load(path) as z:
            v=z['values'];layout=s['layout'];a=v[::5,:,slice(*layout['raw_action'])][:100];valid=(v[::5,:,layout['valid'][0]]>.5)[:100]
        mask=valid[1:]&valid[:-1]
        recorded.append(dict(policy=f'A{n}',scope='Frozen seed43 evaluation first1s; not original exploration training samples',
            action_previous_correlation={side:corr(a[1:,:,i][mask],a[:-1,:,i][mask]) for side,i in [('left',2),('right',5)]},
            eligible_pairs=int(mask.sum())))
        hashes[str(path)]=s['trajectory_sha256']
    alignment=[]
    for label,directory in [('A2200',ROOT/'artifacts/supermimic/supermimic-v13-a2200-prediction-20261101-r1/current-recording'),('A3200',ROOT/'artifacts/supermimic/supermimic-v13-world-response-repair-20261101-r1/anchor-seed41')]:
        meta=json.loads((directory/'record.json').read_text());path=directory/'transitions.pt';assert sha(path)==meta['transitions_sha256']
        r=torch.load(path,map_location='cpu',weights_only=False,mmap=True);d=r['data'];live=r['live_before']
        m1=live[1:]&live[:-1]&~d['done'][:-1]
        m2=live[2:]&live[1:-1]&live[:-2]&~d['done'][1:-1]&~d['done'][:-2]
        error1=(d['previous'][1:]-d['action'][:-1])[m1].abs();error2=(d['previous_previous'][2:]-d['action'][:-2])[m2].abs()
        assert error1.max()==0 and error2.max()==0
        m=live.numpy();a=d['action'].numpy();pr=d['previous'].numpy()
        alignment.append(dict(policy=label,seed=meta['seed'],transitions=meta['transitions'],
            prior_action_label_max_abs=float(error1.max()),prior2_action_label_max_abs=float(error2.max()),
            action_previous_correlation={side:corr(a[:,:,i][m],pr[:,:,i][m]) for side,i in [('left',2),('right',5)]},
            scope='Frozen diagnostic recordings, zero exploration; history alignment only, not original V13 minibatch reconstruction'))
        hashes[str(path)]=meta['transitions_sha256']
    cfg=cp['config'];train_count=coverage['train']['environments']
    configuration=dict(buffer_steps=cfg['buffer_steps'],policy_dt_s=.01,per_environment_retention_s=cfg['buffer_steps']*.01,
        train_transition_capacity=cfg['buffer_steps']*train_count,world_window=cfg['window'],actor_horizon=cfg['horizon'],
        checkpoint_contains_replay_buffer=any('buffer' in key or 'replay' in key for key in cp),
        collection_exploration_std=.2,exploration_scope='Source collect default .2*randn_like(action), called without override by train',
        sampling='Uniform environment then uniform valid start range; no delay-stratified sampling; windows masked after first done',
        loss='Mean normalized squared error across 46 coordinates and five substeps; no paired-action or history-invariance term')
    result=dict(delay_coverage=coverage,original_configuration=configuration,evaluation_correlations=recorded,
        recorded_history_alignment=alignment,input_sha256=hashes,source_manifest_sha256=sha(RUN/'source_sha256.json'),
        scope='Original training initial-physics metadata and source, plus separately labeled frozen evaluation/diagnostic samples; no historical training replay available.',
        optimizer_updates=0,simulator_steps=0)
    (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
