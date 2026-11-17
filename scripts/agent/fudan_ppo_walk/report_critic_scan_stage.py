"""Critic扫描阶段对照的离线复核/曲线/正常速度视频；零新物理和学习。"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import inspect_critic_scan_stage as inspect
from . import inspect_regions_stage as regional
from .inspect_randomization_4000 import write, sha


def audit_recipe(out):
    import torch
    from .resume_pd200_random_from3000 import same
    a,b = [json.loads((spec[1]/'budget.json').read_text()) for spec in inspect.ARMS.values()]
    x = json.loads(json.dumps(b['actual_config']))
    assert x.pop('observations') == dict(critic_height_scan=True)
    assert x == a['actual_config']
    assert a['checkpoint'] == b['checkpoint'] and a['checkpoint_sha256'] == b['checkpoint_sha256']
    assert a['runtime_capacity'] == b['runtime_capacity'] == dict(nconmax=64, njmax=256)
    initial = [torch.load(Path(z['log_dir'])/'model_initial.pt', map_location='cpu', weights_only=False) for z in (a,b)]
    keys = ('model_state_dict','optimizer_state_dict','encoder_optimizer_state_dict','torch_rng','cuda_rng',
            'iterations','transitions','primary_updates','encoder_updates','speed_limit','policy_steps')
    for key in keys:
        assert same(initial[0][key], initial[1][key]), key
    write(out/'paired-training-recipe-audit.json', dict(passed=True,
        only_config_change='observations.critic_height_scan False->True',same_source_checkpoint=True,
        same_runtime_capacity=True,all_initial_learning_state_exact=True,compared_learning_state_keys=keys,
        no_inference_critic_forward=True,optimizer_updates=0))


def host_checkpoint(out):
    import torch
    experiment=inspect.ARMS['scan'][1]
    interrupted=json.loads((experiment/'host-interruption.json').read_text())
    budget=json.loads((experiment/'budget.json').read_text())
    path=Path(interrupted['latest_complete_checkpoint'])
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    n=checkpoint['iterations']
    assert n==5838 and not checkpoint['partial_update']
    assert checkpoint['transitions']==n*4096*48+184320
    assert checkpoint['primary_updates']==checkpoint['encoder_updates']==n*20
    assert checkpoint['critic_observation']==budget['critic_observation']
    assert all(torch.isfinite(v).all() for v in checkpoint['model_state_dict'].values())
    for key in ('optimizer_state_dict','encoder_optimizer_state_dict'):
        assert {int(s['step']) for s in checkpoint[key]['state'].values()}=={n*20}
        assert all(torch.isfinite(v).all() for s in checkpoint[key]['state'].values()
                   for v in s.values() if isinstance(v,torch.Tensor))
    report=dict(passed=True,checkpoint=str(path),sha256=sha(path),saved_iteration=n,
        optimizer_steps_each=n*20,finite_learning_state=True,partial_update=False,
        last_event_iteration=interrupted['last_reward_event'],
        completed_but_not_checkpointed_iterations_at_least=interrupted['last_reward_event']-n,
        restored_iteration_gap_to6000=6000-n,
        original1000_budget_completed_events=interrupted['last_reward_event']-5000,
        original1000_budget_not_observed_completed=6000-interrupted['last_reward_event'],
        budget_note='Replaying unsaved61 rounds would consume extra training; do not silently restart162 rounds to6000',
        training_restarted=False,optimizer_updates=0)
    write(out/'host-checkpoint-audit.json',report)
    write(experiment/'host-checkpoint-audit.json',report)


def plot(out, iteration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for arm, (run, _) in inspect.ARMS.items():
        events = EventAccumulator(str(run), size_guidance={'scalars':0}).Reload()
        for ax, tag in zip(axes, ('Train/reward','Train/speed_limit')):
            rows = [r for r in events.Scalars(tag) if r.step <= iteration]
            x,y = np.array([r.step for r in rows]), np.array([r.value for r in rows])
            if tag == 'Train/reward':
                y = np.convolve(y,np.ones(50)/50,mode='valid');x=x[49:]
            ax.plot(x,y,label=arm)
            ax.set(ylabel=tag);ax.grid(alpha=.25);ax.legend()
    axes[-1].set_xlabel('Iteration');fig.tight_layout()
    fig.savefig(out/'training-paired.png',dpi=150);plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    cases = {r['id']:r for r in regional.routes()}
    for row, mode in enumerate(('up','down')):
        for k in range(4):
            sid = f'pyramid_{k}_{mode}_v1';case=cases[sid];ax=axes[row,k]
            for arm in inspect.ARMS:
                with np.load(inspect.terrain_path(out,arm,'terrain',iteration)/'traces'/(sid+'.npz')) as trace:
                    progress=(trace['qpos'][:,0]-case['start_xy'][0])*(1 if mode=='up' else -1)
                    ax.plot(trace['qpos_time'],progress,label=arm)
            ax.axhline(4.,color='black',ls='--',lw=.8)
            ax.axvline(case['drive_s'],color='gray',ls=':',lw=.8)
            ax.set(title=f'{mode} / {case["slope_deg"]:.2f} deg',xlabel='Time (s)',ylabel='Progress (m)')
            ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(out/'terrain-paired.png',dpi=150);plt.close(fig)


def render(out, iteration):
    view=out/'video-inputs';view.mkdir(exist_ok=False)
    labels=(f'无扫描{iteration}',f'77点扫描{iteration}')
    for folder in ('terrain','terrain-height-extra'):
        (view/folder).mkdir()
        for arm,label in zip(inspect.ARMS,labels):
            (view/folder/label).symlink_to(inspect.terrain_path(out,arm,folder,iteration).resolve(),target_is_directory=True)
    regional.render(view,iteration,labels=labels,output_dir=out/'videos')


def summarize_details(out, iteration):
    from scripts.agent.motion.acceptance import compare_baseline
    result=dict(nominal={},extras_pairing={},terrain_by_height={},optimizer_updates=0)
    for arm in inspect.ARMS:
        summary=json.loads((out/arm/'nominal/summary.json').read_text())
        result['nominal'][arm]=dict(groups=summary['groups'],
            zero=[r for r in summary['rows'] if r['category']=='zero'],
            stops=[r for r in summary['rows'] if r['category'].startswith('stop_')])
        for folder in ('terrain','terrain-height-extra'):
            rows=json.loads((inspect.terrain_path(out,arm,folder,iteration)/'result.json').read_text())['rows']
            for r in rows:
                key=f'{arm}_h{r["height"]:g}_{r["family"]}'
                g=result['terrain_by_height'].setdefault(key,dict(total=0,reached=0,passed=0,fell=0))
                g['total']+=1
                for field in ('reached','passed','fell'):g[field]+=int(r[field])
    for label in ('high-yaw','speed3'):
        left,right=[json.loads((out/arm/label/'acceptance.json').read_text()) for arm in inspect.ARMS]
        compare_baseline(right,left)
        result['extras_pairing'][label]=right['comparison']
    write(out/'behavior-details.json',result)
    shutil.copy2(__file__,out/'report-source.py')
    index={}
    for path in out.rglob('*'):
        if (path.is_file() and path.suffix in ('.json','.npz','.png','.mp4','.py','.md','.log','.pt','.mjb','.xml')
                and 'video-inputs' not in path.parts and path.name!='evidence-index.json'):
            index[str(path.relative_to(out))]=sha(path)
    write(out/'evidence-index.json',dict(files_sha256=index,optimizer_updates=0))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('mode',choices=('recipe','host','plot','render','details'))
    p.add_argument('--iteration',type=int,default=5700)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if a.mode=='recipe':audit_recipe(out)
    elif a.mode=='host':host_checkpoint(out)
    elif a.mode=='details':summarize_details(out,a.iteration)
    else:globals()[a.mode](out,a.iteration)


if __name__=='__main__':main()
