"""Analyze archived 500-iteration rewards, curriculum and existing GPU traces; no learning or simulation."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'logs/mjlab/fudan_ppo_step/20261115_upstairs_4096_resume258_to500_04'
REPLAY=ROOT/'artifacts/fudan_ppo_step/20261115-effect500-01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def method(path,classname,name):
    tree=ast.parse(path.read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==classname)
    return next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name)


def reward_fixture(saved):
    from mjlab.utils.lab_api.math import quat_apply_inverse
    base=RUN/'source/ct_mjlab/envs'
    source=ast.parse((base/'fudan_ppo_walk/posture_reward.py').read_text())
    penalty=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='penalties')
    namespace=dict(torch=torch,math=math,LEGACY=dict(version='fudan-orientation-gravity-xy-v1'),
                   quat_apply_inverse=quat_apply_inverse,LEGS=[0,1,3,4])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[penalty],type_ignores=[])),
                 'archived_posture_penalties','exec'),namespace)
    namespace['posture_reward']=SimpleNamespace(penalties=namespace['penalties'])
    classes=[ast.ClassDef(name='Base',bases=[],keywords=[],decorator_list=[],
                body=[method(base/'fudan_ppo_walk/fudan_ppo_walk.py','FudanEnv','reward')]),
             ast.ClassDef(name='Step',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],decorator_list=[],
                body=[method(base/'fudan_ppo_step/env.py','StepEnv','reward')])]
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),
                 'archived_reward_methods','exec'),namespace)
    cases=[('stopped_cmd.3',.3,0.,True,0.,0.,False),
           ('slow_cmd.3',.3,.077,True,0.,0.,False),
           ('tracked_cmd.3',.3,.3,True,0.,0.,False),
           ('stopped_cmd.5',.5,0.,True,0.,0.,False),
           ('slow_cmd.5',.5,.158,True,0.,0.,False),
           ('tracked_cmd.5',.5,.5,True,0.,0.,False),
           ('lost_double_support',.3,.3,False,0.,0.,False),
           ('roll5_and_lost_support',.3,.3,False,5.,0.,False),
           ('roll10_and_lost_support',.3,.3,False,10.,0.,False),
           ('pitch10_and_lost_support',.3,.3,False,0.,10.,False),
           ('nonwheel_contact',.3,.3,False,0.,0.,True)]
    n=len(cases);env=namespace['Step']();env.dt=.01
    env.cfg=SimpleNamespace(rewards=SimpleNamespace(**saved['training_recipe']['rewards']),
                            step=SimpleNamespace(**saved['training_recipe']['step']))
    env.commands=torch.tensor([[c[1],0.,.35] for c in cases]);env.linear=torch.zeros(n,3)
    env.linear[:,0]=torch.tensor([c[2] for c in cases]);env.linear.requires_grad_()
    env.angular=torch.zeros(n,3);env.height=torch.full((n,),.35)
    env.position=torch.tensor([[1.2,0.,.35]]*n);env.modes=torch.ones(n,dtype=torch.long)
    env.normal_support=torch.tensor([c[3] for c in cases]);env.ray_failed=torch.zeros(n,dtype=torch.bool)
    env.gravity=torch.tensor([[math.sin(math.radians(c[5])),
        -math.sin(math.radians(c[4]))*math.cos(math.radians(c[5])),
        -math.cos(math.radians(c[4]))*math.cos(math.radians(c[5]))] for c in cases])
    env.posture_reward_profile=saved['posture_reward']
    env.contacts=SimpleNamespace(step_contact=torch.tensor([c[6] for c in cases]))
    env.wheel_ids=env.hip_ids=[0,1];env.front_q=list(range(4))
    env.front_limits=torch.tensor([[-3.,3.]]*4)
    env.robot=SimpleNamespace(data=SimpleNamespace(root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]*n)))
    data=SimpleNamespace(xpos=torch.tensor([[[0.,.1,.06],[0.,-.1,.06]]]*n),
        xanchor=torch.tensor([[[0.,.1,.35],[0.,-.1,.35]]]*n),qpos=torch.zeros(n,4))
    env.env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(n,3)),sim=SimpleNamespace(data=data))
    env.action=SimpleNamespace(raw_action=torch.zeros(n,6),torques=torch.zeros(n,6))
    env.velocity=env.acceleration=torch.zeros(n,6);env.previous=torch.zeros(n,6,2);env.sums=torch.zeros(n,2)
    total,terms=env.reward()
    gradient=torch.autograd.grad(terms['tracking_lin_vel'].sum(),env.linear,retain_graph=True)[0][:,0]
    rows=[dict(case=case[0],command_vx=case[1],actual_vx=case[2],
        terms={key:float(value[i].detach()) for key,value in terms.items()},total=float(total[i].detach()),
        main_tracking_velocity_gradient=float(gradient[i])) for i,case in enumerate(cases)]
    assert rows[0]['terms']['tracking_lin_vel']==rows[2]['terms']['tracking_lin_vel']
    assert rows[0]['main_tracking_velocity_gradient']==rows[1]['main_tracking_velocity_gradient']==0.
    scales=saved['training_recipe']['rewards']['scales'];sigma=saved['training_recipe']['rewards']['tracking_sigma']
    plateau=math.sqrt(sigma*math.log(scales['tracking_lin_vel']))
    return dict(rows=rows,tracking_error_plateau_m_s=plateau,
        command_fraction_allowing_zero_main_gradient_from_standstill=(plateau-.2)/(.5-.2),
        reward_method_executed_from_verified_training_snapshot=True,
        scope='Synthetic reward decomposition only, all omitted motion/control costs zero; no feasible-motion or causal-training claim.')


def figure(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    v=np.linspace(-.1,.7,801)
    raw=.015*np.exp(-(.3-v)**2/.25)
    clipped=np.clip(raw,-.01,.01)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(v,raw,'--',label='Before cap')
    axes[0].plot(v,clipped,label='Actual training main velocity reward')
    axes[0].scatter([0.,.077,.3],[.01]*3,color='red',zorder=3)
    axes[0].set(xlabel='Actual vx (m/s)',ylabel='Reward per10ms',title='Command vx=0.30m/s: the main reward is flat')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.3)
    indices=[0,1,2,6,7,8]
    rows=[result['reward']['rows'][i] for i in indices]
    labels=['Stand still','Move at0.077m/s','Track0.30m/s','Lose double support','Lose support + roll5deg','Lose support + roll10deg']
    axes[1].barh(labels,[r['total'] for r in rows]);axes[1].invert_yaxis()
    axes[1].set(xlabel='Synthetic reward per10ms',title='Other motion/control costs held at zero')
    axes[1].grid(axis='x',alpha=.3);fig.tight_layout();fig.savefig(output/'reward-conflict.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    manifest=json.loads((RUN/'source_sha256.json').read_text());checks={}
    for name,digest in manifest.items():
        if name.startswith('mjlab_training/src/ct_mjlab/'):
            path=RUN/'source/ct_mjlab'/name.removeprefix('mjlab_training/src/ct_mjlab/')
            checks[name]=sha(path)==digest
    assert all(checks.values())
    saved=torch.load(RUN/'model_final.pt',map_location='cpu',weights_only=False)
    assert saved['iterations']==500 and saved['training_recipe']['step']['reverse_probability']==.75
    result=dict(checkpoint_sha256=sha(RUN/'model_final.pt'),snapshot_checks=checks,
        reward=reward_fixture(saved),recipe=saved['training_recipe'],physics_steps=0,optimizer_updates=0)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events=EventAccumulator(str(RUN),size_guidance={'scalars':0}).Reload()
    result['scalars']={}
    for tag in events.Tags()['scalars']:
        data=events.Scalars(tag)
        if tag=='Train/reward':assert [r.step for r in data]==list(range(1,501))
        windows={}
        for name,lo,hi in [('first50',1,50),('pre_resize',201,245),('last50',451,500)]:
            values=[r.value for r in data if lo<=r.step<=hi]
            if values:windows[name]=dict(mean=float(np.mean(values)),min=min(values),max=max(values))
        result['scalars'][tag]=windows
    episodes=[]
    for run in sorted(RUN.parent.glob('20261115_upstairs_*')):
        file=run/'episodes.jsonl'
        if file.exists():episodes.extend(json.loads(line) for line in file.open())
    result['curriculum']=[]
    for window,lo in [('all',0),('last50',451)]:
        for mode in (1,2):
            for level in range(10):
                rows=[r for r in episodes if r['iteration']>=lo and r['mode']==mode and r['level']==level]
                if not rows:continue
                result['curriculum'].append(dict(window=window,mode=('flat','up','down')[mode],
                    height_m=saved['training_recipe']['step']['heights'][level],episodes=len(rows),
                    crossed=sum(r['crossed'] for r in rows),success=sum(r['success'] for r in rows),
                    reverse_counts=dict(Counter(str(r['reverse']) for r in rows))))
    result['representative_replays']=[]
    for group in ('baseline-h30','candidate-h30','baseline','candidate'):
        protocol=json.loads((REPLAY/group/'protocol.json').read_text())
        for row in json.loads((REPLAY/group/'metrics.json').read_text()):
            if row['angle']==0 and row['speed']==.3 and row['height'] in (.02,.2):
                result['representative_replays'].append(dict(group=group,command_height=protocol['command_height_m'],**row))
    source=torch.load(saved['transfer']['path'],map_location='cpu',weights_only=False)
    result['std_before_after']=[source['model_state_dict']['std'].tolist(),saved['model_state_dict']['std'].tolist()]
    result['discount']=dict(gamma=saved['config']['gamma'],policy_dt=.01,
        weight_at2s=saved['config']['gamma']**200,weight_at4s=saved['config']['gamma']**400)
    write(args.output/'analysis.json',result)
    figure(result,args.output)
    (args.output/'source_analysis.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(plateau_m_s=result['reward']['tracking_error_plateau_m_s'],
        snapshot_verified=True,reward_fixture_cases=len(result['reward']['rows']),physics_steps=0,optimizer_updates=0),ensure_ascii=False))


if __name__=='__main__':
    main()
