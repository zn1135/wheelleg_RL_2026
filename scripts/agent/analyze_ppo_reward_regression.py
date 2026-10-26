#!/usr/bin/env python3
"""Read saved PPO/BC trajectories and checkpoints; no simulation or optimization."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forward(state,observations):
    x=observations
    for i in (0,2,4,6):
        x=F.linear(x,state[f'mlp.{i}.weight'],state[f'mlp.{i}.bias'])
        if i!=6:x=F.elu(x)
    return x


@torch.inference_mode()
def analyze(args):
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    groups={};inputs={};dt=.01
    for group,path in [('bc',args.baseline),('taskonly',args.run/'taskonly/evaluation'),('smp',args.run/'smp/evaluation')]:
        report=json.loads((path/'result.json').read_text());shares=json.loads((path/'reward-shares.json').read_text())
        assert digest(path/'trace.npz')==report['trace_sha256']
        assert digest(path/'rewards.npz')==shares['rewards_sha256']
        inputs[str(path/'trace.npz')]=report['trace_sha256'];inputs[str(path/'rewards.npz')]=shares['rewards_sha256']
        with np.load(path/'rewards.npz') as d:values=d['contributions'].astype(np.float64);names=d['names'].tolist()
        with np.load(path/'trace.npz') as d:
            alive=d['alive'];features=d['features'];observations=d['observation']
            if group=='bc':bc_obs=torch.from_numpy(observations[::10].reshape(-1,39).copy())
            assert alive.shape==(2000,40) and alive.all()
            assert not observations[:,:,:2].any()
            # 独立公式核对两个最大惩罚，避免仅凭日志标题判断含义。
            expected={'flat_orientation_y_v':-32*features[:,:,1]**2*dt,
                      'stand_still_lin_vel':-2*np.abs(features[:,:,4:6]).sum(-1)*dt}
            formula_error={}
            for key,wanted in expected.items():
                actual=values[:,:,names.index(key)]
                np.testing.assert_allclose(actual,wanted,atol=2e-6,rtol=2e-5)
                formula_error[key]=float(np.max(np.abs(actual-wanted)))
            late=features[1500:]
            motion=dict(mean_abs_pitch_deg=float(np.abs(np.rad2deg(np.arcsin(late[:,:,1].clip(-1,1)))).mean()),
                parking_gravity_gate_fraction=float((late[:,:,3]<-.9).mean()),
                mean_speed_m_s=float(np.linalg.norm(late[:,:,4:6],axis=-1).mean()))
        segments={}
        for label,part in [('all20s',values),('rise',values[20:200]),('late5s',values[1500:])]:
            rates=part.mean((0,1))/dt;contributions=dict(zip(names,map(float,rates)))
            negative=float(-rates[rates<0].sum());smp=contributions['smp_motion']
            segments[label]=dict(weighted_reward_per_second=contributions,total=float(rates.sum()),
                task_only=float(rates.sum()-smp),negative_total=negative,
                pitch_fraction_of_penalties=-contributions['flat_orientation_y_v']/negative,
                absolute_rank=sorted(contributions,key=lambda k:abs(contributions[k]),reverse=True))
        groups[group]=dict(segments=segments,late_motion=motion,formula_error=formula_error)
    for group in ('taskonly','smp'):
        directory=Path(json.loads((args.run/group/'run.json').read_text())['run_dir'])
        logs=json.loads((args.run/group/'budget-check.json').read_text())['metrics']
        groups[group]['training_episode_reward_log']={field:sorted(
            ((k.removeprefix('Episode_Reward/'),v[field]) for k,v in logs.items() if k.startswith('Episode_Reward/')),
            key=lambda pair:abs(pair[1]),reverse=True) for field in ('first20_mean','last20_mean')}
        first_path=args.run/group/'initial.pt';initial=torch.load(first_path,map_location='cpu',weights_only=False)['actor_state_dict']
        reference=forward(initial,bc_obs);std0=initial['distribution.std_param'];changes=[]
        for path in sorted(directory.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1])):
            saved=torch.load(path,map_location='cpu',weights_only=False);state=saved['actor_state_dict'];delta=forward(state,bc_obs)-reference
            std=state['distribution.std_param']
            kl=(torch.log(std/std0)+(std0**2+delta**2)/(2*std**2)-.5).sum(-1)
            changes.append(dict(iterations=saved['iter']+1,sha256=digest(path),
                action_mean_rms_change=float(delta.square().mean().sqrt()),
                per_motor_rms_change=delta.square().mean(0).sqrt().tolist(),std=std.tolist(),
                mean_bc_to_checkpoint_kl=float(kl.mean())))
        groups[group]['checkpoint_action_changes_on_fixed_bc_states']=changes
    result=dict(groups=groups,inputs=inputs,script_sha256=digest(__file__),optimizer_updates=0,
        simulated_steps=0,dt=dt,parking_velocity_factor={str(v):math.exp(-(v/.005)**2) for v in (.005,.01,.02,.05)},
        first_rollout=dict(steps=24,zero_torque_steps=20,active_torque_steps=4,
            scope='Source-derived timing; gated steps still have action history/smoothness effects and enter PPO storage.'),
        scope='Saved nominal40 reward contributions, training log magnitudes, and actor drift on fixed BC states; no per-term policy gradient or causal ablation.')
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,r in groups.items():
        late=r['segments']['late5s'];print(name,'task/s:',late['task_only'],'pitch penalty share:',late['pitch_fraction_of_penalties'])
        print('top terms:',[(k,late['weighted_reward_per_second'][k]) for k in late['absolute_rank'][:6]])
        if name!='bc':print('first update:',r['checkpoint_action_changes_on_fixed_bc_states'][0])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','baseline','output'):p.add_argument('--'+name,type=Path,required=True)
    analyze(p.parse_args())
