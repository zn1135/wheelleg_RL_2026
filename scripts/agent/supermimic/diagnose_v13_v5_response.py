"""Frozen endpoint-model wheel responses from restored native simulator states."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from ct_mjlab.envs import task_registry
from ct_mjlab.supermimic.__main__ import collect
from ct_mjlab.supermimic.analytic_reward import capture
from ct_mjlab.supermimic.buffer import ReplayBuffer
from ct_mjlab.supermimic.learner import Learner,ALGORITHM_CONTRACT
from ct_mjlab.supermimic.motion import WorldModel
from ct_mjlab.supermimic.networks import act,RunningNorm
from ct_mjlab.supermimic.state import capture_state,capture_parameters,NOISE
from scripts.agent.supermimic.diagnose_v13_wheel_response import Snapshot,physical_branch
from scripts.agent.supermimic.assess_chuanliantui_supermimic import make_cases,actor_digest,verify_training_sources
from scripts.agent.supermimic.record_supermimic_physical_prediction import ObservedWalkRewardState

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def outputs(world,path):
    end=path[:,-1];q=end[:,3:7]
    pitch=torch.asin((2*(q[:,0]*q[:,2]-q[:,3]*q[:,1])).clamp(-1,1))
    return torch.stack((world.adapter.reward_features(path)[:,0],end[:,2],pitch,
        end[:,world.adapter.motor_v[2]],end[:,world.adapter.motor_v[5]]),-1)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=43);p.add_argument('--num-envs',type=int,default=96)
    p.add_argument('--times',type=float,nargs='+',default=[.4,.8,3.,10.,20.])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    assert args.num_envs%6==0
    checkpoint=args.run/'final.pt';verify_training_sources(checkpoint,args.output,'supermimic')
    cfg,train=task_registry.get_cfgs('supermimic',play=True);cfg.auto_reset=False;cfg.seed=args.seed
    env,_=task_registry.make_env('supermimic',env_cfg=cfg,train_cfg=train,num_envs=args.num_envs,seed=args.seed,device='cuda:0')
    raw=env.unwrapped
    try:
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends(allow_tf32=False)
        command=raw.command_manager.get_term('standup');learner=Learner('cuda:0')
        learner.load(checkpoint,command,inference=True);learner.actor.eval().requires_grad_(False)
        worlds={};before={};hashes={}
        for label in ('prewarm','final'):
            path=args.run/f'{label}.pt';saved=torch.load(path,map_location='cuda:0',weights_only=False)
            assert saved['algorithm_contract']==ALGORITHM_CONTRACT
            model=WorldModel().cuda().eval().requires_grad_(False);model.load_state_dict(saved['world'])
            worlds[label]=model;before[label]=actor_digest(model);hashes[label]=sha(path)
        actor_before=actor_digest(learner.actor)
        raw._split_walk_rewards=ObservedWalkRewardState(raw);command.speed_curriculum_frozen=True;raw.reset(seed=args.seed)
        targets=torch.empty(args.num_envs,device='cuda:0');acc=torch.empty_like(targets)
        for case in make_cases(args.num_envs):targets[case['env_ids']]=case['speed'];acc[case['env_ids']]=case['acceleration']
        def commands(t):
            command.tracking_episode.fill_(True);command.zero_hold.zero_();command.tracking_yaw_rate.zero_()
            command.tracking_velocity.copy_(targets if 2<=t<16 else torch.zeros_like(targets))
            command.tracking_interval_steps.fill_(1000000);command.tracking_acceleration.copy_(acc)
            command.episode_height.fill_(.26);command._apply_height(None)
        commands(0);raw.observation_manager.reset();raw.obs_buf=raw.observation_manager.compute(update_history=True);obs=raw.get_observations()
        buffer=ReplayBuffer(32,args.num_envs,'cuda:0');samples=[];rows=[]
        sample_ticks={round(t/.01) for t in args.times}
        with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No fitting authorized')),patch.object(RunningNorm,'update',side_effect=AssertionError('No norm updates')):
            for tick in range(max(sample_ticks)+1):
                commands(tick*.01);sample=None
                if tick in sample_ticks:
                    state=capture_state(raw);params=capture_parameters(raw);action=act(learner.actor,obs['history'])
                    context=capture(raw,raw._split_walk_rewards)['context']
                    previous=raw.action_manager.action.clone();previous2=raw.action_manager.prev_action.clone()
                    snapshot=Snapshot(raw);paths={};truth={};pred={label:{} for label in worlds};actions={}
                    for name,wheel,delta in [('base',2,0.),('repeat',2,0.)]+[(f'{w}_{d:+.2f}',w,d) for w in (2,5) for d in (-.1,-.05,.05,.1)]:
                        snapshot.restore();assert torch.equal(capture_state(raw),state)
                        a=action.clone();a[:,wheel]=(a[:,wheel]+delta).clamp(-3,3)
                        path=physical_branch(raw,a,worlds['final'].adapter.motor_v-25)
                        paths[name]=path.cpu();truth[name]=outputs(worlds['final'],path).cpu();actions[name]=a.cpu()
                        for label,world in worlds.items():
                            result=world.step(state,params,a,context[:,:3],context,previous,previous2,state[:,NOISE],return_path=True)
                            pred[label][name]=outputs(world,result[-1]).cpu()
                    snapshot.restore();assert torch.equal(capture_state(raw),state)
                    sample=dict(time_s=tick*.01,state=state.cpu(),parameters=params.cpu(),action=action.cpu(),context=context.cpu(),
                        previous=previous.cpu(),previous_previous=previous2.cpu(),actions=actions,paths=paths,truth=truth,predictions=pred)
                    del snapshot
                obs,_,done=collect(env,learner,buffer,obs,exploration=0.)
                assert not done.any(),'Stop rather than collect reset states'
                if sample is not None:
                    native=buffer.data['path'][(buffer.ptr-1)%buffer.capacity].reshape(args.num_envs,5,64)
                    sample['native_path']=native.cpu();native_out=outputs(worlds['final'],native).cpu()
                    noise=torch.maximum((sample['truth']['base']-sample['truth']['repeat']).abs(),(sample['truth']['base']-native_out).abs())
                    sample['native_outputs']=native_out;sample['noise']=noise
                    # Keep real replay windows for a later explicitly budgeted repair, not synthetic history edits.
                    sample['anchor_windows']=buffer.sample(32,24) if buffer.size>=24 else None
                    if sample['anchor_windows'] is not None:sample['anchor_windows']={k:v.cpu() for k,v in sample['anchor_windows'].items()}
                    for label in worlds:
                        for wheel in (2,5):
                            for eps in (.05,.1):
                                real=sample['truth'][f'{wheel}_{eps:+.2f}']-sample['truth'][f'{wheel}_{-eps:+.2f}']
                                predicted=sample['predictions'][label][f'{wheel}_{eps:+.2f}']-sample['predictions'][label][f'{wheel}_{-eps:+.2f}']
                                interior=(sample['action'][:,wheel].abs()+eps<3)
                                floor=torch.tensor([1e-5,1e-6,1e-5,1e-3,1e-3])
                                eligible=(real.abs()>torch.maximum(5*noise,floor))&interior[:,None]
                                detectable=eligible&(predicted.abs()>floor)
                                for j,metric in enumerate(('LINK_vx','height','pitch_rad','left_wheel_speed','right_wheel_speed')):
                                    rows.append(dict(model=label,time_s=sample['time_s'],wheel=wheel,epsilon=eps,metric=metric,total=args.num_envs,
                                        eligible=int(eligible[:,j].sum()),both_detectable=int(detectable[:,j].sum()),
                                        same_direction=int(((real[:,j]*predicted[:,j]>0)&detectable[:,j]).sum()),
                                        opposite_direction=int(((real[:,j]*predicted[:,j]<0)&detectable[:,j]).sum()),
                                        true_delta_mean=float(real[:,j].mean()),predicted_delta_mean=float(predicted[:,j].mean()),noise_max=float(noise[:,j].max())))
                    samples.append(sample);print(json.dumps(dict(time_s=sample['time_s'],noise_max=noise.max(0).values.tolist())),flush=True)
        assert actor_before==actor_digest(learner.actor)
        assert before=={label:actor_digest(w) for label,w in worlds.items()}
        torch.save(dict(schema='v13-v5-frozen-response-v1',samples=samples,checkpoint_hashes=hashes,
            actor_checkpoint=str(checkpoint.resolve()),actor_sha256=actor_before,seed=args.seed,num_envs=args.num_envs,
            metric_names=['LINK_vx','height','pitch_rad','left_wheel_speed','right_wheel_speed'],updates=0),args.output/'response.pt')
        report=dict(rows=rows,seed=args.seed,parameter_updates=0,norm_updates=0,actor_frozen=True,worlds_frozen=True,
            checkpoint_hashes=hashes,recording_sha256=sha(args.output/'response.pt'),script_sha256=sha(Path(__file__)),
            scope='Local10ms central finite differences, restored same state, nominal physics; not full policy objective causality')
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(completed=True,output=str(args.output)),ensure_ascii=False))
    finally:env.close()


if __name__=='__main__':main()
