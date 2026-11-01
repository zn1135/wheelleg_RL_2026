"""Local wheel finite differences with restored native GPU physics snapshots.

Branches execute only the existing PD and five physical substeps. Native collect
after restoration verifies the baseline path. No training or policy updates.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import warp as wp

from ct_mjlab.envs import task_registry
from ct_mjlab.supermimic.__main__ import collect
from ct_mjlab.supermimic.analytic_reward import capture, differentiable_reward
from ct_mjlab.supermimic.buffer import ReplayBuffer
from ct_mjlab.supermimic.learner import Learner
from ct_mjlab.supermimic.networks import act, RunningNorm
from ct_mjlab.supermimic.state import capture_state, capture_parameters, StateAdapter, NOISE
from scripts.agent.supermimic.record_supermimic_physical_prediction import ObservedWalkRewardState
from scripts.agent.supermimic.assess_chuanliantui_supermimic import verify_training_sources, actor_digest

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


class Snapshot:
    def __init__(self, raw):
        self.arrays=[]
        visited=set()
        def walk(obj,depth=0):
            if id(obj) in visited or depth>3:return
            visited.add(id(obj))
            for value in vars(obj).values():
                if isinstance(value,wp.array) and value.size:
                    self.arrays.append((value,wp.clone(value)))
                elif hasattr(value,'__dict__') and type(value).__module__.startswith('mujoco_warp'):
                    walk(value,depth+1)
        walk(raw.sim.wp_data)
        pd=raw.action_manager.get_term('pd');entity=raw.scene['robot']
        self.tensors=[];self.scalars=[]
        for obj in (pd,entity.data,*entity._actuators):
            for key,value in vars(obj).items():
                if isinstance(value,torch.Tensor):self.tensors.append((value,value.clone()))
                elif type(value) in (int,float,bool):self.scalars.append((obj,key,value))
        self.cpu_rng=torch.get_rng_state();self.gpu_rng=torch.cuda.get_rng_state_all()
        assert self.arrays

    def restore(self):
        for dst,src in self.arrays:wp.copy(dst,src)
        for dst,src in self.tensors:dst.copy_(src)
        for obj,key,value in self.scalars:setattr(obj,key,value)
        torch.set_rng_state(self.cpu_rng);torch.cuda.set_rng_state_all(self.gpu_rng)


def physical_branch(raw, action, motor_v):
    pd=raw.action_manager.get_term('pd');pd.process_actions(action)
    path=[]
    for _ in range(5):
        pd.apply_actions();raw.scene.write_data_to_sim();raw.sim.step()
        raw.scene['robot'].update(raw.physics_dt)
        q=raw.sim.data.qpos.clone();q[:,:3]-=raw.scene.env_origins
        # Contact is excluded here; probes occur before base contact, verified below.
        path.append(torch.cat((q,raw.sim.data.qvel.clone(),raw.sim.data.qfrc_actuator[:,motor_v].clone(),torch.zeros(raw.num_envs,9,device=raw.device)),-1))
    return torch.stack(path,1)


def outputs(path, features, reward):
    return torch.stack((features[:,0],path[:,-1,2],features[:,9],reward),-1)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--history-only',action='store_true',help='Zero-delay negative control: perturb saved wheel FIFO and model previous-action input, holding current action fixed')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    checkpoint=ROOT/'logs/mjlab/chuanliantui_supermimic_walk/20261101_v13_continuation_to5200_r3/final.pt'
    verify_training_sources(checkpoint,args.output,'supermimic')
    cfg,train=task_registry.get_cfgs('supermimic',play=True);cfg.auto_reset=False;cfg.seed=43
    env,_=task_registry.make_env('supermimic',env_cfg=cfg,train_cfg=train,num_envs=96,seed=43,device='cuda:0')
    raw=env.unwrapped
    try:
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends(allow_tf32=False)
        learner=Learner('cuda:0');command=raw.command_manager.get_term('standup')
        learner.load(checkpoint,command,inference=True)
        # Inference load intentionally loads actor only; explicitly load world for diagnostics.
        saved=torch.load(checkpoint,map_location='cuda:0',weights_only=False)
        learner.world.load_state_dict(saved['world'],strict=True)
        for name in ('actor','world','critic'):getattr(learner,name).eval().requires_grad_(False)
        before={n:actor_digest(getattr(learner,n)) for n in ('actor','world')}
        raw._split_walk_rewards=ObservedWalkRewardState(raw)
        command.speed_curriculum_frozen=True;raw.reset(seed=43)
        def commands():
            command.tracking_episode.fill_(True);command.zero_hold.zero_();command.tracking_velocity.zero_();command.tracking_yaw_rate.zero_()
            command.tracking_interval_steps.fill_(1000000);command.tracking_acceleration[:48]=.5;command.tracking_acceleration[48:]=2.
            command.episode_height.fill_(.26);command._apply_height(None)
        commands();raw.observation_manager.reset();raw.obs_buf=raw.observation_manager.compute(update_history=True);obs=raw.get_observations()
        buffer=ReplayBuffer(2,96,'cuda:0');adapter=StateAdapter().cuda().eval()
        motor_v=adapter.motor_v-25
        samples=[]
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.optim.Adam,'step',side_effect=AssertionError('No training')))
            stack.enter_context(patch.object(RunningNorm,'update',side_effect=AssertionError('No norm update')))
            for tick in range(61):
                commands();sample=None
                if tick in (20,40,60):
                    state=capture_state(raw);params=capture_parameters(raw);a=act(learner.actor,obs['history'])
                    record=capture(raw,raw._split_walk_rewards);context=record['context']
                    previous=raw.action_manager.action.clone();previous_previous=raw.action_manager.prev_action.clone()
                    snapshot=Snapshot(raw);initial_q=raw.sim.data.qpos.clone();initial_v=raw.sim.data.qvel.clone()
                    branches={};predicted={};paths={}
                    for label,wheel,delta in [('base',2,0.),('repeat',2,0.)]+[(f'{wheel}_{delta:+.2f}',wheel,delta) for wheel in (2,5) for delta in (-.1,-.05,.05,.1)]:
                        snapshot.restore()
                        assert torch.equal(raw.sim.data.qpos,initial_q) and torch.equal(raw.sim.data.qvel,initial_v)
                        assert torch.equal(capture_state(raw),state)
                        action=a.clone();model_previous=previous.clone()
                        if args.history_only:
                            assert (params[:,24:26]==0).all()
                            assert raw.action_manager.get_term('pd').cfg.motor_response_time_constant_s==0
                            # Zero-delay PD overwrites/uses the current action every substep.
                            raw.action_manager.get_term('pd')._action_history[:,:,wheel].add_(delta)
                            model_previous[:,wheel]+=delta
                        else:
                            action[:,wheel]+=delta
                        path=physical_branch(raw,action,motor_v);feature=adapter.reward_features(path)
                        _,reward=differentiable_reward(feature,action,previous,previous_previous,context)
                        paths[label]=path.cpu();branches[label]=outputs(path,feature,reward).cpu()
                        result=learner.world.step(state,params,action,obs['history'][:,-19:-16],context,model_previous,previous_previous,state[:,NOISE],return_path=True)
                        predicted[label]=outputs(result[-1],result[4],result[2]).cpu()
                    snapshot.restore()
                    assert torch.equal(capture_state(raw),state)
                    sample=dict(time_s=tick*.01,state=state.cpu(),parameters=params.cpu(),action=a.cpu(),context=context.cpu(),previous=previous.cpu(),previous_previous=previous_previous.cpu(),paths=paths,true_outputs=branches,predicted_outputs=predicted,snapshot_arrays=len(snapshot.arrays))
                    del snapshot
                obs,reward,done=collect(env,learner,buffer,obs,exploration=0.)
                assert not done.any()
                if sample is not None:
                    index=(buffer.ptr-1)%buffer.capacity
                    native=buffer.data['path'][index].reshape(96,5,64)
                    feat=buffer.data['features'][index]
                    # Reward context is held fixed to isolate action response, not command updates.
                    _,fixed_reward=differentiable_reward(feat,sample['action'].cuda(),sample['previous'].cuda(),sample['previous_previous'].cuda(),sample['context'].cuda())
                    sample['native_outputs']=outputs(native,feat,fixed_reward).cpu()
                    sample['native_path']=native.cpu();sample['native_features']=feat.cpu()
                    sample['native_reward_context']=buffer.data['context'][index].cpu()
                    sample['context_flags_equal']=torch.equal(sample['context'][:,3:7],sample['native_reward_context'][:,3:7])
                    sample['base_contact_max_N']=float(feat[:,28].max())
                    # True-state prediction compares the same no-contact phase.
                    assert sample['base_contact_max_N'] <= 1e-6
                    assert sample['context_flags_equal']
                    samples.append(sample)
                    print(json.dumps(dict(time_s=sample['time_s'],snapshot_arrays=sample['snapshot_arrays'],base_contact_max_N=sample['base_contact_max_N'],native_path_motion_max_abs=float((native.cpu()[...,:49]-sample['paths']['base'][...,:49]).abs().max()))),flush=True)
        assert before=={n:actor_digest(getattr(learner,n)) for n in before}
        torch.save(dict(samples=samples,intervention='previous_wheel_history_only' if args.history_only else 'current_wheel_action',metrics=['LINK_vx_m_s','root_height_m','reward_height_m','reward_per_step'],checkpoint_sha256=sha(checkpoint),parameter_updates=0,normalization_updates=0,script_sha256=sha(Path(__file__))),args.output/'response.pt')
        print(json.dumps(dict(completed=True,samples=len(samples),output=str(args.output))),flush=True)
    finally:env.close()


if __name__=='__main__':main()
