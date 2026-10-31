#!/usr/bin/env python3
"""Compare frozen walk policies under identical fixed requests; never learn."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sample_physics(env):
    recorder = getattr(env, '_walk_diagnosis', None)
    if recorder is not None:
        recorder.sample()
    return torch.zeros(env.num_envs, device=env.device)


class Recorder:
    def __init__(self, raw, steps):
        self.raw, self.pd = raw, raw.action_manager.get_term('pd')
        self.robot = raw.scene['robot']
        self.state = raw.command_manager.get_term('standup')
        self.live = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
        self.count, self.layout = 0, {}
        self.trace = None
        self.steps = steps
        self.original_apply = self.pd.apply_actions
        self.hips = self.robot.find_bodies(('lf0','rf0'), preserve_order=True)[0]
        self.wheels = self.robot.find_bodies(('lfwheel','rfwheel'), preserve_order=True)[0]
        names = ('lf0','lf00','rf0','rf00','lf1','rf1','lfwheel','rfwheel')
        ids = self.robot.find_joints(names, preserve_order=True)[0]
        self.qadr = self.robot.indexing.joint_q_adr[ids]
        self.vadr = self.robot.indexing.joint_v_adr[ids]
        self.joint_names = names

    def apply(self):
        pd = self.pd
        self.original_apply()
        delayed = pd._action_history[(pd._history_head-pd.action_delay_steps)
            % len(pd._action_history), pd._env_ids]
        legs = pd._leg_action_ids
        desired = pd._nominal_pos[:, legs] + pd.cfg.leg_position_scale*delayed[:, legs]
        target = desired.clamp(pd._reference_pos[:, legs]-3.14, pd._reference_pos[:, legs]+3.14)
        self.pending = dict(leg_target=target.clone(), leg_unclipped=desired.clone(),
            pd_torque=pd._torques.clone(), applied_command=self.state.command.clone())

    def sample(self):
        from mjlab.utils.lab_api.math import quat_apply_inverse
        pd, raw, robot = self.pd, self.raw, self.robot
        data = robot.data
        force = raw.scene['wheel_contact'].data.force
        support = -force[:,:,2]
        relative = data.body_link_pos_w[:,self.wheels]-data.body_link_pos_w[:,self.hips]
        quat = data.root_link_quat_w[:,None,:].expand(-1,2,-1)
        relative = quat_apply_inverse(quat.reshape(-1,4),relative.reshape(-1,3)).reshape(-1,2,3)
        root = raw.sim.data.qpos[:,robot.indexing.free_joint_q_adr]
        root_velocity = raw.sim.data.qvel[:,robot.indexing.free_joint_v_adr]
        qw,qx,qy,qz = root[:,3:7].unbind(-1)
        roll = torch.atan2(2.*(qw*qx+qy*qz), 1.-2.*(qx.square()+qy.square()))
        pitch = torch.asin((2.*(qw*qy-qz*qx)).clamp(-1.,1.))
        height = root[:,2]-raw.scene.env_origins[:,2]
        gz = (-1.+2.*(root[:,4].square()+root[:,5].square())).clamp(-1.,1.)
        base = raw.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
        fields = dict(valid=self.live[:,None].float(), height=height[:,None],
            tilt=torch.acos(-gz)[:,None].rad2deg(), base_contact=base[:,None],
            true_velocity=data.root_link_lin_vel_b, support=support,
            roll_pitch=torch.stack((roll,pitch),-1).rad2deg(),
            angular_velocity_xy=root_velocity[:,3:5],
            root_linear_velocity_w=root_velocity[:,:3],
            swing=torch.atan2(-relative[...,0],-relative[...,2]).rad2deg(),
            leg_length=relative.norm(dim=-1), joint_q=raw.sim.data.qpos[:,self.qadr],
            joint_dq=raw.sim.data.qvel[:,self.vadr],
            actual_torque=raw.sim.data.qfrc_actuator[:,pd._motor_v_adr],
            friction_ratio=force[:,:,:2].norm(dim=-1)/support.clamp_min(5.),
            down_ticks=self.state.walk_down_ticks[:,None].float(),
            encoder_velocity=self.encoder, true_velocity_before=self.true_before,
            raw_action=self.action, **self.pending)
        # 专用反馈对照按策略拍锁存输入，便于离线核对唯一干预。
        for name in ('policy_velocity','policy_wheel_velocity'):
            if hasattr(self,name):
                fields[name] = getattr(self,name)
        if self.trace is None:
            start = 0
            for name,value in fields.items():
                self.layout[name] = [start,start+value.shape[1]]
                start += value.shape[1]
            self.trace = torch.empty(self.steps,raw.num_envs,start,device=raw.device)
        self.trace[self.count] = torch.cat(tuple(fields.values()),-1)
        self.count += 1


def summarize(values, layout, cases, ends, dt, seconds, warmup, weight):
    def get(name):
        lo,hi = layout[name]
        return values[...,lo:hi]
    times = (np.arange(len(values))+1)*dt
    valid = get('valid')[...,0]>.5
    rows, events = [], []
    for case in cases:
        ids = np.asarray(case['env_ids'])
        survive = np.asarray(ends)[ids] >= seconds-1e-6
        full = warmup+abs(case['speed'])/case['acceleration']+.5
        mask = valid[:,ids] & (times[:,None]>=full)
        last = mask & (times[:,None]>=seconds-2.) & survive[None,:]
        vx = get('true_velocity')[:,ids,0]
        row = dict(case=case['name'], speed=case['speed'], acceleration=case['acceleration'],
            environments=len(ids),survived=int(survive.sum()),
            failures_before_request=int((np.asarray(ends)[ids]<warmup).sum()),
            final2s_vx_survivors=None if not last.any() else float(vx[last].mean()),
            final2s_mae_survivors=None if not last.any() else float(abs(vx-case['speed'])[last].mean()),
            full_command_live_samples=int(mask.sum()),
            encoder_bias_full_command=None if not mask.any() else float(
                (get('encoder_velocity')[:,ids,0]-get('true_velocity_before')[:,ids,0])[mask].mean()))
        rows.append(row)
        for e in ids[~survive]:
            end = ends[e]
            recent = valid[:,e] & (times<=end+1e-6) & (times>=max(warmup,end-.8))
            support = get('support')[:,e]
            loaded = (support>=max(.1*weight,5.)).all(-1) & (support.sum(-1)>=.8*weight)
            low = np.flatnonzero(recent & ~loaded)
            onset = int(low[0]) if len(low) else None
            tail = recent if onset is None else recent & (times<=times[onset]) & (times>=times[onset]-.1)
            k = min(len(values)-1,round(end/dt)-1)
            event = dict(case=case['name'],env=int(e),failure_s=float(end),
                low_support_s=None if onset is None else float(times[onset]),
                failure_tilt_deg=float(get('tilt')[k,e,0]),height_m=float(get('height')[k,e,0]),
                support_at_failure_N=support[k].tolist(),
                base_contact_at_failure_N=float(get('base_contact')[k,e,0]),
                max_leg_torque_pre_failure_Nm=float(abs(get('pd_torque')[recent,e][:,[0,1,3,4]]).max()) if recent.any() else None,
                max_wheel_torque_pre_failure_Nm=float(abs(get('actual_torque')[recent,e][:,[2,5]]).max()) if recent.any() else None,
                target_clip_pre_failure=bool((abs(get('leg_target')[recent,e]-get('leg_unclipped')[recent,e])>1e-6).any()),
                swing_before_low_deg=None if not tail.any() else get('swing')[tail,e].mean(0).tolist(),
                length_before_low_m=None if not tail.any() else get('leg_length')[tail,e].mean(0).tolist(),
                request_command_at_failure=get('applied_command')[k,e].tolist(),
                down_ticks_at_failure=int(get('down_ticks')[k,e,0]))
            events.append(event)
    return dict(cases=rows,events=events)


@torch.inference_mode()
def run(args, checkpoint, label, cases, reference, registry):
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from rsl_rl.models import MLPModel
    from ct_mjlab.envs.chuanliantui_split.runner import contracts
    cfg,train = registry.get_cfgs('chuanliantui_walk',play=True)
    cfg.auto_reset,cfg.seed = False,args.seed
    cfg.metrics['walk_diagnosis'] = MetricsTermCfg(func=sample_physics,per_substep=True,reduce='last')
    count = sum(len(c['env_ids']) for c in cases)
    wrapped,_ = registry.make_env('chuanliantui_walk',env_cfg=cfg,train_cfg=train,num_envs=count,device=args.device)
    raw = wrapped.unwrapped
    try:
        built,_ = registry.make_alg_runner(wrapped,'chuanliantui_walk',train_cfg=train,device=args.device)
        saved = torch.load(checkpoint,map_location=args.device,weights_only=False)
        state = raw.command_manager.get_term('standup')
        expected = contracts(state.cfg)
        # v7只做同环境下的权重诊断；不迁移元数据、不宣称正式入口兼容。
        for key in ('observation_contract','control_contract','policy_role','reward_recipe','reward_weights',
                    'physics_dt_s','policy_dt_s','exploration'):
            assert saved['infos'][key] == expected[key],key
        built.alg.actor.load_state_dict(saved['actor_state_dict'],strict=True)
        actor = built.alg.actor
        actor.eval()
        before = {key:value.cpu().clone() for key,value in actor.state_dict().items()}
        previous_actor = None
        if args.action_source != 'original':
            previous_actor = copy.deepcopy(actor)
            previous_actor.load_state_dict(torch.load(args.previous,map_location=args.device,
                weights_only=False)['actor_state_dict'],strict=True)
            previous_actor.eval()
        state.speed_curriculum_frozen = True
        obs,_ = raw.reset(seed=args.seed)
        initial = raw.sim.data.qpos.cpu().numpy().copy()
        initial_v = raw.sim.data.qvel.cpu().numpy().copy()
        pd = raw.action_manager.get_term('pd')
        fixed = dict(qpos=initial,qvel=initial_v,nominal=pd._nominal_pos.cpu().numpy(),
            kp=pd.kp_scale.cpu().numpy(),kd=pd.kd_scale.cpu().numpy(),strength=pd.motor_strength.cpu().numpy(),
            mass=raw.sim.mj_model.body_mass.copy(),friction=raw.sim.mj_model.geom_friction.copy())
        if reference is not None:
            assert all(np.array_equal(fixed[k],reference[k]) for k in fixed),'Initial/physical state differs'
        targets = torch.empty(count,device=args.device)
        accelerations = torch.empty_like(targets)
        for case in cases:
            targets[case['env_ids']]=case['speed']
            accelerations[case['env_ids']]=case['acceleration']
        def commands(time):
            state.tracking_episode.fill_(True)
            state.zero_hold.zero_()
            state.tracking_velocity.copy_(targets if time>=args.warmup else torch.zeros_like(targets))
            state.tracking_yaw_rate.zero_()
            state.tracking_interval_steps.fill_(1000000)
            state.tracking_acceleration.copy_(accelerations)
            state.episode_height.fill_(args.height)
            state._apply_height(None)
        commands(0.)
        raw.observation_manager.reset()
        raw.obs_buf=raw.observation_manager.compute(update_history=True)
        obs=raw.get_observations()
        recorder=Recorder(raw,round(args.seconds/raw.physics_dt))
        raw._walk_diagnosis=recorder
        ends=np.full(count,args.seconds)
        with patch.object(pd,'apply_actions',recorder.apply):
            for tick in range(round(args.seconds/raw.step_dt)):
                commands(tick*raw.step_dt)
                recorder.encoder=actor.encoder(obs['history'])/2.
                recorder.true_before=raw.scene['robot'].data.root_com_lin_vel_b.clone()
                action=actor(obs)
                if args.velocity_source in ('oracle','oracle-vx'):
                    latent=MLPModel.get_latent(actor,obs)
                    velocity=recorder.true_before*2. if args.velocity_source=='oracle' else recorder.encoder*2.
                    if args.velocity_source=='oracle-vx':
                        # 只隔离前向估速反馈，其余两轴保留原编码器输出。
                        velocity[:,0]=recorder.true_before[:,0]*2.
                    action=actor.mlp(torch.cat((latent,velocity),-1))
                if previous_actor is not None:
                    reference_action=previous_actor(obs)
                    ids = pd._leg_action_ids if args.action_source=='previous-legs' else pd._wheel_action_ids
                    action[:,ids]=reference_action[:,ids]
                if train.clip_actions is not None:action=action.clamp(-train.clip_actions,train.clip_actions)
                recorder.action=action
                obs,_,terminated,timeout,_=raw.step(action)
                done=terminated|timeout
                first=done&recorder.live
                ends[first.cpu().numpy()]=(tick+1)*raw.step_dt
                recorder.live&=~done
                if done.any() and tick+1<round(args.seconds/raw.step_dt):
                    raw.reset(env_ids=done.nonzero().flatten())
                    obs=raw.get_observations()
                if (tick+1)%500==0:print(label,(tick+1)*raw.step_dt,'seconds, live',int(recorder.live.sum()),flush=True)
        assert recorder.count==round(args.seconds/raw.physics_dt)
        assert all(torch.equal(before[k],v.cpu()) for k,v in actor.state_dict().items())
        values=recorder.trace.cpu().numpy()
        assert np.isfinite(values).all()
        weight=float(raw.sim.mj_model.body_mass.sum()*9.81)
        summary=summarize(values,recorder.layout,cases,ends,raw.physics_dt,args.seconds,args.warmup,weight)
        result=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),label=label,
            actual_profile=saved['infos']['experiment_profile_contract'],common_environment_profile=expected['experiment_profile_contract'],
            load='strict actor/encoder state_dict; no metadata migration',parameter_updates=0,actor_unchanged=True,
            velocity_source=args.velocity_source,action_source=args.action_source,
            seed=args.seed,seconds=args.seconds,warmup=args.warmup,height=args.height,
            dt=raw.physics_dt,policy_dt=raw.step_dt,weight_N=weight,layout=recorder.layout,
            initial_identical_to_other_policy=reference is not None,joint_names=recorder.joint_names,
            cases=cases,episode_end=ends.tolist(),summary=summary,
            timing='PD target/torque before 2ms integration; physical/contact after integration, derived signals lag <=2ms',
            scope='Fixed-request nominal policy diagnosis; not formal balance acceptance, training or deployment')
        # 保存实际初态和物理数组，供跨轮次配对时逐项复核。
        np.savez_compressed(args.output/(label+'.npz'),values=values,initial_qpos=initial,initial_qvel=initial_v,
                            **{'fixed_'+key:value for key,value in fixed.items()})
        write(args.output/(label+'.json'),result)
        return result,fixed
    finally:
        raw.close()


def main(args):
    torch.set_num_threads(1)
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3]
    training=args.training_artifact.resolve()
    launch=json.loads((training/'launch.json').read_text())
    source_hashes=json.loads((training/'source-sha256.json').read_text())
    source=args.output/'frozen-runtime/mjlab_training/src/ct_mjlab'
    shutil.copytree(Path(launch['log_dir'])/'snapshot/ct_mjlab',source,ignore=shutil.ignore_patterns('__pycache__'))
    (args.output/'frozen-runtime/resources').symlink_to(root/'resources',target_is_directory=True)
    assert all(digest(source/name)==sha for name,sha in source_hashes.items())
    assets=json.loads((training/'physical-files-sha256.json').read_text())
    assert all(digest(root/name)==sha for name,sha in assets.items())
    sys.path.insert(0,str(source.parent))
    from ct_mjlab.envs import task_registry
    cases=[]
    for acceleration in args.accelerations:
        for speed in (0.,.35,-1.):
            start=len(cases)*args.num_per_case
            cases.append(dict(name=f'v{speed:g}-a{acceleration:g}',speed=speed,acceleration=acceleration,
                              env_ids=list(range(start,start+args.num_per_case))))
    current=args.current_checkpoint or Path(launch['log_dir'])/f"model_{launch['last_iteration']}.pt"
    checkpoints=[(args.previous_label,args.previous),(args.current_label,current)]
    checkpoints.extend((label,Path(path)) for label,path in args.extra_checkpoint)
    if args.action_source=='split':
        checkpoints=[('v8-previous-legs',Path(launch['log_dir'])/'model_999.pt'),
                     ('v8-previous-wheels',Path(launch['log_dir'])/'model_999.pt')]
    reference=None;reports=[]
    for label,checkpoint in checkpoints:
        if label.startswith('v8-previous-'):
            args.action_source=label.removeprefix('v8-')
        report,fixed=run(args,checkpoint,label,cases,reference,task_registry)
        if reference is None:reference=fixed
        reports.append(report)
    assert all(digest(root/name)==sha for name,sha in assets.items())
    write(args.output/'result.json',dict(verified=True,parameter_updates=0,source_files=len(source_hashes),
        resource_files=len(assets),runs=reports,initial_and_physical_states_equal=True))
    shutil.copy2(__file__,args.output/'source.py')
    print(json.dumps([dict(label=r['label'],cases=r['summary']['cases']) for r in reports],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-artifact',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--current-checkpoint',type=Path)
    parser.add_argument('--previous-label',default='v7')
    parser.add_argument('--current-label',default='v8')
    parser.add_argument('--extra-checkpoint',nargs=2,action='append',default=[],metavar=('LABEL','PATH'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=20.)
    parser.add_argument('--warmup',type=float,default=2.)
    parser.add_argument('--height',type=float,default=.26)
    parser.add_argument('--num-per-case',type=int,default=16)
    parser.add_argument('--accelerations',type=float,nargs='+',default=[.5,2.])
    parser.add_argument('--velocity-source',choices=('encoder','oracle','oracle-vx'),default='encoder')
    parser.add_argument('--action-source',choices=('original','split'),default='original',
                        help='split compares v8 with v7 legs versus v7 wheels, same observed histories')
    parser.add_argument('--seed',type=int,default=43)
    parser.add_argument('--device',default='cuda:0')
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Diagnosis cannot train')):
        main(parser.parse_args())
