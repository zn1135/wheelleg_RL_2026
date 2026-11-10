#!/usr/bin/env python3
"""Measure a frozen SuperMimic policy or original PPO reference under one protocol.

A single fixed-request run is not full-task, CPU sim2sim or hardware acceptance.
"""
from scripts.agent.motion.assess import STANDARD, THRESHOLDS, supermimic_phase

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'mjlab_training/src'))
from ct_mjlab.validation import require_gpu
from scripts.agent.shared.recording import Recorder, digest, sample_physics


HEIGHT, START, STOP, SETTLE = .26, 2., 16., 1.5
DEFAULT_SPEEDS = (0., .5, -.5)
MODEL_FIELDS = ('body_mass', 'body_ipos', 'body_iquat', 'body_inertia',
                'geom_friction', 'geom_pos', 'geom_quat', 'geom_size',
                'jnt_stiffness', 'dof_damping')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def actor_digest(actor):
    result = hashlib.sha256()
    for key, tensor in sorted(actor.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        result.update(key.encode())
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def snapshot_sources(output):
    source = ROOT/'mjlab_training/src/ct_mjlab'
    paths = list(sorted(source.rglob('*.py')))
    paths += [ROOT/'scripts/agent/motion/assess.py', Path(__file__).resolve(), Path(sys.modules[Recorder.__module__].__file__).resolve()]
    paths += [ROOT/'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml']
    hashes = {}
    for path in paths:
        relative = str(path.relative_to(ROOT))
        destination = output/'snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[relative] = digest(path)
        assert digest(destination) == hashes[relative]
    write(output/'source-sha256.json', hashes)
    return hashes


def verify_training_sources(checkpoint, output, policy_kind):
    current = ROOT/'mjlab_training/src/ct_mjlab'
    snapshot = checkpoint.parent/'snapshot/ct_mjlab'
    manifest = checkpoint.parent/'source_sha256.json'
    if policy_kind == 'ppo-reference':
        if not snapshot.is_dir():
            raise ValueError('PPO reference requires its original snapshot/ct_mjlab')
        hashes = {str(path.relative_to(snapshot)):digest(path) for path in snapshot.rglob('*.py')}
        from ct_mjlab.utils.source_migration import verify_training_tree
        proof = verify_training_tree(hashes, snapshot, current, allowed_added_prefixes=('supermimic/',))
        copied_manifest = output/'training-source-sha256.json'
        write(copied_manifest,hashes)
        return dict(**proof, manifest=str(copied_manifest), manifest_sha256=digest(copied_manifest),
            snapshot_root=str(snapshot),source_files=len(hashes))
    if not manifest.is_file():
        return dict(verified=False,reason='Checkpoint run has no training source manifest')
    hashes = json.loads(manifest.read_text())
    from ct_mjlab.utils.source_migration import verify_training_tree
    proof = verify_training_tree(hashes, snapshot, current)
    shutil.copy2(manifest,output/'training-source-sha256.json')
    return dict(**proof, manifest=str(manifest), manifest_sha256=digest(manifest),
        snapshot_root=str(snapshot),source_files=len(hashes))



def verify_ppo_resources(checkpoint, output):
    candidates = [checkpoint.parent/'physical-files-sha256.json']
    reference = ROOT/'artifacts/legacy_mjlab/walk25-v8-resume2000-20261030-211315'
    launch = reference/'launch.json'
    if launch.is_file() and Path(json.loads(launch.read_text())['log_dir']).resolve() == checkpoint.parent:
        candidates.append(reference/'physical-files-sha256.json')
    manifest = next((path for path in candidates if path.is_file()),None)
    if manifest is None:
        raise ValueError('PPO reference needs an original physical-files-sha256.json tied to its training run')
    hashes = json.loads(manifest.read_text())
    xml = 'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'
    assert xml in hashes, 'PPO physical manifest must cover the actual training XML'
    for name,sha in hashes.items():
        assert digest(ROOT/name) == sha, f'PPO training/current physical resource differs: {name}'
    shutil.copy2(manifest,output/'training-physical-sha256.json')
    return dict(verified=True,manifest=str(manifest),manifest_sha256=digest(manifest),
        resource_files=len(hashes),xml_sha256=hashes[xml])


def make_cases(count, speeds=DEFAULT_SPEEDS, yaw_rates=None, yaw_acceleration=None):
    cases = []
    conditions = ([(speed,0.,acceleration) for acceleration in (.5,2.) for speed in speeds]
                  if yaw_rates is None else [(0.,yaw,None) for yaw in yaw_rates])
    per_case = count//len(conditions)
    for speed,yaw,acceleration in conditions:
        first = len(cases)*per_case
        if yaw_rates is None:
            case = dict(name=f'v{speed:g}-a{acceleration:g}', speed=speed,
                acceleration=acceleration, yaw_rate=yaw, kind='linear',
                yaw_acceleration=None, env_ids=list(range(first, first+per_case)))
        else:
            case = dict(name=f'yaw{yaw:g}',speed=0.,acceleration=None,
                yaw_rate=yaw,kind='yaw',yaw_acceleration=yaw_acceleration,
                env_ids=list(range(first, first+per_case)))
        cases.append(case)
    return cases


class PoseRecorder(Recorder):
    def __init__(self, raw, steps):
        super().__init__(raw, steps)
        # 保存真实500Hz状态，后续录像只读取这些姿态，不重新积分。
        self.poses = torch.empty((steps+1, *raw.sim.data.qpos.shape),
                                device=raw.device, dtype=raw.sim.data.qpos.dtype)
        self.poses[0].copy_(raw.sim.data.qpos)
        self.extra_layout = {name:[index,index+1] for index,name in enumerate(
            ('yaw_rate_body_z','yaw','heading_hold','heading_target','position_locked'))}
        self.extra_trace = torch.empty((steps,raw.num_envs,len(self.extra_layout)),device=raw.device)

    def sample(self):
        super().sample()
        self.poses[self.count].copy_(self.raw.sim.data.qpos)
        data,state = self.robot.data,self.state
        # 独立补充原Recorder没有的转向量；原Recorder和其历史协议保持原样。
        self.extra_trace[self.count-1] = torch.stack((data.root_link_ang_vel_b[:,2],data.heading_w,
            state.heading_hold.float(),state.heading_target,state.position_locked.float()),-1)

    def values_and_layout(self):
        offset = self.trace.shape[-1]
        layout = {**self.layout,**{name:[offset+a,offset+b] for name,(a,b) in self.extra_layout.items()}}
        return torch.cat((self.trace[:self.count],self.extra_trace[:self.count]),-1).cpu().numpy(),layout


def phase_metrics(values, layout, env, start, end, reference, dt, planned_end,
                  height=HEIGHT, yaw_reference=0.):
    def get(name):
        return values[:, env, slice(*layout[name])]
    times = (np.arange(len(values))+1)*dt
    selected = (times > start+1e-9) & (times <= min(end, planned_end)+1e-9)
    valid = get('valid')[:, 0] > .5
    finite = np.isfinite(values[:, env]).all(-1)
    mask = selected & valid & finite
    expected = max(0, round((min(end, planned_end)-start)/dt))
    complete = bool(end <= planned_end+1e-8 and expected > 0 and mask.sum() == expected)

    def stats(array, extra=None):
        sample = np.asarray(array)[mask if extra is None else mask & extra]
        if not sample.size:
            return None
        return dict(mean=float(sample.mean()), mae=float(abs(sample).mean()),
                    p95_abs=float(np.quantile(abs(sample), .95)), max_abs=float(abs(sample).max()))

    velocity = get('root_linear_velocity_w')
    horizontal = np.linalg.norm(velocity[:, :2], axis=-1)
    pitch = get('roll_pitch')[:, 1]
    yaw_rate = get('yaw_rate_body_z')[:,0]
    heading_hold = get('heading_hold')[:,0] > .5
    heading_error = get('heading_target')[:,0]-get('yaw')[:,0]
    heading_error = np.arctan2(np.sin(heading_error),np.cos(heading_error))
    pure_zero_request = reference == 0. and yaw_reference == 0.
    joint_ids = [0, 1, 2, 3]  # Recorder joint_names starts with four physical leg motors.
    motor = np.rad2deg(np.unwrap(get('joint_q')[:, joint_ids], axis=0))
    windows = []
    # t=0锚定的非重叠1秒窗口；不因策略到位慢而后移评分窗口。
    for second in range(math.ceil(start-1e-8), math.floor(min(end, planned_end)+1e-8)):
        window = (times > second+1e-9) & (times <= second+1.+1e-9)
        ok = bool(window.sum() == round(1./dt) and (valid & finite)[window].all())
        if not ok:
            windows.append(dict(start_s=second, complete=False, motor_peak_to_peak_deg=None))
            continue
        angles = motor[window]
        amplitude = np.ptp(angles, axis=0)
        windows.append(dict(start_s=second, complete=True,
            motor_peak_to_peak_deg=amplitude.tolist(), max_motor_peak_to_peak_deg=float(amplitude.max())))
    amplitudes = [w['max_motor_peak_to_peak_deg'] for w in windows if w['complete']]
    pitch_over = (abs(pitch) > 5.) & mask
    longest, current = 0, 0
    for item in pitch_over:
        current = current+1 if item else 0
        longest = max(longest, current)
    motion = supermimic_phase(values, layout, env, start, end, reference, dt, planned_end, yaw_reference)
    return dict(motion_standard=motion, start_s=start, end_s=end, reference_vx_m_s=reference,reference_yaw_rad_s=yaw_reference,
        pure_zero_request=pure_zero_request,
        expected_samples=expected, observed_first_episode_finite_samples=int(mask.sum()), complete=complete,
        vx_error_m_s=stats(get('true_velocity')[:, 0]-reference),
        yaw_error_rad_s=stats(yaw_rate-yaw_reference),
        horizontal_speed_m_s=stats(horizontal), pitch_deg=stats(pitch),
        height_error_m=stats(get('height')[:, 0]-height),
        heading_hold_samples=int((mask & heading_hold).sum()),
        heading_hold_error_deg=stats(np.rad2deg(heading_error),heading_hold),
        position_locked_samples=int((mask & (get('position_locked')[:,0]>.5)).sum()),
        pitch_over_5deg_duration_s=float(pitch_over.sum()*dt),
        pitch_over_5deg_longest_duration_s=float(longest*dt),
        zero_horizontal_speed_le_005=(motion["checks"].get("stopped") if complete and pure_zero_request else None),
        steady_pitch_le_5deg=motion["checks"]["steady_pitch"] if complete else None,
        motor_windows=windows,
        motor_window_max_peak_to_peak_deg=None if not amplitudes else max(amplitudes),
        motor_window_p95_peak_to_peak_deg=None if not amplitudes else float(np.quantile(amplitudes,.95)),
)


def summarize(values, layout, cases, ends, reasons, weight, dt, seconds, height=HEIGHT):
    rows = []
    times = (np.arange(len(values))+1)*dt
    for case in cases:
        speed, acceleration, yaw = case['speed'], case['acceleration'],case['yaw_rate']
        ramp = abs(speed)/acceleration if case['kind']=='linear' else abs(yaw)/case['yaw_acceleration']
        phases = [('initial',0.,START,0.,0.)]
        if speed or yaw:
            phases += [('start',START,START+ramp+SETTLE,speed,yaw),
                       ('cruise',START+ramp+SETTLE,STOP,speed,yaw),
                       ('brake',STOP,STOP+ramp+SETTLE,0.,0.),
                       ('parking',STOP+ramp+SETTLE,seconds,0.,0.)]
        else:
            phases += [('parking',START,seconds,0.,0.)]
        for env in case['env_ids']:
            data = values[:,env]
            get = lambda name: data[:,slice(*layout[name])]
            finite = np.isfinite(data).all(-1)
            valid = get('valid')[:,0] > .5
            all_finite = bool(finite[valid].all())
            survived = bool(ends[env] >= seconds-1e-7 and reasons[env] in ('duration','timeout'))
            support = get('support')
            balance_conditions = dict(
                every_wheel=(support >= max(.1*weight[env],5.)).all(-1),
                total_support=(support.sum(-1) >= .8*weight[env]) & (support.sum(-1) <= 1.2*weight[env]),
                base_clear=get('base_contact')[:,0] <= 1.,
                roll_pitch=np.max(abs(get('roll_pitch')),axis=-1) <= 35.,
                roll_pitch_angular_speed=np.linalg.norm(get('angular_velocity_xy'),axis=-1) <= .5)
            balanced = valid & finite
            for condition in balance_conditions.values():
                balanced &= condition
            hold = 0
            for item in balanced:
                hold = hold+1 if item else 0
            final = times > max(0.,seconds-1.)+1e-9
            observations = {name:bool(condition[final].all()) for name,condition in balance_conditions.items()}
            metrics = {name:phase_metrics(values,layout,env,a,b,target,dt,seconds,height,yaw_target)
                       for name,a,b,target,yaw_target in phases if a < min(b,seconds)}
            steady = [m for name,m in metrics.items() if name in ('cruise','parking')]
            rows.append(dict(env=env, case=case['name'], speed_m_s=speed,
                acceleration_m_s2=acceleration, episode_end_s=float(ends[env]),
                yaw_rate_rad_s=yaw,yaw_acceleration_rad_s2=case['yaw_acceleration'],case_kind=case['kind'],
                episode_end_reason=reasons[env], survived=survived, numerical_ok=all_finite,
                final_balanced=bool(survived and all_finite and hold*dt >= 1.-1e-7),
                final_balance_hold_s=hold*dt, final1s_balance_conditions=observations,
                steady_confirmed_numeric_constraints_met=bool(steady and all(m['complete']
                    and m['steady_pitch_le_5deg'] and (m['zero_horizontal_speed_le_005'] is not False) for m in steady)),
                motion_standard_passed=bool(survived and steady and all(m["motion_standard"]["passed"] for m in steady)),
                phases=metrics))
    return dict(motion_standard=STANDARD, motion_standard_passed=sum(r["motion_standard_passed"] for r in rows),
        per_environment=rows, num_envs=len(rows), survived=sum(r['survived'] for r in rows),
        numerical_failures=sum(not r['numerical_ok'] for r in rows),
        final_balanced=sum(r['final_balanced'] for r in rows),
        steady_confirmed_numeric_constraints_met=sum(r['steady_confirmed_numeric_constraints_met'] for r in rows),
        full_task_accepted=False,
        acceptance_limits=['One fixed-request run does not itself prove all speed/height/yaw/independent-seed coverage, paired PPO retention or CPU physics.',
            'Transient pitch duration, leg oscillation threshold and full coverage remain unconfirmed.',
            'A normal-speed video and independent seeds/conditions are still required for complete behavior acceptance.'])


@torch.inference_mode()
def run(args):
    validation=require_gpu(args.device)
    from dataclasses import asdict
    import mujoco
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    from ct_mjlab.envs import task_registry
    from ct_mjlab.envs.chuanliantui.model import MODEL_XML
    from ct_mjlab.supermimic.learner import Learner
    from ct_mjlab.supermimic.networks import act

    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot_sources(args.output)
    training_sources = verify_training_sources(args.checkpoint,args.output,args.policy_kind)
    physical_sources = (verify_ppo_resources(args.checkpoint,args.output)
                        if args.policy_kind == 'ppo-reference' else None)
    checkpoint_sha, xml_sha = digest(args.checkpoint), digest(MODEL_XML)
    cfg, train = task_registry.get_cfgs('supermimic',play=args.conditions == 'nominal')
    command_cfg = cfg.commands['standup']
    if args.yaw_rates is not None and any(abs(yaw)>command_cfg.yaw_rate_limit for yaw in args.yaw_rates):
        raise ValueError(f'Yaw requests must stay inside production +/-{command_cfg.yaw_rate_limit}rad/s')
    cases = make_cases(args.num_envs,args.speeds,args.yaw_rates,command_cfg.yaw_acceleration)
    cfg.auto_reset, cfg.seed = False, args.seed
    cfg.metrics['supermimic_assessment'] = MetricsTermCfg(func=sample_physics,per_substep=True,reduce='last')
    wrapped,_ = task_registry.make_env('supermimic',env_cfg=cfg,train_cfg=train,
        num_envs=args.num_envs,seed=args.seed,device=args.device)
    raw = wrapped.unwrapped
    try:
        configure_torch_backends(allow_tf32=False)
        assert raw.physics_dt == .002 and raw.step_dt == .01 and wrapped.clip_actions is None
        state = raw.command_manager.get_term('standup')
        if args.policy_kind == 'supermimic':
            learner = Learner(args.device)
            saved = learner.load(args.checkpoint,state,inference=True)
            actor = learner.actor
            algorithm_contract = saved['algorithm_contract']
            checkpoint_iteration, checkpoint_transitions = saved['iteration'],saved['transitions']
            load_method = 'SuperMimic Learner.load(inference=True), strict algorithm/task/XML contract'
        else:
            runner,_ = task_registry.make_alg_runner(wrapped,'supermimic',train_cfg=train,
                                                     device=args.device)
            # 原生PPO读取原始字段及契约；不改检查点键，不恢复优化器或执行学习。
            runner.load(str(args.checkpoint),load_cfg={'actor':True},strict=True,map_location=args.device)
            saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
            actor = runner.alg.actor
            algorithm_contract = 'PPO-reference'
            checkpoint_iteration, checkpoint_transitions = saved['iter'],None
            load_method = 'Original task PPO runner.load(actor only), strict original role/config contract'
        actor.eval()
        actor.requires_grad_(False)
        assert not actor.obs_normalization, 'Shared raw-history inference requires the original unnormalized actor'
        actor_sha = actor_digest(actor)
        state.speed_curriculum_frozen = True
        obs,_ = raw.reset(seed=args.seed)
        assert not state.ground_start.any()
        pd = raw.action_manager.get_term('pd')
        fixed = dict(qpos=raw.sim.data.qpos.cpu().numpy().copy(),qvel=raw.sim.data.qvel.cpu().numpy().copy(),
            nominal=pd._nominal_pos.cpu().numpy().copy(),kp=pd.kp_scale.cpu().numpy().copy(),
            kd=pd.kd_scale.cpu().numpy().copy(),strength=pd.motor_strength.cpu().numpy().copy(),
            position_gains=pd.position_gains.cpu().numpy().copy(),velocity_gains=pd.velocity_gains.cpu().numpy().copy(),
            env_origins=raw.scene.env_origins.cpu().numpy().copy())
        for name in MODEL_FIELDS:
            fixed['nominal_model_'+name] = np.asarray(getattr(raw.sim.mj_model,name)).copy()
            fixed['actual_model_'+name] = getattr(raw.sim.model,name).cpu().numpy().copy()
        weight = raw.sim.model.body_mass.sum(-1).cpu().numpy().reshape(-1)*9.81
        weight = np.broadcast_to(weight,(args.num_envs,)).copy()
        targets = torch.empty(args.num_envs,device=args.device)
        yaw_targets = torch.empty_like(targets)
        accelerations = torch.empty_like(targets) if args.yaw_rates is None else None
        for case in cases:
            targets[case['env_ids']] = case['speed']
            yaw_targets[case['env_ids']] = case['yaw_rate']
            if accelerations is not None:
                accelerations[case['env_ids']] = case['acceleration']

        def commands(time):
            state.tracking_episode.fill_(True)
            state.zero_hold.zero_()
            state.tracking_velocity.copy_(targets if START <= time < STOP else torch.zeros_like(targets))
            state.tracking_yaw_rate.copy_(yaw_targets if START <= time < STOP else torch.zeros_like(targets))
            state.tracking_interval_steps.fill_(1000000)
            if accelerations is not None:
                state.tracking_acceleration.copy_(accelerations)
            state.episode_height.fill_(args.height)
            state._apply_height(None)

        commands(0.)
        raw.observation_manager.reset()
        raw.obs_buf = raw.observation_manager.compute(update_history=True)
        obs = raw.get_observations()
        ticks = round(args.seconds/raw.step_dt)
        recorder = PoseRecorder(raw,round(args.seconds/raw.physics_dt))
        raw._walk_diagnosis = recorder
        ends = np.full(args.num_envs,args.seconds)
        reasons = ['duration']*args.num_envs
        termination_terms = [[] for _ in range(args.num_envs)]
        request_history = np.empty((ticks,args.num_envs),dtype=np.float32)
        request_yaw_history = np.empty_like(request_history)
        dump_yaml(args.output/'env.yaml',asdict(cfg))
        mujoco.mj_saveModel(raw.sim.mj_model,str(args.output/'scene.mjb'),None)
        with patch.object(pd,'apply_actions',recorder.apply):
            for tick in range(ticks):
                commands(tick*raw.step_dt)
                request_history[tick] = state.tracking_velocity.cpu().numpy()
                request_yaw_history[tick] = state.tracking_yaw_rate.cpu().numpy()
                recorder.encoder = actor.encoder(obs['history'])/2.
                recorder.true_before = raw.scene['robot'].data.root_com_lin_vel_b.clone()
                action = act(actor,obs['history'])
                if not torch.isfinite(action).all():
                    raise FloatingPointError('Nonfinite policy output; evaluation stopped before applying it')
                recorder.action = action
                obs,_,terminated,timeout,_ = raw.step(action)
                done = terminated | timeout
                first = done & recorder.live
                for env in first.nonzero().flatten().cpu().tolist():
                    ends[env] = (tick+1)*raw.step_dt
                    reasons[env] = 'failure' if bool(terminated[env]) else 'timeout'
                    manager = raw.termination_manager
                    termination_terms[env] = [name for name in manager.active_terms
                        if bool(manager.get_term(name)[env])]
                recorder.live &= ~done
                if done.any() and tick+1 < ticks:
                    # 其它环境继续完整观察；复位后的原环境保留valid=False，不能补成功。
                    raw.reset(env_ids=done.nonzero().flatten())
                    obs = raw.get_observations()
                if (tick+1)%500 == 0:
                    print(json.dumps(dict(seconds=(tick+1)*raw.step_dt,live=int(recorder.live.sum()))),flush=True)
        assert recorder.count == round(args.seconds/raw.physics_dt)
        assert actor_digest(actor) == actor_sha
        values,layout = recorder.values_and_layout()
        poses = recorder.poses.cpu().numpy()
        summary = summarize(values,layout,cases,ends,reasons,weight,raw.physics_dt,args.seconds,args.height)
        for row in summary['per_environment']:
            row['termination_terms'] = termination_terms[row['env']]
        np.savez_compressed(args.output/'trajectory.npz',values=values,qpos=poses,
            time_s=np.arange(1,len(values)+1)*raw.physics_dt,
            qpos_time_s=np.arange(len(poses))*raw.physics_dt,
            qpos_live=np.concatenate((np.ones((1,args.num_envs),dtype=bool),
                values[:,:,layout['valid'][0]]>.5)),
            requested_vx=request_history,requested_yaw=request_yaw_history,
            policy_time_s=np.arange(ticks)*raw.step_dt,weight_N=weight,
            **{'fixed_'+key:value for key,value in fixed.items()})
        assert digest(args.checkpoint) == checkpoint_sha and digest(MODEL_XML) == xml_sha
        assert all(digest(ROOT/name) == sha for name,sha in source_hashes.items())
        if training_sources['verified']:
            assert digest(Path(training_sources['manifest'])) == training_sources['manifest_sha256']
            for name,sha in json.loads(Path(training_sources['manifest']).read_text()).items():
                assert digest(Path(training_sources['snapshot_root'])/name) == sha
        if physical_sources is not None:
            manifest = Path(physical_sources['manifest'])
            assert digest(manifest) == physical_sources['manifest_sha256']
            assert all(digest(ROOT/name) == sha for name,sha in json.loads(manifest.read_text()).items())
        result = dict(**validation,task='chuanliantui_walk',policy_kind=args.policy_kind,algorithm_contract=algorithm_contract,
            checkpoint_load_method=load_method,
            checkpoint=str(args.checkpoint),checkpoint_sha256=checkpoint_sha,
            checkpoint_iteration=checkpoint_iteration,checkpoint_transitions=checkpoint_transitions,
            checkpoint_infos=saved['infos'],checkpoint_asset_sha256=saved.get('asset_sha256'),
            xml=str(MODEL_XML),xml_sha256=xml_sha,source_files=len(source_hashes),
            source_snapshot_sha256=digest(args.output/'source-sha256.json'),source_unchanged=True,
            training_source_verification=training_sources,training_physical_verification=physical_sources,
            engine_versions=dict(mujoco=mujoco.__version__,torch=torch.__version__,numpy=np.__version__),
            actor_sha256_before=actor_sha,actor_sha256_after=actor_digest(actor),
            actor_parameters_unchanged=True,parameter_updates=0,conditions=args.conditions,
            seed=args.seed,num_envs=args.num_envs,seconds=args.seconds,height_m=args.height,
            physics_dt_s=raw.physics_dt,policy_dt_s=raw.step_dt,layout=layout,
            joint_names=recorder.joint_names,cases=cases,
            coverage_facts=dict(case_kind='linear' if args.yaw_rates is None else 'yaw',
                requested_vx_m_s=list(args.speeds) if args.yaw_rates is None else [0.],
                requested_yaw_rad_s=[0.] if args.yaw_rates is None else args.yaw_rates,
                tested_height_m=args.height,case_count=len(cases),environments_per_case=args.num_envs//len(cases),
                linear_accelerations_m_s2=[.5,2.] if args.yaw_rates is None else [],
                yaw_acceleration_rad_s2=command_cfg.yaw_acceleration,
                production_yaw_limit_rad_s=command_cfg.yaw_rate_limit,
                seed_role='development comparison' if args.seed==43 else 'separate evaluation seed',
                full_task_accepted=False,
                scope='Coverage facts for this one run only; failures remain in each original case denominator'),
            production_direction_speed_config=state.speed_config(),production_yaw_config=state.yaw_config(),
            command_schedule=dict(initial_zero_s=[0.,START],travel_s=[START,STOP],parking_from_s=STOP),
            descriptive_windows=dict(steady_after_nominal_ramp_s=SETTLE,
                status='Fixed descriptive windows; settle duration is not a user-confirmed acceptance tolerance'),
            motion_standard=STANDARD, motion_thresholds=THRESHOLDS.copy(),
            confirmed_thresholds=dict(zero_world_horizontal_speed_max_m_s=THRESHOLDS['parking_horizontal_peak_m_s'],
                steady_pitch_abs_max_deg=THRESHOLDS['steady_pitch_peak_deg']),
            unconfirmed_thresholds=dict(leg_amplitude_or_periodicity='No accepted numeric cutoff',
                absolute_transient_pitch_duration='Not specified'),
            balance_criterion='final-wheel-balance-v3.1: final continuous1s, both wheels>=max(0.1weight,5N),total0.8..1.2weight,base<=1N,max(abs(roll),abs(pitch))<=35deg,xy-angular-speed<=0.5rad/s',
            trajectory_sha256=digest(args.output/'trajectory.npz'),scene_sha256=digest(args.output/'scene.mjb'),
            source_provenance='See training_source_verification: PPO shared training sources must match, with added supermimic modules explicitly listed; PPO physical resources and native loader contract are separately checked.',
            measurement_timing='Physical/contact values after each2ms integration; derived values may lag<=2ms. Encoder/COM target latched before action.',
            summary=summary)
        write(args.output/'summary.json',result)
        print(json.dumps(dict(output=str(args.output),survived=summary['survived'],
            final_balanced=summary['final_balanced'],full_task_accepted=False)),flush=True)
    except Exception as error:
        if 'recorder' in locals() and recorder.count:
            values,layout = recorder.values_and_layout()
            np.savez_compressed(args.output/'incomplete-trajectory.npz',
                values=values[:recorder.count],
                qpos=recorder.poses[:recorder.count+1].cpu().numpy())
        write(args.output/'failure.json',dict(full_task_accepted=False,run_completed=False,
            exception_type=type(error).__name__,message=str(error),
            recorded_physics_samples=recorder.count if 'recorder' in locals() else 0,
            checkpoint_sha256=checkpoint_sha,parameter_updates=0))
        raise
    finally:
        raw.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--policy-kind',choices=('supermimic','ppo-reference'),default='supermimic')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=96)
    parser.add_argument('--seed',type=int,default=43)
    parser.add_argument('--seconds',type=float,default=30.)
    parser.add_argument('--height',type=float,default=HEIGHT)
    parser.add_argument('--speeds',type=float,nargs='+',default=list(DEFAULT_SPEEDS),
                        help='Linear requests, each tested at0.5 and2m/s2; default0,+0.5,-0.5')
    parser.add_argument('--yaw-rates',type=float,nargs='+',
                        help='Yaw requests with production yaw limit/ramp; mutually exclusive with custom--speeds')
    parser.add_argument('--conditions',choices=('nominal','training'),default='nominal')
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    if not math.isfinite(args.height) or not .22 <= args.height <= .30:
        parser.error('--height must be in the current0.22..0.30m command range')
    if not all(math.isfinite(speed) and abs(speed)<=1.5 for speed in args.speeds):
        parser.error('--speeds must be finite and inside the current+/-1.5m/s envelope')
    if len(set(args.speeds)) != len(args.speeds):
        parser.error('--speeds must contain distinct requests')
    if args.yaw_rates is not None:
        if tuple(args.speeds) != DEFAULT_SPEEDS:
            parser.error('--yaw-rates cannot be combined with nondefault--speeds')
        if not all(math.isfinite(yaw) for yaw in args.yaw_rates) or len(set(args.yaw_rates))!=len(args.yaw_rates):
            parser.error('--yaw-rates must contain distinct finite requests')
    case_count = 2*len(args.speeds) if args.yaw_rates is None else len(args.yaw_rates)
    if args.num_envs < case_count or args.num_envs%case_count:
        parser.error(f'--num-envs must be a positive multiple of the{case_count}requested cases')
    if not math.isfinite(args.seconds) or args.seconds < 2. or abs(args.seconds/.01-round(args.seconds/.01)) > 1e-6:
        parser.error('--seconds must be finite, >=2 and an integer number of10ms policy ticks')
    torch.set_num_threads(1)
    with patch.object(torch.optim.Adam,'step',side_effect=AssertionError('Evaluation cannot train')):
        run(args)
