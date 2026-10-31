#!/usr/bin/env python3
"""Frozen getup target slew comparison through the original delayed PD."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import mujoco
import numpy as np
import torch


LEGS = ('lf0', 'lf00', 'rf0', 'rf00')
FIELDS = [f'{prefix}_{joint}' for prefix in ('target', 'desired') for joint in LEGS] + [
    'left_support_N', 'right_support_N', 'base_N', 'height_m', 'tilt_deg',
    'max_roll_pitch_rad', 'roll_pitch_speed_rad_s', 'planar_speed_m_s', 'vz_m_s',
    'gravity_z', 'hold', 'live', 'limited', 'finished', 'has_stood']
CONTROL_FIELDS = ([f'q_{j}' for j in LEGS]
    + [f'{prefix}_{j}' for prefix in ('qd', 'raw', 'control', 'delayed') for j in range(6)]
    + [f'nominal_{j}' for j in LEGS]
    + [f'{prefix}_{j}' for prefix in ('command', 'actual') for j in range(6)])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TargetSlew:
    def __init__(self, pd, raw, rate):
        self.pd, self.raw, self.rate = pd, raw, rate
        self.original_apply, self.original_reset = pd.apply_actions, pd.reset
        self.started = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
        self.finished = torch.zeros_like(self.started)
        self.target = torch.zeros(raw.num_envs, 4, device=raw.device)
        self.original_nominal = pd.nominal_positions.clone()
        self.max_roundtrip_error = torch.zeros((), device=raw.device)
        self.max_control_change = torch.zeros_like(self.max_roundtrip_error)

    def reset(self, env_ids=None):
        self.original_reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.started[ids], self.finished[ids], self.target[ids] = False, False, 0.

    def apply(self):
        pd, raw = self.pd, self.raw
        legs, wheels = pd._leg_action_ids, pd._wheel_action_ids
        head = (pd._history_head+1) % len(pd._action_history)
        # 原PD先写新槽再读FIFO；零延迟必须使用本拍输入，不能读旧槽。
        delayed = pd._action_history[(head-pd.action_delay_steps) % len(pd._action_history), pd._env_ids].clone()
        wheel_delayed = pd._action_history[(head-pd.wheel_action_delay_steps) % len(pd._action_history), pd._env_ids]
        delayed[:, legs] = torch.where((pd.action_delay_steps == 0)[:, None],
            pd._control_actions[:, legs], delayed[:, legs])
        delayed[:, wheels] = torch.where((pd.wheel_action_delay_steps == 0)[:, None],
            pd._control_actions[:, wheels], wheel_delayed[:, wheels])
        q = raw.sim.data.qpos[:, pd._motor_q_adr][:, legs]
        qd = raw.sim.data.qvel[:, pd._motor_v_adr]
        low, high = pd.reference_positions[:, legs]-3.14, pd.reference_positions[:, legs]+3.14
        self.desired = (self.original_nominal[:, legs]+pd.cfg.leg_position_scale*delayed[:, legs]).clamp(low, high)
        self.hold = pd._ground_hold()
        state = raw.command_manager.get_term('standup')
        self.limited = state.ground_start & ~self.hold & ~self.finished & (self.rate > 0.)
        begin = self.limited & ~self.started
        # 在首个实际出力子步锁存角度；首拍只消除P误差，原速度阻尼保留。
        self.target.copy_(torch.where(begin[:, None], q, self.target))
        self.started |= begin
        advance = self.limited & ~begin
        step = (self.desired-self.target).clamp(-self.rate*raw.physics_dt, self.rate*raw.physics_dt)
        self.target.add_(torch.where(advance[:, None], step, 0.))
        self.target.copy_(self.target.clamp(low, high))
        self.applied = torch.where(self.limited[:, None], self.target, self.desired)
        # 首次站稳后追平原目标才退出限速，交接这一子步也受限速约束。
        self.finished |= (self.limited & state.has_stood
                          & ((self.desired-self.target).abs().amax(-1) < 1e-6))
        controls = pd._control_actions.clone()
        temporary = self.applied-pd.cfg.leg_position_scale*delayed[:, legs]
        pd._nominal_pos[:, legs] = torch.where(self.limited[:, None], temporary, self.original_nominal[:, legs])
        actual = (pd._nominal_pos[:, legs]+pd.cfg.leg_position_scale*delayed[:, legs]).clamp(low, high)
        self.applied = actual
        self.max_roundtrip_error = torch.maximum(self.max_roundtrip_error,
            (actual-torch.where(self.limited[:, None], self.target, self.desired)).abs().max())
        try:
            self.original_apply()
        finally:
            pd._nominal_pos.copy_(self.original_nominal)
        assert pd._history_head == head
        self.max_control_change = torch.maximum(self.max_control_change, (pd._control_actions-controls).abs().max())
        self.before = torch.cat((q, qd, pd.raw_action, controls, delayed,
            self.original_nominal[:, legs], pd.torques), -1)


def sample_physics(env):
    recorder = getattr(env, '_target_slew_recorder', None)
    if recorder is not None:
        recorder.sample()
    return torch.zeros(env.num_envs, device=env.device)


class Recorder:
    def __init__(self, raw, limiter, seconds):
        self.raw, self.limiter, self.count = raw, limiter, 0
        self.live = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
        self.values = torch.empty(round(seconds/raw.physics_dt), raw.num_envs, len(FIELDS), device=raw.device)
        self.control = torch.empty(round(1./raw.physics_dt), raw.num_envs, len(CONTROL_FIELDS), device=raw.device)
        self.root_q = raw.scene['robot'].indexing.free_joint_q_adr
        self.root_v = raw.scene['robot'].indexing.free_joint_v_adr

    def sample(self):
        raw, limiter, d = self.raw, self.limiter, self.raw.sim.data
        q, v = d.qpos[:, self.root_q], d.qvel[:, self.root_v]
        w, x, y, z = q[:, 3:].unbind(-1)
        gx, gy, gz = 2*(w*y-x*z), -2*(y*z+w*x), -1+2*(x*x+y*y)
        rp = torch.maximum(torch.atan2(-gy, -gz).abs(), gx.clamp(-1., 1.).asin().abs())
        tilt = torch.rad2deg(torch.acos((-gz).clamp(-1., 1.)))
        support = -raw.scene['wheel_contact'].data.force[:, :, 2]
        base = raw.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
        state = raw.command_manager.get_term('standup')
        self.values[self.count] = torch.cat((limiter.applied, limiter.desired, support,
            torch.stack((base, q[:, 2]-raw.scene.env_origins[:, 2], tilt, rp,
                v[:, 3:5].norm(dim=-1), v[:, :2].norm(dim=-1), v[:, 2], gz,
                limiter.hold.float(), self.live.float(), limiter.limited.float(),
                limiter.finished.float(), state.has_stood.float()), -1)), -1)
        if self.count < len(self.control):
            pd = limiter.pd
            self.control[self.count] = torch.cat((limiter.before, d.qfrc_actuator[:, pd._motor_v_adr]), -1)
        self.count += 1


def episode_summary(values, weight, ends, reasons, stood, rate):
    col = lambda name: values[:, :, FIELDS.index(name)]
    support = values[:, :, [FIELDS.index('left_support_N'), FIELDS.index('right_support_N')]]
    live = col('live') > .5
    good = ((support >= np.maximum(.1*weight, 5.)[None, :, None]).all(-1)
        & (support.sum(-1) >= .8*weight) & (support.sum(-1) <= 1.2*weight)
        & (col('base_N') <= 1.) & (col('max_roll_pitch_rad') <= np.deg2rad(35.))
        & (col('roll_pitch_speed_rad_s') <= .5) & live & np.isfinite(values).all(-1))
    holds = np.zeros(len(weight), dtype=int)
    for sample in good:
        holds = np.where(sample, holds+1, 0)
    rows = []
    for i in range(len(weight)):
        first = np.flatnonzero(live[:, i] & (col('hold')[:, i] < .5))[0]
        finished = np.flatnonzero(live[:, i] & (col('finished')[:, i] > .5))
        full = bool(live[:, i].all() and reasons[i] != 'failure')
        rows.append(dict(env=i, first_stood_s=stood[i], end_s=ends[i], end_reason=reasons[i],
            stood_and_survived_40s=bool(stood[i] is not None and full),
            final_balanced_v31=bool(full and holds[i] >= 500),
            final_balance_hold_s=float(holds[i]*.002),
            limiter_finished_start_s=float(finished[0]*.002) if len(finished) else None))
    return rows


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1)
    assert args.rate >= 0. and args.seconds >= 1.
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    training = args.training_artifact.resolve()
    launch = json.loads((training/'launch.json').read_text())
    checkpoint = Path(launch['log_dir'])/'model_999.pt'
    expected = json.loads((training/'source-sha256.json').read_text())
    snapshot = Path(launch['log_dir'])/'snapshot/ct_mjlab'
    assert all(digest(snapshot/p)==h for p,h in expected.items())
    source = args.output.resolve()/'frozen-runtime/mjlab_training/src/ct_mjlab'
    shutil.copytree(snapshot, source, ignore=shutil.ignore_patterns('__pycache__'))
    (args.output.resolve()/'frozen-runtime/resources').symlink_to(root/'resources', target_is_directory=True)
    assets = json.loads((training/'physical-files-sha256.json').read_text())
    assert all(digest(root/p)==h for p,h in assets.items())
    sys.path.insert(0, str(source.parent))
    from ct_mjlab.envs import task_registry
    from mjlab.managers.metrics_manager import MetricsTermCfg
    assert Path(sys.modules[type(task_registry).__module__].__file__).resolve().is_relative_to(source.parent)
    cfg, agent = task_registry.get_cfgs('chuanliantui_getup', play=True)
    cfg.auto_reset, cfg.seed = False, args.seed
    cfg.metrics['target_slew'] = MetricsTermCfg(func=sample_physics, per_substep=True, reduce='last')
    wrapped, _ = task_registry.make_env('chuanliantui_getup', env_cfg=cfg, train_cfg=agent,
        num_envs=args.num_envs, device=args.device)
    raw = wrapped.unwrapped
    try:
        runner, _ = task_registry.make_alg_runner(wrapped, 'chuanliantui_getup', train_cfg=agent, device=args.device)
        runner.load(str(checkpoint), load_cfg={'actor':True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        before = {k:v.cpu().clone() for k,v in runner.alg.actor.state_dict().items()}
        obs, _ = raw.reset(seed=args.seed)
        initial_qpos, initial_qvel = raw.sim.data.qpos.clone(), raw.sim.data.qvel.clone()
        if args.recording:
            with np.load(args.recording/'trajectory.npz') as old:
                assert np.array_equal(initial_qpos.cpu().numpy(), old['qpos'][0])
        state, pd = raw.command_manager.get_term('standup'), raw.action_manager.get_term('pd')
        assert state.ground_start.all() and raw.physics_dt == .002 and raw.step_dt == .01
        limiter = TargetSlew(pd, raw, args.rate)
        recorder = Recorder(raw, limiter, args.seconds)
        raw._target_slew_recorder = recorder
        mujoco.mj_saveModel(raw.sim.mj_model, str(args.output/'scene.mjb'), None)
        weight = (raw.sim.model.body_mass.sum(-1)*9.81).cpu().numpy()
        meta = dict(kp_scale=pd.kp_scale.cpu().tolist(), kd_scale=pd.kd_scale.cpu().tolist(),
            strength=pd.motor_strength.cpu().tolist(), leg_delay_steps=pd.action_delay_steps.cpu().tolist(),
            wheel_delay_steps=pd.wheel_action_delay_steps.cpu().tolist(), leg_stiffness=pd.cfg.leg_stiffness,
            leg_damping=pd.cfg.leg_damping, leg_position_scale=pd.cfg.leg_position_scale,
            wheel_stiffness=pd.cfg.wheel_stiffness, wheel_damping=pd.cfg.wheel_damping,
            wheel_velocity_scale=pd.cfg.wheel_velocity_scale, max_wheel_velocity=pd.cfg.max_wheel_velocity,
            reference=pd.reference_positions.cpu().tolist())
        ends, reasons, stood, failures = [args.seconds]*args.num_envs, ['duration']*args.num_envs, [None]*args.num_envs, [None]*args.num_envs
        steps = round(args.seconds/raw.step_dt)
        poses = torch.empty(steps+1, *initial_qpos.shape, device=raw.device)
        poses[0] = initial_qpos
        with patch.object(pd, 'apply_actions', limiter.apply), patch.object(pd, 'reset', limiter.reset):
            for tick in range(steps):
                action = policy(obs)
                if agent.clip_actions is not None:
                    action = action.clamp(-agent.clip_actions, agent.clip_actions)
                obs, _, terminated, timeout, _ = raw.step(action)
                live = recorder.live.clone()
                poses[tick+1] = torch.where(live[:, None], raw.sim.data.qpos, poses[tick])
                for i in (live & state.has_stood).nonzero().flatten().tolist():
                    if stood[i] is None:
                        stood[i] = (tick+1)*raw.step_dt
                done = terminated | timeout
                for i in (live & done).nonzero().flatten().tolist():
                    ends[i], reasons[i] = (tick+1)*raw.step_dt, 'failure' if terminated[i] else 'timeout'
                    failures[i] = dict(inverted=bool(raw.scene['robot'].data.projected_gravity_b[i,2]>0),
                        airborne_ticks=int(state.wheels_airborne_steps[i]), has_stood=bool(state.has_stood[i]))
                recorder.live &= ~done
                if tick+1 < steps and done.any():
                    raw.reset(env_ids=done.nonzero(as_tuple=False).flatten())
                    obs = raw.get_observations()
                if (tick+1) % 1000 == 0:
                    print(f'recorded {(tick+1)*raw.step_dt:.0f}/{args.seconds:.0f}s; '
                          f'first episodes live={int(recorder.live.sum())}, '
                          f'ever stood={sum(t is not None for t in stood)}', flush=True)
        assert all(torch.equal(v.cpu(), before[k]) for k,v in runner.alg.actor.state_dict().items())
        assert recorder.count == round(args.seconds/raw.physics_dt)
        assert float(limiter.max_roundtrip_error) < 2e-6 and float(limiter.max_control_change) == 0.
        values, control = recorder.values.cpu().numpy(), recorder.control.cpu().numpy()
        assert np.isfinite(values).all() and np.isfinite(control).all()
        rows = episode_summary(values, weight, ends, reasons, stood, args.rate)
        for i, row in enumerate(rows):
            first = np.flatnonzero((values[:500, i, FIELDS.index('live')]>.5) & (values[:500, i, FIELDS.index('hold')]<.5))[0]
            row['first_active_target_error_deg'] = np.rad2deg(values[first, i, :4]-control[first, i, :4]).tolist()
            row['first_active_leg_torque_Nm'] = control[first, i, [CONTROL_FIELDS.index(f'actual_{j}') for j in (0,1,3,4)]].tolist()
            row['first_failure'] = failures[i]
        np.savez_compressed(args.output/'trace.npz', values=values, fields=np.asarray(FIELDS),
            control=control, control_fields=np.asarray(CONTROL_FIELDS), weight_N=weight,
            initial_qpos=initial_qpos.cpu().numpy(), initial_qvel=initial_qvel.cpu().numpy())
        np.savez_compressed(args.output/'trajectory.npz', qpos=poses.cpu().numpy(), time_s=np.arange(steps+1)*raw.step_dt)
        report = dict(rate_rad_s=args.rate, num_envs=args.num_envs, seconds=args.seconds, seed=args.seed,
            checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), parameter_updates=0,
            actor_unchanged=True, strict_native_load=True, physics_dt_s=raw.physics_dt,
            policy_dt_s=raw.step_dt, source_files_verified=len(expected), physical_files_verified=len(assets),
            controller_meta=meta, rows=rows, limiter_max_roundtrip_error_rad=float(limiter.max_roundtrip_error),
            raw_control_change_max=float(limiter.max_control_change), trace_sha256=digest(args.output/'trace.npz'),
            trajectory_sha256=digest(args.output/'trajectory.npz'), source_sha256=digest(Path(__file__)),
            semantics='2ms target slew after original FIFO; measured-angle latch at first active substep; after first stand chase target before bypass',
            scope='Frozen simulation-only target intervention, first nominal episodes; no training or production control change')
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('controller_meta','rows')}, indent=2), flush=True)
        print(json.dumps({'stood_ids':[r['env'] for r in rows if r['stood_and_survived_40s']],
            'balanced_ids':[r['env'] for r in rows if r['final_balanced_v31']],
            'failed_ids':[r['env'] for r in rows if r['end_reason']=='failure']}, indent=2), flush=True)
    finally:
        raw.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recording', type=Path)
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=40.)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--device', default='cuda:0')
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Frozen assessment cannot train')):
        main(parser.parse_args())
