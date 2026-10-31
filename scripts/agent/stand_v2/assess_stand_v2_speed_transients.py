#!/usr/bin/env python3
"""Frozen ±0.5/1/1.5m/s start/stop/reverse acceptance; separate zero-speed v3.1."""
from scripts.agent.motion.assess import STANDARD, THRESHOLDS, stand_phase, settled_window

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import mujoco
from mjlab.managers.metrics_manager import MetricsTermCfg
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_stand_v2 import mdp
from ct_mjlab.envs.chuanliantui_stand_v2.rewards import restore_tracking_reward_cfg, configure_tight_reward
from ct_mjlab.envs.chuanliantui_stand_v2.observations import get_state
from scripts.agent.stand_v2.assess_chuanliantui_stand_v2 import Recorder, FIELDS, sample_physics, summarize, digest
from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import tensor_digest

# Extra-long deterministic segments allow the slowest ramp to reach its target.
SEGMENTS = ((8., 14., 'start', 1.), (14., 18., 'hold', 1.), (18., 24., 'stop', 0.),
            (24., 30., 'start', 1.), (30., 36., 'reverse', -1.), (36., 40., 'stop', 0.))


class SpeedRecorder(Recorder):
    def __init__(self, raw):
        super().__init__(raw)
        self.trace = torch.empty((20000, raw.num_envs, len(FIELDS)), device=raw.device)
        self.command_trace = torch.empty((20000, raw.num_envs, 5), device=raw.device)

    def sample(self):
        i = self.count
        super().sample()
        q = self.raw.sim.data.qpos[:, self.root_qadr]
        v = self.raw.sim.data.qvel[:, self.root_vadr]
        w, x, y, z = q[:, 3:].unbind(-1)
        vx = ((1-2*(y*y+z*z))*v[:, 0] + 2*(x*y+w*z)*v[:, 1] + 2*(x*z-w*y)*v[:, 2])
        s = self.raw.command_manager.get_term('standup')
        self.command_trace[i] = torch.stack((s.command[:, 0], s.tracking_velocity, vx,
                                            s.tracking_active.float(), v[:, 5]), -1)


def first_hold(mask, dt, hold_s=.5):
    count = 0
    for i, valid in enumerate(mask):
        count = count+1 if valid else 0
        if count >= round(hold_s/dt):
            return (i+1)*dt
    return None


def segment_metrics(p, c, target, start, kind, dt):
    """Uses all physics samples; never conditions success on only surviving samples."""
    live_finite = (p[:, 12] > .5) & (p[:, 13] > .5)
    full = bool(live_finite.all() and (c[:, 3] > .5).all())
    available = live_finite & (c[:, 3] > .5)
    settled = (np.abs(c[:, 0]-target) <= 1e-5) & available
    # Discard first0.5s after the ramp actually reaches its target.
    steady = settled_window(settled, dt)
    enough = bool(steady.sum()*dt >= .5)
    error = float(np.abs(c[steady, 2]-c[steady, 0]).mean()) if enough else None
    tilt = np.rad2deg(np.arccos((-p[:, 14]).clip(-1., 1.)))
    arrival = first_hold((np.abs(c[:, 2]-target) <= .30) & available, dt)
    stopped = first_hold((np.linalg.norm(p[:, 7:9], axis=-1) <= .05) & available, dt)
    contact = bool(((p[:, 2] > 1.) & live_finite).any())
    inverted = bool(((p[:, 14] > 0.) & live_finite).any())
    max_tilt = float(tilt[live_finite].max()) if live_finite.any() else None
    motion = stand_phase(p, c, selected=steady, vx_target=target, complete=bool(full and enough))
    row = dict(motion_standard=motion, passed=motion["passed"], kind=kind, start_s=start, target_vx_m_s=float(target), complete_gate_and_live=full,
        steady_measured_s=float(steady.sum()*dt), steady_mae_m_s=error,
        steady_error_mean_m_s=float((c[steady, 2]-c[steady, 0]).mean()) if enough else None,
        max_tilt_deg=max_tilt,
        max_abs_pitch_deg=float(np.rad2deg(p[live_finite, 4].max())) if live_finite.any() else None,
        body_contact=contact, body_contact_s=float(((p[:, 2] > 1.) & live_finite).sum()*dt), inverted=inverted,
        arrival_confirmed_s=arrival,
        diagnostic_passed=bool(full and enough and max_tilt <= 25. and not contact and not inverted))
    if kind == 'stop':
        # Distance is traveled path from target-zero request until0.5s stop confirmation.
        stop_index = len(p) if stopped is None else round(stopped/dt)
        ended = np.flatnonzero(~live_finite)
        if len(ended):
            stop_index = min(stop_index, int(ended[0]))
        xy = p[:stop_index, 9:11]
        row.update(braking_confirmed_s=stopped,
            braking_path_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum()) if len(xy) else None,
            stopped_within_window=stopped is not None)
        row['diagnostic_passed'] &= stopped is not None
    return row


def tracking_accuracy(rows):
    """保持段分母包含失稳/未就绪/测量不足；误差均值另报有效段数。"""
    def group(selected):
        holds = [s for r in selected for s in r['segments'] if s['kind'] == 'hold']
        measured = [s for s in holds if s['complete_gate_and_live'] and s['steady_mae_m_s'] is not None]
        counts = {str(limit): sum(s['steady_mae_m_s'] <= limit for s in measured) for limit in (.30, .10)}
        return dict(hold_segments=len(holds), measured_segments=len(measured), mae_counts=counts,
            mae_fractions={k: v/len(holds) if holds else None for k,v in counts.items()},
            steady_error_mean_m_s=float(np.mean([s['steady_error_mean_m_s'] for s in measured])) if measured else None,
            steady_mae_mean_m_s=float(np.mean([s['steady_mae_m_s'] for s in measured])) if measured else None)
    return dict(all_speeds=group(rows), by_speed={str(v): group([r for r in rows if r['speed_m_s'] == v])
        for v in sorted({r['speed_m_s'] for r in rows})})


def selfcheck():
    p = np.zeros((3000, 15)); c = np.zeros((3000, 5))
    p[:, 12:14] = 1.; p[:, 14] = -1.; c[:, 3] = 1.
    row = segment_metrics(p, c, 0., 8., 'stop', .002)
    assert row['passed'] and row['braking_confirmed_s'] == .5 and row['braking_path_m'] == 0.
    c[:, 2] = .31
    assert not segment_metrics(p, c, 0., 8., 'hold', .002)['passed']
    c[:, 2] = .10
    tight = segment_metrics(p, c, 0., 8., 'hold', .002)
    c[:, 3] = 0.
    invalid = segment_metrics(p, c, 0., 8., 'hold', .002)
    summary = tracking_accuracy([dict(speed_m_s=.5, segments=[tight, invalid])])['all_speeds']
    assert summary['mae_counts'] == {'0.3': 1, '0.1': 1} and summary['mae_fractions']['0.1'] == .5
    assert abs(summary['steady_error_mean_m_s']-.1) < 1e-12
    assert tracking_accuracy([])['all_speeds']['mae_fractions']['0.1'] is None
    c[:, 3] = 1.
    c[:, 2] = 0.; p[100, 14] = -np.cos(np.deg2rad(26))
    assert not segment_metrics(p, c, 0., 8., 'start', .002)['diagnostic_passed']
    p[:, 14] = -1.; p[100, 2] = 2.
    assert not segment_metrics(p, c, 0., 8., 'stop', .002)['diagnostic_passed']
    p[:, 2] = 0.; p[100, 13] = 0.
    assert not segment_metrics(p, c, 0., 8., 'hold', .002)['passed']
    print('Transient reducer selfcheck passed (no simulation or learning)')


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    contract = saved['infos'].get('command_contract', mdp.ZERO_COMMAND_CONTRACT)
    nominal = args.conditions == 'nominal'
    cfg, agent = task_registry.get_cfgs('chuanliantui_stand_v2', play=nominal)
    cfg.auto_reset = False
    cfg.metrics['speed_physics'] = MetricsTermCfg(func=sample_physics, per_substep=True, reduce='last')
    # Build with the original zero-speed RNG/reset path, then match only loading metadata.
    env, _ = task_registry.make_env('chuanliantui_stand_v2', env_cfg=cfg, train_cfg=agent,
        num_envs=args.num_envs, seed=args.seed, device=args.device)
    try:
        raw, s = env.unwrapped, env.unwrapped.command_manager.get_term('standup')
        s.cfg.post_stand_tracking = contract != mdp.ZERO_COMMAND_CONTRACT
        restore_tracking_reward_cfg(s.cfg, saved['infos'])
        if s.cfg.tracking_tight_term:
            from mjlab.managers.reward_manager import RewardManager
            configure_tight_reward(raw.cfg, s.cfg)
            raw.reward_manager = RewardManager(raw.cfg.rewards, raw, scale_by_dt=raw.cfg.scale_rewards_by_dt)
        s.cfg.speed_curriculum = contract == mdp.SPEED_COMMAND_CONTRACT
        if s.cfg.speed_curriculum:
            for k, v in saved['infos']['standup_curriculum']['speed_curriculum']['config'].items():
                setattr(s.cfg, k, v)
        runner, _ = task_registry.make_alg_runner(env, 'chuanliantui_stand_v2', train_cfg=agent, device=args.device)
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No learning')):
            runner.load(str(args.checkpoint), load_cfg={'actor': True}, map_location=args.device)
            frozen = tensor_digest(runner.alg.actor.state_dict())
            s.cfg.post_stand_tracking = s.cfg.speed_curriculum = True
            s.speed_curriculum_frozen = True
            r = SpeedRecorder(raw)
            raw._stand_v2_acceptance = r
            n, dt = raw.num_envs, raw.physics_dt
            assert dt == .002 and raw.step_dt == .01 and raw.max_episode_length == 4000
            moving = torch.arange(n, device=raw.device) >= n//2
            speeds = torch.tensor([(.5, -.5, 1., -1., 1.5, -1.5)[i % 6] for i in range(n)], device=raw.device)
            s.tracking_episode.copy_(moving)
            s.zero_hold.copy_(~moving)
            s.tracking_acceleration[:] = args.acceleration
            s.tracking_velocity.zero_()
            s.tracking_interval_steps[:] = 1000000
            s.tracking_ready_steps.zero_()
            s._command[:, :2] = 0.
            s.segment_started.zero_()
            schedule_tick = 0
            original_update = s._update_command
            issued_targets = torch.zeros(n, device=raw.device)
            def command(env_ids):
                t = schedule_tick*raw.step_dt
                multiplier = next((v for a,b,_,v in SEGMENTS if a <= t < b), 0.)
                s.tracking_episode.copy_(moving & r.live)
                issued_targets.copy_(torch.where(moving & r.live, speeds*multiplier, 0.))
                s.tracking_velocity.copy_(issued_targets)
                s.tracking_interval_steps[:] = 1000000
                s.segment_started.zero_()
                original_update(env_ids)
            s._update_command = command
            raw.observation_manager.reset()
            raw.obs_buf = raw.observation_manager.compute(update_history=True)
            obs, policy = env.get_observations(), runner.get_inference_policy(device=args.device)
            initial = {k: v.cpu().numpy().copy() for k, v in dict(qpos=raw.sim.data.qpos,
                qvel=raw.sim.data.qvel, mass=raw.sim.model.body_mass,
                friction=raw.sim.model.geom_friction).items()}
            action, observation = raw.action_manager.get_term('pd'), get_state(raw)
            for k, v in dict(inertia=raw.sim.model.body_inertia, com=raw.sim.model.body_ipos,
                solref=raw.sim.model.geom_solref, kp=action.kp_scale, kd=action.kd_scale,
                strength=action.motor_strength, nominal=action._nominal_pos,
                action_delay=action.action_delay_steps, observation_delay=observation.delay.lags).items():
                initial[k] = v.cpu().numpy().copy()
            np.savez_compressed(args.output/'initial.npz', **initial)
            stood = torch.zeros_like(r.live)
            if args.record_qpos:
                qpos = torch.empty((4001,n,raw.sim.data.qpos.shape[-1]),device=raw.device)
                qpos[0].copy_(raw.sim.data.qpos)
                first_end = torch.full((n,),float('nan'),device=raw.device)
                mujoco.mj_saveModel(raw.sim.mj_model,str(args.output/'scene.mjb'),None)
            gate_violations = slope_violations = 0
            for tick in range(4000):
                before = s.command[:, 0].clone()
                obs, _, done, _ = env.step(policy(obs))
                stood |= r.live & s.has_stood
                bad = (s.command[:, 0] != 0.) & (~s.has_stood | s.recover | s.fallen | ~s.tracking_episode | (s.tracking_ready_steps < 50))
                gate_violations += int(bad.sum())
                slope_violations += int((s.tracking_active & ((s.command[:, 0]-before).abs() > args.acceleration*.01+2e-7)).sum())
                failed = r.live & done.bool() & (s._isaac_failure | (tick < 3999))
                if args.record_qpos:
                    qpos[tick+1].copy_(raw.sim.data.qpos)
                    first_end[failed] = (tick+1)*raw.step_dt
                r.live &= ~failed
                if done.any() and tick < 3999:
                    raw.reset(env_ids=done.nonzero(as_tuple=False).flatten())
                    obs = env.get_observations()
                schedule_tick = tick+1
                if (tick+1) % 1000 == 0:
                    print(f'Transient acceptance {(tick+1)/100:.0f}/40s', flush=True)
            assert r.count == 20000
            assert frozen == tensor_digest(runner.alg.actor.state_dict())
            assert gate_violations == slope_violations == 0
            p, c = r.trace.cpu().numpy(), r.command_trace.cpu().numpy()
            weights = raw.sim.model.body_mass.sum(-1).cpu().numpy()*np.linalg.norm(raw.sim.mj_model.opt.gravity)
            rollout = dict(episodes=[dict(ever_isaac_stood=bool(stood[i]), survived_to_end=bool(r.live[i])) for i in range(n)])
            balance_rows = summarize(p, weights, rollout, dt)
            rows = []
            for i in range(n//2, n):
                segments = []
                for a,b,kind,multiplier in SEGMENTS:
                    sl = slice(round(a/dt), round(b/dt))
                    segments.append(segment_metrics(p[sl,i], c[sl,i], float(speeds[i])*multiplier, a, kind, dt))
                rows.append(dict(env=i, speed_m_s=float(speeds[i]), ever_stood=bool(stood[i]),
                    passed=bool(stood[i] and all(v['passed'] for v in segments)), segments=segments))
            zero = balance_rows[:n//2]
            np.savez_compressed(args.output/'trajectories.npz', physical=p, command=c,
                                weight=weights, moving=moving.cpu().numpy())
            if args.record_qpos:
                np.savez_compressed(args.output/'qpos.npz',qpos=qpos.cpu().numpy(),
                    time_s=np.arange(4001)*raw.step_dt,first_end_s=first_end.cpu().numpy())
            sources = [Path(__file__), Path('scripts/agent/motion/assess.py'), Path('scripts/agent/stand_v2/assess_chuanliantui_stand_v2.py')]
            report = dict(sources={str(p.resolve()):digest(p) for p in sources},
                trajectories_sha256=digest(args.output/'trajectories.npz'), physics_dt_s=raw.physics_dt,
                motion_standard=STANDARD, motion_thresholds=THRESHOLDS.copy(), contract='stand-v2-speed-transients-v1', parameter_updates=0,
                checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
                conditions=args.conditions, seed=args.seed, num_envs=n, duration_s=40.,
                acceleration_m_s2=args.acceleration, gate_violations=gate_violations,
                qpos_recorded=args.record_qpos,
                slope_violations=slope_violations, physical_fields=FIELDS,
                command_fields=['ramped_vx', 'hidden_target_vx', 'body_vx', 'tracking_gate', 'body_yaw_rate'],
                zero_balance_v31=dict(total=len(zero), passed=sum(v['passed'] for v in zero), episodes=zero),
                moving=dict(total=len(rows), passed=sum(v['passed'] for v in rows), episodes=rows),
                tracking_accuracy=tracking_accuracy(rows),
                scope='Deterministic external transition schedule; zero balance separate; no H7/CPU approval',
                thresholds=dict(steady_mae_m_s=THRESHOLDS["vx_mae_m_s"], tilt_deg=25., body_contact_N=1., stop_speed_m_s=THRESHOLDS["parking_horizontal_peak_m_s"],
                    arrival_hold_s=.5, steady_wait_after_ramp_s=.5, steady_min_samples_s=.5))
            (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k:v for k,v in report.items() if k not in ('zero_balance_v31','moving')}, indent=2))
    finally:
        env.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--num-envs', type=int, default=128)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--conditions', choices=('nominal','training'), default='training')
    p.add_argument('--acceleration', type=float, default=1.)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--selfcheck', action='store_true')
    p.add_argument('--record-qpos', action='store_true',help='Save exact100Hz poses for video; no re-integration')
    args = p.parse_args()
    if args.selfcheck:
        selfcheck()
    else:
        if not args.checkpoint or not args.output or args.num_envs < 12 or args.num_envs % 2 or not .5 <= args.acceleration <= 2.:
            p.error('Need checkpoint/output, even num-envs>=12, acceleration in[0.5,2]')
        main(args)
