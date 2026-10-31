#!/usr/bin/env python3
"""Frozen getup release diagnosis: actual 500 Hz PD and contact, no learning."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import numpy as np
import torch


LEGS = ('lf0', 'lf00', 'rf0', 'rf00')
GROUPS = ('q', 'qd', 'nominal', 'target', 'raw', 'control', 'p_nominal',
          'p_action', 'd', 'unclipped', 'command', 'actual')
FIELDS = [f'{group}_{joint}' for group in GROUPS for joint in LEGS] + [
    'support_l_N', 'support_r_N', 'base_N', 'height_m', 'vz_m_s', 'tilt_deg',
    'hold', 'first_episode_active']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_physics(env):
    recorder = getattr(env, '_getup_release_recorder', None)
    if recorder is not None:
        recorder.sample()
    return torch.zeros(env.num_envs, device=env.device)


class Recorder:
    def __init__(self, raw, count):
        self.raw, self.pd, self.count = raw, raw.action_manager.get_term('pd'), 0
        self.live = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
        self.trace = torch.empty(count, raw.num_envs, len(FIELDS), device=raw.device)
        self.original_apply = self.pd.apply_actions
        self.root_q = raw.scene['robot'].indexing.free_joint_q_adr
        self.root_v = raw.scene['robot'].indexing.free_joint_v_adr

    def apply(self):
        pd, raw = self.pd, self.raw
        self.original_apply()
        legs = pd._leg_action_ids
        delayed = pd._action_history[
            (pd._history_head-pd.action_delay_steps) % len(pd._action_history), pd._env_ids]
        q = raw.sim.data.qpos[:, pd._motor_q_adr][:, legs]
        qd = raw.sim.data.qvel[:, pd._motor_v_adr][:, legs]
        nominal = pd.nominal_positions[:, legs]
        target = (nominal+pd.cfg.leg_position_scale*delayed[:, legs]).clamp(
            pd.reference_positions[:, legs]-3.14, pd.reference_positions[:, legs]+3.14)
        gain = pd.cfg.leg_stiffness*pd.kp_scale[:, legs]*pd.motor_strength[:, legs]
        # 同一实际状态拆分默认角、网络偏移与速度阻尼，三项相加得到限幅前需求。
        base_p = gain*(nominal-q)
        action_p = gain*(target-nominal)
        damping = -pd.cfg.leg_damping*pd.kd_scale[:, legs]*qd*pd.motor_strength[:, legs]
        unclipped = ((pd.cfg.leg_stiffness*pd.kp_scale[:, legs]*(target-q)
            - pd.cfg.leg_damping*pd.kd_scale[:, legs]*qd)*pd.motor_strength[:, legs])
        self.pending = torch.cat((q, qd, nominal, target, pd.raw_action[:, legs],
            pd._control_actions[:, legs], base_p, action_p, damping, unclipped,
            pd.torques[:, legs]), -1)
        self.hold = pd._ground_hold().clone()

    def sample(self):
        raw, pd = self.raw, self.pd
        data = raw.sim.data
        q = data.qpos[:, self.root_q]
        v = data.qvel[:, self.root_v]
        support = -raw.scene['wheel_contact'].data.force[:, :, 2]
        base = raw.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
        tilt = torch.rad2deg(torch.acos((1.-2.*(q[:, 4]**2+q[:, 5]**2)).clamp(-1., 1.)))
        self.trace[self.count] = torch.cat((self.pending,
            data.qfrc_actuator[:, pd._motor_v_adr][:, pd._leg_action_ids], support,
            torch.stack((base, q[:, 2]-raw.scene.env_origins[:, 2], v[:, 2], tilt,
                         self.hold.float(), self.live.float()), -1)), -1)
        self.count += 1


def summarize(values, dt, failures, ends):
    columns = {name: i for i, name in enumerate(FIELDS)}
    field = lambda name: values[:, :, columns[name]]
    group = lambda name: values[:, :, [columns[f'{name}_{j}'] for j in LEGS]]
    live, hold = field('first_episode_active') > .5, field('hold') > .5
    active = live & ~hold
    demand, command = group('unclipped'), group('command')
    composed = group('p_nominal')+group('p_action')+group('d')
    assert np.allclose(demand, composed, atol=1e-4, rtol=1e-5)
    expected = np.where(hold[:, :, None], 0., np.clip(demand, -40., 40.))
    assert np.allclose(command, expected, atol=1e-5, rtol=0.)
    assert (command[hold] == 0).all() and np.isfinite(values).all()
    rows = []
    for i in range(values.shape[1]):
        first = int(np.flatnonzero(active[:, i])[0])
        support = values[:, i, [columns['support_l_N'], columns['support_r_N']]]
        loaded = support.max(-1) > 1.
        first_contact = np.flatnonzero(live[:, i] & loaded)
        # 仅记录接触过以后首次双轮无支撑；不是凭机身高度推断离地。
        unloaded = np.flatnonzero(live[:, i] & (np.arange(len(values)) >= first)
                                  & ~loaded & (np.maximum.accumulate(loaded)))
        sat = (np.abs(command[:, i]) >= 39.999).any(-1)
        base_only = group('p_nominal')[:, i]+group('d')[:, i]
        until = active[:, i] & (np.arange(len(values))*dt < min(1., ends[i]))
        at = lambda name: group(name)[first, i].tolist()
        rows.append(dict(env=i, first_active_start_s=first*dt,
            first_active_q_rad=at('q'), first_active_q_deg=np.rad2deg(at('q')).tolist(),
            first_active_qd_rad_s=at('qd'), nominal_rad=at('nominal'), target_rad=at('target'),
            first_active_target_error_deg=np.rad2deg(np.array(at('target'))-at('q')).tolist(),
            first_active_raw_action=at('raw'), first_active_control_action=at('control'),
            first_active_p_nominal_Nm=at('p_nominal'), first_active_p_action_Nm=at('p_action'),
            first_active_d_Nm=at('d'), first_active_unclipped_Nm=at('unclipped'),
            first_active_command_Nm=at('command'), first_active_actual_Nm=at('actual'),
            zero_leg_action_same_state_demand_Nm=base_only[first].tolist(),
            first_active_support_post_step_N=support[first].tolist(),
            first_contact_post_step_s=float((first_contact[0]+1)*dt) if len(first_contact) else None,
            first_unloaded_after_release_post_step_s=float((unloaded[0]+1)*dt) if len(unloaded) else None,
            active_first_second_duration_s=float(until.sum()*dt),
            any_leg_saturation_duration_s=float((sat & until).sum()*dt),
            per_leg_saturation_duration_s=(((np.abs(command[:, i]) >= 39.999) & until[:, None]).sum(0)*dt).tolist(),
            first_episode_end_s=ends[i], first_failure=failures[i]))
    return dict(rows=rows, equations_verified=True, waiting_torques_zero=True,
        finite=True, actual_vs_command_max_abs_Nm=float(np.abs(group('actual')[live]-command[live]).max()),
        first_active_all_legs_saturated_count=sum(all(abs(x)>=39.999 for x in r['first_active_command_Nm']) for r in rows),
        zero_leg_same_state_any_saturation_count=sum(any(abs(x)>=40. for x in r['zero_leg_action_same_state_demand_Nm']) for r in rows))


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    training = args.training_artifact.resolve()
    launch = json.loads((training/'launch.json').read_text())
    checkpoint = Path(launch['log_dir'])/'model_999.pt'
    expected = json.loads((training/'source-sha256.json').read_text())
    snapshot = Path(launch['log_dir'])/'snapshot/ct_mjlab'
    assert all(digest(snapshot/p)==h for p, h in expected.items())
    frozen = args.output.resolve()/'frozen-runtime'
    source = frozen/'mjlab_training/src/ct_mjlab'
    shutil.copytree(snapshot, source, ignore=shutil.ignore_patterns('__pycache__'))
    (frozen/'resources').symlink_to(root/'resources', target_is_directory=True)
    assert all(digest(source/p)==h for p, h in expected.items())
    assets = json.loads((training/'physical-files-sha256.json').read_text())
    assert all(digest(root/p)==h for p, h in assets.items())
    sys.path.insert(0, str(source.parent))
    from ct_mjlab.envs import task_registry
    from mjlab.managers.metrics_manager import MetricsTermCfg
    assert Path(sys.modules[type(task_registry).__module__].__file__).resolve().is_relative_to(frozen)
    cfg, agent = task_registry.get_cfgs('chuanliantui_getup', play=True)
    cfg.auto_reset, cfg.seed = False, args.seed
    cfg.metrics['getup_release'] = MetricsTermCfg(func=sample_physics, per_substep=True, reduce='last')
    wrapped, _ = task_registry.make_env('chuanliantui_getup', env_cfg=cfg,
        train_cfg=agent, num_envs=args.num_envs, device=args.device)
    raw = wrapped.unwrapped
    try:
        assert raw.physics_dt == .002 and raw.step_dt == .01
        runner, _ = task_registry.make_alg_runner(wrapped, 'chuanliantui_getup', train_cfg=agent, device=args.device)
        runner.load(str(checkpoint), load_cfg={'actor':True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        before = {k:v.cpu().clone() for k,v in runner.alg.actor.state_dict().items()}
        obs, _ = raw.reset(seed=args.seed)
        initial = raw.sim.data.qpos.cpu().numpy().copy()
        if args.recording:
            with np.load(args.recording/'trajectory.npz') as previous:
                assert np.array_equal(initial, previous['qpos'][0])
        state = raw.command_manager.get_term('standup')
        assert state.ground_start.all()
        recorder = Recorder(raw, round(args.seconds/raw.physics_dt))
        raw._getup_release_recorder = recorder
        pd = recorder.pd
        assert not pd.action_delay_steps.any() and not pd.wheel_action_delay_steps.any()
        if args.mode == 'motors-off':
            # 仅诊断关闭六电机，气弹簧和碰撞物理保持。
            pd.motor_strength.zero_()
        elif args.mode == 'legs-off':
            pd.motor_strength[:, pd._leg_action_ids] = 0.
        failures, ends = [None]*args.num_envs, [args.seconds]*args.num_envs
        policy_records = []
        with patch.object(pd, 'apply_actions', recorder.apply):
            for tick in range(round(args.seconds/raw.step_dt)):
                action = policy(obs)
                if agent.clip_actions is not None:
                    action = action.clamp(-agent.clip_actions, agent.clip_actions)
                if args.mode == 'zero-leg':
                    action[:, pd._leg_action_ids] = 0.
                obs, _, terminated, timeout, _ = raw.step(action)
                done = terminated | timeout
                live = recorder.live.clone()
                policy_records.append(torch.stack((state.has_landed.float(), state.wheels_airborne_steps.float(),
                    state.has_stood.float(), live.float(), terminated.float(), timeout.float()), -1).cpu().numpy())
                for i in (live & done).nonzero().flatten().tolist():
                    ends[i] = (tick+1)*raw.step_dt
                    failures[i] = dict(inverted=bool(raw.scene['robot'].data.projected_gravity_b[i, 2]>0),
                        airborne_ticks=int(state.wheels_airborne_steps[i]), has_stood=bool(state.has_stood[i]))
                recorder.live &= ~done
                if tick+1 < round(args.seconds/raw.step_dt) and done.any():
                    raw.reset(env_ids=done.nonzero(as_tuple=False).flatten())
                    obs = raw.get_observations()
        assert recorder.count == round(args.seconds/raw.physics_dt)
        assert all(torch.equal(v.cpu(), before[k]) for k,v in runner.alg.actor.state_dict().items())
        values = recorder.trace.cpu().numpy()
        np.savez_compressed(args.output/'trace.npz', values=values, fields=np.asarray(FIELDS),
            time_start_s=np.arange(len(values))*raw.physics_dt, initial_qpos=initial,
            policy=np.asarray(policy_records), policy_fields=np.asarray([
                'has_landed', 'wheels_airborne_steps', 'has_stood', 'first_episode_active', 'terminated', 'timeout']))
        report = dict(mode=args.mode, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), seed=args.seed,
            num_envs=args.num_envs, seconds=args.seconds, physics_dt_s=raw.physics_dt,
            source_files_verified=len(expected), physical_files_verified=len(assets),
            actor_unchanged=True, parameter_updates=0, strict_native_load=True,
            initial_qpos_matches_recording=True if args.recording else None,
            sampling='PD inputs before 2ms step; actual torque/contact/height after step. Derived contact follows engine substep timing.',
            scope='Nominal play initial states; first episodes only; no training, control change or formal behavior acceptance',
            **summarize(values, raw.physics_dt, failures, ends))
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='rows'}, indent=2), flush=True)
    finally:
        raw.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-artifact', type=Path, required=True)
    parser.add_argument('--recording', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--mode', choices=('policy', 'zero-leg', 'legs-off', 'motors-off'), default='policy')
    parser.add_argument('--device', default='cuda:0')
    with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('Diagnostic cannot train')):
        main(parser.parse_args())
