"""固定 Jump5200 的 GPU 起跳做功采集与离线报告。"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.config import jump_config
from ct_mjlab.envs.chuanliantui.fudan_ppo_jump.env import JumpEnv
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha
from .evaluate import SEED, STEPS, WORLDS, WALK, policies
from .handoff import LandingHandoff, expected_landing_mode
from .measure_flat import events, summarize

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT/'logs/mjlab/fudan_ppo_jump/20261119_preload4200_to5200_4096_01/model_final.pt'
CHECKPOINT_SHA = '1f1ae7d029c6f741a84b6b703d2ce496a4787c4d29c0934deeb9582c2aa17f80'
REFERENCE = ROOT/'artifacts/fudan_ppo_jump/20261119-max-height5200-01/flat/walk-2.0'
MOTORS = ('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')
SPRINGS = ('left_spring2_joint', 'right_spring2_joint')
ACTUATORS = tuple(n+'_motor' for n in MOTORS) + ('left_gas_spring_motor', 'right_gas_spring_motor')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


class Recorder:
    def __init__(self, env):
        self.env = env
        self.rows = []
        self.slices = {}
        self.held = {}
        model, action = env.env.sim.mj_model, env.action
        self.ids = [model.actuator('robot/'+name).id for name in ACTUATORS]
        self.metadata = dict(
            motor_qadr=action._motor_q_adr.cpu().tolist(),
            motor_vadr=action._motor_v_adr.cpu().tolist(),
            spring_qadr=[int(model.joint('robot/'+n).qposadr[0]) for n in SPRINGS],
            spring_vadr=[int(model.joint('robot/'+n).dofadr[0]) for n in SPRINGS],
            actuator_names=ACTUATORS, actuator_ids=self.ids,
            base_body_id=int(model.body('robot/base_link').id), mass_kg=float(model.body_mass.sum()),
            physics_dt=.001, pd_dt=.005, policy_dt=.01,
            sampling='pre is after controls and before integration; post is after integration; trace geometry/contact uses solver input state')
        assert model.opt.timestep == .001
        assert action.cfg.pd_decimation == 5
        assert not action.cfg.disturbances.action_delay
        assert not action.wheel_motor.enabled and not action._wheel_speed_shaping
        assert np.all(model.actuator_gainprm[self.ids, 0] == 1.)
        assert np.all(model.actuator_gear[self.ids, 0] == 1.)
        apply, step, frame = action.apply_actions, env.env.sim.step, env.trace_frame

        def applied():
            refresh = action._pd_tick == 0
            apply()
            legs = action._leg_action_ids
            if refresh:
                delayed = action._action_history[
                    (action._history_head-action.action_delay_steps) % len(action._action_history),
                    action._env_ids][:, legs]
                center = action._reference_pos[:, legs]
                target = (action._nominal_pos[:, legs] + action.cfg.leg_position_scale*delayed).clamp(center-3.14, center+3.14)
                data = env.env.sim.data
                q = data.qpos[:, action._motor_q_adr[legs]]
                v = data.qvel[:, action._motor_v_adr[legs]]
                requested = (action.cfg.leg_stiffness*action.kp_scale[:, legs]*(target-q)
                             - action.cfg.leg_damping*action.kd_scale[:, legs]*v)*action.motor_strength[:, legs]
                self.held = dict(target=target.clone(), requested=requested.clone())
            self.refresh = refresh

        def stepped():
            data = env.env.sim.data
            # 力矩与本次积分的起、终状态配对，避免混入下一次 PD 刷新。
            self.pre = dict(qpos_pre=data.qpos.clone(), qvel_pre=data.qvel.clone(),
                            pd_torque=action.torques.clone(), ctrl=data.ctrl[:, self.ids].clone())
            return step()

        def framed():
            trace = frame()
            data = env.env.sim.data
            delta = data.xpos[:, env.wheel_ids]-data.xanchor[:, env.hip_ids]
            quat = env.robot.data.root_link_quat_w[:, None].expand(-1, 2, -1)
            local = quat_apply_inverse(quat.reshape(-1, 4), delta.reshape(-1, 3)).reshape(-1, 2, 3)
            length = local[:, :, (0, 2)].norm(dim=-1)
            values = dict(self.pre, **self.held, qpos_post=data.qpos, qvel_post=data.qvel,
                          actual_force=data.actuator_force[:, self.ids], length=length,
                          refresh=torch.full_like(length[:, :1], float(self.refresh)))
            chunks, offset = [], 0
            for name, value in values.items():
                flat = value.reshape(env.num_envs, -1)
                self.slices[name] = [offset, offset+flat.shape[1]]
                offset += flat.shape[1]
                chunks.append(flat)
            self.rows.append(torch.cat(chunks, dim=1))
            return trace

        action.apply_actions, env.env.sim.step, env.trace_frame = applied, stepped, framed

    def arrays(self):
        batch = torch.stack(self.rows).cpu().numpy()
        return {name: batch[:, :, start:end] for name, (start, end) in self.slices.items()}


@torch.inference_mode()
def collect_profile(output, walk, jump, *, probe=False, height=.26, speed=2., modify_action=None):
    output.mkdir()
    env = JumpEnv(WORLDS, SEED, 'cuda:0', False, cfg=jump_config(), record_flat=True)
    try:
        torch.manual_seed(SEED)
        env.reset_idx(torch.arange(WORLDS, device=env.device))
        command = env.commands.new_tensor([speed, 0., height])
        env.commands[:] = command
        obs, _ = env.observations()
        history = obs.repeat(1, 5)
        env.history.copy_(history)
        env._history_needs_fill.zero_()
        initial = [a.cpu().numpy().copy() for a in (env.env.sim.data.qpos, env.env.sim.data.qvel, history)]
        recorder = Recorder(env) if probe else None
        if probe:
            mujoco.mj_saveModel(env.env.sim.mj_model, str(output.parent/'model.mjb'))
            write_json(output.parent/'metadata.json', recorder.metadata)
        start = torch.full((WORLDS,), -1, device=env.device)
        handoff = LandingHandoff(start, buffer_ticks=8)
        live = torch.ones_like(start, dtype=torch.bool)
        count = np.zeros(WORLDS, dtype=int)
        traces, modes, actions = [], [], []
        for tick in range(STEPS):
            if tick == 200:
                start[live] = tick
            active = handoff.active(tick) & live
            env.commands[:] = command
            obs, _ = env.observations()
            history[:, -19:-16] = obs[:, 6:9]
            action, _ = jump.act_inference(obs, history)
            walk_obs, walk_hist = obs.clone(), history.reshape(WORLDS, 5, 25).clone()
            walk_obs[:, 6] *= 2/3
            walk_hist[:, :, 6] *= 2/3
            walk_action, _ = walk.act_inference(walk_obs, walk_hist.flatten(1))
            action = torch.where(active[:, None], action, walk_action)
            if modify_action is not None:
                action = modify_action(env, action, active, tick)
            actions.append(action.cpu().numpy().copy())
            modes.append(active.cpu().numpy())
            count += live.cpu().numpy()
            env.substeps.clear()
            obs, _, _, done, _, history = env.step(action, auto_reset=False, resample=False)
            if len(env.substeps) != 10 or env.env.sim.wp_data.overflow.numpy().any():
                raise RuntimeError('Incomplete GPU recording or solver capacity overflow')
            rows = torch.stack(env.substeps)
            handoff.observe(tick, rows, active)
            traces.append(rows.cpu().numpy())
            live &= ~done
            if not live.any():
                break
            if done.any():
                env.reset_idx(done.nonzero().flatten())
                env.commands[:] = command
            if tick % 100 == 99:
                print(f'{output.name}: {tick+1}/{STEPS} policy steps', flush=True)
        traces, modes, actions = np.concatenate(traces), np.asarray(modes), np.asarray(actions)
        extra = recorder.arrays() if recorder else {}
        cases = []
        for idx, n in enumerate(count):
            trace, mode = traces[:n*10, idx], modes[:n, idx]
            assert np.isfinite(trace).all() and not trace[:, 20].any()
            expected, _ = expected_landing_mode(trace, 200, buffer_ticks=8)
            assert np.array_equal(mode, expected)
            case = dict(id=idx, triggered=bool(mode.any()), jumps=events(trace, mode),
                        raw_highest_lower_wheel_m=float(trace[:, 6:8].min(1).max()))
            cases.append(case)
            np.savez_compressed(output/f'start{idx:02}.npz', trace=trace, mode=mode, actions=actions[:n, idx],
                                qpos0=initial[0][idx], qvel0=initial[1][idx], history0=initial[2][idx],
                                **{name: arr[:n*10, idx] for name, arr in extra.items()})
        write_json(output/'cases.json', cases)
        return summarize(cases)
    finally:
        env.close()


def audit(output, reference):
    cases = []
    for idx in range(WORLDS):
        name = f'start{idx:02}.npz'
        with np.load(reference/name) as old, np.load(output/'baseline'/name) as base, np.load(output/'probe'/name) as probe, np.load(output/'baseline_repeat'/name) as repeat:
            for key in ('qpos0', 'qvel0', 'history0'):
                assert np.array_equal(old[key], base[key]) and np.array_equal(base[key], probe[key]) and np.array_equal(base[key], repeat[key]), (idx, key)
            exact = {key: bool(np.array_equal(base[key], probe[key])) for key in ('actions', 'mode', 'trace')}
            assert len(probe['trace']) == 6000
            for key in probe.files:
                assert np.isfinite(probe[key]).all(), (idx, key)
            assert np.array_equal(probe['qpos_post'][:-1], probe['qpos_pre'][1:])
            assert np.array_equal(probe['qvel_post'][:-1], probe['qvel_pre'][1:])
            legs = [0, 1, 3, 4]
            pd_error = float(np.max(abs(probe['pd_torque'][:, legs]-np.clip(probe['requested'], -40., 40.))))
            ctrl_error = float(np.max(abs(probe['ctrl'][:, :6]-probe['pd_torque'])))
            force_error = float(np.max(abs(probe['actual_force']-probe['ctrl'])))
            assert max(pd_error, ctrl_error, force_error) < 1e-4
            assert np.array_equal(probe['refresh'][:, 0], (np.arange(6000) % 5 == 0))
            before, after = events(old['trace'], old['mode'])[0], events(probe['trace'], probe['mode'])[0]
            peak_delta = after['peak_lower_wheel_m']-before['peak_lower_wheel_m']
            takeoff_delta = after['takeoff_s']-before['takeoff_s']
            baseline_jump = events(base['trace'], base['mode'])[0]
            repeat_jump = events(repeat['trace'], repeat['mode'])[0]
            baseline_first_difference = np.flatnonzero(np.any(base['trace'] != repeat['trace'], axis=1))
            probe_first_difference = np.flatnonzero(np.any(base['trace'] != probe['trace'], axis=1))
            cases.append(dict(id=idx, initial_exact=True, probe_exact=exact,
                              baseline_repeat_trace_exact=bool(np.array_equal(base['trace'], repeat['trace'])),
                              baseline_repeat_first_difference_ms=int(baseline_first_difference[0]) if len(baseline_first_difference) else None,
                              probe_first_difference_ms=int(probe_first_difference[0]) if len(probe_first_difference) else None,
                              baseline_repeat_peak_difference_m=repeat_jump['peak_lower_wheel_m']-baseline_jump['peak_lower_wheel_m'],
                              probe_vs_baseline_peak_difference_m=after['peak_lower_wheel_m']-baseline_jump['peak_lower_wheel_m'],
                              historical_trace_exact=bool(np.array_equal(old['trace'], probe['trace'])),
                              peak_difference_m=peak_delta, takeoff_difference_s=takeoff_delta,
                              within_historical_tolerance=bool(abs(peak_delta) <= .01 and abs(takeoff_delta) <= .01000001),
                              pd_formula_error_Nm=pd_error, ctrl_error_Nm=ctrl_error, force_error_Nm=force_error,
                              pd_refreshes=int(probe['refresh'].sum())))
    result = dict(cases=cases, capture_checks_passed=True,
                  historical_tolerance_passed=all(c['within_historical_tolerance'] for c in cases),
                  bitwise_probe_equivalence_passed=all(all(c['probe_exact'].values()) for c in cases),
                  interpretation='Initial states and force/state pairing are checked. Repeated ordinary GPU rollouts quantify numerical variability. Exact probe neutrality requires bitwise equivalence; otherwise attribution uses this newly measured rollout and reports the limitation.')
    write_json(output/'audit.json', result)
    return result


def collect(output, checkpoint, reference):
    assert sha(checkpoint) == CHECKPOINT_SHA
    output.mkdir(parents=True, exist_ok=False)
    walk, jump, saved = policies(checkpoint, 'cuda:0')
    assert saved['iterations'] == 5200
    networks = [(p, {k: v.cpu().clone() for k, v in p.state_dict().items()}) for p in (walk, jump)]
    with patch('torch.optim.Adam.step', side_effect=RuntimeError('Diagnostic is inference only')):
        baseline = collect_profile(output/'baseline', walk, jump)
        probe = collect_profile(output/'probe', walk, jump, probe=True)
    unchanged = all(torch.equal(v.cpu(), before[k]) for policy, before in networks for k, v in policy.state_dict().items())
    assert unchanged and sha(checkpoint) == CHECKPOINT_SHA
    result = dict(json.loads((output/'metadata.json').read_text()),
                  checkpoint=str(checkpoint.resolve()), checkpoint_sha256=CHECKPOINT_SHA, walk_sha256=sha(WALK),
                  reference=str(reference.resolve()), seed=SEED, worlds=WORLDS, duration_s=6,
                  trigger_s=2., buffer_ms=80, command=[2., 0., .26], backend='mujoco_warp_gpu',
                  optimizer_updates=0, network_parameters_unchanged=unchanged,
                  source_sha256=sha(Path(__file__)), baseline=baseline, probe=probe)
    write_json(output/'capture.json', result)
    verify(output)
    print(json.dumps(dict(capture_passed=True, probe=probe)), flush=True)


def verify(output):
    capture = json.loads((output/'capture.json').read_text())
    if (output/'audit.json').exists():
        raise FileExistsError(output/'audit.json')
    walk, jump, _ = policies(Path(capture['checkpoint']), 'cuda:0')
    networks = [(p, {k: v.cpu().clone() for k, v in p.state_dict().items()}) for p in (walk, jump)]
    with patch('torch.optim.Adam.step', side_effect=RuntimeError('Diagnostic is inference only')):
        collect_profile(output/'baseline_repeat', walk, jump)
    assert all(torch.equal(v.cpu(), before[k]) for policy, before in networks for k, v in policy.state_dict().items())
    audit(output, Path(capture['reference']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('collect')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    run.add_argument('--reference', type=Path, default=REFERENCE)
    report = sub.add_parser('report')
    report.add_argument('--recording', type=Path, required=True)
    report.add_argument('--output-name', default='report')
    verification = sub.add_parser('verify')
    verification.add_argument('--recording', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'collect':
        collect(args.output, args.checkpoint, args.reference)
    elif args.command == 'verify':
        verify(args.recording)
    else:
        from .report_takeoff_work import report
        report(args.recording, output_name=args.output_name)


if __name__ == '__main__':
    main()
