"""Deterministic height-response/action audit of the exact A2500 and source2000.

No learning, optimizer updates, model writes, or imported source modifications.
Outputs must be new directories under artifacts/height-response-a2500-20261012.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import sys
import traceback
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True

import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper

from ct_mjlab.experiments import recovery_factors as factors
from ct_mjlab.experiments import huananhu_v14_height as source
from ct_mjlab.policy import actor_model
from ct_mjlab.reference_training import huananhu_mdp, huananhu_observations
from ct_mjlab.reference_training.entry import serializable
from ct_mjlab.wheel_balance_assessment import STANDARD, components

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / 'artifacts/height-response-a2500-20261012'
LEGS = ('lf0', 'lf00', 'rf0', 'rf00')
KNEES = ('lf1', 'rf1')
PHYSICAL_LEGS = (0, 1, 3, 4)
HEIGHTS = (.20, .22, .25, .27, .30, .34)
SOURCE_HASH = '1c3487a7da8d1fe74c418466ec4068f621c162e36406c14786c972a2839e7027'
EVALUATION = ROOT / 'artifacts/recovery-factors-video-2500-20261012/results/baseline/final500-evaluation.json'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                               indent=2 if path.name in ('run.json', 'summary.json') else None) + '\n')


def finite_list(array):
    array = np.asarray(array)
    if np.isfinite(array).all():
        return array.tolist()
    return np.where(np.isfinite(array), array, None).tolist()


def distribution(array):
    array = np.asarray(array, dtype=np.float64)
    array = array.reshape(-1, array.shape[-1])
    return dict(n=int(len(array)), mean=finite_list(np.nanmean(array, 0)),
        std=finite_list(np.nanstd(array, 0)),
        quantile_probabilities=[0., .05, .25, .5, .75, .95, 1.],
        quantiles=finite_list(np.nanquantile(array, [0., .05, .25, .5, .75, .95, 1.], axis=0)))


def numerical_probe(env):
    huananhu_mdp.failure(env)
    return env.command_manager.get_term('standup').numerical_failure


def load_models(args):
    expected_a = json.loads(EVALUATION.read_text())['checkpoint_sha256']
    models = {}
    for label, path, expected, updates in (
        ('a2500', args.a_checkpoint, expected_a, 2500),
        ('source2000', args.source_checkpoint, SOURCE_HASH, 2000)):
        actual = digest(path)
        if actual != expected:
            raise ValueError(f'{label} SHA256 mismatch: {actual}')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        infos = factors.validate_checkpoint(saved, 'baseline') if label == 'a2500' else source.validate_checkpoint(saved)
        if infos['completed_updates'] != updates:
            raise ValueError('Wrong completed update count')
        actor = actor_model().to(args.device).eval()
        actor.load_state_dict(saved['actor_state_dict'], strict=True)
        if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
            raise ValueError('Nonfinite actor')
        models[label] = dict(actor=actor, metadata=dict(path=str(path), sha256=actual,
            expected_sha256=expected, hash_verified=True, interface_verified=True,
            interface=infos['experiment_profile_contract'], completed_updates=updates,
            training_command_cfg=infos['environment']['commands']['standup'],
            policy_std=saved['actor_state_dict']['distribution.std_param'].tolist(),
            std_type='scalar (per-action; six-action average is not each leg std)'))
    return models


def training_distribution():
    cfg = source.make_env_cfg(4096, 42, play=False)
    return dict(command_cfg=serializable(cfg.commands['standup']),
        action_cfg=serializable(cfg.actions['pd']), runner=source.runner_cfg(42),
        height_sampling='uniform[0.27,0.34] at each command resample',
        velocity_sampling='vx uniform[-2.7,2.7], yaw uniform[-2pi,2pi]',
        heading='50% ordinary modes: clamp(5*wrapped heading error,-2pi,2pi)',
        resampling_seconds='uniform[5,15]', ground_probability=.5,
        ground_first_1_5s='vx/yaw independent uniform[-1,1] override; height unchanged',
        special_modes=[dict(mode=2, starts_at=2000, bucket=.30, vx_abs=[2., 3.],
                            yaw='original sampled yaw, heading feedback disabled'),
            dict(mode=0, starts_at=3000, bucket=.15, vx=[-.1, .1], yaw_abs_pi=[2., 3.25]),
            dict(mode=1, starts_at=4000, bucket=.15, vx=[-.1, .1], yaw_abs_pi=[3.25, 4.5])],
        a2000to2500=dict(only_active_special_mode=2,
            conditional_special_probability_given_first_nonstanding_and_eligible=.30,
            probability_per_resample_if_all_episode_age_eligible=.27,
            ordinary_moving_if_eligible=.567, standing_if_eligible=.163,
            no_special_on_reset_or_age_under5s=True, standing_if_ineligible=.19,
            actual_time_occupancy='unverified; cannot infer from bucket probability without training traces'),
        no_simulation_for_command_distribution=True)


def static_reference(args, model, action, nominal):
    report = json.loads(args.static_solutions.read_text())
    if report['asset_sha256'] != digest(ROOT / 'resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml'):
        raise ValueError('Static solution asset hash mismatch')
    physical_ids = [0, 1, 3, 4]
    reference = action.reference_positions[0, physical_ids].cpu().numpy()
    scale, kp = action.cfg.leg_position_scale, action.cfg.leg_stiffness
    result = dict(input_path=str(args.static_solutions), input_sha256=digest(args.static_solutions),
        nominal_positions=nominal.tolist(), leg_names=LEGS, knee_names=KNEES,
        nominal_knees=[json.loads((ROOT / 'mjlab_training/src/ct_mjlab/poses.json').read_text())['nominal'][n] for n in KNEES],
        cad_reference_positions=reference.tolist(), raw_policy_action_bounds=None,
        policy_outputs_unbounded=True, leg_position_scale=scale, kp=kp, kd=action.cfg.leg_damping,
        target_lower=(reference - 3.14).tolist(), target_upper=(reference + 3.14).tolist(),
        effective_action_lower=((reference - 3.14 - nominal) / scale).tolist(),
        effective_action_upper=((reference + 3.14 - nominal) / scale).tolist(),
        torque_limit_nm=40., poses_json_sha256=digest(ROOT / 'mjlab_training/src/ct_mjlab/poses.json'), cases=[])
    for case in report['cases']:
        q = np.array([case['joint_positions'][n] for n in LEGS])
        tau = np.array(case['force_equilibrium']['motor_torque_nm'])[physical_ids]
        target = q + tau / kp
        result['cases'].append(dict(height=case['target_height_m'], joint_positions=q.tolist(),
            knees=[case['joint_positions'][n] for n in KNEES], motor_torque_nm=tau.tolist(),
            pose_only_actions=((q - nominal) / scale).tolist(),
            loaded_pd_targets=target.tolist(), loaded_pd_actions=((target - nominal) / scale).tolist(),
            target_limit_margin_rad=np.minimum(target - reference + 3.14, reference + 3.14 - target).tolist(),
            knee_limit_margins=case['knee_limit_margin_rad'],
            geometric_and_static_only=True, free_balance_not_proven=True))
    return result


def checks(env, active, height):
    robot = env.scene['robot'].data
    return components(gravity=robot.projected_gravity_b,
        wheel_support=factors.wheel_force(env), wheel_contact=env.scene['wheel_ground_probe'].data.found > 0,
        nonwheel_support=factors.other_force(env), robot_weight=factors.weight(env),
        base_contact=env.scene['base_contact'].data.force.norm(dim=-1).amax(-1),
        linear_velocity=robot.root_link_lin_vel_b, angular_velocity=robot.root_link_ang_vel_b,
        height=height, target_height=env.command_manager.get_term('standup').command[:, 2],
        valid=active, parameters=factors.PARAMETERS)


def knee_limit_constraints(env):
    """Read native constraint rows; never recompute or apply forces."""
    data, model = env.sim.wp_data, env.sim.mj_model
    types, ids, force = (wp.to_torch(getattr(data.efc, n)) for n in ('type', 'id', 'force'))
    valid = torch.arange(force.shape[1], device=env.device)[None, :] < wp.to_torch(data.nefc)[:, None]
    limit_type = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
    magnitude, counts, torque = [], [], []
    jacobian = wp.to_torch(data.efc.J)
    if data.efc.J_rownnz.shape[1] != 0:
        raise ValueError('This read-only limit probe expects the native dense Jacobian')
    for name in KNEES:
        joint = model.joint('robot/' + name)
        selected = valid & (types == limit_type) & (ids == joint.id)
        weighted = selected * force
        magnitude.append(weighted.abs().sum(-1))
        counts.append(selected.sum(-1))
        torque.append((weighted * jacobian[:, :force.shape[1], int(joint.dofadr[0])]).sum(-1))
    return dict(knee_limit_force_magnitude=torch.stack(magnitude, -1),
        knee_limit_constraint_count=torch.stack(counts, -1),
        knee_limit_generalized_torque=torch.stack(torque, -1))


def one_pose(args, pose, models):
    torch.manual_seed(args.seed)
    cfg = factors.make_env_cfg('baseline', 64, args.seed, play=True)
    cfg.auto_reset = False
    cfg.commands['standup'].ground_start_probability = float(pose == 'ground')
    cfg.commands['standup'].zero_velocity_commands = True
    cfg.commands['standup'].height_range = (.30, .30)
    cfg.commands['standup'].resampling_time_range = (1e6, 1e6)
    cfg.terminations['failure'] = TerminationTermCfg(func=numerical_probe)
    wrapped = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=args.device), clip_actions=None)
    env = wrapped.unwrapped
    try:
        state = env.command_manager.get_term('standup')
        actor_action = env.action_manager.get_term('pd')
        model = env.sim.mj_model
        path = args.initial_ground if pose == 'ground' else args.initial_normal
        if path:
            with np.load(path) as saved:
                initial = {k: torch.as_tensor(saved[k], device=args.device).clone() for k in ('qpos', 'qvel', 'origins')}
        else:
            initial = {k: v.clone() for k, v in dict(qpos=env.sim.data.qpos, qvel=env.sim.data.qvel, origins=env.scene.env_origins).items()}
        if not torch.equal(initial['origins'], env.scene.env_origins):
            raise ValueError('Canonical64 scene origins differ')
        initial_hash = hashlib.sha256(initial['qpos'].cpu().numpy().tobytes() + initial['qvel'].cpu().numpy().tobytes()).hexdigest()
        historical = json.loads(EVALUATION.read_text())['poses'][pose]['initial_state_sha256']
        if initial_hash != historical:
            raise ValueError(f'Canonical {pose} initial fingerprint differs: {initial_hash} vs {historical}')
        write(args.output / f'initial-{pose}.json', dict(input=str(path), fingerprint=initial_hash,
            historical_fingerprint=historical, exact_match=True,
            **{k: v.cpu().numpy().tolist() for k, v in initial.items()}))
        qids = [int(model.joint('robot/' + n).qposadr[0]) for n in LEGS]
        kids = [int(model.joint('robot/' + n).qposadr[0]) for n in KNEES]
        knee_ranges = np.array([model.jnt_range[model.joint('robot/' + n).id] for n in KNEES])
        nominal = actor_action.nominal_positions[0, list(PHYSICAL_LEGS)].cpu().numpy()
        static = static_reference(args, model, actor_action, nominal)
        static['knee_ranges'] = knee_ranges.tolist()
        write(args.output / 'static-reference.json', static)
        data = {}
        original_apply = actor_action.apply_actions
        telemetry = {}

        def apply_actions():
            original_apply()
            legs = actor_action._leg_action_ids
            delayed = actor_action._action_history[(actor_action._history_head - actor_action.action_delay_steps)
                                                   % len(actor_action._action_history), actor_action._env_ids]
            unbounded = actor_action.nominal_positions[:, legs] + actor_action.cfg.leg_position_scale * delayed[:, legs]
            ref = actor_action.reference_positions[:, legs]
            target = unbounded.clamp(ref - 3.14, ref + 3.14)
            telemetry['target'] = target.clone()
            torque = actor_action.torques[:, legs]
            target_hit = (unbounded - target).abs() > 1e-6
            torque_hit = torque.abs() >= 40. - 1e-4
            q = env.sim.data.qpos[:, kids]
            margin = torch.minimum(q - telemetry['knee_low'], telemetry['knee_high'] - q)
            telemetry['full_target_hits'] += target_hit
            telemetry['full_torque_hits'] += torque_hit
            telemetry['full_min_knee_margin'] = torch.minimum(telemetry['full_min_knee_margin'], margin)
            if telemetry['tail']:
                telemetry['target_hits'] += target_hit
                telemetry['torque_hits'] += torque_hit
                telemetry['min_knee_margin'] = torch.minimum(telemetry['min_knee_margin'], margin)
                telemetry['maximum_abs_torque'] = torch.maximum(telemetry['maximum_abs_torque'], torque.abs())

        actor_action.apply_actions = apply_actions
        nsteps = round(args.seconds / env.step_dt)
        tail_steps = min(round(5. / env.step_dt), nsteps)
        tail_start = nsteps - tail_steps
        for label in args.models:
            actor = models[label]['actor']
            for command in args.heights:
                started = now()
                env.common_step_counter = 0
                env._sim_step_counter = 0
                env.reset(seed=args.seed)
                ids = torch.arange(64, device=args.device)
                env.scene.reset(ids)
                env.sim.data.qpos.copy_(initial['qpos'])
                env.sim.data.qvel.copy_(initial['qvel'])
                state.sampled[:, :2] = state.command[:, :2] = 0.
                state.sampled[:, 2] = state.command[:, 2] = command
                state.ground_command.zero_()
                state.ground_start.fill_(pose == 'ground')
                state.numerical_failure.zero_()
                env.sim.forward()
                env.sim.sense()
                huananhu_observations.reset_observations(env, ids)
                env.observation_manager.reset(ids)
                env.obs_buf = env.observation_manager.compute(update_history=True)
                obs = wrapped.get_observations()
                if not torch.equal(env.sim.data.qpos, initial['qpos']) or not torch.equal(env.sim.data.qvel, initial['qvel']):
                    raise ValueError('Initial states were not exactly restored')
                active = torch.ones(64, dtype=torch.bool, device=args.device)
                telemetry.clear()
                telemetry.update(tail=False, knee_low=torch.tensor(knee_ranges[:, 0], device=args.device),
                    knee_high=torch.tensor(knee_ranges[:, 1], device=args.device))
                for name in ('target_hits', 'torque_hits', 'full_target_hits', 'full_torque_hits', 'maximum_abs_torque'):
                    telemetry[name] = torch.zeros(64, 4, device=args.device)
                for name in ('min_knee_margin', 'full_min_knee_margin'):
                    telemetry[name] = torch.full((64, 2), float('inf'), device=args.device)
                heights, trace, joint_records = [], [], []
                legacy_tail = torch.zeros(64, device=args.device)
                sync_tail = torch.zeros_like(legacy_tail)
                component_tail = {}
                height_band_hold = torch.zeros(64, device=args.device)
                height_band_max_hold = torch.zeros_like(height_band_hold)
                max_after2s = torch.full((64,), -float('inf'), device=args.device)
                # History tensors persist across trials; inference-mode tensors
                # cannot later be reset in place outside that mode.
                with torch.no_grad():
                    for step in range(nsteps):
                        telemetry['tail'] = step >= tail_start
                        action = torch.where(active[:, None], actor(obs), 0.)
                        obs, _, _, _ = wrapped.step(action)
                        active &= ~state.numerical_failure
                        h = env.scene['robot'].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
                        legacy = checks(env, active, h)
                        # Diagnostic reads only: no history rolling or physical step.
                        for name in ('base_contact', 'wheel_contact', 'wheel_ground_probe', 'nonwheel_ground_probe'):
                            env.scene[name]._invalidate_cache()
                        synchronized = checks(env, active, h)
                        if step >= 100:
                            max_after2s = torch.maximum(max_after2s, h)
                            height_band_hold = torch.where(synchronized['target_height'] & active,
                                height_band_hold + env.step_dt, 0.)
                            height_band_max_hold = torch.maximum(height_band_max_hold, height_band_hold)
                        if step >= tail_start:
                            heights.append(h.clone())
                            legacy_tail += legacy['wheel_balance']
                            sync_tail += synchronized['wheel_balance']
                            for name, flag in synchronized.items():
                                component_tail.setdefault(name, torch.zeros(64, device=args.device))
                                component_tail[name] += flag & active
                            if abs(command - .30) < 1e-9 and pose == 'normal':
                                record = dict(actions=action[:, :4].clone(), target=telemetry['target'].clone(),
                                    q=env.sim.data.qpos[:, qids].clone(), torque=actor_action.torques[:, list(PHYSICAL_LEGS)].clone(),
                                    knees=env.sim.data.qpos[:, kids].clone())
                                if args.record_knee_limit_forces:
                                    record.update({k:v.clone() for k,v in knee_limit_constraints(env).items()})
                                joint_records.append(record)
                        if (step + 1) % 50 == 0:
                            trace.append(dict(time_s=(step + 1) * env.step_dt, height=h.cpu().tolist(),
                                synchronized_balanced=int(synchronized['wheel_balance'].sum()),
                                numerical_failures=int((~active).sum())))
                        # A failed world remains permanently excluded. Reset only it
                        # to prevent NaNs from contaminating further batch stepping.
                        bad = (env.reset_buf & ~env.reset_time_outs).nonzero().flatten()
                        if len(bad):
                            env.reset(env_ids=bad)
                            state.sampled[bad, :2] = state.command[bad, :2] = 0.
                            state.sampled[bad, 2] = state.command[bad, 2] = command
                            obs = wrapped.get_observations()
                array = torch.stack(heights).cpu().numpy()
                valid = active.cpu().numpy()
                values = array[:, valid]
                sync = ((sync_tail >= .95 * tail_steps) & active).cpu().numpy()
                legacy = ((legacy_tail >= .95 * tail_steps) & active).cpu().numpy()
                summary = dict(model=label, pose=pose, command_height_m=command, initial_state_sha256=initial_hash,
                    seconds=args.seconds, tail_seconds=tail_steps * env.step_dt, started=started, ended=now(),
                    mean_height_m=float(values.mean()) if valid.any() else None,
                    pooled_height_std_m=float(values.std()) if valid.any() else None,
                    between_env_mean_height_std_m=float(values.mean(0).std()) if valid.any() else None,
                    mean_within_env_temporal_std_m=float(values.std(0).mean()) if valid.any() else None,
                    per_env_mean_height_m=finite_list(array.mean(0)),
                    per_env_temporal_std_m=finite_list(array.std(0)),
                    stable_balance=int(sync.sum()), stable_samples=np.flatnonzero(sync).tolist(),
                    legacy_stable_balance=int(legacy.sum()), legacy_stable_samples=np.flatnonzero(legacy).tolist(),
                    numerical_failures=int((~valid).sum()),
                    component_fractions={k: (v / tail_steps).cpu().tolist() for k, v in component_tail.items()},
                    max_height_after2s=finite_list(max_after2s.cpu().numpy()),
                    max_continuous_height_band_hold_after2s=height_band_max_hold.cpu().tolist())
                limitation = dict(leg_names=LEGS, knee_names=KNEES, tail_substeps=tail_steps * cfg.decimation,
                    full_substeps=nsteps * cfg.decimation)
                for key in ('target_hits', 'torque_hits', 'full_target_hits', 'full_torque_hits',
                            'min_knee_margin', 'full_min_knee_margin', 'maximum_abs_torque'):
                    limitation[key] = telemetry[key].cpu().tolist()
                raw_joints, joint_summary = {}, {}
                if joint_records:
                    for key in joint_records[0]:
                        arr = torch.stack([r[key] for r in joint_records]).cpu().numpy()
                        raw_joints[key] = finite_list(arr)
                        joint_summary[key] = distribution(arr[:, valid])
                    mean_action = np.array(joint_summary['actions']['mean'])
                    std = np.array(models[label]['metadata']['policy_std'])[:4]
                    joint_summary['static_comparison'] = []
                    for reference in static['cases']:
                        target_action = np.array(reference['loaded_pd_actions'])
                        delta = target_action - mean_action
                        joint_summary['static_comparison'].append(dict(height=reference['height'],
                            delta_loaded_pd_actions=delta.tolist(), delta_over_0_16=(delta / .16).tolist(),
                            delta_over_actual_leg_std=(delta / std).tolist(),
                            joint_standardized_distance=float(np.linalg.norm(delta / std))))
                filename = f'{label}-{pose}-h{command:.2f}.json'
                write(args.output / filename, dict(summary=summary, limitations=limitation,
                    joint_summary=joint_summary, tail_height=finite_list(array), tail_joint_samples=raw_joints, trace=trace))
                data[f'{label}:{command:.2f}'] = summary | {'output_file': filename, 'joint_summary': joint_summary,
                                                        'limitations': limitation}
                print('HEIGHT_RESPONSE ' + json.dumps({k: summary[k] for k in (
                    'model', 'pose', 'command_height_m', 'mean_height_m', 'pooled_height_std_m',
                    'stable_balance', 'legacy_stable_balance', 'numerical_failures')}), flush=True)
        return data
    finally:
        wrapped.close()


def fits(records, heights):
    result = {}
    for label in ('a2500', 'source2000'):
        for pose in ('normal', 'ground'):
            rows = [v for v in records.values() if v['model'] == label and v['pose'] == pose]
            if len(rows) < 2:
                continue
            rows.sort(key=lambda x: x['command_height_m'])
            x = np.array([r['command_height_m'] for r in rows])
            y = np.array([r['mean_height_m'] for r in rows])
            slope, intercept = np.polyfit(x, y, 1)
            common = sorted(set.intersection(*[set(r['stable_samples']) for r in rows]))
            result[f'{label}:{pose}'] = dict(slope_m_per_m=float(slope), intercept_m=float(intercept),
                commanded_range_m=float(x.max()-x.min()), actual_mean_range_m=float(y.max()-y.min()),
                residual_rmse_m=float(np.sqrt(np.mean((y-slope*x-intercept)**2))),
                common_balanced_samples=common,
                common_balanced_slope_m_per_m=float(np.polyfit(x, [np.mean(np.array(r['per_env_mean_height_m'])[common])
                                                                  for r in rows], 1)[0]) if common else None)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a-checkpoint', type=Path, default=OUT_ROOT / 'models/model_2499.pt')
    p.add_argument('--source-checkpoint', type=Path, default=OUT_ROOT / 'models/model_1999.pt')
    p.add_argument('--initial-ground', type=Path, required=True)
    p.add_argument('--initial-normal', type=Path)
    p.add_argument('--static-solutions', type=Path, default=ROOT / 'artifacts/wheel-support-xml-height-check-20261012/jobs/20261012-xml-wheel-support-height-v5/xml-height-audit.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seconds', type=float, default=20.)
    p.add_argument('--heights', type=float, nargs='+', default=list(HEIGHTS))
    p.add_argument('--poses', choices=('normal', 'ground'), nargs='+', default=['normal', 'ground'])
    p.add_argument('--models', choices=('a2500', 'source2000'), nargs='+', default=['a2500', 'source2000'])
    p.add_argument('--record-knee-limit-forces', action='store_true',
                   help='Also record native knee joint-limit constraints at 0.30m/normal')
    args = p.parse_args()
    if not args.output.resolve().is_relative_to(OUT_ROOT.resolve()):
        p.error('Output must be under artifacts/height-response-a2500-20261012')
    if args.seconds <= 0 or not np.isfinite(args.seconds) or any(not np.isfinite(h) or h <= 0 for h in args.heights):
        p.error('Seconds and heights must be positive finite values')
    if len(set(args.heights)) != len(args.heights):
        p.error('Duplicate heights')
    args.models, args.poses = list(dict.fromkeys(args.models)), list(dict.fromkeys(args.poses))
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), cwd=str(Path.cwd()),
        started=now(), ended=None, exit_code=None, python=sys.version, platform=platform.platform(),
        device=args.device, cuda_available=torch.cuda.is_available(), torch_version=torch.__version__,
        versions={name: importlib.metadata.version(name) for name in ('mjlab', 'mujoco', 'mujoco-warp', 'rsl-rl-lib', 'warp-lang')},
        num_envs=64, maximum_concurrent_envs=64, learning=False, optimizer_loaded=False,
        evaluation_standard=STANDARD, nominal_physics=True, deterministic_actor=True,
        noise=False, delays=False, raw_action_clipping=False,
        height_outside_training_range=[h for h in args.heights if not .27 <= h <= .34],
        expected_full_protocol=args.seconds == 20. and set(args.heights) == set(HEIGHTS)
            and len(args.models)==2 and len(args.poses)==2,
        arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        script_sha256=digest(__file__), frozen_sources={})
    for folder in ('mjlab_training/src/ct_mjlab/experiments', 'mjlab_training/src/ct_mjlab/reference_training'):
        for path in (ROOT / folder).rglob('*.py'):
            metadata['frozen_sources'][str(path.relative_to(ROOT))] = digest(path)
    for path in (ROOT / 'mjlab_training/src/ct_mjlab/wheel_balance_assessment.py',
                 ROOT / 'scripts/agent/eval_chuanliantui_recovery_factors.py'):
        metadata['frozen_sources'][str(path.relative_to(ROOT))] = digest(path)
    write(args.output / 'run.json', metadata)
    try:
        if args.device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise ValueError('Requested GPU is unavailable')
            metadata['device_name'] = torch.cuda.get_device_name(args.device)
            metadata['gpu_total_memory_bytes'] = torch.cuda.get_device_properties(args.device).total_memory
        models = load_models(args)
        metadata['models'] = {k:v['metadata'] for k,v in models.items()}
        training = training_distribution()
        if models['a2500']['metadata']['training_command_cfg'] != training['command_cfg']:
            raise ValueError('Frozen model and current training command configuration differ')
        write(args.output / 'training-command-distribution.json', training)
        metadata['initial_input_sha256'] = {str(path):digest(path) for path in (args.initial_ground,args.initial_normal) if path}
        records = {}
        for pose in args.poses:
            data = one_pose(args, pose, models)
            records.update({f'{pose}:{k}':v for k,v in data.items()})
            write(args.output / 'summary.json', dict(records=records, fits=fits(records,args.heights)))
        metadata['exit_code'] = 0
        if args.device.startswith('cuda'):
            metadata['peak_torch_gpu_memory_bytes'] = torch.cuda.max_memory_allocated(args.device)
    except Exception:
        metadata['exit_code'] = 1
        metadata['exception'] = traceback.format_exc()
        raise
    finally:
        metadata['ended'] = now()
        metadata['frozen_sources_unchanged'] = all(digest(ROOT / path) == sha for path,sha in metadata['frozen_sources'].items())
        write(args.output / 'run.json', metadata)


if __name__ == '__main__':
    main()
