#!/usr/bin/env python3
"""Read saved feedback traces and MuJoCo quaternion math; never step physics."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(value):
    value = np.asarray(value)
    return dict(minimum=float(value.min()), median=float(np.median(value)),
        maximum=float(value.max()))


def error_metrics(predicted, target, mask):
    if not mask.any():
        return None
    difference = (predicted-target)[mask]
    return dict(samples=int(mask.sum()), bias_m_s=float(difference.mean()),
        mae_m_s=float(abs(difference).mean()),
        abs_error_p95_m_s=float(np.percentile(abs(difference), 95)),
        abs_error_max_m_s=float(abs(difference).max()))


def audit(root, mode):
    metadata = root/(mode+'.json')
    trace = root/(mode+'.npz')
    meta = json.loads(metadata.read_text())
    scene = Path(meta['scene_mjb'])
    model = mujoco.MjModel.from_binary_path(str(scene))
    with np.load(trace, allow_pickle=False) as saved:
        values = saved['values']
        poses = saved['qpos']
        initial_v = saved['fixed_qvel']
        initial_q = saved['fixed_qpos']
        live_at_end = saved['qpos_live'][-1]
    assert np.array_equal(poses[0], initial_q)
    assert meta['feedback_mode'] == mode and meta['velocity_source'] == mode
    assert meta['wheel_radius_m'] == .06 and meta['wheel_vx_coefficient_m'] == .03
    assert meta['parameter_updates'] == 0 and meta['actor_unchanged']
    assert meta['actor_state_sha256'] == meta['actor_state_sha256_after']

    def get(key):
        return values[..., slice(*meta['layout'][key])]

    valid = get('valid')[..., 0] > .5
    policy_stride = round(meta['policy_dt']/meta['dt'])
    assert policy_stride == 5
    policy_samples = np.arange(0, len(values), policy_stride)
    policy_times = np.arange(len(policy_samples))*meta['policy_dt']
    wheel_names = ('lfwheel', 'rfwheel')
    joint_ids = [model.joint('robot/'+name).id for name in wheel_names]
    joint_vadr = model.jnt_dofadr[joint_ids]
    assert np.array_equal(model.jnt_axis[joint_ids], np.asarray(((0., 1., 0.), (0., -1., 0.))))
    wheel_trace_indices = [meta['joint_names'].index(name) for name in wheel_names]
    wheels = get('policy_wheel_velocity')[policy_samples]
    previous = np.empty_like(wheels)
    previous[0] = initial_v[:, joint_vadr]
    previous[1:] = get('joint_dq')[policy_samples[1:]-1][..., wheel_trace_indices]
    policy_live = valid[policy_samples]
    difference = abs(wheels-previous)
    assert np.allclose(wheels[policy_live], previous[policy_live], rtol=1e-6, atol=1e-6)
    held = get('policy_wheel_velocity').reshape(-1, policy_stride, wheels.shape[1], 2)
    hold_residual = float(abs(held-held[:, :1]).max())
    assert hold_residual == 0.
    proxy = .03*(wheels[..., 0]-wheels[..., 1])
    used = get('policy_velocity')[policy_samples]
    encoder = get('encoder_velocity')[policy_samples]
    expected_vx = proxy if mode == 'wheel-vx' else encoder[..., 0]
    feedback_residual = float(abs(used[..., 0]-expected_vx)[policy_live].max())
    assert np.allclose(used[..., 0][policy_live], expected_vx[policy_live], rtol=1e-6, atol=1e-7)
    assert np.array_equal(used[..., 1:], encoder[..., 1:])

    # 只调用MuJoCo四元数数学函数，独立于Recorder中的展开公式。
    free_joint = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
    qadr = int(model.jnt_qposadr[free_joint])
    quaternions = poses[1:, :, qadr+3:qadr+7]
    angles = np.empty((*quaternions.shape[:2], 2), dtype=np.float64)
    rotation = np.empty(9)
    for index in np.ndindex(quaternions.shape[:2]):
        mujoco.mju_quat2Mat(rotation, quaternions[index].astype(np.float64))
        angles[index] = np.rad2deg((np.arctan2(rotation[7], rotation[8]),
            np.arcsin(np.clip(-rotation[6], -1., 1.))))
    angle_error = abs(angles-get('roll_pitch')[policy_stride-1::policy_stride])
    assert float(angle_error.max()) < 1e-3
    true_com = get('true_velocity_before')[policy_samples, :, 0]
    steady = (policy_times >= 5.) & (policy_times < 15.)
    cases = []
    for case in meta['cases']:
        ids = case['env_ids']
        live = policy_live[:, ids]
        stable = live & steady[:, None]
        cases.append(dict(case=case['name'], speed_m_s=case['speed'],
            acceleration_m_s2=case['acceleration'],
            observed_first_episode=error_metrics(proxy[:, ids], true_com[:, ids], live),
            encoder_observed_first_episode=error_metrics(encoder[:, ids, 0], true_com[:, ids], live),
            steady_5_to_15s_wheel_proxy=error_metrics(proxy[:, ids], true_com[:, ids], stable),
            steady_5_to_15s_encoder=error_metrics(encoder[:, ids, 0], true_com[:, ids], stable)))

    ended = np.asarray(meta['episode_done'], dtype=bool)
    ends = np.asarray(meta['episode_end'])
    assert np.array_equal(~ended, live_at_end)
    physics_times = (np.arange(len(values))+1)*meta['dt']
    failures = []
    for env in np.flatnonzero(ended):
        last = round(ends[env]/meta['dt'])-1
        recent = valid[:, env] & (physics_times > ends[env]-.1-1e-9)
        request = get('applied_command')[last, env]
        tilt = float(get('tilt')[last, env, 0])
        support = get('support')[last, env]
        base = float(get('base_contact')[last, env, 0])
        height = float(get('height')[last, env, 0])
        loaded = bool((support >= max(.1*meta['weight_N'], 5.)).all()
            and support.sum() >= .8*meta['weight_N'])
        failures.append(dict(env=int(env), end_s=float(ends[env]),
            ended_during_zero_request_warmup=bool(ends[env] < meta['warmup']),
            applied_command_at_end=request.tolist(), height_m=height,
            roll_pitch_deg=get('roll_pitch')[last, env].tolist(), tilt_deg=tilt,
            support_N=support.tolist(), support_total_weight_ratio=float(support.sum()/meta['weight_N']),
            base_contact_N=base, down_ticks=float(get('down_ticks')[last, env, 0]),
            observed_conditions=dict(tilt_over90_inversion=tilt > 90., tilt_over40=tilt > 40.,
                sufficient_load=loaded, base_contact_over10=base > 10., height_le18cm=height <= .18,
                armed_1s_elapsed=bool(ends[env] >= 1.)),
            final100ms=dict(observed_samples=int(recent.sum()),
                height_range_m=[float(get('height')[recent, env, 0].min()),
                                float(get('height')[recent, env, 0].max())],
                pitch_abs_max_deg=float(abs(get('roll_pitch')[recent, env, 1]).max()),
                support_total_range_N=[float(get('support')[recent, env].sum(-1).min()),
                                       float(get('support')[recent, env].sum(-1).max())],
                base_contact_max_N=float(get('base_contact')[recent, env, 0].max()),
                down_ticks_max=float(get('down_ticks')[recent, env, 0].max()))))
    summary = None
    if failures:
        summary = dict(count=len(failures), end_s=stats([e['end_s'] for e in failures]),
            all_before_movement_request=all(e['ended_during_zero_request_warmup'] for e in failures),
            height_m=stats([e['height_m'] for e in failures]),
            pitch_deg=stats([e['roll_pitch_deg'][1] for e in failures]),
            roll_deg=stats([e['roll_pitch_deg'][0] for e in failures]),
            tilt_deg=stats([e['tilt_deg'] for e in failures]),
            support_total_weight_ratio=stats([e['support_total_weight_ratio'] for e in failures]),
            down_ticks=stats([e['down_ticks'] for e in failures]),
            at_end_tilt_over90_count=sum(e['observed_conditions']['tilt_over90_inversion'] for e in failures),
            at_end_base_contact_over10_count=sum(e['observed_conditions']['base_contact_over10'] for e in failures),
            at_end_insufficient_load_count=sum(not e['observed_conditions']['sufficient_load'] for e in failures))
    return dict(mode=mode, checkpoint_sha256=meta['checkpoint_sha256'],
        actor_state_sha256=meta['actor_state_sha256'], survived=int((~ended).sum()),
        inputs={str(p.resolve()):digest(p) for p in (metadata, trace, scene)},
        wheel_mapping=dict(physical_joint_names=list(wheel_names), actor_slots=[15, 18],
            observation_scale=.05, radius_m=.06, forward_signs=[1, -1], mean_coefficient_m=.03,
            source_chain='raw sensors MOTOR_NAMES lf0,lf00,rf0,rf00,lfwheel,rfwheel; velocity permutation 0,1,4,2,3,5; six speeds begin actor slot13',
            previous_policy_boundary_qvel_max_abs_residual_rad_s=float(difference[policy_live].max()),
            reversed_joint_order_mean_abs_residual_rad_s=float(abs(wheels-previous[..., ::-1])[policy_live].mean()),
            feedback_held_5_physics_substeps_max_abs_residual_rad_s=hold_residual,
            actual_policy_vx_formula_max_abs_residual_m_s=feedback_residual,
            encoder_vy_vz_preserved=True),
        roll_pitch_verification=dict(method='Saved qpos -> mujoco.mju_quat2Mat -> ZYX roll/pitch; no forward/step',
            order=['roll', 'pitch'], samples=int(np.prod(quaternions.shape[:2])),
            max_abs_residual_deg=float(angle_error.max()), tolerance_deg=1e-3),
        error_window='Policy input time in [5,15)s, first episode only; same pre-action COM target. First-episode rows use all available live policy ticks.',
        cases=cases, failure_summary=summary, failures=failures,
        failure_interpretation='Recorded down_ticks precede terminal computation by up to one policy tick. Before1s, the new100ms down gate is unarmed; >90deg tilt matches the unconditional inversion emergency. This distinguishes observed termination conditions from low load alone. It does not identify why the feedback destabilized the policy.')


def main(args):
    args.measurement = args.measurement.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = [audit(args.measurement, mode) for mode in ('encoder', 'wheel-vx')]
    assert reports[0]['checkpoint_sha256'] == reports[1]['checkpoint_sha256']
    assert reports[0]['actor_state_sha256'] == reports[1]['actor_state_sha256']
    script = Path(__file__).resolve()
    result = dict(verified=True, simulation_steps=0, optimizer_steps=0, device='cpu',
        script=str(script), script_sha256=digest(script), runs=reports,
        scope='Audit of this recorded intervention and its failures only; no new candidate, resimulation, training or task-acceptance claim')
    target = args.output/'input-and-failure-audit.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    shutil.copy2(script, args.output/'source.py')
    print(json.dumps(dict(verified=True, output=str(target),
        runs=[dict(mode=r['mode'], survived=r['survived'],
                   mapping_max_abs_residual_rad_s=r['wheel_mapping']['previous_policy_boundary_qvel_max_abs_residual_rad_s'],
                   roll_pitch_max_abs_residual_deg=r['roll_pitch_verification']['max_abs_residual_deg'],
                   failure_summary=r['failure_summary']) for r in reports]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
