#!/usr/bin/env python3
"""Isolated recorded-data check of analytic motor torques and integrated wheel q.

No training, new simulation, production changes or actor/critic inference. The
frozen-world substitution is a feasibility probe, not a reduced-output model.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
import torch
import yaml

from scripts.agent.supermimic import score_supermimic_v11_paired_prediction as bridge
from ct_mjlab.envs.chuanliantui.model import ACTUATED_JOINTS, load_model
from ct_mjlab.envs.chuanliantui_split.env_cfg import command_cfg
from ct_mjlab.envs.chuanliantui_split.runner import contracts
from ct_mjlab.supermimic.networks import RunningNorm
from ct_mjlab.supermimic.state import NOISE

core = bridge.core
MODES = ('baseline', 'torque', 'wheel_q', 'both')


def scalar_mapping(node):
    return {key.value: value for key, value in node.value}


class AnalyticAuxiliaries:
    def __init__(self, model, config):
        action = scalar_mapping(scalar_mapping(yaml.compose(config.read_text()))['actions'])['pd']
        values = scalar_mapping(action)
        self.cfg = {name: float(values[name].value) for name in (
            'leg_position_scale', 'wheel_velocity_scale', 'max_wheel_velocity',
            'wheel_stiffness', 'wheel_damping', 'motor_response_time_constant_s')}
        if self.cfg['motor_response_time_constant_s'] != 0.:
            raise ValueError('This recorded-data probe does not reconstruct nonzero motor-response memory')
        self.q = torch.tensor([model.jnt_qposadr[model.joint(n).id] for n in ACTUATED_JOINTS])
        self.v = torch.tensor([25+model.jnt_dofadr[model.joint(n).id] for n in ACTUATED_JOINTS])
        actuator = [model.actuator(n+'_motor').id for n in ACTUATED_JOINTS]
        self.reference = torch.tensor(model.qpos0[self.q.numpy()], dtype=torch.float32)
        self.gain = torch.tensor(model.actuator_gainprm[actuator, 0], dtype=torch.float32)
        self.response = torch.tensor(model.actuator_gainprm[actuator, :3], dtype=torch.float32)
        self.limits = torch.tensor(model.actuator_ctrlrange[actuator], dtype=torch.float32)
        self.legs, self.wheels = [0, 1, 3, 4], [2, 5]
        self.wq, self.wv = self.q[self.wheels], self.v[self.wheels]

    def torque(self, physical, parameters, action, previous, phase):
        delayed = torch.where((parameters[:, 24:25] <= phase), action, previous)
        wheel_delayed = torch.where((parameters[:, 25:26] <= phase), action, previous)
        q, qd = physical[:, self.q], physical[:, self.v]
        kp, kd, strength = parameters[:, 6:12], parameters[:, 12:18], parameters[:, 18:24]
        result = torch.zeros_like(action)
        target = (parameters[:, self.legs]+self.cfg['leg_position_scale']*delayed[:, self.legs]).clamp(
            self.reference[self.legs]-3.14, self.reference[self.legs]+3.14)
        result[:, self.legs] = kp[:, self.legs]*(target-q[:, self.legs])-kd[:, self.legs]*qd[:, self.legs]
        velocity = (self.cfg['wheel_velocity_scale']*wheel_delayed[:, self.wheels]).clamp(
            -self.cfg['max_wheel_velocity'], self.cfg['max_wheel_velocity'])
        stiffness, damping = self.cfg['wheel_stiffness'], self.cfg['wheel_damping']
        kd_scale = kd[:, self.wheels]/(stiffness+damping)
        result[:, self.wheels] = kd_scale*(stiffness*(velocity-qd[:, self.wheels])-damping*qd[:, self.wheels])
        result = (result*strength).clamp(self.limits[:, 0], self.limits[:, 1])
        # Apply the XML speed attenuation and fixed gain once, as the real backend does.
        command = result[:, self.wheels]
        along = (command.sign()*qd[:, self.wheels]).clamp_min(0.)
        response = self.response[self.wheels]
        fraction = (1.-along/response[:, 1]).clamp_min(0.)
        result[:, self.wheels] = command*fraction.pow(response[:, 2])
        return result*self.gain

    def path(self, state, parameters, path, action, previous, mode):
        result = path.clone()
        before = state[:, :64]
        for phase in range(5):
            if mode in ('wheel_q', 'both'):
                # implicitfast advances scalar hinge q using the integrated end-step velocity.
                result[:, phase, self.wq] = before[:, self.wq]+.002*result[:, phase, self.wv]
            if mode in ('torque', 'both'):
                result[:, phase, 49:55] = self.torque(before, parameters, action, previous, phase)
            before = result[:, phase]
        return result


class SubstitutedWorld:
    def __init__(self, world, auxiliaries, mode):
        self.world, self.auxiliaries, self.mode = world, auxiliaries, mode
        self.original_advance = world.adapter.advance

    def advance(self, state, parameters, path, command, action, previous, previous_previous, context, noise):
        # Modify only this fresh predicted path; weights, recorded labels and simulator stay untouched.
        path.copy_(self.auxiliaries.path(state, parameters, path, action, previous, self.mode))
        return self.original_advance(state, parameters, path, command, action,
            previous, previous_previous, context, noise)

    def step(self, *args, **kwargs):
        with patch.object(self.world.adapter, 'advance', side_effect=self.advance):
            return self.world.step(*args, **kwargs)


def errors(value):
    flat = value.double().reshape(-1, value.shape[-1])
    return dict(mae=flat.abs().mean(0).tolist(), max_abs=flat.abs().amax(0).tolist(),
        scalar_values=flat.numel())


def reconstruction(record, world, auxiliaries):
    data = {name: value.flatten(0, 1) for name, value in record['data'].items()}
    if not record['live'].all() or record['data']['done'].any():
        raise ValueError('This feasibility probe requires the original complete first episodes')
    parameters = data['parameters']
    if (parameters[:, 63] != 0.).any() or (parameters[:, 24:26] < 0).any() or (parameters[:, 24:26] > 5).any():
        raise ValueError('Unsupported ground-start gate or delay beyond one policy step')
    torque_error, wheel_error = [], []
    checks = {mode: dict(reward_max_abs=0., frame_max_abs=0., feature_max_abs=0.,
        failure_mismatches=0, memory_max_abs=0.) for mode in MODES}
    for first in range(0, len(parameters), 2048):
        last = min(first+2048, len(parameters)); get = lambda name: data[name][first:last]
        true_path = get('path').reshape(-1, 5, 64)
        revised = auxiliaries.path(get('state'), get('parameters'), true_path, get('action'), get('previous'), 'both')
        torque_error.append(revised[..., 49:55]-true_path[..., 49:55])
        wheel_error.append(revised[..., auxiliaries.wq]-true_path[..., auxiliaries.wq])
        baseline_values = None
        for mode in MODES:
            path = true_path if mode == 'baseline' else auxiliaries.path(
                get('state'), get('parameters'), true_path, get('action'), get('previous'), mode)
            state, frame, reward, failure, features = world.adapter.advance(get('state'), get('parameters'),
                path, get('next_frame')[:, 6:9], get('action'), get('previous'),
                get('previous_previous'), get('context'), get('next_state')[:, NOISE])
            if mode == 'baseline':
                # Keep the original CPU oracle tolerance; full recordings cover more than its sampled endpoints.
                torch.testing.assert_close(reward, get('reward'), atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(frame, get('next_frame'), atol=2e-5, rtol=2e-5)
                baseline_values = (state, frame, reward, failure, features)
            row = checks[mode]
            for field, actual, truth in (('reward', reward, get('reward')), ('frame', frame, get('next_frame')),
                    ('feature', features, get('features')), ('memory', state[:, 244:254], get('next_state')[:, 244:254])):
                row[field+'_max_abs'] = max(row[field+'_max_abs'], float((actual-truth).abs().max()))
            row['failure_mismatches'] += int((failure != get('failure')).sum())
            row['reward_vs_CPU_baseline_max_abs'] = max(row.get('reward_vs_CPU_baseline_max_abs', 0.),
                float((reward-baseline_values[2]).abs().max()))
            row['frame_vs_CPU_baseline_max_abs'] = max(row.get('frame_vs_CPU_baseline_max_abs', 0.),
                float((frame-baseline_values[1]).abs().max()))
    # Integrate all 15000 physical ticks per environment without resetting q every control tick.
    source = record['data']; true_path = source['path'].reshape(3000, 24, 5, 64)
    velocities = true_path[..., auxiliaries.wv].permute(0, 2, 1, 3).reshape(15000, 24, 2)
    actual_q = true_path[..., auxiliaries.wq].permute(0, 2, 1, 3).reshape(15000, 24, 2)
    integrated = source['state'][0, :, auxiliaries.wq].clone()
    continuous_error = []
    for tick in range(15000):
        integrated = integrated+.002*velocities[tick]
        continuous_error.append(integrated-actual_q[tick])
    baseline = checks['baseline']
    assert baseline['memory_max_abs'] == 0. and baseline['failure_mismatches'] == 0
    return dict(transitions=72000, physical_substeps=360000, motor_order=list(ACTUATED_JOINTS),
        torque_Nm=errors(torch.cat(torque_error)), wheel_q_rad=errors(torch.cat(wheel_error)),
        wheel_q_continuous_30s_rad=errors(torch.stack(continuous_error)), adapter_checks=checks,
        actual_delay_ticks=sorted(parameters[:, 24:26].unique().tolist()),
        actual_strength_min=float(parameters[:, 18:24].min()), actual_strength_max=float(parameters[:, 18:24].max()),
        scope='True recorded motion supplied; no forecast accuracy claim. No failures in these two recordings; failure-boundary and randomized-delay fidelity remain untested.')


def summary(pred, windows):
    arrays, _, _, _ = core.metric_arrays(pred, windows['truth'])
    result = {}
    keys = ('reward_link_vx_m_s', 'rotation_geodesic_deg', 'support_left_N', 'support_right_N',
            'support_total_bodyweight', 'reward_weighted_dt')
    for h in (1, 8, 24):
        mask = windows['mask'][:, h-1]
        row = {key: core.stats(arrays[key][0][:, h-1], arrays[key][1][:, h-1], mask) for key in keys}
        row['hard_failure'] = core.confusion(pred['failure'][:, h-1], windows['truth']['failure'][:, h-1],
            mask, torch.ones_like(mask))
        row['nonfinite_path_state_features'] = sum(int((~pred[key][:, :h].isfinite()).sum()) for key in ('path', 'state', 'features'))
        row['cumulative_reward'] = core.stats(pred['reward'][:, :h].sum(1),
            windows['truth']['reward'][:, :h].sum(1), mask)
        result[str(h)] = row
    return result


@torch.inference_mode()
def run(args):
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/Path(__file__).name)
    started = time.monotonic()
    current = core.current_training_sources()
    old, bridge_proof = bridge.source_bridge(current)
    infos = contracts(command_cfg('walk'))
    loaded = bridge.load_world('V11-final', args.checkpoint.resolve(), current, infos)
    world = loaded['world']; world_digest = core.state_digest(world)
    model = load_model()
    sealed_path = bridge.NEW_TRIAL/'score/predictions.pt'
    sealed_score = json.loads((bridge.NEW_TRIAL/'score/score.json').read_text())
    assert core.digest(sealed_path) == sealed_score['predictions_sha256']
    sealed = torch.load(sealed_path, map_location='cpu', weights_only=False, mmap=True)
    results, predictions, inputs = {}, [], []
    for index, name in enumerate(('midpoint-actor', 'final-actor')):
        directory = bridge.OLD_TRIAL/name
        record = bridge.load_capture(directory, old, current, infos)
        auxiliaries = AnalyticAuxiliaries(model, directory/'env.yaml')
        true_check = reconstruction(record, world, auxiliaries)
        windows = core.make_windows(record)
        assert windows['mask'].all() and len(windows['plan']) == 360
        frozen = {}
        for mode in MODES:
            candidate = world if mode == 'baseline' else SubstitutedWorld(world, auxiliaries, mode)
            pred = core.predict(candidate, windows, 128)
            if mode == 'baseline':
                reference = next(item['tensors'] for item in sealed['predictions'] if
                    item['world_label'] == 'final' and item['recording_index'] == index and item['mode'] == 'free')
                assert pred.keys() == reference.keys()
                assert all(torch.equal(pred[key], reference[key]) for key in pred), 'Baseline differs from sealed V11 predictions'
            frozen[mode] = summary(pred, windows)
            predictions.append(dict(recording=name, mode=mode, tensors=pred))
        results[name] = dict(reconstruction=true_check, frozen_H24_substitution=frozen)
        inputs.append(dict(recording=name, files=record['proof']['files'], config_sha256=core.digest(directory/'env.yaml')))
        print(json.dumps(dict(recording=name, reconstruction=true_check, H24={m:v['24'] for m,v in frozen.items()}), allow_nan=False), flush=True)
        del record, windows
    assert core.state_digest(world) == world_digest
    assert core.current_training_sources() == current
    for source in inputs:
        for meta in source['files'].values():
            assert core.digest(meta['path']) == meta['sha256']
    assert core.digest(sealed_path) == sealed_score['predictions_sha256']
    torch.save(dict(predictions=predictions, inputs=inputs, checkpoint=loaded['proof']), args.output/'predictions.pt')
    report = dict(completed=True, diagnostic_kind='analytic_auxiliaries_isolated_feasibility',
        generated_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic()-started,
        script_sha256=core.digest(__file__), checkpoint=loaded['proof'], bridge_proof=bridge_proof,
        source99_unchanged=True, original_recordings_unchanged=True, sealed_V11_baseline_bitwise_equal=True,
        world_and_norm_unchanged=True, optimizer_steps=0, normalization_updates=0,
        actor_forward_calls=0, critic_forward_calls=0, simulator_steps=0,
        predictions_sha256=core.digest(args.output/'predictions.pt'), inputs=inputs, results=results,
        limitations=['No reduced-output network was trained; original network still produces all outputs.',
            'Frozen substitution changes its input distribution and is not evidence of retraining benefit.',
            'True-motion reconstruction uses true q/qd; free forecast substitution uses only predicted motion.',
            'Only the two existing nominal-contact recordings are tested, with zero delays, unity motor strengths and no failures.',
            'Contact-force outputs remain learned and unchanged by this candidate; production/XML/H7/checkpoints are not edited.'])
    core.write(args.output/'report.json', report)
    lines = ['# 解析力矩与轮速积分：隔离验证', '',
        '仅新增诊断入口与本目录。无训练、新仿真、生产替换；冻结V11原模型。', '',
        '| 轨迹 | 方案 | H24速度MAE m/s | 姿态MAE° | 左/右承重MAE N | 失败误报 |',
        '|---|---|---:|---:|---:|---:|']
    for name, result in results.items():
        for mode, horizons in result['frozen_H24_substitution'].items():
            h = horizons['24']
            number = lambda key: '—' if h[key]['mae'] is None else f"{h[key]['mae']:.6f}"
            lines.append(f"| {name} | {mode} | {number('reward_link_vx_m_s')} | {number('rotation_geodesic_deg')} | {number('support_left_N')}/{number('support_right_N')} | {h['hard_failure']['fp']} |")
    lines += ['', '真实运动重建与每电机误差见report.json。原模型七类封存预测逐位复现。', '',
        '该试验不能证明删减输出后的训练效果；延迟/强度随机化、失败边界与策略闭环尚未验证。']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No training authorized')))
        stack.enter_context(patch.object(RunningNorm, 'update', side_effect=AssertionError('Norms frozen')))
        run(args)
