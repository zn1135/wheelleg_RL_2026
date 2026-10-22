#!/usr/bin/env python3
"""Offline measured-input MuJoCo predictions; never opens a hardware connection.

The fixed fixture identifies conditional command-response parameters, not physical
wheel inertia. Full-body predictions use recorded actions and the original PD.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

import mj_sim2sim_huananhu_multimode as sim
import real2sim_huananhu as mirror

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'deployment/H7_RL_ros/imcalib/config/machine_config.c'
FIELDS = ('armature', 'damping', 'frictionloss')
UPPER = np.array((.012, .03, .5))
I2N = [mirror.MOTORS.index(n) for n in sim.MOTORS]
N2I = [sim.MOTORS.index(n) for n in mirror.MOTORS]
HORIZONS = (10, 20, 50, 100)


def identity(path):
    return mirror.file_identity(Path(path))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def stats(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    return {'mae': float(np.mean(abs(a))), 'rmse': float(np.sqrt(np.mean(a*a))),
            'mean': float(a.mean()), 'p95_abs': float(np.percentile(abs(a), 95))}


def signed_tick_delta(value, reference):
    return ((int(value) - int(reference) + 2**31) % 2**32) - 2**31


class Session:
    """Reparse the hashed raw stream, never trust a possibly stale analysis CSV."""

    def __init__(self, directory, model=None, config=CONFIG):
        self.directory = Path(directory)
        self.meta, self.model_path, self.config = mirror.load_session(self.directory, model, config)
        parser, clock = mirror.FrameParser(), mirror.FrameClock()
        frames, receive, ticks, segments = [], [], [], []
        segment, previous_tick, elapsed = 0, None, 0.
        connections = []
        for event in mirror.iter_session_events(self.directory):
            if event['type'] != 'data':
                connections.append({'type': event['type'], 't_s': event['t_ns']/1e9})
                parser.buffer.clear()
                segment += 1
                continue
            for frame in parser.feed(bytes.fromhex(event['data_hex'])):
                old = (clock.missing, clock.restarts)
                if not clock.accept(frame):
                    continue
                if frame[4] != 2:
                    raise ValueError('Dynamics comparison requires mirror v2 control telemetry')
                tick = mirror.read_u32(frame[8:10])
                if old != (clock.missing, clock.restarts):
                    segment += 1
                if previous_tick is not None:
                    delta = (tick - previous_tick) % 2**32
                    elapsed += delta if delta < 2**31 and delta else 1
                previous_tick = tick
                frames.append(frame); receive.append(event['t_ns']/1e9)
                ticks.append(elapsed); segments.append(segment)
        if len(frames) < 12:
            raise ValueError('Not enough v2 frames')
        self.f = np.asarray(frames)
        modes = np.unique(self.f[:, 158].astype(int))
        if len(modes) != 1 or str(modes[0]) not in mirror.WHEEL_RESPONSE_CATALOG:
            raise ValueError('Mixed or unknown wheel response profiles; use a separate session per firmware')
        self.wheel_response_mode = int(modes[0])
        profile = mirror.WHEEL_RESPONSE_CATALOG[str(self.wheel_response_mode)]
        if self.wheel_response_mode and self.meta.get('wheel_response_catalog', {}).get(str(self.wheel_response_mode)) != profile:
            raise ValueError('Recorded wheel response catalog missing or changed; update recorder before collection')
        self.wheel_friction = np.array(profile['friction_nm']) if self.wheel_response_mode else None
        self.rx, self.ms, self.segment = np.array(receive), np.array(ticks), np.array(segments)
        self.sign, self.zero = mirror.load_mapping(self.config)
        self.q = np.column_stack((self.f[:, 10:14] - self.zero, self.f[:, 18:20])) * self.sign
        self.v = np.column_stack((self.f[:, 14:18], self.f[:, 20:22])) * self.sign
        self.torque = self.f[:, 132:138].astype(float) * self.sign
        self.source_ms = self.ms[:, None] - self.f[:, 138:144]
        self.output_ms = self.ms + np.array([signed_tick_delta(mirror.read_u32(f[105:107]),
                                      mirror.read_u32(f[8:10])) for f in self.f])
        ages = self.f[:, 138:144]
        self.valid = ((self.f[:, 29] == 127) & (self.f[:, 2] == 0)
                      & ((self.f[:, 47].astype(int) & 2) != 0)
                      & np.all((ages >= 0) & (ages < 10), axis=1)
                      & np.isfinite(self.q).all(axis=1) & np.isfinite(self.v).all(axis=1)
                      & np.isfinite(self.torque).all(axis=1)
                      & (abs(self.output_ms-self.ms) < 10)
                      & (self.f[:, 30] >= 0) & (self.f[:, 30] < 200)
                      & (self.f[:, 31] >= 0) & (self.f[:, 31] < 200)
                      & ((self.f[:, 109].astype(int) & 8) != 0)
                      & np.isfinite(self.f[:, 22:29]).all(axis=1)
                      & (abs(np.linalg.norm(self.f[:, 22:26], axis=1)-1) < .001))
        applied = self.valid & ((self.f[:, 109].astype(int) & 63) == 61) & (self.f[:, 154] == 3)
        wheel_velocity = self.f[applied, 130:132].astype(float)
        expected = .1*(self.f[applied, 114:116]-wheel_velocity)
        if self.wheel_friction is not None:
            expected += self.wheel_friction*np.tanh(wheel_velocity/profile['smoothing_rad_s'])
        expected = np.clip(expected, -3.9, 3.9)
        if expected.size and np.max(abs(expected-self.torque[applied, 4:])) > 1e-4:
            raise ValueError('Wheel control snapshot disagrees with recorded response profile')
        self.wheel_response_verified = int(applied.sum())
        self.summary = {'directory': str(self.directory), 'frames': len(frames),
                        'stream_sha256': self.meta['stream_sha256'],
                        'crc_rejected_candidates': parser.rejected, 'missing': clock.missing,
                        'restarts': clock.restarts, 'duplicates': clock.duplicates,
                        'out_of_order': clock.out_of_order, 'connections': connections,
                        'valid_frames': int(self.valid.sum()), 'duration_s': float(self.rx[-1]-self.rx[0])}

    def continuous(self, start, end):
        return (0 <= start <= end < len(self.f) and self.valid[start:end+1].all()
                and self.segment[start] == self.segment[end]
                and np.all((np.diff(self.ms[start:end+1]) > 0)
                           & (np.diff(self.ms[start:end+1]) <= 20)))


def numeric_expression(text):
    """Read C numeric arithmetic without evaluating names, calls or Python code."""
    tree = ast.parse(re.sub(r'(?<=\d)[fFuUlL]\b', '', text.strip()), mode='eval')
    def number(node):
        if isinstance(node, ast.Constant) and type(node.value) in (float, int):
            return float(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div, ast.Add, ast.Sub)):
            a, b = number(node.left), number(node.right)
            return {ast.Mult: lambda: a*b, ast.Div: lambda: a/b,
                    ast.Add: lambda: a+b, ast.Sub: lambda: a-b}[type(node.op)]()
        raise ValueError('Unsupported C numeric expression')
    return number(tree.body)


def current_scale(config):
    text = Path(config).read_text().split('[MACHINE_ID_BIG_WHEELLEG] = {', 1)[1]
    text = text.split('[MACHINE_ID_SMALL_WHEELLEG]', 1)[0]
    motor_value = re.search(r'\.dji_type\s*=\s*([^,]+)', text).group(1).strip()
    if motor_value.startswith('DJI_'):
        motor = motor_value[4:]
    else:
        enums = (Path(config).parent.parent / 'common/motor_types.h').read_text()
        motor = next(n for n, v in re.findall(r'DJI_(M\d+)\s*=\s*(\d+)', enums)
                     if int(v) == numeric_expression(motor_value))
    ratio = numeric_expression(re.search(r'\.dji_gear_ratio\s*=\s*([^,]+)', text).group(1))
    header = Path(config).parent.parent / 'drivers/dji.h'
    defines = header.read_text()
    def constant(prefix):
        return numeric_expression(re.search(r'^#define\s+' + prefix + '_' + motor +
                                            r'\s+([^\r\n]+)', defines, re.M).group(1))
    return constant('DJI_NM_FULL') / constant('DJI_CURRENT_MAX') * ratio / constant('DJI_RATIO_STD')


def range_excess(model, qpos):
    return {model.joint(i).name: float(max(model.jnt_range[i, 0]-qpos[model.jnt_qposadr[i]],
                                           qpos[model.jnt_qposadr[i]]-model.jnt_range[i, 1], 0))
            for i in range(model.njnt) if model.jnt_limited[i]
            and not model.jnt_range[i, 0]-1e-8 <= qpos[model.jnt_qposadr[i]] <= model.jnt_range[i, 1]+1e-8}


def pose_from_sample(session, index):
    pose = mirror.MirrorPose(session.model_path, session.config)
    pose.data.qpos[:3] = 0
    pose.data.qpos[3:7] = session.f[index, 22:26]
    for side in (0, 1):
        if not pose._solve_leg(side, session.q[index, 2*side:2*side+2]):
            raise ValueError(f'Closure failed at frame {index}')
    pose.data.qpos[pose.wheel_q] = session.q[index, 4:]
    mujoco.mj_forward(pose.model, pose.data)
    return pose


class Fixture:
    """Bake measured nonwheel poses into a rigid fixture; only two hinges remain."""

    def __init__(self, model_path, qpos, dt=.001):
        source = mujoco.MjModel.from_xml_path(str(model_path))
        from wheel_response import prepare_wheel_response
        prepare_wheel_response(source)
        data = mujoco.MjData(source)
        data.qpos[:] = qpos
        # Wheel angle remains a simulated coordinate; do not bake it twice.
        for name in mirror.WHEELS:
            data.qpos[source.joint(name).qposadr] = 0
        mujoco.mj_forward(source, data)
        spec = mujoco.MjSpec.from_file(str(model_path))
        for i in range(1, source.nbody):
            parent = source.body_parentid[i]
            rp, rb = data.xmat[parent].reshape(3, 3), data.xmat[i].reshape(3, 3)
            body = spec.body(source.body(i).name)
            body.pos = rp.T @ (data.xpos[i]-data.xpos[parent])
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, (rp.T @ rb).ravel())
            body.quat = quat
        for joint in list(spec.joints):
            if joint.name not in mirror.WHEELS:
                spec.delete(joint)
        for eq in list(spec.equalities):
            spec.delete(eq)
        for actuator in list(spec.actuators):
            if actuator.name not in tuple(n+'_motor' for n in mirror.WHEELS):
                spec.delete(actuator)
        self.m = spec.compile()
        self.m.geom_contype[:] = self.m.geom_conaffinity[:] = 0
        self.m.opt.timestep = dt
        self.m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.m.opt.tolerance = 1e-10
        self.m.opt.iterations = 100
        self.d = mujoco.MjData(self.m)
        self.q = np.array([self.m.joint(n).qposadr[0] for n in mirror.WHEELS])
        self.v = np.array([self.m.joint(n).dofadr[0] for n in mirror.WHEELS])
        self.a = np.array([self.m.actuator(n+'_motor').id for n in mirror.WHEELS])
        self.original = np.column_stack([getattr(self.m, 'dof_'+n)[self.v] for n in FIELDS])
        assert self.m.nv == 2 and self.m.neq == 0
        mujoco.mj_forward(self.m, self.d)
        self.fixed_positions = self.d.xpos.copy()
        self.bake_error_m = float(max(np.linalg.norm(self.d.body(source.body(i).name).xpos-data.xpos[i])
                                      for i in range(1, source.nbody)))
        if self.bake_error_m > 1e-8:
            raise ValueError(f'Fixture pose bake failed: {self.bake_error_m}')

    def parameters(self, values):
        for k, name in enumerate(FIELDS):
            getattr(self.m, 'dof_'+name)[self.v] = self.original[:, k] + values[:, k]
        mujoco.mj_setConst(self.m, self.d)

    def predict(self, inputs, q0, v0, parameters):
        self.parameters(parameters)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.q], self.d.qvel[self.v] = q0, v0
        q, v = [q0.copy()], [v0.copy()]
        # No measurement is available inside this integration loop.
        for torque in inputs:
            self.d.ctrl[self.a] = torque
            mujoco.mj_step(self.m, self.d)
            q.append(self.d.qpos[self.q].copy()); v.append(self.d.qvel[self.v].copy())
        if max(self.d.warning.number) or not np.isfinite(v).all():
            raise RuntimeError('Fixture simulation warning or nonfinite state')
        return np.asarray(q), np.asarray(v)


class Pulse:
    def __init__(self, session, indices, number, scale, next_start):
        self.number, self.s, self.e = number, int(indices[0]), int(indices[-1])
        self.side = int(np.argmax(abs(session.torque[self.s, 4:])))
        self.direction = int(np.sign(session.torque[self.s, self.side+4]))
        self.group = mirror.WHEELS[self.side] + ('+' if self.direction > 0 else '-')
        self.start = self.s-2
        self.end = min(int(np.searchsorted(session.ms, session.ms[self.e]+1800)), next_start-2, len(session.f)-1)
        if not session.continuous(self.start, self.end):
            raise ValueError(f'Pulse {number}: invalid, missing, stale or disconnected feedback in window')
        if np.any(abs(session.torque[self.start:self.end+1, :4]) > 1e-6):
            raise ValueError(f'Pulse {number}: nonzero leg request')
        if np.any(abs(session.torque[self.start:self.end+1, 5-self.side]) > 1e-6):
            raise ValueError(f'Pulse {number}: simultaneous wheel request')
        self.t0 = float(session.source_ms[self.start, self.side+4])/1000
        self.time = (session.source_ms[self.start:self.end+1, self.side+4]/1000)-self.t0
        if np.any(np.diff(self.time) <= 0):
            raise ValueError('Nonmonotonic source time')
        self.actual_q = session.q[self.start:self.end+1, self.side+4].copy()
        self.actual_v = session.v[self.start:self.end+1, self.side+4].copy()
        self.grid = np.arange(int(np.ceil(self.time[-1]/.001))+1)*.001
        self.q0 = np.array([np.interp(self.t0*1000, session.source_ms[:, j+4], session.q[:, j+4]) for j in (0, 1)])
        self.v0 = np.array([np.interp(self.t0*1000, session.source_ms[:, j+4], session.v[:, j+4]) for j in (0, 1)])
        self.amplitude = float(session.torque[self.s, self.side+4])
        if not np.allclose(session.torque[indices, self.side+4], self.amplitude):
            raise ValueError('Pulse amplitude changed; fixed-amplitude bench protocol required')
        on = session.output_ms[self.s-1:self.s+1]/1000-self.t0
        off = session.output_ms[self.e:self.e+2]/1000-self.t0
        self.on_interval, self.request_off_interval = on.copy(), off.copy()
        self.on, self.off = float(np.mean(on)), float(np.mean(off))
        self.current_time = (session.source_ms[self.start:self.end+1, self.side+4]/1000-self.t0)
        self.current = session.f[self.start:self.end+1, 150+self.side].astype(float)*scale*session.sign[self.side+4]
        # Current is timing evidence, not an independently measured mechanical torque.
        near = np.arange(self.e, min(self.e+5, self.end)+1)
        threshold = .5*np.median(abs(session.f[indices, 150+self.side]))
        below = near[abs(session.f[near, 150+self.side]) < threshold]
        crossing = int(below[0]) if len(below) else int(near[-1])
        current_off = session.source_ms[max(crossing-1, self.start):crossing+1, self.side+4]/1000-self.t0
        self.off_interval = np.array([min(off[0], current_off[0]-.001), max(off[1], current_off[-1]+.001)])
        self.fit_mask = ((self.time >= self.on+.015) & (self.time <= self.off-.015)) | (self.time >= self.off_interval[1]+.03)
        pose = pose_from_sample(session, self.start)
        self.full_qpos, self.model_path = pose.data.qpos.copy(), session.model_path
        self.fixture = Fixture(session.model_path, self.full_qpos)
        self.input_cache = {}
        self.metadata = {'pulse': number, 'group': self.group, 'rows': [self.s, self.e],
                         'samples': len(indices), 'nominal_request_nm': self.amplitude,
                         'on_interval_s': on.tolist(), 'request_off_interval_s': off.tolist(),
                         'current_evidence_off_interval_s': current_off.tolist(),
                         'off_sensitivity_interval_s': self.off_interval.tolist(),
                         'fixture_pose_error_m': self.fixture.bake_error_m,
                         'initial_model_range_excess_rad': range_excess(pose.model, pose.data.qpos),
                         'leg_motion_peak_to_peak_rad': np.ptp(session.q[self.start:self.end+1, :4], axis=0).tolist()}

    def inputs(self, hypothesis, edges=None, shift=0):
        key = (hypothesis, edges, shift)
        if key not in self.input_cache:
            t = self.grid[:-1]+.0005
            values = np.zeros((len(t), 2))
            if hypothesis == 'request':
                on, off = (self.on, self.off) if edges is None else edges
                values[:, self.side] = self.amplitude*((t >= on) & (t < off))
            else:
                values[:, self.side] = np.interp(t-shift, self.current_time, self.current)
            self.input_cache[key] = values
        return self.input_cache[key]

    def predict(self, parameters, hypothesis, edges=None, shift=0):
        return self.fixture.predict(self.inputs(hypothesis, edges, shift), self.q0, self.v0, parameters)

    def sampled(self, values):
        return np.interp(self.time, self.grid, values[:, self.side])


def pulses_from_session(session):
    active = np.flatnonzero(np.any(abs(session.torque[:, 4:]) > 1e-6, axis=1))
    groups = np.split(active, np.flatnonzero(np.diff(active) > 1)+1)
    if len(active) == 0:
        raise ValueError('No wheel pulses')
    scale = current_scale(session.config)
    pulses, rejected = [], []
    for k, group in enumerate(groups):
        try:
            pulses.append(Pulse(session, group, k+1, scale, int(groups[k+1][0]) if k+1 < len(groups) else len(session.f)))
        except ValueError as exc:
            rejected.append({'pulse': k+1, 'reason': str(exc)})
    return pulses, rejected, scale


def fit_parameters(pulses, hypothesis):
    values, diagnostics = np.zeros((2, 3)), []
    for side in (0, 1):
        selected = [p for p in pulses if p.side == side]
        if {p.direction for p in selected} != {-1, 1}:
            raise ValueError('Fit needs both wheel directions on each side')
        def residual(x):
            params = np.zeros((2, 3)); params[side] = x
            return np.concatenate([(p.sampled(p.predict(params, hypothesis)[1])-p.actual_v)[p.fit_mask]
                                   /np.sqrt(p.fit_mask.sum()) for p in selected])
        fits = [least_squares(residual, initial, bounds=(np.zeros(3), UPPER),
                              x_scale=np.array((.004, .004, .15)), max_nfev=65,
                              ftol=2e-6, xtol=2e-6, gtol=2e-6)
                for initial in (np.array((.004, .003, .12)), np.array((.001, .015, .3)))]
        best = min(fits, key=lambda x: np.sum(x.fun**2))
        values[side] = best.x
        diagnostics.append({'wheel': mirror.WHEELS[side], 'parameters': dict(zip(FIELDS, best.x.tolist())),
                            'multistart': [{'cost': float(np.sum(f.fun**2)), 'x': f.x.tolist(),
                                            'success': bool(f.success), 'nfev': f.nfev} for f in fits],
                            'jacobian_singular_values': np.linalg.svd(best.jac, compute_uv=False).tolist()})
    return values, diagnostics


def profile_parameters(pulses, hypothesis, values):
    profiles = []
    for side in (0, 1):
        selected = [p for p in pulses if p.side == side]
        for k, field in enumerate(FIELDS):
            free = [j for j in range(3) if j != k]
            for fixed in sorted(set((0., float(values[side, k]), min(float(values[side, k]*2+.00001), UPPER[k])))):
                def residual(x):
                    pmat = values.copy(); pmat[side, k] = fixed; pmat[side, free] = x
                    return np.concatenate([(p.sampled(p.predict(pmat, hypothesis)[1])-p.actual_v)[p.fit_mask]
                                           /np.sqrt(p.fit_mask.sum()) for p in selected])
                fit = least_squares(residual, values[side, free], bounds=(np.zeros(2), UPPER[free]),
                                    x_scale=np.array((.004, .004, .15))[free], max_nfev=22)
                profiles.append({'wheel': mirror.WHEELS[side], 'field': field, 'fixed': fixed,
                                 'profile_cost': float(np.sum(fit.fun**2)), 'other_parameters': fit.x.tolist(),
                                 'converged': bool(fit.success)})
    return profiles


def pulse_metrics(pulse, q, v):
    pq, pv = pulse.sampled(q), pulse.sampled(v)
    stop = lambda a: next((float(pulse.time[i]) for i in range(len(a)-2)
                          if pulse.time[i] > pulse.off_interval[1]+.03 and np.all(abs(a[i:i+3]) < .2)), None)
    middle = (pulse.time >= pulse.on+.015) & (pulse.time <= pulse.off-.015)
    slope_error = None
    if middle.sum() >= 2:
        actual_slope = np.polyfit(pulse.time[middle], pulse.actual_v[middle], 1)[0]
        predicted_slope = np.polyfit(pulse.time[middle], pv[middle], 1)[0]
        slope_error = {'measured_rad_s2': float(actual_slope), 'predicted_rad_s2': float(predicted_slope),
                       'relative_error': float(abs(predicted_slope-actual_slope)/max(abs(actual_slope), 1e-9))}
    return {'q_error_rad': stats(pq-pulse.actual_q), 'velocity_error_rad_s': stats(pv-pulse.actual_v),
            'interior_velocity_error_rad_s': stats((pv-pulse.actual_v)[pulse.fit_mask]),
            'coast_velocity_error_rad_s': stats((pv-pulse.actual_v)[pulse.time > pulse.off_interval[1]+.03]),
            'predicted_peak_rad_s': float(max(abs(pv))), 'measured_peak_rad_s': float(max(abs(pulse.actual_v))),
            'predicted_stop_s': stop(pv), 'measured_stop_s': stop(pulse.actual_v),
            'interior_acceleration': slope_error}


def export_candidate(model_path, output, values):
    tree = ET.parse(model_path)
    original = mujoco.MjModel.from_xml_path(str(model_path))
    for side, name in enumerate(mirror.WHEELS):
        joint = tree.find(f".//joint[@name='{name}']")
        address = original.joint(name).dofadr[0]
        for k, field in enumerate(FIELDS):
            joint.set(field, str(float(getattr(original, 'dof_'+field)[address]+values[side, k])))
    compiler = tree.find('compiler')
    compiler.set('meshdir', str((Path(model_path).parent / compiler.get('meshdir', '')).resolve()))
    tree.write(output, encoding='utf-8', xml_declaration=True)
    candidate = mujoco.MjModel.from_xml_path(str(output))
    # Compile both files and allow only the six requested wheel fields to differ.
    for name in ('body_mass', 'body_inertia', 'body_pos', 'body_quat', 'body_ipos', 'body_iquat',
                 'jnt_range', 'jnt_axis', 'jnt_limited', 'eq_data', 'eq_solref', 'eq_solimp',
                 'geom_pos', 'geom_quat', 'geom_friction', 'geom_contype', 'geom_conaffinity',
                 'mesh_vertnum', 'mesh_vert', 'actuator_gainprm', 'actuator_ctrlrange', 'actuator_gear'):
        if not np.allclose(getattr(original, name), getattr(candidate, name), rtol=1e-12, atol=1e-12):
            raise ValueError(f'Candidate modified forbidden field: {name}')
    wheels = [original.joint(n).dofadr[0] for n in mirror.WHEELS]
    for field in FIELDS:
        a, b = getattr(original, 'dof_'+field), getattr(candidate, 'dof_'+field)
        if not np.array_equal(np.delete(a, wheels), np.delete(b, wheels)):
            raise ValueError(f'Candidate modified a nonwheel {field}')
    if candidate.nq != original.nq or candidate.nv != original.nv or candidate.neq != original.neq:
        raise ValueError('Fixture constraints leaked into full-body candidate')
    return {'verified_fields': 26, 'nq': candidate.nq, 'nv': candidate.nv, 'neq': candidate.neq,
            'sha256': identity(output)['sha256']}


class RecordedRL:
    def __init__(self, session):
        self.s = session
        good = ((session.f[:, 29].astype(int) & 1) != 0) & np.isfinite(session.f[:, 22:26]).all(axis=1)
        good &= abs(np.linalg.norm(session.f[:, 22:26], axis=1)-1) < .001
        self.quats = Slerp(session.ms[good], Rotation.from_quat(session.f[good][:, [23, 24, 25, 22]]))
        self.policies = {}
        for i, f in enumerate(session.f):
            if not int(f[47]) & 1:
                continue
            seq = mirror.read_u32(f[48:50])
            key = (int(session.segment[i]), seq)
            tick = session.ms[i] + signed_tick_delta(mirror.read_u32(f[52:54]), mirror.read_u32(f[8:10]))
            value = (tick, f[96:102].astype(float))
            if key in self.policies and (tick != self.policies[key][0] or
                                        not np.array_equal(value[1], self.policies[key][1])):
                raise ValueError('Recorded policy pair changed for the same sequence')
            self.policies[key] = value
        self.by_segment = {}
        for segment in np.unique(session.segment):
            pairs = sorted((t, seq, a) for (seg, seq), (t, a) in self.policies.items() if seg == segment)
            if pairs:
                self.by_segment[segment] = (np.array([v[0] for v in pairs]), np.array([v[1] for v in pairs]),
                                            np.array([v[2] for v in pairs]))
        self.schedule_mismatches = {}
        for lag in (2, 10):
            bad, checked = 0, 0
            for i, f in enumerate(session.f):
                if not session.valid[i] or not int(f[109]) & 1 or f[154] != 3:
                    continue
                ticks, seq, _ = self.by_segment[session.segment[i]]
                ix = np.searchsorted(ticks+lag, session.output_ms[i], side='right')-1
                checked += 1
                bad += int(ix < 0 or seq[ix] != mirror.read_u32(f[107:109]))
            self.schedule_mismatches[str(lag)] = {'checked': checked, 'mismatched': bad}

    def attitude(self, tick):
        return self.quats(float(tick)).as_quat()[[3, 0, 1, 2]]

    def actions(self, index, lag):
        ticks, seq, actions = self.by_segment[self.s.segment[index]]
        at = self.s.output_ms[index]+np.arange(50)*2
        ix = np.searchsorted(ticks+lag, at, side='right')-1
        if np.any(ix < 0):
            raise ValueError('No preceding policy action for this window')
        return actions[ix].copy(), seq[ix]


def pitch(q):
    w, x, y, z = q
    return float(np.arcsin(np.clip(2*(w*y-x*z), -1, 1)))


class BodyPredictor:
    """Reusable model5849 PD kernel; prediction receives no future measurements."""

    def __init__(self, model_path, config, wheel_friction_compensation=None):
        self.env = sim.Playback('normal', (0., 0., .27), model_path=Path(model_path),
                                wheel_friction_compensation=wheel_friction_compensation)
        self.pose = mirror.MirrorPose(model_path, config)
        self.m, self.d = self.env.model, self.env.data
        self.passive = np.array([int(self.m.joint(n).dofadr[0]) for n in
                                 sum(mirror.PASSIVE, ()) + tuple(f'{s}_{p}_joint' for s in ('left', 'right')
                                                               for p in ('spring1', 'spring2'))])
        self.pairs = [p for side in mirror.PIN_PAIRS for p in side]
        self.pairs += [(f'{s}_spring_tip', f'{s}_gas_spring_lower') for s in ('left', 'right')]
        self.wheel_geoms = [[g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.m.body(n).id
                            and (self.m.geom_contype[g] or self.m.geom_conaffinity[g])] for n in mirror.WHEELS]
        self.contact = {n: getattr(self.m, 'geom_'+n).copy() for n in ('friction', 'solref')}

    def bottoms(self):
        points = []
        for geoms in self.wheel_geoms:
            vertices = []
            for g in geoms:
                mesh = self.m.geom_dataid[g]
                begin, count = self.m.mesh_vertadr[mesh], self.m.mesh_vertnum[mesh]
                vertices.append(self.m.mesh_vert[begin:begin+count] @ self.d.geom_xmat[g].reshape(3, 3).T
                                + self.d.geom_xpos[g])
            v = np.concatenate(vertices)
            points.append(v[np.argmin(v[:, 2])])
        return np.array(points)

    def measured_state(self, recorded, index):
        s, tick = recorded.s, recorded.s.output_ms[index]
        q, velocity = s.f[index, 122:126].astype(float), s.f[index, 126:132].astype(float)
        self.pose.data.qpos[:3] = 0
        self.pose.data.qpos[3:7] = recorded.attitude(tick)
        for side in (0, 1):
            if not self.pose._solve_leg(side, q[2*side:2*side+2]):
                raise ValueError('closure_failed')
            self.pose.data.qpos[self.pose.wheel_q[side]] = np.interp(tick, s.source_ms[:, side+4], s.q[:, side+4])
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.pose.data.qpos
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
        self.d.qvel[self.env.v] = velocity[I2N]
        self.d.qvel[3:6] = [np.interp(tick, s.ms, s.f[:, 26+k]) for k in range(3)]
        blocks = []
        for a, b in self.pairs:
            ja, jb = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.d, ja, None, self.m.site(a).id)
            mujoco.mj_jacSite(self.m, self.d, jb, None, self.m.site(b).id)
            blocks.append(ja-jb)
        jac = np.vstack(blocks)
        pv, _, rank, _ = np.linalg.lstsq(jac[:, self.passive], -jac @ self.d.qvel, rcond=1e-10)
        if rank != len(self.passive):
            raise ValueError('passive_velocity_rank')
        self.d.qvel[self.passive] = pv
        self.d.qpos[2] -= float(self.bottoms()[:, 2].min())
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
        points, relative = self.bottoms(), []
        for side, point in enumerate(points):
            j = np.zeros((3, self.m.nv))
            mujoco.mj_jac(self.m, self.d, j, None, point, self.m.body(mirror.WHEELS[side]).id)
            relative.append(j @ self.d.qvel)
        mujoco.mj_forward(self.m, self.d)
        excess = range_excess(self.m, self.d.qpos)
        if excess:
            raise ValueError('joint_range:'+json.dumps(excess))
        wheel_ids = sum(self.wheel_geoms, [])
        floor = self.m.geom('sim2sim_flat_floor').id
        if any(floor in (c.geom1, c.geom2) and c.dist < -1e-6 and
               not any(g in wheel_ids for g in (c.geom1, c.geom2)) for c in self.d.contact):
            raise ValueError('nonwheel_floor_penetration')
        lower = int(np.argmin(points[:, 2]))
        return self.d.qpos.copy(), self.d.qvel.copy(), {
            'lower_velocity': (-np.asarray(relative)[lower]).tolist(),
            'rolling_velocity': (-np.mean(relative, axis=0)).tolist(),
            'wheel_floor_gaps_m': points[:, 2].tolist(), 'closure_m': max(self.pose.residual_m),
            'velocity_closure_m_s': float(np.max(abs(jac @ self.d.qvel))),
            'lower_wheel': lower}

    def predict(self, qpos, qvel, actions, case):
        for name, value in self.contact.items():
            getattr(self.m, 'geom_'+name)[:] = value
        if case == 'friction_1':
            for geoms in self.wheel_geoms:
                self.m.geom_friction[geoms, 0] = 1.
        if case == 'solref_half':
            self.m.geom_solref[:, 0] *= .5
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:], self.d.qvel[:] = qpos, qvel
        self.env.ground_start, self.env.recovered, self.env.steps = True, True, 1000
        mujoco.mj_forward(self.m, self.d)
        qs, vs = [qpos.copy()], [qvel.copy()]
        for action in actions:
            torque, spring = self.env.controls(action)
            self.d.ctrl[self.env.motor_ids] = torque
            self.d.ctrl[self.env.spring_ids] = spring
            mujoco.mj_step(self.m, self.d)
            qs.append(self.d.qpos.copy()); vs.append(self.d.qvel.copy())
        if max(self.d.warning.number) or not np.isfinite(qs).all() or not np.isfinite(vs).all():
            raise RuntimeError('Full-body MuJoCo warning/nonfinite prediction')
        return np.array(qs), np.array(vs)


BODY_CASES = {
    'lower_contact': ('lower', -1e-6, 2), 'late_action': ('lower', -1e-6, 10),
    'zero_translation': ('zero', 0., 2), 'rolling_inferred': ('rolling', 0., 2),
    'rolling_late': ('rolling', 0., 10), 'floor_gap_2mm': ('rolling', .002, 2),
    'friction_1': ('lower', -1e-6, 2), 'solref_half': ('lower', -1e-6, 2),
}


def analyze_rl(session, candidate, out, stride):
    r = RecordedRL(session)
    predictors = {'original': BodyPredictor(session.model_path, session.config, session.wheel_friction),
                  'candidate': BodyPredictor(candidate, session.config, session.wheel_friction)}
    first = predictors['original']
    initialized, rejected, results, states_q, states_v, state_keys = [], [], [], [], [], []
    # Inspect the whole recording, including recovery boundaries and high speeds.
    possible = np.arange(max(0, len(session.f)-10))
    accepted = 0
    for i in possible:
        i = int(i)
        window = slice(i, i+11)
        if not session.continuous(i, i+10):
            rejected.append({'row': i, 'reason': 'invalid_or_discontinuous'}); continue
        # A computed PD snapshot alone does not prove the request passed the final gate.
        if (not np.all((session.f[window, 109].astype(int) & 63) == 61) or
                not np.all((session.f[window, 47].astype(int) & 1) != 0) or
                not np.isfinite(session.f[window, 96:102]).all() or
                not np.isfinite(session.f[window, 122:132]).all() or
                not np.all(session.f[window, 154] == 3) or
                not np.all(np.diff(session.output_ms[window]) == 10)):
            rejected.append({'row': i, 'reason': 'not_normal_applied_pd_or_irregular_tick'}); continue
        try:
            q, v, meta = first.measured_state(r, i)
        except ValueError as exc:
            rejected.append({'row': i, 'reason': str(exc)}); continue
        accepted += 1
        if (accepted-1) % stride:
            continue
        initialized.append({'row': i, 'host_s': float(session.rx[i]), **meta})
        t0 = session.output_ms[i]
        extrapolated = bool(np.max(abs(session.f[window, 130:132])) > 37)
        inputs = {lag: r.actions(i, lag)[0] for lag in (2, 10)}
        for case, (translation, gap, lag) in BODY_CASES.items():
            initial_q, initial_v = q.copy(), v.copy()
            initial_q[2] += gap
            if translation != 'zero':
                initial_v[:3] = meta['lower_velocity' if translation == 'lower' else 'rolling_velocity']
            for model_name, predictor in predictors.items():
                pq, pv = predictor.predict(initial_q, initial_v, inputs[lag], case)
                states_q.append(pq); states_v.append(pv)
                state_keys.append((i, case, model_name))
                for horizon in HORIZONS:
                    j, step = i+horizon//10, horizon//2
                    actual_q, actual_v = session.f[j, 122:126], session.f[j, 126:132]
                    pred_q = pq[step, predictor.env.q][N2I[:4]]
                    pred_v = pv[step, predictor.env.v][N2I]
                    continuation = q.copy()
                    mujoco.mj_integratePos(first.m, continuation, v, horizon/1000)
                    row = {'row': i, 'host_s': float(session.rx[i]), 'case': case, 'model': model_name,
                           'horizon_ms': horizon, 'extrapolated_over_37': int(extrapolated),
                           'pitch_error_deg': float(np.degrees(pitch(pq[step, 3:7])-pitch(r.attitude(t0+horizon)))),
                           'constant_gyro_pitch_error_deg': float(np.degrees(pitch(continuation[3:7])-pitch(r.attitude(t0+horizon))))}
                    for k, name in enumerate(mirror.MOTORS):
                        row[name+'_v_error'] = float(pred_v[k]-actual_v[k])
                        row[name+'_v_pred'] = float(pred_v[k]); row[name+'_v_real'] = float(actual_v[k])
                        row[name+'_constant_v_error'] = float(session.f[i, 126+k]-actual_v[k])
                        if k < 4:
                            row[name+'_q_error_deg'] = float(np.degrees(pred_q[k]-actual_q[k]))
                    results.append(row)
        if len(initialized) % 20 == 0:
            print(json.dumps({'phase': 'rl', 'windows': len(initialized), 'row': i}), flush=True)
    if not results:
        raise ValueError('No usable RL windows; see session validity and range constraints')
    with (out/'rl-predictions.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    np.savez_compressed(out/'rl-states.npz', qpos=np.array(states_q), qvel=np.array(states_v),
                        row=np.array([k[0] for k in state_keys]), case=np.array([k[1] for k in state_keys]),
                        model=np.array([k[2] for k in state_keys]), timestep_s=.002)
    columns = ([n+'_v_error' for n in mirror.MOTORS] + [n+'_q_error_deg' for n in mirror.MOTORS[:4]]
               + [n+'_constant_v_error' for n in mirror.MOTORS] + ['pitch_error_deg', 'constant_gyro_pitch_error_deg'])
    summaries = {}
    for case in BODY_CASES:
        for h in HORIZONS:
            for model in predictors:
                for coverage, flag in (('within_bench', 0), ('extrapolated', 1), ('all', None)):
                    rows = [x for x in results if x['case'] == case and x['horizon_ms'] == h and x['model'] == model
                            and (flag is None or x['extrapolated_over_37'] == flag)]
                    summaries[f'{case}/{h}/{model}/{coverage}'] = {'windows': len(rows),
                        **{c: stats([x[c] for x in rows]) for c in columns}}
    # This is an exploratory regression guard, never permission to modify the official XML.
    guard = []
    for h in HORIZONS:
        a, b = [summaries[f'lower_contact/{h}/{m}/within_bench'] for m in predictors]
        ok = bool(a['windows'])
        if ok:
            ok &= all(b[n+'_v_error']['mae'] < a[n+'_v_error']['mae'] for n in mirror.WHEELS)
            ok &= b['pitch_error_deg']['mae'] <= 1.1*a['pitch_error_deg']['mae']
            ok &= np.mean([b[n+'_q_error_deg']['mae'] for n in mirror.MOTORS[:4]]) <= 1.1*np.mean(
                [a[n+'_q_error_deg']['mae'] for n in mirror.MOTORS[:4]])
        guard.append({'horizon_ms': h, 'exploratory_guard_passed': bool(ok)})
    report = {'session': session.summary, 'attempted_windows': len(possible), 'valid_before_stride': accepted,
              'evaluated_windows': len(initialized), 'stride': stride, 'rejected_windows': rejected,
              'initialization': initialized, 'summaries': summaries, 'regression_guard': guard,
              'automatic_adoption': False, 'action_schedule_at_observed_ticks': r.schedule_mismatches,
              'wheel_response_mode': session.wheel_response_mode,
              'wheel_response_profile': mirror.WHEEL_RESPONSE_CATALOG[str(session.wheel_response_mode)],
              'wheel_control_snapshots_verified': session.wheel_response_verified,
              'window_selection': 'all recorded starts with 100ms subsequent data; invalid/non-NORMAL/limit-conflicting windows listed as rejected',
              'whole_session_wheel_speed_min_rad_s': np.min(session.v[:, 4:], axis=0).tolist(),
              'whole_session_wheel_speed_max_rad_s': np.max(session.v[:, 4:], axis=0).tolist(),
              'whole_session_frames_over_37_rad_s': int(np.any(abs(session.v[:, 4:]) > 37, axis=1).sum()),
              'assumptions': {'translation': 'unmeasured: lower-wheel or two-wheel no-slip inference, and zero sensitivity',
                              'support': 'unknown human support is not reproduced',
                              'source_alignment': 'PD snapshot at output Mono tick; public gyro/quaternion/wheel angle interpolated',
                              'actions': 'recorded actions; original 500Hz PD evaluated on predicted state; lag 2/10ms hypotheses',
                              'contacts': BODY_CASES,
                              'coverage': 'above 37rad/s separately reported as extrapolation; never silently omitted'}}
    save_json(out/'rl-report.json', report)
    return report, results


def analyze_bench(session, out):
    pulses, rejected, scale = pulses_from_session(session)
    groups = sorted({p.group for p in pulses})
    if len(groups) != 4 or any(sum(p.group == g for p in pulses) < 2 for g in groups):
        raise ValueError(f'Need two repetitions of each wheel direction; rejected pulses: {rejected}')
    forward = [next(p for p in pulses if p.group == g) for g in groups]
    reverse = [next(p for p in reversed(pulses) if p.group == g) for g in groups]
    report = {'session': session.summary, 'pulses': [p.metadata for p in pulses], 'rejected_pulses': rejected,
              'current_same_firmware_scale_nm_per_raw': scale, 'fits': {}, 'metrics': [],
              'edge_sensitivity': [], 'profiles': {}, 'automatic_adoption': False,
              'scope': 'conditional fixed-fixture response: mj_step reduces to two independent wheel equations; not whole-body calibration'}
    traces, csv_rows, primary = {}, [], None
    for hypothesis in ('request', 'current'):
        for fold, fitting in (('forward', forward), ('reverse', reverse)):
            print(json.dumps({'phase': 'bench_fit', 'hypothesis': hypothesis, 'fold': fold}), flush=True)
            params, fit_info = fit_parameters(fitting, hypothesis)
            key = f'{hypothesis}/{fold}'
            report['fits'][key] = {'parameters': params.tolist(), 'diagnostics': fit_info,
                                    'fit_pulses': [p.number for p in fitting],
                                    'held_out_pulses': [p.number for p in pulses if p not in fitting]}
            if key == 'request/forward':
                primary = params.copy()
                report['profiles'][key] = profile_parameters(fitting, hypothesis, params)
            for p in pulses:
                held_out = p not in fitting
                for model, values in (('original', np.zeros((2, 3))), ('candidate', params)):
                    q, v = p.predict(values, hypothesis)
                    report['metrics'].append({'hypothesis': hypothesis, 'fold': fold, 'pulse': p.number,
                                              'model': model, 'held_out': held_out, **pulse_metrics(p, q, v)})
                    if key == 'request/forward':
                        traces[f'p{p.number}_{model}_q'] = q
                        traces[f'p{p.number}_{model}_v'] = v
                    for j, t in enumerate(p.time):
                        csv_rows.append({'hypothesis': hypothesis, 'fold': fold, 'pulse': p.number,
                                         'model': model, 'held_out': int(held_out), 'time_s': float(t),
                                         'pred_q_rad': float(p.sampled(q)[j]), 'real_q_rad': float(p.actual_q[j]),
                                         'pred_v_rad_s': float(p.sampled(v)[j]), 'real_v_rad_s': float(p.actual_v[j])})
                if key == 'request/forward':
                    traces[f'p{p.number}_time'] = p.grid
                    traces[f'p{p.number}_input'] = p.inputs(hypothesis)
                    traces[f'p{p.number}_full_initial_qpos'] = p.full_qpos
                if held_out:
                    scenarios = ([(float(a), float(b)) for a in p.on_interval for b in p.off_interval]
                                 if hypothesis == 'request' else [-.005, .005])
                    for scenario in scenarios:
                        kwargs = {'edges': scenario} if hypothesis == 'request' else {'shift': scenario}
                        for model, values in (('original', np.zeros((2, 3))), ('candidate', params)):
                            q, v = p.predict(values, hypothesis, **kwargs)
                            report['edge_sensitivity'].append({'hypothesis': hypothesis, 'fold': fold,
                                'pulse': p.number, 'scenario': scenario, 'model': model, **pulse_metrics(p, q, v)})
        save_json(out/'bench-progress.json', report)
    # Same held-out input is supplied to both models at every sensitivity corner.
    report['held_out_comparison'] = []
    for hypothesis in ('request', 'current'):
        for fold in ('forward', 'reverse'):
            pairs = []
            for p in pulses:
                rows = [x for x in report['metrics'] if x['hypothesis'] == hypothesis and x['fold'] == fold
                        and x['pulse'] == p.number and x['held_out']]
                if rows:
                    original, candidate = [next(x for x in rows if x['model'] == m) for m in ('original', 'candidate')]
                    pairs.append({'pulse': p.number, 'original_rmse': original['velocity_error_rad_s']['rmse'],
                                  'candidate_rmse': candidate['velocity_error_rad_s']['rmse']})
            report['held_out_comparison'].append({'hypothesis': hypothesis, 'fold': fold, 'pulses': pairs,
                'all_improved_against_original': all(x['candidate_rmse'] < x['original_rmse'] for x in pairs),
                'interpretation': 'relative improvement over the original XML alone does not establish a calibrated model'})
    # Repeat the primary fixture with the deployed CPU model's 2ms timestep.
    report['timestep_sensitivity'] = []
    for p in pulses:
        fixture = Fixture(p.model_path, p.full_qpos, dt=.002)
        q, v = fixture.predict(p.inputs('request')[::2], p.q0, p.v0, primary)
        times = np.arange(len(v))*.002
        a = p.predict(primary, 'request')[1]
        difference = np.interp(p.time, times, v[:, p.side])-p.sampled(a)
        report['timestep_sensitivity'].append({'pulse': p.number, '2ms_minus_1ms_velocity_rad_s': stats(difference)})
    report['candidate_integrity'] = export_candidate(session.model_path, out/'candidate.xml', primary)
    save_json(out/'candidate-parameters.json', {'kind': 'conditional request-response candidate',
              'wheel_order': mirror.WHEELS, 'parameter_order': FIELDS, 'additional_parameters': primary.tolist(),
              'automatic_adoption': False, 'fit_pulses': report['fits']['request/forward']['fit_pulses'],
              'limitations': ['torque gain and absolute inertia cannot be separated',
                             'armature and wheel-body inertia have different free-base coupling',
                             'direction-asymmetric losses remain model residuals',
                             'current conversion is not independent shaft torque']})
    with (out/'bench-predictions.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    np.savez_compressed(out/'bench-states.npz', **traces)
    save_json(out/'bench-report.json', report)
    return report, pulses, traces


def plot_results(out, pulses, traces, rl_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for detail in (False, True):
        fig, axes = plt.subplots(2, 4, figsize=(17, 7), constrained_layout=True)
        for ax, p in zip(axes.ravel(), pulses):
            ax.plot(p.time, p.actual_v, color='black', label='Measured', linewidth=1.5)
            for model, color in (('original', 'tab:red'), ('candidate', 'tab:blue')):
                if detail and model == 'original':
                    continue
                ax.plot(p.grid, traces[f'p{p.number}_{model}_v'][:, p.side], color=color, label=model)
            ax.axvspan(*p.off_interval, alpha=.15, color='orange', label='Off-time uncertainty')
            ax.set(title=f'P{p.number} {p.group} (model axis)', xlabel='Time from initialization (s)', ylabel='Wheel speed (rad/s)')
            ax.grid(alpha=.25)
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle('Measured input -> MuJoCo free prediction (request convention, forward split)')
        fig.savefig(out/('bench-detail.png' if detail else 'bench-comparison.png'), dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(17, 6), constrained_layout=True)
    for ax, p in zip(axes.ravel(), pulses):
        ax.plot(p.grid[:-1], p.inputs('request')[:, p.side], label='Nominal request')
        ax.plot(p.current_time, p.current, '.-', label='Current converted by firmware scale')
        ax.axvspan(*p.off_interval, color='orange', alpha=.2)
        ax.set(xlim=(max(0, p.on-.02), p.off_interval[1]+.04), title=f'P{p.number} {p.group}',
               xlabel='Time (s)', ylabel='Nominal Nm (not measured shaft torque)')
    axes.ravel()[0].legend(fontsize=7)
    fig.savefig(out/'input-timing.png', dpi=160); plt.close(fig)
    if rl_rows:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for ax, field in zip(axes, ('lfwheel_v_error', 'rfwheel_v_error', 'pitch_error_deg')):
            for model in ('original', 'candidate'):
                ys = [np.mean([abs(r[field]) for r in rl_rows if r['case'] == 'lower_contact'
                               and r['model'] == model and r['horizon_ms'] == h]) for h in HORIZONS]
                ax.plot(HORIZONS, ys, 'o-', label=model)
            ax.set(xlabel='Prediction horizon (ms)', ylabel='Mean absolute error', title=field)
            ax.grid(alpha=.3); ax.legend()
        fig.suptitle('Held-supported RL recording: regression only, all speed ranges')
        fig.savefig(out/'rl-comparison.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-session', type=Path, required=True)
    parser.add_argument('--rl-session', type=Path, required=True)
    parser.add_argument('--machine-config', type=Path, default=CONFIG)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rl-stride', type=int, default=1, help='Evaluate every Nth valid RL start (default: every 10ms start)')
    args = parser.parse_args()
    if args.rl_stride < 1:
        parser.error('--rl-stride must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out/'run.json', {'complete': False, 'phase': 'loading'})
    try:
        bench = Session(args.bench_session, config=args.machine_config)
        rl = Session(args.rl_session, model=bench.model_path, config=args.machine_config)
        before = {'model': identity(bench.model_path), 'config': identity(args.machine_config)}
        source = args.out/'source'; source.mkdir()
        for path in (Path(__file__), Path(sim.__file__), Path(mirror.__file__), args.machine_config):
            shutil.copy2(path, source/path.name)
        firmware = args.machine_config.parent.parent
        for path in (firmware/'drivers/dji.h', firmware/'drivers/dji.c', firmware/'common/motor_types.h',
                     ROOT/'scripts/agent/check_real2sim_dynamics_huananhu.py'):
            shutil.copy2(path, source/path.name)
        (source/'tracked.diff').write_text(subprocess.run(
            ['git', 'diff', '--', 'sim2sim/mj_sim2sim_huananhu_multimode.py', 'sim2sim/real2sim_huananhu.py'],
            cwd=ROOT, text=True, capture_output=True, check=True).stdout)
        bench_report, pulses, traces = analyze_bench(bench, args.out)
        save_json(args.out/'run.json', {'complete': False, 'phase': 'rl'})
        rl_report, rows = analyze_rl(rl, args.out/'candidate.xml', args.out, args.rl_stride)
        plot_results(args.out, pulses, traces, rows)
        if before != {'model': identity(bench.model_path), 'config': identity(args.machine_config)}:
            raise RuntimeError('Official model or calibration changed during analysis')
        report = {'complete': True, 'scope': 'offline measured-input MuJoCo bench and full-body RL predictions',
                  'inputs': before, 'program': identity(Path(__file__)), 'simulator': identity(Path(sim.__file__)),
                  'mujoco_version': mujoco.__version__, 'bench': 'bench-report.json', 'rl': 'rl-report.json',
                  'command_arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  'source_snapshot': {p.name: identity(p)['sha256'] for p in source.iterdir() if p.is_file()},
                  'bench_held_out': bench_report['held_out_comparison'], 'rl_guard': rl_report['regression_guard'],
                  'automatic_adoption': False, 'physical_gap_eliminated': False,
                  'limitations': ['nominal current/torque scale is not calibrated shaft torque',
                                  'left/right loss and input timing uncertainty remain',
                                  'fixed fixture cannot distinguish armature from wheel-body inertia',
                                  'RL translation, human support and exact contact state are unmeasured',
                                  'above 37rad/s is extrapolation; existing RL recording is not independent acceptance']}
        save_json(args.out/'report.json', report)
        save_json(args.out/'run.json', {'complete': True, 'phase': 'finished'})
        print(json.dumps({'complete': True, 'out': str(args.out), 'rl_windows': rl_report['evaluated_windows']}), flush=True)
    except BaseException as exc:
        save_json(args.out/'run.json', {'complete': False, 'phase': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
