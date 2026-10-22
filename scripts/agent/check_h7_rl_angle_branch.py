#!/usr/bin/env python3
"""Compile the actual H7 angle mapper, observation builder and PD for offline checks.

No port is opened and no alternative dynamics/actuator response is claimed.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
H7 = ROOT / 'deployment/H7_RL_ros'


def initializer(text, name):
    start = text.index('{', text.index('.' + name + ' ='))
    depth = 0
    for i in range(start, len(text)):
        depth += text[i] == '{'
        depth -= text[i] == '}'
        if depth == 0:
            return text[start:i + 1]
    raise ValueError(name)


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    config_path = H7 / 'imcalib/config/machine_config.c'
    text = config_path.read_text().split('[MACHINE_ID_SMALL_WHEELLEG]')[0]
    rl = initializer(text, 'rl')
    def values(name):
        return np.array([float(x.strip().rstrip('f')) for x in
                         re.search(r'\.' + name + r'\s*=\s*\{([^}]+)\}', rl).group(1).split(',')], dtype=np.float32)
    sign, zero, reference, nominal = map(values, ('motor_sign', 'motor_zero', 'reference_pos', 'nominal_pos'))
    half = np.float32(re.search(r'\.dm_pos_max\s*=\s*([\d.]+)f', text).group(1))
    period = np.float32(2 * half)
    wrapper = args.out / 'wrapper.c'
    wrapper.write_text('''#include "rl_observation.h"
#include "rl_torque.h"
#include <string.h>
static machine_rl_cfg_t config = ''' + rl + ''';
static rl_motor_map_state_t state;
static rl_motor_state_t motor;
int step(int reset, int valid, const float *q, const float *v, const float *w, float *position, float *velocity, float *offset)
{
    if (reset) { state.initialized = 0u; }
    int ok = RL_Motor_Map(&config, ''' + str(float(period)) + '''f, q, v, w, valid, &state, &motor);
    memcpy(position, motor.pos, sizeof(motor.pos));
    memcpy(velocity, motor.vel, sizeof(motor.vel));
    memcpy(offset, state.position_offset, sizeof(state.position_offset));
    return ok;
}
int shared(const float *action, float *obs, float *pd_position, float *torque_out)
{
    rl_observation_state_t observation = {0};
    rl_observation_sample_t sample;
    rl_recovery_state_t recovery = {.phase=RL_RECOVERY_NORMAL};
    rc_command_t command = {.len=.27f};
    rl_torque_state_t pd = {0};
    torque_output_t torque;
    const float gyro[3] = {0}, quat[4] = {1,0,0,0};
    RL_Observation_Sample(&observation, gyro, quat, &motor, 1u);
    if (!RL_Observation_Read_Sample(&observation, &sample)) { return 0; }
    if (!RL_Observation_Build(&observation, &config, &sample, &command, &recovery, 1u)) { return 0; }
    if (!RL_Torque_Compute(&motor, &config, action, &pd, &torque)) { return 0; }
    memcpy(obs, observation.obs, sizeof(observation.obs));
    memcpy(pd_position, pd.pos, sizeof(pd.pos));
    memcpy(torque_out, torque.dm, sizeof(torque.dm));
    memcpy(torque_out+4, torque.dji, sizeof(torque.dji));
    return 1;
}
''')
    library = args.out / 'mapper.so'
    cc = os.environ.get('CC', 'cc')
    command = [cc, '-std=c11', '-O0', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
               '-I' + str(H7 / 'imcalib/Algorithm'), '-I' + str(H7 / 'imcalib/common'),
               '-I' + str(H7 / 'imcalib/config'), str(wrapper),
               str(H7 / 'imcalib/Algorithm/rl_observation.c'),
               str(H7 / 'imcalib/Algorithm/rl_torque.c'), '-lm', '-o', str(library)]
    compiled = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.out / 'compile.log').write_text(compiled.stdout)
    (args.out / 'compile-command.json').write_text(json.dumps(command, indent=2) + '\n')
    compiled.check_returncode()
    lib = ctypes.CDLL(str(library.resolve()))
    ptr = ctypes.POINTER(ctypes.c_float)
    lib.step.argtypes = [ctypes.c_int, ctypes.c_int] + [ptr] * 6
    lib.shared.argtypes = [ptr] * 4
    def pointer(a):
        return a.ctypes.data_as(ptr)
    def step(q, reset=False, valid=True, v=None, w=None):
        q = np.array(q, dtype=np.float32)
        v = np.zeros(4, dtype=np.float32) if v is None else np.array(v, dtype=np.float32)
        w = np.zeros(2, dtype=np.float32) if w is None else np.array(w, dtype=np.float32)
        pos, vel, offset = (np.empty(n, dtype=np.float32) for n in (4, 6, 4))
        ok = lib.step(reset, valid, *map(pointer, (q, v, w, pos, vel, offset)))
        return ok, pos, vel, offset
    def firmware(q):
        return np.asarray(q, dtype=np.float32) * sign[:4] + zero

    max_error, cases = 0., 0
    # A new RL selection chooses the same representation for prior ±1/±3 turns.
    for delta in (-3., -1., -.1, 0., .1, 1., 3.):
        expected = reference + np.float32(delta)
        for turns in (-3, -1, 0, 1, 3):
            ok, pos, vel, _ = step(firmware(expected) + np.float32(turns) * period,
                                    reset=True, v=[1, -2, 3, -4], w=[5, -6])
            assert ok
            error = float(np.max(np.abs(pos - expected)))
            max_error = max(max_error, error)
            assert error < 5e-6, (delta, turns, pos, expected)
            assert np.array_equal(vel, sign * np.array([1, -2, 3, -4, 5, -6], dtype=np.float32))
            cases += 1
    # Retain continuous motion past both the reference cut and encoder cut.
    for direction in (-1, 1):
        for j, delta in enumerate((0., 2.8, 3.13, 3.15, 3.4, 6.4)):
            expected = reference + np.float32(direction * delta)
            ok, pos, _, _ = step(firmware(expected) + 3 * period, reset=j == 0)
            assert ok and np.max(np.abs(pos - expected)) < 5e-6
            cases += 1
    # Loss of feedback must not silently re-anchor while staying in RL.
    _, _, _, before = step(firmware(reference) - 3 * period, reset=True)
    ok, _, _, after = step(firmware(reference), valid=False)
    assert not ok and np.array_equal(before, after)
    ok, pos, _, after = step(firmware(reference + 3.4) - 3 * period)
    assert ok and np.max(np.abs(pos - (reference + 3.4))) < 5e-6
    assert np.array_equal(before, after)
    # Invalid first feedback does not latch NaN offsets; next good sample works.
    for invalid in (np.nan, np.inf, -np.inf):
        q = firmware(reference); q[0] = invalid
        ok, _, _, _ = step(q, reset=True)
        assert not ok
        ok, pos, _, _ = step(firmware(reference))
        assert ok and np.max(np.abs(pos - reference)) < 1e-6
        cases += 1
    for polarity in (-1, 1):
        # Feed a continuous driver crossing its wrapped encoder endpoint.
        base = zero + polarity * half
        _, p0, _, _ = step(base - .001, reset=True)
        _, p1, _, _ = step(base + .001)
        assert np.max(np.abs((p1 - p0) - sign[:4] * .002)) < 1e-6
        cases += 1
    # Check actual obs and PD both consume the newly mapped position.
    expected = nominal + .1
    ok, pos, _, _ = step(firmware(expected) + 3 * period, reset=True)
    assert ok
    action, obs, pd_pos, torque = (np.zeros(n, dtype=np.float32) for n in (6, 39, 4, 6))
    assert lib.shared(*map(pointer, (action, obs, pd_pos, torque)))
    assert np.array_equal(pos, pd_pos)
    assert np.max(np.abs(obs[10:14] - (pos - nominal))) < 1e-6
    assert np.max(np.abs(torque[:4] - sign[:4] * 60 * (nominal - pos))) < 1e-6
    assert np.array_equal(torque[4:], np.zeros(2, dtype=np.float32))

    sys.path.insert(0, str(ROOT / 'sim2sim'))
    import real2sim_huananhu as mirror
    meta, _, _ = mirror.load_session(args.session)
    parser, clock = mirror.FrameParser(), mirror.FrameClock()
    recorded, positions = [], []
    reset, lost = True, 0
    for event in mirror.iter_session_events(args.session):
        if event['t_ns'] > 30e9:
            break
        if event['type'] != 'data':
            continue
        for f in parser.feed(bytes.fromhex(event['data_hex'])):
            if not clock.accept(f):
                continue
            if f[152] != 0:
                reset = True
                continue
            source_ok = not (int(f[2]) & 4) and (int(f[29]) & 30) == 30
            ok, pos, _, offset = step(f[10:14], reset=reset, valid=source_ok,
                                      v=f[14:18], w=f[20:22])
            reset = False
            if ok:
                positions.append(pos)
                recorded.append([event['t_ns'] / 1e9, *f[10:14], *pos, *offset])
            else:
                lost += 1
    positions = np.asarray(positions)
    assert len(positions) and np.max(np.abs(positions)) < 3.5
    np.savetxt(args.out / 'recorded-mapping.csv', np.asarray(recorded), delimiter=',',
               header='receive_s,fw_lf0,fw_lf00,fw_rf0,fw_rf00,model_lf0,model_lf00,model_rf0,model_rf00,offset0,offset1,offset2,offset3')
    paths = [H7 / p for p in ('imcalib/Algorithm/rl_observation.c', 'imcalib/Algorithm/rl_observation.h',
                             'imcalib/Algorithm/rl_torque.c', 'imcalib/task/task_actuation.c',
                             'imcalib/config/machine_config.c')]
    report = dict(ok=True, compiler=cc, functional_cases=cases, max_period_invariance_error_rad=max_error,
                  source_stream_sha256=meta['stream_sha256'], mapped_RL_frames=len(positions),
                  invalid_source_frames=lost, first_corrected_position_rad=positions[0].tolist(),
                  corrected_min_rad=positions.min(axis=0).tolist(), corrected_max_rad=positions.max(axis=0).tolist(),
                  shared_observation_PD_position_exact=True, preserved_velocity_signs=True,
                  continuous_past_reference_and_encoder_cuts=True, offset_preserved_through_invalid_source=True,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  scope='Actual C mapping/obs/PD arithmetic and recorded-angle conversion only; no alternative policy inputs, physics, motor commands or measured behavior improvement.')
    (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
