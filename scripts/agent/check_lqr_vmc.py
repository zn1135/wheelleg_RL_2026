#!/usr/bin/env python3
"""Formal H7 numerical checks; run with mjlab_training/.venv/bin/python.

Compiles unchanged H7 C with host libm replacing only CMSIS sin/cos/sqrt.
This checks equations and state transitions, not MCU timing or robot behavior.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from ct_mjlab.teacher.gains import gain_matrix
from ct_mjlab.teacher.mechanics import leg_geometry, force_map, spring_force, gravity_torque, wrap
from ct_mjlab.teacher.controller import LqrVmcController
from ct_mjlab.teacher.estimators import VelocityFilter, AirDetector
from ct_mjlab.teacher.standup import Standup, DONE, FAILED, REAR, EXTEND, RETRACT, SWING

ROOT = Path(__file__).resolve().parents[2]
H7 = ROOT / 'deployment/H7_RL_ros/imcalib'
PTR = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')


def compile_reference(directory, compiler):
    compiler = compiler or os.environ.get('CC') or shutil.which('cc')
    if not compiler:
        fallback = Path.home() / '.local/opt/h7-host-gcc/usr/bin/x86_64-linux-gnu-gcc-13'
        if fallback.is_file():
            compiler = str(fallback)
    if not compiler:
        raise RuntimeError('需要本机C编译器；用--cc指定，不会修改固件')
    (directory / 'host_math.h').write_text('''#include <math.h>
#define ARM_MATH_SUCCESS 0
#define arm_cos_f32 cosf
#define arm_sin_f32 sinf
static inline int arm_sqrt_f32(float x, float *out) { *out=sqrtf(x); return x<0?-1:0; }
''')
    paths = []
    sources = ['Algorithm/' + name + '.c' for name in (
        'leg_solver', 'gas_spring', 'gravity_comp', 'leg_compensation', 'lqr_gain_big',
        'lqr_gain_table', 'lqr_balance', 'leg_balance', 'standup', 'slip', 'air_detection')]
    sources += ['config/machine_config.c', 'common/pid.c', 'common/simple-function.c']
    for name in sources:
        original = H7 / name
        if name in ('Algorithm/leg_solver.c', 'Algorithm/gas_spring.c'):
            dest = directory / original.name
            dest.write_text(original.read_text().replace(
                '"../../Drivers/CMSIS/DSP/Include/arm_math.h"', '"host_math.h"'))
            paths.append(str(dest))
        else:
            paths.append(str(original))
    library = directory / 'reference.so'
    command = [compiler, '-std=c11', '-shared', '-fPIC', '-O0', '-ffp-contract=off',
               '-B' + str(Path(compiler).parent), '-o', str(library)]
    # GCC的collect2查找无前缀ld；仅在临时目录提供现有工具的别名。
    for name in ('ld', 'as'):
        prefixed = Path(compiler).parent / ('x86_64-linux-gnu-' + name)
        if prefixed.is_file():
            (directory / name).symlink_to(prefixed)
    command += ['-B' + str(directory) + '/']
    # 本机拆包安装的工具链：前端在系统目录，链接运行库在私有目录。
    private_runtime = Path(compiler).parent.parent / 'lib/gcc/x86_64-linux-gnu/13'
    if private_runtime.is_dir() and Path('/usr/libexec/gcc/x86_64-linux-gnu/13/cc1').is_file():
        command += ['-B/usr/libexec/gcc/x86_64-linux-gnu/13/', '-B' + str(private_runtime) + '/',
                    '-isystem', str(private_runtime / 'include')]
    command += ['-I' + str(H7 / folder) for folder in ('Algorithm', 'config', 'common')]
    command += ['-I' + str(directory), str(Path(__file__).with_name('lqr_vmc_reference.c')), *paths, '-lm']
    compiler_env = os.environ.copy()
    private_libs = Path(compiler).parent.parent / 'lib/x86_64-linux-gnu'
    if private_libs.is_dir():
        compiler_env['LD_LIBRARY_PATH'] = str(private_libs) + ':' + compiler_env.get('LD_LIBRARY_PATH', '')
    subprocess.run(command, check=True, capture_output=True, text=True, env=compiler_env)
    lib = ctypes.CDLL(str(library))
    lib.ref_geometry.argtypes = [PTR, ctypes.c_float, ctypes.c_float, PTR]
    lib.ref_models.argtypes = [ctypes.c_float, ctypes.c_float, PTR]
    lib.ref_control.argtypes = [PTR, PTR, PTR, ctypes.c_int, PTR]
    lib.ref_standup.argtypes = [PTR, ctypes.c_float, PTR]
    lib.ref_balance.argtypes = [PTR, PTR, ctypes.c_float, ctypes.c_float, ctypes.c_int, PTR]
    lib.ref_velocity.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_int, PTR]
    lib.ref_air.argtypes = [PTR, PTR, ctypes.c_float, ctypes.c_float, ctypes.c_float, PTR]
    return lib, {name: hashlib.sha256((H7 / name).read_bytes()).hexdigest() for name in sources}


def hip_angles(length, theta):
    alpha = math.acos((length*length + .21**2 - .25**2)/(2*length*.21))
    return [math.pi/2-theta+alpha, math.pi/2-theta-alpha]


def numerical(lib):
    torch.manual_seed(42)
    q = torch.rand(256, 2) * .7 + torch.tensor([2.3, .2])
    dq = torch.randn_like(q) * 2
    length, theta, rates, jac, valid = leg_geometry(q, dq)
    force, torque = torch.randn(256)*100, torch.randn(256)*10
    mapped = force_map(jac, force, torque)
    actual = torch.cat((length[:, None], theta[:, None], rates, jac.flatten(1), mapped, valid[:, None]), -1)
    expected = np.empty((256, 11), dtype=np.float32)
    models = np.empty((256, 2), dtype=np.float32)
    for i in range(256):
        lib.ref_geometry(torch.cat((q[i], dq[i])).numpy(), force[i].item(), torque[i].item(), expected[i])
        lib.ref_models(length[i].item(), theta[i].item(), models[i])
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-4, rtol=2e-5)
    np.testing.assert_allclose(torch.stack((spring_force(length), gravity_torque(length, theta)), -1),
                               models, atol=1e-4, rtol=1e-5)
    # Independent virtual-work / finite-difference check, not another copy of C.
    h = 1e-6
    q64 = q.double()
    fd = []
    for col in range(2):
        delta = torch.zeros_like(q64)
        delta[:, col] = h
        lp, tp, *_ = leg_geometry(q64+delta, dq.double())
        lm, tm, *_ = leg_geometry(q64-delta, dq.double())
        fd.append(torch.stack(((lp-lm)/(2*h), wrap(tp-tm)/(2*h)), -1))
    np.testing.assert_allclose(jac, torch.stack(fd, -1), atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose((mapped*dq).sum(-1), force*rates[:, 0]+torque*rates[:, 1], atol=2e-5, rtol=2e-5)
    max_control = 0.
    for air in (0, 1):
        for i in range(128):
            lengths = (torch.rand(2)*.3+.08).numpy()
            x, target = torch.randn(10), torch.randn(10)
            expected_control = np.empty(44, dtype=np.float32)
            lib.ref_control(lengths, x.numpy(), target.numpy(), air, expected_control)
            k = gain_matrix(torch.from_numpy(lengths).clamp(.15, .31)[None])[0]
            error = target-x
            error[2] = wrap(error[2])
            if air:
                error[[4, 6]] = wrap(error[[4, 6]])
            contribution = k*error
            if air:
                contribution[:2] = 0
                contribution[2:, :4] = 0
                contribution[2:, 8:] = 0
            output = contribution.sum(-1)
            output[:2].clamp_(-4.84375, 4.84375)
            output[2:].clamp_(-10 if air else -54, 10 if air else 54)
            np.testing.assert_allclose(k.flatten(), expected_control[4:], atol=3e-5, rtol=2e-5)
            np.testing.assert_allclose(output, expected_control[:4], atol=3e-5, rtol=2e-5)
            max_control = max(max_control, float(np.max(np.abs(output.numpy()-expected_control[:4]))))
    return dict(samples=256, geometry_vmc_max_abs=float(np.max(np.abs(actual.numpy()-expected))),
                compensation_max_abs=float(np.max(np.abs(models-torch.stack((spring_force(length), gravity_torque(length, theta)), -1).numpy()))),
                lqr_max_abs=max_control, finite_difference_and_virtual_work=True)


def standup_checks(lib):
    lib.ref_reset()
    st = Standup(1, 'cpu')
    max_error = 0.
    phases = set()
    length, angle = .18, -.8
    for tick in range(2500):
        q = torch.tensor([hip_angles(length, angle)], dtype=torch.float32).repeat(2, 1)
        l, t, rates, _, valid = leg_geometry(q, torch.zeros_like(q))
        raw = torch.cat((q, torch.zeros_like(q)), -1).flatten().numpy()
        expected = np.empty(14, dtype=np.float32)
        lib.ref_standup(raw, 0., expected)
        f, tp, preparing, _ = st.step(l[None], t[None], rates[:, 1][None],
                                     torch.zeros(1), torch.ones(1), valid.all()[None])
        if int(st.phase[0]) != int(expected[0]):
            raise AssertionError(f'standup tick={tick}: torch phase={st.phase.tolist()} C={expected.tolist()}')
        phases.add(int(st.phase[0]))
        if preparing:
            actual = torch.cat((f[0], tp[0], st.length_cmd[0], st.angle_cmd[0])).numpy()
            np.testing.assert_allclose(actual, expected[2:10], atol=.01, rtol=2e-4,
                                       err_msg=f'standup tick={tick}, phase={st.phase.item()}')
            max_error = max(max_error, float(np.max(np.abs(actual-expected[2:10]))))
        if st.phase == DONE:
            break
        # Scripted measured state follows each stage; this is not a physics result.
        if st.phase == EXTEND:
            length = min(.25, length+.0003)
        elif st.phase == REAR:
            angle = float(st.angle_cmd[0, 0])
        elif st.phase == RETRACT:
            length = max(.15, length-.0005)
        elif st.phase == SWING:
            angle += max(-.006, min(.006, -angle))
    assert st.phase == DONE, 'synthetic normal standup did not hand over'
    assert phases == {EXTEND, REAR, RETRACT, SWING, DONE}, phases
    completed_tick = tick+1
    lib.ref_reset()
    st.reset(slice(None))
    q = torch.tensor([hip_angles(.25, -.8)], dtype=torch.float32).repeat(2,1)
    l, t, rates, _, valid = leg_geometry(q, torch.zeros_like(q))
    raw = torch.cat((q, torch.zeros_like(q)), -1).flatten().numpy()
    for tick in range(4500):
        expected = np.empty(14, dtype=np.float32)
        lib.ref_standup(raw, 0., expected)
        st.step(l[None], t[None], rates[:,1][None], torch.zeros(1), torch.ones(1), valid.all()[None])
        assert int(st.phase[0]) == int(expected[0]), f'timeout/retry phase tick={tick}'
    assert st.phase == FAILED and st.retry == 3 and st.fault == 3
    return dict(phases=sorted(phases), ticks=tick+1, max_abs=max_error,
                normal_handover_tick=completed_tick, timeout_and_three_retries=True,
                scope='scripted sensor trajectory against actual H7 C; not physical standup')


def batch_checks(*, velocity_integral=False):
    n = 20
    batched = LqrVmcController(n, 'cpu',velocity_integral=velocity_integral)
    single = [LqrVmcController(1, 'cpu',velocity_integral=velocity_integral) for _ in range(n)]
    batched.reset(slice(None), standing=True)
    for ctl in single:
        ctl.reset(slice(None), standing=True)
    max_error = 0.
    for tick in range(50):
        q = torch.tensor(hip_angles(.22, .02))[None, None].repeat(n, 2, 1)
        q += torch.arange(n)[:, None, None] * .002
        z = torch.zeros(n)
        command = torch.stack((z, z, torch.full_like(z, .15)), -1)
        upright = torch.ones(n)
        if tick >= 10:
            upright[3] = -1
        args = (q, torch.zeros_like(q), torch.zeros(n, 2), z, z, z, z, z, upright,
                z, z, torch.zeros_like(q), command)
        if tick == 25:
            batched.reset(torch.tensor([2, 7]), standing=True)
            for i in (2, 7):
                single[i].reset(slice(None), standing=True)
        result = batched.step(*args)
        reference = torch.cat([ctl.step(*(a[i:i+1] for a in args)) for i, ctl in enumerate(single)])
        torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)
        max_error = max(max_error, float((result-reference).abs().max()))
        if tick >= 10:
            assert batched.standup.phase[3] == FAILED and not result[3].any()
    return dict(environments=n, ticks=50, partial_reset=[2, 7], max_abs=max_error, inverted_zero_output=True)


def integral_checks():
    ctl = LqrVmcController(3,'cpu',velocity_integral=True)
    ctl.reset(slice(None),standing=True)
    q = torch.tensor(hip_angles(.30,.2))[None,None].repeat(3,2,1)
    zero = torch.zeros(3)
    command = torch.tensor([[.5,0.,.30],[-.5,0.,.30],[-.5,0.,.30]])
    args = (q,torch.zeros_like(q),torch.zeros(3,2),zero,zero,zero,zero,zero,
            torch.ones(3),zero,zero,torch.zeros_like(q),command)
    ctl.pos[2] = 3.4999
    for _ in range(20):
        ctl.step(*args)
    torch.testing.assert_close(ctl.pos,torch.tensor([-.01,.01,3.5]),atol=1e-7,rtol=1e-6)
    other = ctl.pos[1:].clone()
    ctl.reset(torch.tensor([0]),standing=True)
    assert ctl.velocity_integral and not ctl.pos[0]
    torch.testing.assert_close(ctl.pos[1:],other,atol=0,rtol=0)
    ctl.air.flight[1] = True
    ctl.standup.phase[0] = EXTEND
    before = ctl.pos.clone()
    ctl.step(*args)
    torch.testing.assert_close(ctl.pos[:2],before[:2],atol=0,rtol=0)
    return dict(velocity_error_integral_sign_and_limit=True,partial_reset=True,
                frozen_in_flight_and_preparation=True,scope='simulation extension; not H7 C equivalence')


def estimator_checks(lib):
    lib.ref_reset()
    velocity = VelocityFilter(1, 'cpu')
    air = AirDetector(1, 'cpu')
    maximum_v, maximum_f = 0., 0.
    flights = set()
    for tick in range(500):
        w = torch.tensor([math.sin(tick*.03) + (2 if tick == 180 else 0)])
        a = torch.tensor([math.cos(tick*.03)])
        contact = torch.tensor([tick < 200 or tick > 300])
        velocity.step(w, a, contact)
        expected_v = np.empty(6, dtype=np.float32)
        lib.ref_velocity(w.item(), a.item(), int(contact.item()), expected_v)
        actual_v = torch.cat((velocity.state.flatten(), velocity.cov.flatten())).numpy()
        np.testing.assert_allclose(actual_v, expected_v, atol=1e-6, rtol=1e-5)
        maximum_v = max(maximum_v, float(np.max(np.abs(actual_v-expected_v))))
        length = .23 if tick < 350 else .19
        q = torch.tensor(hip_angles(length, .03))[None, None].repeat(1, 2, 1)
        l, t, rates, jac, _ = leg_geometry(q, torch.zeros_like(q))
        spring, gravity = spring_force(l), gravity_torque(l, t)
        # Sustained support → release → compression landing.
        support = 100 if tick < 120 or tick > 400 else 0
        motor = force_map(jac, torch.full_like(l, support)-spring, gravity)
        air.step(l, t, rates, torch.zeros(1), torch.zeros(1), jac, motor, spring, gravity,
                 torch.zeros(1), torch.ones(1, dtype=torch.bool))
        raw = torch.cat((q[0], torch.zeros_like(q[0])), -1).flatten().numpy()
        expected_f = np.empty(4, dtype=np.float32)
        lib.ref_air(raw, motor.flatten().numpy(), 0., 0., 0., expected_f)
        actual_f = torch.cat((air.force.flatten(), air.armed, air.flight)).numpy()
        np.testing.assert_allclose(actual_f, expected_f, atol=1e-4, rtol=1e-5, err_msg=f'air tick={tick}')
        maximum_f = max(maximum_f, float(np.max(np.abs(actual_f-expected_f))))
        flights.add(bool(air.flight[0]))
    assert flights == {False, True}
    return dict(ticks=500, velocity_max_abs=maximum_v, support_max_abs=maximum_f, air_and_landing=True)


def balance_checks(lib):
    lib.ref_reset()
    ctl = LqrVmcController(1, 'cpu')
    ctl.reset(slice(None), standing=True)
    maximum = 0.
    for tick in range(200):
        q = torch.tensor(hip_angles(.2+.01*math.sin(tick*.03), .01))[None, None].repeat(1,2,1)
        z = torch.zeros(1)
        roll = torch.tensor([.03*math.cos(tick*.03)])
        command = torch.tensor([[0.,0.,.15]])
        actual = ctl.step(q, torch.zeros_like(q), torch.zeros(1,2), z, roll, z, z, z,
                          torch.ones(1), z, z, torch.zeros_like(q), command)
        expected = np.empty(8, dtype=np.float32)
        raw = torch.cat((q[0], torch.zeros_like(q[0])), -1).flatten().numpy()
        lib.ref_balance(raw, ctl.u[0].numpy(), float(roll[0]), .15, 0, expected)
        values = torch.cat((ctl.force[0], actual[0])).numpy()
        np.testing.assert_allclose(values, expected, atol=.003, rtol=2e-5, err_msg=f'balance tick={tick}')
        maximum = max(maximum, float(np.max(np.abs(values-expected))))
    return dict(ticks=200, pd_vmc_compensation_limits_max_abs=maximum)


def mapping_checks():
    import mujoco
    from ct_mjlab.envs.chuanliantui.model import load_model, solve_standing_joint_positions, wheel_response_parameters
    from ct_mjlab.teacher.env import body_angles
    model = load_model()
    data = mujoco.MjData(model)
    differences = []
    for shift in (-.1, 0., .1):
        positions = solve_standing_joint_positions(model, front_joint_targets={
            'left': (.06+shift, -.1), 'right': (-.06-shift, .1)})
        for name, value in positions.items():
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        for front, back, wheel, sign in [('lf0','lf00','lfwheel',1), ('rf0','rf00','rfwheel',-1)]:
            q = torch.tensor([[positions[front]*sign+math.pi, positions[back]*sign]], dtype=torch.float64)
            length, theta, *_ = leg_geometry(q, torch.zeros_like(q))
            vector = data.xpos[model.body(wheel).id]-data.xpos[model.body(front).id]
            xml_length = float(np.linalg.norm(vector[[0,2]]))
            xml_theta = float(np.arctan2(vector[0], -vector[2]))
            differences.append([float(length)-xml_length, float(theta)-xml_theta])
        # Joint +Y/-Y rotation and torque use the same sign, preserving power.
        assert np.allclose(data.xaxis[model.joint('lfwheel').id], [0, 1, 0])
        assert np.allclose(data.xaxis[model.joint('rfwheel').id], [0, -1, 0])
    for p in (-.2, .2):
        pitch, _, _, upright = body_angles(torch.tensor([[math.cos(p/2), 0, math.sin(p/2), 0]]))
        assert abs(float(pitch)-p) < 1e-6 and upright > 0
    np.testing.assert_allclose(differences, 0, atol=1e-8)
    # The serial CAD linkage reproduces H7 virtual-leg geometry at these closed poses.
    return dict(mass_kg=float(model.body_mass.sum()), wheel_response=wheel_response_parameters(model).tolist(),
                h7_minus_xml_length_m=[row[0] for row in differences],
                h7_minus_xml_angle_rad=[row[1] for row in differences],
                wheel_axes_and_pitch=True)


def command_checks():
    from ct_mjlab.scripts.teacher import command_schedule
    preset = ROOT / 'mjlab_training/configs/teacher_motion.json'
    args = SimpleNamespace(vx=0., yaw=0., leg_length=.15, sequence=preset)
    rows = command_schedule(args)
    assert len(rows) == 7 and rows[-1]['time'] == 24
    for bad in (float('nan'), 2.):
        args.sequence, args.vx = None, bad
        try:
            command_schedule(args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid command accepted')
    return dict(preset_rows=7, invalid_commands_rejected=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(passed=False)
    try:
        with tempfile.TemporaryDirectory(prefix='h7-teacher-check-') as tmp:
            lib, hashes = compile_reference(Path(tmp), args.cc)
            report['source_sha256'] = hashes
            report['numerical'] = numerical(lib)
            report['normal_standup'] = standup_checks(lib)
            report['estimators'] = estimator_checks(lib)
            report['balance_output'] = balance_checks(lib)
            report['batch_reset'] = batch_checks()
            report['velocity_integral'] = integral_checks()
            report['integral_batch_reset'] = batch_checks(velocity_integral=True)
            report['mapping'] = mapping_checks()
            report['commands'] = command_checks()
        report['passed'] = True
    except subprocess.CalledProcessError as exc:
        report['error'] = exc.stderr
        raise
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
