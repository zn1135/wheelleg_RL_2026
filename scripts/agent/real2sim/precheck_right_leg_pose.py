"""Exercise actual C right-leg controller, then integrate it in fixed-base Warp GPU."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import mujoco
import numpy as np

from scripts.agent.real2sim.control_leg_pose_gpu import mirror, gas_spring_force, leg_pose, inverse_pose, initial_preload
from scripts.agent.real2sim.check_h7_output_capture import extract


class State(ctypes.Structure):
    _fields_ = [('center', ctypes.c_float * 4), ('target', ctypes.c_float * 4), ('integral', ctypes.c_float * 4), ('takeover', ctypes.c_float * 4),
                ('start_ms', ctypes.c_uint32), ('elapsed_ms', ctypes.c_uint32),
                ('running', ctypes.c_uint8), ('blocked', ctypes.c_uint8), ('stage', ctypes.c_uint8),
                ('axis', ctypes.c_uint8), ('saturated', ctypes.c_uint16)]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--observation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepared-length-mm', type=float)
    parser.add_argument('--prepared-hold', action='store_true', help='Synthetic static torque handover from the existing length-preparation mode')
    parser.add_argument('--profile', choices=('pose','length-only'), default='pose')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    h7 = root / 'deployment/H7_RL_ros'
    algorithm = h7 / 'imcalib/Algorithm'
    reference = root / 'artifacts/pace-joint-capture/20261108-lowfreq-load-fit-01'
    params = json.loads((reference / 'parameters.json').read_text())
    observed = json.loads(args.observation.read_text())
    fw = np.array(observed['motor_positions_rad'])
    expected_angle=float(leg_pose(fw[None])[1][0,1]) if args.profile=='length-only' else np.pi/6
    if args.prepared_length_mm is not None:
        start_angle = leg_pose(fw[None])[1][0, 1]
        fw[2:] = inverse_pose(args.prepared_length_mm / 1000, start_angle)
    pose = mirror.MirrorPose(machine_config=h7 / 'imcalib/config/machine_config.c')
    model = mujoco.MjModel.from_xml_path(str(reference / 'gpu-replay/model.xml'))
    model.geom_contype[:] = model.geom_conaffinity[:] = 0
    mujoco.mj_saveLastXML(str(args.output / 'model.xml'), model)
    data = mujoco.MjData(model)
    mq = np.array([model.joint(n).qposadr[0] for n in mirror.MOTORS[:4]])
    mv = np.array([model.joint(n).dofadr[0] for n in mirror.MOTORS[:4]])
    aq = [pose.model.joint(model.joint(j).name).qposadr[0] for j in range(model.njnt)]
    for side in (0, 1):
        assert pose._solve_leg(side, (fw - pose.zero)[2 * side:2 * side + 2] * pose.sign[2 * side:2 * side + 2])
    initial = pose.data.qpos[aq].copy()
    motor = np.array([model.actuator(n + '_motor').id for n in mirror.MOTORS])
    spring = [model.actuator(s + '_gas_spring_motor').id for s in ('left', 'right')]
    sq = [model.joint(s + '_spring2_joint').qposadr[0] for s in ('left', 'right')]
    sv = [model.joint(s + '_spring2_joint').dofadr[0] for s in ('left', 'right')]
    sources = [algorithm / name for name in ('joint_capture.c', 'joint_chirp.c')]
    ffour = ctypes.c_float * 4
    signs = (ctypes.c_int8 * 6)(*pose.sign.astype(int))
    zero = ffour(*pose.zero)
    kp, kd = ffour(60, 60, 60, 60), ffour(2, 2, 2, 2)
    with tempfile.TemporaryDirectory() as temp:
        library = Path(temp) / 'controller.so'
        compiler = os.environ.get('CC', 'cc')
        subprocess.run([compiler, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC', *map(str, sources), '-lm', '-o', str(library)], check=True)
        lib = ctypes.CDLL(str(library))
        update = lib.Joint_Capture_Update
        update.argtypes = [ctypes.POINTER(State), ctypes.c_uint32, ctypes.c_uint8, *([ctypes.POINTER(ctypes.c_float)] * 3), ctypes.POINTER(ctypes.c_int8), *([ctypes.POINTER(ctypes.c_float)] * 3), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
        update.restype = ctypes.c_uint8
        guard = lib.Joint_Capture_Geometry
        guard.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float]
        guard.restype = ctypes.c_uint8
        target_function = lib.Joint_Pose_Target
        target_function.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int8), *([ctypes.POINTER(ctypes.c_float)] * 1), ctypes.c_float, ctypes.c_float, *([ctypes.POINTER(ctypes.c_float)] * 3), ctypes.POINTER(ctypes.c_uint8)]
        target_function.restype = ctypes.c_uint8
        def call(state, ms, q, v, previous, valid=1):
            output = ffour()
            ok = update(ctypes.byref(state), ms, valid, ffour(*q), ffour(*v), ffour(*previous), signs, zero, kp, kd, .21, .25, 40., output)
            return ok, np.array(output[:])
        state = State()
        ok, torque = call(state, 0, fw, np.zeros(4), np.zeros(4))
        assert ok and max(abs(torque)) < 1e-4, dict(ok=ok, integral=list(state.integral), blocked=state.blocked, torque=torque.tolist())
        assert not call(state, 2, fw, np.zeros(4), torque, valid=0)[0]
        assert not call(state, 4, fw, np.zeros(4), torque)[0]  # 锁止后不会自行恢复。
        state = State()
        assert not call(state, 0, fw + np.array([0, 0, np.nan, 0]), np.zeros(4), np.zeros(4))[0]
        state = State()
        assert not call(state, 0, np.array(observed['motor_positions_rad']), np.zeros(4), np.zeros(4))[0]  # 原337mm起点明确拒绝。
        center = (fw - pose.zero) * pose.sign[:4]
        target, velocity, feedforward, stage = ffour(), ffour(), ffour(), ctypes.c_uint8()
        path, speed = [], []
        for ms in range(0, 25001):
            assert target_function(ms, ffour(*center), signs, zero, .21, .25, target, velocity, feedforward, ctypes.byref(stage))
            target_fw = np.array(target[:]) * pose.sign[:4] + pose.zero
            assert guard(ffour(*target_fw), .21, .25)
            assert target[:2] == ffour(*center)[:2] and velocity[:2] == [0, 0] and feedforward[:2] == [0, 0]
            path.append(target[:]); speed.append(velocity[:])
        path, speed = np.array(path), np.array(speed)
        final_l, final_a, _ = leg_pose(path[-1:] * pose.sign[:4] + pose.zero)
        assert abs(final_l[0, 1] - .19) < 1e-6 and abs(final_a[0, 1] - expected_angle) < 1e-6
        if args.profile=='length-only':
            path_length,path_angle,_=leg_pose(path*pose.sign[:4]+pose.zero)
            assert np.max(abs(path_angle[:,1]-expected_angle))<1e-6
        assert np.max(abs(np.gradient(path[1:-1], .001, axis=0) - speed[1:-1])) < .001
        # 使用实际任务包装，核对失能等待/2ms保持/5ms过期锁止及零轮请求。
        wrapper = extract((h7 / 'imcalib/task/task_actuation.c').read_text(), 'solve_joint_capture')
        harness = r'''
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "joint_capture.h"
#define FAULT_NONE 0
#define MACHINE_RL_CTRL_DIV 2
typedef struct {float dm[4],dji[2];uint8_t valid;} torque_output_t;
static joint_capture_t joint_capture;static uint8_t joint_capture_divider;static torque_output_t joint_capture_torque;
static struct {uint8_t drive,joint_ready,motor_valid;struct {uint8_t online;} rc;float motor_pos[4],motor_vel[4];} control_frame;
static struct {float leg_lu,leg_lg,dm_trq_clamp;struct {int8_t motor_sign[6];float motor_zero[4];struct {float p_gains[4],d_gains[4];} torque;} rl;} config;
static __typeof__(config) *machine=&config;
static uint64_t now,latency_source_rx_ns[8];static float rl_output_dm_cmd_nm[4];static uint32_t ctrl_fault;
static uint64_t Mono_Ns_Get(void){return now;}
static void Torque_Output_Clear(torque_output_t *t){memset(t,0,sizeof(*t));}
@WRAPPER@
static void fresh(void){for(int i=0;i<4;i++)latency_source_rx_ns[i+1]=now-1000000;}
int main(void){
 torque_output_t t={0};now=1000000000;config.leg_lu=.21f;config.leg_lg=.25f;config.dm_trq_clamp=54.f;
 control_frame.rc.online=control_frame.motor_valid=control_frame.joint_ready=1;
 for(int i=0;i<4;i++){config.rl.motor_sign[i]=i<2?1:-1;config.rl.torque.p_gains[i]=60;config.rl.torque.d_gains[i]=2;control_frame.motor_pos[i]=i%2?.48435023f:-.48435023f;}
 fresh();solve_joint_capture(&t);assert(!t.valid&&!joint_capture.blocked);
 control_frame.drive=1;solve_joint_capture(&t);assert(t.valid&&joint_capture.running&&t.dm[0]==0&&t.dm[1]==0&&t.dji[0]==0&&t.dji[1]==0);
 now+=1000000;fresh();solve_joint_capture(&t);assert(t.valid&&joint_capture_divider==1);
 now+=1000000;fresh();solve_joint_capture(&t);assert(t.valid&&joint_capture.elapsed_ms==2);
 now+=6000000;solve_joint_capture(&t);assert(!t.valid&&joint_capture.blocked);
 fresh();solve_joint_capture(&t);assert(!t.valid);
 return 0;
}
'''.replace('@WRAPPER@', wrapper)
        harness_path = Path(temp) / 'wrapper.c'
        harness_path.write_text(harness)
        executable = Path(temp) / 'wrapper'
        subprocess.run([compiler, '-std=gnu11', '-Wall', '-Werror', '-I', str(algorithm), str(harness_path), *map(str, sources), '-lm', '-o', str(executable)], check=True)
        subprocess.run([str(executable)], check=True)
        import warp as wp
        import mujoco_warp as mw
        from scripts.agent.real2sim import ordered_leg_constraints as ordered
        wp.init()
        count, steps, dt = 3, 18500, .002
        states = [State() for _ in range(count)]
        previous = np.zeros((count, 4))
        factors = np.array([.7, 1., 1.3])
        qref = np.array(params['reference_q_model_rad'])
        stiffness = np.array(params['stiffness_nm_rad'])
        if args.prepared_hold:
            data.qpos[:] = initial
            data.qvel[:] = 0
            residual, _ = initial_preload(model, data, np.zeros(4), initial_spring_force=gas_spring_force(data.qpos[sq], np.zeros(2)))
            local_load = np.asarray(params['bias_nm']) + (initial[mq] - qref) @ stiffness.T
            # 准备档在各假设负载下已静置；右腿继承其保持矩，左腿在定位档归零。
            previous = (-residual[None] + local_load[None] * factors[:, None]) * pose.sign[:4]
        initial_previous = previous.copy()
        saved_q, saved_v, saved_torque, saved_target, stopped = [], [], [], [], [None] * count
        max_gap = 0.
        with wp.ScopedDevice('cuda:0'):
            gm = mw.put_model(model)
            gd = mw.put_data(model, data, nworld=count, nconmax=128, njmax=256)
            gd.qpos.assign(np.tile(initial, (count, 1)).astype(np.float32)); gd.qvel.zero_()
            buffers = ordered.buffers(gm, gd)
            with wp.ScopedCapture() as capture:
                for _ in range(10):
                    mw.step1(gm, gd); ordered.apply(gd, buffers); mw.step2(gm, gd)
            assert not np.any(gd.time.numpy())
            for step in range(steps + 1):
                q, v = gd.qpos.numpy(), gd.qvel.numpy()
                torque = np.zeros((count, 4))
                for trial in range(count):
                    fw_q = q[trial, mq] * pose.sign[:4] + pose.zero
                    valid = int(guard(ffour(*fw_q), .21, .25))
                    ok, firmware_torque = call(states[trial], step * 2, fw_q, v[trial, mv] * pose.sign[:4], previous[trial], valid)
                    if not ok and stopped[trial] is None: stopped[trial] = step * dt
                    torque[trial] = firmware_torque * pose.sign[:4]
                    previous[trial] = firmware_torque
                assert not np.any(torque[:, :2])
                saved_q.append(q.copy()); saved_v.append(v.copy()); saved_torque.append(torque.copy())
                saved_target.append([list(state.target) for state in states])
                if step % 50 == 0:
                    for row in q:
                        data.qpos[:] = row; mujoco.mj_kinematics(model, data)
                        max_gap = max(max_gap, max(np.linalg.norm(data.site(a).xpos - data.site(b).xpos) for group in mirror.PIN_PAIRS for a, b in group))
                if step == steps: break
                ctrl = np.zeros((count, model.nu), dtype=np.float32)
                ctrl[:, motor[:4]] = torque
                ctrl[:, spring] = gas_spring_force(q[:, sq], v[:, sv])
                load = np.zeros((count, model.nv), dtype=np.float32)
                load[:, mv] = -(params['bias_nm'] + (q[:, mq] - qref) @ stiffness.T) * factors[:, None]
                gd.ctrl.assign(ctrl); gd.qfrc_applied.assign(load)
                wp.capture_launch(capture.graph)
        saved_q, saved_v, saved_torque = map(np.asarray, (saved_q, saved_v, saved_torque))
        length, angle, _ = leg_pose(saved_q[:, :, mq] * pose.sign[:4] + pose.zero)
        final_length_error = np.max(abs(length[-1500:, :, 1] - .19), axis=0) * 1000
        final_angle_error = np.degrees(np.max(abs(angle[-1500:, :, 1] - expected_angle), axis=0))
        report = dict(passed=bool(not any(x is not None for x in stopped) and max(final_length_error) < 1 and max(final_angle_error) < .2 and max_gap < .0001),
                      synthetic_prepared_length_mm=args.prepared_length_mm,
                      profile=args.profile,target_right_angle_deg=float(np.degrees(expected_angle)),
                      synthetic_prepared_holding_torque=args.prepared_hold,
                      frozen_parameters_sha256=hashlib.sha256((reference / 'parameters.json').read_bytes()).hexdigest(),
                      local_load_factors=factors.tolist(), guard_stop_s=stopped, max_pin_gap_mm=max_gap * 1000,
                      last3s_right_length_max_error_mm=final_length_error.tolist(), last3s_right_angle_max_error_deg=final_angle_error.tolist(),
                      final_right_length_mm=(length[-1, :, 1] * 1000).tolist(), final_right_angle_deg=np.degrees(angle[-1, :, 1]).tolist(),
                      peak_request_nm=np.max(abs(saved_torque), axis=(0, 1)).tolist(),
                      peak_actual_right_speed_rad_s=float(np.max(abs(saved_v[:, :, mv[2:]]))),
                      peak_target_speed_rad_s=float(abs(speed).max()),
                      zero_left_and_wheel_requests=True, wrapper_checked=True, new_gpu_environment_steps=steps * 10 * count,
                      hardware_commands=False, scope='Necessary-condition model test of actual C; not hardware proof')
        np.savez_compressed(args.output / 'traces.npz', qpos=saved_q, qvel=saved_v, torque_nm=saved_torque,
                            target_q=np.array(saved_target), initial_previous_firmware_nm=initial_previous, length_m=length, angle_rad=angle, time_s=np.arange(steps + 1) * dt)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
        for source in (*sources, algorithm / 'joint_capture.h', algorithm / 'joint_chirp.h', algorithm / 'joint_pose_load.h'):
            shutil.copy2(source, args.output / source.name)
        assert report['passed']


if __name__ == '__main__':
    main()
