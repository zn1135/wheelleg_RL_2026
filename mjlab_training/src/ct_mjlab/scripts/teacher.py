"""Run the H7 LQR+VMC teacher in mjlab; never starts PPO or loads a policy."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from ct_mjlab.teacher.env import make_teacher_env_cfg
from ct_mjlab.teacher.mechanics import DT
from ct_mjlab.teacher.standup import PHASE_NAMES, FAULT_NAMES, FAILED
from ct_mjlab.envs.chuanliantui.model import MODEL_XML

ROOT = Path(__file__).resolve().parents[4]


class PlaybackComplete(BaseException):
    """Exit the viewer loop without its generic Exception handler pausing playback."""


def command_schedule(args):
    rows = [{'time': 0., 'vx': args.vx, 'yaw': args.yaw, 'leg_length': args.leg_length}]
    if args.sequence:
        rows = json.loads(args.sequence.read_text())
    if not isinstance(rows, list) or not rows or rows[0].get('time') != 0:
        raise ValueError('命令序列必须为非空JSON数组，第一项time=0')
    for i, row in enumerate(rows):
        if set(row) != {'time', 'vx', 'yaw', 'leg_length'}:
            raise ValueError('每项必须包含time/vx/yaw/leg_length')
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError('命令必须为有限数值')
        if (row['time'] < 0 or (i and row['time'] <= rows[i-1]['time']) or
                abs(row['vx']) > 1.5 or abs(row['yaw']) > 1.5 or not .13 <= row['leg_length'] <= .34):
            raise ValueError('time须递增，vx/yaw限±1.5，leg_length限0.13～0.34m')
    return rows


class Recorder:
    def __init__(self, env, output, args, schedule):
        self.env, self.output, self.args, self.schedule = env, output, args, schedule
        self.term = env.action_manager.get_term('teacher')
        self.count = 0
        self.manual_resets = 0
        self.run_error = None
        self.frames = []
        self.t0 = time.monotonic()
        self.weight = float(env.sim.get_default_field('body_mass')[env.scene['robot'].indexing.body_ids].sum()) * 9.81
        self.hold = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.failed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.max_tilt = torch.zeros(env.num_envs, device=env.device)
        self.base_contact = torch.zeros_like(self.failed)
        self.tracking_tilt = torch.zeros_like(self.max_tilt)
        self.tracking_contact = torch.zeros_like(self.failed)
        self.hip_bodies = env.scene['robot'].find_bodies(('lf0', 'rf0'), preserve_order=True)[0]
        self.wheel_bodies = env.scene['robot'].find_bodies(('lfwheel', 'rfwheel'), preserve_order=True)[0]
        self.errors = torch.zeros(env.num_envs, 4, device=env.device)
        self.measured = 0
        self.initial_qpos = env.sim.data.qpos.cpu().numpy().copy()
        sources = list((ROOT / 'mjlab_training/src/ct_mjlab/teacher').glob('*.py')) + [Path(__file__)]
        self.code_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}

    def sample(self, command):
        env, term = self.env, self.term
        robot = env.scene['robot'].data
        g = robot.projected_gravity_b
        roll = torch.atan2(-g[:, 1], -g[:, 2])
        pitch = torch.asin(g[:, 0].clamp(-1, 1))
        tilt = torch.maximum(roll.abs(), pitch.abs())
        base = env.scene['base_contact'].data.force.norm(dim=-1).amax(-1)
        wheels = -env.scene['wheel_contact'].data.force[:, :, 2]
        balance = ((tilt <= math.radians(35)) & (base <= 1) &
                   (wheels >= max(.1*self.weight, 5.)).all(-1) &
                   (wheels.sum(-1) >= .8*self.weight) & (wheels.sum(-1) <= 1.2*self.weight) &
                   (robot.root_link_ang_vel_b[:, :2].norm(dim=-1) <= .5))
        self.failed |= term.controller.standup.phase == FAILED
        self.hold.copy_(torch.where(balance & ~self.failed, self.hold + 1, 0))
        self.max_tilt.copy_(torch.maximum(self.max_tilt, tilt))
        self.base_contact |= base > 1
        height = robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        leg = robot.body_link_pos_w[:, self.wheel_bodies] - robot.body_link_pos_w[:, self.hip_bodies]
        quat = robot.root_link_quat_w[:, None].expand(-1, 2, -1)
        leg = quat_apply_inverse(quat.reshape(-1, 4), leg.reshape(-1, 3)).reshape(-1, 2, 3)
        xml_length = leg[:, :, (0, 2)].norm(dim=-1)
        if self.count * DT > 2.:
            self.tracking_tilt.copy_(torch.maximum(self.tracking_tilt, tilt))
            history = env.scene['base_contact'].data.force_history
            self.tracking_contact |= history.norm(dim=-1).amax(dim=(1, 2)) > 1
            self.errors += torch.stack(((robot.root_link_lin_vel_b[:, 0]-command[:, 0]).abs(),
                                        (robot.root_link_ang_vel_b[:, 2]-command[:, 1]).abs(),
                                        (height-command[:, 2]-.03648).abs(),
                                        (xml_length-command[:, 2:3]).abs().mean(-1)), -1)
            self.measured += 1
        if self.count % 10 == 0:
            root = torch.cat((robot.root_link_pos_w, robot.root_link_quat_w,
                              robot.root_link_lin_vel_w, robot.root_link_ang_vel_w), -1)
            frame = dict(time=self.count*DT, command=command, root=root,
                         qpos=env.sim.data.qpos[:], qvel=env.sim.data.qvel[:],
                         motor_q=env.sim.data.qpos[:, term.qadr], motor_dq=env.sim.data.qvel[:, term.vadr],
                         length=term.controller.length, xml_length=xml_length, theta=term.controller.theta,
                         h7_torque=term.h7_torques, torque=term.torques,
                         control_active=term.control_active,
                         phase=term.controller.standup.phase, fault=term.controller.standup.fault,
                         wheel_support=wheels, base_contact=base, lqr_state=term.controller.x,
                         lqr_output=term.controller.u, air=term.controller.air.flight,
                         support_estimate=term.controller.air.force,
                         axial_force=term.controller.force, swing_torque=term.controller.swing_torque,
                         spring_compensation=term.controller.spring, gravity_compensation=term.controller.gravity,
                         raw_h7_torque=term.controller.raw_torque,
                         actuator_torque=env.sim.data.actuator_force[:, term.motor_actuators])
            self.frames.append({k: v.detach().cpu().numpy().copy() if isinstance(v, torch.Tensor) else v
                                for k, v in frame.items()})

    def save(self, completed):
        if self.frames:
            np.savez_compressed(self.output / 'trace.npz',
                                **{k: np.asarray([f[k] for f in self.frames]) for k in self.frames[0]},
                                initial_qpos=self.initial_qpos)
        term = self.term
        errors = (self.errors / max(self.measured, 1)).cpu().tolist()
        phases = term.controller.standup.phase.cpu().tolist()
        faults = term.controller.standup.fault.cpu().tolist()
        rows = []
        for i in range(self.env.num_envs):
            rows.append(dict(env=i, phase=PHASE_NAMES[phases[i]], fault=FAULT_NAMES[faults[i]],
                             final_balance_hold_s=float(self.hold[i])*DT,
                             final_window_balance=bool(completed and not self.manual_resets and
                                                    self.hold[i] >= 1000 and not self.failed[i]),
                             final_balance_v31=bool(completed and self.count * DT >= 20 and
                                                    not self.manual_resets and self.hold[i] >= 1000 and not self.failed[i]),
                             max_tilt_deg=math.degrees(float(self.max_tilt[i])),
                             base_contact=bool(self.base_contact[i]), ever_failed=bool(self.failed[i]),
                             mean_abs_vx_error=errors[i][0] if self.measured else None,
                             mean_abs_yaw_error=errors[i][1] if self.measured else None,
                             mean_abs_height_error=errors[i][2] if self.measured else None,
                             mean_abs_leg_length_error=errors[i][3] if self.measured else None,
                             tracking_pass=bool(completed and not self.manual_resets and
                                                self.count * DT >= 12 and self.measured and
                                                self.args.initial == 'standing' and len(self.schedule) == 1 and
                                                not self.failed[i] and not self.tracking_contact[i] and
                                                self.tracking_tilt[i] <= math.radians(25) and
                                                errors[i][0] <= .30 and errors[i][1] <= .30 and errors[i][2] <= .03)))
        h7 = ROOT / 'deployment/H7_RL_ros/imcalib'
        sources = list(h7.glob('Algorithm/*.c')) + [h7 / 'config/machine_config.c']
        record = dict(created_at=datetime.now().astimezone().isoformat(),
                      git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                      completed=completed, simulated_seconds=self.count*DT, wall_seconds=time.monotonic()-self.t0,
                      manual_resets=self.manual_resets,
                      run_error=self.run_error,
                      physics_dt=DT, controller_dt=DT, trace_hz=100, mass_kg=self.weight/9.81,
                      ground_start_hold_s=term.cfg.ground_start_hold_s if term.cfg.initial == 'ground' else 0.,
                      h7_support_mass_kg=20.534, h7_wheel_radius=.0525,
                      position_target_m=0., h7_original_position_target_m=.4,
                      teacher_source_sha256=self.code_hashes,
                      xml_sha256=hashlib.sha256(MODEL_XML.read_bytes()).hexdigest(),
                      h7_sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                      arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()},
                      schedule=self.schedule, results=rows,
                      scope='mjlab teacher only; no PPO, no inverted recovery, no hardware validation; '
                            'tracking uses H7 leg-length command + 0.03648m geometric height proxy; '
                            'v3.1 requires 20s and final one-second balance; '
                            'tracking measures after 2s with >=12s total; sticky failure, no episode restart')
        (self.output / 'result.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps({'output': str(self.output), 'completed': completed, 'results': rows}, ensure_ascii=False), flush=True)


def run(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    schedule = command_schedule(args)
    if args.num_envs < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError('num-envs和seconds必须为正')
    output = args.output or ROOT / 'logs/mjlab/teacher' / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output.mkdir(parents=True, exist_ok=False)
    configure_torch_backends()
    cfg = make_teacher_env_cfg(initial=args.initial, num_envs=args.num_envs, seed=args.seed,
                               velocity_integral=getattr(args,'velocity_integral',False),
                               ground_start_hold_s=getattr(args,'ground_start_hold',0.))
    env = ManagerBasedRlEnv(cfg, device=args.device)
    recorder = None
    completed = False
    try:
        env.reset(seed=args.seed)
        recorder = Recorder(env, output, args, schedule)
        steps = max(1, round(args.seconds / DT))
        command = torch.tensor([schedule[0][k] for k in ('vx', 'yaw', 'leg_length')],
                               device=args.device).repeat(args.num_envs, 1)
        index = 0

        def policy(_obs):
            nonlocal index, completed
            if recorder.count:
                recorder.sample(command)
            if recorder.count >= steps:
                completed = True
                raise PlaybackComplete
            while index+1 < len(schedule) and schedule[index+1]['time'] <= recorder.count*DT:
                index += 1
                command[:] = command.new_tensor([schedule[index][k] for k in ('vx', 'yaw', 'leg_length')])
            recorder.count += 1
            return command

        def reset_recording():
            recorder.manual_resets += 1
            recorder.hold.zero_()

        policy.reset = reset_recording
        with torch.inference_mode():
            if args.headless:
                while True:
                    env.step(policy(None))
            else:
                viewer = args.viewer
                if viewer == 'auto':
                    viewer = 'native' if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') else 'viser'
                if viewer == 'native':
                    from mjlab.viewer import NativeMujocoViewer
                    NativeMujocoViewer(env, policy).run()
                else:
                    from mjlab.viewer import ViserPlayViewer
                    ViserPlayViewer(env, policy).run()
    except PlaybackComplete:
        pass
    except Exception as exc:
        if recorder is not None:
            recorder.run_error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            if recorder is not None:
                recorder.save(completed)
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--viewer', choices=('auto', 'native', 'viser'), default='auto')
    parser.add_argument('--initial', choices=('standing', 'ground'), default='standing')
    parser.add_argument('--ground-start-hold', type=float, default=0.,
                        help='Zero active motor torque before ground-start takeover (seconds)')
    parser.add_argument('--vx', type=float, default=0.)
    parser.add_argument('--yaw', type=float, default=0.)
    parser.add_argument('--leg-length', type=float, default=.15)
    parser.add_argument('--sequence', type=Path)
    parser.add_argument('--velocity-integral', action='store_true', help='Simulation velocity-error integral in LQR position slot')
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
