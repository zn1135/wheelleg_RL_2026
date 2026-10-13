"""V11 actor in native CPU MuJoCo, with keyboard commands (Python 3.12).

This entry point is pinned to the documented wheelleg V11 checkpoint. It uses
this repository's closed-chain robot, nominal physics, and fixed NORMAL mode.
It does not train, export, or implement the legacy H7 controller.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mjlab_training/src'))

import mujoco
import numpy as np
import torch
from ct_mjlab.asset import POSES, cpu_model
from ct_mjlab.config import ASSET
from ct_mjlab.policy import actor_model
from ct_mjlab.reference_training.gas_spring import gas_spring_force

V11_SHA = '62efa60cf3c0884d0abf4a2378d778371135a49c475ed3453ebec08136d22366'
MOTORS = ('lf0', 'lf00', 'lfwheel', 'rf0', 'rf00', 'rfwheel')
OBS_ORDER = np.array([0, 1, 3, 4, 2, 5])
LEGS, WHEELS = np.array([0, 1, 3, 4]), np.array([2, 5])
DT, DECIMATION = .002, 10
HELP = 'Up/Down: speed | Left/Right: yaw rate | PgUp/PgDn: height\nEnter: stop | Home: reset standing | End: reset ground | F1: pause'


def load_actor(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != V11_SHA:
        raise ValueError('Checkpoint is not the documented V11 model')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    info = saved['infos']
    if (info['observation_contract'] != 'chuanliantui-huananhu-flat-35-78-v1'
            or not info['control_contract'].endswith('sourcexml-spring90mm-v10')
            or info['huananhu_training_iteration'] != 4000 or saved['iter'] != 999):
        raise ValueError('V11 metadata mismatch')
    actor = actor_model().eval()
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    if not all(torch.isfinite(x).all() for x in actor.state_dict().values()):
        raise ValueError('Nonfinite policy parameters')
    return actor


class Playback:
    def __init__(self, height=.30, pose='normal'):
        if hashlib.sha256(ASSET.read_bytes()).hexdigest() != POSES['asset_sha256']:
            raise ValueError('Robot asset does not match the solved pose cache')
        self.model = cpu_model()
        self.model.opt.ls_iterations = 20
        self.data = mujoco.MjData(self.model)
        self.q = np.array([self.model.joint(n).qposadr[0] for n in MOTORS])
        self.v = np.array([self.model.joint(n).dofadr[0] for n in MOTORS])
        self.motor_ids = [self.model.actuator(n + '_motor').id for n in MOTORS]
        springs = ['left_spring2_joint', 'right_spring2_joint']
        self.sq = [self.model.joint(n).qposadr[0] for n in springs]
        self.sv = [self.model.joint(n).dofadr[0] for n in springs]
        self.spring_ids = [self.model.actuator(s + '_gas_spring_motor').id for s in ('left', 'right')]
        self.nominal = np.array([POSES['nominal'][n] for n in MOTORS], dtype=np.float32)
        self.reference = self.model.qpos0[self.q].astype(np.float32)
        self.command = np.array([0., 0., height], dtype=np.float32)
        self.reset(pose)

    def reset(self, pose='normal'):
        mujoco.mj_resetData(self.model, self.data)
        # Closed poses already solved against this exact XML; no online IK.
        values = POSES['nominal' if pose == 'normal' else 'reset']
        self.data.qpos[:7] = (0., 0., .38 if pose == 'normal' else POSES['root_height'], 1., 0., 0., 0.)
        for name, value in values.items():
            self.data.qpos[self.model.joint(name).qposadr[0]] = value
        self.ground = pose == 'ground'
        self.steps = 0
        self.last_action = np.zeros(6, dtype=np.float32)
        self.command[:2] = 0.
        mujoco.mj_forward(self.model, self.data)

    def observation(self):
        d = self.data
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, d.qpos[3:7])
        gravity = rotation.reshape(3, 3).T @ np.array([0., 0., -1.])
        q = d.qpos[self.q].astype(np.float32) - self.nominal
        q[WHEELS] = 0.
        return np.concatenate((
            [self.command[0], 0., self.command[1], np.clip(self.command[2], 0., 1.) * 5],
            np.clip(d.qvel[3:6], -100, 100) * .5,
            np.clip(gravity, -100, 100), np.clip(q[OBS_ORDER], -100, 100),
            np.clip(d.qvel[self.v][OBS_ORDER], -200, 200) * .1,
            np.clip(self.last_action[OBS_ORDER], -100, 100), [1., 0., 0., 0., 0., 0., 0.],
        )).astype(np.float32)

    def controls(self, action):
        q, v = self.data.qpos[self.q], self.data.qvel[self.v]
        target = np.clip(self.nominal[LEGS] + .5 * action[LEGS],
                         self.reference[LEGS] - 3.14, self.reference[LEGS] + 3.14)
        tau = np.zeros(6)
        tau[LEGS] = 60. * (target - q[LEGS]) - 2. * v[LEGS]
        tau[WHEELS] = .1 * (np.clip(10. * action[WHEELS], -150, 150) - v[WHEELS])
        tau = np.clip(tau, -np.array([40, 40, 3.9, 40, 40, 3.9]), [40, 40, 3.9, 40, 40, 3.9])
        if self.ground and self.steps < 10:
            tau[:] = 0.
        return tau, gas_spring_force(self.data.qpos[self.sq], self.data.qvel[self.sv])

    def step(self, actor):
        with torch.inference_mode():
            action = actor({'actor': torch.from_numpy(self.observation()[None])}).numpy()[0]
        if not np.isfinite(action).all():
            raise FloatingPointError('Nonfinite actor output')
        self.last_action = action.copy()
        for _ in range(DECIMATION):
            tau, spring = self.controls(action)
            self.data.ctrl[self.motor_ids] = tau
            self.data.ctrl[self.spring_ids] = spring
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('Nonfinite simulation state')
        if max(self.data.warning.number):
            raise FloatingPointError('MuJoCo reported a simulation warning')

    def metrics(self):
        obs = self.observation()
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        return dict(time=float(self.data.time), height=float(self.data.qpos[2]),
                    vx=float((rotation.reshape(3, 3).T @ self.data.qvel[:3])[0]),
                    yaw_rate=float(self.data.qvel[5]),
                    tilt_deg=float(np.rad2deg(np.arccos(np.clip(-obs[9], -1, 1)))),
                    command=self.command.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--render', action='store_true')
    p.add_argument('--seconds', type=float, default=30., help='0 keeps a rendered window open until closed')
    p.add_argument('--cmd-vx', type=float, default=0.)
    p.add_argument('--cmd-yaw', type=float, default=0., help='yaw rate, rad/s')
    p.add_argument('--cmd-height', type=float, default=.30)
    p.add_argument('--pose', choices=('normal', 'ground'), default='normal')
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    if not all(np.isfinite(x) for x in (args.seconds, args.cmd_vx, args.cmd_yaw, args.cmd_height)):
        p.error('All numeric arguments must be finite')
    if args.seconds < 0 or (args.seconds == 0 and not args.render):
        p.error('Use positive seconds, or --render --seconds 0')
    if abs(args.cmd_vx) > 1.5 or abs(args.cmd_yaw) > 1.5 or not .20 <= args.cmd_height <= .34:
        p.error('Command bounds: speed/yaw ±1.5; height .20–.34 m')
    if args.report and args.report.exists():
        raise FileExistsError(args.report)
    torch.set_num_threads(1)
    actor = load_actor(args.checkpoint)
    env = Playback(args.cmd_height, args.pose)
    env.command[:] = (args.cmd_vx, args.cmd_yaw, args.cmd_height)
    events = deque()
    viewer, paused, rows = None, False, []
    total_steps, completed = 0, False
    print('V11 native CPU MuJoCo | ' + V11_SHA, flush=True)
    print(HELP, flush=True)
    try:
        if args.render:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=events.append)
            with viewer.lock():
                viewer.cam.distance = 2.2
                viewer.cam.azimuth = 135.
                viewer.cam.elevation = -18.
                viewer.opt.geomgroup[3] = 0
        while (args.seconds == 0 or total_steps * DT * DECIMATION < args.seconds):
            start = time.monotonic()
            if viewer and not viewer.is_running():
                break
            while events:
                key = events.popleft()
                if key == 265: env.command[0] += .1
                elif key == 264: env.command[0] -= .1
                elif key == 263: env.command[1] += .1
                elif key == 262: env.command[1] -= .1
                elif key == 266: env.command[2] += .01
                elif key == 267: env.command[2] -= .01
                elif key in (257, 335): env.command[:2] = 0.
                elif key == 268:
                    with viewer.lock(): env.reset('normal')
                elif key == 269:
                    with viewer.lock(): env.reset('ground')
                elif key == 290: paused = not paused
                env.command[:] = np.clip(env.command, [-1.5, -1.5, .20], [1.5, 1.5, .34])
            if not paused:
                if viewer:
                    with viewer.lock(): env.step(actor)
                else:
                    env.step(actor)
                total_steps += 1
                row = env.metrics()
                if args.report: rows.append(row)
                if total_steps % 50 == 0:
                    print(json.dumps(row), flush=True)
            if viewer:
                with viewer.lock(): viewer.cam.lookat[:] = env.data.qpos[:3]
                row = env.metrics()
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100.value,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                    f'V11 | {"PAUSED" if paused else "RUNNING"}\n'
                    f'vx {env.command[0]:+.2f} m/s | yaw {env.command[1]:+.2f} rad/s\n'
                    f'height {row["height"]:.3f} / {env.command[2]:.2f} m\n{HELP}', '')])
                viewer.sync()
                time.sleep(max(0., DT * DECIMATION - (time.monotonic() - start)))
        completed = True
    finally:
        if viewer: viewer.close()
        if args.report:
            with args.report.open('x') as f:
                json.dump(dict(checkpoint_sha256=V11_SHA, asset_sha256=POSES['asset_sha256'],
                               backend='native-mujoco-cpu', pose=args.pose, dt=DT,
                               decimation=DECIMATION, completed=completed, trajectory=rows), f, indent=2)


if __name__ == '__main__':
    main()
