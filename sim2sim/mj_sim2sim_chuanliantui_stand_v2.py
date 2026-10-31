#!/usr/bin/env python3
"""Independent native CPU MuJoCo, stand_v2 history160 ONNX, nominal play only.

Reuse the shared physical model/PD helpers; no training environment drives physics.
Initial states are recorded nominal-play states, translated to the origin. The
renderer alone uses mjlab's Viser scene adapter. H7 firmware is not emulated.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from mj_sim2sim_huananhu_multimode import (
    Playback, DT, DECIMATION, LEGS, MOTORS, MODEL_ASSET, sha256,
)


CONTRACT = "stand-v2-native-cpu-history160-integrator4-v2"
POLICY_DT = DT * DECIMATION


class OnnxPolicy:
    def __init__(self, checkpoint, path):
        import onnx
        from onnx.reference import ReferenceEvaluator

        record = json.loads(path.with_name(path.stem+"_export.json").read_text())
        if record["checkpoint_sha256"] != sha256(checkpoint) or record["onnx_sha256"] != sha256(path):
            raise ValueError("ONNX export does not match checkpoint or graph hash")
        if record["observation_contract"] != "stand-v2-isaac32-history160-critic89-physical-v3":
            raise ValueError("This entry requires the stand_v2 32/160 observation contract")
        if record["recovery_contract"] != "stand-v2-normal-wheel-weight-v1":
            raise ValueError("This entry requires the wheel-weight NORMAL rule")
        if record.get("command_contract", "stand-v2-zero-command-v1") != "stand-v2-zero-command-v1":
            raise ValueError("CPU entry supports zero-command models only; post-stand tracking is not adapted")
        infos = torch.load(checkpoint, map_location="cpu", weights_only=False)["infos"]
        if not infos["standup_curriculum"]["unlocked"]:
            raise ValueError("This nominal CPU entry supports the unlocked 0.22 m command")
        if infos["physics_dt_s"] != DT or infos["policy_dt_s"] != POLICY_DT:
            raise ValueError("Checkpoint timing differs from CPU playback")
        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in graph.graph.input}
        if shapes != {"observation_history": [1, 160]}:
            raise ValueError(f"Unexpected ONNX input: {shapes}")
        self.evaluator = ReferenceEvaluator(graph)
        self.velocity = np.zeros(3)

    def __call__(self, history):
        actions, velocity = self.evaluator.run(None, {"observation_history": history.reshape(1, 160)})
        self.velocity = velocity[0]
        return actions[0]


class StandV2Playback(Playback):
    def __init__(self, initial_states, sample=0, integration_substeps=4):
        with np.load(initial_states) as saved:
            self.initial_qpos = saved["qpos"].copy()
            self.initial_qvel = saved["qvel"].copy()
        self.sample = sample
        super().__init__("ground", (0., 0., .22), motor_response_time_constant_s=0.)
        # 原生CPU在部分落地姿态下2ms积分会异常注能；保持PD 2ms，用四个0.5ms子步积分。
        self.integration_substeps = integration_substeps
        self.model.opt.timestep = DT/integration_substeps
        # 本任务名义play没有动作时延，不能继承multimode的H7实测一拍等待。
        self.action_delay_steps = 0
        self._action_history = np.zeros((1, 6), dtype=np.float32)
        self.weight = self.model.body_mass.sum() * np.linalg.norm(self.model.opt.gravity)

    def reset(self, pose=None):
        super().reset("normal")
        if self.initial_qpos.shape[1] != self.model.nq or self.initial_qvel.shape[1] != self.model.nv:
            raise ValueError("Recorded initial state differs from the shared model ordering")
        self.data.qpos[:] = self.initial_qpos[self.sample]
        self.data.qvel[:] = self.initial_qvel[self.sample]
        self.data.qpos[:2] = 0.  # 去掉批量训练环境的平面平移，不改变机器人姿态。
        self.ground_start = True
        self.history = None
        self.stable_ticks = self.standing_ticks = self.airborne_ticks = self.fallen_ticks = 0
        self.has_stood = self.has_landed = False
        self.done = False
        self.wheel_support = np.zeros(2)
        self.base_force = 0.
        mujoco.mj_forward(self.model, self.data)

    def observation(self):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        gravity = rotation.reshape(3, 3).T @ np.array((0., 0., -1.))
        mode = np.zeros(7, dtype=np.float32)
        mode[0 if self.recovered else 3] = 1.
        return np.clip(np.concatenate((self.data.qvel[3:6]*.25, gravity,
            self.command*np.array((2., .25, 5.)),
            self.data.qpos[self.q[LEGS]]-self.nominal[LEGS],
            self.data.qvel[self.v]*.05, self.last_action, mode)), -100., 100.).astype(np.float32)

    def update_history(self, frame):
        # 每次策略调用仅推进一帧，最旧到最新；复位首帧填满五格。
        if self.history is None:
            self.history = np.tile(frame, (5, 1))
        else:
            self.history[:-1] = self.history[1:].copy()
            self.history[-1] = frame
        return self.history.reshape(160)

    def contact_forces(self):
        bodies = (self.base_body, *self.wheel_bodies)
        forces = np.zeros((3, 3))
        local = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            mujoco.mj_contactForce(self.model, self.data, i, local)
            world = c.frame.reshape(3, 3).T @ local[:3]
            for j, body in enumerate(bodies):
                if b2 == body:
                    forces[j] += world
                if b1 == body:
                    forces[j] -= world
        return forces

    def update_mode(self):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        g = rotation.reshape(3, 3).T @ np.array((0., 0., -1.))
        roll, pitch = abs(np.arctan2(-g[1], -g[2])), abs(np.arcsin(np.clip(g[0], -1., 1.)))
        loads = bool((self.wheel_support >= max(.1*self.weight, 5.)).all()
                     and self.wheel_support.sum() >= .8*self.weight)
        down = max(roll, pitch) > np.deg2rad(40) or self.data.qpos[2] <= .20 or not loads
        self.stable_ticks = 0 if down else self.stable_ticks+1
        self.recovered = not down and (self.recovered or self.stable_ticks >= 50)
        supported = bool((self.wheel_support > 1.).any())
        self.has_landed |= supported
        stable = (self.has_landed and self.data.qpos[2] >= .20
                  and g[2] <= -np.cos(np.deg2rad(15)) and self.base_force <= .1 and loads)
        self.standing_ticks = self.standing_ticks+1 if stable else 0
        self.has_stood |= self.standing_ticks >= 50
        self.airborne_ticks = self.airborne_ticks+1 if self.has_landed and not supported else 0
        fallen = self.has_landed and self.has_stood and (self.base_force > 10. or g[2] > -.1)
        self.fallen_ticks = self.fallen_ticks+1 if fallen else 0
        self.done = bool((self.has_landed and self.has_stood and
            (g[2] > 0. or self.airborne_ticks >= 20 or self.fallen_ticks > 100)) or self.steps >= 4000)

    def step(self, controller):
        history = self.update_history(self.observation())
        action = np.asarray(controller(history), dtype=np.float32)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise FloatingPointError("Malformed policy action")
        self.last_action = action.copy()
        for _ in range(DECIMATION):
            torque, spring = self.controls(action)
            # XML原生回调完成轮端响应，不能再乘一遍训练侧的速度增益。
            self.data.ctrl[self.motor_ids] = torque
            self.data.ctrl[self.spring_ids] = spring
            for _ in range(self.integration_substeps):
                mujoco.mj_step(self.model, self.data)
            forces = self.contact_forces()
        self.base_force = float(np.linalg.norm(forces[0]))
        self.wheel_support = forces[1:, 2].copy()
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        self.update_mode()
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Non-finite native MuJoCo state")
        if max(self.data.warning.number):
            raise FloatingPointError("Native MuJoCo issued a simulation warning")
        return action, torque

    def metrics(self, action=None, torque=None):
        row = super().metrics(action, torque)
        row.update(sample=self.sample, normal=bool(self.recovered), has_stood=bool(self.has_stood),
            base_force_N=self.base_force, wheel_support_N=self.wheel_support.tolist(), done=self.done)
        return row


def main(args):
    policy = OnnxPolicy(args.checkpoint, args.onnx)
    env = StandV2Playback(args.initial_states, args.sample)
    server = scene = None
    events = deque()
    rows = []
    if args.viewer:
        import viser
        from mjlab.viewer.viser.scene import MjlabViserScene
        server = viser.ViserServer(host="127.0.0.1", port=args.port, label="stand_v2 CPU sim2sim")
        scene = MjlabViserScene(server, env.model, 1)
        scene.create_scene_gui(camera_distance=2.0, show_debug_viz_control=False)
        paused = server.gui.add_checkbox("暂停", initial_value=False)
        reset = server.gui.add_button("重放当前初态")
        select = server.gui.add_slider("初态编号", min=0, max=len(env.initial_qpos)-1, step=1, initial_value=args.sample)
        reset.on_click(lambda _: events.append(int(select.value)))
        def select_sample(_):
            if int(select.value) != env.sample:
                events.append(int(select.value))
        select.on_update(select_sample)
        info = server.gui.add_markdown("原生CPU MuJoCo／100Hz策略／500Hz PD／2000Hz积分；ONNX160维历史")
        scene.update_from_mjdata(env.data)
        print(f"CPU sim2sim viewer: http://127.0.0.1:{server.get_port()}", flush=True)
    ticks = 0
    try:
        while args.viewer or ticks < round(args.seconds/POLICY_DT):
            started = time.monotonic()
            while events:
                env.sample = events.popleft()
                env.reset()
            if server is None or not paused.value:
                if env.done:
                    if server is None:
                        break
                    env.sample = (env.sample+1) % len(env.initial_qpos)
                    env.reset()
                    select.value = env.sample
                action, torque = env.step(policy)
                ticks += 1
                row = env.metrics(action, torque)
                if args.report:
                    rows.append(row)
                if ticks % 100 == 0:
                    print(json.dumps(row), flush=True)
            if scene is not None:
                if ticks % 3 == 0 or paused.value:
                    with server.atomic():
                        scene.update_from_mjdata(env.data)
                        info.content = (f"**{args.checkpoint.stem} CPU sim2sim** · 初态{env.sample} · {env.data.time:.1f}s\n\n"
                            f"模式：{'NORMAL' if env.recovered else 'RECOVER'} · 曾站起：{env.has_stood}\n\n"
                            f"高度 {env.data.qpos[2]:.3f}m · 轮支撑 {env.wheel_support.round(1)}N")
                time.sleep(max(0., POLICY_DT-(time.monotonic()-started)))
    finally:
        if server is not None:
            server.stop()
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(dict(contract=CONTRACT, rows=rows,
                checkpoint_sha256=sha256(args.checkpoint), onnx_sha256=sha256(args.onnx),
                xml_sha256=sha256(MODEL_ASSET), initial_states_sha256=sha256(args.initial_states),
                sample=args.sample, policy_dt_s=POLICY_DT, pd_dt_s=DT,
                physics_dt_s=env.model.opt.timestep, integration_substeps=env.integration_substeps,
                parameter_updates=0, scope="Single initial state native CPU replay; not hardware acceptance"), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--initial-states", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=40.)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--port", type=int, default=8093)
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("Report already exists")
    main(args)
