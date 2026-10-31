#!/usr/bin/env python3
"""Compare native stand_v2 preprocessing/PD/FIFO/mode timers with the real task."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"sim2sim"))
from mj_sim2sim_chuanliantui_stand_v2 import StandV2Playback, OnnxPolicy
from mj_sim2sim_huananhu_multimode import sha256, MOTORS, MODEL_ASSET
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_stand_v2.observations import get_state
from ct_mjlab.envs.chuanliantui_stand_v2 import mdp


def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    native = StandV2Playback(args.initial_states)
    policy = OnnxPolicy(args.checkpoint, args.onnx)
    cfg, agent = task_registry.get_cfgs("chuanliantui_stand_v2", play=True)
    env, _ = task_registry.make_env("chuanliantui_stand_v2", env_cfg=cfg, train_cfg=agent,
        num_envs=1, seed=42, device="cuda:0")
    obs_error = torque_error = 0.
    try:
        raw = env.unwrapped
        observer, pd = get_state(raw), raw.action_manager.get_term("pd")
        state = raw.command_manager.get_term("standup")
        assert native.model.nq == raw.sim.mj_model.nq and native.model.nv == raw.sim.mj_model.nv
        for i in range(native.model.njnt):
            assert native.model.joint(i).name == raw.sim.mj_model.joint(i).name.removeprefix("robot/")
            assert native.model.jnt_qposadr[i] == raw.sim.mj_model.jnt_qposadr[i]
        np.testing.assert_allclose(native.nominal, pd.nominal_positions[0].cpu(), atol=1e-6, rtol=0)
        np.testing.assert_allclose(native.weight, float(raw.sim.model.body_mass.sum()*9.81), atol=1e-4)
        rng = np.random.default_rng(42)
        ids = torch.tensor([0], device=raw.device)
        for index in range(16):
            native.reset()
            if index:
                q = np.array((1., .15*rng.normal(), .15*rng.normal(), .15*rng.normal()))
                native.data.qpos[3:7] = q/np.linalg.norm(q)
                native.data.qvel[:] = rng.uniform(-4., 4., native.model.nv)
            native.recovered = bool(index % 2)
            native.last_action = rng.uniform(-2., 2., 6).astype(np.float32)
            raw.sim.data.qpos[0] = torch.as_tensor(native.data.qpos, device=raw.device, dtype=torch.float32)
            raw.sim.data.qvel[0] = torch.as_tensor(native.data.qvel, device=raw.device, dtype=torch.float32)
            pd._raw_actions[0] = torch.as_tensor(native.last_action, device=raw.device)
            state.sampled[:] = state._command[:] = torch.tensor([0., 0., .22], device=raw.device)
            state.observation_recovered[:] = native.recovered
            raw.sim.forward()
            observer.reset(ids)
            actual, reference = native.observation(), observer.actor()[0].cpu().numpy()
            obs_error = max(obs_error, float(np.max(np.abs(actual-reference))))
            np.testing.assert_allclose(actual, reference, atol=2e-6, rtol=0)
            action = rng.uniform(-3., 3., 6).astype(np.float32)
            for age in (0, 19, 20, 100):
                native.steps = age
                raw.episode_length_buf[:] = age
                pd.process_actions(torch.as_tensor(action[None], device=raw.device))
                pd.apply_actions()
                torque, spring = native.controls(action)
                expected = pd.torques[0].cpu().numpy()
                torque_error = max(torque_error, float(np.max(np.abs(torque-expected))))
                np.testing.assert_allclose(torque, expected, atol=2e-5, rtol=0)
        native.history = None
        frames = [np.arange(32, dtype=np.float32)+i for i in range(9)]
        for index, frame in enumerate(frames):
            actual = native.update_history(frame).reshape(5, 32)
            expected = np.asarray([frames[max(0, index-4+k)] for k in range(5)])
            np.testing.assert_array_equal(actual, expected)
            assert np.isfinite(policy(actual.reshape(160))).all()
        native.reset()
        calls = {"pd":0, "integrator":0}
        original_pd, original_step = native.controls, mujoco.mj_step
        def count_pd(action):
            calls["pd"] += 1
            return original_pd(action)
        def count_step(model, data):
            calls["integrator"] += 1
            return original_step(model, data)
        native.controls = count_pd
        mujoco.mj_step = count_step
        try:
            native.step(lambda _: np.zeros(6, dtype=np.float32))
        finally:
            native.controls = original_pd
            mujoco.mj_step = original_step
        assert calls == {"pd":5, "integrator":20}
        assert abs(native.data.time-.01) < 1e-12
        # 同一组人工状态逐拍喂入真实任务和独立入口，核对49/50拍与掉承重退出。
        native.reset()
        native.data.qpos[:7] = [0, 0, .25, 1, 0, 0, 0]
        raw.sim.data.qpos[0] = torch.as_tensor(native.data.qpos, device=raw.device, dtype=torch.float32)
        raw.sim.data.qvel.zero_(); raw.sim.forward()
        state.reset(ids)
        for tick in range(102):
            support = native.weight*np.array([.5, .5]) if tick != 50 else np.array([1., 1.])
            native.wheel_support = support
            raw.scene["wheel_contact"].data.force.zero_()
            raw.scene["wheel_contact"].data.force[0, :, 2] = -torch.as_tensor(support, device=raw.device, dtype=torch.float32)
            raw.scene["base_contact"].data.force.zero_()
            raw.common_step_counter += 1
            mdp.failure(raw)
            native.update_mode()
            assert native.recovered == bool(state.observation_recovered[0])
            assert native.has_stood == bool(state.has_stood[0])
            if tick == 48:
                assert not native.recovered
            if tick == 49:
                assert native.recovered
            if tick == 50:
                assert not native.recovered
        report = dict(passed=True, parameter_updates=0, observation_samples=16,
            maximum_observation_error=obs_error, pd_cases=64, maximum_pd_error_Nm=torque_error,
            motor_order=list(MOTORS), joint_order_equal=True, normal_gate_timer_steps=102,
            FIFO_oldest_to_latest_and_first_frame_fill=True, zero_delay_and_20step_hold=True,
            native_integrator_dt_s=.0005, pd_dt_s=.002, policy_dt_s=.01,
            native_integrator_steps_per_policy=20, pd_evaluations_per_policy=5,
            onnx_sha256=sha256(args.onnx), checkpoint_sha256=sha256(args.checkpoint),
            xml_sha256=sha256(MODEL_ASSET), scope="Finite interface checks; CPU behavior measured separately")
        (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--initial-states", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
