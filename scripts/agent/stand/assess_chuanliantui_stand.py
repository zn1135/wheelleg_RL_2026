#!/usr/bin/env python3
"""Evaluate the independent standing task; never train or operate hardware.

Raw motor coordinates are sampled on every 2 ms physics tick. A settled window
is anchored once: later failure cannot restart the clock or move that anchor.
Terminated worlds are reset only after their failed result has been frozen, so
other worlds can finish under mjlab's explicit-reset requirement.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "mjlab_training/src"))
TASK = "chuanliantui_stand"
MOTORS = ("lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel")
LEG_INDICES = (0, 1, 3, 4)
PHYSICS_DT, POLICY_DT = .002, .01
WINDOW_S, HEIGHT_M = 20., .27
CRITERIA_VERSION = "stand-speed-v2"
LEG_SPEED_RAD_S = .5
BODY_SPEED_M_S = .05
CHANNELS = ("q_lf0", "q_lf00", "q_lfwheel", "q_rf0", "q_rf00", "q_rfwheel",
            "qd_lf0", "qd_lf00", "qd_lfwheel", "qd_rf0", "qd_rf00", "qd_rfwheel",
            "x", "y", "roll_deg", "pitch_deg", "height_m", "wheel_supported",
            "base_contact", "finite", "vx_world_m_s", "vy_world_m_s")


def window_verdict(*, duration_s, maximum_displacement_m, maximum_tilt_deg,
                   maximum_height_error_m, wheel_support_fraction, base_contact_fraction,
                   leg_peak_to_peak_rad, maximum_body_speed_m_s, leg_max_abs_speed_rad_s, failed=False):
    """The same pure arithmetic is used by the report and the offline check."""
    numbers = np.r_[duration_s, maximum_displacement_m, maximum_tilt_deg,
                    maximum_height_error_m, wheel_support_fraction, base_contact_fraction,
                    leg_peak_to_peak_rad, maximum_body_speed_m_s, leg_max_abs_speed_rad_s]
    return bool(np.isfinite(numbers).all() and not failed
                and duration_s >= WINDOW_S - 1e-6
                and maximum_body_speed_m_s <= BODY_SPEED_M_S + 1e-8
                and maximum_tilt_deg <= 25. + 1e-6
                and maximum_height_error_m <= .03 + 1e-8
                and wheel_support_fraction >= .95 - 1e-8
                and base_contact_fraction <= .01 + 1e-8
                and np.max(leg_max_abs_speed_rad_s) <= LEG_SPEED_RAD_S + 1e-7)


def dominant_frequency(q, dt):
    q = np.asarray(q, dtype=np.float64)
    if len(q) < 2:
        return [0.] * q.shape[-1]
    centred = q - q.mean(axis=0)
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(len(q))[:, None], axis=0))
    spectrum[0] = 0.
    frequency = np.fft.rfftfreq(len(q), dt)[spectrum.argmax(axis=0)]
    frequency[np.ptp(q, axis=0) < 1e-8] = 0.
    return frequency.tolist()


def total_tilt_degrees(roll_pitch_deg):
    """Angle of body +Z to world +Z; the two raw Euler columns stay unchanged."""
    roll_pitch = np.deg2rad(np.asarray(roll_pitch_deg, dtype=np.float64))
    cosine = np.cos(roll_pitch[..., 0]) * np.cos(roll_pitch[..., 1])
    return np.rad2deg(np.arccos(np.clip(cosine, -1., 1.)))


def summarize_window(samples, dt=PHYSICS_DT):
    """Summarize raw samples, not policy-step means or action changes.

    samples has q/qd [T,6], xy/tilt_deg [T,2] (raw roll/pitch degrees), height_m,
    wheel_supported/base_contact/finite [T], and anchor_xy [2].
    """
    q, qd = np.asarray(samples["q"]), np.asarray(samples["qd"])
    n = len(q)
    if not n:
        return {"duration_s": 0., "maximum_displacement_m": None,
                "maximum_tilt_deg": None, "maximum_height_error_m": None,
                "wheel_support_fraction": 0., "base_contact_fraction": 0.,
                "leg_peak_to_peak_rad": [None] * 4, "qd_rms_rad_s": [None] * 6,
                "leg_max_abs_speed_rad_s": [None] * 4, "maximum_body_speed_m_s": None,
                "leg_dominant_frequency_hz": [None] * 4, "finite": False, "passed": False}
    range_q = np.vstack((samples["anchor_q"], q)) if "anchor_q" in samples else q
    body_velocity = np.asarray(samples["body_velocity_xy"])
    speed_qd = np.vstack((samples["anchor_qd"], qd)) if "anchor_qd" in samples else qd
    speed_xy = np.vstack((samples["anchor_velocity_xy"], body_velocity)) if "anchor_velocity_xy" in samples else body_velocity
    result = dict(duration_s=n * dt,
        maximum_body_speed_m_s=float(np.linalg.norm(speed_xy, axis=-1).max()),
        leg_max_abs_speed_rad_s=np.abs(speed_qd[:, LEG_INDICES]).max(axis=0).tolist(),
        maximum_displacement_m=float(np.linalg.norm(
            np.asarray(samples["xy"]) - samples["anchor_xy"], axis=-1).max()),
        maximum_tilt_deg=float(total_tilt_degrees(samples["tilt_deg"]).max()),
        maximum_height_error_m=float(np.abs(np.asarray(samples["height_m"]) - HEIGHT_M).max()),
        wheel_support_fraction=float(np.mean(samples["wheel_supported"])),
        base_contact_fraction=float(np.mean(samples["base_contact"])),
        leg_peak_to_peak_rad=np.ptp(range_q[:, LEG_INDICES], axis=0).tolist())
    finite = bool(np.asarray(samples["finite"]).all() and np.isfinite(q).all()
                  and np.isfinite(speed_qd).all() and np.isfinite(speed_xy).all())
    passed = window_verdict(**result, failed=not finite)
    return {**result, "qd_rms_rad_s": np.sqrt(np.mean(qd.astype(np.float64) ** 2, axis=0)).tolist(),
            "leg_dominant_frequency_hz": dominant_frequency(q[:, LEG_INDICES], dt),
            "finite": finite, "passed": passed}


@dataclass
class AcceptanceState:
    """One sample's irreversible qualification/delivery/measurement lifecycle."""
    case_kind: str
    phase: str = "waiting"
    stable_run_s: float = 0.
    delivered_s: float | None = None
    anchor_s: float | None = None
    fell: bool = False
    recovered: bool = False
    failed_reason: str | None = None
    ended_s: float | None = None
    previous_s: float = 0.

    def fail(self, reason):
        if self.phase != "done":
            self.phase, self.failed_reason = "failed", self.failed_reason or reason

    def mark_delivered(self, t):
        if self.phase != "waiting" or self.delivered_s is not None:
            raise ValueError("Disturbance can only be delivered once")
        self.delivered_s, self.phase, self.stable_run_s = t, "post_push", 0.

    def update(self, t, stable, fallen, terminal=False, nonfinite=False, unstable=False):
        dt, self.previous_s = max(0., t - self.previous_s), t
        if self.phase in ("failed", "done"):
            return "none"
        if terminal or nonfinite:
            self.fail("nonfinite" if nonfinite else "terminated_or_timeout")
            self.ended_s = t
            return "none"
        if self.delivered_s is not None:
            self.fell |= bool(fallen)
        if self.phase == "measuring":
            if fallen or unstable:
                self.fail("lost_balance_after_anchor")
                self.ended_s = t
            return "none"
        self.stable_run_s = self.stable_run_s + dt if stable else 0.
        if self.phase == "waiting":
            deadline = 5. if self.case_kind == "hold" else 10.
            eligible = self.stable_run_s >= .5 - 1e-8
            if self.case_kind == "hold" and eligible and t <= deadline + 1e-8:
                self.phase, self.anchor_s = "measuring", t
                return "anchor"
            if self.case_kind != "hold" and eligible and 5. - 1e-8 <= t <= deadline + 1e-8:
                return "deliver"
            if t >= deadline - 1e-8:
                self.fail("initial_stability_deadline" if self.case_kind == "hold" else "delivery_deadline")
                self.ended_s = t
        elif self.phase == "post_push":
            elapsed = t - self.delivered_s
            if self.stable_run_s >= .5 - 1e-8 and elapsed <= 5. + 1e-8:
                self.phase, self.anchor_s = "measuring", t
                self.recovered = self.fell
                return "anchor"
            if elapsed >= 5. - 1e-8:
                self.fail("post_push_stability_deadline")
                self.ended_s = t
        return "none"

    def finish(self, t):
        if self.phase != "measuring" or t - self.anchor_s < WINDOW_S - 1e-6:
            raise ValueError("A full uninterrupted window is required")
        self.phase, self.ended_s = "done", t


def make_cases(suite="all", *, hold_envs=32, envs_per_case=1):
    cases = []
    if suite in ("all", "hold"):
        cases += [dict(kind="hold", vector=[0., 0.], magnitude=0., direction=None,
                       replicate=i) for i in range(hold_envs * envs_per_case)]
    for kind in ("push", "knockdown"):
        if suite not in ("all", kind):
            continue
        magnitudes = (.3, .6) if kind == "push" else (12., 16.)
        for magnitude in magnitudes:
            for direction in range(8):
                angle = direction * math.pi / 4
                vector = np.array([math.cos(angle), math.sin(angle)])
                # Linear diagonal cases deliberately reach (+/-.6,+/-.6) m/s.
                if kind == "push":
                    vector /= np.max(np.abs(vector))
                for repeat in range(envs_per_case):
                    cases.append(dict(kind=kind, vector=(magnitude * vector).tolist(),
                                      magnitude=magnitude, direction=direction, replicate=repeat))
    return [{"case_id": i, **case} for i, case in enumerate(cases)]


def sample_physics(env):
    recorder = getattr(env, "_stand_acceptance_recorder", None)
    if recorder is not None:
        recorder.sample()
    import torch
    return torch.zeros(env.num_envs, device=env.device)


class PhysicsRecorder:
    """GPU recording plus independent per-window extrema at the real PD rate."""
    def __init__(self, env, maximum_steps):
        import torch
        self.env, self.count = env, 0
        robot = env.scene["robot"]
        ids, _ = robot.find_joints(MOTORS, preserve_order=True)
        self.qadr, self.vadr = robot.indexing.joint_q_adr[ids], robot.indexing.joint_v_adr[ids]
        self.root_adr = robot.indexing.free_joint_q_adr
        self.root_vadr = robot.indexing.free_joint_v_adr
        n, device = env.num_envs, env.device
        self.trace = torch.empty((maximum_steps, n, len(CHANNELS)), device=device)
        self.measuring = torch.zeros(n, dtype=torch.bool, device=device)
        self.anchor = torch.zeros(n, 2, device=device)
        self.anchor_q = torch.zeros(n, 6, device=device)
        self.anchor_qd = torch.zeros(n, 6, device=device)
        self.anchor_velocity_xy = torch.zeros(n, 2, device=device)
        self.max_leg_speed = torch.zeros(n, 4, device=device)
        self.max_body_speed = torch.zeros(n, device=device)
        self.samples = torch.zeros(n, dtype=torch.long, device=device)
        self.qmin = torch.full((n, 4), float("inf"), device=device)
        self.qmax = -self.qmin.clone()
        self.qd_square_sum = torch.zeros(n, 6, dtype=torch.float64, device=device)
        self.max_displacement = torch.zeros(n, device=device)
        self.max_tilt = self.max_displacement.clone()
        self.max_height_error = self.max_displacement.clone()
        self.support_count = torch.zeros(n, dtype=torch.long, device=device)
        self.base_count = self.support_count.clone()
        self.nonfinite = torch.zeros(n, dtype=torch.bool, device=device)
        self.window_nonfinite = self.nonfinite.clone()
        self.step_fallen = self.nonfinite.clone()
        self.start_index = [-1] * n
        self.end_index = [-1] * n

    def arm(self, index):
        if self.start_index[index] != -1:
            raise ValueError("A sample cannot be re-anchored")
        self.start_index[index] = self.count
        self.anchor[index] = self.env.sim.data.qpos[index, self.root_adr[:2]]
        self.anchor_q[index] = self.env.sim.data.qpos[index, self.qadr]
        self.anchor_qd[index] = self.env.sim.data.qvel[index, self.vadr]
        self.anchor_velocity_xy[index] = self.env.sim.data.qvel[index, self.root_vadr[:2]]
        self.max_leg_speed[index] = self.anchor_qd[index, LEG_INDICES].abs()
        self.max_body_speed[index] = self.anchor_velocity_xy[index].norm()
        self.qmin[index] = self.anchor_q[index, LEG_INDICES]
        self.qmax[index] = self.anchor_q[index, LEG_INDICES]
        self.measuring[index] = True

    def stop(self, index):
        self.measuring[index] = False
        if self.end_index[index] == -1:
            self.end_index[index] = self.count

    def sample(self):
        import torch
        env, data = self.env, self.env.sim.data
        q, qd = data.qpos[:, self.qadr], data.qvel[:, self.vadr]
        root = data.qpos[:, self.root_adr]
        # Free-joint translational qvel is world velocity; read the actual 2 ms state.
        velocity_xy = data.qvel[:, self.root_vadr[:2]]
        # Double precision avoids acos losing small angles close to upright.
        w, x, y, z = root[:, 3:].double().unbind(-1)
        gx, gy, gz = 2 * (w*y-x*z), -2 * (y*z+w*x), -1 + 2 * (x*x+y*y)
        tilt = torch.rad2deg(torch.stack((torch.atan2(-gy, -gz), torch.asin(gx.clamp(-1, 1))), -1))
        total_tilt = torch.rad2deg(torch.atan2(torch.sqrt(gx.square() + gy.square()), -gz))
        height = root[:, 2] - env.scene.env_origins[:, 2]
        wheel_up = -env.scene["wheel_contact"].data.force[:, :, 2]
        weight = env.sim.model.body_mass.sum(dim=-1) * np.linalg.norm(env.sim.mj_model.opt.gravity)
        support = (wheel_up > 5.).all(-1) & (wheel_up.sum(-1) >= .7 * weight)
        base = env.scene["base_contact"].data.force.norm(dim=-1).amax(-1) > 1.
        finite = torch.isfinite(data.qpos).all(-1) & torch.isfinite(data.qvel).all(-1)
        finite &= torch.isfinite(wheel_up).all(-1)
        row = self.trace[self.count]
        row[:, :6], row[:, 6:12], row[:, 12:14] = q, qd, root[:, :2]
        row[:, 14:16], row[:, 16] = tilt, height
        row[:, 17], row[:, 18], row[:, 19] = support, base, finite
        row[:, 20:22] = velocity_xy
        mask = self.measuring
        self.max_leg_speed = torch.where(mask[:, None],
            torch.maximum(self.max_leg_speed, qd[:, LEG_INDICES].abs()), self.max_leg_speed)
        self.max_body_speed = torch.where(mask,
            torch.maximum(self.max_body_speed, velocity_xy.norm(dim=-1)), self.max_body_speed)
        self.samples += mask.long()
        legs = q[:, LEG_INDICES]
        self.qmin = torch.where(mask[:, None], torch.minimum(self.qmin, legs), self.qmin)
        self.qmax = torch.where(mask[:, None], torch.maximum(self.qmax, legs), self.qmax)
        self.qd_square_sum += torch.where(mask[:, None], qd.double().square(), 0.)
        self.max_displacement = torch.where(mask, torch.maximum(self.max_displacement,
            (root[:, :2] - self.anchor).norm(dim=-1)), self.max_displacement)
        self.max_tilt = torch.where(mask, torch.maximum(self.max_tilt, total_tilt), self.max_tilt)
        self.max_height_error = torch.where(mask, torch.maximum(self.max_height_error,
            (height - HEIGHT_M).abs()), self.max_height_error)
        self.support_count += (mask & support).long()
        self.base_count += (mask & base).long()
        self.nonfinite |= ~finite
        self.window_nonfinite |= mask & ~finite
        self.step_fallen |= base | (tilt.abs().amax(-1) > 40.)
        self.count += 1

    def violations(self):
        return (self.window_nonfinite | (self.max_body_speed > BODY_SPEED_M_S + 1e-8)
                | (self.max_tilt > 25. + 1e-6) | (self.max_height_error > .03 + 1e-8)
                | (self.max_leg_speed.amax(-1) > LEG_SPEED_RAD_S + 1e-7))


def geometry_report(model, poses, times, states, body_ids):
    """Offline 20 Hz body geometry; never changes the measured 500 Hz leg range."""
    import mujoco
    data = mujoco.MjData(model)
    output = []
    for i, state in enumerate(states):
        values = []
        if state.anchor_s is not None:
            for pose, t in zip(poses[:, i], times, strict=True):
                if state.anchor_s - 1e-8 <= t <= (state.ended_s or t) + 1e-8:
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    rotation = data.xmat[body_ids[0]].reshape(3, 3)
                    delta = (data.xpos[body_ids[3:5]] - data.xpos[body_ids[1:3]]) @ rotation
                    length = np.linalg.norm(delta[:, (0, 2)], axis=-1)
                    angle = np.arctan2(delta[:, 0], -delta[:, 2])
                    values.append(np.r_[angle, length])
        a = np.asarray(values)
        output.append({"sample_hz": 20., "samples": len(values),
            "hip_wheel_angle_peak_to_peak_deg": np.rad2deg(np.ptp(a[:, :2], axis=0)).tolist() if len(a) else None,
            "hip_wheel_length_min_m": a[:, 2:].min(0).tolist() if len(a) else None,
            "hip_wheel_length_max_m": a[:, 2:].max(0).tolist() if len(a) else None})
    return output


def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def json_default(value):
    if isinstance(value, (Path, np.generic)):
        return str(value) if isinstance(value, Path) else value.item()
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def evaluate_batch(args, cases, profile, seed, batch_index, output_dir):
    import mujoco
    import torch
    from ct_mjlab.envs import task_registry
    from ct_mjlab.envs.chuanliantui_stand import mdp
    from ct_mjlab.envs.chuanliantui_stand.env_cfg import configure_evaluation
    from mjlab.managers.metrics_manager import MetricsTermCfg

    cfg, agent = task_registry.get_cfgs(TASK, play=profile == "nominal")
    configure_evaluation(cfg)
    removed = [name for name in cfg.events if name in (
        "stand_disturbance", "huananhu_velocity_push", "huananhu_local_wrench")]
    for name in removed:
        del cfg.events[name]
    cfg.commands["standup"].disturbances = False
    # The original fixed matrix remains near-standing, even when training mixes starts.
    start_mode = getattr(args, "start_mode", "near")
    cfg.commands["standup"].ground_start_probability = 1. if start_mode == "ground" else 0.
    cfg.auto_reset, cfg.seed = False, seed
    cfg.metrics["acceptance_physics"] = MetricsTermCfg(func=sample_physics, per_substep=True, reduce="last")
    batch_dir = output_dir / f"{profile}-seed{seed}-batch{batch_index:03d}"
    batch_dir.mkdir()
    wrapped, cfg = task_registry.make_env(TASK, env_cfg=cfg, train_cfg=agent,
        num_envs=len(cases), seed=seed, device=args.device)
    env = wrapped.unwrapped
    try:
        if not math.isclose(env.physics_dt, PHYSICS_DT) or not math.isclose(env.step_dt, POLICY_DT):
            raise ValueError("Assessment requires unchanged 500 Hz physics / 100 Hz policy")
        runner, _ = task_registry.make_alg_runner(wrapped, TASK, train_cfg=agent, device=args.device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        max_steps = round(35.1 / env.step_dt)
        recorder = PhysicsRecorder(env, max_steps * cfg.decimation)
        env._stand_acceptance_recorder = recorder
        states = [AcceptanceState(case["kind"]) for case in cases]
        obs, _ = env.reset(seed=seed)
        initial_qvel = env.sim.data.qvel.clone().cpu().numpy()
        pd = env.action_manager.get_term("pd")
        initial_delay_ticks = torch.stack((pd.action_delay_steps,
            pd.wheel_action_delay_steps), dim=1).cpu().numpy()
        initial_delay_policy_steps = initial_delay_ticks // cfg.decimation
        command = env.command_manager.get_term("standup")
        if not bool((command.ground_start == (start_mode == "ground")).all()):
            raise AssertionError("Assessment reset mode differs from its requested initial state")
        full_qpos = torch.empty((max_steps + 1, env.num_envs, env.sim.mj_model.nq), device=env.device)
        full_qpos[0] = env.sim.data.qpos
        reset_count = np.zeros(env.num_envs, dtype=np.int64)
        mujoco.mj_saveModel(env.sim.mj_model, str(batch_dir / "scene.mjb"), None)
        params = {"environment": asdict(cfg), "agent": asdict(agent), "removed_events": removed,
                  "scene_mjb_scope": "compiled scene; randomized model fields saved separately"}
        (batch_dir / "config.json").write_text(json.dumps(params, default=json_default, indent=2) + "\n")
        fields = ("body_mass", "body_inertia", "body_ipos", "geom_friction", "geom_solref",
                  "dof_frictionloss", "dof_damping")
        np.savez_compressed(batch_dir / "runtime_model.npz", **{
            name: getattr(env.sim.model, name).cpu().numpy() for name in fields})
        runtime_error, completed_steps = None, 0
        with torch.inference_mode():
            for step in range(max_steps):
                live = [i for i, s in enumerate(states) if s.phase not in ("failed", "done")]
                if not live:
                    break
                recorder.step_fallen.zero_()
                try:
                    wanted = torch.tensor([0., 0., 0., HEIGHT_M * 5], device=env.device)
                    torch.testing.assert_close(obs["actor"][live, :4], wanted.expand(len(live), -1), atol=1e-6, rtol=0.)
                    action = policy(obs)
                    if not torch.isfinite(action).all():
                        raise FloatingPointError("Non-finite policy output")
                    obs, _, terminated, timeout, _ = env.step(action)
                except (RuntimeError, AssertionError, FloatingPointError) as exc:
                    runtime_error = f"{type(exc).__name__}: {exc}"
                    for i in live:
                        states[i].fail("runtime_or_nonfinite_failure")
                        states[i].ended_s = recorder.count * PHYSICS_DT
                        recorder.stop(i)
                    break
                completed_steps, t = step + 1, (step + 1) * env.step_dt
                full_qpos[completed_steps] = env.sim.data.qpos
                stable = mdp.standing_mask(env) & ~command.recover & ~command.airborne
                packed = torch.stack((stable, command.fallen | recorder.step_fallen, command.recover | command.airborne,
                    terminated | timeout, recorder.nonfinite | command.numerical_failure,
                    recorder.violations()), dim=0).cpu().numpy()
                delivered_this_step = False
                for i in live:
                    s = states[i]
                    action = s.update(t, bool(packed[0, i]), bool(packed[1, i]),
                        terminal=bool(packed[3, i]), nonfinite=bool(packed[4, i]), unstable=bool(packed[2, i]))
                    if s.phase == "measuring" and packed[5, i]:
                        s.fail("physics_window_limit"); s.ended_s = t
                    if action == "deliver":
                        ids = torch.tensor([i], device=env.device)
                        velocity = torch.tensor([cases[i]["vector"]], device=env.device)
                        apply = mdp.apply_velocity_push if cases[i]["kind"] == "push" else mdp.apply_knockdown
                        apply(env, ids, velocity)
                        s.mark_delivered(t)
                        delivered_this_step = True
                    elif action == "anchor":
                        recorder.arm(i)
                    if s.phase == "measuring" and t - s.anchor_s >= WINDOW_S - 1e-8:
                        s.finish(t)
                    if s.phase in ("failed", "done"):
                        recorder.stop(i)
                # Terminated cases keep their failed grade; only their dead world is reset.
                reset_ids = (terminated | timeout).nonzero(as_tuple=False).flatten()
                if len(reset_ids):
                    ids_cpu = reset_ids.cpu().numpy()
                    reset_count[ids_cpu] += 1
                    env.reset(env_ids=reset_ids)
                if delivered_this_step:
                    # Refresh root-velocity derived values without advancing sensor histories.
                    env.sim.forward()
                    env.sim.sense()
                if len(reset_ids) or delivered_this_step:
                    obs = env.get_observations()
    except Exception:
        # Configuration/checkpoint errors should fail before producing a misleading assessment.
        raise
    finally:
        # The simulation objects are still readable after close; no subsequent stepping occurs.
        env.close()

    for i, state in enumerate(states):
        if state.phase not in ("done", "failed"):
            state.fail("assessment_time_limit")
            state.ended_s = completed_steps * POLICY_DT
        recorder.stop(i)
    raw = recorder.trace[:recorder.count].cpu().numpy()
    poses = full_qpos[:completed_steps + 1].cpu().numpy()
    anchors, anchor_q = recorder.anchor.cpu().numpy(), recorder.anchor_q.cpu().numpy()
    body_names = ("base_link", "lf0", "rf0", "lfwheel", "rfwheel")
    local_ids, _ = env.scene["robot"].find_bodies(body_names, preserve_order=True)
    body_ids = env.scene["robot"].indexing.body_ids[local_ids].cpu().numpy()
    geometry = geometry_report(env.sim.mj_model, poses[::5], np.arange(len(poses))[::5] * POLICY_DT, states, body_ids)
    np.savez_compressed(batch_dir / "trajectory.npz", physics=raw, initial_qvel=initial_qvel,
        initial_action_delay_physics_ticks=initial_delay_ticks,
        initial_action_delay_policy_steps=initial_delay_policy_steps,
        physics_time_s=(np.arange(len(raw)) + 1) * PHYSICS_DT,
        physics_channels=np.asarray(CHANNELS), qpos=poses,
        policy_time_s=np.arange(len(poses)) * POLICY_DT,
        anchor_xy=anchors, anchor_q=anchor_q,
        anchor_qd=recorder.anchor_qd.cpu().numpy(),
        anchor_velocity_xy=recorder.anchor_velocity_xy.cpu().numpy(),
        criteria_version=np.asarray(CRITERIA_VERSION), start_index=recorder.start_index, end_index=recorder.end_index)
    artifact_metadata = {name: identity(batch_dir / name) for name in
                         ("trajectory.npz", "scene.mjb", "runtime_model.npz", "config.json")}
    (batch_dir / "artifacts.json").write_text(json.dumps(artifact_metadata, indent=2) + "\n")
    rows = []
    for i, (case, state) in enumerate(zip(cases, states, strict=True)):
        start, end = recorder.start_index[i], recorder.end_index[i]
        window = raw[start:end, i] if start >= 0 else raw[:0, i]
        summary = summarize_window(dict(q=window[:, :6], qd=window[:, 6:12], xy=window[:, 12:14],
            tilt_deg=window[:, 14:16], height_m=window[:, 16], wheel_supported=window[:, 17],
            base_contact=window[:, 18], finite=window[:, 19], body_velocity_xy=window[:, 20:22],
            anchor_xy=anchors[i], anchor_q=anchor_q[i],
            anchor_qd=recorder.anchor_qd[i].cpu().numpy(),
            anchor_velocity_xy=recorder.anchor_velocity_xy[i].cpu().numpy()))
        # Cross-check raw saved samples against the independent 500 Hz accumulators.
        if len(window):
            np.testing.assert_allclose(summary["leg_max_abs_speed_rad_s"], recorder.max_leg_speed[i].cpu().numpy(), atol=1e-7, rtol=0.)
            np.testing.assert_allclose(summary["maximum_body_speed_m_s"], recorder.max_body_speed[i].cpu().numpy(), atol=1e-7, rtol=0.)
            np.testing.assert_allclose(summary["leg_peak_to_peak_rad"],
                (recorder.qmax[i] - recorder.qmin[i]).cpu().numpy(), atol=1e-7, rtol=0.)
            np.testing.assert_allclose(summary["qd_rms_rad_s"],
                torch.sqrt(recorder.qd_square_sum[i] / recorder.samples[i]).cpu().numpy(), atol=1e-7)
            np.testing.assert_allclose(summary["maximum_tilt_deg"],
                recorder.max_tilt[i].cpu().numpy(), atol=2e-4, rtol=0.)
        passed = state.phase == "done" and summary["passed"]
        rows.append({**case, "env": i, "profile": profile, "seed": seed,
            "batch": batch_index, "initial_state_mode": start_mode,
            "initial_action_delay_policy_steps": {
                "legs": int(initial_delay_policy_steps[i, 0]),
                "wheels": int(initial_delay_policy_steps[i, 1])},
            **asdict(state), "delivered": state.delivered_s is not None,
            "passed": passed, "window": summary, "geometry": geometry[i],
            "resets_after_result_frozen": int(reset_count[i]),
            "runtime_error": runtime_error,
            "anchor_xy_m": anchors[i].tolist() if start >= 0 else None,
            "trajectory": str(batch_dir / "trajectory.npz"), "scene": str(batch_dir / "scene.mjb"),
            "artifact_metadata": str(batch_dir / "artifacts.json")})
    (batch_dir / "episodes.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def check_arithmetic():
    q = np.zeros((10_000, 6))
    samples = dict(q=q, qd=q.copy(), xy=np.zeros((len(q), 2)), tilt_deg=np.zeros((len(q), 2)),
        body_velocity_xy=np.zeros((len(q), 2)), height_m=np.full(len(q), HEIGHT_M), wheel_supported=np.ones(len(q)),
        base_contact=np.zeros(len(q)), finite=np.ones(len(q), dtype=bool), anchor_xy=np.zeros(2))
    assert summarize_window(samples)["passed"]
    q[3, 0] = math.radians(10.)
    samples["xy"][:, 0] = 1.
    assert summarize_window(samples)["passed"]  # Angle range and displacement are diagnostic in v2.
    samples["qd"][3, 0] = .5001
    assert not summarize_window(samples)["passed"]
    samples["qd"][:] = 0.
    samples["body_velocity_xy"][3] = [.04, .04]
    assert not summarize_window(samples)["passed"]
    samples["body_velocity_xy"][:] = 0.
    samples["tilt_deg"][3] = [20., 20.]
    assert not summarize_window(samples)["passed"]  # Combined inclination is about 28 degrees.
    state = AcceptanceState("hold")
    for tick in range(1, 51):
        action = state.update(tick * .01, True, False)
    assert action == "anchor"
    state.update(.51, False, True)
    assert state.phase == "failed" and state.update(1., True, False) == "none"
    return {"scope": "CPU arithmetic only; no physics or training", "passed": True}


def acceptance_criteria():
    return {"target_command": [0., 0., HEIGHT_M], "qualification_continuous_s": .5,
            "initial_stable_deadline_s": 5., "delivery_window_s": [5., 10.],
            "post_push_stable_deadline_s": 5., "window_s": WINDOW_S,
            "version": CRITERIA_VERSION, "maximum_body_planar_speed_m_s": BODY_SPEED_M_S, "maximum_tilt_deg": 25., "height_error_m": .03,
            "tilt_definition": "angle between body +Z and world +Z; acos(cos(roll)*cos(pitch))",
            "wheel_support_fraction": .95, "base_contact_fraction": .01,
            "each_physical_leg_maximum_absolute_speed_rad_s": LEG_SPEED_RAD_S,
            "speed_capture": "raw500Hz qvel; each sample must satisfy both speed limits",
            "diagnostic_only": ["displacement_m", "leg_peak_to_peak_deg"],
            "anchor_rule": "one anchor per case after qualification; never re-anchor after measurement failure",
            "reset_rule": "terminated samples fail permanently before their worlds are reset for batch continuation",
            "recovery_rule": "a delivered impulse without an observed fall is not a recovery"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("hold", "push", "knockdown", "all"), default="all")
    parser.add_argument("--profile", choices=("nominal", "randomized", "both"), default="both")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--num-envs", type=int, default=32, help="Maximum simultaneous worlds per batch")
    parser.add_argument("--envs-per-case", type=int, default=1)
    parser.add_argument("--hold-envs", type=int, default=32, help="Standing cases per seed/profile")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix("").exists():
        parser.error("Output exists; choose a new path")
    if min(args.num_envs, args.envs_per_case, args.hold_envs) < 1:
        parser.error("Environment counts must be positive")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error("Provide nonempty, distinct seeds")
    if not args.check_only and (args.checkpoint is None or not args.checkpoint.is_file()):
        parser.error("--checkpoint must name an existing checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        result = check_arithmetic()
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result)); return
    from ct_mjlab.envs.chuanliantui.model import MODEL_XML
    profiles = ("nominal", "randomized") if args.profile == "both" else (args.profile,)
    output_dir = args.output.with_suffix("")
    output_dir.mkdir()
    cases = make_cases(args.suite, hold_envs=args.hold_envs, envs_per_case=args.envs_per_case)
    rows = []
    for profile in profiles:
        for seed in seeds:
            for offset in range(0, len(cases), args.num_envs):
                batch = evaluate_batch(args, cases[offset:offset + args.num_envs], profile, seed,
                                       offset // args.num_envs, output_dir)
                rows.extend(batch)
                print(json.dumps({"profile": profile, "seed": seed, "batch": offset // args.num_envs,
                                  "passed": sum(row["passed"] for row in batch), "cases": len(batch)}), flush=True)
    expected = len(cases) * len(profiles) * len(seeds)
    actual_recoveries = sum(row["fell"] and row["recovered"] and row["passed"] for row in rows)
    needs_recovery = args.suite in ("all", "knockdown")
    passed = len(rows) == expected and all(row["passed"] for row in rows) and (not needs_recovery or actual_recoveries > 0)
    sources = [Path(__file__), *sorted((ROOT / "mjlab_training/src/ct_mjlab").rglob("*.py"))]
    report = {"kind": "standing/push/recovery behavioral assessment; no learning or hardware",
        "task": TASK, "checkpoint": identity(args.checkpoint), "xml": identity(MODEL_XML),
        "sources": [identity(path) for path in sources], "command": sys.argv,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "physics_dt_s": PHYSICS_DT, "policy_dt_s": POLICY_DT,
        "physics_capture_hz": 500, "full_qpos_capture_hz": 100, "geometry_measurement_hz": 20,
        "initial_state_mode": "near",
        "motor_order": MOTORS, "measured_leg_indices": LEG_INDICES,
        "suite": args.suite, "profiles": profiles, "seeds": seeds,
        "expected_cases": expected, "completed_cases": len(rows), "passed_cases": sum(r["passed"] for r in rows),
        "actual_fall_recoveries": actual_recoveries, "passed": passed,
        "selected_matrix_passed": passed,
        "full_acceptance_passed": (passed and args.suite == "all" and args.profile == "both"
            and args.hold_envs * args.envs_per_case >= 32 and {42, 43, 44, 45}.issubset(seeds)),
        "full_acceptance_requirements": {
            "suite": "all", "profiles": ["nominal", "randomized"], "seeds": [42, 43, 44, 45],
            "minimum_hold_cases_per_seed_profile": 32,
            "note": "Separate 42/43 and 44/45 runs require an explicit combined review; a reduced matrix is diagnostic."},
        "criteria_version": CRITERIA_VERSION, "criteria": acceptance_criteria(),
        "episodes": rows}
    args.output.write_text(json.dumps(report, default=json_default, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "passed": passed,
                      "passed_cases": report["passed_cases"], "expected_cases": expected,
                      "actual_fall_recoveries": actual_recoveries}), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
