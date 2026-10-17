#!/usr/bin/env python3
"""Measure standing-still leg motion of a 39-D multimode checkpoint in two backends.

Scope: the 2026-10-17 standing leg-swing investigation and its regression evidence.
Two backends, one reduction:

  cpu    the independent CPU MuJoCo entry (sim2sim/mj_sim2sim_huananhu_multimode.py,
         Playback class): deterministic normal start, zero-velocity command.
  mjlab  the same training task in the play configuration (no domain randomization,
         no observation noise or delay) on the GPU/Warp backend. `--reset stance`
         reproduces the CPU entry's normal start exactly (init stance at 0.38 m,
         level, zero velocity); `--reset bank` uses the training pose-bank normal
         branch for a distributional companion run. Commands are pinned constant.

Both backends log the six motor joints, their velocities, the raw action, the
observation leg slots, base height, tilt and base contact at the 50 Hz control
rate, and reduce the window t >= --settle with the same code, so the rows are
directly comparable.

Reported per motor joint (physical order lf0, lf00, lfwheel, rf0, rf00, rfwheel):
q mean / peak-to-peak (rad), qd RMS (rad/s), action mean / peak-to-peak,
dominant oscillation frequency (detrended FFT peak in 0.1-10 Hz) of q and of the
action, half the action pk-pk as the target swing (rad; the 0.5 scale and the
+-3.14 clamp make this the leg target pk-pk while the clamp is inactive), plus
mean/pk-pk of observation slots 10:16 (q minus the leg nominal in training
semantics; a nonzero mean is a static reference bias) and base height / tilt /
base contact. This is a diagnostic measurement, not an acceptance: it defines no
pass threshold.

Usage (repo root, mjlab venv for both backends):
  mjlab_training/.venv/bin/python scripts/agent/measure_multimode_leg_motion.py \
      --backend cpu --checkpoint <model_*.pt> --output artifacts/<new>/leg-motion-cpu.json
  mjlab_training/.venv/bin/python scripts/agent/measure_multimode_leg_motion.py \
      --backend mjlab --checkpoint <model_*.pt> --device cuda:0 --output <...>.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_DT = 0.02
MOTORS = ("lf0", "lf00", "lfwheel", "rf0", "rf00", "rfwheel")
LEG_MOTORS = ("lf0", "lf00", "rf0", "rf00")
OBS_LEG_SLOTS = slice(10, 16)
OBS_LEG_ORDER = ("lf0", "lf00", "rf0", "rf00", "lfwheel", "rfwheel")
TASK = "chuanliantui"
BASE_CONTACT_N = 1.0
JUMP_COOLDOWN_PIN = 1.0e9


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def dominant_frequency(values: np.ndarray, dt: float, band=(0.1, 10.0)):
    x = np.asarray(values, dtype=np.float64)
    if x.size < 8:
        return None
    x = x - x.mean()
    freqs = np.fft.rfftfreq(x.size, dt)
    spectrum = np.abs(np.fft.rfft(x))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return None
    return float(freqs[mask][np.argmax(spectrum[mask])])


def summarize(t, q, qd, action, obs_leg, height, tilt_deg, base_contact, settle_s, control_dt):
    t = np.asarray(t, dtype=np.float64)
    mask = t >= settle_s
    out = {"window_s": [float(settle_s), float(t[-1]) if t.size else None],
           "samples": int(mask.sum()), "dt_s": control_dt, "joints": {}}
    for index, name in enumerate(MOTORS):
        col_q = np.asarray(q)[:, index][mask]
        col_qd = np.asarray(qd)[:, index][mask]
        col_a = np.asarray(action)[:, index][mask]
        out["joints"][name] = {
            "q_mean_rad": float(col_q.mean()),
            "q_pk_pk_rad": float(col_q.max() - col_q.min()),
            "qd_rms_rad_s": float(np.sqrt(np.mean(col_qd ** 2))),
            "action_mean": float(col_a.mean()),
            "action_pk_pk": float(col_a.max() - col_a.min()),
            "action_half_pk_pk_rad": float(.5 * (col_a.max() - col_a.min())),
            "q_dom_freq_hz": dominant_frequency(col_q, control_dt),
            "action_dom_freq_hz": dominant_frequency(col_a, control_dt),
        }
    obs_leg = np.asarray(obs_leg)[mask]
    out["obs_leg_slots_10_16"] = {
        "order": list(OBS_LEG_ORDER),
        "mean": [float(v) for v in obs_leg.mean(axis=0)],
        "pk_pk": [float(v) for v in (obs_leg.max(axis=0) - obs_leg.min(axis=0))],
    }
    h = np.asarray(height)[mask]
    out["base_height_m"] = {"mean": float(h.mean()), "min": float(h.min()), "max": float(h.max())}
    out["tilt_max_deg"] = float(np.asarray(tilt_deg)[mask].max())
    out["base_contact_count"] = int(np.asarray(base_contact)[mask].sum())
    return out


def print_summary(label, summaries, settle_s):
    for index, summary in enumerate(summaries):
        print(f"--- {label} env {index} | window t>={settle_s}s | samples {summary['samples']}")
        for name in MOTORS:
            line = summary["joints"][name]
            print(f"  {name:8s} q pk_pk {line['q_pk_pk_rad']:.4f} rad | q mean {line['q_mean_rad']:+.4f} | "
                  f"qd rms {line['qd_rms_rad_s']:.4f} | a mean {line['action_mean']:+.4f} "
                  f"a pk_pk {line['action_pk_pk']:.4f} | q f {line['q_dom_freq_hz']} | "
                  f"a f {line['action_dom_freq_hz']}")
        print(f"  obs legs mean {['%.4f' % v for v in summary['obs_leg_slots_10_16']['mean']]} "
              f"pk_pk {['%.4f' % v for v in summary['obs_leg_slots_10_16']['pk_pk']]}")
        print(f"  height {summary['base_height_m']} tilt_max {summary['tilt_max_deg']:.3f} deg "
              f"base_contact {summary['base_contact_count']}")


def _run_cpu(args):
    import torch

    sys.path.insert(0, str(REPO_ROOT / "sim2sim"))
    import mj_sim2sim_huananhu_multimode as entry

    torch.set_num_threads(1)
    actor = entry.load_actor(args.checkpoint, args.expect_sha256)
    env = entry.Playback("normal", (args.vx, args.yaw, args.height))

    def controller(obs):
        with torch.inference_mode():
            return actor(torch.from_numpy(obs)[None]).numpy()[0]

    control_dt = entry.DT * entry.DECIMATION
    total = int(round(args.seconds / control_dt))
    t, q, qd, action, obs_leg, height, tilt, contact = [], [], [], [], [], [], [], []
    for step in range(total):
        applied, _ = env.step(controller)
        t.append((step + 1) * control_dt)
        q.append(env.data.qpos[env.q].copy()[None, :])
        qd.append(env.data.qvel[env.v].copy()[None, :])
        action.append(np.asarray(applied, dtype=np.float64)[None, :])
        obs_leg.append(np.asarray(env.observation()[OBS_LEG_SLOTS], dtype=np.float64)[None, :])
        height.append(np.array([float(env.data.qpos[2])]))
        row = env.metrics()
        tilt.append(np.array([float(max(abs(v) for v in row["roll_pitch_deg"]))]))
        contact.append(np.array([bool(row["base_contact"])]))
    arrays = {"t": t, "q": q, "qd": qd, "action": action, "obs_leg": obs_leg,
              "height": height, "tilt": tilt, "contact": contact}
    meta = {"initial_pose": "normal", "command": [args.vx, args.yaw, args.height],
            "sim2sim_entry_sha256": sha256(REPO_ROOT / "sim2sim" / "mj_sim2sim_huananhu_multimode.py"),
            "controller": "pytorch-actor"}
    return arrays, control_dt, meta


def _pin_commands(state, command, env_ids=None):
    """Copied from scripts/agent/assess_multimode_tracking.py: pin the fixed command and
    make every later resample produce it too, with jump requests suppressed."""
    ids = slice(None) if env_ids is None else env_ids

    def pinned_resample(resample_ids):
        state.sampled[resample_ids] = command[resample_ids]
        state.standing[resample_ids] = False
        state.heading[resample_ids] = False
        state.special_mode[resample_ids] = -1
        state.ground_start[resample_ids] = False
        state.ground_command[resample_ids] = 0.
        state.jump_cooldown[resample_ids] = JUMP_COOLDOWN_PIN

    state._resample_command = pinned_resample
    pinned_resample(ids)
    state._command[ids] = command[ids]


def _stance_reset(env, env_ids):
    """Reproduce the CPU entry's normal start on the training backend: init stance
    (default_joint_pos = solved closed pose) at 0.38 m, level, zero velocity."""
    import torch

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    state = env.command_manager.get_term("standup")
    state.ground_start[env_ids] = False
    robot = env.scene["robot"]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] = env.scene.env_origins[env_ids]
    root_state[:, 2] += 0.38
    root_state[:, 3] = 1.
    root_state[:, 4:7] = 0.
    root_state[:, 7:13] = 0.
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)


def _bank_reset(env, env_ids):
    """Training's own normal branch: the V14 pose bank, for the distributional run."""
    import torch

    from ct_mjlab.envs.chuanliantui.huananhu_poses import sample_reset_state

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    state = env.command_manager.get_term("standup")
    ground = torch.zeros(len(env_ids), device=env.device, dtype=torch.bool)
    state.ground_start[env_ids] = ground
    sample_reset_state(env, env_ids, ground)
    if hasattr(env, "_huananhu_rewards"):
        env._huananhu_rewards.step = -1


def _run_mjlab(args):
    sys.path.insert(0, str(REPO_ROOT / "mjlab_training" / "src"))
    import torch

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from ct_mjlab.envs import task_registry

    configure_torch_backends()
    reset_func = {"stance": _stance_reset, "bank": _bank_reset}[args.reset]
    cfg, agent = task_registry.get_cfgs(TASK, play=True)
    cfg.scene.num_envs = args.envs
    cfg.seed = args.seed
    cfg.auto_reset = False
    cfg.curriculum = {}
    cfg.commands["standup"].resampling_time_range = (1e9, 1e9)
    cfg.events["reset_pose"] = EventTermCfg(func=reset_func, mode="reset")
    env = ManagerBasedRlEnv(cfg, device=args.device)
    ended_step = None
    try:
        runner = task_registry._task(TASK).runner_class(
            RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions), asdict(agent), device=args.device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        command = torch.tensor([[args.vx, args.yaw, args.height]] * args.envs,
                               device=env.device, dtype=torch.float32)
        obs, _ = env.reset(seed=args.seed)
        state = env.command_manager.get_term("standup")
        _pin_commands(state, command)
        obs, _, _, _, _ = env.step(torch.zeros((args.envs, 6), device=env.device))
        robot = env.scene["robot"]
        ids, order = robot.find_joints(MOTORS, preserve_order=True)
        if tuple(order) != MOTORS:
            raise RuntimeError(f"unexpected motor order {order}")
        joint_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        q_adr = robot.indexing.joint_q_adr[joint_ids]
        v_adr = robot.indexing.joint_v_adr[joint_ids]
        control_dt = float(env.step_dt)
        total = int(round(args.seconds / control_dt))
        t, q, qd, action, obs_leg, height, tilt, contact = [], [], [], [], [], [], [], []
        with torch.inference_mode():
            for step in range(total):
                _pin_commands(state, command)
                obs, _, terminated, timeout, _ = env.step(policy(obs))
                t.append((step + 1) * control_dt)
                q.append(env.sim.data.qpos[:, q_adr].detach().cpu().numpy().copy())
                qd.append(env.sim.data.qvel[:, v_adr].detach().cpu().numpy().copy())
                action.append(env.action_manager.get_term("pd").raw_action.detach().cpu().numpy().copy())
                obs_leg.append(obs["actor"][:, OBS_LEG_SLOTS].detach().cpu().numpy().copy())
                height.append(robot.data.root_link_pos_w[:, 2].detach().cpu().numpy().copy())
                gravity = robot.data.projected_gravity_b
                roll = torch.atan2(-gravity[:, 1], -gravity[:, 2]).abs()
                pitch = torch.asin(gravity[:, 0].clamp(-1, 1)).abs()
                tilt.append(torch.rad2deg(torch.maximum(roll, pitch)).detach().cpu().numpy().copy())
                force = env.scene["base_contact"].data.force_history
                contact.append((force.norm(dim=-1).amax(dim=(1, 2)) > BASE_CONTACT_N)
                               .detach().cpu().numpy().copy())
                if bool((terminated | timeout).any()):
                    ended_step = step + 1
                    break
    finally:
        env.close()
    arrays = {"t": t, "q": q, "qd": qd, "action": action, "obs_leg": obs_leg,
              "height": height, "tilt": tilt, "contact": contact}
    meta = {"reset": args.reset, "seed": args.seed, "envs": args.envs, "device": args.device,
            "command": [args.vx, args.yaw, args.height], "ended_step": ended_step,
            "checkpoint_sha256": sha256(args.checkpoint)}
    return arrays, control_dt, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("cpu", "mjlab"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--settle", type=float, default=2.)
    parser.add_argument("--vx", type=float, default=0.)
    parser.add_argument("--yaw", type=float, default=0.)
    parser.add_argument("--height", type=float, default=.30)
    parser.add_argument("--expect-sha256", default=None)
    parser.add_argument("--reset", choices=("stance", "bank"), default="stance",
                        help="mjlab backend start pose: CPU entry stance or training pose bank")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.seconds <= args.settle:
        raise SystemExit("--seconds must exceed --settle")

    if args.backend == "cpu":
        arrays, control_dt, meta = _run_cpu(args)
    else:
        arrays, control_dt, meta = _run_mjlab(args)

    envs = int(np.asarray(arrays["height"]).shape[1])
    summaries = []
    for index in range(envs):
        summaries.append(summarize(
            arrays["t"],
            np.asarray(arrays["q"])[:, index],
            np.asarray(arrays["qd"])[:, index],
            np.asarray(arrays["action"])[:, index],
            np.asarray(arrays["obs_leg"])[:, index],
            np.asarray(arrays["height"])[:, index],
            np.asarray(arrays["tilt"])[:, index],
            np.asarray(arrays["contact"])[:, index],
            args.settle, control_dt))
    print_summary(f"{args.backend} {args.checkpoint.name}", summaries, args.settle)

    report = {
        "kind": "standing-still leg-motion measurement; diagnostic, defines no pass threshold",
        "backend": args.backend, "task": TASK if args.backend == "mjlab" else "CPU sim2sim entry",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
        "git_head": git_head(),
        "args": {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items()},
        "control_dt_s": control_dt, **meta,
        "envs": summaries,
        "scope": "one checkpoint, one backend, one constant command; comparison unit for the "
                 "standing leg-swing investigation, not a training acceptance",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
