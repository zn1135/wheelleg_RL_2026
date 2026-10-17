#!/usr/bin/env python3
"""Check the 39-D multimode CPU sim2sim entry against the training semantics it replays.

Scope (no learning, no long simulation, no GPU):
  robot    the entry's compiled model has the same XML qpos0 as the training model (the
           two asset files differ only in the meshdir path), and its closed-stance solve
           equals ct_mjlab model.solve_standing_joint_positions on the training model
  centres  observation leg slots and the leg PD target centre on that solved stance
           (q_nominal, exactly pd_action._nominal_pos = huananhu_env_cfg's
           nominal_leg_positions), while the soft-limit clamp stays centred on XML qpos0
           (pd_action._reference_pos); a one-step torque probe with q == q_nominal and a
           zero action must give zero leg torque, which fails under either single-centre
           scheme (this is the 2026-10-17 standing leg-swing regression)
  gains    leg target scale and stiffness/damping, wheel velocity target clip(10*a,
           +-150) and the huananhu_env_cfg wheel gains, and the effort clamps 40 / 3.9
           (mirroring ChuanliantuiPdAction._effort_limits) are asserted against a torque
           probe with the values read from the training config class
  spring   entry gas_spring_force equals ct_mjlab gas_spring.gas_spring_force on a grid
           with identical constants
  tail     action order and observation permutation match v14_observations, a normal
           start reports mode NORMAL / flags [0,1,0,0], and the report tolerances equal
           the single source of truth in scripts/agent/assess_multimode_tracking.py
  options  simulator options (timestep/integrator/iterations/ls_iterations/tolerance/
           cone/impratio) equal the multimodal env's MujocoCfg; the vx/yaw command slots
           clamp to +-100 like v14_observations.basic; the ground-start activation flag
           flips on the 11th completed step (the training float32 boundary, not the 10th)

Static + one-step consistency between two implementations of one contract. It is not a
behavior acceptance and says nothing about policy quality.

Run from the repo root:
  mjlab_training/.venv/bin/python scripts/agent/check_multimode_cpu_sim2sim.py [--out <json>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim2sim"))
sys.path.insert(0, str(ROOT / "mjlab_training" / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "agent"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="optional JSON evidence path")
    args = parser.parse_args()

    import mujoco
    import mj_sim2sim_huananhu_multimode as entry
    from ct_mjlab.envs.chuanliantui import gas_spring, model, v14_observations
    from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdActionCfg
    import assess_multimode_tracking as assess

    results = {}
    cfg = HuananhuPdActionCfg(entity_name="robot")

    # 1. Same robot and same solved closed stance on both sides.
    training = model.load_model()
    training_stance = model.solve_standing_joint_positions(training)
    env = entry.Playback()
    entry_stance = entry.solve_standing_joint_positions(env.model)
    qpos0_gap = max(
        abs(float(training.qpos0[training.joint(name).qposadr[0]])
            - float(env.model.qpos0[env.model.joint(name).qposadr[0]]))
        for name in entry.POSE_JOINTS)
    stance_gap = max(abs(training_stance[name] - entry_stance[name]) for name in entry.POSE_JOINTS)
    assert qpos0_gap < 1e-9, f"qpos0 differs between the two XMLs: {qpos0_gap}"
    assert stance_gap < 1e-6, f"closed-stance solve differs: {stance_gap}"
    results["qpos0_max_abs_gap_rad"] = qpos0_gap
    results["stance_max_abs_gap_rad"] = stance_gap

    # 2. Centres: q_nominal for the observation and the leg target, XML qpos0 for the clamp.
    nominal = np.array([training_stance[name] for name in entry.MOTORS], dtype=np.float64)
    reference = np.array([float(env.model.qpos0[env.model.joint(name).qposadr[0]])
                          for name in entry.MOTORS])
    assert np.allclose(env.nominal, nominal, atol=1e-9), "entry.nominal is not the solved stance"
    assert np.allclose(env.reference, reference, atol=1e-9), "entry.reference is not XML qpos0"
    env.data.qpos[env.q] = env.nominal
    env.data.qvel[env.v] = 0.0
    mujoco.mj_forward(env.model, env.data)
    obs = env.observation()
    assert np.allclose(obs[10:16], 0.0, atol=1e-6), f"leg observation slots nonzero at q_nominal: {obs[10:16]}"
    torque, _ = env.controls(np.zeros(6))
    assert np.allclose(torque, 0.0, atol=1e-9), f"zero torque expected at q_nominal with zero action: {torque}"
    results["observation_leg_slots_at_nominal"] = [float(v) for v in obs[10:16]]
    results["torque_at_nominal_zero_action_nm"] = [float(v) for v in torque]

    # 3. Gains, scales and effort clamps (training literal 40 / 3.9, pd_action.py).
    action = np.zeros(6)
    action[0] = 1.0
    torque, _ = env.controls(action)
    leg_expected = cfg.leg_position_scale * cfg.leg_stiffness  # q == q_nominal, qd == 0
    assert abs(torque[0] - leg_expected) < 1e-9, f"leg probe {torque[0]} != {leg_expected}"
    results["leg_probe_nm"] = {"measured": float(torque[0]), "expected": float(leg_expected)}
    action[0] = 100.0
    torque, _ = env.controls(action)
    assert abs(torque[0] - 40.0) < 1e-9, f"leg effort clamp {torque[0]}"
    results["leg_clamp_nm"] = float(torque[0])
    action[:] = 0.0
    action[2] = 1.0
    torque, _ = env.controls(action)
    wheel_expected = cfg.wheel_stiffness * cfg.wheel_velocity_scale  # qd == 0
    assert abs(torque[2] - wheel_expected) < 1e-9, f"wheel probe {torque[2]} != {wheel_expected}"
    results["wheel_probe_nm"] = {"measured": float(torque[2]), "expected": float(wheel_expected)}
    action[2] = 100.0
    torque, _ = env.controls(action)
    assert abs(torque[2] - 3.9) < 1e-9, f"wheel effort clamp {torque[2]}"
    results["wheel_clamp_nm"] = float(torque[2])

    # 4. Gas-spring parity on a deterministic grid (the entry inlines the 90 mm stroke).
    positions = np.linspace(-0.02, 0.09, 41)
    velocities = np.linspace(-1.0, 1.0, 9)
    p, v = np.meshgrid(positions, velocities, indexing="ij")
    ours = np.asarray(entry.gas_spring_force(p, v), dtype=np.float64)
    theirs = np.asarray(gas_spring.gas_spring_force(p, v), dtype=np.float64)
    spring_gap = float(np.abs(ours - theirs).max())
    assert spring_gap < 1e-12, f"gas spring functions differ: {spring_gap}"
    for name in ("SPRING_INITIAL_COMPRESSION", "SPRING_DAMPING", "SPRING_FORCE_LIMIT"):
        assert getattr(entry, name) == getattr(gas_spring, name), f"{name} differs"
    results["gas_spring_max_abs_gap_n"] = spring_gap

    # 5. Order, task tail, and the tolerances the report shares with the assessment.
    assert tuple(entry.MOTORS) == tuple(model.ACTUATED_JOINTS), "motor order differs"
    assert tuple(int(v) for v in entry.OBS_ORDER) == tuple(v14_observations.ACTION_ORDER), \
        "observation permutation differs"
    mode, flags = env.mode_and_flags()
    assert np.array_equal(mode, entry.NORMAL_MODE), f"normal-start mode {mode}"
    assert np.array_equal(flags, np.array((0., 1., 0., 0.), dtype=np.float32)), f"normal-start flags {flags}"
    for entry_name, assess_name in (("VX_TOL", "VX_TOL"), ("YAW_TOL", "YAW_TOL"),
                                    ("HEIGHT_TOL", "HEIGHT_TOL"), ("TILT_TOL_DEG", "TILT_TOL_DEG"),
                                    ("TRACKING_SETTLE_S", "SETTLE_S")):
        assert getattr(entry, entry_name) == getattr(assess, assess_name), \
            f"{entry_name} != assess.{assess_name}"
    results["normal_start_mode"] = [float(v) for v in mode]
    results["normal_start_flags"] = [float(v) for v in flags]
    results["shared_tolerances"] = {"vx": entry.VX_TOL, "yaw": entry.YAW_TOL,
                                    "height": entry.HEIGHT_TOL, "tilt_deg": entry.TILT_TOL_DEG,
                                    "settle_s": entry.TRACKING_SETTLE_S}

    # 6. Command-slot clamps and the ground-start activation boundary match training.
    env.command[:] = (200., -200., 5.)
    try:
        clipped = env.observation()
        assert clipped[0] == 100. and clipped[2] == -100., \
            f"vx/yaw command slots not clamped: {clipped[:4]}"
        assert clipped[3] == 5., f"height command slot not clamped to 1 x 5: {clipped[3]}"
        results["command_clamp_probe"] = {"vx": float(clipped[0]), "yaw": float(clipped[2]),
                                          "height_x5": float(clipped[3])}
    finally:
        env.command[:] = (0., 0., .30)
    ground_env = entry.Playback("ground")
    ground_env.steps = 10
    at_ten = float(ground_env.mode_and_flags()[1][1])
    ground_env.steps = 11
    at_eleven = float(ground_env.mode_and_flags()[1][1])
    assert (at_ten, at_eleven) == (0., 1.), \
        f"ground policy_active boundary: 10 steps -> {at_ten}, 11 steps -> {at_eleven}"
    results["ground_policy_active_steps"] = {"after_10": at_ten, "after_11": at_eleven}

    # 7. Simulator options equal the multimodal env's MujocoCfg (the training settings).
    from ct_mjlab.envs.chuanliantui.multimode_env_cfg import make_multimode_env_cfg
    mujoco_cfg = make_multimode_env_cfg(play=True).sim.mujoco
    integrators = {"euler": mujoco.mjtIntegrator.mjINT_EULER,
                   "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                   "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST}
    cones = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
             "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}
    option_pairs = {
        "timestep": float(env.model.opt.timestep) == float(mujoco_cfg.timestep),
        "integrator": int(env.model.opt.integrator) == int(integrators[str(mujoco_cfg.integrator)]),
        "iterations": int(env.model.opt.iterations) == int(mujoco_cfg.iterations),
        "ls_iterations": int(env.model.opt.ls_iterations) == int(mujoco_cfg.ls_iterations),
        "tolerance": float(env.model.opt.tolerance) == float(mujoco_cfg.tolerance),
        "cone": int(env.model.opt.cone) == int(cones[str(mujoco_cfg.cone)]),
        "impratio": float(env.model.opt.impratio) == float(mujoco_cfg.impratio),
    }
    assert env.model.opt.solver == training.opt.solver, "solver differs from the training model default"
    results["simulator_options"] = {name: bool(ok) for name, ok in option_pairs.items()}
    failures = sorted(name for name, ok in option_pairs.items() if not ok)
    assert not failures, f"simulator options differ from the training MujocoCfg: {failures}"

    summary = {"ok": True, "script": str(Path(__file__).resolve()),
               "scope": "static + one-step entry-vs-training consistency; not a behavior acceptance",
               "results": results}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
