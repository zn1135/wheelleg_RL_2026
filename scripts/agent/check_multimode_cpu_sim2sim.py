#!/usr/bin/env python3
"""Check the 39-D multimode CPU sim2sim entry against the training semantics it replays.

Scope (no learning, no long simulation, no GPU):
  robot    training, native playback and real2sim share one XML, the legacy XML path
           links to it, and the entry's closed-stance solve
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
           start reports mode NORMAL / flags [0,1,0,0] at the reset instant, and the report tolerances equal
           the single source of truth in scripts/agent/assess_multimode_tracking.py
  options  simulator options (timestep/integrator/iterations/ls_iterations/tolerance/
           cone/impratio) equal the multimodal env's MujocoCfg; the vx/yaw command slots
           clamp to +-100 like v14_observations.basic; the ground-start activation flag
           flips on the 11th completed step (the training float32 boundary, not the 10th)
  airborne the airborne flag (slot 37) mirrors multimode_mdp.MultimodeCommand.advance:
           both wheel centers clear of the floor by more than WHEEL_RADIUS + 2 cm with
           peak wheel contact force below 5 N for one control period; asserted to flip
           after the first completed step from the 0.38 m stance and to clear after the
           wheels land (the 2026-10-17 second-audit observation fix)
  contacts every robot geom carries the training scene's priority=1 patch, so robot-floor
           contact friction equals the training value (0.5 sliding), not the equal-priority
           elementwise-max mix with the floor's 1.0 (the 2026-10-17 audit finding)
  recovery CPU replay intentionally uses body-frame mean L*cos(theta) for recovery
           height, matching H7's quantity rather than training world base height;
           check translation/rotation invariance, split legs and the actual mode gate

Static consistency and limited-step probes, including the CPU-only recovery-height
adaptation. This is not a behavior acceptance and says nothing about policy quality.

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
    import real2sim_huananhu as mirror
    from ct_mjlab.envs.chuanliantui import gas_spring, model, v14_observations
    from ct_mjlab.envs.chuanliantui.huananhu_action import HuananhuPdActionCfg
    import assess_multimode_tracking as assess

    results = {}
    cfg = HuananhuPdActionCfg(entity_name="robot")

    # A single editable asset prevents training/playback model copies from drifting.
    legacy = ROOT / "resources/robots/chuanliantui_new_1/multimode_model3023/chuanliantui.xml"
    assert entry.MODEL_ASSET.samefile(model.MODEL_XML)
    assert mirror.MODEL.samefile(model.MODEL_XML)
    assert legacy.is_symlink() and legacy.samefile(model.MODEL_XML)
    assert entry.POSE_BANK.parent == legacy.parent, "versioned pose bank moved with the XML"
    results["shared_xml"] = str(model.MODEL_XML.relative_to(ROOT))
    results["legacy_xml_symlink"] = str(legacy.readlink())

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

    # Optional real2sim compensation must preserve leg control, wheel clamps and hold.
    coefficients = np.array((.12, .06))
    compensated = entry.Playback(wheel_friction_compensation=coefficients)
    compensation_error = 0.
    for speeds in ((0., 0.), (.01, -.01), (2., -2.), (-2., 2.), (80., -80.)):
        env.data.qvel[env.v[entry.WHEELS]] = speeds
        compensated.data.qpos[:] = env.data.qpos
        compensated.data.qvel[:] = env.data.qvel
        action = np.array((.2, -.2, .3, -.2, .2, -.3))
        baseline, _ = env.controls(action)
        actual, _ = compensated.controls(action)
        expected = np.clip(.1*(10*action[entry.WHEELS]-speeds) +
                           coefficients*np.tanh(np.asarray(speeds)/.5), -3.9, 3.9)
        compensation_error = max(compensation_error, float(np.max(abs(actual[entry.WHEELS]-expected))))
        assert np.array_equal(actual[entry.LEGS], baseline[entry.LEGS]), 'compensation changed leg control'
        assert np.all(abs(actual[entry.WHEELS]) <= 3.9), 'compensation bypassed wheel effort clamp'
    assert compensation_error < 1e-12
    compensated.ground_start, compensated.steps = True, 0
    assert np.all(compensated.controls(action)[0] == 0), 'compensation bypassed ground-start hold'
    env.data.qvel[env.v] = 0.
    results['optional_wheel_friction_compensation'] = {
        'max_formula_difference': compensation_error, 'leg_control_preserved': True,
        'zero_crossing_sign_clamp_and_ground_hold_checked': True, 'default_disabled': True}

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

    # 5b. The airborne flag (slot 37) is not pinned to 0: from the 0.38 m stance the
    #     wheels leave the floor, so after one full control period it must read 1
    #     (as the training backend does), and it must clear once the wheels land.
    airborne_env = entry.Playback()
    airborne_env.step(lambda obs: np.zeros(6))
    flags_airborne = airborne_env.mode_and_flags()[1]
    assert np.array_equal(flags_airborne, np.array((0., 1., 1., 0.), dtype=np.float32)), \
        f"airborne after one control step: {flags_airborne}"
    for _ in range(40):
        airborne_env.step(lambda obs: np.zeros(6))
    flags_landed = airborne_env.mode_and_flags()[1]
    assert np.array_equal(flags_landed, np.array((0., 1., 0., 0.), dtype=np.float32)), \
        f"airborne after landing: {flags_landed}"
    results["airborne_flag"] = {"reset": [float(v) for v in flags],
                                "after_one_step": [float(v) for v in flags_airborne],
                                "after_landing": [float(v) for v in flags_landed]}
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
    multimode_cfg = make_multimode_env_cfg(play=True)
    mujoco_cfg = multimode_cfg.sim.mujoco
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

    # 8. Contact parameters mirror the training scene's robot-geom priority patch
    #    (huananhu_env_cfg: GeomCfg(geom_names_expr=('.*',), priority=1)): robot-floor
    #    contacts take the robot's own friction, not the equal-priority max with the
    #    floor's 1.0. This is the assertion that would have caught the 2026-10-17
    #    audit finding (entry ran at twice the training sliding friction).
    train_priority = int(multimode_cfg.scene.entities["robot"].geoms[0].priority)
    train_terrain_friction = tuple(float(v) for v in multimode_cfg.scene.terrain.geoms[0].friction)
    floor_id = env.model.geom("sim2sim_flat_floor").id
    assert int(env.model.geom_priority[floor_id]) == 0, "floor geom priority is not 0"
    robot_priorities = {int(env.model.geom_priority[i]) for i in range(env.model.ngeom)
                        if i != floor_id}
    assert robot_priorities == {train_priority}, \
        f"robot geom priority {sorted(robot_priorities)} != training {train_priority}"
    assert np.allclose(env.model.geom_friction[floor_id], train_terrain_friction), \
        f"floor friction {env.model.geom_friction[floor_id]} != training terrain {train_terrain_friction}"
    robot_frictions = {tuple(round(float(v), 6) for v in row) for row in training.geom_friction}
    assert len(robot_frictions) == 1, f"unexpected robot geom frictions: {robot_frictions}"
    slide, spin, roll = next(iter(robot_frictions))
    expected_contact = (slide, slide, spin, roll, roll)
    probe = mujoco.MjData(env.model)
    mujoco.mj_forward(env.model, probe)
    floor_contacts = [i for i in range(probe.ncon)
                      if floor_id in (probe.contact[i].geom1, probe.contact[i].geom2)]
    assert floor_contacts, "no robot-floor contacts at the qpos0 stance"
    contacts = {}
    for contact_id in floor_contacts:
        contact = probe.contact[contact_id]
        robot_id = contact.geom2 if contact.geom1 == floor_id else contact.geom1
        measured = tuple(round(float(v), 6) for v in contact.friction)
        assert measured == expected_contact, \
            f"robot-floor contact geom {robot_id} friction {measured} != training {expected_contact}"
        geom_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, robot_id) or f"geom{robot_id}"
        contacts[geom_name] = list(measured)
    results["contact_parameters"] = {
        "robot_geom_priority": train_priority,
        "floor_friction": list(train_terrain_friction),
        "robot_floor_contacts": contacts,
    }

    # 8. Recovery height is geometric and cannot be replaced by world base Z.
    height_env = entry.Playback()
    nominal_qpos = height_env.data.qpos.copy()
    baseline = height_env.hip_wheel_height()
    height_env.data.qpos[:3] += (2., -3., 4.)
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, np.array((0., 1., 0.)), .6)
    height_env.data.qpos[3:7] = quat
    mujoco.mj_forward(height_env.model, height_env.data)
    assert abs(height_env.hip_wheel_height() - baseline) < 1e-12, \
        "hip-wheel estimate depends on root translation/rotation"
    assert abs(height_env.metrics()["height_m"] - 4.38) < 1e-12, \
        "world height report changed meaning"

    height_env.rng = np.random.default_rng(42)
    height_env.reset("ground")
    folded_qpos = height_env.data.qpos.copy()
    # One leg extended and one folded: project each leg before averaging.
    for name in entry.RIGHT_JOINTS:
        adr = height_env.model.joint(name).qposadr[0]
        height_env.data.qpos[adr] = nominal_qpos[adr]
    mujoco.mj_forward(height_env.model, height_env.data)
    rotation = height_env.data.xmat[height_env.base_body].reshape(3, 3)
    projected = [float(-(rotation.T @ (height_env.data.xpos[wheel]
                                       - height_env.data.xanchor[hip]))[2])
                 for hip, wheel in zip(height_env.hip_joints, height_env.wheel_bodies, strict=True)]
    assert abs(projected[0] - projected[1]) > .1, "split-leg probe is not asymmetric"
    assert abs(height_env.hip_wheel_height() - np.mean(projected)) < 1e-12

    # Controlled state probes: restore each pose before stepping so height alone
    # distinguishes folded/extended legs, without claiming a balancing rollout.
    recovery_probes = {}
    for label, qpos, expected in (("folded", folded_qpos, False),
                                  ("extended", nominal_qpos, True)):
        height_env.reset("ground")
        for _ in range(26):
            height_env.data.qpos[:] = qpos
            height_env.data.qpos[:3] = (0., 0., 4.)
            height_env.data.qvel.fill(0.)
            mujoco.mj_forward(height_env.model, height_env.data)
            height_env.step(lambda obs: np.zeros(6))
            row = height_env.metrics()
            assert not row["base_contact"] and max(map(abs, row["roll_pitch_deg"])) < 40.
            assert (row["hip_wheel_height_m"] > entry.RECOVERY_HEIGHT_M) == expected
        assert bool(height_env.recovered) == expected, \
            f"{label} recovery used world base height instead of hip-wheel projection"
        assert bool(np.array_equal(height_env.mode_and_flags()[0], entry.NORMAL_MODE)) == expected
        recovery_probes[label] = {"world_height_m": row["height_m"],
                                  "hip_wheel_height_m": row["hip_wheel_height_m"],
                                  "recovered": bool(height_env.recovered)}
    results["recovery_height"] = {"stance_m": baseline, "split_leg_projections_m": projected,
                                   "threshold_m": entry.RECOVERY_HEIGHT_M, "probes": recovery_probes}

    # At a 0.15 m vertical leg, a low command may complete recovery; the old
    # command band must still require the historical 0.20 m leg projection.
    height_env.command[2] = .1865
    low_threshold = height_env.recovery_height_threshold()
    assert .13 < low_threshold < .15
    height_env.command[2] = .27
    assert height_env.recovery_height_threshold() == entry.RECOVERY_HEIGHT_M
    results["recovery_height"]["low_command_threshold_m"] = low_threshold

    summary = {"ok": True, "script": str(Path(__file__).resolve()),
               "scope": "entry-vs-training consistency + H7 recovery-height probes; not a behavior acceptance",
               "results": results}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
