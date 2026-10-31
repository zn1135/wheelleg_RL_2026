#!/usr/bin/env python3
"""Isaac32 history, get-up boundaries, gradients and finite physics; no parameter updates."""

import argparse
import ast
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import mujoco
from scipy.optimize import brentq

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.model import (
    MODEL_XML, load_model, solve_standing_joint_positions, closed_chain_residual)
from ct_mjlab.envs.chuanliantui.huananhu_poses import _front_kinematics, _mesh_vertices
from ct_mjlab.envs.chuanliantui_stand_v2 import mdp, rewards
from ct_mjlab.envs.chuanliantui_stand_v2.observations import ACTOR_DIM, CRITIC_DIM
from scripts.agent.isaac.check_isaac_standup_rewards import SOURCES, source_class


ROOT = Path(__file__).resolve().parents[3]


def reward_parity():
    torch.manual_seed(42)
    namespace = {"torch": torch}
    source_class(SOURCES[0], "LeggedRobot", None, namespace)
    cls = source_class(SOURCES[1], "ChuanliantuiStandup", "LeggedRobot", namespace)
    source = cls()
    source.cfg = NS(rewards=NS(tracking_sigma=.25, height_reward_sigma=.01,
        height_reward_sigma_post_unlock=.0025, height_reward_contact_factor=.2,
        base_link_reward_force_scale=5., recovered_reward_duration_s=.5, knee_limit_margin_rad=.05),
        standup=NS(initial_base_height=.15, wheel_contact_force_threshold=1.))
    tree = ast.parse(SOURCES[2].read_text())
    cfg_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ChuanliantuiStandupCfg")
    reward_cfg = next(n for n in cfg_class.body if isinstance(n, ast.ClassDef) and n.name == "rewards")
    scales = next(n for n in reward_cfg.body if isinstance(n, ast.ClassDef) and n.name == "scales")
    weights = {n.targets[0].id: ast.literal_eval(n.value) for n in scales.body if isinstance(n, ast.Assign)}
    assert {name: value for name, value in weights.items() if value} == rewards.ISAAC_WEIGHTS
    n = 128
    q, dq = torch.randn(n, 4), torch.randn(n, 4) * 300.
    limits = torch.tensor([[-.5, .8]]*4)
    knee_limits = torch.tensor([[-.12, .77], [-.77, .12]])
    swing = (torch.rand(n, 2) - .5) * 6.
    positions = torch.zeros(n, 4, 3)
    positions[:, 2:, 0] = -swing.sin()
    positions[:, 2:, 2] = -swing.cos()
    data = NS(body_link_pos_w=positions, root_link_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(n, 1),
        projected_gravity_b=torch.randn(n, 3), root_link_lin_vel_b=torch.randn(n, 3)*3,
        root_link_ang_vel_b=torch.randn(n, 3)*4, joint_pos=q, joint_vel=dq)
    state = NS(cfg=NS(post_stand_tracking=False, tracking_tight_term=False), height_above_ground=torch.rand(n)*.5, command=torch.zeros(n, 3),
        has_landed=torch.rand(n)>.5, has_stood=torch.rand(n)>.5,
        standing_time=torch.rand(n), reward_active=torch.ones(n, dtype=torch.bool), numerical_failure=torch.zeros(n, dtype=torch.bool))
    base_force, support = torch.rand(n)*10., torch.rand(n, 2)>.5
    contacts = torch.zeros(n, 3, 3)
    contacts[:, 0, 2] = base_force
    contacts[:, 1:, 2] = support*2.
    manager = NS(action=torch.randn(n, 6)*10., prev_action=torch.randn(n, 6), prev_prev_action=torch.randn(n, 6))
    torque = torch.randn(n, 6)*80.
    sensors = {"base_contact": NS(data=NS(force=contacts[:, :1])),
               "wheel_contact": NS(data=NS(force=-contacts[:, 1:]))}
    env = NS(command_manager=NS(get_term=lambda name: state), scene=sensors,
        action_manager=manager, sim=NS(data=NS(qfrc_actuator=torque)), common_step_counter=1)
    port = rewards.RewardState.__new__(rewards.RewardState)
    port.robot = NS(data=data)
    port.legs, port.knees, port.motor_vadr = [0, 1, 2, 3], [1, 3], [0, 1, 2, 3, 4, 5]
    port.hips, port.wheels = [0, 1], [2, 3]
    port.soft_limits, port.knee_limits = limits, knee_limits
    source.base_height = state.height_above_ground
    source.projected_gravity, source.base_lin_vel, source.base_ang_vel = (
        data.projected_gravity_b, data.root_link_lin_vel_b, data.root_link_ang_vel_b)
    source.has_landed, source.has_stood, source.standing_time = state.has_landed, state.has_stood, state.standing_time
    source.theta0, source.contact_forces = swing, contacts
    source.feet_indices, source.termination_contact_indices = torch.tensor([1, 2]), torch.tensor([0])
    source.dof_pos, source.dof_vel = q[:, [0, 1, 0, 2, 3, 0]], dq[:, [0, 1, 0, 2, 3, 0]]
    source.dof_pos_limits, source.knee_dof_indices = limits[[0, 1, 0, 2, 3, 0]], torch.tensor([1, 4])
    source.knee_hard_limits, source.torques, source.actions = knee_limits, torque, manager.action
    source.last_actions = torch.stack((manager.prev_action, manager.prev_prev_action), -1)
    errors = {}
    for unlocked in (False, True):
        state.unlocked = source.standup_curriculum_unlocked = unlocked
        state.command[:, 2] = .22 if unlocked else .30
        source.commands = state.command
        source.reward_scales = {**rewards.ISAAC_WEIGHTS, "orientation": -10. if unlocked else -1.}
        port.step = -1
        actual = port.compute(env)
        for name, weight in source.reward_scales.items():
            if name in ("base_height", "orientation", "lin_vel_z", "ang_vel_xy"):
                continue  # These four terms deliberately depart from the Isaac recipe.
            expected = (getattr(source, "_reward_"+name)()*weight).clamp(-1., 1.)*.01
            if name == "nominal_state":
                expected *= .1 + .9 * torch.where(state.has_stood, 1.,
                    (state.standing_time / .5).clamp(0., 1.))
            value = actual[name]*rewards.WEIGHTS[name]*.01
            torch.testing.assert_close(value, expected, atol=2e-7, rtol=1e-5, msg=name)
            if name != "nominal_state":
                errors[f"{int(unlocked)}:{name}"] = float((value-expected).abs().max())
    assert "getup_inverted" not in actual and len(actual) == 19
    state.command[:, 2] = state.height_above_ground[:] = .22
    state.has_stood[:] = False
    state.standing_time.zero_()
    sensors["base_contact"].data.force.zero_()
    base_force = sensors["base_contact"].data.force[:, 0, 2]
    angles = torch.deg2rad(torch.tensor([0., 15., 27.5, 40., 90., 120., 150., 180.]))
    data.projected_gravity_b.zero_()
    data.projected_gravity_b[:, 2] = -1.
    data.projected_gravity_b[:8, 0] = angles.sin()
    data.projected_gravity_b[:8, 2] = -angles.cos()
    data.root_link_lin_vel_b[:] = torch.tensor([0., 0., .2])
    data.root_link_ang_vel_b[:] = torch.tensor([.2, -.3, 0.])
    def compute():
        env.common_step_counter += 1
        return port.compute(env)
    # 大角差使原项饱和；只有限幅后缩放才能把起立前的加权惩罚降到-0.1。
    positions[:, 2:, 0] = torch.tensor([-1., 1.])
    positions[:, 2:, 2] = 0.
    state.standing_time[:6] = torch.tensor([0., .125, .25, .49, .5, 1.])
    nominal = compute()["nominal_state"].clone()
    torch.testing.assert_close(nominal[:6] * rewards.WEIGHTS["nominal_state"],
        -torch.tensor([.1, .325, .55, .982, 1., 1.]))
    # 同一策略拍重复取值不应重复乘系数。
    torch.testing.assert_close(port.compute(env)["nominal_state"], nominal, atol=0., rtol=0.)
    state.standing_time.zero_()
    torch.testing.assert_close(compute()["nominal_state"], torch.full((n,), .1/3.))
    state.has_stood[:] = True
    torch.testing.assert_close(compute()["nominal_state"], torch.full((n,), 1./3.))
    # 小角差也保留原平方公式；站稳锁存优先于当前站立计时。
    positions[:, 2:, 0] = -torch.tensor([.1, -.1]).sin()
    positions[:, 2:, 2] = -torch.tensor([.1, -.1]).cos()
    torch.testing.assert_close(compute()["nominal_state"], torch.full((n,), .04))
    state.has_stood[:] = False
    torch.testing.assert_close(compute()["nominal_state"], torch.full((n,), .004))
    value = compute()
    torch.testing.assert_close(value["base_height"][:8],
        torch.tensor([1., 1., .5, 0., 0., 0., 0., 0.]), atol=5e-7, rtol=1e-5)
    torch.testing.assert_close(value["orientation"][:8], 1.-angles.cos())
    assert float(value["orientation"][7]*rewards.WEIGHTS["orientation"]) == -2.
    assert (value["orientation"][4:8].diff() > 0).all(), "No orientation saturation after 90 degrees"
    for key, target in (("lin_vel_z", .004), ("ang_vel_xy", .013)):
        torch.testing.assert_close(value[key], torch.full((n,), target))
    # Low height must not suppress orientation; a separate inverted penalty is unnecessary.
    state.height_above_ground[:] = .10
    torch.testing.assert_close(compute()["orientation"], value["orientation"])
    state.height_above_ground[:] = .27
    data.projected_gravity_b.zero_(); data.projected_gravity_b[:, 2] = -1.
    # Even after global unlock and an earlier success, current stance controls precision.
    state.unlocked = True
    state.has_stood[:] = True
    state.standing_time[:] = .49
    broad = compute()
    torch.testing.assert_close(broad["base_height"], torch.full((n,), math.exp(-.25)))
    state.standing_time[:] = .5
    narrow = compute()
    torch.testing.assert_close(narrow["base_height"], torch.full((n,), math.exp(-1.)))
    for key, target in (("lin_vel_z", .04), ("ang_vel_xy", .13)):
        torch.testing.assert_close(narrow[key], torch.full((n,), target))
    state.standing_time.zero_()
    torch.testing.assert_close(compute()["base_height"], broad["base_height"])
    state.unlocked = False
    state.standing_time[:] = .5
    torch.testing.assert_close(compute()["base_height"], broad["base_height"])
    base_force[:] = 5.
    torch.testing.assert_close(compute()["base_height"], broad["base_height"]*.2)
    state.reward_active[:] = False
    env.common_step_counter += 1
    assert all(not v.any() for v in port.compute(env).values())
    state.reward_active[:] = True
    state.numerical_failure[:] = True
    assert all(not v.any() for v in compute().values())
    return {"unchanged_isaac_terms_max_abs_error": errors,
            "nominal_state_post_clip_first_stand_scale": True,
            "nominal_state_0_125_250_490_500_1000ms": True,
            "nominal_state_interrupted_ramp_and_stood_latch_and_cache": True,
            "upright_gate_15_27p5_40_degrees": True, "unclipped_orientation_through_inversion": True,
            "low_height_orientation_active": True, "pre_stand_damping_10pct": True,
            "precision_requires_global_unlock_and_current_50_ticks": True,
            "loss_of_stance_returns_to_broad_reward": True, "contact_and_hold_and_numerical_masks": True}


def termination_parity():
    # Execute only the source's tensor-based termination method, never Isaac runtime.
    tree = ast.parse(SOURCES[1].read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ChuanliantuiStandup")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "check_termination")
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(SOURCES[1]), "exec"), namespace)
    n, dt = 6, .01
    source = NS(has_landed=torch.zeros(n, dtype=torch.bool), has_stood=torch.zeros(n, dtype=torch.bool),
        standing_time=torch.zeros(n), fail_buf=torch.zeros(n), wheels_airborne_steps=torch.zeros(n, dtype=torch.long),
        episode_length_buf=torch.zeros(n, dtype=torch.long), dt=dt, max_episode_length=4000,
        cfg=NS(standup=NS(success_base_contact_force_threshold=.1, success_projected_gravity_z=-.9,
            success_requires_base_contact_free=True, success_duration_s=.5, wheels_airborne_timeout_s=.2),
            env=NS(fail_to_terminal_time_s=1.)))
    support, base_force = torch.ones(n, dtype=torch.bool), torch.zeros(n)
    gravity, height = torch.zeros(n, 3), torch.full((n,), .30)
    gravity[:, 2] = -1.
    source.projected_gravity, source.base_height = gravity, height
    source._has_wheel_contact = lambda: support
    source._both_wheels_airborne = lambda: ~support
    source._base_link_has_contact = lambda threshold: base_force > threshold
    source._current_standup_success_height = lambda: .20
    state = NS(has_landed=torch.zeros(n, dtype=torch.bool), has_stood=torch.zeros(n, dtype=torch.bool),
        standing_time=torch.zeros(n), stable_time=torch.zeros(n), wheels_airborne_steps=torch.zeros(n, dtype=torch.long),
        isaac_fallen_steps=torch.zeros(n, dtype=torch.long), landed_steps=torch.zeros(n, dtype=torch.long),
        fallen=torch.zeros(n, dtype=torch.bool), knockdown=torch.zeros(n, dtype=torch.bool),
        numerical_failure=torch.zeros(n, dtype=torch.bool), completed=torch.zeros(n, dtype=torch.bool),
        height_above_ground=torch.zeros(n), success_height=.20, _last_failure_step=-1,
        metrics={"isaac_recovered": torch.zeros(n)},
        observation_stable_steps=torch.zeros(n, dtype=torch.long),
        observation_recovered=torch.zeros(n, dtype=torch.bool))
    pos = torch.zeros(n, 3); pos[:, 2] = height
    robot = NS(data=NS(root_link_pos_w=pos, projected_gravity_b=gravity))
    class Scene(dict):
        env_origins = torch.zeros(n, 3)
    scene = Scene(robot=robot, wheel_contact=NS(data=NS(force=torch.zeros(n, 2, 3))),
                  base_contact=NS(data=NS(force=torch.zeros(n, 1, 3))))
    env = NS(command_manager=NS(get_term=lambda name: state), scene=scene,
        sim=NS(data=NS(qpos=torch.zeros(n, 25), qvel=torch.zeros(n, 24)),
               model=NS(body_mass=torch.full((n, 1), 10.)), mj_model=NS(opt=NS(gravity=(0., 0., -10.)))),
        step_dt=dt, max_episode_length=4000, episode_length_buf=torch.zeros(n, dtype=torch.long))
    # Isolate unchanged post-stand failures; new success qualification is tested below.
    source.has_stood[:5] = state.has_stood[:5] = True
    for step in range(1, 170):
        # Qualified stance, exact airborne/fall boundaries, inversion, recovery and never-landed.
        support[:] = True; base_force[:] = 0.; gravity[:, 2] = -1.
        support[5] = False
        if step > 60:
            support[0] = False
            base_force[1] = 11.
            gravity[2, 2] = .01
            support[3] = step % 10 == 0
        scene["wheel_contact"].data.force[:, :, 2] = -support[:, None].float()*50.
        scene["base_contact"].data.force[:, 0, 2] = base_force
        env.common_step_counter = step
        source.episode_length_buf += 1
        env.episode_length_buf += 1
        namespace["check_termination"](source)
        actual = mdp.failure(env)
        # Isaac returns a single reset mask; mjlab splits failure from timeout.
        torch.testing.assert_close(actual | mdp.timeout(env), source.reset_buf)
        torch.testing.assert_close(state.has_stood, source.has_stood)
        # Current support now resets the stance timer; Isaac's timer only remembers landing.
        torch.testing.assert_close(mdp.timeout(env), source.time_out_buf)
        before = state.standing_time.clone()
        mdp.failure(env)
        torch.testing.assert_close(before, state.standing_time)
    # Deliberate changes from Isaac: get-up grace and total-reset timeout.
    for name in ("has_stood", "standing_time", "wheels_airborne_steps", "isaac_fallen_steps",
                 "observation_stable_steps", "observation_recovered", "completed"):
        getattr(state, name).zero_()
    state.has_landed[:] = True
    pos[:, 2] = .15
    gravity[:, 2] = 1.
    scene["wheel_contact"].data.force.zero_()
    scene["base_contact"].data.force.zero_()
    for step in range(170, 200):
        env.common_step_counter = step
        assert not mdp.failure(env).any(), "Pre-stand inversion/airborne must not terminate"
    # At 0.27 m and 25 degrees NORMAL may activate, but upright-first success must not.
    gravity[:, 2] = -1.
    gravity[:, 0] = math.sin(math.radians(25))
    gravity[:, 2] = -math.cos(math.radians(25))
    pos[:, 2] = .27
    scene["wheel_contact"].data.force[:, :, 2] = -50.
    for step in range(200, 250):
        env.common_step_counter = step
        mdp.failure(env)
    assert state.observation_recovered.all() and not state.has_stood.any()
    # Both wheels touching is insufficient when the body still carries most of the weight.
    scene["wheel_contact"].data.force[:, :, 2] = -10.
    env.common_step_counter = 250
    mdp.failure(env)
    assert not state.observation_recovered.any()
    scene["wheel_contact"].data.force[:, :, 2] = -50.
    for step in range(251, 300):
        env.common_step_counter = step
        mdp.failure(env)
        assert not state.observation_recovered.any()
    env.common_step_counter = 300
    mdp.failure(env)
    assert state.observation_recovered.all(), "NORMAL requires 50 consecutive supported policy steps"
    gravity[:, 0] = .76604444
    gravity[:, 2] = -.64278761  # 50-degree pitch: re-enter RECOVER immediately.
    env.common_step_counter = 301
    mdp.failure(env)
    assert not state.observation_recovered.any()
    # Success is independent of NORMAL and requires upright+height+both wheels+free base.
    gravity.zero_(); gravity[:, 2] = -1.
    pos[:, 2] = .20
    pos[0, 2] = .1999
    gravity[1, 0] = math.sin(math.radians(15.1))
    gravity[1, 2] = -math.cos(math.radians(15.1))
    scene["wheel_contact"].data.force[:, :, 2] = -50.
    scene["wheel_contact"].data.force[2, :, 2] = -39.
    scene["wheel_contact"].data.force[3, :, 2] = torch.tensor([-100., 0.])
    scene["base_contact"].data.force[4, 0, 2] = .101
    state.standing_time.zero_(); state.has_stood.zero_()
    for step in range(302, 351):
        env.common_step_counter = step
        mdp.failure(env)
        assert not state.has_stood.any()
    env.common_step_counter = 351
    mdp.failure(env)
    torch.testing.assert_close(state.has_stood, torch.tensor([False]*5+[True]))
    torch.testing.assert_close(state.standing_time, torch.tensor([0.]*5+[.5]))
    pos[5, 2] = .19
    env.common_step_counter = 352
    mdp.failure(env)
    assert state.has_stood[5] and state.standing_time[5] == 0.
    state.has_landed[5] = False
    scene["wheel_contact"].data.force[5] = 0.
    for step, length in ((353, 3999), (354, 4000)):
        env.common_step_counter = step
        env.episode_length_buf[:] = length
        mdp.failure(env)
        assert bool(mdp.timeout(env).all()) == (length == 4000)
    assert state.completed.all(), "Never-landed timeout must count in curriculum"
    assert not state.has_landed[5]
    env.sim.data.qpos[0, 0] = float("nan")
    env.common_step_counter += 1
    assert mdp.failure(env)[0], "Numerical failure stays active before standing"
    return {"post_stand_isaac_parity_steps": 169, "cases": n,
            "pre_stand_grace": True, "never_landed_timeout": True, "independent_recover": True,
            "normal_wheel_load_and_50_step_hold": True,
            "upright_success_15deg_20cm_both_wheels_free_base_exact_50ticks": True,
            "current_stance_resets_but_success_memory_remains": True}


def height_geometry_checks():
    model, result = load_model(), {}
    data = mujoco.MjData(model)
    wheel_bodies = {model.body(name).id for name in ("lfwheel", "rfwheel")}
    wheel_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] in wheel_bodies
                   and (model.geom_contype[i] or model.geom_conaffinity[i])]
    def solve(knee):
        _, phi = _front_kinematics(model, data, knee)
        positions = solve_standing_joint_positions(model,
            front_joint_targets={"left": (phi, knee), "right": (-phi, -knee)})
        data.qpos[:] = model.qpos0
        data.qpos[:7] = (0., 0., 0., 1., 0., 0., 0.)
        for name, value in positions.items():
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        return -min(float(_mesh_vertices(model, data, i)[:, 2].min()) for i in wheel_geoms)
    lower, upper = model.joint("lf1").range
    for height in (.30, .22):
        knee = brentq(lambda value: solve(value)-height, lower+1e-4, upper-1e-4)
        actual = solve(knee)
        assert abs(actual-height) < 1e-8
        assert closed_chain_residual(model, data) < 1e-8
        for i in range(model.njnt):
            if model.jnt_limited[i]:
                value = data.qpos[model.jnt_qposadr[i]]
                assert model.jnt_range[i, 0] <= value <= model.jnt_range[i, 1]
        result[str(height)] = {"wheel_supported_height_m": actual,
                              "closed_chain_residual_m": closed_chain_residual(model, data)}
    return result


def contact_metric_checks():
    import numpy as np
    from mjlab.managers.metrics_manager import MetricsManager, MetricsTermCfg
    from ct_mjlab.envs.chuanliantui_stand_v2 import diagnostics
    from scripts.agent.stand_v2.diagnose_chuanliantui_stand_v2 import FIELDS, contact_summary

    state = NS(recover=torch.zeros(3, dtype=torch.bool))
    force = torch.zeros(3, 1, 3)
    env = NS(num_envs=3, device="cpu", scene={"base_contact": NS(data=NS(force=force))},
        command_manager=NS(get_term=lambda name: state))
    manager = MetricsManager({
        "base_contact_force_N": MetricsTermCfg(func=mdp.base_contact_force),
        "base_contact_fraction": MetricsTermCfg(func=diagnostics.base_contact_fraction),
        "normal_base_contact_fraction": MetricsTermCfg(func=diagnostics.normal_base_contact_fraction)}, env)
    forces = ((0., 2., 5.), (2., 2., 0.), (1., 0., 5.), (2., 2., 0.))
    normal = ((True, False, True), (False, True, True),
              (True, True, False), (True, False, True))
    values = np.zeros((4, 3, len(FIELDS)))
    for step, (contact, mode) in enumerate(zip(forces, normal, strict=True)):
        # Horizontal contact counts too: diagnostics use vector magnitude, not only Z.
        force[:, 0, 0] = torch.tensor(contact)
        state.recover = ~torch.tensor(mode)
        before = state.recover.clone()
        manager.compute()
        torch.testing.assert_close(state.recover, before)
        values[step, :, FIELDS.index("base_force")] = contact
        values[step, :, FIELDS.index("recover_mode")] = state.recover.numpy()
    result = manager.reset(torch.tensor([0, 2]))
    for name, expected in (("base_contact_force_N", 1.875), ("base_contact_fraction", .5),
                           ("normal_base_contact_fraction", .25)):
        assert float(result["Episode_Metrics/"+name]) == expected
    retained = manager.reset(torch.tensor([1]))
    assert float(retained["Episode_Metrics/base_contact_force_N"]) == 1.5
    assert float(retained["Episode_Metrics/base_contact_fraction"]) == .75
    assert float(retained["Episode_Metrics/normal_base_contact_fraction"]) == .25
    active = np.ones((4, 3), dtype=bool)
    late = active.copy(); late[:2] = False
    offline = contact_summary(values, active, late, .01)
    assert offline["normal_base_contact_fraction"] == .25
    assert offline["normal_base_contact_envs"] == 3
    assert offline["last5_normal_base_contact_envs"] == 1
    assert offline["last5_persistent_normal_base_contact_envs"] == 1
    for row in offline["episodes"]:
        assert row["normal_base_contact_s"] == .01
        assert row["normal_base_contact_fraction"] == .25
    return {"force_threshold_boundary_N": 1., "normal_and_contact_same_sample": True,
            "episode_mean_and_partial_reset": True, "no_mode_mutation": True}


def normal_wheel_support_checks():
    forces = torch.tensor([[40., 40.], [39.999, 40.], [9.999, 90.001], [90., 10.],
                           [100., 0.], [80., 80.], [19., 5.], [19.1, 4.9]])
    mass = torch.tensor([10., 10., 10., 10., 10., 20., 3., 3.])[:, None]
    sensor = torch.zeros(8, 2, 3)
    sensor[:, :, 2] = -forces
    env = NS(scene={"wheel_contact": NS(data=NS(force=sensor))},
        sim=NS(model=NS(body_mass=mass), mj_model=NS(opt=NS(gravity=(0., 0., -10.)))))
    torch.testing.assert_close(mdp.normal_wheel_support(env),
        torch.tensor([True, False, False, True, False, True, True, False]))
    mass[0] = 11.
    assert not mdp.normal_wheel_support(env)[0], "Use the current randomized weight"
    sensor[1, :, 0] = 1000.
    assert not mdp.normal_wheel_support(env)[1], "Horizontal force is not upward support"
    return {"total_80pct_and_each_10pct_boundaries": True, "each_5N_floor": True,
            "randomized_mass": True, "upward_force_only": True}


def observation_checks(wrapped):
    raw = wrapped.unwrapped
    state = raw._multimode_observations
    sensors = torch.arange(18, device=raw.device).float().repeat(raw.num_envs, 1) * .1
    values = sensors + state.noise if state.cfg.noise else sensors
    # Separate explicit field checks catch ordering and scaling mistakes, not just shape.
    with patch.object(state.delay, "read", return_value=sensors):
        obs = state.actor()
    torch.testing.assert_close(obs[:, :3], values[:, :3] * .25)
    torch.testing.assert_close(obs[:, 3:6], values[:, 3:6])
    torch.testing.assert_close(obs[:, 6:9], state.command.command *
        torch.tensor((2., .25, 5.), device=raw.device))
    torch.testing.assert_close(obs[:, 9:13], values[:, 6:10])
    torch.testing.assert_close(obs[:, 13:19], values[:, (12, 13, 16, 14, 15, 17)] * .05)
    torch.testing.assert_close(obs[:, 19:25], state.action.raw_action)
    torch.testing.assert_close(obs[:, 25:32], state.command.mode())
    critic = state.critic()
    current = wrapped.get_observations()
    torch.testing.assert_close(current["velocity_target"], critic[:, 28:31] * 2.)
    phase = state.command
    expected = torch.stack((phase.has_landed.float(), phase.has_stood.float(),
        (phase.standing_time / .5).clamp(0, 1), (phase.wheels_airborne_steps / 20).clamp(0, 1),
        (phase.isaac_fallen_steps / 100).clamp(0, 1),
        (phase.observation_stable_steps / 50).clamp(0, 1),
        (raw.episode_length_buf / raw.max_episode_length).clamp(0, 1)), -1)
    torch.testing.assert_close(critic[:, 82:], expected)
    return {"actor_fields_and_scales": True, "critic_memory_channels": 7,
            "velocity_target_scale": 2., "physical_motor_positions": True}


def network_checks(wrapped, runner):
    """Exercise the actual PPO/encoder backward path with optimizer.step disabled."""
    alg = runner.alg
    obs = wrapped.get_observations()
    model = alg.get_policy()
    with torch.no_grad():
        baseline = model(obs).clone()
        changed = obs.clone()
        changed["critic"].fill_(100.)
        changed["velocity_target"].fill_(100.)
        torch.testing.assert_close(model(changed), baseline)
    # Policy loss cannot backpropagate into the velocity estimator.
    model.zero_grad(set_to_none=True)
    model(obs).square().mean().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    weights_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for _ in range(alg.storage.num_transitions_per_env):
            actions = alg.act(obs)
            obs, reward, done, extras = wrapped.step(actions)
            alg.process_env_step(obs, reward, done, extras)
        alg.compute_returns(obs)
    # Suppressing both optimizer steps makes this a gradient/interface check, not training.
    with patch.object(torch.optim.Adam, "step", return_value=None) as suppressed:
        losses = alg.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, weights_before[k], atol=0, rtol=0)
    return {"privileged_input_isolation": True, "ppo_encoder_detached": True,
            "supervised_encoder_gradient": True, "suppressed_optimizer_steps": suppressed.call_count,
            "losses": losses, "parameter_updates": 0}


def ground_probe(args):
    """First-episode diagnostics under zero/random controls, not a learned-policy score."""
    result = {}
    for std in (0., .5):
        cfg, agent = task_registry.get_cfgs("chuanliantui_stand_v2")
        cfg.seed = 42
        wrapped, _ = task_registry.make_env("chuanliantui_stand_v2", env_cfg=cfg,
            train_cfg=agent, num_envs=args.probe_envs, device=args.device)
        raw = wrapped.unwrapped
        try:
            n = raw.num_envs
            live = torch.ones(n, dtype=torch.bool, device=args.device)
            ended = torch.full((n,), -1., device=args.device)
            returns = torch.zeros(n, device=args.device)
            active_seconds = torch.zeros(n, device=args.device)
            observed_stood = torch.zeros_like(live)
            generator = torch.Generator(device=args.device).manual_seed(123)
            state = raw.command_manager.get_term("standup")
            with torch.no_grad(), patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No training")):
                for step in range(round(args.probe_seconds / raw.step_dt)):
                    observed_stood |= live & state.has_stood
                    active_seconds += live * state.policy_active * raw.step_dt
                    action = torch.randn(n, 6, generator=generator, device=args.device) * std
                    obs, reward, done, _ = wrapped.step(action)
                    assert all(torch.isfinite(v).all() for v in obs.values())
                    assert torch.isfinite(reward).all()
                    returns += live * reward
                    first_done = live & done.bool()
                    ended[first_done] = (step + 1) * raw.step_dt
                    observed_stood |= live & state.has_stood
                    live &= ~done.bool()
            completed = ended[ended >= 0]
            result[str(std)] = {"num_envs": n, "duration_s": args.probe_seconds,
                "first_episode_terminated": int((~live).sum()),
                "first_termination_median_s": float(completed.median()) if len(completed) else None,
                "observed_stood": int(observed_stood.sum()),
                "mean_first_episode_return": float(returns.mean()),
                "reward_per_active_second": float(returns.sum() / active_seconds.sum().clamp_min(.01)),
                "parameter_updates": 0}
        finally:
            wrapped.close()
    return result


def check(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "source formulas, deliberate termination changes, history, gradients, export; no parameter updates",
              "reward_parity": reward_parity(), "termination_parity": termination_parity(),
              "height_geometry": height_geometry_checks(), "contact_metrics": contact_metric_checks(),
              "normal_wheel_support": normal_wheel_support_checks()}
    cfg, agent = task_registry.get_cfgs("chuanliantui_stand_v2")
    original, original_agent = task_registry.get_cfgs("chuanliantui_stand")
    a, b = asdict(agent), asdict(original_agent)
    a.pop("experiment_name"); b.pop("experiment_name")
    a["actor"]["class_name"] = b["actor"]["class_name"]
    assert a["actor"]["distribution_cfg"]["init_std"] == .5
    a["actor"]["distribution_cfg"]["init_std"] = b["actor"]["distribution_cfg"]["init_std"]
    a["algorithm"]["class_name"] = b["algorithm"]["class_name"]
    assert a == b
    for name in ("scene", "sim", "decimation", "episode_length_s", "viewer"):
        left, right = getattr(cfg, name), getattr(original, name)
        if name in ("scene", "sim", "viewer"):
            # Factories create distinct but equivalent default terrain lambdas.
            encode = lambda value: value.__module__+"."+value.__qualname__ if callable(value) else repr(value)
            left = json.dumps(asdict(left), default=encode, sort_keys=True)
            right = json.dumps(asdict(right), default=encode, sort_keys=True)
        assert left == right, name
    assert len(cfg.rewards) == 19 and "smp_motion" not in cfg.rewards
    cfg.seed = 42
    wrapped, _ = task_registry.make_env("chuanliantui_stand_v2", env_cfg=cfg, train_cfg=agent,
        num_envs=20, device=args.device)
    raw = wrapped.unwrapped
    try:
        runner, _ = task_registry.make_alg_runner(wrapped, "chuanliantui_stand_v2", train_cfg=agent, device=args.device)
        state, action = raw.command_manager.get_term("standup"), raw.action_manager.get_term("pd")
        physical = raw._standup_randomization
        assert state.cfg.ground_start_probability == 1. and state.ground_start.all()
        torch.testing.assert_close(action.action_delay_steps, action.wheel_action_delay_steps)
        assert int(action.action_delay_steps.max()) <= 5
        assert ((action.kp_scale >= .95) & (action.kp_scale <= 1.05)).all()
        assert ((action.motor_strength >= .8) & (action.motor_strength <= 1.2)).all()
        assert (action._spring_force_bias == 0).all() and (action.spring_damping == 50).all()
        assert ((physical.robot_friction <= .2) | (physical.robot_friction >= .5)).all()
        assert (physical.com_offset.abs() <= .02).all()
        assert ((physical.restitution >= .6) & (physical.restitution <= 1.)).all()
        noise = raw._multimode_observations
        torch.testing.assert_close(noise.noise_bounds.cpu(), torch.tensor(
            [.2]*3+[.05]*3+[.02]*4+[0.]*2+[1.5]*6))
        assert torch.isin(noise.delay.lags, torch.tensor([3, 5, 8], device=args.device)).all()
        frozen = [v.clone() for v in (action.kp_scale, action.kd_scale, action.motor_strength,
                                      action.nominal_positions, action.action_delay_steps)]
        optimizer_before = copy.deepcopy(runner.alg.optimizer.state_dict())
        with torch.no_grad(), patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No learning")):
            obs = wrapped.get_observations()
            initial_history = obs["history"].reshape(20, 5, ACTOR_DIM)
            torch.testing.assert_close(initial_history, obs["actor"][:, None].expand(-1, 5, -1))
            for step in range(args.steps):
                previous_history = obs["history"].clone().reshape(20, 5, ACTOR_DIM)
                obs, reward, done, extras = wrapped.step(torch.zeros(20, 6, device=args.device))
                assert obs["actor"].shape == (20, ACTOR_DIM) and obs["critic"].shape == (20, CRITIC_DIM)
                assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
                assert torch.isfinite(reward).all()
                torch.testing.assert_close(obs["history"][:, -ACTOR_DIM:], obs["actor"])
                alive = ~done.bool()
                torch.testing.assert_close(obs["history"].reshape(20, 5, ACTOR_DIM)[alive, :-1],
                                           previous_history[alive, 1:])
                if step < 20:
                    assert not reward.any(), "Motor-hold reward leaked"
                assert not state.airborne.any()
                torch.testing.assert_close(raw.reward_manager._step_reward.sum(-1)*raw.step_dt, reward)
        # Partial resets retain sampled parameters and unrelated episode state.
        before_other = state.standing_time[3:].clone()
        history_other = wrapped.get_observations()["history"][3:].clone()
        raw.reset(env_ids=torch.tensor([0, 1, 2], device=args.device))
        assert state.ground_start.all()
        after_reset = wrapped.get_observations()
        torch.testing.assert_close(after_reset["history"][3:], history_other)
        torch.testing.assert_close(after_reset["history"][:3].reshape(3, 5, ACTOR_DIM),
                                   after_reset["actor"][:3, None].expand(-1, 5, -1))
        torch.testing.assert_close(state.standing_time[3:], before_other)
        for before, after in zip(frozen, (action.kp_scale, action.kd_scale, action.motor_strength,
                                         action.nominal_positions, action.action_delay_steps), strict=True):
            torch.testing.assert_close(before, after, atol=0, rtol=0)
        physical.reset(torch.arange(20, device=args.device))
        state.has_landed[:] = False
        state.has_landed[:2] = True
        state.has_stood[:2] = torch.tensor([False, True], device=args.device)
        state.standing_time[:2] = torch.tensor([0., 1.], device=args.device)
        physical.push_wait_steps[:2] = 0
        action.process_actions(torch.zeros(20, 6, device=args.device))
        magnitude = physical.push_force.norm(dim=-1)
        assert 1. <= magnitude[0] <= 5. and 5. <= magnitude[1] <= 15.
        assert not physical.push_count[2:].any()
        force = physical.push_force.clone()
        for _ in range(9):
            action.process_actions(torch.zeros(20, 6, device=args.device))
            torch.testing.assert_close(physical.push_force, force)
        action.process_actions(torch.zeros(20, 6, device=args.device))
        assert not physical.push_force.any()
        # Trigger the actual 4096-episode threshold without running training updates.
        state.completed_episodes, state.recovered_episodes = 4095, 1229
        state.completed[0] = state.has_stood[0] = True
        raw.reset(env_ids=torch.tensor([0], device=args.device))
        assert state.unlocked
        torch.testing.assert_close(state.command[:, 2], torch.full((20,), .22, device=args.device))
        report["observation_checks"] = observation_checks(wrapped)
        report["network_checks"] = network_checks(wrapped, runner)
        checkpoint = args.output / "untrained-interface.pt"
        runner.save(str(checkpoint))
        saved = state.checkpoint_state()
        encoder_saved = copy.deepcopy(runner.alg.actor.encoder.state_dict())
        with torch.no_grad():
            next(runner.alg.actor.encoder.parameters()).add_(1.)
        state.unlocked = False
        state.completed_episodes = 17
        runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=args.device)
        assert state.checkpoint_state() == saved
        for name, value in runner.alg.actor.encoder.state_dict().items():
            torch.testing.assert_close(value, encoder_saved[name])
        runner.load(str(checkpoint), map_location=args.device)
        assert "encoder_optimizer_state_dict" in torch.load(checkpoint, weights_only=False)
        saved_checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved_checkpoint["infos"]["recovery_contract"] == mdp.RECOVERY_CONTRACT
        saved_checkpoint["infos"].pop("recovery_contract")
        legacy_path = args.output / "legacy-recovery-interface.pt"
        torch.save(saved_checkpoint, legacy_path)
        try:
            runner.load(str(legacy_path), load_cfg={"actor": True}, map_location=args.device)
        except ValueError as error:
            assert "NORMAL rule changed" in str(error)
        else:
            raise AssertionError("Legacy recovery rule was silently changed")
        runner.load(str(legacy_path), load_cfg={"actor": True}, map_location=args.device,
                    allow_recovery_change=True)
        from ct_mjlab.scripts.export import export
        report["export"] = export(checkpoint, args.output / "untrained-interface.onnx", samples=32)
        assert report["export"]["recipe_version"] == rewards.RECIPE_VERSION
        assert report["export"]["curriculum_version"] == mdp.CURRICULUM_VERSION
        assert runner.alg.optimizer.state_dict() == optimizer_before
        # A matching dimension does not authorize loading old task recipes.
        bad = torch.load(checkpoint, map_location="cpu", weights_only=False)
        bad["infos"]["experiment_profile_contract"] = "chuanliantui-stand-39-82-v1"
        bad_path = args.output / "wrong-profile.pt"
        torch.save(bad, bad_path)
        try:
            runner.load(str(bad_path), load_cfg={"actor": True}, map_location=args.device)
        except ValueError:
            pass
        else:
            raise AssertionError("Old standing profile was accepted")
        for key, old in (("recipe_version", "stand-v2-isaac-getup-mask-inverted-v2"),
                         ("curriculum_version", "stand-v2-isaac-height-ground-getup-v3")):
            bad = torch.load(checkpoint, map_location="cpu", weights_only=False)
            bad["infos"][key] = old
            bad_path = args.output / (key+"-old.pt")
            torch.save(bad, bad_path)
            try:
                runner.load(str(bad_path), load_cfg={"actor": True}, map_location=args.device)
            except ValueError as error:
                assert key in str(error)
            else:
                raise AssertionError(f"Old {key} was accepted")
            try:
                export(bad_path, args.output / (key+"-rejected.onnx"), samples=1)
            except ValueError as error:
                assert key in str(error)
            else:
                raise AssertionError(f"Old {key} was exported as current recipe")
        report.update(passed=True, num_envs=20, steps=args.steps,
            physics_dt_s=raw.physics_dt, policy_dt_s=raw.step_dt,
            noise_bounds=noise.noise_bounds.cpu().tolist(),
            action_delay_ticks=action.action_delay_steps.cpu().tolist(),
            curriculum_restored=saved, optimizer_updates=0,
            checkpoint_scope="untrained interface check only; not a trained model")
    finally:
        wrapped.close()
    play_cfg, play_agent = task_registry.get_cfgs("chuanliantui_stand_v2", play=True)
    replay, _ = task_registry.make_env("chuanliantui_stand_v2", env_cfg=play_cfg,
        train_cfg=play_agent, num_envs=2, device=args.device)
    try:
        replay_runner, _ = task_registry.make_alg_runner(replay, "chuanliantui_stand_v2",
            train_cfg=play_agent, device=args.device)
        replay_runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=args.device)
        replay_state = replay.unwrapped.command_manager.get_term("standup")
        assert replay_state.unlocked and replay_state.ground_start.all()
        replay_obs = replay.get_observations()
        torch.testing.assert_close(replay_obs["actor"][:, 8],
            torch.full((2,), .22*5, device=args.device))
        replay_sensors = replay.unwrapped._multimode_observations
        torch.testing.assert_close(replay_obs["history"].reshape(2, 5, ACTOR_DIM)[:, :, 8],
            torch.full((2, 5), .22*5, device=args.device))
        assert not replay_sensors.cfg.noise and not replay_sensors.cfg.delay
        assert not replay.unwrapped.action_manager.get_term("pd").action_delay_steps.any()
        with torch.no_grad():
            policy = replay_runner.get_inference_policy()
            for _ in range(8):
                replay_obs, value, _, _ = replay.step(policy(replay_obs))
                assert torch.isfinite(replay_obs["actor"]).all() and torch.isfinite(value).all()
        report["nominal_replay"] = {"num_envs": 2, "steps": 8, "unlocked_restored": True}
    finally:
        replay.close()
    if args.probe_envs:
        report["ground_probe"] = ground_probe(args)
    encode = lambda value: value.__module__+"."+value.__qualname__ if callable(value) else repr(value)
    (args.output/"config.json").write_text(json.dumps({"train_check_env": asdict(cfg),
        "agent": asdict(agent), "play_check_env": asdict(play_cfg)}, default=encode, indent=2)+"\n")
    sources = [Path(__file__), *sorted((ROOT/"mjlab_training/src/ct_mjlab").rglob("*.py")),
               *SOURCES, Path(__file__).resolve().parents[3] / 'scripts/agent/isaac/check_isaac_standup_rewards.py', MODEL_XML]
    report["sources"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "passed": True, "optimizer_updates": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--probe-envs", type=int, default=0)
    parser.add_argument("--probe-seconds", type=float, default=5.)
    check(parser.parse_args())
