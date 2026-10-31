#!/usr/bin/env python3
"""Finite checks for the independent 0.27 m standing task; never calls learn().

--check-only checks configuration isolation, XML/pose geometry and the actual
assessment arithmetic on CPU. The default additionally creates four mjlab
environments and exercises resets, disturbance scheduling and checkpoint I/O.
The full check saves untrained-model.pt beside its report for negative behavior
checks and video smoke checks. Neither mode runs learning or proves stability.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "mjlab_training/src"))
XML_SHA256 = "cac8061de2850602bfd1421fc856bf217c4e5040a11c8f5d33cf1abe61d5f0cf"
TASK = "chuanliantui_stand"


def normalized(value):
    """Compare fresh configs by value, including callables but not object IDs."""
    if is_dataclass(value) and not isinstance(value, type):
        return normalized(asdict(value))
    if isinstance(value, dict):
        return {str(key): normalized(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return re.sub(r" at 0x[0-9a-f]+", "", f"{type(value).__name__}:{value!r}")


def equal(actual, expected, label):
    if normalized(actual) != normalized(expected):
        raise AssertionError(f"{label} changed: {normalized(actual)!r} != {normalized(expected)!r}")


def require(value, message):
    if not bool(value):
        raise AssertionError(message)


def expect_value_error(call, label):
    try:
        call()
    except ValueError as error:
        return str(error)
    raise AssertionError(f"{label}: incompatible input was accepted")


def configuration_checks():
    from ct_mjlab.envs import task_registry
    from ct_mjlab.envs.chuanliantui.model import MODEL_XML
    from ct_mjlab.envs.chuanliantui_stand import mdp, rewards

    digest = hashlib.sha256(MODEL_XML.read_bytes()).hexdigest()
    equal(digest, XML_SHA256, "shared XML SHA-256")
    require(TASK in task_registry.names(), "stand task is not registered")
    modes = {}
    changed_events = {"reset_pose", "huananhu_velocity_push", "huananhu_local_wrench",
                      "reset_observations", "huananhu_domain_randomization"}
    for play in (False, True):
        old, old_train = task_registry.get_cfgs("chuanliantui", play=play)
        original = normalized(old)
        new, new_train = task_registry.get_cfgs(TASK, play=play)
        equal(old, original, "old config mutated during new factory call")
        fresh_old, fresh_old_train = task_registry.get_cfgs("chuanliantui", play=play)
        equal(fresh_old, original, "old factory output after new factory call")
        equal(fresh_old_train, old_train, "old PPO factory")
        equal(new.sim, old.sim, "physics settings")
        equal(new.sim.mujoco.timestep, .002, "physics dt")
        equal(new.decimation, 5, "policy decimation")
        old_action, new_action = normalized(old.actions), normalized(new.actions)
        for key in ("class_type", "action_delay_s", "action_delay_choices_s", "randomize_delays"):
            old_action["pd"].pop(key)
            new_action["pd"].pop(key)
        equal(new_action["pd"].pop("action_delay_policy_steps"), [1, 2], "policy-step delay choices")
        equal(new_action["pd"].pop("randomization_scale"), 1. if play else .25, "motor randomization curriculum")
        expected_offset = old_action["pd"].pop("nominal_leg_offset_range")
        equal(new_action["pd"].pop("nominal_leg_offset_range"),
              expected_offset if play else [v*.25 for v in expected_offset], "randomized reference offset range")
        equal(new_action, old_action, "PD, action limits, reference angles and motor randomization")
        require(new.actions["pd"].action_delay_s is None, "fixed-time action delay remains")
        equal(new.actions["pd"].action_delay_choices_s, (), "seconds-based delay choices disabled")
        require(new.actions["pd"].randomize_delays, "policy-step delay must be resampled at reset")
        from ct_mjlab.envs.chuanliantui_stand import observations
        for name, func in (("actor", observations.actor_observation), ("critic", observations.critic_observation)):
            term = new.observations[name].terms["proprioception"]
            require(term.func is func, "standing observation implementation")
            equal(term.params["cfg"].noise_scale, .4, "training noise scale")
            equal(term.params["cfg"].delay_seconds, (.005, .010, .015), "original sensor delays")
            equal(term.params["cfg"].noise, not play, "observation noise enabled")
        equal(old.scene.num_envs, 1 if play else 4096, "old default environment count")
        equal(new.scene.num_envs, 1 if play else 2048, "new default environment count")
        old_scene, new_scene = normalized(old.scene), normalized(new.scene)
        new_scene["num_envs"] = old_scene["num_envs"]
        for field in ("pos", "joint_pos"):
            new_scene["entities"]["robot"]["init_state"][field] = (
                old_scene["entities"]["robot"]["init_state"][field])
        equal(new_scene, old_scene, "robot, terrain and sensors outside reset posture")
        equal(new.terminations, old.terminations, "termination configuration")
        equal(new.episode_length_s, 40., "standing episode duration")
        for name, event in old.events.items():
            if name not in changed_events:
                require(name in new.events, f"missing retained event: {name}")
                equal(new.events[name], event, f"retained domain randomization/event {name}")
        for name in ("huananhu_velocity_push", "huananhu_local_wrench"):
            require(name not in new.events, f"old repeated disturbance remains: {name}")
        equal(new.commands["standup"].recovery_window_s, 2., "continuous fallen timeout")
        equal(new.commands["standup"].ground_start_probability, .5, "mixed reset probability")
        require("stand_hold_position" not in new.rewards, "hidden position reward remains registered")
        require("stand_body_speed_cost" not in new.rewards, "rejected body cost remains registered")
        equal(new.rewards["stand_leg_speed_cost"].weight, -20., "true leg overspeed cost weight")
        equal(new.rewards["stand_body_speed_margin_cost"].weight, -40., "body speed margin cost weight")
        equal(new.rewards["stand_quiet_wheels"].weight, 1., "bounded wheel rest reward weight")
        for name in ("stand_still_lin_vel", "no_fork", "no_fork_square", "leg_joint_vel", "wheel_vel"):
            require(new.rewards[name].func is rewards.hold_cost_outside_recovery,
                    f"legacy static cost {name} still constrains recovery")
        equal(new_train.experiment_name, "chuanliantui_stand_v1", "separate experiment name")
        old_ppo, new_ppo = asdict(old_train), asdict(new_train)
        for name in ("experiment_name", "run_name"):
            old_ppo.pop(name, None)
            new_ppo.pop(name, None)
        equal(new_ppo, old_ppo, "network and PPO defaults")
        modes["play" if play else "train"] = {
            "old_factory_isolated": True, "nominal_pd_and_observation_delays_retained": True,
            "authorized_training_noise_scale": .4, "authorized_training_randomization_scale": .25,
            "action_delay_policy_steps": [1, 2],
            "old_num_envs": old.scene.num_envs, "new_num_envs": new.scene.num_envs,
            "retained_events": sorted(set(old.events) - changed_events),
            "physics_hz": 1. / new.sim.mujoco.timestep,
            "policy_hz": 1. / (new.sim.mujoco.timestep * new.decimation),
        }
    equal(new.commands["standup"].curriculum_stage, None, "default resumes saved stage")
    return {"xml_sha256": digest, "modes": modes,
            "curriculum_version": mdp.CURRICULUM_VERSION, "fresh_stage": 0}


def runner_contract_checks():
    from ct_mjlab.envs.chuanliantui.huananhu_runner import HuananhuOnPolicyRunner
    from ct_mjlab.envs.chuanliantui_stand.observations import OBSERVATION_CONTRACT
    from ct_mjlab.envs.chuanliantui.multimode_runner import (
        EXPERIMENT_PROFILE_CONTRACT as OLD_PROFILE, MultimodeOnPolicyRunner)
    from ct_mjlab.envs.chuanliantui.pd_action import CONTROL_CONTRACT
    from ct_mjlab.envs.chuanliantui.runner import DirectMotorOnPolicyRunner
    from ct_mjlab.envs.chuanliantui_stand.mdp import CURRICULUM_VERSION
    from ct_mjlab.envs.chuanliantui_stand.runner import EXPERIMENT_PROFILE_CONTRACT, StandOnPolicyRunner

    require(StandOnPolicyRunner.__bases__ == (DirectMotorOnPolicyRunner,),
            "stand runner must bypass legacy auto-initialization runner")
    require(not issubclass(StandOnPolicyRunner, HuananhuOnPolicyRunner), "legacy init runner inherited")
    require(not hasattr(StandOnPolicyRunner, "initialize_from"), "legacy initialization entry inherited")
    require("os.environ" not in inspect.getsource(StandOnPolicyRunner.learn),
            "standing learn reads legacy environment initialization")
    info = {"observation_contract": OBSERVATION_CONTRACT, "control_contract": CONTROL_CONTRACT,
            "experiment_profile_contract": EXPERIMENT_PROFILE_CONTRACT,
            "physics_dt_s": .002, "policy_dt_s": .01,
            "curriculum_version": CURRICULUM_VERSION, "stand_training_iteration": 2000,
            "stand_curriculum_stage": 0}
    errors = {}
    with tempfile.TemporaryDirectory(prefix="stand-contract-") as tmp:
        new_file, old_file = Path(tmp) / "stand.pt", Path(tmp) / "old.pt"
        torch.save({"infos": info}, new_file)
        old_info = info | {"experiment_profile_contract": OLD_PROFILE, "huananhu_training_iteration": 2000}
        torch.save({"infos": old_info}, old_file)
        # Load rejects before touching runner state; no simulator or PPO is needed.
        errors["new_rejects_old"] = expect_value_error(
            lambda: StandOnPolicyRunner.load(None, str(old_file)), "new task loads old profile")
        errors["old_rejects_new"] = expect_value_error(
            lambda: MultimodeOnPolicyRunner.load(None, str(new_file)), "old task loads new profile")
        torch.save({"infos": info | {
            "observation_contract": "chuanliantui-stand-41-82-planar-velocity-v1",
            "experiment_profile_contract": "chuanliantui-stand-41-82-v1"}}, old_file)
        errors["rejects_historical_41"] = expect_value_error(
            lambda: StandOnPolicyRunner.load(None, str(old_file)), "39-input task loads a 41-input checkpoint")
        for key, bad in (("physics_dt_s", .001), ("policy_dt_s", .02),
                         ("curriculum_version", "other"), ("stand_training_iteration", -1),
                         ("stand_curriculum_stage", -1), ("stand_curriculum_stage", 3)):
            torch.save({"infos": info | {key: bad}}, new_file)
            errors[key] = expect_value_error(
                lambda: StandOnPolicyRunner.load(None, str(new_file)), key)
    return {"legacy_initializer_not_in_method_resolution": True, "guards": errors,
            "scope": "profile/timing/curriculum guards only; real parameter roundtrip is in runtime checks"}


def reward_sample_checks():
    from ct_mjlab.envs.chuanliantui_stand import rewards

    state = SimpleNamespace(leg_v_adr=torch.arange(4), leg_velocity_sum=torch.zeros(2),
                            leg_absolute_velocity_sum=torch.zeros(2),
                            leg_speed_peak=torch.zeros(2),
                            wheel_v_adr=torch.arange(4, 6), wheel_velocity_sum=torch.zeros(2),
                            body_xy_v_adr=torch.arange(6, 8), body_speed_peak=torch.zeros(2),
                            leg_velocity_count=0, leg_velocity_step=-1,
                            quiet_ready=torch.tensor([False, False]),
                            settle_required=torch.tensor([False, False]),
                            stable_time=torch.tensor([.5, .5]),
                            fallen=torch.tensor([False, False]), push_applied=torch.tensor([False, False]),
                            recover=torch.tensor([False, False]), airborne=torch.tensor([False, False]),
                            numerical_failure=torch.tensor([False, False]))
    env = SimpleNamespace(common_step_counter=0,
                          command_manager=SimpleNamespace(get_term=lambda _: state),
                          scene={"robot": SimpleNamespace(data=SimpleNamespace(
                              root_link_lin_vel_b=torch.zeros(2, 3)))},
                          sim=SimpleNamespace(data=SimpleNamespace(qvel=torch.zeros(2, 8))))
    for sample in range(5):
        # 只有中间2ms采样有腿速，末个策略端点为零。
        env.sim.data.qvel.zero_()
        if sample == 2:
            env.sim.data.qvel[0, :4] = 1.
            env.sim.data.qvel[0, 4:] = 1.
            env.sim.data.qvel[0, 6:] = torch.tensor([.03, .04])
        rewards.sample_leg_velocity(env)
    equal(state.leg_velocity_count, 5, "physical leg reward sample count")
    torch.testing.assert_close(state.leg_velocity_sum, torch.tensor([1., 0.]))
    torch.testing.assert_close(state.leg_absolute_velocity_sum, torch.tensor([1., 0.]))
    def quiet_reward(standing=True):
        with patch.object(rewards, "standing_mask", return_value=torch.full((2,), standing)):
            return rewards.quiet_legs(env)

    value = quiet_reward()
    torch.testing.assert_close(value, torch.tensor([1. / 6., 1.]))
    velocity = env.scene["robot"].data.root_link_lin_vel_b
    velocity[:, 0] = .03
    rolling = quiet_reward()
    torch.testing.assert_close(rolling, value)
    velocity[:, 0] = -.03
    torch.testing.assert_close(quiet_reward(), rolling)
    velocity.zero_()
    torch.testing.assert_close(quiet_reward(), value)
    def wheel_reward(supported=True):
        with patch.object(rewards, "supported_pose_mask", return_value=torch.full((2,), supported)):
            return rewards.quiet_wheels(env)

    wheel_value = wheel_reward()
    torch.testing.assert_close(wheel_value, torch.tensor([1. / 81., 1.]))
    def leg_cost(supported=True):
        with patch.object(rewards, "supported_pose_mask", return_value=torch.full((2,), supported)):
            return rewards.quiet_leg_speed_cost(env)

    def body_cost(supported=True):
        with patch.object(rewards, "supported_pose_mask", return_value=torch.full((2,), supported)):
            return rewards.body_speed_margin_cost(env)

    torch.testing.assert_close(body_cost(), torch.tensor([.02, 0.]), atol=1e-7, rtol=0.)
    require(not bool(body_cost(False).any()), "unsupported pose receives body margin cost")

    torch.testing.assert_close(leg_cost(), torch.tensor([.5, 0.]))
    require(not bool(leg_cost(False).any()), "unsupported pose receives quiet-leg speed cost")
    require(bool((wheel_reward(False) == 0.).all()), "unsupported wheel rest was rewarded")
    state.settle_required.fill_(True)
    require(not bool(quiet_reward().any()), "legs constrained before recovery settling")
    require(not bool(wheel_reward().any()), "wheels constrained before recovery settling")
    require(not bool(leg_cost().any()), "linear leg speed cost constrains ground get-up")
    require(not bool(body_cost().any()), "body margin cost constrains ground get-up")
    state.settle_required.zero_()
    require(bool((quiet_reward(False) == 0.).all()), "non-standing pose receives quiet reward")
    state.stable_time[:] = .49
    require(bool((quiet_reward() == 0.).all()), "initial pose timer was bypassed")
    require(bool((wheel_reward() == 0.).all()), "wheel initial pose timer was bypassed")
    require(not bool(leg_cost().any()), "linear leg cost bypassed initial pose timer")
    require(not bool(body_cost().any()), "body margin cost bypassed initial pose timer")
    state.stable_time[:] = .5
    state.fallen[:] = True
    require(bool((quiet_reward() == 0.).all()), "first fallen tick was not exempted")
    require(bool((wheel_reward() == 0.).all()), "fallen wheels were constrained")
    require(not bool(leg_cost().any()), "linear leg speed cost constrains first fallen tick")
    require(not bool(body_cost().any()), "body margin cost constrains first fallen tick")
    state.fallen[:] = False
    state.push_applied[:] = True
    require(bool((quiet_reward() == 0.).all()), "push recovery received quiet reward before restabilization")
    require(bool((wheel_reward() == 0.).all()), "pushed wheels were constrained before restabilization")
    require(not bool(leg_cost().any()), "linear leg cost constrains push recovery")
    require(not bool(body_cost().any()), "body margin cost constrains push recovery")
    state.quiet_ready[:] = True
    torch.testing.assert_close(quiet_reward(), value)
    torch.testing.assert_close(wheel_reward(), wheel_value)
    torch.testing.assert_close(leg_cost(), torch.tensor([.5, 0.]))
    state.recover[0], state.airborne[1] = True, True
    require(bool((quiet_reward() == 0.).all()), "recovery/airborne leg reward was not exempted")
    require(bool((wheel_reward() == 0.).all()), "recovery/airborne wheel reward was not exempted")
    require(not bool(leg_cost().any()), "linear leg cost constrains recovery or airborne phase")
    require(not bool(body_cost().any()), "body margin cost constrains recovery or airborne phase")
    env.common_step_counter = 1
    rewards.sample_leg_velocity(env)
    equal(state.leg_velocity_count, 1, "next policy step resets sample counter")
    require(bool((state.leg_velocity_sum == 0.).all()), "next policy step retained old velocities")
    require(not bool(state.leg_absolute_velocity_sum.any()), "next policy step retained absolute leg speeds")
    require(not bool(state.leg_speed_peak.any()), "next policy step retained leg speed peak")
    require(bool((state.wheel_velocity_sum == 0.).all()), "next policy step retained wheel velocities")
    require(not bool(state.body_speed_peak.any()), "next policy step retained body speed peak")
    state.recover.zero_()
    state.airborne.zero_()
    env.common_step_counter = 2
    env.sim.data.qvel[0, 4:] = -2.
    for _ in range(5):
        rewards.sample_leg_velocity(env)
    # Both signs use the same bounded reward, including the original five substeps.
    unchanged = env.sim.data.qvel.clone()
    torch.testing.assert_close(wheel_reward(), torch.tensor([1. / 1601., 1.]))
    torch.testing.assert_close(env.sim.data.qvel, unchanged)
    env.sim.data.qvel[0, 4:] = 2.
    env.common_step_counter += 1
    for _ in range(5):
        rewards.sample_leg_velocity(env)
    torch.testing.assert_close(wheel_reward(), torch.tensor([1. / 1601., 1.]))
    env.common_step_counter = 4
    env.sim.data.qvel.zero_()
    env.sim.data.qvel[0, :4] = .02
    env.sim.data.qvel[1, :4] = -.02
    for _ in range(5):
        rewards.sample_leg_velocity(env)
    torch.testing.assert_close(leg_cost(), torch.zeros(2))
    env.scene["robot"].data.root_link_lin_vel_b[:, 0] = .04
    torch.testing.assert_close(leg_cost(), torch.zeros(2))
    state.leg_speed_peak[:] = .5
    torch.testing.assert_close(leg_cost(), torch.zeros(2))
    env.common_step_counter = 5
    env.sim.data.qvel[0, :4] = .6
    env.sim.data.qvel[1, :4] = -.6
    for _ in range(5):
        rewards.sample_leg_velocity(env)
    torch.testing.assert_close(leg_cost(), torch.full((2,), .1))
    state.numerical_failure[0] = True
    torch.testing.assert_close(leg_cost(), torch.tensor([0., .1]))
    state.body_speed_peak[:] = torch.tensor([.05, .02])
    require(not bool(body_cost().any()), "numerical failure body cost is not exempt")
    state.numerical_failure.zero_()
    state.body_speed_peak[:] = torch.tensor([.03, .02])
    torch.testing.assert_close(body_cost(), torch.zeros(2))
    state.body_speed_peak[:] = .05
    state.stable_time[0] = 0.
    torch.testing.assert_close(body_cost(), torch.tensor([0., .02]), atol=1e-7, rtol=0.)
    state.stable_time[:] = .5
    state.body_speed_peak[:] = .5
    torch.testing.assert_close(body_cost(), torch.full((2,), .1))
    return {"five_substeps_accumulated": True, "middle_substep_velocity_not_omitted": True,
            "recovery_and_airborne_exempted": True, "next_policy_step_isolated": True,
            "quiet_phase_does_not_require_quiet_legs": True,
            "pure_hold_uses_current_standing_mask": True,
            "push_requires_continuous_restabilization": True, "first_fallen_tick_exempted": True,
            "wheel_middle_substep_not_omitted": True, "wheel_rest_is_reward_only": True,
            "wheel_rest_uses_five_substep_mean_square": True, "wheel_velocity_is_not_modified": True,
            "leg_overspeed_cost_keeps_worst_of_five_substeps": True,
            "leg_overspeed_both_signs_and_below_limit_exempt": True,
            "leg_overspeed_cost_not_scaled_away_by_body_motion": True,
            "body_cost_world_xy_norm_uses_worst_physical_substep": True,
            "body_cost_zero_within_0_03_margin_and_capped_at_0_1": True,
            "body_cost_recovery_and_partial_reset_exempt": True,
            "leg_overspeed_cost_ground_recovery_airborne_exempt": True}


def mixed_sampling_checks():
    from ct_mjlab.envs.chuanliantui_stand import mdp

    selected = torch.tensor([1, 3, 4, 7])
    state = SimpleNamespace(cfg=SimpleNamespace(ground_start_probability=.5),
        settle_required=torch.ones(8, dtype=torch.bool), ground_clearance_lift=torch.arange(8) * .001)
    env = SimpleNamespace(device="cpu", num_envs=8,
        command_manager=SimpleNamespace(get_term=lambda _: state))
    untouched = state.ground_clearance_lift[[0, 2, 5, 6]].clone()
    with patch.object(mdp.torch, "rand", return_value=torch.tensor([.1, .9, .4, .8])), \
         patch.object(mdp, "original_reset_pose") as original, \
         patch.object(mdp, "_clear_ground_penetration") as clearance, \
         patch.object(mdp, "_reset_near_standing_pose") as near:
        mdp.reset_pose(env, selected)
        torch.testing.assert_close(original.call_args.args[1], torch.tensor([1, 4]))
        equal(original.call_args.kwargs, {"ground_probability": 1.}, "original ground delegate arguments")
        torch.testing.assert_close(clearance.call_args.args[1], torch.tensor([1, 4]))
        torch.testing.assert_close(near.call_args.args[1], torch.tensor([3, 7]))
    torch.testing.assert_close(state.ground_clearance_lift[[0, 2, 5, 6]], untouched)
    torch.testing.assert_close(state.settle_required, torch.tensor([True, True, True, False, True, True, True, False]))
    state.cfg.ground_start_probability = 0.
    with patch.object(mdp.torch, "rand", side_effect=AssertionError("pinned near path added random draw")), \
         patch.object(mdp, "original_reset_pose") as original, \
         patch.object(mdp, "_reset_near_standing_pose") as near:
        mdp.reset_pose(env, selected)
        original.assert_not_called()
        torch.testing.assert_close(near.call_args.args[1], selected)
    return {"half_probability_routes_selected_envs_only": True,
            "actual_original_reset_delegated": True, "pinned_near_path_has_no_extra_random_draw": True}


def position_and_recovery_reward_checks():
    from ct_mjlab.envs.chuanliantui_stand import rewards

    class Scene(dict):
        env_origins = torch.zeros(2, 3)

    state = SimpleNamespace(quiet_ready=torch.zeros(2, dtype=torch.bool),
        settle_required=torch.zeros(2, dtype=torch.bool), recover=torch.zeros(2, dtype=torch.bool),
        fallen=torch.zeros(2, dtype=torch.bool), airborne=torch.zeros(2, dtype=torch.bool),
        numerical_failure=torch.zeros(2, dtype=torch.bool), hold_anchor_xy=torch.zeros(2, 2),
        hold_anchor_valid=torch.zeros(2, dtype=torch.bool))
    data = SimpleNamespace(root_link_pos_w=torch.tensor([[1., 2., .27], [3., 4., .27]]),
                           projected_gravity_b=torch.tensor([[0., 0., -1.], [0., 0., 1.]]))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: state),
                          scene=Scene(robot=SimpleNamespace(data=data)))
    with patch.object(rewards, "supported_pose_mask", return_value=torch.ones(2, dtype=torch.bool)):
        torch.testing.assert_close(rewards.hold_position(env), torch.zeros(2))
        require(not bool(state.hold_anchor_valid.any()), "position anchor set before body qualification")
        state.quiet_ready.fill_(True)
        torch.testing.assert_close(rewards.hold_position(env), torch.zeros(2))
        original = state.hold_anchor_xy.clone()
        data.root_link_pos_w[:, 0] += .02
        torch.testing.assert_close(rewards.hold_position(env), torch.full((2,), .02), atol=1e-6, rtol=0.)
        state.quiet_ready.zero_()
        data.root_link_pos_w[:, 0] += .2
        torch.testing.assert_close(rewards.hold_position(env), torch.full((2,), .1))
        torch.testing.assert_close(state.hold_anchor_xy, original)
        state.recover[0] = True
        state.settle_required[0] = True
        torch.testing.assert_close(rewards.hold_position(env), torch.tensor([0., .1]))
        require(not bool(state.hold_anchor_valid[0]) and bool(state.hold_anchor_valid[1]),
                "recovery anchor invalidation affected the wrong environment")
        state.recover[0] = False
        state.settle_required[0] = False
        state.quiet_ready[0] = True
        torch.testing.assert_close(rewards.hold_position(env), torch.tensor([0., .1]))
        torch.testing.assert_close(state.hold_anchor_xy[0], data.root_link_pos_w[0, :2])
        torch.testing.assert_close(state.hold_anchor_xy[1], original[1])
    state.settle_required.fill_(True)
    torch.testing.assert_close(rewards.recovery_upright(env), torch.tensor([1., -1.]))
    data.root_link_pos_w[:, 2] = .16
    torch.testing.assert_close(rewards.recovery_upright(env), torch.tensor([0., -2.]))
    data.root_link_pos_w[:, 2] = .215
    torch.testing.assert_close(rewards.recovery_upright(env), torch.tensor([.5, -1.5]), atol=1e-6, rtol=0.)
    # World translation must not change height progress relative to the ground.
    env.scene.env_origins[:, 2] = 1.
    data.root_link_pos_w[:, 2] += 1.
    torch.testing.assert_close(rewards.recovery_upright(env), torch.tensor([.5, -1.5]), atol=1e-6, rtol=0.)
    env.scene.env_origins.zero_()
    data.root_link_pos_w[:, 2] = .27
    data.projected_gravity_b[:, 2] = 0.
    torch.testing.assert_close(rewards.recovery_upright(env), torch.zeros(2))
    with patch.object(rewards.huananhu_rewards, "reward", return_value=torch.tensor([2., 3.])):
        torch.testing.assert_close(rewards.orientation_outside_recovery(env, name="flat_orientation_y_exp"),
                                   torch.zeros(2))
        state.settle_required.zero_()
        torch.testing.assert_close(rewards.orientation_outside_recovery(env, name="flat_orientation_y_exp"),
                                   torch.tensor([2., 3.]))
    return {"position_anchor_requires_body_qualification_only": True,
            "no_reanchor_after_drift_or_temporary_speed_increase": True,
            "cost_capped_at_10cm": True, "recovery_can_lock_a_new_position": True,
            "other_environment_anchor_preserved": True,
            "recovery_orientation_distinguishes_upside_down": True,
            "ordinary_standing_orientation_formulas_retained": True}


def recovery_settle_checks():
    from ct_mjlab.envs.chuanliantui_stand import mdp

    state = object.__new__(mdp.StandCommand)
    state._env = SimpleNamespace(step_dt=.01, cfg=SimpleNamespace(episode_length_s=40.))
    state.cfg = SimpleNamespace(episode_length_s=40.)
    state.ground_start = torch.ones(2, dtype=torch.bool)
    state.getting_up = torch.zeros(2, dtype=torch.bool)
    state.recovered = torch.zeros(2, dtype=torch.bool)
    state.fallen = torch.zeros(2, dtype=torch.bool)
    state.airborne = torch.zeros(2, dtype=torch.bool)
    state.strict_stable_time = torch.zeros(2)
    state.quiet_ready = torch.zeros(2, dtype=torch.bool)
    state.settle_required = torch.ones(2, dtype=torch.bool)
    state.metrics = {"strict_standing_fraction": torch.zeros(2), "curriculum_stage": torch.zeros(2)}
    state.curriculum_stage = 0
    # No leg velocity is present: eligibility depends on body stability alone.
    with patch.object(mdp.MultimodeCommand, "advance"), \
         patch.object(mdp, "standing_mask", return_value=torch.ones(2, dtype=torch.bool)):
        state.advance()
        require(bool(state.settle_required.all()), "ground get-up was not exempted")
        state.recovered.fill_(True)
        for _ in range(49):
            state.advance()
        require(bool(state.settle_required.all()), "recovery settling shorter than .5s")
        for _ in range(11):
            state.advance()
        require(not bool(state.settle_required.any()), "hold rewards did not reenable after physical settling")
        state.ground_start[1] = False
        state.getting_up[1] = True
        state.recovered[1] = False
        state.fallen[1] = True
        state.advance()
        require(bool(state.settle_required[1]) and not bool(state.settle_required[0]),
                "mid-episode fall did not rearm only its own exemption")
    return {"ground_getup_and_mid_episode_recovery_exempted": True,
            "requires_continuous_body_stability": True, "no_leg_motion_gate": True}


def minimum_collision_z(model, data):
    from ct_mjlab.envs.chuanliantui.huananhu_poses import _mesh_vertices

    values = []
    for gid in range(model.ngeom):
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            raise AssertionError(f"Unexpected non-mesh robot collision geom {model.geom(gid).name}")
        values.append(float(_mesh_vertices(model, data, gid)[:, 2].min()))
    require(values, "robot has no collision geometry")
    return min(values)


def pose_checks():
    from ct_mjlab.envs.chuanliantui.model import closed_chain_residual, load_model, standing_motor_positions
    from ct_mjlab.envs.chuanliantui_stand import mdp

    model = load_model()
    data = mujoco.MjData(model)
    pose = mdp.standing_pose()
    for name, value in pose["joint_positions"].items():
        joint = model.joint(name)
        data.qpos[joint.qposadr[0]] = value
        if joint.limited:
            require(joint.range[0] - 1e-8 <= value <= joint.range[1] + 1e-8,
                    f"initial joint {name} outside XML limit")
    data.qpos[2] = pose["height"]
    mujoco.mj_forward(model, data)
    residual = closed_chain_residual(model, data)
    require(residual <= 1e-8, f"six closed chains do not close: {residual} m")
    require(abs(float(data.qpos[2]) - .27) <= .005, "nominal reset height not near 0.27 m")
    motors = tuple(pose["joint_positions"][name] for name in mdp.LEG_MOTORS)
    require(not np.allclose(motors, standing_motor_positions()),
            "near 0.27 m reset incorrectly reused the old PD reference pose")
    return {"height_m": float(data.qpos[2]), "closed_chain_residual_m": residual,
            "lowest_collision_z_m": minimum_collision_z(model, data),
            "all_joint_limits_respected": True, "reset_is_distinct_from_pd_reference": True}


def posture_reward_checks():
    from ct_mjlab.envs.chuanliantui_stand import rewards
    from ct_mjlab.envs import task_registry

    before, _ = task_registry.get_cfgs(TASK)
    targets = rewards.balanced_standing_motor_targets()
    after, _ = task_registry.get_cfgs(TASK)
    equal(before.actions, after.actions, "reward pose solver changed PD reference or limits")
    target = torch.tensor(targets, dtype=torch.float32)
    state = SimpleNamespace(stable_time=torch.full((2,), .5),
        settle_required=torch.zeros(2, dtype=torch.bool),
        recover=torch.zeros(2, dtype=torch.bool), airborne=torch.zeros(2, dtype=torch.bool),
        fallen=torch.zeros(2, dtype=torch.bool), push_applied=torch.zeros(2, dtype=torch.bool),
        quiet_ready=torch.zeros(2, dtype=torch.bool), numerical_failure=torch.zeros(2, dtype=torch.bool), posture_q_adr=torch.arange(4),
        posture_target=target)
    velocity = torch.tensor([[0., 0., 0.], [.01, 0., 0.]])
    env = SimpleNamespace(device="cpu", scene={"robot": SimpleNamespace(data=SimpleNamespace(
        root_link_lin_vel_b=velocity))}, sim=SimpleNamespace(data=SimpleNamespace(
        qpos=target.expand(2, -1).clone())),
        command_manager=SimpleNamespace(get_term=lambda _: state))
    env.sim.data.qpos[1] += rewards.POSTURE_SCALE_RAD
    unchanged = env.sim.data.qpos.clone()
    with patch.object(rewards, "supported_pose_mask", return_value=torch.ones(2, dtype=torch.bool)):
        torch.testing.assert_close(rewards.standing_posture(env), torch.tensor([1., np.exp(-1.)], dtype=torch.float32))
        torch.testing.assert_close(rewards.low_velocity_hold(env), torch.tensor([1., .5]))
        velocity.zero_()
        torch.testing.assert_close(rewards.standing_posture(env),
            torch.tensor([1., np.exp(-1.)], dtype=torch.float32))
        for flag in ("recover", "airborne", "fallen", "settle_required", "numerical_failure"):
            getattr(state, flag).fill_(True)
            require(not bool(rewards.standing_posture(env).any()), f"posture reward constrains {flag}")
            if flag == "settle_required":
                require(not bool(rewards.low_velocity_hold(env).any()), "low-speed hold before recovery settling")
            getattr(state, flag).zero_()
        state.push_applied.fill_(True)
        require(not bool(rewards.standing_posture(env).any()), "posture reward before push restabilization")
        state.quiet_ready.fill_(True)
        require(bool((rewards.standing_posture(env) > 0).all()), "posture reward not restored")
    torch.testing.assert_close(env.sim.data.qpos, unchanged)
    return {"physical_motor_target_rad": targets, "target_geometry_validated_by_solver": True,
            "pose_is_reward_only": True, "recovery_airborne_fallen_exempted": True,
            "posture_target_retained_while_braking": True,
            "body_still_speed_scale_m_s": rewards.BODY_STILL_SPEED_SCALE_M_S,
            "push_requires_restabilization": True, "speed_scale_m_s": rewards.HOLD_SPEED_SCALE_M_S,
            "posture_scale_rad": rewards.POSTURE_SCALE_RAD}


def assessment_checks():
    from scripts.agent.stand import assess_chuanliantui_stand as assessment

    verdict = assessment.window_verdict
    base = dict(duration_s=20., maximum_displacement_m=0., maximum_tilt_deg=0.,
                maximum_height_error_m=0., wheel_support_fraction=1., base_contact_fraction=0.,
                leg_peak_to_peak_rad=np.zeros(4), maximum_body_speed_m_s=0.,
                leg_max_abs_speed_rad_s=np.zeros(4), failed=False)
    cases = []

    def case(name, expected, **overrides):
        passed = bool(verdict(**(base | overrides)))
        require(passed == expected, f"criterion case {name}: got {passed}, expected {expected}")
        cases.append(name)

    case("complete_20_seconds", True)
    case("19_99_seconds_incomplete", False, duration_s=19.99)
    case("leg_speed_exactly_0_5_rad_s", True, leg_max_abs_speed_rad_s=np.full(4, .5))
    for joint in range(4):
        spike = np.zeros(4); spike[joint] = .5001
        case(f"leg_{joint}_speed_above_limit", False, leg_max_abs_speed_rad_s=spike)
    case("body_planar_speed_boundary", True, maximum_body_speed_m_s=.05)
    case("body_planar_speed_above_boundary", False, maximum_body_speed_m_s=.050001)
    case("angle_range_and_displacement_are_diagnostic", True,
         maximum_displacement_m=2., leg_peak_to_peak_rad=np.full(4, np.deg2rad(60.)))
    case("tilt_boundary", True, maximum_tilt_deg=25.)
    case("tilt_over_boundary", False, maximum_tilt_deg=25.0001)
    case("height_boundary", True, maximum_height_error_m=.03)
    case("height_over_boundary", False, maximum_height_error_m=.030001)
    case("support_boundary", True, wheel_support_fraction=.95)
    case("support_below_boundary", False, wheel_support_fraction=.94999)
    case("base_contact_boundary", True, base_contact_fraction=.01)
    case("base_contact_above_boundary", False, base_contact_fraction=.01001)
    case("fallen_or_reset", False, failed=True)
    case("nonfinite_joint_state", False, leg_peak_to_peak_rad=np.array([np.nan, 0., 0., 0.]))
    samples = {
        "q": np.zeros((10000, 6)), "qd": np.zeros((10000, 6)),
        "xy": np.zeros((10000, 2)), "body_velocity_xy": np.zeros((10000, 2)),
        "tilt_deg": np.zeros((10000, 2)),
        "height_m": np.full(10000, .27), "wheel_supported": np.ones(10000, dtype=bool),
        "base_contact": np.zeros(10000, dtype=bool), "finite": np.ones(10000, dtype=bool),
        "anchor_xy": np.zeros(2),
    }
    summary = assessment.summarize_window(samples, dt=.002)
    require(bool(summary["passed"]), "perfect raw 500 Hz window did not pass")
    samples["tilt_deg"][:] = 20.
    combined = assessment.summarize_window(samples, dt=.002)
    require(27.9 < combined["maximum_tilt_deg"] < 28.1 and not combined["passed"],
            "combined 20 degree roll/pitch escaped the 25 degree tilt limit")
    cases.append("combined_roll_pitch_20_degrees_fails")
    samples["tilt_deg"][:] = 0.
    # A 2 ms speed spike lies between policy endpoints; neither averaging nor RMS is allowed.
    for sign in (1., -1.):
        samples["qd"][:] = 0.
        samples["qd"][2, 1] = sign * .5001
        require(np.abs(samples["qd"][4::5]).max() == 0., "spike is not between policy samples")
        require(not assessment.summarize_window(samples)["passed"], "single leg speed spike was omitted")
    samples["qd"][:] = 0.
    samples["body_velocity_xy"][2] = [.04, .04]
    require(not assessment.summarize_window(samples)["passed"], "planar speed used components instead of norm")
    samples["body_velocity_xy"][:] = 0.
    samples["xy"][:, 0] = 2.
    samples["q"][2, 1] = np.deg2rad(10.)
    require(assessment.summarize_window(samples)["passed"], "removed displacement/angle limits still reject")
    samples["body_velocity_xy"][2, 0] = np.nan
    require(not assessment.summarize_window(samples)["passed"], "nonfinite body speed passed")
    return {"criteria_version": assessment.CRITERIA_VERSION, "cases": cases,
            "single_positive_negative_leg_speed_spike_detected": True,
            "single_planar_speed_spike_detected": True,
            "removed_angle_and_position_limits_are_diagnostic": True,
            "state_machine": acceptance_state_checks(assessment),
            "production_recorder": recorder_checks(assessment)}


def recorder_checks(assessment):
    """CPU fixture calls the real recorder on every physical sample."""
    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    scene = Scene(robot=SimpleNamespace(
        find_joints=lambda names, preserve_order: (list(range(6)), names),
        indexing=SimpleNamespace(joint_q_adr=torch.arange(7, 13), joint_v_adr=torch.arange(6, 12),
                                 free_joint_q_adr=torch.arange(7), free_joint_v_adr=torch.arange(6))),
        wheel_contact=SimpleNamespace(data=SimpleNamespace(force=torch.tensor([[[0., 0., -50.], [0., 0., -50.]]]))),
        base_contact=SimpleNamespace(data=SimpleNamespace(force=torch.zeros(1, 1, 3))))
    data = SimpleNamespace(qpos=torch.zeros(1, 13), qvel=torch.zeros(1, 12))
    data.qpos[0, 2], data.qpos[0, 3] = .27, 1.
    env = SimpleNamespace(num_envs=1, device="cpu", scene=scene, sim=SimpleNamespace(data=data,
        model=SimpleNamespace(body_mass=torch.tensor([[10.]])),
        mj_model=SimpleNamespace(opt=SimpleNamespace(gravity=np.array([0., 0., -9.81])))))
    recorder = assessment.PhysicsRecorder(env, 5)
    recorder.arm(0)
    for index in range(5):
        data.qvel[0, 7] = -.5001 if index == 2 else 0.
        recorder.sample()
    require(bool(recorder.violations()[0]), "production recorder missed substep joint-speed spike")
    require(float(recorder.trace[4, 0, 7]) == 0., "synthetic endpoint was not still")
    recorder.stop(0)
    expect_value_error(lambda: recorder.arm(0), "production recorder reanchor")
    # 起点也是20s窗口的真实极值，不能只保留后续采样中的最大最小。
    data.qvel[0, 7] = .5001
    anchored = assessment.PhysicsRecorder(env, 1)
    anchored.arm(0)
    data.qvel[0, 7] = 0.
    anchored.sample()
    require(bool(anchored.violations()[0]), "production recorder omitted anchor speed")
    body = assessment.PhysicsRecorder(env, 5)
    body.arm(0)
    for index in range(5):
        data.qvel[0, :2] = torch.tensor([.04, .04]) if index == 2 else 0.
        body.sample()
    require(bool(body.violations()[0]), "production recorder omitted 2ms planar speed spike")
    torch.testing.assert_close(body.trace[2, 0, 20:22], torch.tensor([.04, .04]))
    data.qvel.zero_()
    diagnostic = assessment.PhysicsRecorder(env, 1)
    diagnostic.arm(0)
    data.qpos[0, 0], data.qpos[0, 8] = 2., .2
    diagnostic.sample()
    require(not bool(diagnostic.violations()[0]), "removed position/angle criteria still active")
    from mjlab.utils.lab_api.math import quat_from_euler_xyz
    angle = torch.tensor([np.deg2rad(20.)], dtype=torch.float32)
    data.qpos[0, 3:7] = quat_from_euler_xyz(angle, angle, torch.zeros_like(angle))[0]
    tilted = assessment.PhysicsRecorder(env, 1)
    tilted.arm(0)
    tilted.sample()
    require(27.9 < float(tilted.max_tilt[0]) < 28.1 and bool(tilted.violations()[0]),
            "production recorder missed combined tilt")
    return {"middle_physics_tick_not_omitted": True, "single_anchor_enforced": True,
            "anchor_velocity_is_in_extrema": True, "body_speed_substep_spike_detected": True,
            "combined_roll_pitch_uses_total_tilt": True,
            "scope": "synthetic CPU arithmetic; real wheel force direction is checked separately in runtime"}


def acceptance_state_checks(assessment):
    """Exercise the production anchor state machine, not a duplicate predicate."""
    state = assessment.AcceptanceState("hold")
    result = state.update(1., stable=True, fallen=False)
    require(result == "anchor", f"stable static case did not anchor: {result!r}")
    first_anchor = state.anchor_s
    state.update(2., stable=False, fallen=True)
    state.update(3., stable=True, fallen=False)
    equal(state.anchor_s, first_anchor, "anchor after a fall")
    equal(state.phase, "failed", "fall during measurement must latch failure")
    short = assessment.AcceptanceState("hold")
    short.update(1., stable=True, fallen=False)
    expect_value_error(lambda: short.finish(20.99), "19.99 s state-machine window")
    equal(short.phase, "measuring", "incomplete window")
    short.finish(21.)
    equal(short.phase, "done", "complete window")
    late = assessment.AcceptanceState("hold")
    late.update(5.001, stable=True, fallen=False)
    equal(late.phase, "failed", "first stable state after 5 s deadline")
    pushed = assessment.AcceptanceState("push")
    require(pushed.update(5., stable=True, fallen=False) == "deliver", "push not requested after stability")
    pushed.mark_delivered(5.)
    pushed.update(5.1, stable=False, fallen=True)
    require(pushed.update(6., stable=True, fallen=False) == "anchor", "recovered push case did not anchor")
    pushed_anchor = pushed.anchor_s
    pushed.fail("synthetic displacement failure")
    require(pushed.update(9., stable=True, fallen=False) != "anchor", "failed push case reanchored")
    equal(pushed.anchor_s, pushed_anchor, "failed pushed anchor")
    return {"no_reanchor_after_failure": True, "duration_boundary": True,
            "start_deadline": True, "recovery_anchor_after_delivery": True}


def runtime_pose_checks(env):
    from ct_mjlab.envs.chuanliantui.model import closed_chain_residual, load_model
    from ct_mjlab.envs.chuanliantui_stand.mdp import RESET_TILT_RAD

    model, rows = load_model(), []
    robot = env.scene["robot"]
    data = mujoco.MjData(model)
    state = env.command_manager.get_term("standup")
    for index in range(env.num_envs):
        root = env.sim.data.qpos[index, robot.indexing.free_joint_q_adr].cpu().numpy()
        data.qpos[:7] = root
        data.qpos[:3] -= env.scene.env_origins[index].cpu().numpy()
        for local, name in enumerate(robot.joint_names):
            data.qpos[model.joint(name).qposadr[0]] = float(
                env.sim.data.qpos[index, robot.indexing.joint_q_adr[local]])
        mujoco.mj_forward(model, data)
        residual, minimum_z = closed_chain_residual(model, data), minimum_collision_z(model, data)
        matrix = data.xmat[model.body("base_link").id].reshape(3, 3)
        roll, pitch = np.arctan2(matrix[2, 1], matrix[2, 2]), np.arcsin(-matrix[2, 0])
        require(residual <= 2e-7, f"runtime reset {index} violates six closed chains: {residual}")
        ground = bool(state.ground_start[index])
        require(minimum_z >= -2e-7, f"runtime reset {index} penetrates floor: {minimum_z}")
        if ground:
            require(.25 - 1e-6 <= data.qpos[2] <= .25401, f"ground reset {index} height changed excessively")
            require(max(abs(roll), abs(pitch)) < 1e-6, f"original ground reset {index} is not level")
            require(bool(state.settle_required[index]), "ground reset bypassed reward exemption")
        else:
            require(abs(minimum_z - .002) <= 2e-7, f"near reset {index} collision clearance {minimum_z}")
            require(max(abs(roll), abs(pitch)) <= RESET_TILT_RAD + 1e-6, f"near reset {index} tilt")
            require(abs(data.qpos[2] - .27) < .03, f"near reset {index} is not near 0.27 m")
            require(bool((env.sim.data.qvel[index].abs() < 1e-7).all()), "near reset velocities are not zero")
        for joint in range(1, model.njnt):
            if model.jnt_limited[joint]:
                value = data.qpos[model.jnt_qposadr[joint]]
                require(model.jnt_range[joint, 0] - 1e-7 <= value <= model.jnt_range[joint, 1] + 1e-7,
                        f"reset {index} violates joint limit {model.joint(joint).name}")
        rows.append({"env": index, "height_m": float(data.qpos[2]),
                     "closed_chain_residual_m": residual, "minimum_collision_z_m": minimum_z,
                     "roll_pitch_deg": np.rad2deg([roll, pitch]).tolist(),
                     "ground_start": ground, "clearance_lift_m": float(state.ground_clearance_lift[index])})
    return rows


def runtime_command_checks(env):
    state = env.command_manager.get_term("standup")
    ids = torch.arange(env.num_envs, device=env.device)
    expected = torch.tensor([0., 0., .27], device=env.device).expand(env.num_envs, -1)
    torch.testing.assert_close(state.command, expected)
    for name, value in (("ground_start", True), ("getting_up", True), ("recovered", False)):
        getattr(state, name).fill_(value)
    state.ground_command[:] = .7
    state._resample_command(ids)
    state._update_command(ids)
    torch.testing.assert_close(state.command, expected)
    # 重采样不能重新安排本回合扰动或清除已经施扰的标志。
    state.push_applied[:] = True
    before = {name: getattr(state, name).clone() for name in ("disturbance_kind", "disturbance_time", "push_applied")}
    state._resample_command(ids[2:])
    state._update_command(ids[2:])
    torch.testing.assert_close(state.command, expected)
    for name, value in before.items():
        torch.testing.assert_close(getattr(state, name), value)
    # 部分reset后，未reset环境的物理状态与所有逐环境命令状态必须保留。
    state.hold_anchor_valid.fill_(True)
    state.hold_anchor_xy[:] = torch.arange(env.num_envs, device=env.device)[:, None]
    saved = {name: value[:2].clone() for name, value in vars(state).items()
             if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == env.num_envs}
    qpos, qvel = env.sim.data.qpos[:2].clone(), env.sim.data.qvel[:2].clone()
    env.reset(env_ids=ids[2:])
    for name, value in saved.items():
        torch.testing.assert_close(getattr(state, name)[:2], value, msg=f"partial reset affected {name}")
    torch.testing.assert_close(env.sim.data.qpos[:2], qpos)
    torch.testing.assert_close(env.sim.data.qvel[:2], qvel)
    require(not bool(state.push_applied[2:].any()), "partial reset did not rearm selected worlds")
    require(not bool(state.hold_anchor_valid[2:].any()), "partial reset retained selected reward anchors")
    torch.testing.assert_close(state.command, expected)
    env.reset()
    return {"fixed_command_during_recovery_and_resampling": True,
            "resampling_does_not_rearm_disturbance": True, "partial_reset_isolation": True}


def runtime_ground_reuse_checks(env):
    from ct_mjlab.envs.chuanliantui import multimode_mdp
    from ct_mjlab.envs.chuanliantui_stand import mdp

    state = env.command_manager.get_term("standup")
    probability = state.cfg.ground_start_probability
    ids = torch.arange(env.num_envs, device=env.device)
    nominal = env.action_manager.get_term("pd").nominal_positions.clone()
    require(mdp.original_reset_pose is multimode_mdp.reset_pose, "ground sampler was replaced")
    state.cfg.ground_start_probability = 1.
    torch.manual_seed(103)
    multimode_mdp.reset_pose(env, ids, ground_probability=1.)
    source_qpos, source_qvel = env.sim.data.qpos.clone(), env.sim.data.qvel.clone()
    torch.manual_seed(103)
    mdp.reset_pose(env, ids)
    z_address = int(env.scene["robot"].indexing.free_joint_q_adr[2])
    columns = torch.arange(env.sim.data.qpos.shape[1], device=env.device) != z_address
    torch.testing.assert_close(env.sim.data.qpos[:, columns], source_qpos[:, columns], rtol=0., atol=0.)
    torch.testing.assert_close(env.sim.data.qvel.clone(), source_qvel, rtol=0., atol=0.)
    lift = env.sim.data.qpos[:, z_address] - source_qpos[:, z_address]
    require(bool(((lift >= -1e-7) & (lift <= .004001)).all()), "ground clearance changed original reset excessively")
    torch.testing.assert_close(env.action_manager.get_term("pd").nominal_positions, nominal, rtol=0., atol=0.)
    ground_poses = runtime_pose_checks(env)
    require(all(row["ground_start"] for row in ground_poses), "forced ground reset was not ground")
    state.cfg.ground_start_probability = 0.
    env.reset(seed=42)
    near_poses = runtime_pose_checks(env)
    require(not any(row["ground_start"] for row in near_poses), "fixed near reset included ground samples")
    state.cfg.ground_start_probability = probability
    env.reset(seed=42)
    return {"original_ground_joint_positions_and_all_velocities_bitwise_identical": True,
            "only_root_z_clearance_correction": True, "pd_reference_unchanged": True,
            "ground_poses": ground_poses, "fixed_near_poses": near_poses}


def runtime_disturbance_checks(env):
    from ct_mjlab.envs.chuanliantui_stand import mdp

    state = env.command_manager.get_term("standup")
    ids = torch.arange(env.num_envs, device=env.device)
    results = {}
    # 固定随机抽签，覆盖课程边界和各桶；不靠四个环境碰巧抽到每一类。
    draws = torch.tensor([.1, .49, .6, .9], device=env.device)
    for stage, expected in ((0, [0, 0, 0, 0]), (1, [0, 0, 1, 1]), (2, [0, 0, 2, 3])):
        state.curriculum_stage = stage
        state.training_iteration = 7000 - env.common_step_counter // mdp.STEPS_PER_ITERATION
        with patch.object(mdp.torch, "rand", return_value=draws.clone()):
            state.assign_disturbances(ids)
        equal(state.disturbance_kind.tolist(), expected, f"runtime curriculum stage {stage} at iteration7000")
        require(bool(((state.disturbance_time >= 5.) & (state.disturbance_time <= 10.)).all()),
                "sampled disturbance time is outside 5-10 seconds")
        results[str(stage)] = state.disturbance_kind.tolist()
    state.curriculum_stage = 0
    env.episode_length_buf[:] = 600
    state.disturbance_kind[:] = torch.tensor([1, 2, 3, 0], device=env.device)
    state.disturbance_time[:] = 5.
    state.push_applied[:] = False
    state.strict_stable_time[:] = .5
    state.ground_start[:], state.getting_up[:], state.airborne[:] = False, False, False
    state.recovered[:] = True
    require(not bool(env.termination_manager.dones.any()), "scheduler fixture has a terminated world")
    state.hold_anchor_valid.fill_(True)
    before = env.sim.data.qvel.clone()
    with patch.object(mdp, "standing_mask", return_value=torch.ones(4, dtype=torch.bool, device=env.device)):
        mdp.disturb_once(env, None)
        after = env.sim.data.qvel.clone()
        applied = state.push_applied.clone()
        state.strict_stable_time[:] = .5
        mdp.disturb_once(env, None)
    torch.testing.assert_close(env.sim.data.qvel[:], after)
    equal(applied.tolist(), [True, True, True, False], "one disturbance per eligible episode")
    equal(state.hold_anchor_valid.tolist(), [False, False, False, True],
          "actual disturbance must unlock only its selected reward anchors")
    torch.testing.assert_close(after[3], before[3])
    require(bool((after[:3] != before[:3]).any(dim=1).all()), "requested disturbance did not write velocity")
    # 已错过10秒窗口的回合不能在第11秒补推。
    env.reset()
    env.episode_length_buf[:] = 1100
    state.disturbance_kind[:] = 1
    state.disturbance_time[:] = 8.
    state.strict_stable_time[:] = .5
    state.recovered[:] = True
    before = env.sim.data.qvel.clone()
    with patch.object(mdp, "standing_mask", return_value=torch.ones(4, dtype=torch.bool, device=env.device)):
        mdp.disturb_once(env, None)
    torch.testing.assert_close(env.sim.data.qvel[:], before)
    require(not bool(state.push_applied.any()), "late disturbance was delivered after 10 seconds")
    env.reset()
    return {"curriculum_buckets": results, "one_shot_velocity_write": True,
            "quiet_world_untouched": True, "late_delivery_rejected": True,
            "scope": "standing gate mocked true to isolate scheduler; actual contact sign checked in rollout"}


def runtime_runner_checks(wrapped, train_cfg, device):
    from ct_mjlab.envs import task_registry
    from ct_mjlab.envs.chuanliantui.multimode_runner import MultimodeOnPolicyRunner
    from ct_mjlab.envs.chuanliantui_stand import mdp

    env, records = wrapped.unwrapped, []
    state = env.command_manager.get_term("standup")
    with tempfile.TemporaryDirectory(prefix="stand-checkpoint-") as tmp:
        legacy_path = str(Path(tmp) / "must-not-be-loaded.pt")
        with patch.dict(os.environ, {"WHEELLEG_MULTIMODE_INIT_CHECKPOINT": legacy_path,
                                     "WHEELLEG_HUANANHU_INIT_CHECKPOINT": legacy_path}):
            runner, _ = task_registry.make_alg_runner(wrapped, TASK, train_cfg=train_cfg, device=device)
        require(not hasattr(runner, "init_source"), "runner implicitly initialized from old task")
        equal(runner.alg.gamma, .99, "runtime standing reward discount")
        equal(torch.backends.cuda.matmul.fp32_precision, "ieee", "standing network FP32 precision")
        from tensordict import TensorDict
        observations = wrapped.get_observations()
        # Compare the actual 2048 rollout / 12288 optimization shapes without learning.
        batch = TensorDict({k: v.repeat((12288 + env.num_envs - 1) // env.num_envs, 1)[:12288]
                            for k, v in observations.items()}, batch_size=[12288])
        with torch.no_grad():
            for module in (runner.alg._raw_actor, runner.alg._raw_critic):
                whole = module(batch)
                chunks = torch.cat([module(batch[i:i+2048]) for i in range(0, 12288, 2048)])
                torch.testing.assert_close(whole, chunks, rtol=2e-5, atol=2e-5)
        saved = Path(tmp) / "roundtrip.pt"
        state.cfg.curriculum_stage = None
        for iteration, stage in ((999, 0), (1000, 2), (1999, 1), (7000, 0)):
            state.curriculum_stage = stage
            env.common_step_counter = 72
            state.training_iteration = iteration - 3
            runner.current_learning_iteration = iteration
            runner.save(str(saved))
            original = torch.load(saved, map_location="cpu", weights_only=False)
            equal(original["infos"]["stand_training_iteration"], iteration, "saved effective curriculum")
            equal(original["infos"]["network_matmul_precision"], "ieee", "saved network precision")
            actor = runner.alg._raw_actor
            with torch.no_grad():
                next(actor.parameters()).add_(1.)
            env.common_step_counter, state.training_iteration = 0, 0
            state.curriculum_stage = (stage + 1) % 3
            runner.load(str(saved), strict=True, map_location=device)
            equal(runner.alg.gamma, .99, "resume preserves configured reward discount")
            equal(state.effective_iteration, iteration, "restored effective curriculum")
            equal(state.curriculum_stage, stage, "restored actual curriculum stage")
            equal(original["infos"]["stand_curriculum_stage"], stage, "saved actual stage")
            equal(runner.current_learning_iteration, iteration, "restored runner iteration")
            for group, module in (("actor_state_dict", runner.alg._raw_actor),
                                  ("critic_state_dict", runner.alg._raw_critic)):
                for name, value in module.state_dict().items():
                    torch.testing.assert_close(value.cpu(), original[group][name], rtol=0., atol=0.)
            records.append({"iteration": iteration, "stage": stage})
        # RSL完成1000次更新时保存it=999；续训入口必须从已完成次数1000开始。
        env.common_step_counter, state.training_iteration = 24000, 0
        runner.current_learning_iteration = 999
        runner.save(str(saved))
        last_index = torch.load(saved, map_location="cpu", weights_only=False)
        equal(last_index["iter"], 999, "saved RSL zero-based last iteration")
        equal(last_index["infos"]["stand_training_iteration"], 1000, "saved completed update count")
        resumed = []
        for load_cfg in (None, {"actor": True, "critic": True, "optimizer": True, "iteration": True}):
            runner.current_learning_iteration = 0
            env.common_step_counter, state.training_iteration = 0, 0
            runner.load(str(saved), load_cfg=load_cfg, strict=True, map_location=device)
            equal(runner.current_learning_iteration, 1000, "next iteration after 999-index checkpoint")
            equal(state.effective_iteration, 1000, "curriculum after 999-index checkpoint")
            resumed.append("default" if load_cfg is None else "explicit_iteration_true")
        actor_only_counts = []
        for load_cfg in ({"actor": True}, {"actor": True, "iteration": False}):
            runner.current_learning_iteration = 37
            runner.load(str(saved), load_cfg=load_cfg, strict=True, map_location=device)
            equal(runner.current_learning_iteration, 37, "actor-only load changed runner iteration")
            actor_only_counts.append(runner.current_learning_iteration)
        state.cfg.curriculum_stage = 1
        runner.load(str(saved), strict=True, map_location=device)
        equal(state.curriculum_stage, 1, "explicit between-batch curriculum selection")
        state.cfg.curriculum_stage = None
        legacy = torch.load(saved, map_location="cpu", weights_only=False)
        legacy["infos"]["curriculum_version"] = mdp.LEGACY_CURRICULUM_VERSION
        legacy["infos"].pop("stand_curriculum_stage")
        legacy_path = Path(tmp) / "own-v1.pt"
        torch.save(legacy, legacy_path)
        runner.load(str(legacy_path), strict=True, map_location=device)
        equal(state.curriculum_stage, 0, "own legacy checkpoint migration starts easy")
        equal(state.effective_iteration, 1000, "legacy migration preserves completed updates")
        reverse = expect_value_error(lambda: MultimodeOnPolicyRunner.load(runner, str(saved)),
                                     "old runner real stand checkpoint")
        env.common_step_counter, state.training_iteration = 0, 0
        runner.current_learning_iteration = 0
        env.reset()
    return runner, {"roundtrips": records, "weights_identical": True,
                    "old_runner_rejection": reverse, "legacy_env_variables_ignored": True,
                    "zero_based_resume": {"saved_iter": 999, "completed_updates": 1000,
                                          "next_runner_iteration": 1000, "load_modes": resumed},
                    "actor_only_runner_iterations_unchanged": actor_only_counts,
                    "actual_stage_roundtrip": True, "explicit_between_batch_override": True,
                    "network_matmul_precision": "ieee", "batch_shape_consistency": [2048, 12288],
                    "own_legacy_migration_preserves_weights_and_iteration": True}


def runtime_posture_reward_checks(env):
    from ct_mjlab.envs.chuanliantui_stand import rewards

    state = SimpleNamespace(stable_time=torch.full((env.num_envs,), .5, device=env.device))
    for flag in ("recover", "airborne", "fallen", "settle_required", "push_applied", "quiet_ready", "numerical_failure"):
        setattr(state, flag, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    before = env.sim.data.qpos.clone()
    pd = env.action_manager.get_term("pd")
    reference = pd.nominal_positions.clone()
    with patch.object(env.command_manager, "get_term", return_value=state), \
         patch.object(rewards, "supported_pose_mask", return_value=torch.ones(env.num_envs, dtype=torch.bool, device=env.device)):
        actual = rewards.standing_posture(env).cpu().numpy()
    q = env.sim.data.qpos[:, state.posture_q_adr].cpu().numpy()
    target = np.asarray(rewards.balanced_standing_motor_targets())
    expected = np.exp(-np.mean((q - target)**2, axis=-1) / rewards.POSTURE_SCALE_RAD**2)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0.)
    torch.testing.assert_close(env.sim.data.qpos.clone(), before)
    torch.testing.assert_close(pd.nominal_positions, reference)
    return {"worlds": env.num_envs, "independent_numpy_real_joint_reward": True,
            "maximum_reward_absolute_error": float(np.max(np.abs(actual - expected))),
            "pd_reference_and_actual_state_unchanged": True,
            "scope": "soft reward arithmetic only; no joint locking or behavioral acceptance"}


def runtime_action_delay_checks(env):
    from ct_mjlab.envs.chuanliantui_stand.action import StandPdAction

    pd = env.action_manager.get_term("pd")
    require(isinstance(pd, StandPdAction), "standing action-step buffer not installed")
    equal(pd._delay_choices.tolist(), [5, 10], "1/2 policy steps in the 500Hz FIFO")
    require(pd.cfg.action_delay_s is None, "fixed seconds still override the action buffer")
    observed = set()
    traces = []
    with torch.inference_mode():
        for _ in range(16):
            pd.reset()
            observed.update(pd.action_delay_steps.tolist())
            observed.update(pd.wheel_action_delay_steps.tolist())
        equal(sorted(observed), [5, 10], "both random delay choices exercised")
        pd.reset()
        pd.action_delay_steps[:] = torch.tensor([5, 10, 5, 10], device=env.device)
        pd.wheel_action_delay_steps[:] = 15 - pd.action_delay_steps
        commands = []
        for policy_step in range(4):
            action = torch.full((4, 6), .02 * (policy_step + 1), device=env.device)
            action += torch.arange(6, device=env.device) * .001
            commands.append(action.clone())
            pd.process_actions(action)
            torch.testing.assert_close(pd.raw_action, action)
            for substep in range(5):
                pd.apply_actions()
                for delay, joints in ((pd.action_delay_steps, pd._leg_action_ids),
                                      (pd.wheel_action_delay_steps, pd._wheel_action_ids)):
                    actual = pd._action_history[
                        (pd._history_head - delay) % len(pd._action_history), pd._env_ids]
                    expected = torch.zeros_like(action)
                    for i in range(4):
                        source_step = policy_step - int(delay[i]) // 5
                        if source_step >= 0:
                            expected[i] = commands[source_step][i]
                    torch.testing.assert_close(actual[:, joints], expected[:, joints], rtol=0, atol=0)
                traces.append({"policy_step": policy_step, "physics_substep": substep,
                               "leg0_action": float(pd._action_history[
                                   (pd._history_head - 5) % len(pd._action_history), 0, 0])})
        before_history = pd._action_history.clone()
        before_leg, before_wheel = pd.action_delay_steps.clone(), pd.wheel_action_delay_steps.clone()
        pd.reset(torch.tensor([1, 3], device=env.device))
        require(not bool(pd._action_history[:, [1, 3]].any()), "partial reset leaked previous actions")
        torch.testing.assert_close(pd._action_history[:, [0, 2]], before_history[:, [0, 2]])
        torch.testing.assert_close(pd.action_delay_steps[[0, 2]], before_leg[[0, 2]])
        torch.testing.assert_close(pd.wheel_action_delay_steps[[0, 2]], before_wheel[[0, 2]])
        require(bool(torch.isin(pd.action_delay_steps, pd._delay_choices).all()), "invalid reset leg delay")
        require(bool(torch.isin(pd.wheel_action_delay_steps, pd._delay_choices).all()), "invalid reset wheel delay")
        env.reset(seed=42)
    return {"delay_unit": "policy steps", "choices": [1, 2],
            "physical_ticks_at_current_decimation": [5, 10],
            "all_five_substeps_checked": True, "independent_leg_wheel_delays": True,
            "partial_reset_clears_selected_history_only": True, "trace": traces}


def runtime_observation_checks(env):
    from ct_mjlab.envs.chuanliantui_stand.observations import get_state

    state = get_state(env)
    original_qpos, original_qvel = env.sim.data.qpos.clone(), env.sim.data.qvel.clone()
    sensors, actor = state.raw_sensors().clone(), state.actor().clone()
    try:
        # Wheel rotation and world XY translation are not actor inputs.
        env.sim.data.qpos[:, state._motor_q_adr[4:]] += 17.
        env.sim.data.qvel[:, state.robot.indexing.free_joint_v_adr[:2]] += 3.
        require(torch.equal(state.raw_sensors(), sensors), "raw actor sensors expose wheel angle or planar speed")
        require(torch.equal(state.actor(), actor), "actor exposes planar speed outside original sensor cache")
        require(not bool(sensors[:, 10:12].any()), "wheel-angle raw placeholders are not zero")
        expected = state.noise[:, 10:12] if state.cfg.noise else torch.zeros_like(actor[:, 14:16])
        torch.testing.assert_close(actor[:, 14:16], expected, rtol=0., atol=0.)
        require(not hasattr(state, "velocity_history"), "removed planar sensor still has a history buffer")
        history, noise = state.delay.history.clone(), state.noise.clone()
        state.reset(torch.tensor([1, 3], device=env.device))
        require(torch.equal(state.delay.history[:, [0, 2]], history[:, [0, 2]]),
                "partial observation reset changed other histories")
        require(torch.equal(state.noise[[0, 2]], noise[[0, 2]]),
                "partial observation reset changed other noise")
    finally:
        env.sim.data.qpos[:] = original_qpos
        env.sim.data.qvel[:] = original_qvel
        env.reset(seed=42)
    return {"actor_channels": 39, "critic_channels": 82,
            "wheel_placeholders_one_based": [15, 16], "original_placeholder_noise_retained": True,
            "no_actor_planar_velocity": True, "partial_reset_isolated": True}


def runtime_reset_precision_checks(env):
    state = env.command_manager.get_term("standup")
    probability = state.cfg.ground_start_probability
    precision = torch.backends.cuda.matmul.fp32_precision
    poses = []
    try:
        state.cfg.ground_start_probability = 0.
        for mode in ("tf32", "ieee"):
            torch.backends.cuda.matmul.fp32_precision = mode
            env.reset(seed=42)
            equal(torch.backends.cuda.matmul.fp32_precision, mode, "reset leaked geometry precision")
            poses.append(env.sim.data.qpos.clone())
        require(torch.equal(*poses), "IEEE policy changed historical near-reset height bits")
        before = env.sim.data.qpos.clone()
        env.reset(env_ids=torch.tensor([1, 3], device=env.device))
        require(torch.equal(env.sim.data.qpos[[0, 2]].clone(), before[[0, 2]]),
                "precision-preserving partial reset changed other worlds")
    finally:
        torch.backends.cuda.matmul.fp32_precision = precision
        state.cfg.ground_start_probability = probability
        env.reset(seed=42)
    return {"initial_qpos_bit_identical_across_network_precision": True,
            "network_precision_restored": True, "partial_reset_isolated": True}


def runtime_checks(device, steps, output_dir):
    from ct_mjlab.envs import task_registry

    cfg, train_cfg = task_registry.get_cfgs(TASK)
    wrapped, _ = task_registry.make_env(TASK, env_cfg=cfg, train_cfg=train_cfg,
                                        num_envs=4, seed=42, device=device)
    env = wrapped.unwrapped
    try:
        env.reset(seed=42)
        equal(env.physics_dt, .002, "actual physics dt")
        equal(env.step_dt, .01, "actual policy dt")
        action_delay = runtime_action_delay_checks(env)
        poses = runtime_pose_checks(env)
        posture = runtime_posture_reward_checks(env)
        ground_reuse = runtime_ground_reuse_checks(env)
        pd = env.action_manager.get_term("pd")
        legs = pd._leg_action_ids
        nominal = pd.nominal_positions[:, legs] - pd.nominal_offset[:, legs]
        expected = torch.tensor(cfg.actions["pd"].nominal_leg_positions, device=env.device).expand_as(nominal)
        torch.testing.assert_close(nominal, expected)
        commands = runtime_command_checks(env)
        disturbances = runtime_disturbance_checks(env)
        runner, checkpoints = runtime_runner_checks(wrapped, train_cfg, device)
        reset_precision = runtime_reset_precision_checks(env)
        observation = runtime_observation_checks(env)
        obs = wrapped.get_observations()
        equal({name: list(obs[name].shape) for name in ("actor", "critic")},
              {"actor": [4, 39], "critic": [4, 82]}, "runtime observation dimensions")
        equal(wrapped.num_actions, 6, "runtime action dimension")
        contact_samples, z_min, z_max = 0, 0., 0.
        policy = runner.get_inference_policy(device=device)
        # 真实接触的原始Z符号用于检查支撑门控；不是假定合成force的方向。
        with torch.inference_mode():
            for _ in range(max(steps, 32)):
                obs, reward, _, _ = wrapped.step(policy(obs))
                for name in ("actor", "critic"):
                    require(bool(torch.isfinite(obs[name]).all()), f"non-finite {name} observation")
                require(bool(torch.isfinite(reward).all()), "non-finite finite-check reward")
                force = env.scene["wheel_contact"].data.force
                z = force[:, :, 2]
                useful = force.norm(dim=-1) > 5.
                if bool(useful.any()):
                    sample = z[useful]
                    contact_samples += int(useful.sum())
                    z_min, z_max = min(z_min, float(sample.min())), max(z_max, float(sample.max()))
        require(contact_samples > 0, "no actual wheel contact sampled; force sign remains unverified")
        require(z_min < -5. and z_max <= 1e-4,
                f"wheel contact raw Z sign differs from support=-force_z: min={z_min}, max={z_max}")
        from ct_mjlab.envs.chuanliantui_stand import rewards
        state = env.command_manager.get_term("standup")
        selected = torch.tensor([1, 3], device=env.device)
        with torch.inference_mode():
            state.body_speed_peak[:] = .2
            before_peak = state.body_speed_peak.clone()
            env.reset(env_ids=selected)
            require(not bool(rewards.body_speed_margin_cost(env)[selected].any()),
                    "partial reset leaked a previous body speed cost")
            torch.testing.assert_close(state.body_speed_peak[[0, 2]], before_peak[[0, 2]])
        # 权重从未优化；清零检查用的课程计数，交付可复现的未训练负例。
        env.common_step_counter = 0
        env.command_manager.get_term("standup").training_iteration = 0
        runner.current_learning_iteration = 0
        # 上面的步进已创建inference sensor缓存，复位也须留在同一上下文。
        with torch.inference_mode():
            env.reset(seed=42)
        model_path = output_dir / "untrained-model.pt"
        if model_path.exists():
            raise FileExistsError(f"Refusing to overwrite untrained check artifact: {model_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        runner.save(str(model_path), infos={"artifact_kind": "untrained-random-policy-for-finite-checks",
                                          "learning_updates": 0, "trained": False})
        untrained = {"path": str(model_path.resolve()),
                     "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                     "trained": False, "learning_updates": 0, "iteration": 0,
                     "scope": "random initial policy for negative acceptance and video smoke checks only"}
        return {"poses": poses, "posture_reward": posture, "ground_reuse": ground_reuse,
                "action_delay": action_delay, "reset_precision": reset_precision,
                "observation": observation,
                "partial_reset_body_cost_exempt_and_other_peak_preserved": True,
                "pd_reference_unchanged": True, "commands": commands,
                "disturbances": disturbances, "checkpoints": checkpoints,
                "untrained_model": untrained,
                "finite_steps": max(steps, 32), "observation_shapes": {"actor": [4, 39], "critic": [4, 82]},
                "wheel_contact_sign": {"real_contact_samples": contact_samples,
                                       "raw_z_min_n": z_min, "raw_z_max_n": z_max,
                                       "positive_support_is_negative_sensor_z": True}}
    finally:
        wrapped.close()


def visible_speed_and_recovery_cost_checks():
    from ct_mjlab.envs.chuanliantui_stand import rewards

    state = SimpleNamespace(stable_time=torch.full((2,), .5),
        settle_required=torch.zeros(2, dtype=torch.bool), recover=torch.zeros(2, dtype=torch.bool),
        fallen=torch.zeros(2, dtype=torch.bool), airborne=torch.zeros(2, dtype=torch.bool),
        push_applied=torch.zeros(2, dtype=torch.bool), quiet_ready=torch.zeros(2, dtype=torch.bool))
    velocity = torch.tensor([[.03, 0., 0.], [0., .004, 0.]])
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: state),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_w=velocity))})
    with patch.object(rewards, "supported_pose_mask", return_value=torch.ones(2, dtype=torch.bool)):
        torch.testing.assert_close(rewards.body_speed_cost(env), torch.tensor([.03, .004]))
        velocity[0] = torch.tensor([-.2, 0., 10.])
        torch.testing.assert_close(rewards.body_speed_cost(env), torch.tensor([.1, .004]))
        for flag in ("settle_required", "recover", "fallen", "airborne"):
            getattr(state, flag)[0] = True
            torch.testing.assert_close(rewards.body_speed_cost(env), torch.tensor([0., .004]))
            getattr(state, flag)[0] = False
        state.push_applied[0] = True
        torch.testing.assert_close(rewards.body_speed_cost(env), torch.tensor([0., .004]))
        state.quiet_ready[0] = True
        torch.testing.assert_close(rewards.body_speed_cost(env), torch.tensor([.1, .004]))
    for flag in ("settle_required", "recover", "fallen"):
        for name in ("stand_still_lin_vel", "no_fork", "no_fork_square", "leg_joint_vel", "wheel_vel"):
            with patch.object(rewards.huananhu_rewards, "reward", return_value=torch.tensor([2., 3.])):
                torch.testing.assert_close(rewards.hold_cost_outside_recovery(env, name=name),
                    torch.tensor([2., 3.]))
                getattr(state, flag)[0] = True
                torch.testing.assert_close(rewards.hold_cost_outside_recovery(env, name=name),
                    torch.tensor([0., 3.]))
                getattr(state, flag)[0] = False
    return {"world_horizontal_velocity_only": True, "cost_caps_at_0_1_m_s": True,
            "no_position_anchor_input": True, "no_leg_motion_eligibility": True,
            "legacy_static_costs_exempt_through_getup_recovery": True,
            "normal_values_and_other_env_preserved": True}


def recovery_wheel_height_checks():
    from ct_mjlab.envs.chuanliantui_stand import rewards

    class Scene(dict):
        env_origins = torch.zeros(5, 3)

    state = SimpleNamespace(settle_required=torch.ones(5, dtype=torch.bool),
        recover=torch.zeros(5, dtype=torch.bool), fallen=torch.zeros(5, dtype=torch.bool),
        numerical_failure=torch.zeros(5, dtype=torch.bool))
    data = SimpleNamespace(root_link_pos_w=torch.tensor([
        [0., 0., .16], [0., 0., .215], [0., 0., .27], [0., 0., .27], [0., 0., .27]]),
        projected_gravity_b=torch.tensor([[0., 0., -1.]]).expand(5, -1).clone())
    force = torch.zeros(5, 2, 3)
    force[3, 0, 2] = -49.05
    force[4, :, 2] = -49.05
    scene = Scene(robot=SimpleNamespace(data=data, indexing=SimpleNamespace(body_ids=[0])),
        wheel_contact=SimpleNamespace(data=SimpleNamespace(force=force)))
    env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_term=lambda _: state),
        sim=SimpleNamespace(model=SimpleNamespace(body_mass=torch.full((5, 1), 10.)),
            mj_model=SimpleNamespace(opt=SimpleNamespace(gravity=np.array([0., 0., -9.81])))))
    torch.testing.assert_close(rewards.recovery_wheel_height(env),
        torch.tensor([0., .125, .25, .625, 1.]), atol=1e-6, rtol=0.)
    force[4, :, 2] = -10000.
    torch.testing.assert_close(rewards.recovery_wheel_height(env)[4], torch.tensor(1.))
    data.projected_gravity_b[4, 2] = 1.
    torch.testing.assert_close(rewards.recovery_wheel_height(env)[4], torch.tensor(0.))
    data.projected_gravity_b[4, 2] = -1.
    state.numerical_failure[4] = True
    torch.testing.assert_close(rewards.recovery_wheel_height(env)[4], torch.tensor(0.))
    state.settle_required.zero_()
    require(not bool(rewards.recovery_wheel_height(env).any()),
            "recovery shaping persists after settling")
    return {"no_wheel_load_cannot_earn_full_reward": True,
            "single_wheel_cannot_earn_full_reward": True,
            "impact_force_capped": True, "signed_upright": True,
            "numerical_failure_and_normal_standing_exempt": True,
            "height_floor_is_reward_only_m": rewards.RECOVERY_HEIGHT_FLOOR_M,
            "command_height_unchanged_m": .27}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    if not args.check_only and (args.output.parent / "untrained-model.pt").exists():
        parser.error("Output directory already contains untrained-model.pt; use a new directory")
    if args.steps < 1:
        parser.error("--steps must be positive")
    report = {"kind": "finite standing task checks; no learning and no behavior acceptance",
              "task": TASK, "check_only": args.check_only, "learn_called": False,
              "checks": {}, "failures": []}
    checks = [("configuration", configuration_checks), ("pose", pose_checks),
              ("position_recovery_rewards", position_and_recovery_reward_checks),
              ("recovery_wheel_height", recovery_wheel_height_checks),
              ("visible_speed_and_legacy_recovery_costs", visible_speed_and_recovery_cost_checks),
              ("runner_contract", runner_contract_checks), ("reward_samples", reward_sample_checks),
              ("posture_reward", posture_reward_checks),
              ("mixed_sampling", mixed_sampling_checks), ("recovery_settle", recovery_settle_checks),
              ("assessment", assessment_checks)]
    if not args.check_only:
        checks.append(("runtime", lambda: runtime_checks(args.device, args.steps, args.output.parent)))
    for name, check in checks:
        try:
            report["checks"][name] = {"passed": True, "details": check()}
        except Exception as error:
            report["checks"][name] = {"passed": False, "error": str(error),
                                      "traceback": traceback.format_exc()}
            report["failures"].append(name)
    report["passed"] = not report["failures"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": report["passed"], "failures": report["failures"],
                      "report": str(args.output)}, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
