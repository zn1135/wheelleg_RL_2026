#!/usr/bin/env python3
"""Historical r25/r26 velocity-profile checks; use its frozen source snapshot."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "mjlab_training/src"))
from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui import huananhu_randomization
from ct_mjlab.envs.chuanliantui_stand import observations, randomization
from ct_mjlab.envs.chuanliantui_stand.env_cfg import configure_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--migrated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if observations.OBSERVATION_CONTRACT != "chuanliantui-stand-41-82-planar-velocity-v1":
        parser.error("This historical check requires frozen r25/r26 sources; "
                     "use check_chuanliantui_stand.py for the current 39-input profile")
    assert not args.output.exists()
    cfg, agent = task_registry.get_cfgs("chuanliantui_stand")
    wrapped, _ = task_registry.make_env("chuanliantui_stand", env_cfg=cfg, train_cfg=agent,
                                      num_envs=4, seed=42, device=args.device)
    env = wrapped.unwrapped
    report = {"scope": __doc__, "training_updates": 0}
    try:
        runner, _ = task_registry.make_alg_runner(wrapped, "chuanliantui_stand", train_cfg=agent, device=args.device)
        runner.load(str(args.migrated), map_location=args.device)
        source = torch.load(args.source, map_location="cpu", weights_only=False)
        migrated = torch.load(args.migrated, map_location="cpu", weights_only=False)
        actor = runner.alg._raw_actor
        legacy = copy.deepcopy(actor.mlp)
        legacy[0] = torch.nn.Linear(39, 256, device=args.device)
        legacy.load_state_dict({k.removeprefix("mlp."): v for k, v in source["actor_state_dict"].items()
                                if k.startswith("mlp.")})
        obs = wrapped.get_observations()
        with torch.no_grad():
            torch.testing.assert_close(actor(obs), legacy(obs["actor"][:, :39]), rtol=2e-5, atol=2e-6)
        for name, value in source["actor_state_dict"].items():
            actual = migrated["actor_state_dict"][name]
            if name == "mlp.0.weight":
                torch.testing.assert_close(actual[:, :39], value, rtol=0, atol=0)
                assert not bool(actual[:, 39:].any())
            else:
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
        for name, value in source["critic_state_dict"].items():
            torch.testing.assert_close(migrated["critic_state_dict"][name], value, rtol=0, atol=0)
        for pid, state in source["optimizer_state_dict"]["state"].items():
            for name, value in state.items():
                actual = migrated["optimizer_state_dict"]["state"][pid][name]
                if isinstance(value, torch.Tensor) and value.shape == (256, 39):
                    torch.testing.assert_close(actual[:, :39], value, rtol=0, atol=0)
                    assert not bool(actual[:, 39:].any())
                elif isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual, value, rtol=0, atol=0)
                else:
                    assert actual == value
        assert runner.current_learning_iteration == source["infos"]["stand_training_iteration"]
        report["migration"] = {"original_actor_function_preserved": True, "critic_unchanged": True,
                               "all_adam_history_preserved_with_zero_new_columns": True,
                               "resume_iteration": runner.current_learning_iteration}
        state = observations.get_state(env)
        expected_bounds = torch.tensor([.25]*3 + [.05]*3 + [.025]*6 + [.5]*4 + [1.]*2,
                                       device=args.device) * .4
        torch.testing.assert_close(state.noise_bounds, expected_bounds)
        ids, others = torch.tensor([1, 3], device=args.device), torch.tensor([0, 2], device=args.device)
        with torch.inference_mode():
            state.velocity_history[:, ids] = 123.
            before = state.velocity_history[:, others].clone()
            state.reset(ids)
            torch.testing.assert_close(state.velocity_history[:, others], before, rtol=0, atol=0)
            torch.testing.assert_close(state.velocity_history[:, ids],
                state.raw_planar_velocity()[ids].unsqueeze(0).expand_as(state.velocity_history[:, ids]), rtol=0, atol=0)
            rng = torch.cuda.get_rng_state(args.device)
            state._sample_velocity_noise(ids)
            torch.testing.assert_close(torch.cuda.get_rng_state(args.device), rng, rtol=0, atol=0)
            state.velocity_history.zero_()
            state.velocity_noise.zero_()
            state.delay.head = 0
            state.delay.lags[:, 0] = 3
            for tick in range(6):
                state.delay.head = (state.delay.head + 1) % len(state.velocity_history)
                state.velocity_history[state.delay.head] = float(tick + 1)
                expected = max(0, tick - 2)
                torch.testing.assert_close(state.actor()[:, -2:],
                    torch.full((4, 2), float(expected), device=args.device), rtol=0, atol=0)
            qpos, qvel = env.sim.data.qpos.clone(), env.sim.data.qvel.clone()
            indexing = env.scene["robot"].indexing
            yaw = torch.tensor([0., torch.pi/2, torch.pi, -torch.pi/2], device=args.device)
            quat = torch.zeros((4, 4), device=args.device)
            quat[:, 0], quat[:, 3] = (yaw/2).cos(), (yaw/2).sin()
            env.sim.data.qpos[:, indexing.free_joint_q_adr[3:7]] = quat
            env.sim.data.qvel[:, indexing.free_joint_v_adr[:2]] = torch.tensor([1., 2.], device=args.device)
            torch.testing.assert_close(state.raw_planar_velocity(),
                torch.tensor([[1., 2.], [2., -1.], [-1., -2.], [-2., 1.]], device=args.device),
                rtol=0, atol=1e-6)
            env.sim.data.qpos.copy_(qpos)
            env.sim.data.qvel.copy_(qvel)
            env.reset(seed=42)
        report["velocity_sensor"] = {"dimensions": 2, "partial_reset_isolated": True,
            "separate_noise_rng": True, "physical_sample_delay_checked": True,
            "heading_frame_four_quadrants_checked": True,
            "training_noise_scale": .4, "velocity_noise_bound_m_s": .01}
        # Directly compare full-scale wrapper with the original implementation,
        # using identical RNG; restore all model fields before returning to simulation.
        before_fields = {name: getattr(env.sim.model, name).clone() for name in randomization.MODEL_FIELDS}
        original_state = env._huananhu_randomization
        rng = torch.cuda.get_rng_state(args.device)
        try:
            with torch.inference_mode():
                randomization.randomize_at_startup(env, None, scale=1.)
                full = {name: getattr(env.sim.model, name).clone() for name in randomization.MODEL_FIELDS}
                torch.cuda.set_rng_state(rng, args.device)
                huananhu_randomization.randomize_at_startup(env, None)
                for name in randomization.MODEL_FIELDS:
                    torch.testing.assert_close(getattr(env.sim.model, name).clone(), full[name], rtol=0, atol=0)
        finally:
            with torch.inference_mode():
                for name, value in before_fields.items():
                    getattr(env.sim.model, name).copy_(value)
            env._huananhu_randomization = original_state
            torch.cuda.set_rng_state(rng, args.device)
        evaluation, _ = task_registry.get_cfgs("chuanliantui_stand")
        configure_evaluation(evaluation)
        assert evaluation.events["huananhu_domain_randomization"].params["scale"] == 1.
        assert evaluation.actions["pd"].randomization_scale == 1.
        assert evaluation.observations["actor"].terms["proprioception"].params["cfg"].noise_scale == 1.
        report["fixed_evaluation_full_original_randomization_identical"] = True
        report["passed"] = True
    finally:
        wrapped.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
