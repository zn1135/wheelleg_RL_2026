#!/usr/bin/env python3
"""Check command gates, reward isolation, resume/export and frozen rollouts; no learning."""

import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch

import torch

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui_stand.mdp import StandCommand
from ct_mjlab.envs.chuanliantui_stand_v2 import rewards
from ct_mjlab.envs.chuanliantui_stand_v2.export import export
from ct_mjlab.envs.chuanliantui_stand_v2.runner import task_contracts
from scripts.agent.stand_v2.experiment_stand_v2_phase_mix import digest, tensor_digest


@torch.inference_mode()
def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, agent = task_registry.get_cfgs("chuanliantui_stand_v2")
    cfg.seed = agent.seed = 42
    agent.actor.distribution_cfg["std_range"] = (.05, 1e6)
    env, _ = task_registry.make_env("chuanliantui_stand_v2", env_cfg=cfg, train_cfg=agent,
        num_envs=128, seed=42, device=args.device)
    try:
        runner, _ = task_registry.make_alg_runner(env, "chuanliantui_stand_v2", train_cfg=agent, device=args.device)
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No learning")):
            runner.load(str(args.checkpoint), map_location=args.device)
            parent = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            assert tensor_digest(runner.alg.actor.state_dict()) == tensor_digest(parent["actor_state_dict"])
            assert tensor_digest(runner.alg.critic.state_dict()) == tensor_digest(parent["critic_state_dict"])
            optimizers = [copy.deepcopy(o.state_dict()) for o in (runner.alg.optimizer, runner.alg.encoder_optimizer)]
            raw = env.unwrapped
            state = raw.command_manager.get_term("standup")
            assert state.unlocked and state.ground_start.all() and state.cfg.ground_start_probability == 1.
            ids = torch.arange(raw.num_envs, device=raw.device)
            state.cfg.post_stand_tracking = True
            state._resample_command(ids)
            selected = int(state.tracking_episode.sum())
            assert 0 < selected < raw.num_envs
            assert ((state.tracking_velocity.abs() >= .1) & (state.tracking_velocity.abs() <= .3)).all()
            # 使用同一生产门控，传感器与模式更新由既有正式检查覆盖；此处控制就绪边界。
            state.has_stood[:] = state.observation_recovered[:] = True
            state.fallen[:] = False
            with patch.object(StandCommand, "_update_command"):
                for _ in range(49):
                    raw.common_step_counter += 1
                    state._update_command(None)
                    assert not state.command[:, :2].any()
                raw.common_step_counter += 1
                state._update_command(None)
                assert torch.equal(state.command[:, 0] != 0., state.tracking_episode)
                counter = state.tracking_ready_steps.clone()
                state._update_command(None)
                assert torch.equal(counter, state.tracking_ready_steps), "Observation reads advanced gate"
                state.observation_recovered[:] = False
                raw.common_step_counter += 1
                state._update_command(None)
                assert not state.command[:, :2].any() and not state.tracking_ready_steps.any()
                state.observation_recovered[:] = True
                for _ in range(49):
                    raw.common_step_counter += 1
                    state._update_command(None)
                    assert not state.command[:, :2].any()
                state.has_stood[:] = False
                raw.common_step_counter += 1
                state._update_command(None)
                assert not state.command[:, :2].any() and not state.tracking_ready_steps.any()
            state.tracking_ready_steps[:] = 17
            others = state.tracking_velocity[3:].clone()
            raw.reset(env_ids=ids[:3])
            assert not state.tracking_ready_steps[:3].any() and (state.tracking_ready_steps[3:] == 17).all()
            assert torch.equal(others, state.tracking_velocity[3:])
            assert not state.command[:3, :2].any()
            state._resample_command(ids[:0])
            # 当前真实命令槽进入actor和历史末帧；维度/缩放保持原约定。
            state._command[:, 0] = .3
            obs = raw.observation_manager.compute(update_history=True)
            torch.testing.assert_close(obs["actor"][:, 6], torch.full((128,), .6, device=raw.device))
            assert torch.equal(obs["history"][:, -32:], obs["actor"])
            assert obs["critic"].shape[-1] == 89
            state.has_stood[:] = state.reward_active[:] = True
            state.numerical_failure[:] = False
            port = rewards.RewardState(raw)
            def values(enabled, vx):
                state.cfg.post_stand_tracking = enabled
                state._command[:, 0] = vx
                port.step = -1
                return {k:v.clone() for k,v in port.compute(raw).items()}
            old_zero, new_zero = values(False, 0.), values(True, 0.)
            assert all(torch.equal(old_zero[k], new_zero[k]) for k in old_zero)
            old_move, new_move = values(False, .3), values(True, .3)
            torch.testing.assert_close(new_move["tracking_lin_vel"], old_move["tracking_lin_vel"]*5.)
            assert all(torch.equal(old_move[k], new_move[k]) for k in old_move if k != "tracking_lin_vel")
            state.reward_active[:] = False
            port.step = -1
            assert not any(v.any() for v in port.compute(raw).values())
            # 新配方显式版本保存；原2999普通加载拒绝，防止静默混用。
            raw.reset()
            interface = args.output/"interface-only-parent2999.pt"
            runner.save(str(interface))
            saved = torch.load(interface, map_location="cpu", weights_only=False)
            assert all(saved["infos"][k] == v for k,v in task_contracts(True).items())
            try:
                runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device)
            except ValueError as error:
                assert "experiment_profile_contract" in str(error)
            else:
                raise AssertionError("Old zero-command recipe was silently accepted")
            runner.load(str(interface), map_location=args.device)
            export_report = export(interface, args.output/"interface-only.onnx", samples=32)
            assert all(export_report[k] == v for k,v in task_contracts(True).items())
            raw.reset()
            policy = runner.get_inference_policy(device=args.device)
            obs = env.get_observations()
            moving_steps = gate_violations = 0
            ever_stood = torch.zeros(128, dtype=torch.bool, device=raw.device)
            for tick in range(4000):
                obs, value, done, _ = env.step(policy(obs))
                assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(value).all()
                moving = state.command[:, 0] != 0.
                invalid = moving & (~state.has_stood | state.recover | state.fallen
                    | ~state.tracking_episode | (state.tracking_ready_steps < 50))
                moving_steps += int(moving.sum())
                gate_violations += int(invalid.sum())
                ever_stood |= state.has_stood
                assert not state.command[:, 1].any() and state.ground_start.all()
                assert (state.command[:, 0].abs() <= .3).all()
                assert torch.equal(obs["history"][:, -32:], obs["actor"])
                torch.testing.assert_close(obs["actor"][:, 6:9], state.command*state.command.new_tensor((2., .25, 5.)))
                if (tick+1) % 1000 == 0:
                    print(f"Frozen tracking gate check {(tick+1)/100:.0f}/40s; moving env-steps={moving_steps}", flush=True)
            assert moving_steps > 0 and gate_violations == 0
            assert tensor_digest(runner.alg.actor.state_dict()) == tensor_digest(parent["actor_state_dict"])
            assert tensor_digest(runner.alg.critic.state_dict()) == tensor_digest(parent["critic_state_dict"])
            for before, optimizer in zip(optimizers, (runner.alg.optimizer, runner.alg.encoder_optimizer)):
                after = optimizer.state_dict()
                assert before["param_groups"] == after["param_groups"]
                for k, row in before["state"].items():
                    for name, v in row.items():
                        assert torch.equal(v, after["state"][k][name]) if torch.is_tensor(v) else v == after["state"][k][name]
            report = dict(passed=True, parameter_updates=0, checkpoint_sha256=digest(args.checkpoint),
                contracts=task_contracts(True), actor_critic_optimizers_unchanged=True,
                ground_probability=1., tracking_probability=.5, initial_selected_episodes=selected,
                hold_49_50_boundary=True, lost_NORMAL_clears_and_rearms=True, partial_reset_isolated=True,
                zero_command_rewards_equal=True, moving_main_effective_weight=1., other_rewards_equal=True,
                observation_dimensions=[32,160,89,6], export=export_report,
                frozen_128env_40s=dict(nonzero_command_env_steps=moving_steps, gate_violations=gate_violations,
                    observed_stood=int(ever_stood.sum())),
                scope="Interface and finite command-gate checks with frozen2999; not tracking capability acceptance",
                source_sha256=digest(__file__))
            (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps({k:v for k,v in report.items() if k != "export"}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
