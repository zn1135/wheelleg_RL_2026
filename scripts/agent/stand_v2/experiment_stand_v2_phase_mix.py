#!/usr/bin/env python3
"""Physically stagger stand_v2 resets; isolated experiment, production recipe unchanged."""

import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from mjlab.utils.os import dump_yaml

from ct_mjlab.envs import task_registry
from ct_mjlab.envs.chuanliantui.model import MODEL_XML
from ct_mjlab.envs.chuanliantui_stand_v2.mdp import TRACKING_COMMAND_CONTRACT, SPEED_COMMAND_CONTRACT, YAW_COMMAND_CONTRACT, POSITION_COMMAND_CONTRACT
from ct_mjlab.envs.chuanliantui_stand_v2.rewards import (
    restore_tracking_reward_cfg, configure_tight_reward)


CONTRACT = "stand-v2-physical-phase-mix-experiment-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(mapping):
    result = hashlib.sha256()
    for key, value in sorted(mapping.items()):
        result.update(key.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class Monitor:
    def __init__(self, runner):
        self.runner, self.raw = runner, runner.env.unwrapped
        self.state = self.raw.command_manager.get_term("standup")
        self.action = self.raw.action_manager.get_term("pd")
        self.stage = "prelude"
        self.tick = 0
        self.records = []
        self.completed = {key: [0, 0] for key in ("prelude", "check", "train")}
        self.counts = torch.zeros(7, dtype=torch.long, device=self.raw.device)
        self.hold_violations = torch.zeros((), dtype=torch.long, device=self.raw.device)
        self.std_floor = runner.alg.actor.distribution.std_range[0]
        self.std_floor_violations = torch.zeros_like(self.hold_violations)
        self.lowest_std = torch.full((6,), float("inf"), device=self.raw.device)
        self.tracking_counts = torch.zeros(3, dtype=torch.long, device=self.raw.device)
        self.yaw_counts = torch.zeros(2, dtype=torch.long, device=self.raw.device)
        self.position_counts = torch.zeros(3, dtype=torch.long, device=self.raw.device)
        original_reset = self.state.reset

        def reset(env_ids):
            completed = self.state.completed[env_ids]
            counts = self.completed[self.stage]
            counts[0] += int(completed.sum())
            counts[1] += int((completed & self.state.has_stood[env_ids]).sum())
            return original_reset(env_ids)

        self.state.reset = reset
        original_step = runner.env.step

        def step(actions):
            age = self.raw.episode_length_buf.clone()
            hold = self.action._ground_hold().clone()
            std = runner.alg.actor.output_std[0].detach()
            self.lowest_std = torch.minimum(self.lowest_std, std)
            self.std_floor_violations += (std < self.std_floor-1e-8).sum()
            result = original_step(actions)
            moving = self.state.command[:, 0] != 0.
            role = self.state.tracking_episode
            if self.state.cfg.position_hold:
                role = role | self.state.position_locked
            invalid = moving & (~self.state.has_stood | self.state.recover | self.state.fallen
                | ~role | (self.state.tracking_ready_steps < 50))
            self.tracking_counts += torch.stack((moving.sum(), invalid.sum(),
                (self.state.ground_start == 0).sum()))
            if self.state.cfg.yaw_control:
                turning = self.state.command[:, 1] != 0.
                invalid_yaw = turning & (~self.state.has_stood | self.state.recover | self.state.fallen
                    | ~self.state.yaw_active | (self.state.tracking_ready_steps < 50))
                self.yaw_counts += torch.stack((turning.sum(), invalid_yaw.sum()))
            if self.state.cfg.position_hold:
                locked = self.state.position_locked
                invalid_position = locked & (~self.state.yaw_active | (self.state.tracking_velocity != 0.))
                self.position_counts += torch.stack((locked.sum(), (self.state.position_velocity != 0.).sum(), invalid_position.sum()))
            self.hold_violations += (self.action.torques[hold] != 0).sum()
            self.counts += torch.stack(((age < 20).sum(), ((age >= 20) & (age < 200)).sum(),
                ((age >= 200) & (age < 500)).sum(), (age >= 500).sum(),
                result[2].bool().sum(), self.raw.reset_terminated.sum(),
                self.raw.reset_time_outs.sum()))
            self.tick += 1
            if self.tick % 24 == 0:
                self.records.append([self.stage, self.tick, *self.counts.cpu().tolist(),
                                     *self.completed[self.stage]])
                self.counts.zero_()
                assert all(torch.isfinite(v).all() for v in result[0].values())
            return result

        runner.env.step = step

    def begin(self, stage):
        self.stage, self.tick = stage, 0
        self.counts.zero_()


@torch.inference_mode()
def prepare(runner, monitor, output):
    raw, actor = runner.env.unwrapped, runner.alg.actor
    assert raw.max_episode_length == 4000 and not raw.episode_length_buf.any()
    assert raw.step_dt == .01 and raw.physics_dt == .002
    start_counter = raw.common_step_counter
    actor_before = tensor_digest(actor.state_dict())
    critic_before = tensor_digest(runner.alg.critic.state_dict())
    optimizer_before = copy.deepcopy(runner.alg.optimizer.state_dict())
    encoder_before = copy.deepcopy(runner.alg.encoder_optimizer.state_dict())
    # 真实走到不同回合时刻，保持每次复位的20步电机等待，不改写回合年龄。
    ages = torch.arange(raw.num_envs, device=raw.device) * 4000 // raw.num_envs
    ages = ages[torch.randperm(raw.num_envs, device=raw.device)]
    schedule = {}
    for env_id, when in enumerate((4000-ages).cpu().tolist()):
        schedule.setdefault(when, []).append(env_id)
    obs = runner.env.get_observations()
    for tick in range(1, 4001):
        obs, _, _, _ = runner.env.step(actor(obs, stochastic_output=True))
        if tick in schedule:
            ids = torch.tensor(schedule[tick], device=raw.device)
            raw.reset(env_ids=ids)
            obs = runner.env.get_observations()
            assert (raw.episode_length_buf[ids] == 0).all()
            assert monitor.action._ground_hold()[ids].all()
            assert torch.equal(obs["history"][ids], obs["actor"][ids].repeat(1, 5))
        if tick % 1000 == 0:
            print(f"Physical prelude {tick/100:.0f}/40 s; optimizer updates=0", flush=True)
    actual = raw.episode_length_buf.clone()
    assert (actual <= ages).all()  # Physical failures may reset an environment early.
    assert monitor.hold_violations == 0
    assert actor_before == tensor_digest(actor.state_dict())
    assert critic_before == tensor_digest(runner.alg.critic.state_dict())
    for before, after in ((optimizer_before, runner.alg.optimizer.state_dict()),
                          (encoder_before, runner.alg.encoder_optimizer.state_dict())):
        for key, state in before["state"].items():
            for name, value in state.items():
                assert torch.equal(value, after["state"][key][name]) if torch.is_tensor(value) else value == after["state"][key][name]
        assert before["param_groups"] == after["param_groups"]
    # 全局计数继续表示PPO采样步；预运行的4000步另记，保留真实局部时钟和所有历史。
    raw.common_step_counter = start_counter
    stats = dict(contract=CONTRACT, physical_policy_steps=4000, parameter_updates=0,
        common_step_counter=start_counter, target_age_steps=ages.cpu().tolist(),
        actual_age_steps=actual.cpu().tolist(), physical_completed=monitor.completed["prelude"],
        actor_unchanged=True, critic_unchanged=True, optimizers_unchanged=True,
        hold_violations=int(monitor.hold_violations), history_refilled_after_each_reset=True)
    torch.save(dict(qpos=raw.sim.data.qpos.cpu(), qvel=raw.sim.data.qvel.cpu(),
        age=actual.cpu(), observations={k: v.cpu() for k, v in obs.items()}), output/"phase-start.pt")
    (output/"prelude.json").write_text(json.dumps(stats, indent=2)+"\n")
    return stats


def main(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, agent = task_registry.get_cfgs("chuanliantui_stand_v2")
    cfg.seed = agent.seed = args.seed
    agent.max_iterations = args.iterations
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    parent_contract = checkpoint.get("infos", {}).get("command_contract")
    parent_position = parent_contract == POSITION_COMMAND_CONTRACT
    parent_yaw = parent_contract in (YAW_COMMAND_CONTRACT, POSITION_COMMAND_CONTRACT)
    parent_speed = parent_contract in (SPEED_COMMAND_CONTRACT, YAW_COMMAND_CONTRACT, POSITION_COMMAND_CONTRACT)
    parent_tracking = parent_contract in (TRACKING_COMMAND_CONTRACT, SPEED_COMMAND_CONTRACT, YAW_COMMAND_CONTRACT, POSITION_COMMAND_CONTRACT)
    yaw_control = getattr(args, 'yaw_control', False)
    position_hold = getattr(args, 'position_hold', False)
    restore_tracking_reward_cfg(cfg.commands["standup"], checkpoint.get("infos", {}))
    parent_active_weight = cfg.commands["standup"].tracking_weight_while_active
    parent_tight = cfg.commands["standup"].tracking_tight_term
    cfg.commands["standup"].post_stand_tracking = parent_tracking
    cfg.commands["standup"].speed_curriculum = parent_speed
    cfg.commands["standup"].yaw_control = parent_yaw
    cfg.commands['standup'].position_hold = parent_position
    if parent_yaw:
        for key, value in checkpoint['infos']['yaw_control_config'].items():
            setattr(cfg.commands['standup'], key, value)
    if parent_yaw and not yaw_control:
        raise ValueError('Yaw checkpoints require --yaw-control')
    if parent_position:
        for key, value in checkpoint['infos']['position_hold_config'].items():
            setattr(cfg.commands['standup'], key, value)
        if not position_hold:
            raise ValueError('Position checkpoints require --position-hold')
    if position_hold and not yaw_control:
        raise ValueError('--position-hold requires --yaw-control')
    if yaw_control and not (args.post_stand_tracking and args.speed_curriculum):
        raise ValueError('--yaw-control requires --post-stand-tracking and --speed-curriculum')
    cfg.commands["standup"].tracking_weight_while_active = parent_active_weight
    if args.tracking_weight_while_active and not args.speed_curriculum:
        raise ValueError("--tracking-weight-while-active requires --speed-curriculum")
    if parent_active_weight and not args.tracking_weight_while_active:
        raise ValueError("Active-weight checkpoints require --tracking-weight-while-active")
    if args.tracking_tight_term and not args.tracking_weight_while_active:
        raise ValueError("--tracking-tight-term requires --tracking-weight-while-active")
    if parent_tight and not args.tracking_tight_term:
        raise ValueError("Tight checkpoints require --tracking-tight-term")
    if parent_speed:
        for key, value in checkpoint["infos"]["standup_curriculum"]["speed_curriculum"]["config"].items():
            setattr(cfg.commands["standup"], key, value)
    if parent_speed and not args.speed_curriculum:
        raise ValueError("Speed checkpoints require --speed-curriculum")
    if args.speed_curriculum and not args.post_stand_tracking:
        raise ValueError("--speed-curriculum requires --post-stand-tracking")
    if parent_tracking and not args.post_stand_tracking:
        raise ValueError("Tracking checkpoints require --post-stand-tracking; cannot silently change recipe")
    start_iteration = checkpoint["iter"] + 1
    if args.min_std is not None:
        # 使用RSL-RL原生采样下限，保持载入的网络参数与优化器状态。
        agent.actor.distribution_cfg["std_range"] = (args.min_std, 1e6)
    exploration = dict(min_std=args.min_std, entropy_coef=agent.algorithm.entropy_coef)
    label = f"_phase_mix{start_iteration-1}to{start_iteration+args.iterations-1}"
    if args.min_std is not None:
        label += "_stdmin005"
    if args.post_stand_tracking:
        label += "_tracking50_vx15_slew" if args.speed_curriculum else "_tracking50_vx01to03"
    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + label + f"_{args.num_envs}"
    if args.tracking_weight_while_active:
        run_name += "_activew1"
    if args.tracking_tight_term:
        run_name += "_tight"
    if yaw_control:
        run_name += "_yawhold"
    if position_hold:
        run_name += '_poshold'
    if Path(run_name).name != run_name:
        raise ValueError("run-name must be a single directory name")
    log_dir = None if args.check_only else Path("logs/mjlab/chuanliantui_stand_v2")/run_name
    if log_dir is not None and log_dir.exists():
        raise FileExistsError(log_dir)
    sources = [Path(__file__).resolve().relative_to(Path.cwd()),
               *Path("mjlab_training/src/ct_mjlab").rglob("*.py")]
    report = dict(contract=CONTRACT, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=digest(args.checkpoint), seed=args.seed, num_envs=args.num_envs,
        xml_sha256=digest(MODEL_XML),
        authorized_iterations=args.iterations, check_only=args.check_only,
        start_iteration=start_iteration, exploration=exploration,
        post_stand_tracking=args.post_stand_tracking,
        speed_curriculum=args.speed_curriculum,
        tracking_weight_while_active=args.tracking_weight_while_active,
        tracking_tight_term=args.tracking_tight_term,
        yaw_control=yaw_control,
        position_hold=position_hold,
        log_dir=None if log_dir is None else str(log_dir.resolve()),
        sources={str(p): digest(p) for p in sources},
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    (args.output/"run-info.json").write_text(json.dumps(report, indent=2)+"\n")
    (args.output/"workspace.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    for source in sources:
        target = args.output/"source"/source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    env, cfg = task_registry.make_env("chuanliantui_stand_v2", env_cfg=cfg, train_cfg=agent,
        num_envs=args.num_envs, device=args.device)
    try:
        runner, _ = task_registry.make_alg_runner(env, "chuanliantui_stand_v2", train_cfg=agent,
            log_dir=log_dir, resume=args.checkpoint, device=args.device)
        assert runner.current_learning_iteration == start_iteration
        assert env.unwrapped.command_manager.get_term("standup").unlocked
        state = env.unwrapped.command_manager.get_term("standup")
        if args.post_stand_tracking and not parent_tracking:
            # 先完整恢复2999旧配方，再显式启用新配方，不绕过旧版本拒载检查。
            cfg.commands["standup"].post_stand_tracking = state.cfg.post_stand_tracking = True
            ids = torch.arange(env.num_envs, device=env.unwrapped.device)
            state._resample_command(ids)
            state._update_command(ids)
            assert not state.command[:, :2].any()
        if args.speed_curriculum and not parent_speed:
            state.cfg.speed_curriculum = True
            state.cfg.tracking_acceleration_range = tuple(args.acceleration_range)
            ids = torch.arange(env.num_envs, device=env.unwrapped.device)
            state._resample_command(ids)
            state._update_command(ids)
        if parent_speed and tuple(args.acceleration_range) != tuple(state.cfg.tracking_acceleration_range):
            raise ValueError("Acceleration range must match resumed speed checkpoint")
        if yaw_control and not parent_yaw:
            # 先按5999原契约完整续载；新转弯分布不能沿用旧直行统计窗口。
            report['yaw_migration_previous_speed_window'] = state.checkpoint_state()['speed_curriculum']
            state.cfg.yaw_control = True
            state.speed_segments = state.speed_successes = 0
            state.last_speed_success_fraction = 0.
            ids = torch.arange(env.num_envs, device=env.unwrapped.device)
            state._resample_command(ids)
            state._update_command(ids)
            env.unwrapped.observation_manager.reset()
            env.unwrapped.obs_buf = env.unwrapped.observation_manager.compute(update_history=True)
        if yaw_control:
            report['yaw_control_config'] = state.yaw_config()
        if position_hold and not parent_position:
            # 原契约完整恢复后才迁移；停车纠偏不能沿用旧速度统计窗口。
            report['position_migration_previous_speed_window'] = state.checkpoint_state()['speed_curriculum']
            state.cfg.position_hold = True
            state.position_config()
            state.speed_segments = state.speed_successes = 0
            state.last_speed_success_fraction = 0.
            ids = torch.arange(env.num_envs, device=env.unwrapped.device)
            state._resample_command(ids)
            state._update_command(ids)
            env.unwrapped.observation_manager.reset()
            env.unwrapped.obs_buf = env.unwrapped.observation_manager.compute(update_history=True)
        if position_hold:
            report['position_hold_config'] = state.position_config()
        if args.tracking_weight_while_active and not parent_active_weight:
            # 完整恢复旧配方及优化器后才迁移奖励；不更改命令、课程或环境历史。
            state.cfg.tracking_weight_while_active = True
        if args.tracking_tight_term and not parent_tight:
            # 父模型已按原契约完整加载；在预运行前才注册B的第20项。
            from mjlab.managers.reward_manager import RewardManager
            state.cfg.tracking_tight_term = True
            configure_tight_reward(env.unwrapped.cfg, state.cfg)
            env.unwrapped.reward_manager = RewardManager(env.unwrapped.cfg.rewards,
                env.unwrapped, scale_by_dt=env.unwrapped.cfg.scale_rewards_by_dt)
            if hasattr(env.unwrapped, "_isaac_v2_rewards"):
                env.unwrapped._isaac_v2_rewards.step = -1
        if args.tracking_tight_term:
            report["tracking_tight_config"] = dict(
                kernel_denominator=state.cfg.tracking_tight_kernel_denominator,
                weight=state.cfg.tracking_tight_weight)
        obs = env.get_observations()
        with torch.no_grad():
            runner.alg.actor(obs, stochastic_output=True)
            effective_std = runner.alg.actor.output_std[0].detach().cpu().tolist()
        assert tensor_digest(runner.alg.actor.state_dict()) == tensor_digest(checkpoint["actor_state_dict"])
        assert tensor_digest(runner.alg.critic.state_dict()) == tensor_digest(checkpoint["critic_state_dict"])
        if args.min_std is not None:
            assert min(effective_std) >= args.min_std - 1e-8
            assert runner.alg.actor.distribution.std_range == [args.min_std, 1e6]
        report["initial_effective_std"] = effective_std
        report["initial_raw_std"] = runner.alg.actor.distribution.std_param.detach().cpu().tolist()
        report["restored_model_parameters_exact"] = True
        (args.output/"run-info.json").write_text(json.dumps(report, indent=2)+"\n")
        dump_yaml(args.output/"env.yaml", asdict(cfg))
        dump_yaml(args.output/"agent.yaml", asdict(agent))
        monitor = Monitor(runner)
        # 打散复位预运行不用于晋级，也不改写续载的课程统计窗口。
        state.speed_curriculum_frozen = True
        prelude = prepare(runner, monitor, args.output)
        state.speed_curriculum_frozen = args.check_only
        if args.check_only:
            monitor.begin("check")
            before = tensor_digest(runner.alg.actor.state_dict())
            obs = env.get_observations()
            with torch.inference_mode():
                for tick in range(4100):
                    obs, _, _, _ = env.step(runner.alg.actor(obs, stochastic_output=True))
                    if (tick+1) % 1000 == 0:
                        print(f"Checking staggered rollouts {(tick+1)/100:.0f}/41 s", flush=True)
            assert before == tensor_digest(runner.alg.actor.state_dict())
        else:
            monitor.begin("train")
            save = runner.save

            def save_experiment(path, infos=None):
                return save(path, {**(infos or {}), "phase_mix_experiment": dict(
                    contract=CONTRACT, parent_sha256=report["checkpoint_sha256"],
                    prelude_policy_steps=4000, num_envs=args.num_envs, seed=args.seed,
                    exploration=exploration)})

            runner.save = save_experiment
            print(f"Training output: {log_dir.resolve()}", flush=True)
            runner.learn(num_learning_iterations=args.iterations)
            assert runner.current_learning_iteration == start_iteration+args.iterations-1
        assert monitor.hold_violations == 0
        assert monitor.std_floor_violations == 0
        assert monitor.tracking_counts[1:].sum() == 0
        assert monitor.yaw_counts[1] == 0
        assert monitor.position_counts[2] == 0
        report.update(completed=True, actual_optimizer_iterations=0 if args.check_only else args.iterations,
            final_iteration=runner.current_learning_iteration, hold_violations=int(monitor.hold_violations),
            std_floor_violations=int(monitor.std_floor_violations),
            lowest_observed_std=monitor.lowest_std.cpu().tolist(),
            tracking_monitor=dict(zip(("nonzero_command_env_steps", "gate_violations", "nonground_env_steps"),
                                      monitor.tracking_counts.cpu().tolist())),
            yaw_monitor=dict(zip(('nonzero_command_env_steps', 'gate_violations'), monitor.yaw_counts.cpu().tolist())),
            position_monitor=dict(zip(('locked_env_steps', 'correction_env_steps', 'gate_violations'), monitor.position_counts.cpu().tolist())),
            completed_and_stood=monitor.completed, monitor_fields=["stage", "policy_tick", "age_0_02",
                "age_02_2", "age_2_5", "age_5_40", "reset", "failure", "timeout",
                "completed_cumulative", "stood_cumulative"],
            monitor_records=monitor.records, prelude=prelude)
        (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k not in ("sources", "monitor_records", "prelude")}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--post-stand-tracking", action="store_true",
                        help="All ground starts; half of episodes track vx only after standing and NORMAL hold")
    parser.add_argument("--speed-curriculum", action="store_true")
    parser.add_argument("--tracking-weight-while-active", action="store_true")
    parser.add_argument("--tracking-tight-term", action="store_true")
    parser.add_argument("--yaw-control", action="store_true", help="Yaw rate tracking and heading hold on turn release")
    parser.add_argument('--position-hold', action='store_true', help='Parking position outer loop; requires yaw control')
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--acceleration-range", nargs=2, type=float, default=(.5, 2.))
    parser.add_argument("--min-std", type=float, choices=(.05,), default=None,
                        help="Native Gaussian sampling lower bound; entropy stays at the task value")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.tracking_weight_while_active:
        if (args.check_only and args.iterations != 0) or (not args.check_only and args.iterations <= 0):
            parser.error("Active-weight recipe: check-only requires0; training needs explicit positive iterations")
    elif args.iterations != (0 if args.check_only else (1000 if args.post_stand_tracking else 500)):
        parser.error("Budget: check-only=0; original phase experiment=500; post-stand tracking=1000")
    main(args)
