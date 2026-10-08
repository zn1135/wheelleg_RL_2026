#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isaac Gym 干净环境对照评估：与 mj_sim2sim 相同场景（平地、无随机化、固定命令）。

用途：sim2sim 排查的对照组。MuJoCo 里失稳时先跑这个——
    Isaac 也失稳  -> 策略本身的问题（回去调训练）
    Isaac 稳      -> 两引擎的 gap（调 MJCF 接触参数或加强域随机化）

用法:
    python sim2sim/eval_isaac.py --load_run Jul21_06-00-36_ --checkpoint 1000 --cmd_vx 0.5

chuanliantui_standup 可追加：
    --pd_error_report /tmp/pd_error.json  逐个 PD 子步记录四个腿关节的位置误差。
    --standup_config_snapshot <run>/chuanliantui_standup_config.py
        加载可信的本地起立配置快照；继承的父类仍使用当前工作区版本。
    --stochastic --noise --domain_rand  使用采样动作、观测噪声及域随机化。
    --pushes  单独启用首次轮接地后的随机推扰，其他域随机化仍默认关闭。
    --gas_spring_force 0  chuanliantui 每侧气弹簧推力；0 回放旧无弹簧条件。
    --action_delay_ms 6  固定动作延迟（按物理步量化），不启用其他随机化。
    --standup_post_unlock  加载后统一为解锁阶段，使用该阶段的成功判据。
    --metrics_report /tmp/metrics.json  保存逐控制步轨迹及 5 秒之后的统计。
    --render  打开跟随机器人视角的回放窗口。
    --frame_dir <新目录>  配合 --render，每秒保存一张回放画面。
统计是有限采样诊断，不替代可视行为验收或历史训练全程记录。
"""
import isaacgym  # noqa: F401  必须在 torch 之前 import
from wheel_legged_gym.envs import *  # noqa: F401,F403  触发任务注册
from wheel_legged_gym.utils import get_args, task_registry
import torch
import sys
import math
import json
import runpy
import numpy as np
from pathlib import Path


class StandupMetricsRecorder:
    """单环境轨迹；重置步只保留事件，不记录已清零的下一回合推扰状态。"""

    def __init__(self, env):
        self.env = env
        self.rows = []
        self.episode = 0
        self.origin = env.root_states[0, :2].clone()

    def record(self, step, active, actions, done, timeout, policy_feedback=None):
        env = self.env
        row = {"t": (step + 1) * env.dt, "episode": self.episode,
               "reset": done, "timeout": timeout, "policy_active": active}
        if done:
            self.episode += 1
            self.origin = env.root_states[0, :2].clone()
        else:
            pg = env.projected_gravity[0].tolist()
            row.update({
                "z_m": env.root_states[0, 2].item(),
                "height_m": env.base_height[0].item(),
                "xy_m": env.root_states[0, :2].tolist(),
                "drift_m": torch.norm(env.root_states[0, :2] - self.origin).item(),
                "vx_m_s": env.base_lin_vel[0, 0].item(),
                "pitch_deg": math.degrees(math.asin(max(-1., min(1., pg[0])))),
                "tilt_deg": math.degrees(math.acos(max(-1., min(1., -pg[2])))),
                "has_stood": bool(env.has_stood[0].item()),
                "standing_now": bool(env.standing_time[0].item() > 0),
                "stable_now": bool(env.standing_time[0].item() >= env.cfg.standup.success_duration_s),
                "actions": actions[0].tolist(),
                "motor_torque_nm": env.torques[0].tolist(),
                "gas_spring_knee_torque_nm": (env.gas_spring_knee_torques[0].tolist()
                                              if hasattr(env, "gas_spring_knee_torques") else [0., 0.]),
                # 此 buffer 保留刚完成的物理子步已施加的力，不能读未来待施加的目标。
                "push_force_world_n": (env.standup_push_force[0].tolist()
                                       if hasattr(env, "standup_push_force") else [0., 0., 0.]),
                "push_count": (int(env.standup_push_count[0].item())
                               if hasattr(env, "standup_push_count") else 0),
            })
            if policy_feedback is not None:
                row.update(policy_feedback)
        self.rows.append(row)

    def save(self, path, metadata):
        summaries = {}
        for name, start in [("whole", 0.), ("after_5s", 5.)]:
            selected = [r for r in self.rows if r["t"] > start]
            valid = [r for r in selected if not r["reset"]]
            stats = {"samples": len(valid), "valid_duration_s": len(valid) * self.env.dt,
                     "resets": sum(r["reset"] for r in selected),
                     "failures": sum(r["reset"] and not r["timeout"] for r in selected),
                     "push_force_nonzero_duration_s": 0., "push_force_peak_n": 0.}
            if valid:
                push_norm = np.linalg.norm([r["push_force_world_n"] for r in valid], axis=1)
                stats["push_force_nonzero_duration_s"] = float(np.count_nonzero(push_norm) * self.env.dt)
                stats["push_force_peak_n"] = float(push_norm.max())
                for key in ("z_m", "height_m", "pitch_deg", "vx_m_s", "drift_m", "tilt_deg"):
                    x = np.asarray([r[key] for r in valid])
                    stats[key] = {"mean": float(x.mean()), "std": float(x.std()),
                                  "min": float(x.min()), "max": float(x.max()),
                                  "rms": float(np.sqrt(np.mean(x * x)))}
                stats["height_mae_m"] = float(np.mean([abs(r["height_m"] - metadata["cmd_height"]) for r in valid]))
                for key in ("standing_now", "stable_now"):
                    stats[key + "_fraction"] = sum(r[key] for r in valid) / len(valid)
                changes = [np.asarray(b["actions"]) - np.asarray(a["actions"])
                           for a, b in zip(selected, selected[1:])
                           if not a["reset"] and not b["reset"] and a["policy_active"] and b["policy_active"]
                           and a["episode"] == b["episode"]]
                stats["action_delta_rms"] = float(np.sqrt(np.mean(np.square(changes)))) if changes else None
            summaries[name] = stats
        summaries["first_stood_s"] = next((r["t"] for r in self.rows if r.get("has_stood")), None)
        with Path(path).open("x") as stream:
            json.dump({"metadata": metadata, "summary": summaries, "samples": self.rows},
                      stream, indent=2, allow_nan=False)
        print("METRICS_SUMMARY=" + json.dumps(summaries))
        print(f"完整轨迹: {Path(path).resolve()}")


class LegErrorRecorder:
    """只读记录实际送入 PD 的四个腿关节误差；每次调用对应一个物理子步。"""

    def __init__(self, env):
        self.env = env
        self.compute = env._compute_torques
        self.rows = []

    def __call__(self, actions):
        env = self.env
        ids = [0, 1, 3, 4]
        q = env.dof_pos[0, ids]
        target = (actions * env.cfg.control.pos_action_scale + env.default_dof_pos)[0, ids]
        error = target - q
        active = bool(env.get_policy_action_mask()[0].item()) if hasattr(env, "get_policy_action_mask") else True
        stood = bool(env.has_stood[0].item()) if hasattr(env, "has_stood") else False
        torque = self.compute(actions)
        self.rows.append({
            "t": len(self.rows) * env.sim_params.dt,
            "active": active, "has_stood": stood,
            "q": q.detach().cpu().tolist(),
            "target": target.detach().cpu().tolist(),
            "error": error.detach().cpu().tolist(),
            "qd": env.dof_vel[0, ids].detach().cpu().tolist(),
            "torque": torque[0, ids].detach().cpu().tolist(),
        })
        return torque

    def save(self, path, metadata):
        names = [self.env.dof_names[i] for i in [0, 1, 3, 4]]
        summary = {}
        for phase, predicate in [
            ("before_policy", lambda r: not r["active"]),
            ("policy_active", lambda r: r["active"]),
            ("after_stood", lambda r: r["active"] and r["has_stood"]),
        ]:
            rows = [r for r in self.rows if predicate(r)]
            if not rows:
                summary[phase] = {"substeps": 0}
                continue
            errors = torch.tensor([r["error"] for r in rows]).abs()
            maximum = errors.max(dim=0).values.tolist()
            summary[phase] = {
                "substeps": len(rows), "max_abs_error_rad": dict(zip(names, maximum)),
                "fraction_abs_gt_pi": dict(zip(names, (errors > math.pi).float().mean(0).tolist())),
                "fraction_abs_ge_3_5": dict(zip(names, (errors >= 3.5).float().mean(0).tolist())),
                "peak_example": rows[int(errors.max(1).values.argmax())],
            }
        result = {"metadata": metadata, "joint_names": names, "summary": summary, "samples": self.rows}
        # 避免覆盖已有诊断证据或历史训练文件。
        with Path(path).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print("PD_ERROR_SUMMARY=" + json.dumps(summary))
        print(f"PD 内环原始记录: {Path(path).resolve()}")


def main():
    # 复用仓库的参数解析（--task/--load_run/--checkpoint/--headless），追加命令参数
    cmd_vx = 0.5
    cmd_yaw = 0.0
    cmd_height = 0.30
    sim_seconds = 15.0
    stochastic = False
    with_noise = False
    with_trimesh = False
    with_dr = False
    with_pushes = False
    contact_friction = None
    pd_error_report = None
    standup_config_snapshot = None
    render = False
    frame_dir = None
    action_delay_ms = None
    metrics_report = None
    standup_post_unlock = False
    gas_spring_force = None
    push_schedule_path = None
    argv = []
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--cmd_vx":
            cmd_vx = float(next(it))
        elif a == "--cmd_yaw":
            cmd_yaw = float(next(it))
        elif a == "--cmd_height":
            cmd_height = float(next(it))
        elif a == "--sim_time":
            sim_seconds = float(next(it))
        elif a == "--stochastic":
            stochastic = True  # 用训练收集时的采样动作（含 std~0.27 噪声）而非确定性均值
        elif a == "--noise":
            with_noise = True  # 消融：保留训练的观测噪声
        elif a == "--trimesh":
            with_trimesh = True  # 消融：保留训练的 trimesh 地形（不强制 plane）
        elif a == "--domain_rand":
            with_dr = True  # 消融：保留训练的全部域随机化
        elif a == "--pushes":
            with_pushes = True
        elif a == "--gas_spring_force":
            gas_spring_force = float(next(it))
        elif a == "--push_schedule":
            push_schedule_path = next(it)
        elif a == "--contact_friction":
            contact_friction = float(next(it))
        elif a == "--pd_error_report":
            pd_error_report = next(it)
        elif a == "--action_delay_ms":
            action_delay_ms = float(next(it))
        elif a == "--metrics_report":
            metrics_report = next(it)
        elif a == "--standup_post_unlock":
            standup_post_unlock = True
        elif a == "--standup_config_snapshot":
            standup_config_snapshot = next(it)
        elif a == "--render":
            render = True
        elif a == "--frame_dir":
            frame_dir = Path(next(it))
        else:
            argv.append(a)
    sys.argv = ["eval_isaac", "--task=mini_wheel_legged", "--headless"] + argv
    args = get_args()
    args.headless = not render
    if not math.isfinite(sim_seconds) or sim_seconds <= 0:
        raise ValueError("--sim_time 必须是正有限数")
    if (metrics_report or standup_post_unlock) and args.task != "chuanliantui_standup":
        raise ValueError("轨迹指标及课程阶段覆盖仅支持 chuanliantui_standup")
    if with_pushes and args.task != "chuanliantui_standup":
        raise ValueError("--pushes 仅支持 chuanliantui_standup")
    if push_schedule_path and (args.task != "chuanliantui_standup" or with_pushes or with_dr):
        raise ValueError("--push_schedule 仅用于起立对照，不能与 --pushes/--domain_rand 合用")
    for path in (metrics_report, pd_error_report):
        if path and Path(path).exists():
            raise FileExistsError(f"不覆盖已有报告: {path}")
    if action_delay_ms is not None and (not math.isfinite(action_delay_ms) or action_delay_ms < 0):
        raise ValueError("--action_delay_ms 必须是非负有限数")
    if frame_dir:
        if not render:
            raise ValueError("--frame_dir 需要 --render")
        frame_dir.mkdir(parents=True, exist_ok=False)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if standup_config_snapshot:
        if args.task != "chuanliantui_standup":
            raise ValueError("--standup_config_snapshot 仅用于 chuanliantui_standup")
        snapshot = runpy.run_path(standup_config_snapshot)
        env_cfg = snapshot["ChuanliantuiStandupCfg"]()
        train_cfg = snapshot["ChuanliantuiStandupCfgPPO"]()
        env_cfg.seed = train_cfg.seed
        print(f"加载起立配置快照（父类仍用当前代码）: {standup_config_snapshot}")
    if gas_spring_force is not None:
        if not hasattr(env_cfg, "gas_spring"):
            raise ValueError("--gas_spring_force 仅用于具有气弹簧配置的 chuanliantui")
        if not math.isfinite(gas_spring_force) or not 0 <= gas_spring_force <= 150:
            raise ValueError("--gas_spring_force 必须是 0~150 N 的有限数")
        env_cfg.gas_spring.force_n = gas_spring_force
    env_cfg.env.num_envs = 1
    # 有限时长评估观察同一回合；保留失稳终止，避免训练20s超时截断30s站立。
    env_cfg.env.episode_length_s = max(env_cfg.env.episode_length_s, sim_seconds + 1.0)
    if not with_trimesh:
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.curriculum = False
    else:
        env_cfg.terrain.num_rows = 3
        env_cfg.terrain.num_cols = 3
        env_cfg.terrain.max_init_terrain_level = 0  # 最简单档地形
    env_cfg.noise.add_noise = with_noise
    if not with_dr:
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_restitution = False
        env_cfg.domain_rand.randomize_base_com = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_inertia = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_Kp = False
        env_cfg.domain_rand.randomize_Kd = False
        env_cfg.domain_rand.randomize_motor_torque = False
        env_cfg.domain_rand.randomize_default_dof_pos = False
        env_cfg.domain_rand.randomize_action_delay = False
        env_cfg.domain_rand.randomize_compliance = False
    push_parameter_names = (
        "standup_push_force_range", "standup_push_unstable_force_range",
        "standup_push_duration_s", "standup_push_interval_s_range",
    )
    if with_pushes:
        missing = [name for name in push_parameter_names if not hasattr(env_cfg.domain_rand, name)]
        if missing:
            raise ValueError(
                "--pushes 需要新的起立推扰配置；当前配置/旧快照缺少: " + ", ".join(missing)
            )
        env_cfg.domain_rand.push_robots = True
    push_metadata = {"push_robots": bool(env_cfg.domain_rand.push_robots)}
    push_metadata.update({name: getattr(env_cfg.domain_rand, name, None)
                          for name in push_parameter_names})
    if action_delay_ms is not None:
        env_cfg.domain_rand.randomize_action_delay = True
        env_cfg.domain_rand.delay_ms_range = [action_delay_ms, action_delay_ms]
    if contact_friction is not None:
        if args.task != "chuanliantui_standup" or with_dr:
            raise ValueError("--contact_friction 仅用于起立任务的无域随机化对照")
        if env_cfg.terrain.static_friction != env_cfg.terrain.dynamic_friction:
            raise ValueError("地面静、动摩擦不同，无法用单个接触摩擦目标评估")
        robot_friction = 2.0 * contact_friction - env_cfg.terrain.static_friction
        if robot_friction < 0:
            raise ValueError("目标接触摩擦过低，所需机器人摩擦为负")
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.friction_range = [robot_friction, robot_friction]
        env_cfg.domain_rand.friction_ranges = [[robot_friction, robot_friction]]
    env_cfg.commands.curriculum = False
    # 固定命令（resample 也只会采到同一个值）
    env_cfg.commands.ranges.lin_vel_x = [cmd_vx, cmd_vx]
    env_cfg.commands.ranges.ang_vel_yaw = [cmd_yaw, cmd_yaw]
    env_cfg.commands.ranges.height = [cmd_height, cmd_height]
    # 关键：基类 heading_command=True 时 commands[:,1] 每步被航向控制器覆盖
    # （1.5*(目标航向-当前航向) 剪到±5），ang_vel_yaw 范围无效——必须把目标航向也钉死为 0，
    # 否则评估中每 resampling_time 秒收到一个 [-π,π] 随机航向目标，机器人被命令猛转弯。
    env_cfg.commands.ranges.heading = [0.0, 0.0]
    if args.task == "chuanliantui_standup":
        # 起立环境重采样时按课程写高度，必须同时固定两个课程阶段。
        env_cfg.standup_curriculum.pre_unlock_target_height = cmd_height
        env_cfg.standup_curriculum.post_unlock_target_height = cmd_height
    forward_axis = 1 if args.task == "mini_wheel_legged" else 0
    axis_name = "xyz"[forward_axis]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if with_pushes and not all(hasattr(env, name) for name in (
        "standup_push_force", "standup_push_count"
    )):
        raise RuntimeError("--pushes 需要支持 standup_push_force/count 的起立环境实现")
    if contact_friction is not None:
        print(
            f"轮地目标接触摩擦={contact_friction:.3f}，"
            f"地面={env_cfg.terrain.static_friction:.3f}，"
            f"机器人材质={env.friction_coef[0].item():.3f}"
        )
    # 关键：必须置 resume=True 才会加载 --load_run/--checkpoint 指定的权重
    # （对齐 play.py:71；漏掉这行会静默地跑随机初始化网络！）
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    print(f"已加载权重: load_run={train_cfg.runner.load_run} checkpoint={train_cfg.runner.checkpoint}")
    if standup_post_unlock:
        env.standup_curriculum_unlocked = True
        env.reward_scales["orientation"] = env_cfg.standup_curriculum.post_unlock_orientation_scale * env.dt
        print(f"统一解锁阶段，成功高度门槛={env._current_standup_success_height():.3f}m")
    policy = ppo_runner.get_inference_policy(device=env.device)
    recorder = LegErrorRecorder(env) if pd_error_report else None
    if recorder:
        if args.task != "chuanliantui_standup" or env.num_envs != 1:
            raise ValueError("PD 误差统计仅支持单环境 chuanliantui_standup")
        env._compute_torques = recorder
    # runner 构造时会 reset，加载权重又会恢复课程；重新取得匹配当前状态的观测。
    env.reset()
    scheduled_push = None
    scheduled_impulse = np.zeros(3)
    if push_schedule_path:
        from sim2sim.standing_push_schedule import load
        scheduled_push = load(push_schedule_path)
        schedule_dt = scheduled_push.to_dict()["physics_dt_s"]
        if not math.isclose(schedule_dt, env.sim_params.dt, abs_tol=1e-8):
            raise ValueError("推力计划的物理步长与Isaac不一致")
        schedule_step = 0
        env.cfg.domain_rand.push_robots = True
        push_metadata["push_robots"] = True
        schedule_starts = {round(e["t_start_s"] / schedule_dt)
                           for e in scheduled_push.to_dict()["events"]}
        env._update_standup_pushes = lambda: None

        def accumulate_scheduled_push():
            nonlocal schedule_step
            force = scheduled_push.force_at(schedule_step * schedule_dt)
            force32 = force.astype(np.float32)
            env.standup_push_force[0] = torch.as_tensor(force32, device=env.device)
            env.rigid_body_external_forces[0, env.standup_push_body_index] += env.standup_push_force[0]
            scheduled_impulse[:] += force32.astype(np.float64) * schedule_dt
            if schedule_step in schedule_starts:
                env.standup_push_count += 1
            schedule_step += 1

        env._accumulate_push_forces = accumulate_scheduled_push
    obs, obs_history = env.get_observations()
    start_xy = env.root_states[0, :2].clone()
    metrics = StandupMetricsRecorder(env) if metrics_report else None
    actual_delay_ms = float(env.action_delay_idx[0].item() * env.sim_params.dt * 1000) if env_cfg.domain_rand.randomize_action_delay else 0.
    print(f"实际动作延迟={actual_delay_ms:.3f}ms")
    if with_pushes:
        print("随机推扰配置=" + json.dumps(push_metadata, ensure_ascii=False))

    n_steps = int(round(sim_seconds / env.dt))
    ac = ppo_runner.alg.actor_critic
    print(f"Isaac 对照评估：{sim_seconds}s (cmd_vx={cmd_vx}, yaw={cmd_yaw}, h={cmd_height}, "
          f"动作={'采样(训练同款噪声)' if stochastic else '确定性均值'})")
    print(f"字段：xy 世界系位置 | z 高度 | v_fwd 机体系前向({axis_name})速度 | pg_xyz 重力投影 | done 重置")
    n_dones = 0
    n_timeouts = 0
    max_drift = 0.0
    max_speed = 0.0
    max_tilt = 0.0
    ever_stood = False
    for step in range(n_steps):
        if render:
            target = env.root_states[0, :3].detach().cpu().numpy()
            env.set_camera(target + [1.4, -1.4, 0.8], target)
        with torch.no_grad():
            mask = env.get_policy_action_mask().clone() if hasattr(env, "get_policy_action_mask") else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            latent = torch.zeros(env.num_envs, 3, device=env.device)
            if mask.any():
                if stochastic:
                    actions[mask] = ac.act(obs[mask], obs_history[mask])
                    latent[mask] = ac.latent
                else:
                    actions[mask], latent[mask] = policy(obs[mask], obs_history[mask])
        policy_feedback = {
            "policy_t_s": step * env.dt,
            "policy_true_velocity_m_s": env.base_lin_vel[0].tolist(),
            "estimated_velocity_m_s": (latent[0] / 2.).tolist(),
            "velocity_estimation_error_m_s": (latent[0] / 2. - env.base_lin_vel[0]).tolist(),
        } if metrics else None
        obs, _, _, dones, infos, obs_history = env.step(actions.detach())
        if metrics:
            done = bool(dones[0].item())
            metrics.record(step, bool(mask[0].item()), actions,
                           done, done and bool(infos.get("time_outs", env.time_out_buf)[0].item()),
                           policy_feedback)
        if frame_dir and step % 100 == 0:
            env.gym.write_viewer_image_to_file(
                env.viewer, str(frame_dir / f"step_{step:05d}.png")
            )
        n_dones += int(dones[0].item())
        if dones[0].item():
            n_timeouts += int(infos.get("time_outs", env.time_out_buf)[0].item())
            start_xy = env.root_states[0, :2].clone()
        else:
            max_drift = max(max_drift, torch.norm(env.root_states[0, :2] - start_xy).item())
            max_speed = max(max_speed, abs(env.base_lin_vel[0, forward_axis].item()))
            max_tilt = max(max_tilt, math.degrees(math.acos(
                max(-1.0, min(1.0, -env.projected_gravity[0, 2].item())))))
        if hasattr(env, "has_stood"):
            ever_stood |= bool(env.has_stood[0].item())
        if step % 100 == 0 or dones[0].item():
            z = env.root_states[0, 2].item()
            pg = env.projected_gravity[0].cpu().numpy()
            v_fwd = env.base_lin_vel[0, forward_axis].item()
            v_hat = latent[0, forward_axis].item() / 2.0  # 推理前状态的速度估计
            cmd = env.commands[0, :3].cpu().numpy()
            xy = env.root_states[0, :2].cpu().numpy()
            stood = int(env.has_stood[0].item()) if hasattr(env, "has_stood") else -1
            print(f"[{step:5d}] xy=[{xy[0]:+.3f},{xy[1]:+.3f}] z={z:.3f} v_fwd={v_fwd:+.3f} v̂={v_hat:+.3f} "
                  f"pg_xyz=[{pg[0]:+.2f},{pg[1]:+.2f},{pg[2]:+.2f}] has_stood={stood} "
                  f"cmd=[{cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:.2f}] done={int(dones[0].item())}")
            if dones[0].item():
                print(f">>> 第 {step} 步({step*env.dt:.1f}s)触发重置；该行位置可能已复位，需区分超时和失败。")
    print(f"评估结束。{sim_seconds}s 内重置 {n_dones} 次，其中超时 {n_timeouts} 次。")
    print(f"非重置步统计：回合内最大水平位移={max_drift:.3f}m 最大前向速率={max_speed:.3f}m/s "
          f"最大倾角={max_tilt:.1f}deg（包含起立瞬态）")
    if hasattr(env, "has_stood"):
        print(f"是否曾满足训练稳站判据：{ever_stood}（不代表此后持续稳定）")
    print("未触发重置不等于稳定；结合高度、位移、速度、姿态和稳站判据判断。")
    if metrics:
        metrics.save(metrics_report, {
            "task": args.task, "load_run": train_cfg.runner.load_run,
            "checkpoint": train_cfg.runner.checkpoint, "seed": env_cfg.seed,
            "cmd_vx": cmd_vx, "cmd_yaw": cmd_yaw, "cmd_height": cmd_height,
            "sim_seconds": sim_seconds, "control_dt": env.dt, "physics_dt": env.sim_params.dt,
            "episode_timeout_s": env_cfg.env.episode_length_s,
            "action_delay_ms": actual_delay_ms, "stochastic": stochastic,
            "gas_spring_force_n": getattr(env, "gas_spring_force_n", 0.0),
            "push_schedule": scheduled_push.to_dict() if scheduled_push else None,
            "scheduled_push_impulse_world_ns": scheduled_impulse.tolist(),
            "push_mode": "schedule" if scheduled_push else ("random" if with_pushes else "none"),
            "velocity_error_sampling": "actor推理前的encoder与同一时刻10ms位置差分机体系真速度",
            "noise": with_noise, "domain_rand": with_dr, "terrain": env_cfg.terrain.mesh_type,
            **push_metadata,
            "push_force_sampling": "末个物理子步已施加的世界系力，每个策略步记录一次；重置行不记录力和计数",
            "push_duration_measurement": "非零力采样数乘 control_dt，精度为策略步；排除重置步",
            "push_count_scope": "每回合已开始的推扰次数；重置后重新计数",
            "contact_friction_requested": contact_friction,
            "ground_friction": env_cfg.terrain.static_friction,
            "robot_friction": env.friction_coef[0].item() if env_cfg.domain_rand.randomize_friction else None,
            "curriculum_unlocked": env.standup_curriculum_unlocked,
            "success_height": env._current_standup_success_height(),
            "success_pg_z": env_cfg.standup.success_projected_gravity_z,
            "success_duration_s": env_cfg.standup.success_duration_s,
            "initial_position": env_cfg.init_state.pos,
            "initial_dof_pos": env_cfg.standup.initial_dof_pos,
            "config_snapshot": standup_config_snapshot,
        })
    if recorder:
        recorder.save(pd_error_report, {
            "task": args.task, "load_run": train_cfg.runner.load_run,
            "checkpoint": train_cfg.runner.checkpoint, "seed": env_cfg.seed,
            "stochastic": stochastic, "noise": with_noise, "domain_rand": with_dr,
            "gas_spring_force_n": getattr(env, "gas_spring_force_n", 0.0),
            **push_metadata,
            "config_snapshot": standup_config_snapshot,
            "sim_seconds": sim_seconds, "dt": env.sim_params.dt,
            "terrain_friction": env_cfg.terrain.static_friction,
            "randomized_robot_friction": (
                env.friction_coef[0].item()
                if env_cfg.domain_rand.randomize_friction else None
            ),
            "resets": n_dones, "timeouts": n_timeouts, "ever_stood": ever_stood,
        })


if __name__ == "__main__":
    main()
