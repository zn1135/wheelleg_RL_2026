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
from pathlib import Path


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
    contact_friction = None
    pd_error_report = None
    standup_config_snapshot = None
    render = False
    frame_dir = None
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
        elif a == "--contact_friction":
            contact_friction = float(next(it))
        elif a == "--pd_error_report":
            pd_error_report = next(it)
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
    env_cfg.env.num_envs = 1
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
    policy = ppo_runner.get_inference_policy(device=env.device)
    recorder = LegErrorRecorder(env) if pd_error_report else None
    if recorder:
        if args.task != "chuanliantui_standup" or env.num_envs != 1:
            raise ValueError("PD 误差统计仅支持单环境 chuanliantui_standup")
        env._compute_torques = recorder
    # runner 构造时会 reset，加载权重又会恢复课程；重新取得匹配当前状态的观测。
    env.reset()
    obs, obs_history = env.get_observations()
    start_xy = env.root_states[0, :2].clone()

    n_steps = int(sim_seconds / env.dt)
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
            mask = env.get_policy_action_mask() if hasattr(env, "get_policy_action_mask") else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            latent = torch.zeros(env.num_envs, 3, device=env.device)
            if mask.any():
                if stochastic:
                    actions[mask] = ac.act(obs[mask], obs_history[mask])
                    latent[mask] = ac.latent
                else:
                    actions[mask], latent[mask] = policy(obs[mask], obs_history[mask])
        obs, _, _, dones, infos, obs_history = env.step(actions.detach())
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
    if recorder:
        recorder.save(pd_error_report, {
            "task": args.task, "load_run": train_cfg.runner.load_run,
            "checkpoint": train_cfg.runner.checkpoint, "seed": env_cfg.seed,
            "stochastic": stochastic, "noise": with_noise, "domain_rand": with_dr,
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
