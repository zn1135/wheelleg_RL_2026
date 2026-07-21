#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isaac Gym 干净环境对照评估：与 mj_sim2sim 相同场景（平地、无随机化、固定命令）。

用途：sim2sim 排查的对照组。MuJoCo 里失稳时先跑这个——
    Isaac 也失稳  -> 策略本身的问题（回去调训练）
    Isaac 稳      -> 两引擎的 gap（调 MJCF 接触参数或加强域随机化）

用法:
    python sim2sim/eval_isaac.py --load_run Jul21_06-00-36_ --checkpoint 1000 --cmd_vx 0.5
"""
import isaacgym  # noqa: F401  必须在 torch 之前 import
from wheel_legged_gym.envs import *  # noqa: F401,F403  触发任务注册
from wheel_legged_gym.utils import get_args, task_registry
import torch
import sys


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
        else:
            argv.append(a)
    sys.argv = ["eval_isaac", "--task=mini_wheel_legged", "--headless"] + argv
    args = get_args()

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
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
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_Kp = False
        env_cfg.domain_rand.randomize_Kd = False
        env_cfg.domain_rand.randomize_motor_torque = False
        env_cfg.domain_rand.randomize_default_dof_pos = False
        env_cfg.domain_rand.randomize_action_delay = False
        env_cfg.domain_rand.randomize_compliance = False
    env_cfg.commands.curriculum = False
    # 固定命令（resample 也只会采到同一个值）
    env_cfg.commands.ranges.lin_vel_x = [cmd_vx, cmd_vx]
    env_cfg.commands.ranges.ang_vel_yaw = [cmd_yaw, cmd_yaw]
    env_cfg.commands.ranges.height = [cmd_height, cmd_height]
    # 关键：基类 heading_command=True 时 commands[:,1] 每步被航向控制器覆盖
    # （1.5*(目标航向-当前航向) 剪到±5），ang_vel_yaw 范围无效——必须把目标航向也钉死为 0，
    # 否则评估中每 resampling_time 秒收到一个 [-π,π] 随机航向目标，机器人被命令猛转弯。
    env_cfg.commands.ranges.heading = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_history = env.get_observations()
    # 关键：必须置 resume=True 才会加载 --load_run/--checkpoint 指定的权重
    # （对齐 play.py:71；漏掉这行会静默地跑随机初始化网络！）
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    print(f"已加载权重: load_run={train_cfg.runner.load_run} checkpoint={train_cfg.runner.checkpoint}")
    policy = ppo_runner.get_inference_policy(device=env.device)

    n_steps = int(sim_seconds / env.dt)
    ac = ppo_runner.alg.actor_critic
    print(f"Isaac 对照评估：{sim_seconds}s (cmd_vx={cmd_vx}, yaw={cmd_yaw}, h={cmd_height}, "
          f"动作={'采样(训练同款噪声)' if stochastic else '确定性均值'})")
    print("字段：z 高度 | v_fwd 机体系前向(y)速度 | pg_yz 重力投影 | done 是否触发终止(重置)")
    n_dones = 0
    for step in range(n_steps):
        with torch.no_grad():
            if stochastic:
                actions = ac.act(obs, obs_history)
            else:
                actions, _ = policy(obs, obs_history)
        obs, _, _, dones, infos, obs_history = env.step(actions.detach())
        n_dones += int(dones[0].item())
        if step % 100 == 0 or dones[0].item():
            z = env.root_states[0, 2].item()
            pg = env.projected_gravity[0].cpu().numpy()
            v_fwd = env.base_lin_vel[0, 1].item()
            cmd = env.commands[0, :3].cpu().numpy()
            print(f"[{step:5d}] z={z:.3f} v_fwd={v_fwd:+.3f} "
                  f"pg_yz=[{pg[1]:+.2f},{pg[2]:+.2f}] "
                  f"cmd=[{cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:.2f}] done={int(dones[0].item())}")
            if dones[0].item():
                print(f">>> 第 {step} 步({step*env.dt:.1f}s)触发终止：Isaac 里同样失稳")
    print(f"评估结束。{sim_seconds}s 内终止 {n_dones} 次（0 次 = 稳定）。")


if __name__ == "__main__":
    main()
