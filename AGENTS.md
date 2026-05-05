# AGENTS.md — Wheel-Legged-Gym 项目速览

## 项目概述

**Wheel-Legged-Gym** 是基于 NVIDIA Isaac Gym 的 **轮足机器人强化学习训练框架**，用于训练轮足机器人（腿关节+轮子）在复杂地形上的运动策略。基於 ETH Zurich RSL 的 [legged_gym](https://github.com/leggedrobotics/legged_gym) 和 [rsl_rl](https://github.com/leggedrobotics/rsl_rl)。

- 算法: PPO (Proximal Policy Optimization)
- 物理引擎: Isaac Gym GPU 并行仿真
- 机器人: 6-DOF 轮足机器人（2条腿×2关节 + 2个轮子），有开链（wl）和闭链（xwl）两个型号

---

## 目录结构

```
Wheel-Legged-Gym/
├── wheel_legged_gym/            # 主 Python 包
│   ├── __init__.py              # 定义 ROOT_DIR / ENVS_DIR 路径常量
│   ├── scripts/
│   │   ├── train.py             # 训练入口
│   │   └── play.py              # 评估/部署入口
│   ├── envs/                    # 任务环境定义
│   │   ├── __init__.py          # 注册4个任务到 task_registry
│   │   ├── base/
│   │   │   ├── base_task.py     # BaseTask（VecEnv 兼容的抽象基类）
│   │   │   ├── base_config.py   # BaseConfig（递归类→实例的配置系统）
│   │   │   ├── legged_robot.py  # LeggedRobot（主环境类，~1800行）
│   │   │   └── legged_robot_config.py  # 基础配置类
│   │   ├── wheel_legged/        # 任务1: 端到端开链机器人（PD控制）
│   │   │   └── wheel_legged_config.py
│   │   ├── wheel_legged_vmc/    # 任务2: VMC控制（粗糙地形）
│   │   │   ├── wheel_legged_vmc.py      # LeggedRobotVMC 类
│   │   │   └── wheel_legged_vmc_config.py
│   │   ├── wheel_legged_vmc_flat/  # 任务3: VMC控制（平地，低VRAM）
│   │   │   └── wheel_legged_vmc_flat_config.py
│   │   └── mini_wheel_legged/   # 任务4: 迷你轮足机器人（xwl）
│   │       └── mini_wheel_legged_config.py
│   ├── rsl_rl/                  # PPO 算法库（从 rsl_rl fork）
│   │   ├── algorithms/ppo.py    # PPO 实现
│   │   ├── runners/on_policy_runner.py  # 训练循环
│   │   ├── modules/
│   │   │   ├── actor_critic.py           # ActorCritic (MLP)
│   │   │   ├── actor_critic_sequence.py   # ActorCriticSequence (encoder+MLP)
│   │   │   ├── actor_critic_recurrent.py  # RNN-based
│   │   │   └── normalizer.py
│   │   ├── storage/rollout_storage.py  # GAE + mini-batch
│   │   └── utils/
│   └── utils/
│       ├── task_registry.py     # TaskRegistry 全局单例
│       ├── helpers.py           # get_args, class_to_dict, export_policy_as_jit
│       ├── math.py              # quat_apply_yaw, wrap_to_pi
│       ├── terrain.py           # 地形生成
│       └── logger.py            # 评估日志绘图
├── resources/robots/
│   ├── wl/                      # 开链轮足机器人 URDF（8.8kg）
│   │   └── urdf/wl.urdf
│   └── xwl/                     # 闭链迷你轮足机器人 URDF（~1kg）
│       └── urdf/xwl.urdf
└── logs/                        # 训练日志与模型保存
```

---

## 架构总览

```
train.py / play.py
  │
  └─→ task_registry (全局单例, wheel_legged_gym/utils/task_registry.py)
        │
        ├─→ make_env(task_name)
        │     └─→ 实例化 LeggedRobot 或 LeggedRobotVMC
        │           └─→ 继承链: BaseTask → LeggedRobot → LeggedRobotVMC
        │
        └─→ make_alg_runner(env, task_name)
              └─→ OnPolicyRunner
                    ├─→ ActorCritic / ActorCriticSequence  (策略网络)
                    ├─→ PPO                                 (算法)
                    └─→ RolloutStorage                      (经验存储)
```

### 训练循环

```
OnPolicyRunner.learn():
  for iteration in range(max_iterations):
    1. ROLLOUT (no_grad):
       actions = ppo.act(obs, obs_history, critic_obs)
       obs, rew, done, infos, obs_history = env.step(actions)
       ppo.process_env_step(rew, done, infos, next_obs)
    2. COMPUTE RETURNS:
       ppo.compute_returns(critic_obs)   # GAE
    3. UPDATE:
       ppo.update()                      # PPO clipped surrogate loss
```

### env.step() 内部流程

```
env.step(actions):
  for i in range(decimation):     # 默认 = 2
    1. action_fifo 延迟缓冲区 (domain rand)
    2. _compute_torques(action)   # 动作→关节力矩
    3. gym.apply_rigid_body_force_tensors (推扰)
    4. gym.simulate(sim)
    5. gym.refresh_dof_state_tensor(sim)
  post_physics_step():
    1. 刷新 root_states, contact_forces, rigid_body_states
    2. 计算 base_lin_vel, base_ang_vel, projected_gravity
    3. 计算正运动学 L0, theta0
    4. check_termination()        # 碰撞/翻倒/超时/出界
    5. compute_reward()           # 遍历 _reward_* 方法
    6. reset_idx(env_ids)         # 重置已终止的环境
    7. compute_observations()     # 组装 obs_buf + obs_history
```

---

## 核心类

### LeggedRobot (`wheel_legged_gym/envs/base/legged_robot.py:57`)

主环境类，继承 BaseTask。

| 方法 | 功能 |
|------|------|
| `__init__` | 解析配置、初始化 buffer、准备 reward 函数 |
| `step(actions)` | 执行 decimation 次物理步进，调用 post_physics_step |
| `post_physics_step` | 刷新状态、检查终止、计算奖励、重置、计算观测 |
| `check_termination` | 检查接触终止(>10N)、翻倒(projected_gravity.z>-0.1)、超时、出界 |
| `compute_reward` | 聚合所有非零 scale 的 `_reward_*` 方法 |
| `compute_observations` | 组装 27 维观测 + 观测历史 + 特权观测 |
| `_compute_torques` | PD 控制器：动作→关节力矩 |
| `reset_idx` | 重置指定环境：清 buffer、重采样指令、记录 episode 统计 |
| `_init_buffers` | 初始化所有 GPU tensor（关节状态、接触力、运动学等）|
| `_prepare_reward_function` | 根据 reward_scales 构建 reward_functions 列表 |

### LeggedRobotVMC (`wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc.py:1`)

继承 LeggedRobot，重写力矩计算和观测结构。核心区别在于使用 **VMC（Virtual Model Control）** ：

- 动作空间变为 `[theta0, L0, wheel_vel]` ×2（腿角度、腿长度、轮速）
- `_compute_torques`: PD 控制 theta0/L0 → VMC → 关节力矩
- `VMC(F, T)`: 虚拟力/力矩 → 关节力矩的雅可比映射
- `compute_proprioception_observations`: 观测包含 theta0, theta0_dot, L0, L0_dot

### BaseTask (`wheel_legged_gym/envs/base/base_task.py`)

VecEnv 兼容的抽象基类，负责创建 simulation、分配基础 buffer、viewer 管理。

### BaseConfig (`wheel_legged_gym/envs/base/base_config.py`)

递归配置系统：`__init__` 自动将嵌套的类定义实例化为对象。

### TaskRegistry (`wheel_legged_gym/utils/task_registry.py`)

全局单例 `task_registry`，管理任务注册、环境创建、算法 runner 创建、配置文件备份。

---

## 已注册的任务

| 任务名 | 环境类 | 描述 |
|--------|--------|------|
| `wheel_legged` | LeggedRobot | 端到端 PD 控制，开链机器人，各种地形 |
| `wheel_legged_vmc` | LeggedRobotVMC | VMC 控制，开链机器人，粗糙地形 |
| `wheel_legged_vmc_flat` | LeggedRobotVMC | VMC 控制，平地（低 VRAM） |
| `mini_wheel_legged` | LeggedRobot | PD 控制，迷你闭链机器人 xwl |

所有注册在 `wheel_legged_gym/envs/__init__.py:52-70`。

---

## 观测空间 & 动作空间

### 标准 LeggedRobot 观测 (27维)

```
[0:3]   base_ang_vel × obs_scales.ang_vel
[3:6]   projected_gravity
[6:9]   commands[:, :3] × commands_scale    # lin_vel_x, ang_vel_yaw, height
[9:15]  (dof_pos - default_dof_pos) × obs_scales.dof_pos   # 6 joints
[15:21] dof_vel × obs_scales.dof_vel
[21:27] actions                              # 上一步动作
```

### LeggedRobotVMC 观测 (27维，结构不同)

```
[0:3]   base_ang_vel × obs_scales.ang_vel
[3:6]   projected_gravity
[6:9]   commands[:, :3] × commands_scale
[9:11]  theta0 × obs_scales.dof_pos         # 腿角度 (2)
[11:13] theta0_dot × obs_scales.dof_vel     # 腿角速度 (2)
[13:15] L0 × obs_scales.l0                  # 腿长度 (2)
[15:17] L0_dot × obs_scales.l0_dot          # 腿长度变化率 (2)
[17:19] dof_pos[:, [2,5]] × obs_scales.dof_pos  # 轮位置 (2)
[19:21] dof_vel[:, [2,5]] × obs_scales.dof_vel  # 轮速度 (2)
[21:27] actions
```

### 特权观测（仅训练时 critic 使用）

额外包含: base_lin_vel, 上一步动作, dof_acc, height_measurements, torques, base_mass, base_com, default_dof_pos offsets, friction/restitution。

### 动作空间 (6维)

| 模式 | 6维含义 |
|------|---------|
| PD (LeggedRobot) | [左腿关节0, 左腿关节1, 左轮, 右腿关节0, 右腿关节1, 右轮] |
| VMC (LeggedRobotVMC) | [左 theta0, 左 L0, 左轮速, 右 theta0, 右 L0, 右轮速] |

---

## 奖励函数（LeggedRobot 定义的所有 `_reward_*`）

位于 `wheel_legged_gym/envs/base/legged_robot.py:1628-1811`。只有 `reward_scales` 中非零的才会生效。

| 奖励函数 | 含义 |
|----------|------|
| `tracking_lin_vel` | 跟踪指令线速度 x：exp(-error²/σ) |
| `tracking_lin_vel_enhance` | 增强线速度跟踪 |
| `tracking_ang_vel` | 跟踪指令角速度 yaw |
| `tracking_ang_vel_enhance` | 增强角速度跟踪 |
| `tracking_lin_vel_pbrs` | PBRS 线速度势能奖励 |
| `tracking_ang_vel_pbrs` | PBRS 角速度势能奖励 |
| `lin_vel_z` | 惩罚基座 z 轴线速度 |
| `ang_vel_xy` | 惩罚基座 xy 轴角速度 |
| `orientation` | 惩罚俯仰/翻滚角 |
| `base_height` | 惩罚偏离目标高度 |
| `base_height_enhance` | 指数高度误差奖励 |
| `torques` | 惩罚大关节力矩 |
| `power` | 惩罚机械功率 (τ·ω) |
| `dof_vel` | 惩罚关节速度 |
| `dof_acc` | 惩罚关节加速度 |
| `action_rate` | 惩罚动作变化率 |
| `action_smooth` | 惩罚动作二阶差分 |
| `collision` | 惩罚指定部位接触 |
| `termination` | 终止奖励（非超时） |
| `dof_pos_limits` | 惩罚超出关节限位 |
| `dof_vel_limits` | 惩罚接近速度限值 |
| `torque_limits` | 惩罚接近力矩限值 |
| `stumble` | 惩罚足端水平接触力 |
| `stand_still` | 惩罚零指令时的运动 |
| `nominal_state` | 惩罚双腿不对称 |
| `feet_contact_forces` | 惩罚过大的足端接触力 |

---

## 配置继承链

```
BaseConfig
  └── LeggedRobotCfg               # 基础环境配置
        ├── WheelLeggedCfg          # wl 机器人，PD 控制
        │     └── WheelLeggedVMCCfg       # VMC 控制（粗糙地形）
        │           └── WheelLeggedVMCFlatCfg  # VMC（平地）
        └── Mini_WheelLeggedCfg     # xwl 迷你机器人

BaseConfig
  └── LeggedRobotCfgPPO
        ├── WheelLeggedCfgPPO
        │     └── WheelLeggedVMCCfgPPO
        │           └── WheelLeggedVMCFlatCfgPPO
        └── Mini_WheelLeggedCfgPPO
```

---

## 常用命令

```bash
# 训练 VMC 平地
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_flat

# 训练 VMC 粗糙地形
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc

# 训练端到端
python wheel_legged_gym/scripts/train.py --task=wheel_legged

# 使用 TensorBoard
tensorboard --logdir=./ --port=8080

# 评估/演示
python wheel_legged_gym/scripts/play.py --task=wheel_legged_vmc_flat

# 其他参数: --headless, --num_envs=1024, --seed=42, --max_iterations=5000

# 查看所有参数
python wheel_legged_gym/scripts/train.py --help
```

---

## VMC 工作原理

VMC（Virtual Model Control）将低层关节控制抽象为**虚拟腿**控制：

1. 策略输出 `theta0`（腿角度）、`L0`（腿长度）、`wheel_vel`（轮速）
2. PD 控制器在虚拟腿空间计算:
   - `torque_leg = kp_theta * (theta0_ref - theta0) - kd_theta * theta0_dot`
   - `force_leg = kp_l0 * (L0_ref - L0) - kd_l0 * L0_dot`
3. `VMC(F, T)` 通过雅可比将虚拟力/力矩转换为两关节力矩 `[T1, T2]`
4. 轮子直接 PD 速度控制

VMC 的优势：将开链(wl)和闭链(xwl)机器人的运动控制统一到虚拟腿空间，无需针对每个机器人重新训练。

---

## 添加新任务的步骤

1. 在 `wheel_legged_gym/envs/` 下新建目录
2. 创建 `your_config.py`，继承已有配置类
3. 如需新逻辑，创建 `your_env.py` 继承 LeggedRobot/LeggedRobotVMC
4. 在 `wheel_legged_gym/envs/__init__.py` 注册
5. 将奖励 scale 设为 0 可禁用奖励，设为非零可启用

---

## 关键文件索引

| 文件 | 行数 | 功能 |
|------|------|------|
| `wheel_legged_gym/envs/base/legged_robot.py` | 1811 | 主环境类，step/reward/obs/termination |
| `wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc.py` | 817 | VMC 力矩映射和观测 |
| `wheel_legged_gym/rsl_rl/algorithms/ppo.py` | 295 | PPO 算法：act/update/GAE |
| `wheel_legged_gym/rsl_rl/runners/on_policy_runner.py` | 340 | 训练循环：rollout+learn+log+save |
| `wheel_legged_gym/rsl_rl/modules/actor_critic.py` | 177 | MLP 策略网络 |
| `wheel_legged_gym/rsl_rl/modules/actor_critic_sequence.py` | 227 | 序列策略（encoder+MLP） |
| `wheel_legged_gym/rsl_rl/storage/rollout_storage.py` | 334 | GAE 计算和 mini-batch 生成 |
| `wheel_legged_gym/utils/task_registry.py` | 246 | 任务注册中心 |
| `wheel_legged_gym/utils/terrain.py` | 243 | 地形生成 |
| `wheel_legged_gym/scripts/train.py` | 55 | 训练入口 |
| `wheel_legged_gym/scripts/play.py` | 231 | 评估入口 |
