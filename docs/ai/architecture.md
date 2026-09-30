# 架构与数据流

## 来源

本仓库 fork 自 [legged_gym](https://github.com/leggedrobotics/legged_gym)，并把 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) **内嵌**为 `wheel_legged_gym/rsl_rl/`（不是 pip 依赖）。改 RL 算法直接改这个目录，不要去装外部 rsl_rl。

## 训练与部署仓库

| 仓库 | 职责 | 本机位置 |
|---|---|---|
| Wheel-Legged-Gym（本仓库） | 强化学习训练、策略导出与 MuJoCo sim2sim 验证 | 当前克隆的仓库根目录 |
| [H7_RL](https://github.com/zn1135/H7_RL.git) | 板端策略部署与真机控制 | `.env.local` 中的 `H7_REPO_PATH` |

本机配置与团队流程见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。涉及板端实现时，以 H7_RL 对应版本的代码和仓库说明为准；远程名称因克隆而异，跨仓库操作前分别核对 URL、分支、版本与未提交改动。接口差异及同步要求见 [部署接口约定](../deployment-contract.md)。

## 训练数据流

```
train.py
  └─ task_registry.make_env(name)      # envs/__init__.py 里注册的 (EnvClass, EnvCfg, TrainCfg)
  └─ task_registry.make_alg_runner()
       └─ OnPolicyRunner (rsl_rl/runners/on_policy_runner.py)
            ├─ ActorCriticSequence (rsl_rl/modules/)
            └─ PPO (rsl_rl/algorithms/ppo.py)
                 └─ RolloutStorage (rsl_rl/storage/)

每步：
  env.step(actions)                     # envs/base/legged_robot.py
    ├─ action_fifo 延迟随机化（0~10ms）
    ├─ decimation 次内环：compute_torques → gym.simulate → 差分更新 dof_vel
    ├─ post_physics_step：算观测、奖励、终止、reset
    └─ 返回 obs / privileged_obs / rewards / dones
```

## 观测结构（mini_wheel_legged / imcawl）

三路观测，缺一不可：

| 名称 | 维度 | 去处 |
|---|---|---|
| `obs` | 27 | actor 输入的一半 |
| `obs_history` | 135 = 27×5 | 喂 encoder，输出 3 维 latent |
| `privileged_obs` | 143 | 只给 critic，部署时不存在 |

`action = actor(cat(obs_27, encoder(history_135)))`。

latent 的前 3 维被监督为 `base_lin_vel * 2`（见 `rsl_rl/algorithms/ppo.py`），即 encoder 兼任**速度估计器**。部署端没有真实 `base_lin_vel`，全靠这个估计——它是当前性能上限的来源，详见 [sim2sim.md](sim2sim.md)。

`obs_history_length = 5` 意味着 encoder 只有 50ms 窗口。

## 配置继承

`base_config.py` 提供属性递归实例化机制；`LeggedRobotCfg` / `LeggedRobotCfgPPO` 是所有任务的基类。子类**继承内部类**来局部覆盖：

```python
class Mini_WheelLeggedCfg(LeggedRobotCfg):
    class sim(LeggedRobotCfg.sim):
        class physx(LeggedRobotCfg.sim.physx):   # 必须继承，否则未列出的字段全丢
            contact_offset = 0.02
```

**不继承就会丢掉基类的兄弟字段**（`solver_type`、`num_threads` 等）。这是本仓库最容易踩的配置坑。

奖励函数按约定装配：`rewards.scales` 里每个非零字段名 `foo` 都会去找环境类的 `_reward_foo()` 方法。把 scale 设为 0 即禁用该项。

## 奖励函数清单

奖励函数随分支和版本变化，以下为 imcawl 相关实现的索引。`MiniWheelLegged` override 了 `_reward_tracking_lin_vel`、`_reward_tracking_lin_vel_enhance`、`check_termination`。

跟踪类：`tracking_lin_vel`、`tracking_lin_vel_enhance`、`tracking_ang_vel`、`tracking_ang_vel_enhance`、`tracking_lin_vel_pbrs`、`tracking_ang_vel_pbrs`（后两个是 PBRS 势能形式）

姿态与高度：`lin_vel_z`、`ang_vel_xy`、`orientation`、`base_height`、`base_height_enhance`、`nominal_state`（惩罚双腿不对称）

能耗与平滑：`torques`、`power`（τ·ω）、`dof_vel`、`dof_acc`、`action_rate`、`action_smooth`（动作二阶差分）

限位与接触：`dof_pos_limits`、`dof_vel_limits`、`torque_limits`、`collision`、`stumble`、`feet_contact_forces`、`termination`、`stand_still`

跨维护线移植奖励时核对所选版本的实现与配置；不能仅凭任务同名认定奖励一致。

## VMC 工作原理

VMC（Virtual Model Control）把关节控制抽象到**虚拟腿**空间，实现见 `envs/wheel_legged_vmc/wheel_legged_vmc.py`：

1. 策略输出 `theta0`（腿摆角）、`L0`（腿长）、轮速
2. 正运动学 `forward_kinematics(theta1, theta2)` 求出当前 `L0 / theta0`，并差分得到 `L0_dot / theta0_dot`
3. 虚拟腿空间 PD 得到虚拟力 `F` 与力矩 `T`，再经雅可比映射为两个关节力矩
4. 轮子仍是直接 PD 速度控制

用途是把开链（wl）与闭链机构的运动控制统一到同一套虚拟腿接口，换机构不必重训策略。注意 `LeggedRobotVMC` 的 27 维观测与基类结构不同（用 `L0/theta0` 替代原始关节量）。

## 坐标约定

imcawl 的**前进方向是机体 +y 轴，不是 x**。腿装在机体 x 两侧（x=±0.179），轮绕 x 轴滚动。

基类 `LeggedRobot` 是为 wl 机器人（x 前向）写的，`MiniWheelLegged` 子类 override 了 `_reward_tracking_lin_vel` 系列改用 `base_lin_vel[:, 1]`。**给 imcawl 写任何涉及方向的代码前，先确认用的是哪个轴。**
