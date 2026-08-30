# 模块职责

## 顶层

| 路径 | 职责 |
|---|---|
| `wheel_legged_gym/envs/` | 环境实现与配置，任务注册 |
| `wheel_legged_gym/rsl_rl/` | 内嵌的 rsl_rl（PPO / ActorCritic / runner / storage） |
| `wheel_legged_gym/scripts/` | `train.py`、`play.py` 入口 |
| `wheel_legged_gym/utils/` | `task_registry`、`helpers`（参数解析）、`terrain`、`logger`、`math` |
| `sim2sim/` | MuJoCo 部署验证与对照工具 |
| `resources/robots/` | URDF / mesh / 关节配置 |
| `logs/` | 训练输出（权重 + tensorboard）。**已 gitignore，且不可删** — 见下 |
| `docs/ai/` | 本文档目录 |

## 环境层

| 类 | 文件 | 说明 |
|---|---|---|
| `BaseTask` | `envs/base/base_task.py` | gym 接口骨架 |
| `LeggedRobot` | `envs/base/legged_robot.py` | 主力基类：仿真步进、观测构造、奖励、域随机化、地形 |
| `LeggedRobotVMC` | `envs/wheel_legged_vmc/wheel_legged_vmc.py` | 用 VMC 统一开链/闭链机构的运动控制 |
| `MiniWheelLegged` | `envs/mini_wheel_legged/mini_wheel_legged.py` | **当前主力**，imcawl 机器人。override 前向轴（+y）与倾角终止阈值 |

## 已注册任务

在 `envs/__init__.py` 里注册；新任务必须加到这里才能用 `--task` 调用。

| `--task` 值 | 环境类 | 机器人 | 说明 |
|---|---|---|---|
| `mini_wheel_legged` | `MiniWheelLegged` | imcawl | **当前主力**，`experiment_name = mini_wheel_legged` |
| `wheel_legged` | `LeggedRobot` | wl | 端到端开链，多地形 |
| `wheel_legged_vmc` | `LeggedRobotVMC` | wl | VMC，可迁移到闭链实机 |
| `wheel_legged_vmc_flat` | `LeggedRobotVMC` | wl | 平地，显存需求低 |
| `chuanliantui` | `Chuanliantui` | chuanliantui | 串联腿平地站立/行走 |
| `chuanliantui_standup` | `ChuanliantuiStandup` | chuanliantui | 固定趴姿起立并稳站 |

## 机器人资产

| 名称 | 路径 | 状态 |
|---|---|---|
| imcawl | `resources/robots/imcawl/urdf/imcawl.urdf` | 当前分支主力。MuJoCo 侧对应 `sim2sim/imcawl.xml`，两者必须手工保持一致 |
| wl | `resources/robots/wl/urdf/wl.urdf` | **在用**。`wheel_legged` 任务直接引用；`wheel_legged_vmc` / `wheel_legged_vmc_flat` 继承其 `asset` 段，也用它 |
| xwl | `resources/robots/xwl/urdf/xwl.urdf` | **在用，但 env config 在 `XML` 分支上**，见下节 |

imcawl 关键参数（`mini_wheel_legged_config.py`）：

- DOF 顺序 `[lf0, lf1, l_wheel, rf0, rf1, r_wheel]`
- 默认关节角 `[0.9, -1.62, 0, -0.9, 1.62, 0]`，对应站高约 0.30 m
- 连杆 `l1 = 0.21`（髋→膝）、`l2 = 0.25`（膝→轮心）；改 URDF 必须同步这两个值
- 腿 Kp=60 / Kd=2，轮 Kp=0 / Kd=0.5（轮是纯速度控制）
- 力矩上限 `[30, 30, 5, 30, 30, 5]` N·m
- 站高命令范围 `[0.20, 0.40]` m
- `fail_tilt_pg_z = -0.85`（约 32°）：基类 -0.1 太松，策略曾学会 ~40° 前倾斜靠停机作弊

## 分支分叉：mini_wheel_legged 绑定两台不同机器人

`mini_wheel_legged` 任务在两个分支上指向不同资产：

| 分支 | `Mini_WheelLeggedCfg.asset.file` |
|---|---|
| `大腿`（当前） | `resources/robots/imcawl/urdf/imcawl.urdf` |
| `XML` | `resources/robots/xwl/urdf/xwl.urdf` |

**xwl 不是废弃资产，是同一任务的前一代机器人**，它的 env config 活在 `XML` 分支。`logs/xwl_*` 下的 147 个权重是那条线的产物。

两分支在 `5892702` 分叉，`XML` 独有 6 个 commit，含基类改动（`alive` 奖励、reset 静态启动、屏蔽轮位置观测）——这些**当前分支没有**。跨分支引用代码或权重前先确认自己在哪个分支。`XML` 分支根目录另有一份 335 行的旧 AGENTS.md（`e02e598`），内容与本 `docs/ai/` 有重叠也有冲突，以当前分支为准。

## sim2sim 工具

| 文件 | 职责 |
|---|---|
| `mj_sim2sim.py` | MuJoCo 部署验证主脚本。文件头注释是**部署契约的权威来源**，真机实现照它抄 |
| `eval_isaac.py` | Isaac 干净环境对照组，分辨「策略问题」vs「引擎 gap」 |
| `check_model.py` | `imcawl.xml` 碰撞对与轮几何自检，改完 XML 跑一次 |

## 两个叫 logs 的目录，别搞混

| 路径 | 是什么 | 能不能动 |
|---|---|---|
| `logs/`（仓库根） | 训练产物：`model_*.pt` 权重 + tensorboard 记录，按 `<experiment_name>/<日期时间_run_name>/` 组织 | **只读。不可删、不可移动、不可覆盖** |
| `wheel_legged_gym/logs/` | `envs/` 的陈旧 `.py` 代码副本（2026-01-13 快照），不含任何训练产物 | 不要 import、不要改、不要参考 |

根 `logs/` 是不可再生资产：历史权重用于回放旧策略、与新训练结果做效果对比。它被 `.gitignore` 忽略，因此**没有 Git 兜底，删了就找不回来**。需要腾空间时先问用户，不要自行判断哪个 run「没用了」。

`wheel_legged_gym/logs/` 则是另一回事——它之所以也没进 Git，是因为 `.gitignore` 里 `logs` 那条规则不带路径前缀，把它一起连带忽略了。它是 `envs/` 在 2026-01-13 的快照，与当前 `envs/` 已分叉（`legged_robot_config.py` 差 162 行、`legged_robot.py` 差 18 行）且缺 `mini_wheel_legged`。全仓搜索确认**没有任何代码 import 它**，但它自带 `__init__.py` 并在其中调 `task_registry.register`，误 import 会注册出一套过时配置。真实环境代码只在 `wheel_legged_gym/envs/`。
