# AGENTS.md

本文件约束 AI 助手的工作方式。团队通用流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，AI 同样遵守；本文中的用户确认要求不作为团队成员日常提交的审批流程。

## 项目是什么

串联腿的新训练入口为独立的 `mjlab_training/` Python 3.12 / mjlab 环境，采用华南虎 V14 flat v0 普通 PPO；迁移范围、实体关节接口、命令和验证见 [mjlab 串联腿说明](docs/mjlab-chuanliantui.md)。下文 Isaac Gym / Python 3.8 约定仍适用于旧环境、旧模型和旧检查脚本；不得把新依赖装进旧环境。新任务无需 `import isaacgym`，其普通 actor 没有旧 encoder，旧 `policy_1.pt` 禁令不适用于新接口的独立 ONNX 导出。历史 `logs/` 保护和默认不启动训练／TensorBoard仍适用于两个入口。

轮足机器人强化学习训练仓，fork 自 legged_gym 并内嵌 rsl_rl。在 Isaac Gym 里训练策略，经 MuJoCo sim2sim 验证后通向真机部署。训练仓为纯 Python，无编译步骤，Linux + NVIDIA GPU；H7 部署仓有独立的固件构建流程。

已记录的维护线包括 `26_wheelleg`（含 chuanliantui 与 imcawl）、`大腿`（imcawl）、`XML`（`mini_wheel_legged` 绑定 xwl）。动手前先 `git branch --show-current`，不假定合入目标为 `main`；分支用途与机器人映射见 [docs/ai/modules.md](docs/ai/modules.md) 的“分支分叉与维护线”。

## 关联部署仓库

- 真机部署代码已纳入 [deployment/H7_RL](deployment/H7_RL/README.md)，与训练代码使用同一 Git 仓库；原始来源为 [H7_RL](https://github.com/zn1135/H7_RL.git)，导入版本及文件哈希见 `deployment/H7_RL/SOURCE_IMPORT.json`。
- 本机代码路径由 `.env.local` 的 `H7_REPO_PATH` 提供，推荐指向本仓 `deployment/H7_RL`；仍可显式使用外部部署克隆。配置方式见 [协作指南](CONTRIBUTING.md)，公共示例见 [.env.example](.env.example)。
- 本仓库负责强化学习训练与 MuJoCo sim2sim 验证；H7_RL 负责板端策略部署与真机控制。
- 两人维护：用户本人负责RL训练侧，另一位协作者负责 `deployment/H7_RL/`；接口与模型交付共同核对，具体目录分工见 [协作指南](CONTRIBUTING.md#两人维护分工)。AI按本次任务范围开发，不因同仓管理擅自改动另一侧的控制代码或物理参数。
- 涉及部署端代码时，先检查实际路径，读取 `deployment/H7_RL/AGENTS.md` 及其引用的固件规范。内置目录核对本仓分支与改动；外部克隆另核对其分支、改动及远程 URL。内置部署代码仍是旧接口，不能直接运行新 mjlab 35维实体关节策略。

## 开始任务前先读

- 团队流程：[CONTRIBUTING.md](CONTRIBUTING.md)
- 架构与数据流：[docs/ai/architecture.md](docs/ai/architecture.md)
- 模块职责与任务注册：[docs/ai/modules.md](docs/ai/modules.md)
- 构建与测试：[docs/ai/build-test.md](docs/ai/build-test.md)
- 改动观测/控制时序/URDF/XML，或做真机部署时**另读**：[docs/ai/sim2sim.md](docs/ai/sim2sim.md)
- 涉及模型或跨仓库交付时另读：[部署接口约定](docs/deployment-contract.md)，填写其中关联的接口／模型交付模板
- 命令速查：[COMMANDS.md](COMMANDS.md)

## 正式构建与测试

无编译。必须用装了 Isaac Gym 的 Python 3.8 conda 环境，不能用系统 python；本机路径见 [COMMANDS.md](COMMANDS.md) 开头，下文 `python` 均指它。`import isaacgym` 要在 `import torch` 之前。

```bash
# imcawl 训练示例；其他机器人按 COMMANDS.md 选择任务
python wheel_legged_gym/scripts/train.py --task=mini_wheel_legged --headless
# 在另一终端监控；仅交付命令，默认不启动
tensorboard --logdir=logs/mini_wheel_legged --port=8080

# Isaac 内回放
python wheel_legged_gym/scripts/play.py --task=mini_wheel_legged

# MuJoCo sim2sim：自检 → 站立 → 行走，按顺序走完
python sim2sim/mj_sim2sim.py --selfcheck --checkpoint <model_*.pt>
python sim2sim/mj_sim2sim.py --render --checkpoint <model_*.pt> --cmd_vx 0 --cmd_height 0.30 --init_height 0.30
python sim2sim/mj_sim2sim.py --render --checkpoint <model_*.pt> --cmd_vx 1.0 --init_height 0.30

# 改了 imcawl.xml / URDF
python sim2sim/check_model.py

# 失稳时的 Isaac 对照组
python sim2sim/eval_isaac.py --load_run <run> --checkpoint <n> --cmd_vx 0.5
```

**本仓库没有单元测试框架，也没有 CI。** 验证 = 上面训练 → 回放 → sim2sim 三步，靠人眼判读终端输出。`--selfcheck` 只检查网络形状、不检查行为，不能单独当作验证通过。不要声称跑过单测，也不要用这些之外的临时命令代替正式流程。

**训练命令交付：默认只向用户提供可执行命令，不得自行启动训练或 TensorBoard。每次提供训练命令时，必须同时提供 TensorBoard 命令，用训练输出所在的 `logs/` 目录作为 `--logdir`。仅当改动机器人的初始位置（`init_state.pos` 或等价的根节点初始位置）时，训练命令才应先使用非无头、`--num_envs 20` 的可视预检；其他训练命令可直接大规模 headless。**

## 代码入口

| 模块 | 路径 | 职责 |
|---|---|---|
| 环境基类 | [wheel_legged_gym/envs/base/legged_robot.py](wheel_legged_gym/envs/base/legged_robot.py) | 仿真步进、观测构造、奖励、域随机化 |
| 基础配置 | [wheel_legged_gym/envs/base/legged_robot_config.py](wheel_legged_gym/envs/base/legged_robot_config.py) | 所有任务的配置基类 |
| imcawl 环境 | [wheel_legged_gym/envs/mini_wheel_legged/](wheel_legged_gym/envs/mini_wheel_legged/) | override 前向轴与倾角终止 |
| chuanliantui 环境 | [wheel_legged_gym/envs/chuanliantui/](wheel_legged_gym/envs/chuanliantui/) | 25 维观测、+x 前向；起立任务在 [chuanliantui_standup/](wheel_legged_gym/envs/chuanliantui_standup/) |
| 任务注册 | [wheel_legged_gym/envs/__init__.py](wheel_legged_gym/envs/__init__.py) | 新任务必须注册才能用 `--task` |
| RL 算法 | [wheel_legged_gym/rsl_rl/](wheel_legged_gym/rsl_rl/) | 内嵌 rsl_rl：PPO / ActorCriticSequence / runner |
| 训练回放入口 | [wheel_legged_gym/scripts/](wheel_legged_gym/scripts/) | `train.py` / `play.py` |
| sim2sim | [sim2sim/](sim2sim/) | imcawl 以 `mj_sim2sim.py` 头注释为契约来源；chuanliantui 见 `mj_sim2sim_ct.py` 与对应训练实现 |
| 机器人资产 | [resources/robots/imcawl/](resources/robots/imcawl/) | URDF / mesh / 关节配置 |

## 禁止事项

- 不得用系统 python 或其他环境跑本仓库；`import isaacgym` 必须在 `import torch` 之前。
- 不得用导出的 `policy_1.pt` 做 sim2sim / 部署——它缺 encoder。只用 `model_*.pt`。
- 不得回退 `sim2sim/mj_sim2sim.py` 里 MuJoCo 观测读取的三处修复（`mjOBJ_XBODY`、读状态前 `mj_forward`、PD 内环「算力矩→步进→差分 dof_vel」顺序）。理由见 docs/ai/sim2sim.md。
- 不得给 imcawl 写基于「x 是前向」的代码——它的前进方向是机体 **+y**。
- imcawl 部署端不得把 yaw 命令当常数喂——训练时它是航向保持外环的反馈值。chuanliantui 使用偏航角速度命令，不套用该外环；按 [部署接口约定](docs/deployment-contract.md) 区分。
- 配置子类覆盖内部类时不得省略继承（写 `class physx(LeggedRobotCfg.sim.physx)`，不是 `class physx`），否则静默丢掉基类兄弟字段。
- 改 `resources/robots/imcawl/urdf/` 后不得不同步 `sim2sim/imcawl.xml`（以及 config 里的 `asset.l1/l2`），改完必须跑 `check_model.py`。
- **不得删除、移动、覆盖仓库根 `logs/` 下的任何内容。** 那是历史训练产物（`model_*.pt` 与 tensorboard 记录），用途是回放旧策略、与新训练结果做效果对比，属于不可再生资产。它被 gitignore 因此没有 Git 兜底，删了就没了。需要腾空间时先问，不要自行判断哪个 run「没用了」。
- 不得 import 或修改 `wheel_legged_gym/logs/`——注意这是**另一个目录**，里面只有 `envs/` 的陈旧 `.py` 副本、不含任何训练产物，被 gitignore 的 `logs` 规则连带忽略，已与 `envs/` 分叉。真实环境代码在 `wheel_legged_gym/envs/`。
- 当前策略安全包线 **±1.5 m/s**，不要在验证脚本里默认超过它（≥1.8 m/s 刹车瞬态会发散翻车，根因见 docs/ai/sim2sim.md）。

## 工作区约定

默认直接在当前工作区和分支开发。团队使用功能分支的约定不授权 AI 自动切换分支；除非用户在当次请求中明确要求，不得创建或切换分支、创建或使用 Git worktree，也不启用任务注册、资源锁或私有任务目录等 worktree 工作流。开始时核对并保留已有暂存与未提交改动，提交范围不得夹带他人修改。

## AI 的 Git 操作授权

- Git 提交信息（标题和正文）必须使用中文，团队提交与 PR 规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。
- 每次实际执行 `git commit` 前，必须先向用户展示暂存文件范围和拟用的中文提交信息，并等待用户明确确认；未确认时不得提交、amend、推送或创建 PR。
