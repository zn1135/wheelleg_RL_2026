# AGENTS.md

本文件约束 AI 助手。团队流程遵守 [CONTRIBUTING.md](CONTRIBUTING.md)；下文用户确认要求仅约束 AI，不作为团队成员日常审批流程。

## 项目与文档入口

- **mjlab 当前主线**：`mjlab_training/`，入口 `scripts/mjlab.sh`；任务包括 `chuanliantui`、`chuanliantui_stand`、`chuanliantui_stand_v2`。任务配方、接口与日志分别核对，不互套模型。
- **Isaac 旧任务**：`wheel_legged_gym/`，用于 imcawl、旧串联腿与历史模型；与 mjlab 环境隔离。
- 当前最好模型、验收结论和逐日实验只记录在[项目实验记录](docs/ai/project-log.md)顶部；此文件不重复任务参数或实验状态。

开始时核对当前分支及已有改动（`git branch --show-current`、`git status --short`），不假定目标为 `main`。按本次任务读取下表对应文档的相关章节；已读且未变化的内容不重复整篇读取。先用搜索定位，再读取必要段落。

| 本次任务 | 需要查阅 |
|---|---|
| 协作流程、分工、分支 | [CONTRIBUTING.md](CONTRIBUTING.md)；维护线与机器人映射见[模块职责](docs/ai/modules.md) |
| 定位代码、改实现 | [架构与数据流](docs/ai/architecture.md)、[模块职责与任务注册](docs/ai/modules.md) |
| mjlab 任务参数、命令、验证 | 对应任务：[multimode](docs/mjlab-chuanliantui.md)、[稳站](docs/mjlab-chuanliantui-stand.md)、[稳站v2](docs/mjlab-chuanliantui-stand-v2.md)；[构建与测试](docs/ai/build-test.md)、[命令速查](COMMANDS.md) |
| Isaac 旧任务或历史入口 | **先读**[旧任务保护规则](docs/ai/isaac-legacy.md)，再查其中对应命令与模块 |
| 改观测、控制时序、URDF/XML，或真机部署 | **另读**[sim2sim约定](docs/ai/sim2sim.md) |
| 模型或跨侧交付 | **另读**[部署接口约定](docs/deployment-contract.md)，填写其关联的接口／模型交付模板 |
| 部署端代码 | 核对实际路径，**先读**该目录的协作规则（内置：[H7规则](deployment/H7_RL/md/AGENTS.md)）及其引用的固件规范 |

## 运行与验证约束

- **不得用系统 Python 或混用环境**：mjlab 只用 `scripts/mjlab.sh` 或 `mjlab_training/.venv/bin/python`；Isaac 只用已安装 Isaac Gym 的 Python 3.8 环境，且先导入 `isaacgym` 再导入 `torch`。不得向旧环境安装新依赖。
- 默认交付命令，不自行启动训练或 TensorBoard。只有用户明确授权本次训练并给定预算才启动；预算到达即停止，检查通过不授权追加。
- 每次交付训练命令，同时提供 TensorBoard 命令，`--logdir` 指向实际训练的 `logs/` 子目录；mjlab 使用本仓虚拟环境的 `python -m tensorboard.main`。仅改初始位置或根复位分布时，交付训练命令前先给出20环境可视预检命令；其他情况可直接 headless，具体命令查对应任务文档。
- 本仓库无单元测试框架、无 CI。执行对应任务的正式检查与行为验收；`check` 的有限物理／张量检查、`--selfcheck` 的形状检查和 `eval` 的地面起立统计均不能单独代表行为验收。新增自动检查放 `scripts/agent/` 并说明范围，不以一次性临时命令替代正式验证、不声称跑过单测。
- 串联腿按[双轮平衡标准](docs/interfaces/chuanliantui-wheel-balance-acceptance-v2.md)及对应任务文档验收，原成功样本任一失效即判退化；[multimode平地跟踪](docs/interfaces/chuanliantui-huananhu-multimode-tracking-v1.md)另行测量，两者互不替代。
- 保留已有 `logs/` 模型与日志，新增运行和导出使用新目录；不得删除、移动或覆盖历史产物。**不得 import 或修改 `wheel_legged_gym/logs/`**，那是陈旧源码副本，真实代码在 `wheel_legged_gym/envs/`。

## 工作区与部署分工

- 默认在当前工作区、当前分支开发并保留已有暂存与未提交改动。未经用户当次明确要求，不创建或切换分支、不创建或使用 worktree，不启用任务注册、资源锁或私有任务目录等工作流。
- 用户负责训练侧，另一位协作者负责 H7。同仓管理不授权擅改另一侧的控制代码或物理参数；接口／模型交付共同核对。
- H7 默认位于 `deployment/H7_RL/`，实际路径由 `.env.local` 的 `H7_REPO_PATH` 或用户显式指定；配置见 [CONTRIBUTING.md](CONTRIBUTING.md)。内置目录核对本仓分支与改动，外部克隆另核对分支、改动及远程 URL；实现与真实时序以[H7代码总览](deployment/H7_RL/md/file-guide.md)及实际源码为准。
- 接口、时序或模型交付须完成相应记录与验证；H7 负责人未核对时标“待确认”，不得把 AI 自查写成对方复查，也不得把仿真成绩写成真机通过。

## AI 的 Git 操作授权

- 提交标题与正文使用中文。每次 `git commit` 前，先展示暂存文件范围、拟用提交信息、是否推送及目标远程分支，并等待用户明确确认。未确认不得提交、amend、推送或创建 PR；只确认提交时不得自行推送。
- 用户本人维护的改动按其指定维护分支直接提交／推送，不自动要求 PR 或创建功能分支；其他协作者按 [CONTRIBUTING.md](CONTRIBUTING.md) 执行。影响部署接口也不自动改为 PR，但仍须接口记录、部署同步和验证，H7 核对标“待确认”；推送后提醒用户通知对方，不代发通知。
- 免 PR 不等于授权 Git 操作，推送须有用户当次明确授权。遇到 detached HEAD 或维护分支被其他工作区占用，先核对并保留两处改动，按用户授权整合，不因此强制要求 PR。提交不得夹带他人修改，推送前检查服务器地址、凭据等不宜公开的信息。
