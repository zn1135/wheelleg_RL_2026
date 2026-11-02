# Isaac 旧任务与历史入口

最终验收现统一为GPU、共用运动门槛及报告，见[统一GPU验收](../interfaces/20261102-unified-gpu-acceptance.md)。主线统一入口为`scripts/mjlab.sh accept --task <任务> --recording <GPU记录> --output <新目录>`；起立单列，旧CPU流程仅保留历史含义。

涉及 `wheel_legged_gym/`、Isaac 历史模型或对应 sim2sim 时先读本文；mjlab 当前任务无需例行读取。本文件由根 AGENTS.md 的旧任务说明迁移而来，保护要求继续适用。

## 环境与验证

- 只用已安装 Isaac Gym 的 Python 3.8 环境，禁止系统 Python；`import isaacgym` 必须早于 `import torch`，不得把 mjlab 新依赖装进该环境。
- 环境激活、训练、TensorBoard、回放和 sim2sim 命令统一查 [COMMANDS.md](../../COMMANDS.md) 的“Isaac旧环境准备”及后续章节；本机路径按其当前配置核对。
- 正式流程查[构建与测试](build-test.md)的“环境”“正式流程”：策略／控制改动按对应任务执行训练 → Isaac GPU回放与验收；imcawl sim2sim 按自检 → 站立 → 行走执行。执行训练仍须用户逐次授权明确预算。
- `--selfcheck` 只验网络形状；没有单元测试框架或 CI，不得据此宣称行为通过。失稳对照用 `sim2sim/eval_isaac.py`，改机器人模型后用 `sim2sim/check_model.py`，具体参数见上述文档。

## 必须保留的保护规则

- Isaac sim2sim／部署只用完整 `model_*.pt`，禁止使用缺 encoder 的 `policy_1.pt`。这条不代表所有 mjlab Actor 都没有 encoder；按对应任务的导出接口核对。
- imcawl 的前进方向是机体 **+y**，不得按 +x 写观测或控制；其 yaw 命令是航向保持外环的实时反馈，不得常数喂入。Isaac 串联腿使用 +x 和偏航角速度命令，不能套用 imcawl 外环。详见[部署契约](sim2sim.md)。
- 不得回退 `sim2sim/mj_sim2sim.py` 的三处修复：`mjOBJ_XBODY`、读派生状态前 `mj_forward`、PD 内环“算力矩 → 步进 → 差分 dof_vel”。原因见[sim2sim约定](sim2sim.md)。
- 配置子类覆盖内部类必须继承原内部类，例如 `class physx(LeggedRobotCfg.sim.physx)`；不能写成不继承的 `class physx`，否则会丢失兄弟字段。
- 改 `resources/robots/imcawl/urdf/` 时必须同步 `sim2sim/imcawl.xml` 及配置 `asset.l1/l2`，并执行 `check_model.py`；URDF／XML 同步要求见[sim2sim约定](sim2sim.md)。
- imcawl／Isaac 旧策略默认验证不得超过 **±1.5 m/s**；≥1.8 m/s 刹车瞬态失稳的原因见[sim2sim约定](sim2sim.md)。mjlab 串联腿使用其自身接口包线。
- 不得 import 或修改 `wheel_legged_gym/logs/` 的陈旧源码副本；真实代码在 `wheel_legged_gym/envs/`，训练资产在根 `logs/`，禁止删除、移动或覆盖历史资产。

## 代码定位

先按[模块职责](modules.md)核对分支和机器人映射；不能假定所有维护线上的 `mini_wheel_legged` 都对应 imcawl。

| 职责 | 入口 |
|---|---|
| 环境基类、基础配置 | [legged_robot.py](../../wheel_legged_gym/envs/base/legged_robot.py)、[legged_robot_config.py](../../wheel_legged_gym/envs/base/legged_robot_config.py) |
| imcawl 环境 | [mini_wheel_legged/](../../wheel_legged_gym/envs/mini_wheel_legged/) |
| Isaac 串联腿与起立 | [chuanliantui/](../../wheel_legged_gym/envs/chuanliantui/)、[chuanliantui_standup/](../../wheel_legged_gym/envs/chuanliantui_standup/) |
| 任务注册 | [envs/__init__.py](../../wheel_legged_gym/envs/__init__.py)，新任务必须注册 |
| 内嵌算法与训练／回放 | [rsl_rl/](../../wheel_legged_gym/rsl_rl/)、[scripts/](../../wheel_legged_gym/scripts/) |
| sim2sim、机器人资产 | [sim2sim/](../../sim2sim/)、[imcawl资产](../../resources/robots/imcawl/) |

## 已删除配方的历史查找

旧 mjlab V14／r5～r15／recover／V11 配方于2026-10-15删除，历史可从 `backup/gpt-structure-20261015` 查找。该信息不是切换分支、恢复旧实现或替换当前任务的授权；当前任务仍查根 AGENTS.md 的对应入口。
