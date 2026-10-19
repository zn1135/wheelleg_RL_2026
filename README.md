# Wheel-Legged-Gym

轮腿机器人的强化学习训练、MuJoCo仿真验证与H7部署工程。当前串联腿主线使用 **mjlab / Python 3.12**；Isaac Gym / Python 3.8保留给旧任务。两套环境分别使用，不能混装依赖或互换模型。

## 从哪里开始

| 要做什么 | 阅读入口 |
| --- | --- |
| 运行串联腿训练、回放、评估与导出 | [mjlab串联腿说明](docs/mjlab-chuanliantui.md) |
| 查看当前模型与实验结论 | [项目实验记录顶部](docs/ai/project-log.md) |
| 在独立MuJoCo中运行策略 | [multimode ONNX使用说明](docs/onnx-huananhu-multimode-model5849.md) |
| 理解训练代码与修改位置 | [mjlab代码结构](mjlab_training/README.md)、[模块职责](docs/ai/modules.md) |
| 理解H7反馈、推理和电机执行 | [H7工程代码总览](deployment/H7_RL/md/file-guide.md) |
| 核对训练与实机接口 | [部署接口约定](docs/deployment-contract.md) |
| Jetson只读实机镜像、串口与后续电脑复算 | [实机调试说明（镜像软件已实现，待联机）](docs/deployment-contract.md#host-debug-plan) |
| 运行Isaac旧任务 | [命令速查](COMMANDS.md)、[构建与测试](docs/ai/build-test.md) |

团队协作、本机路径和提交约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。AI助手另读 [AGENTS.md](AGENTS.md)；涉及部署端时还须读 [H7协作规则](deployment/H7_RL/md/AGENTS.md)。

## 当前代码怎样运行

```text
mjlab_training/：仿真反馈 → 39维策略观测 → 6维动作 → 控制器 → 仿真机器人
sim2sim/：相同接口的策略 → 独立CPU MuJoCo → 数值与运动对照
deployment/H7_RL/：实机反馈 → 板端策略 → 电机控制 → DM/DJI驱动
```

“39维”表示每次输入策略的39个数；82维critic观测只供训练时评价动作使用，不是实机推理输入。当前接口为 `chuanliantui-huananhu-multimode-39-82-v1`，默认大机策略50Hz、电机控制500Hz。六个动作中，四个控制腿的位置目标、两个控制轮速目标，不能把网络输出直接当作力矩。

H7源码已同步 `model_5849` 的39维观测／6维动作接口，支持公共遥控和平地NORMAL／RECOVER；当前48路遥测改为只读镜像，旧LQR／RL布局保留为源码注释。Jetson接收与闭链显示入口已实现，实际联机／标定仍待确认，见[镜像说明](docs/deployment-contract.md#host-debug-plan)。板端数值、真实时序、物理映射与恢复效果仍待确认，见[部署同步记录](docs/interfaces/20261018-h7-model5849-deployment-sync.md)。当前没有接通39维电脑接管协议，导出、模型、板端和真机结果分别记录。

## 运行环境与验证

| 范围 | 使用的环境 | 命令入口 |
| --- | --- | --- |
| mjlab当前主线及39维CPU MuJoCo回放 | `mjlab_training/.venv`，Python 3.12 | `scripts/mjlab.sh` 的 train、play、eval、export、check；CPU回放见ONNX说明 |
| Isaac旧任务与旧25/125维工具 | 已安装Isaac Gym的Python 3.8 conda环境 | `wheel_legged_gym/scripts/`；先 `import isaacgym`，再 `import torch` |
| H7固件 | 部署工程的eIDE/GCC或Keil配置 | [H7构建说明](deployment/H7_RL/md/file-guide.md#build) |

安装与完整命令以 [mjlab说明](docs/mjlab-chuanliantui.md#可执行命令) 和 [COMMANDS.md](COMMANDS.md) 为准。训练默认只交付命令，启动和追加训练需要用户逐次授权。

本仓没有单元测试框架或CI。`check` 只检查有限物理与张量状态；`eval` 统计地面起立，均不能代替[双轮平衡验收](docs/interfaces/chuanliantui-wheel-balance-acceptance-v2.md)或[平地跟踪测量](docs/interfaces/chuanliantui-huananhu-multimode-tracking-v1.md)。板端和真机验证另行完成。

## 目录

| 路径 | 内容 |
| --- | --- |
| `mjlab_training/src/ct_mjlab/` | 当前训练任务、观测、动作控制、奖励与训练入口 |
| `sim2sim/` | 独立MuJoCo回放及历史接口诊断工具；先确认模型版本 |
| `resources/robots/` | 机器人XML、URDF、网格和姿态库 |
| `deployment/H7_RL/` | 同仓管理的H7固件；当前入口在 `md/file-guide.md` |
| `wheel_legged_gym/` | Isaac旧任务与内嵌rsl_rl |
| `scripts/agent/` | 有明确检查范围的验证与诊断脚本 |
| `docs/` | 接口、模型交付、使用说明和实验记录 |
| `logs/` | 历史训练产物；不可删除、移动或覆盖 |

不要使用 `wheel_legged_gym/logs/` 中的陈旧代码副本。旧25/125维S2R1／HPI1工具的适用范围见[历史上位机工具说明](docs/ai/s2r-offline.md)，不能直接用于当前39维固件。

## 来源

本项目基于 [legged_gym](https://github.com/leggedrobotics/legged_gym) 和 [rsl_rl](https://github.com/leggedrobotics/rsl_rl)。相关论文：[Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)、[Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion](https://arxiv.org/abs/2202.05481)。
