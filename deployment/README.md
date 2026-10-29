# H7 工程与原版备份

2026-10-30本次PR将`H7_RL`同步为本地部署仓29ef907：1999/35维、50Hz策略/500Hz PD、UART8 VOFA与独立UART10 Jetson，连同控制结构及LQR调参成套更新。`H7_RL_ros`不随本次替换。下方2026-10-20内容为历史说明；当前来源与验证见[本次同步记录](../docs/interfaces/20261030-h7-model1999-deployment-sync.md)，模型见[1999导出目录](../exports/onnx/model_1999/)。

2026-10-20撤销PR #4新版同步时，主工程与原版备份各1609个来源文件均与 `83290520b76ef55a04cf84e54c58763967cea129` 一致。随后按用户要求，仅在 `H7_RL_ros` 接入板载USB实机姿态镜像，两目录现已产生下述差异。

**`H7_RL_ros` 通过板载USB CDC向Jetson上传姿态，`H7_RL` 保持原UART实现。** `H7_RL_ros` 的目录后缀不表示包含ROS节点；Jetson直接用MuJoCo显示模型。

| 目录 | 来源 | 用途 |
| --- | --- | --- |
| [H7_RL](H7_RL/md/file-guide.md) | 当前维护分支 `26_wheelleg`，提交 `aa676b2a5bd7b5ae4fd9f2238709995dbd82c924`；H7 源码与 `8329052` 一致 | 受版本管理的主工程，使用现有默认部署路径 |
| [H7_RL_ros](H7_RL_ros/md/file-guide.md#vofa) | 基于原版 `83290520b76ef55a04cf84e54c58763967cea129` 增加USB镜像 | 独立USB姿态同步工程；H7控制、Jetson本地屏幕显示 |

| 比较项 | `H7_RL/` | `H7_RL_ros/` |
| --- | --- | --- |
| 文件内容 | 原主工程保持 | USB镜像发送与对应说明已修改 |
| 控制算法、网络权重与参数 | 原版实现 | 保持原值；USB镜像另加只读策略／输出诊断，UART探测回报槽置零 |
| 工程配置与编译入口 | 本目录内的 eIDE、Keil 工程 | 相同配置，使用备份目录内的工程入口 |
| 默认使用路径 | 现有部署说明和默认路径指向此处 | 需要显式打开此目录或将 `H7_REPO_PATH` 指向此处 |
| 维护用途 | 日常修改和部署的主工程 | 板载USB连接Jetson进行实机姿态同步 |

两份都是独立文件，修改其中一份不会自动同步到另一份。回退完成时的逐文件校验为各1609个文件、独有0个、不同0个；该结果是USB改动之前的历史记录，不代表当前仍相同。

分别打开各目录内的 `CtrBoard-H7_ALL.code-workspace`；Keil 工程分别在各自的 `MDK-ARM/CtrBoard-H7_ALL.uvprojx`。源码、工程配置和库独立存放，不使用指向另一份的链接，不复制旧编译缓存。使用外部缓存的构建命令前，核对 `rootDir` 指向所选工程，输出目录也分别设置，避免误编译或误用另一版固件。

当前分支为 `26_wheelleg`，本地 `codex/h7-uart10-probe` 分支已删除。[PR #4](https://github.com/zn1135/wheelleg_RL_2026/pull/4) 的新版代码未保留在这两个目录中。`.env.local` 的 `H7_REPO_PATH` 未改动；若其指向 `deployment/H7_RL`，现在读取的是恢复后的原版。

USB版的构建、烧录及Jetson联机结果见[接口记录](../docs/interfaces/20261020-h7-ros-usb-mirror.md)。历史验证只适用于其原记录版本，物理映射、控制行为及H7部署负责人核对仍待确认。
