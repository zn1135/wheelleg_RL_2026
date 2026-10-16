# 华南虎 multimode `model_3023` ONNX 使用说明

本文只适用于历史 checkpoint `model_3023.pt`（SHA-256 `624ded630635f0fb51a7378cdd04a7350eee98ffb305acbe1104c3cc7705a88a`）。它是 39 维 actor，不能替换仓库中旧 25/125 维、100 Hz 的 H7 网络，也不能加载到 V11 的 35/78 actor。

checkpoint 位于 `artifacts/multimode-v1-replay-20261014/models/`；专用 XML 与地面姿态库位于版本化资源目录 `resources/robots/chuanliantui_new_1/multimode_model3023/`。正式入口为：

```text
scripts/export_huananhu_multimode_onnx.py
sim2sim/mj_sim2sim_huananhu_multimode.py
```

前者只接受该 checkpoint，并在导出前校验 SHA-256；后者只读取专用 XML 和姿态库，不改写默认训练资产。

## 本次导出的文件

2026-10-15 导出结果：

| 项目 | 内容 |
|---|---|
| ONNX 文件 | [model_3023_opset13.onnx](model_3023_opset13.onnx) |
| 文件大小 | 208259 字节，权重内嵌，无需额外权重文件 |
| ONNX SHA-256 | `fff95187813a11e066a4467db261579cfcfba2ee0e73df8b0d2e06053f2f0777` |
| 导出配置 | opset 13、float32、固定 batch=1 |
| 数值核对 | 256 组输入，最大绝对误差 `1.9073486328125e-06`，小于 `1e-4` |
| 详细记录 | [本次模型交付记录](20261015-model3023-onnx-export.md) |

给 H7 协作者的完整文件集见 [H7 适配交付包](../../chuanliantui_multimode/model3023-h7-20261015/README.md)，包含 ONNX、参考输入输出（NPZ／CSV）、导出记录和校验清单；该目录随本说明纳入版本管理，协作者拉取包含它的提交后即可取得。旧 `artifacts/` 导出目录继续保留作本机证据。

ONNX 是保存网络计算和权重的通用格式；**opset 是计算操作的版本号，13 不表示输入维数**。batch=1 表示每次推理一台机器人的一组观测。actor 是根据观测产生动作的策略网络；本图结构为 `39 → 256 → 128 → 64 → 6`，三个隐藏层使用 ELU 激活函数，最后输出不经过 `tanh` 限幅。

与以前 Isaac Gym 串联腿模型的区别：旧模型输入为 25 维当前观测和 125 维历史观测，带用于处理历史的编码器；本模型只有一个 39 维输入，不需要历史观测输入，也没有 `latent` 输出。旧脚本默认 opset 17、H723 示例也使用过 opset 13，因此这不是 Isaac Gym 与 mjlab 各自固定的格式区别。

## 环境与导出

该历史模型依赖原仓的 Python 3.12/mjlab 环境。以下命令在仓库根目录执行；先按本机实际位置设置解释器。导出目录使用新目录，避免覆盖已有交付文件。

```bash
MLAB_PYTHON=./mjlab_training/.venv/bin/python
OUT_DIR=$(mktemp -d artifacts/multimode-v1-replay-20261014/manual-onnx-XXXXXX)

# 固定 batch=1、opset 13：优先给 Cube.AI 导入。
"$MLAB_PYTHON" scripts/export_huananhu_multimode_onnx.py \
  --opset 13 --output "$OUT_DIR/model_3023_opset13.onnx"

# 固定 batch=1、opset 18：电脑端参考版本。
"$MLAB_PYTHON" scripts/export_huananhu_multimode_onnx.py \
  --opset 18 --output "$OUT_DIR/model_3023_opset18.onnx"
```

导出器执行 ONNX checker，并以 256 个固定随机 `float32[1,39]` 输入逐项比较 PyTorch actor 与 `onnx.reference.ReferenceEvaluator`。成功后同目录会生成：

| 文件 | 内容 |
|---|---|
| `*.onnx` | 内嵌权重的单文件 ONNX 图 |
| `*_reference_vectors.npz` | `obs` 与配对的 PyTorch `actions`，供板端数值核对 |
| `*_export.json` | checkpoint／ONNX SHA-256、图形状、数值误差和动作顺序 |

先前在本机 `artifacts/` 中生成的 ONNX、参考向量和记录是被忽略的历史证据，不随 Git 发布。请使用上述命令在新的输出目录重新导出，并以同目录 `*_export.json` 的 SHA-256 与数值检查为准。

## 推理接口

图中只有确定性 actor，不包含观测构造、历史缓存、PD、力矩限制、气弹簧或实体关节映射。

| 张量 | 类型与形状 | 含义 |
|---|---|---|
| `obs` | `float32[1,39]` | 已缩放、已裁剪的单机器人 actor 观测 |
| `actions` | `float32[1,6]` | 动作均值，顺序 `[lf0, lf00, lfwheel, rf0, rf00, rfwheel]` |

`obs` 的顺序如下。所有区间都是从零开始、右端不含。

| 索引 | 内容 |
|---|---|
| `0:4` | `[vx_command, 0, yaw_rate_command, height_command × 5]` |
| `4:7` | 机体系角速度 × 0.5 |
| `7:10` | 机体系投影重力 |
| `10:16` | 腿关节角减去默认角度；顺序 `[lf0, lf00, rf0, rf00, lfwheel, rfwheel]`，轮位置固定为 0 |
| `16:22` | 同一顺序的关节速度 × 0.1 |
| `22:28` | 同一顺序的上一拍动作 |
| `28:35` | 模式 `[normal, stair, slope, recover, jump, jump_target_height×5, jump_state_time]` |
| `35:39` | 状态标志 `[ground_start, policy_active, airborne, step_active]` |

### 39 维逐项定义

下表每一行对应一个输入槽，索引从 0 开始。`q_nominal` 是训练控制器使用的默认电机关节角，不是每次启动时把当前读数置零；`a_prev` 是上一拍网络原始输出，不是 PD 力矩或缩放后的角度／轮速。角度使用弧度，角速度使用 rad/s，高度使用 m。

| 索引 | 字段／送入模型的值 | 含义 |
|---|---|---|
| 0 | `vx_command` | 机体 +x 前进速度命令，m/s |
| 1 | `0` | 横向速度命令占位，固定为 0 |
| 2 | `yaw_rate_command` | 绕机体 z 轴的转向角速度命令，rad/s；不是航向角 |
| 3 | `height_command × 5` | 目标机身高度 |
| 4 | `angular_velocity_x × 0.5` | 机体系 x 轴角速度 |
| 5 | `angular_velocity_y × 0.5` | 机体系 y 轴角速度 |
| 6 | `angular_velocity_z × 0.5` | 机体系 z 轴角速度 |
| 7 | `projected_gravity_x` | 重力单位方向在机体 x 轴的分量 |
| 8 | `projected_gravity_y` | 重力单位方向在机体 y 轴的分量 |
| 9 | `projected_gravity_z` | 重力单位方向在机体 z 轴的分量 |
| 10 | `q_lf0 - q_nominal_lf0` | 左侧 lf0 电机关节角偏差 |
| 11 | `q_lf00 - q_nominal_lf00` | 左侧 lf00 电机关节角偏差 |
| 12 | `q_rf0 - q_nominal_rf0` | 右侧 rf0 电机关节角偏差 |
| 13 | `q_rf00 - q_nominal_rf00` | 右侧 rf00 电机关节角偏差 |
| 14 | `0` | 左轮位置占位 |
| 15 | `0` | 右轮位置占位 |
| 16 | `velocity_lf0 × 0.1` | lf0 电机关节速度 |
| 17 | `velocity_lf00 × 0.1` | lf00 电机关节速度 |
| 18 | `velocity_rf0 × 0.1` | rf0 电机关节速度 |
| 19 | `velocity_rf00 × 0.1` | rf00 电机关节速度 |
| 20 | `velocity_lfwheel × 0.1` | 左轮角速度 |
| 21 | `velocity_rfwheel × 0.1` | 右轮角速度 |
| 22 | `a_prev[0]` | 上一拍 lf0 原始动作 |
| 23 | `a_prev[1]` | 上一拍 lf00 原始动作 |
| 24 | `a_prev[3]` | 上一拍 rf0 原始动作 |
| 25 | `a_prev[4]` | 上一拍 rf00 原始动作 |
| 26 | `a_prev[2]` | 上一拍左轮原始动作 |
| 27 | `a_prev[5]` | 上一拍右轮原始动作 |
| 28 | `normal` | 普通模式，0/1 |
| 29 | `stair` | 楼梯／台阶模式，0/1 |
| 30 | `slope` | 坡地模式，0/1 |
| 31 | `recover` | 恢复模式，0/1 |
| 32 | `jump` | 跳跃模式，0/1 |
| 33 | `jump_target_height × 5` | 跳跃目标高度；非跳跃时为 0 |
| 34 | `jump_state_time` | 当前跳跃阶段计时，秒；非跳跃时为 0 |
| 35 | `ground_start` | 本回合是否采用地面起立初态，0/1；不是实时接地信号 |
| 36 | `policy_active` | 策略是否已激活，0/1 |
| 37 | `airborne` | 是否腾空，0/1 |
| 38 | `step_active` | 台阶辅助是否激活，0/1 |

重力方向表示机器人朝哪边倾斜，直立时约为 `[0, 0, -1]`，不能直接填 IMU 的原始加速度或欧拉角。`lf0/lf00/rf0/rf00` 是模型内的实体驱动关节名，不能直接把 `lf00/rf00` 当作旧串联模型的虚拟膝角。

模式槽 28～32 只允许一个为 1，训练代码优先级为跳跃、楼梯／台阶、恢复、坡地、普通。模式和状态由图外逻辑提供，ONNX 不会自动生成这些输入，也不直接接收地形扫描数据。

### 输入裁剪与初始化

裁剪表示超过上下限时取边界值。按训练观测构造代码，**先裁剪、后乘缩放系数**：

| 原始量 | 裁剪范围 | 裁剪后缩放 |
|---|---|---|
| 前进／转向速度命令 | `[-100, 100]` | ×1 |
| 目标高度 | `[0, 1]` m | ×5 |
| 三轴角速度 | `[-100, 100]` rad/s | ×0.5 |
| 重力分量、关节角偏差 | `[-100, 100]` | ×1 |
| 关节速度 | `[-200, 200]` rad/s | ×0.1 |
| 上一拍原始动作 | `[-100, 100]` | ×1 |

这些是观测的数值裁剪范围，不代表已验证的机器人运动范围。正常推理不主动添加训练用噪声或延迟，也不额外做均值／方差归一化；模型按这里的固定缩放训练。复位时将上一拍动作清零。轮位置原始槽固定为 0；训练开启噪声时这些槽可能叠加训练噪声，实际调用无需模拟该噪声。

静站交付固定使用 `mode[28:35]=[1,0,0,0,0,0,0]`、`flags[35:39]=[0,1,0,0]`，即普通起始、策略已激活、未腾空、未遇到台阶。地面起立回放开始时使用 `ground_start=1`、`policy_active=0` 和 recover 模式；0.2 秒后才激活策略。

### 6 维输出与动作顺序

| 输出索引 | 电机 | 图外控制用途 |
|---|---|---|
| 0 | `lf0` | 左腿位置目标偏移 |
| 1 | `lf00` | 左腿位置目标偏移 |
| 2 | `lfwheel` | 左轮速度目标 |
| 3 | `rf0` | 右腿位置目标偏移 |
| 4 | `rf00` | 右腿位置目标偏移 |
| 5 | `rfwheel` | 右轮速度目标 |

**输出按左右两侧排列，观测按四腿、两轮排列，两者顺序不同。** 将输出保存为下一拍输入时，使用以下重排；代码中切片右端不包含在内：

```python
# last_action 保存上一拍 actions[0]，首次推理或复位时为六个 0。
obs[22:28] = np.clip(last_action[[0, 1, 3, 4, 2, 5]], -100.0, 100.0)
```

动作是无量纲原始网络输出，可能超过 `[-1, 1]`，不能自行加上该限幅。调用端每 20 ms（50 Hz）构造一次观测并推理，每个动作保持十个 2 ms 控制步。

PD 控制是根据目标与实际状态的偏差计算电机力矩。现有独立 CPU sim2sim 使用以下图外控制；这里的 `q_reference` 来自专用 XML 的 `qpos0`，不是实机编码器零点：

```text
腿角目标 = clip(q_reference + 0.5 × action, q_reference - 3.14, q_reference + 3.14)
腿力矩   = clip(60 × (腿角目标 - 当前角度) - 2 × 当前角速度, -40, 40) N·m
轮速目标 = clip(10 × action, -150, 150) rad/s
轮力矩   = clip(0.12 × (轮速目标 - 当前轮速) - 0.02 × 当前轮速, -3.9, 3.9) N·m
```

训练侧腿目标使用 `q_nominal + 0.5 × action`，`q_nominal` 由任务默认姿态配置决定，限幅中心是 XML 的 `q_reference`；对接时须同时核对观测减去的默认角和控制目标使用的默认角。上述参数说明已有仿真实现，不是实机零点或极性标定。气弹簧、闭链映射、力矩限幅和策略激活门控都在 ONNX 图外。

### Python 推理调用示例

以下使用已安装的 ONNX 参考执行器，适合核对数值；输入 `obs` 须已按上表构造，不能用全零观测代替真实反馈来控制机器人。

```python
import numpy as np
from onnx.reference import ReferenceEvaluator

# 初始化时只加载一次模型。
policy = ReferenceEvaluator(
    "exports/chuanliantui_multimode/model3023-h7-20261015/model_3023_opset13.onnx"
)
last_action = np.zeros(6, dtype=np.float32)

# 每个策略周期：obs 是调用方构造好的 39 项观测。
obs[22:28] = np.clip(last_action[[0, 1, 3, 4, 2, 5]], -100.0, 100.0)
model_input = np.asarray(obs, dtype=np.float32).reshape(1, 39)
actions = policy.run(None, {"obs": model_input})[0]  # 输出形状 [1, 6]
last_action = actions[0].copy()  # 保存原始动作，供下一策略周期使用。
```

此片段仅展示一次推理的数据衔接，不包含传感器采集、循环调度或电机控制。

## 仿真验证

已交付的 GPU/Warp 固定 NORMAL 静站证据保存在模型交付记录中；正式代码入口不依赖被忽略的原仓快照。独立 CPU MuJoCo sim2sim 显式接收同一份 opset 13 ONNX。平地 normal 静站运行 20 秒：

```bash
"$MLAB_PYTHON" sim2sim/mj_sim2sim_huananhu_multimode.py \
  --onnx "$OUT_DIR/model_3023_opset13.onnx" --seconds 20
```

可视地面起立回放会持续到窗口关闭；点击 MuJoCo 窗口后按 `R` 可随机抽取下一组历史地面姿态，`Home` 重置当前启动姿态，`F1` 暂停或继续：

```bash
"$MLAB_PYTHON" sim2sim/mj_sim2sim_huananhu_multimode.py \
  --onnx "$OUT_DIR/model_3023_opset13.onnx" --render --seconds 0 --initial-pose ground
```

已完成的行为与数值证据见 [multimode 接口记录](../../../docs/interfaces/chuanliantui-huananhu-multimode-replay-v1.md) 和 [模型交付记录](../../../docs/model-deliveries/20261014-huananhu-multimode-model3023-replay.md)。这些结果仅覆盖仿真，不代表 H7 已适配或真机已验证。

本次重新导出只执行 ONNX 图检查和数值比较，没有重跑行为仿真、Cube.AI 导入或板端／真机验证。H7 部署负责人对 39 维输入、50 Hz 调度、关节映射及板端数值的核对状态为待确认。

## 定义来源

- [基础观测构造](../../../mjlab_training/src/ct_mjlab/envs/chuanliantui/v14_observations.py)：关节排列、默认角扣除、缩放、裁剪及上一拍动作。
- [39 维观测拼接](../../../mjlab_training/src/ct_mjlab/envs/chuanliantui/multimode_observations.py)：在原 35 维后追加四个状态标志。
- [模式及状态构造](../../../mjlab_training/src/ct_mjlab/envs/chuanliantui/multimode_mdp.py)：模式优先级、跳跃信息、状态标志。
- [训练侧动作控制](../../../mjlab_training/src/ct_mjlab/envs/chuanliantui/huananhu_action.py)：原始动作保存、PD、动作延迟和激活门控。
- [ONNX 导出器](../../../scripts/export_huananhu_multimode_onnx.py)：网络结构、模型哈希检查、固定形状和数值验证。
- [独立 CPU sim2sim](../../../sim2sim/mj_sim2sim_huananhu_multimode.py)：观测打包与仿真控制实现。
