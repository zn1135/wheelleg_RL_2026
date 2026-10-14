# 华南虎 multimode `model_3023` ONNX 使用说明

本文只适用于历史 checkpoint `model_3023.pt`（SHA-256 `624ded630635f0fb51a7378cdd04a7350eee98ffb305acbe1104c3cc7705a88a`）。它是 39 维 actor，不能替换仓库中旧 25/125 维、100 Hz 的 H7 网络，也不能加载到 V11 的 35/78 actor。

checkpoint 位于 `artifacts/multimode-v1-replay-20261014/models/`；专用 XML 与地面姿态库位于版本化资源目录 `resources/robots/chuanliantui_new_1/multimode_model3023/`。正式入口为：

```text
scripts/export_huananhu_multimode_onnx.py
sim2sim/mj_sim2sim_huananhu_multimode.py
```

前者只接受该 checkpoint，并在导出前校验 SHA-256；后者只读取专用 XML 和姿态库，不改写默认训练资产。

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
| `10:16` | 关节位置；顺序 `[lf0, lf00, rf0, rf00, lfwheel, rfwheel]`，轮位置固定为 0 |
| `16:22` | 同一顺序的关节速度 × 0.1 |
| `22:28` | 同一顺序的上一拍动作 |
| `28:35` | 模式 `[normal, stair, slope, recover, jump, jump_target_height×5, jump_state_time]` |
| `35:39` | 状态标志 `[ground_start, policy_active, airborne, step_active]` |

静站交付固定使用 `mode[28:35]=[1,0,0,0,0,0,0]`、`flags[35:39]=[0,1,0,0]`，即普通起始、策略已激活、未腾空、未遇到台阶。地面起立回放开始时使用 `ground_start=1`、`policy_active=0` 和 recover 模式；0.2 秒后才激活策略。

调用端每 20 ms（50 Hz）构造一次观测并推理。每个动作保持十个 2 ms 物理步。腿目标是 `q_reference + 0.5 × action`，腿 PD 为 `Kp=60`、`Kd=2`；轮目标速度为 `10 × action` rad/s，轮 PD 为 `Kp=0.12`、`Kd=0.02`，目标速度限幅 150 rad/s。力矩限幅为腿 40 N·m、轮 3.9 N·m。上述规则、气弹簧和闭链实体映射都在 ONNX 图外。

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

已完成的行为与数值证据见 [multimode 接口记录](interfaces/chuanliantui-huananhu-multimode-replay-v1.md) 和 [模型交付记录](model-deliveries/20261014-huananhu-multimode-model3023-replay.md)。这些结果仅覆盖仿真，不代表 H7 已适配或真机已验证。
