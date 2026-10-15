# mjlab 串联腿训练（multimode 单线）

mjlab 训练侧只保留一条线：华南虎 multimode 任务，基线模型 `model_3023.pt`。2026-10-15 起 V14、r5～r15、recover、V11 等旧配方已删除，代码与文档可从分支 `backup/gpt-structure-20261015`（`e451b3d`）或其父提交 `109da6d` 找回。

## 接口

| 项目 | 内容 |
|---|---|
| 任务名 | `chuanliantui`（原 `Chuanliantui-Huananhu-Multimode-v0`） |
| 观测契约 | `chuanliantui-huananhu-multimode-39-82-v1` |
| Actor／Critic／动作 | 39／82／6 |
| 动作顺序 | `[lf0, lf00, lfwheel, rf0, rf00, rfwheel]` |
| 时序 | 2 ms 物理、20 ms／50 Hz 策略 |
| 地形 | 生成地形 flat 50%、slope 20%、step 15%、stair 15% |
| 初态 | 50% 普通／50% 地面，姿态库 `resources/robots/chuanliantui_new_1/mjlab/huananhu_pose_bank.npz` |
| 基线 checkpoint | `artifacts/multimode-v1-replay-20261014/models/model_3023.pt`，SHA256 `624ded630635f0fb51a7378cdd04a7350eee98ffb305acbe1104c3cc7705a88a` |

观测槽、模式与标志位见 [multimode 接口](interfaces/chuanliantui-huananhu-multimode-replay-v1.md)。

## 代码结构

与 Isaac 侧 `wheel_legged_gym/` 同样分工，见 [mjlab_training/README.md](../mjlab_training/README.md)。环境源码逐字节取自 `model_3023` 训练时的冻结源码快照，只改了 `model.py` 的资源路径；训练 XML 与姿态库同样取自该快照，SHA256 分别为 `dfe01afb…` 与 `ca82470a…`。

## 可执行命令

均在仓库根目录执行。环境安装：`scripts/setup_mjlab.sh <已装 mjlab 的 python>` 或 `uv sync --project mjlab_training --locked --python 3.12`。

```bash
# 有限检查（不训练）：加载模型、核对 39/82 维、步进若干次
scripts/mjlab.sh check --num-envs 2 --steps 16 \
  --checkpoint artifacts/multimode-v1-replay-20261014/models/model_3023.pt

# 训练（从零）；--resume 从 multimode checkpoint 完整续训
scripts/mjlab.sh train --num-envs 4096 --iterations 3000 --run-name <label>
scripts/mjlab.sh train --num-envs 2048 --iterations 300 --run-name <label> \
  --resume artifacts/multimode-v1-replay-20261014/models/model_3023.pt
# 训练监控
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui_huananhu_multimode_v1 --port 8080

# 可视回放（play 配置：无随机化、观测噪声与延迟）
scripts/mjlab.sh play --num-envs 1 --checkpoint <model_*.pt>

# 地面起立评估：64 个强制地面初态、20 秒
scripts/mjlab.sh eval --checkpoint <model_*.pt> --num-envs 64 --seeds 42 \
  --start-mode force --terrain flat --output logs/mjlab/evaluation/<name>/result.json

# ONNX 导出（固定 batch=1，opset 13 供 Cube.AI，18 供电脑回放）
scripts/mjlab.sh export --checkpoint <model_*.pt> --output exports/<name>.onnx --opset 13
```

改动初态或复位分布后，训练前先做 20 环境可视预检：`scripts/mjlab.sh play --num-envs 20`。

RTX 4060 Laptop（8 GiB）上 4096 环境曾在创建 Warp CUDA 图时显存不足，本机续训用 2048 环境。

独立 CPU MuJoCo sim2sim 和历史 ONNX 交付见 [ONNX 使用说明](onnx-huananhu-multimode-model3023.md)。

## 基线复核（2026-10-15）

迁移后用 `eval` 在 64 个强制地面初态、seed 42、20 秒下复跑 `model_3023`：初态签名与原基线一致（`458f15a3…`），严格起立 64/64，平地 31/31 全部存活，数值失败 0。存活 61/64，原记录为 62/64：差在 56 号楼梯样本，它在 19.16 s 越出 2.5 m 地块，不是跌倒，属于 GPU/Warp 逐轨迹不严格确定的范围。

`check` 和 `eval` 只做有限检查和地面起立统计，不是行为验收。行为验收按 [双轮平衡验收 v2](interfaces/chuanliantui-wheel-balance-acceptance-v2.md) 执行。
