# model_3023 ONNX 再导出交付

## 任务与来源

2026-10-15 按用户指定导出 `model_3023.pt`，保留原 39 维输入和 6 维动作，不修改模型或训练代码。机器人为串联腿，任务为 `Chuanliantui-Huananhu-Multimode-v0`；交付用途为模型格式转换与后续数值核对，不新增行为验收结论。接收人为用户，独立复查及 H7 部署负责人核对状态为待确认。

- 接口：`chuanliantui-huananhu-multimode-39-82-v1`，见[接口记录](../../../docs/interfaces/chuanliantui-huananhu-multimode-replay-v1.md)。
- 训练仓：`https://github.com/zn1135/wheelleg_RL_2026.git`，分支 `26_wheelleg`，完整提交 `269ba935eeb3a39fc3fedcb87b554a4f654ffcec`。
- 导出时工作区干净；交付准备增加本记录、ONNX 说明、接口交付说明及 `exports/` 文件包，无运行源码差异或新增运行源码，无需运行源码差异快照。
- 内置部署目录与训练仓同一提交；本次未修改或运行 H7，实际板端固件版本待核实，无提交或 PR。
- 历史训练配置、累计进度、资产及仿真证据沿用[原模型记录](../../../docs/model-deliveries/20261014-huananhu-multimode-model3023-replay.md)；本次未训练，不产生新训练命令或 TensorBoard 日志。未在本次重新核实历史训练 seed 和完整软件环境。
- 导出环境：仓库 `mjlab_training/.venv`，Python 3.12.3、PyTorch 2.14.0+cu130、ONNX 1.22.0、NumPy 2.5.3；CPU 导出，不运行 Isaac Gym 或 MuJoCo。

## 模型与产物

原始产物位于本机 [导出目录](../../../artifacts/model3023-onnx-20261015-FmsnA2)，被 Git 忽略。为交给 H7 协作者适配，已将模型及参考向量逐字节复制到 [H7 适配交付包](../../chuanliantui_multimode/model3023-h7-20261015/README.md)，随本交付记录一同纳入版本管理，供协作者拉取。

交付包另含无表头的 256×39 输入 CSV、256×6 期望输出 CSV，与 NPZ 的 float32 数值逐项一致。导出 JSON 保留原验证结果，将本机绝对路径改为仓库相对路径，并补充导出代码 SHA、工具版本及 H7 待确认状态。`SHA256SUMS` 覆盖交付目录内其他六个文件。源 checkpoint 已在仓库中跟踪，本次无需再次复制。

| 产物 | SHA-256 |
|---|---|
| [源 model_3023.pt](../../../artifacts/multimode-v1-replay-20261014/models/model_3023.pt) | `624ded630635f0fb51a7378cdd04a7350eee98ffb305acbe1104c3cc7705a88a` |
| [model_3023_opset13.onnx](model_3023_opset13.onnx) | `fff95187813a11e066a4467db261579cfcfba2ee0e73df8b0d2e06053f2f0777` |

ONNX 为 208259 字节，内嵌权重、固定 batch=1、opset 13；输入 `obs: float32[1,39]`，输出 `actions: float32[1,6]`，动作顺序 `[lf0, lf00, lfwheel, rf0, rf00, rfwheel]`。只导出 actor，不含观测构造、PD、限幅或电机映射。

## 导出与验证

2026-10-15 18:25（北京时间），在仓库根目录执行，退出码 0：

```bash
mjlab_training/.venv/bin/python scripts/export_huananhu_multimode_onnx.py \
  --opset 13 \
  --output artifacts/model3023-onnx-20261015-FmsnA2/model_3023_opset13.onnx
```

ONNX 图检查、固定输入输出形状检查通过。256 组固定随机输入逐个比较 ONNX 参考执行器与 PyTorch，最大绝对误差 `1.9073486328125e-06`，小于容差 `1e-4`。源 checkpoint 哈希由导出器强制检查。

- [导出日志](../../../artifacts/model3023-onnx-20261015-FmsnA2/export.log)
- [形状、校验值及数值结果（可移植交付副本）](../../chuanliantui_multimode/model3023-h7-20261015/model_3023_opset13_export.json)
- [板端核对参考向量](../../chuanliantui_multimode/model3023-h7-20261015/model_3023_opset13_reference_vectors.npz)

本次未运行训练、MuJoCo 行为回放、Cube.AI 导入、板端数值／时序或真机验证；Isaac 回放不适用。历史仿真证据仅作为源模型已有记录，不表述为本次重跑。H7 的 39 维观测、50 Hz 调度及数值对齐仍由部署负责人待确认；此次导出不代表旧 H7 固件可以直接使用。
