# 模型交付记录：2026-11-03复旦8000轮v2 ONNX

## 任务与接口

| 项目 | 内容 |
|---|---|
| 交付/接收/复查 | Codex按用户请求生成本地导出；接收人为用户；独立复查/H7负责人待确认 |
| 机器人/任务 | 本机闭链轮腿，fudan_ppo_walk，保持当前共享XML与实体电机接口 |
| 用途 | 冻结8000轮行走模型导出；仅检查图和数值，不代表停车已修复或部署通过 |
| 合同 | fudan-ppo-walk-physical25-history125-critic144-v1；reset/history v2，[接口说明](../interfaces/20261101-fudan-ppo-walk.md) |
| PR | 无，本轮未提交/推送，未修改H7 |

## 代码与训练来源

训练仓https://github.com/zn1135/wheelleg_RL_2026.git，分支26_wheelleg，HEAD `7e8903308160b9d3a9505838d45c7eac322f96d3`，有未提交与未跟踪改动；HEAD不能代表本次运行内容。实际导出使用整理后的唯一生产网络/模型加载实现。

[导出源码完整快照](../../exports/fudan_ppo_walk/20261103_model8000_v2/source/)、[逐文件SHA](../../exports/fudan_ppo_walk/20261103_model8000_v2/source-sha256.json)、[已跟踪差异快照](../../exports/fudan_ppo_walk/20261103_model8000_v2/working-tree.patch)覆盖当前导出源码，包含参与运行的未跟踪文件；原训练来源以原run的source、source_sha256.json和training_report.json为准，未修改历史内容。

原训练为4096环境、seed1，7000→8000新增1000轮，恢复两Adam与RNG、物理回合重新初始化，显式保留v2历史。原训练命令/配置/审计见[完成记录](../../artifacts/fudan-parking-resume7000-to8000-20261102/COMPLETED.md)，日志在`logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000/`。本轮无新训练。

Python运行于mjlab_training虚拟环境；PyTorch2.14.0、ONNX1.22.0、MuJoCo3.11.0。导出及数值检查仅CPU，无物理积分；XML路径/实际SHA、源快照/差异SHA详见[export-report.json](../../exports/fudan_ppo_walk/20261103_model8000_v2/export-report.json)。部署仓未修改，负责人核对待确认。

## 模型与导出产物

| 产物 | 路径 | SHA256 |
|---|---|---|
| 源PT | [model_final.pt](../../logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000/model_final.pt) | 39fa104d176f82a33db8317a75b4db28749362daf6bc4d2e06cb83d2c5ae32c9 |
| ONNX | [policy.onnx](../../exports/fudan_ppo_walk/20261103_model8000_v2/policy.onnx) | ceccf25e06f0f2598c498c35ad58942423ae344050bb888d7f5009104c0e5fcb |
| 接口/来源/校验 | [导出目录](../../exports/fudan_ppo_walk/20261103_model8000_v2/) | 各文件以目录内SHA256SUMS为准 |
| 板端网络/固件 | 未生成 | 不适用 |

实际命令：

```bash
scripts/mjlab.sh fudan-ppo-walk export --checkpoint logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000/model_final.pt --output exports/fudan_ppo_walk/20261103_model8000_v2
```

固定batch1/opset17：obs float32[1,25]、obs_history float32[1,125] → actions float32[1,6]。图内含原Encoder和Actor，输出原动作，顺序lf0/lf00/lfwheel/rf0/rf00/rfwheel；无积分、Critic、PD、历史FIFO或新估速输入。v2历史预处理以interface.json为准，策略10ms/PD2ms。

## 分阶段验证与结论

| 阶段 | 状态 | 证据 |
|---|---|---|
| 原训练完成 | 历史已完成8000 | 原run报告，未重复训练 |
| ONNX结构/数值 | 通过 | 导出退出0，onnx.checker通过；导出器1组随机输入，ONNX ReferenceEvaluator相对CPU PyTorch最大动作差9.5367431640625e-7，阈值1e-5 |
| 模型完整性 | 通过 | 导出后PT实际SHA仍为上表值 |
| 物理回放/GPU行为 | 本轮未执行 | 原8000回放及积分r2记录保持，不作新验收结论 |
| 板端数值/最坏耗时/真机 | 未执行 | H7未适配/未部署，负责人待确认 |

允许用作离线模型接口/推理验证；不视为板端或真机通过。积分r2“真值9/9、估速0/9”结论不变，不将ONNX导出当成停车修复。回退使用原8000 PT及其归档训练源码，不改写历史产物。
