# 复旦最终验证统一GPU（2026-11-02）

后续用户已确认扩展为所有task统一GPU验收，且复旦运动门槛改为通用0.10/0.10/0.05/5°，以[统一GPU验收](20261102-unified-gpu-acceptance.md)为准。本页“v2评分保持”是前一步变更的历史描述，新验收用accept，原score_v2只保留历史复查。

用户要求“最终验证sim2sim用gpu，cpu删除”。本条只适用于`fudan_ppo_walk`：最终验证使用mjlab / MuJoCo Warp GPU，CPU物理评估及活动诊断入口删除；其他任务的`sim2sim/`、共享XML、PD、训练配方和H7保持。

GPU与原生CPU在同状态、同输入下生成的轮地接触点不同，证据见[单拍对照](../../artifacts/fudan-single-step-7000-20261102/report-final/README.md)。选择GPU是统一后续验证后端，不是证明CPU更不准确，也不把GPU结果当作CPU跨引擎迁移或真机通过。

## 当前入口

```bash
# 默认GPU；显式写 evaluate gpu 也可。截止须来自本次授权预算。
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.evaluate \
  --checkpoint <model.pt> --output <新GPU目录> --deadline <本次截止Unix秒>
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.score_v2 \
  --recording <GPU目录> --output <新v2评分.json>
```

完整矩阵保持47工况×21种子=987样本，仅GPU统计；诊断用`--diagnostic`，高度0.30m子集再加`--diagnostic-height .30`，仍只算15例诊断。模型reset/history v2/v3选择、随机物理、命令、原失败分母、v2评分和正常速度完整视频要求保持。最终通过仍待用户视频确认。模型与真实部署接口未改，H7负责人核对状态仍待确认。

`evaluate cpu`及CPU专属`--fixtures/--workers/--integration-substeps`参数拒绝。`protocol.json`与`execution.json`记录`validation_engine=gpu`、`validation_backend=mujoco_warp_gpu`。20环境预检从GPU几何位置检查碰撞网格最低点，不再构造CPU仿真或要求CPU/GPU逐点观测相等。

## 删除与依赖

- 删除`ct_mjlab/fudan_ppo_walk/native.py`；原`load_policy`函数原样迁到`policy.py`，GPU评估和ONNX导出共用。
- 删除活动脚本`check_native`、`diagnose_native_integration`、`probe_refresh_timing`、`check_refresh_probe`、`audit_refresh_repeat`、`probe_single_step`。已完成的CPU动态诊断不再作为当前入口。
- 输入/历史/扰动检查去掉CPU物理分支，保留GPU使用的张量、CLI、保存恢复检查。CPU上的离线评分、模型读取、ONNX数值校验和历史结果读取不属于CPU物理仿真，不删除。
- 旧CPU轨迹、视频、日志、模型和报告原位保留；清理前的整套复旦源码及相关文档另存于[清理记录](../../artifacts/fudan-gpu-only-validation-20261102/README.md)。历史报告不改标为GPU结果。

本次只改入口、依赖和文档，未新训练、回放或导出。当前检查与未验证项见清理记录。
