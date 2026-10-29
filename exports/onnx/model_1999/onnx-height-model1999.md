# 高度门控model1999：35维ONNX使用说明

只适用于`artifacts/height-response-a2500-20261012/models/model_1999.pt`，模型目录遵守上级README分类约定。25维通用说明及当前39维导出器不适用本检查点。

## 导出命令

在仓库根目录使用导出专用环境运行，output必须不存在，避免覆盖历史产物：

```powershell
& mjlab_training/.venv/Scripts/python.exe scripts/agent/export_h7_height1999.py --checkpoint artifacts/height-response-a2500-20261012/models/model_1999.pt --output exports/onnx/model_1999-new
```

入口固定校验指定PT哈希，读取实际actor配置，严格加载全部MLP权重；旧mjlab配置类仅惰性读取元数据，不构造环境或运行回调。确定性actor为256/128/64 ELU，无归一化、历史或估速，策略标准差不进入图。

## 网络契约

float32/batch1/opset13，输入observation[1,35]，输出action[1,6]，动作按LF前/LF后/RF前/RF后/左轮/右轮排列。

| obs下标 | 定义 |
|---|---|
| 0～3 | vx、0、yaw速度、目标机高×5 |
| 4～6 | 角速度先裁剪±100再×0.5 |
| 7～9 | 机体系单位重力 |
| 10～15 | 四髋原生角减名义角、两个零轮位置；裁剪±100 |
| 16～21 | 四腿两轮速度先裁剪±200再×0.1 |
| 22～27 | 四腿两轮上一原始action，裁剪±100 |
| 28～34 | 实际训练dynamic_modes=false，固定[1,0,0,0,0,0,0] |

预处理、PD和电机映射在图外。实际配置为50Hz策略/500Hz内环，腿位置缩放0.5、PD60/2、轮速缩放10、轮P0.1、目标轮速上限100rad/s；完整冻结配置见delivery.json。

本目录保存ONNX、delivery.json、reference_vectors.npz、参考输入/动作CSV、export.log、source-snapshot.zip、workspace.patch及SHA256SUMS。使用说明与[实际导出交付记录](20261030-model1999-onnx-export.md)同目录保存。
