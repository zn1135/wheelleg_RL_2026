# 复旦8000轮v2 ONNX

[policy.onnx](policy.onnx)：obs float32[1,25]、obs_history float32[1,125] → actions float32[1,6]，固定batch1、opset17。

保留源v2历史（4净1噪初填、历史先于当前帧裁剪），图含Encoder和Actor，不含积分、PD或图外历史维护。动作顺序lf0/lf00/lfwheel/rf0/rf00/rfwheel。

图检查通过；导出器1组随机输入的ONNX ReferenceEvaluator/CPU PyTorch最大动作差9.5367431640625e-07，阈值1e-5。仅数值检查，无新仿真/训练/板端或真机验证；H7负责人核对待确认。

[接口元数据](interface.json) · [导出来源和检查](export-report.json) · [完整源码指纹](source-sha256.json)

[完整Markdown使用说明](model8000-onnx.md)包含25维槽位、v2历史、动作顺序与数值检查范围。
