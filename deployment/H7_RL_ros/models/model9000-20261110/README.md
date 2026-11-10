# 复旦9000轮导出包

独立导出，未替换active networkzn1或烧录；板上仍是14000轮LQR→RL版。

[交付说明](../../../../docs/model-deliveries/20261110-fudan-model9000-onnx.md)。obs[1,25]＋obs_history[1,125]→actions[1,6]，float32、opset17、history v2。完整PT保留两优化器；图仅含encoder＋actor。

来源9000训练于2026-11-04，早于正式采纳双腿Dahl拟合物理；检查点没有该拟合profile身份，不把它说成新物理重新训练的9000。板端接管和运动表现未验证；当前14000切RL无出力问题未定位，换模型不代表修复。
