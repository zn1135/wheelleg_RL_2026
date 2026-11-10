# 复旦14000 H7_ros候选

截至此次提交，板上为14000轮LQR→RL切换版HEX72377777…；用户已反馈切RL后无出力，尚未读取实际状态/定位根因或修复。另导出的9000未替换板上模型，不能把导出、编译或烧录成功当作实际接管通过。

最新状态：2026-11-10 18:40:08（北京时间），用户本次确认电机动力断开后，已实际写入H7_ros的14000轮固件，Programming Finished / Verified OK / Resetting Target齐备，OpenOCD退出0。未使能电机、未执行冷态采集/板端数值/最坏周期/真机运动验证；GPU11/987未达标结论保持。 [烧录证据](../../../../artifacts/fudan_ppo_walk/20261110-model14000-h7ros-flash-01/README.md)。下文为导出阶段历史记录。

用户本次最终选择：暂不烧录，仅保留导出、源码适配和固件；未连接/暂停/复位目标板。

已导出、接入源码并编译；未烧录/使能，GPU完整运动11/987未达标。

[完整交付](../../../../docs/model-deliveries/20261110-fudan-model14000-h7ros.md)、[接口](../../../../docs/interfaces/20261110-h7ros-fudan-model14000.md)。

policy.onnx含编码器＋actor，obs[1,25]、obs_history[1,125]→actions[1,6]；硬件适配状态以delivery-manifest.json为准，interface.json保留原导出器的待H7核对标记。训练负载JSON仅作来源记录，不能叠加为实机补偿矩。
