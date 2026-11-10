# 复旦9000轮：独立ONNX导出

2026-11-10按用户要求另导出此前续训起点9000轮模型。仅新增模型包，不替换当前14000的networkzn1、固件或板端权重；没有新增训练、烧录或运动验收。当前14000切RL后无出力的问题尚未定位，本次导出不表示该问题已修复。

| 项目 | 实际值 |
|---|---|
| 来源 | logs/mjlab/fudan_ppo_walk/20261104_resume8000_to9000_2ms/model_final.pt |
| 轮数/累计转移 | 9000 / 1253376000 |
| PT SHA256 | 4ff0c038f26b7aa9d9d9919c97e59cd4e8124f1ddee5eaec1d2944436f8a37a3 |
| ONNX | exports/fudan_ppo_walk/20261110_model9000_v2/policy.onnx |
| ONNX SHA256 | 727b2abbf4756eafeb72652192fc089329ed453f515c5d7fc920897704f1e201 |
| 同步模型包 | deployment/H7_RL_ros/models/model9000-20261110/，完整PT、ONNX、256组参考向量、接口、检查JSON及SHA256SUMS |
| 合同 | fudan-ppo-walk-physical25-history125-critic144-v1 |
| 历史 | fudan-physical-pd-centred-warmup-clean4-noisy1-unclipped-history-v2 |
| 输入/输出 | float32、batch1；obs[1,25]＋obs_history[1,125]→actions[1,6]，opset17 |
| 时序/扰动 | 策略10ms、PD2ms；源模型训练动作/传感各2ms、主动推扰关闭；图外维护，不向真机额外添加等待 |
| 工具 | PyTorch2.14.0+cu130、ONNX1.22.0；本仓mjlab虚拟环境 |

该9000训练于2026-11-04，早于正式采用镜像Dahl负载，检查点未记录该拟合profile身份；不能把9000称为新拟合物理重新训练的模型。此后的9000→11000才迁入双腿拟合默认物理。左腿独立辨识、步长、跨姿态及独立实录等未完成项不因导出而改变。

图内包括编码器和actor，latent保持训练原尺度；图不含传感预处理、历史队列、图外动作±100、PD或保护。H7仍使用14000和[当前LQR切换接口](../interfaces/20261110-h7ros-lqr-to-fudan.md)；若以后改烧9000，必须重新生成并校验C网络，不能只改文件名或直接加载PT。

实际导出命令，退出0：

```bash
scripts/mjlab.sh fudan-ppo-walk export \
  --checkpoint logs/mjlab/fudan_ppo_walk/20261104_resume8000_to9000_2ms/model_final.pt \
  --output exports/fudan_ppo_walk/20261110_model9000_v2
```

图结构检查通过；原导出单组误差1.90734863e-6。固定seed9000的256组[-1,1]输入，PyTorch/ONNX均有限，最大绝对差1.04904175e-5，小于预先使用的1e-4容差。参考NPZ和检查JSON随包；完整日志保留于`artifacts/fudan_ppo_walk/20261110-model9000-export-01/`。这是软件数值验证，不是9000在当前拟合物理或真机上的行为通过。

来源HEAD c3c8fdffe0739aed314fb71c69f2a2d3135b3fb9、分支26_wheelleg及脏工作区；导出器/模型加载代码仍沿当前主线。完整PT与原训练日志保留，未覆盖旧产物；H7负责人独立核对待确认。
