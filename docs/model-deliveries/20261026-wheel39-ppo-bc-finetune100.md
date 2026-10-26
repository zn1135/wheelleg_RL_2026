# 模型交付记录：轮角39 BC初始化PPO微调100轮

2026-10-26，Codex执行，用户接收；独立成员复查与H7负责人待确认。**仅供本地仿真诊断，停车未通过，尚未超过LQR，不替换默认稳定BC，不用于真机。**

## 任务与来源

当前共享XML串联腿，任务`chuanliantui`，正常趴地起立、零速平衡和平滑／停车；不训练移动和倒置恢复。actor39／critic82／动作6，接口`chuanliantui-huananhu-wheel-angle-39-82-handover-v2`，新训练模式见[配置记录](../interfaces/20261026-wheel39-bc-ppo-finetune.md)。

训练仓当前detached HEAD `2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`，含大量已有未提交／未跟踪实现；不是该HEAD单独可重现的模型。实际运行60个生产源码及逐文件哈希见[训练源码记录](../../artifacts/ppo-bc-finetune100-20261026-v1/training-source-manifest.json)，对应运行目录`snapshot/ct_mjlab`；实际env与PPO配置见运行目录`params`。没有Git提交、PR、推送或H7改动。

本地RTX4060 Laptop 8GB，mjlab虚拟环境；seed42、2048环境、100轮、每轮24步与5×4个小批次。前20轮仅价值网络，动作噪声腿0.001／轮0.005，固定学习率、原BC软约束、冻结SMP先验、名义物理、无观测噪声／传感延迟、腿轮动作2或4ms。实际时间北京时间13:41:46～13:46:27。

BC初始化SHA `2ac61f711e17e9e37347fefeb4a72149b17f02ee9219ebad18225bc919baaa89`；只复制actor，critic与Adam新建。SMP SHA `1bff6d8bd8f9200090fd14b1883cbcb3b3c5fab809c020e0f80dcabb6b90eda4`；XML SHA `cac8061de2850602bfd1421fc856bf217c4e5040a11c8f5d33cf1abe61d5f0cf`。BC／SMP额外更新0，XML不改。

## 产物与验证

完整checkpoint：`logs/mjlab/chuanliantui_huananhu_multimode_smp_wheel_handover_v4_bc_finetune_v1/2026-10-26_13-41-45_wheel39-bcfinetune-smp-100/model_99.pt`。

SHA256：`226cccc53ecd836adda711b147980008ae10a2f3485d897e8347d9d08e0c6fb4`。含actor／critic、Adam、冻结BC参考和训练计数，不含encoder；没有ONNX、板端网络或固件产物。

| 阶段 | 状态与证据 |
|---|---|
| 训练预算 | 通过：100轮，价值更新2000、动作721，见报告`budget-audit.json` |
| 原40组20秒起立／平衡 | 通过：严格起立、最终v3.1平衡、腿部平滑均40/40 |
| 停车 | 失败：0/40；末5秒均速0.01098468m/s，XY偏离均值0.05125570m |
| 原生LQR对照 | LQR相同10姿态起立／平衡10/10、停车4/10，速度0.00590787m/s、全频腿速RMS0.00560840rad/s；PPO未超过 |
| 视频 | 完整20秒固定case0三画面对照，原始轨迹／视频哈希均保存 |
| 续训状态恢复 | 非空Adam、actor／critic、BC参考及计数核对通过；并非逐物理状态无缝续跑 |
| Isaac | 不适用：本轮为mjlab |
| 移动、10ms压力、CPU sim2sim、ONNX、板端、真机 | 未执行，不在本轮范围 |

完整执行脚本、命令、失败检查记录及逐物理步独立复算见[报告](../../artifacts/ppo-bc-finetune100-20261026-v1/README.md)。原40组已用于模型选择，不能当新泛化；原生LQR1ms／0秒门控与学生2ms／0.2秒门控差异已明确记录。

本模型保留为100轮对照；用户允许自主配置，下一段新增300轮、累计400轮已启动，尚未交付其结果。当前整体目标仍未达到；稳定BC和旧PPO均保留供对照与回退，H7负责人待确认。
