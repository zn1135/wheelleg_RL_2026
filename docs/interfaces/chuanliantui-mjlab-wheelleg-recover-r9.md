# 串联腿关闭新增膝限位惩罚 r9

2026-10-10，用户要求“关闭新增膝限位惩罚，再训练”。26_wheelleg，HEAD58947e8fabc4683b2025d42bda742b06c84dc566加未提交改动；助手编写，独立复查待执行。当前接口`chuanliantui-mjlab-wheelleg-recover-r9`，训练奖励配置`wheelleg-recover-v12-no-knee-limit`。

## 改动

仅把新增knee_limit惩罚权重从-0.5改为0。mjlab RewardManager对0权重项跳过累加，每步及回合奖励贡献均为0；保留原始公式供历史配置评估。XML实体膝关节硬限位、闭链及气弹簧资产保持原样。

其余26项奖励公式／权重、3～6cm速度高度门控、PPO、初态／随机化、PD和物理步长沿用r8。此前提出的姿态奖励门控、起立奖励和初态课程没有实施。

Actor35/Critic78/Action6；动作及观测关节／上一拍请求动作顺序仍为`[lf0,lf00,rf0,rf00,lfwheel,rfwheel]`。动态normal/recover、恢复目标和计时沿用r7。控制／观测合同不变，完整动作和物理控制见 [r8接口](chuanliantui-mjlab-wheelleg-recover-r8.md)。本次版本变化标识训练奖励配置差异。

## 兼容与训练

r5旧direct入口继续保留；r6/r7/r8按历史manifest恢复各自动作顺序、mode及knee_limit权重-0.5回放／CPU评估。历史manifest文件为reference_training/r6_interface.json、r7_interface.json及r8_interface.json。完整resume到r9被拒绝；本次使用随机权重新训练，不覆盖历史logs。

r9 checkpoint、environment.json及导出JSON记录knee_limit_weight=0和完整奖励权重。ONNX维度与输出顺序沿用r8；1000轮最终模型行为评估未通过，见 [r9效果评估](../diagnostics/20261010-mjlab-r9-model999-evaluation.md)；H7适配与真机验收尚未执行。

有限预检见 [诊断](../diagnostics/20261010-mjlab-r9-no-knee-limit.md)。用户授权的4096环境／1000轮训练状态见 [训练记录](../diagnostics/20261010-mjlab-r9-formal-training.md)。未提交Git，TensorBoard默认仅交付命令。
