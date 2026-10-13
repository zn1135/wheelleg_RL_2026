# 串联腿恢复膝限位惩罚 r11

2026-10-10，按用户要求“加上限位惩罚”。分支 `26_wheelleg`，HEAD `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交改动。接口 `chuanliantui-mjlab-wheelleg-recover-r11`。

## 奖励变化

恢复 wheelleg_RL 当前 `huananhu_rewards.py` 的 `knee_limit`，权重 **0→-0.5**。

- 关节为实体被动膝 `lf1、rf1`，从资产读取各自硬限位；不是腿电机 `lf00、rf00`。
- `d=min(q-lower,upper-q)`，软区余量 `m=0.15 rad`。
- 原始项 `sum(clamp((m-d)/m,0,1.5)^2)`；两侧求和。
- 距限位≥0.15 rad为0；距限位0.075 rad时单侧为0.25；到止挡时单侧为1；越界单侧最高2.25。
- RewardManager仅乘一次控制步长0.02 s；两膝恰在止挡时，本项贡献每步-0.02。原硬限位和闭链物理参数保持原值。

27项奖励公式与权重均与当前参考一致。PPO、动态normal/recover、四腿两轮动作顺序、50%普通／50%地面初态、零速与0.27～0.34 m高度命令、3～6 cm高度门控和气弹簧沿用 [r10](chuanliantui-mjlab-wheelleg-recover-r10.md)。初态阶段名仍为 `recover_mixed_reset_r10`，奖励版本由新接口及实验合同区分。

## 兼容与验证

Actor35/Critic78/Action6、50 Hz和部署预处理／控制语义保持原值。实验合同改为 `chuanliantui-huananhu-flat-35-78-ppo-sourcexml-spring90mm-velocity-reciprocal-knee-limit-mixed-reset-r11`。

新增保存历史 `reference_training/r10_interface.json`；r6～r10回放／CPU评估仍读取checkpoint的原膝权重与初态参数。旧模型直接完整resume到r11会拒绝，跨奖励版本需要显式迁移。没有自动加载r10或参考工程权重。

有限检查见 [诊断](../diagnostics/20261010-mjlab-r11-knee-limit.md)。配置修改后用户授权训练，已在远端4090启动4096环境／1000轮、随机权重训练，见 [运行记录](../diagnostics/20261010-mjlab-r11-cloud-training.md)；尚未完成新策略行为评估，不能据配置检查声称起立效果改善。H7代码及模型交付待新策略训练、评估后另行核对。
