# 串联腿混合初态 r10

2026-10-10，用户要求将所选初态配置改为参考 v7。分支 `26_wheelleg`，HEAD `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交改动；独立复查待执行。接口 `chuanliantui-mjlab-wheelleg-recover-r10`，阶段 `recover_mixed_reset_r10`。

## 本次配置

- 网络默认仍随机初始化；未提供 resume 或初始化来源时不加载历史策略。
- 每次复位独立以 0.5 概率选择普通／地面初态，并非每批强制各一半。
- 地面 root 高度 0.25 m，相对环境原点；机体姿态为单位四元数，无额外 root 倾斜、yaw 或高度扰动。地面双腿仍独立从合法闭链姿态库采样，关节及轮速度仍随机。
- 普通初态沿用姿态库的碰撞净空高度，最低 0.38 m；roll/pitch 各 ±0.15 rad，yaw ±π。不是固定站稳姿态。

仅将 r9 的 ground_start_probability 1→0.5，ground_randomization_probability 0.5→0，并将扰动幅度和偏移范围置零。零速命令、0.27～0.34 m 高度命令、3～6 cm 高度门控、关闭膝惩罚、动态模式、四腿两轮顺序、PPO、机器人和气弹簧沿用 [r9](chuanliantui-mjlab-wheelleg-recover-r9.md)。本次没有完整改成 v7 奖励、速度命令或训练阶段流程。

## 模型兼容

Actor35/Critic78/Action6、输入输出语义和实体控制保持 r9。新 manifest 和实验合同标记初态配置变化。r9 历史 manifest 保存为 `reference_training/r9_interface.json`；历史 r6～r9 的原生回放和 CPU 评估读取各自 checkpoint 保存的初态参数，不随当前默认分布变化。完整 resume 到 r10 仍须显式迁移，不自动使用旧权重。

本次没有训练、模型导出或 H7 代码改动。有限检查和 20 环境可视预检见 [诊断记录](../diagnostics/20261010-mjlab-r10-mixed-reset.md)。尚无 r10 已训练模型，也未完成策略行为或真机验收。
