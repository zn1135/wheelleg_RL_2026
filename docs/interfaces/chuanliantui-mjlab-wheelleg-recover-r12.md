# 串联腿关闭knee_limit并从r11续训 r12

2026-10-10，用户要求“关闭knee_limit从1000轮续训2000轮”。分支26_wheelleg，HEAD `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交改动。新接口 `chuanliantui-mjlab-wheelleg-recover-r12`。

## 配置与续训语义

- `knee_limit`权重-0.5→0；公式保留以支持历史回放。原硬限位保持。
- Actor35/Critic78/Action6、四腿两轮顺序、动态normal/recover、初态、PPO、其他26项奖励、控制时序、机器人和气弹簧保持r11。
- 从r11 `model_999.pt`恢复actor、critic、Adam全部moment／step、学习率和训练计数；不是随机重新训练，也不是只加载actor。
- 输入模型完成了1000次PPO更新，保存iter=999、有效累计更新=1000。下一轮索引1000；追加2000次更新，最终索引2999、累计3000轮。
- 仿真状态、回合／延迟历史重新初始化，恢复的是checkpoint已保存的训练状态；不是精确重放原运行的物理轨迹或RNG。

## 显式入口与审计

使用 `--resume <r11-model_999.pt> --resume-disable-knee-limit --iterations 2000`。普通`--resume`仍拒绝跨版本manifest；显式入口仅接受已保存完整r11 manifest、控制／观测／实验合同相符、actor／critic／优化器／计数齐全的模型，且当前语义差异只能是knee_limit权重。其他资产、动作、初态、PPO、奖励或内核变化会拒绝。

历史r11 manifest保存为 `reference_training/r11_interface.json`；r5～r11推理按原配置路由。新实验合同为 `chuanliantui-huananhu-flat-35-78-ppo-sourcexml-spring90mm-velocity-reciprocal-no-knee-limit-mixed-reset-r12`。

迁移保存源路径／SHA256、旧新接口、权重、已完成更新、下一轮索引和恢复标志到每个新checkpoint的`reward_migration_history`。正式新run另保存更新前`resume_state.pt`，用于核对实际远端actor／critic／Adam是否与源checkpoint逐项一致；它不是已完成新PPO更新的model文件。

本机及远端检查见 [诊断](../diagnostics/20261010-mjlab-r12-resume.md)。新日志目录保存所有续训输出；不覆盖r11模型和历史logs。此次只改变训练奖励和恢复入口，H7输入输出及固件保持原值，后续策略效果／模型交付需重新评估。
