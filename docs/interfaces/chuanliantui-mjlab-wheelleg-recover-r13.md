# 串联腿宽高度门控 r13

2026-10-10，用户要求“改为宽高度门控”。分支 `26_wheelleg`，HEAD `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交改动。当前接口 `chuanliantui-mjlab-wheelleg-recover-r13`，前版见 [r12](chuanliantui-mjlab-wheelleg-recover-r12.md)。

## 本次改变

四项速度跟踪奖励的高度门控从3～6 cm放宽为参考基础阶段的5～10 cm。令 `e = abs(height - command_height)`：

```text
gate = clamp((0.10 - e) / (0.10 - 0.05), 0, 1)
```

- 误差≤5 cm：全额；7.5 cm：半额；≥10 cm：零。
- 生效项：`track_lin_vel_xy`、`track_lin_vel_xy_square`、`track_ang_vel_z`、`track_ang_vel_z_square`。
- 正奖励和对应负权重平方项均乘该门控，继续遵循参考公式的同一门控规则。
- `track_height_exp_tight`／`track_height_square`公式保持r12；严格起立和normal/recover切换仍要求高度误差≤3 cm。
- `knee_limit`权重保持0。PPO、其余奖励、50%普通／50%地面初态、动态模式、机器人／气弹簧、控制时序保持r12。

Actor35／Critic78／Action6、50 Hz；动作、关节观测及动作历史均为 `[lf0,lf00,rf0,rf00,lfwheel,rfwheel]`。观测／控制合同不变，实验合同增加 `wide-height-gate` 标识，checkpoint记录门控 `[0.05,0.10]`。

## 模型兼容

修改源码前将r12 manifest保存到 `reference_training/r12_interface.json`。r6～r12历史模型回放读取对应门控，r6～r8缺少奖励元数据时使用其原内核3～6 cm门控；r5仍走旧实现。同版本r13保存／加载可用。

当前普通续训拒绝旧版本模型；旧 `--resume-disable-knee-limit` 只属于历史r11→r12迁移，在r13会拒绝，避免将门控变化错误记为单项膝惩罚迁移。用户随后明确要求从1000轮续训，并选择追加1000轮。新增 `--resume-wide-height-gate` 显式入口仅接受完整历史r11：恢复actor／critic／Adam／学习率／计数，同时记录门控3～6→5～10 cm、膝惩罚-0.5→0两项变化。其余manifest／合同必须相符；实际验证与远端运行见 [续训记录](../diagnostics/20261010-mjlab-r13-cloud-training.md)。

有限检查见 [诊断](../diagnostics/20261010-mjlab-r13-wide-height-gate.md)。首次门控修改检查时未启动训练；后续授权续训以运行记录为准，尚无完成的r13模型交付；有限物理和接口检查不能证明策略效果。H7部署代码和真机参数未改，新策略仍需按部署接口约定适配后验收。
