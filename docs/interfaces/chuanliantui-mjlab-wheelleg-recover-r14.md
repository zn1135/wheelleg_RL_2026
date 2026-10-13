# 串联腿平地速度与航向命令 r14

2026-10-10，用户要求加入华南虎 Flat-v0 的前进／偏航速度命令。分支 `26_wheelleg`，HEAD `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交改动。旧接口见 [r13](chuanliantui-mjlab-wheelleg-recover-r13.md)，新接口 `chuanliantui-mjlab-wheelleg-recover-r14`。作者：Codex；独立复查：待进行。

## 命令定义

- 当前默认训练关闭 `zero_velocity_commands`：前进速度均匀采样 −2.7～2.7 m/s，偏航速度均匀采样 −2π～2π rad/s，侧向速度为0。
- 每5～15 s重采样。50%的环境采样航向目标 −π～π；偏航命令每控制步由 `5 * wrap_to_pi(target_heading - current_heading)` 求得，再限幅至 ±2π。
- 沿用原普通采样中的静止环境抽样，两次10%抽样合计约19%；静止环境的速度命令为0。
- 地面初态前1.5 s沿用源代码过渡命令：前进／偏航各 −1～1；之后使用普通命令。恢复标志仍按原状态机切换，不把恢复环境强制改成零速度。
- 本次仅加入普通速度／航向命令。2000／3000／4000轮后的冲刺、自旋模式关闭，避免其范围覆盖本次指定范围。

权威实现：`huananhu_mdp.HuananhuCommandCfg`／`HuananhuCommand`、`huananhu_env_cfg.make_huananhu_env_cfg`。来源为本机华南虎 `wheelbipe_V14/env_cfg.py` 的 V14 Flat-v0 普通命令及原移植的命令实现。

## 接口与兼容

Actor35／Critic78／Action6、命令观测缩放、动态normal/recover、四腿两轮顺序、50 Hz、50%普通／50%地面初态均沿用r13。高度命令仍0.27～0.34 m，速度奖励高度门控仍5～10 cm，膝惩罚权重0。机器人／气弹簧／PD、其余奖励和PPO超参数未改变；实验名与合同升为r14。

命令观测仍为前进／侧向／偏航／高度，但前进与偏航槽现在有非零输入；此前零速度训练的模型未验证这些输入下的行为。旧r6～r13按各自manifest恢复原命令；新命令配置与模型保存到新版本，普通续训拒绝隐式跨版本。历史r11→r13的 `--resume-wide-height-gate` 不能用来迁入r14。用户尚未指定本次训练起点或要求启动训练。

H7源码未修改，仍不是当前35维接口的部署实现。新策略的命令范围、航向误差计算位置与板端适配需双方核对；模型交付／板端及真机验证待进行。本次没有产生新训练模型。

## 有限检查

8192份命令采样验证范围、航向比例、角差环绕／限幅、部分环境更新、地面过渡、后期不启用特殊模式；r11和r13模型均成功加载并恢复零速度命令。见 [检查JSON](../diagnostics/20261010-mjlab-r14-velocity-commands-check.json)。GPU有限32步接口检查另见 [诊断](../diagnostics/20261010-mjlab-r14-velocity-commands.md)。这些检查没有PPO学习，不证明行走／恢复策略效果。
