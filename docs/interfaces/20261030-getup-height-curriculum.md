# 接口记录：串联腿getup恢复高度课程

2026-10-30后续：新增[四腿动作50%平滑](20261030-getup-leg-action-smoothing.md)，每100Hz拍混合当前raw与上一拍已平滑动作，等待期/复位清缓存，轮子及raw观测不变。新增图外滤波契约；CPU/H7尚未适配，CPU明确拒载。本次未训练，冻结旧1999的4初态试验丢失原2个成功，不能直接给旧策略套用；新配方重训效果未测。[检查与录像](../../artifacts/getup25-leg-smoothing-20261030-143250/README.md)。

2026-10-30后续：用户要求加大关节力矩惩罚，先将独立getup的`torques`权重−1e−4→−2e−4，未饱和时加倍，加权后仍限到[−1,0]再乘dt；原六主动电机力矩平方和及“曾站稳”门控保持，因此首次站稳前该项仍为0。共享公式只对role=getup按实际权重补限幅，旧stand_v2/walk不改；其余18项奖励及观测/课程/终止/物理不变。getup reward recipe升为`split-getup-torque2-full-strength-nominal-state-v3`，严格拒载此前配方（包括刚完成的1999），无隐式迁移，历史产物保留。未启动训练，旧模型动作不会因改奖励而变化。[检查与范围](../../artifacts/getup25-torque2-20261030-140050/README.md)。

2026-10-30后续：按用户标注，将getup起立前倒置/腾空终止对齐Isaac：首次任一轮触地后，重力投影z严格>0立即失败；双轮连续无支撑≥0.2秒失败，不要求曾站稳。任一轮恢复>1N支撑即清零腾空计时，首次触地前两项均不触发。仅修改split getup；旧stand_v2和walk保持。新增`termination_contract=split-getup-post-contact-inverted-airborne200ms-v1`，拒载缺少该终止配方的旧getup checkpoint，避免续训时静默改变规则；历史产物保留。其余成功姿态/承重要求及40秒复位计时不变，未启动训练。[本次检查](../../artifacts/getup25-isaac-termination-20261030-124700/README.md)。

2026-10-30后续：按用户要求，仅将getup成功高度门槛对齐Isaac原站立任务：解锁前≥0.28m、解锁后≥0.20m。目标高度仍0.30→0.22m，窗口4096/成功率30%不变；姿态≤15°、双轮承重、机身离地连续0.5秒、全强度对称惩罚和25/125/71/6均保持，不宣称完整照搬Isaac所有条件。walk仍使用原配置。checkpoint已通过command_config严格保存/核对两个success_height，旧0.20/0.20配方拒载，不隐式迁移。未启动训练。[本次检查](../../artifacts/getup25-success-height-20261030-122904/README.md)。

2026-10-30后续按用户要求，仅将getup的左右腿对称惩罚nominal_state恢复为原1999的全程100%强度，权重仍−3、先限幅再加权/dt缩放保持；不再乘首次站稳前0.1→1的系数。高度课程、25/125/71/6、训练评分网络输入和其余18项奖励不变；独立stand_v2保留原渐进缩放，walk不变。getup profile仍v2，reward recipe改为`split-getup-full-strength-nominal-state-v2`，严格拒载之前缩放配方（包括本轮2000轮model_1999），不隐式迁移；旧模型/logs/快照保留。此次未启动训练，行为效果未验证。[检查证据](../../artifacts/getup25-full-nominal-20261030-120612/README.md)。

| 项目 | 内容 |
| --- | --- |
| 日期、作者、复查人 | 2026-10-30；Codex；H7负责人待确认 |
| 机器人、任务 | 当前串联腿闭链模型；`chuanliantui_getup` |
| 旧接口 | [双策略起立v1](20261030-chuanliantui-two-policy.md)，全程0.22m |
| 新profile | `chuanliantui-getup-physical25-v2` |
| 新课程标识 | `split-getup-height30to22-recovered30pct-window4096-v1` |
| 训练仓分支、SHA | `26_wheelleg`；`518360847bc804de5396edd75abc565174b1d96e`，含既有未提交改动；实际训练源码以新run的snapshot及启动哈希为准 |
| 部署版本、复查 | 未改H7、未交付模型；负责人核对待确认 |
| 实现 | `chuanliantui_split/env_cfg.py::command_cfg`、`mdp.py::SplitCommand`、`runner.py::contracts/validate` |

## 本次差异

只恢复起立高度课程：初始`unlocked=False`，目标0.30m；沿原`StandupCommand.update_curriculum`累计4096个完成回合，成功比例达到30%后全体切为0.22m，未达标重新累计下一窗口。当前成功高度按解锁前≥0.28m、解锁后≥0.20m，倾角≤15°、双轮承重及机身离地连续0.5秒判定，单个成功不提前结束回合。全局解锁后不重新锁回，局部/物理复位和checkpoint续载都保留阶段及累计计数。

Actor25/五帧历史125/Critic71/动作6，字段顺序、单位、缩放、FIFO更新、原始动作和实体PD、XML、物理500Hz/策略100Hz均不变；仍不输入模式或阶段。高度课程首次接入时保留了19项奖励及首次站稳前10%→100%的左右腿对称惩罚；当前已按上述后续要求恢复该项全程100%，其余18项保持。行走随机高度和17项奖励不变。

checkpoint新增课程标识，严格核对0.30/0.22m、4096窗口及30%门槛；允许保存和恢复未解锁/已解锁阶段。旧起立v1固定0.22m模型拒载，不隐式续训；旧模型与训练快照保留，可用其对应源码回看。用户授权本轮从头2000轮，2048环境/seed42，预算0～1999，独立新run，不追加。

## 兼容性与同步

mjlab单模型play恢复模型保存的课程目标；未解锁模型以0.30m运行，已解锁以0.22m运行。ONNX仍为`observation_history[1,125] → actions[1,6], velocity[1,3]`，目标高度由图外命令输入。独立CPU双策略入口固定起立0.22m，因此明确拒绝未解锁的getup checkpoint；已解锁模型可沿原交接规则读取。本次不把课程训练成绩当作独立CPU或真机通过。

| 范围 | 状态 |
| --- | --- |
| 训练、单模型play、保存/续载及ONNX接口 | 随本次配方检查；实际结果见下方产物 |
| 独立CPU双策略 | 保持0.22m；增加未解锁模型拒载检查 |
| H7模型/固件 | 未修改、未交付；负责人适配及核对待确认 |

## 验证与模型记录

本轮独立正式入口为`scripts/agent/check_getup_height_curriculum.py`，复用`check_chuanliantui_two_policy.py`的起立检查：新增未解锁0.30m有限推演、4095/4096与1228/1229成功门槛、局部复位、未解锁保存/恢复、旧v1与错误门槛拒载；本轮起立20环境先64步未解锁、再256步有限推演、两个优化器精确恢复、ONNX及两阶段各32步有限play通过。实际旧getup v1拒载、CPU未解锁拒载/已解锁ONNX加载通过。行走正在其他任务并行修改，本轮不把行走失败当作起立失败、不声称完成当前行走或完整CPU推演；这些是接口和有限检查，不代表行为改善。

[本次检查与运行产物](../../artifacts/getup25-height-curriculum-20261030-023926/README.md)。训练完成核对用`check_getup_height_training.py --training <本次产物> --output <新JSON>`，`--startup`只检查首模型，不把启动核对称作完成。新训练模型未完成/未交付时，模型交付表不填虚构结果；本轮log_dir、命令、源码哈希、实际退出码和完成检查分别由launch、source-sha256、exit及completion-check记录。
