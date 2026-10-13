# 恢复退化的单项对照训练接口

2026-10-12用户要求启动多组训练、逐项分析，并选择每组追加500轮。全部从已评估高度门控组2000轮独立续训，4096环境、seed42、完整恢复actor/critic/Adam和1e-5学习率；各组最终累计2500轮。保留默认V14与全部历史内核，不改机器人或H7控制代码。

## 分组与比较关系

接口为`chuanliantui-mjlab-recovery-factors-v1:<case>`，每组独立标识与checkpoint，禁止隐式跨组加载。

| case | 配置 | 用途 |
|---|---|---|
| `baseline` | 原2000轮`huananhu-v14-height`继续训练 | 检查沿用旧目标是否也会丢失双轮平衡 |
| `recovery_reference` | 原recovery课程，从扶正重新开始 | 与其他四组采用完全相同500轮预算的参照 |
| `skip_righting` | 只将课程初始阶段设为双轮平衡 | 检查重学只扶正是否削弱已有能力 |
| `keep_height` | 只在前两阶段保留原V14两项高度奖励 | 检查移除高度目标的影响 |
| `keep_contact` | 只保留扶正阶段的非期望接触惩罚 | 检查允许长期撑地的影响 |
| `bearing_reward` | 只改变平衡/高度阶段的轮支撑奖励公式 | 检查5N就饱和的奖励是否偏弱 |

四个实验组各自与`recovery_reference`比较；它们不依次继承前一组模型，也不把多个修正混入同组。baseline与recovery_reference之间存在多项历史目标差异，不能称为单变量比较。已有3000轮结果只作较长训练的参照，不能直接与500轮组比较收益大小。

## 三项奖励修正

- `keep_height`：前两阶段高度指数恢复`exp(-clamp(error,-0.15,0.15)^2/0.001)`、高度平方恢复`(10*error)^2`，沿用原权重+1/−1；最终阶段保留原recovery的门控与公式。
- `keep_contact`：只在扶正阶段恢复原接触判定，即undesired_contact的力历史模长最大值>3 N，原权重−2；其他阶段不变。
- `bearing_reward`：`min(clamp(每轮向上对地支撑/(0.5*实际整机重量),0,1))*upright`，还须两轮确实接触地面、非轮对地支撑≤整机重量5%。权重仍+2，扶正阶段仍不启用；不改变课程成功判据、终止条件或腿关节目标。

所有组共用两个新增的只读对地接触探针`wheel_ground_probe`与`nonwheel_ground_probe`。原wheel/base/undesired传感器保持原配置，原actor35/critic78、动作顺序、PD、闭链、气弹簧和物理随机化均保留。探针只测量接触，不施力；除bearing_reward之外不参与训练奖励。

## 完整迁移与版本保护

独立模块：[recovery_factors.py](../../mjlab_training/src/ct_mjlab/experiments/recovery_factors.py)。独立入口：[run_chuanliantui_recovery_factors.py](../../scripts/agent/run_chuanliantui_recovery_factors.py)，不修改默认CLI配方。

首次训练只接受已评估`model_1999.pt`，SHA256：

`1c3487a7da8d1fe74c418466ec4068f621c162e36406c14786c972a2839e7027`

训练前核对原接口、实际环境、4096环境与seed42；保存`model_resume_initial.pt`并逐项检查actor、critic、Adam完全相同，未通过则不进入PPO。累计计数从2000恢复，学习率从Adam恢复为1e-5，继续原自适应PPO。每组保存独立接口、实际环境、runner参数及完整源码快照。每100轮保存中间checkpoint，最终保存`model_2499.pt`，其完成轮数是2500。

该独立入口用于本次已授权训练；新checkpoint应使用本接口对应评估脚本，不能隐式当成历史CLI配方、H7可部署策略或原ONNX交付。

## 评估与自动分析

2026-10-12补充[双轮平衡配对验收v2](chuanliantui-wheel-balance-acceptance-v2.md)：原成功样本失效即判退化、验收不通过，新增成功不抵消丢失。E组56/64且丢失0、55为明确退化。逐条件轮端承重、稳定平衡、高度跟踪分开报告；已完成的冻结训练与原报告保持历史内容。

每组完成训练后，自动评估`model_2100.pt`（累计2101、追加101轮）与`model_2499.pt`（累计2500、追加500轮）。用checkpoint中的真实计数，不把前者误称为追加100轮。

评估入口：[eval_chuanliantui_recovery_factors.py](../../scripts/agent/eval_chuanliantui_recovery_factors.py)。各时间点普通/地面各64份、20秒、seed42、零速度、目标0.30 m、确定性actor、名义物理、关闭噪声与延迟、无辅助力、不更新策略。显式还原同一qpos/qvel和场景原点、重置动作与观测历史；地面样本复用退化审计的保存初态，普通初态由源2000模型首次评估保存后复用。

评价按用户要求：轮端真实对地接触，每轮≥10%整机重量、总支撑80%～120%、非轮承重≤5%、机身倾角≤15°，以及原线速度≤0.15 m/s/角速度≤0.5 rad/s；不限制腿关节角。高度达标另外要求目标±3 cm。记录连续0.5秒成功、末5秒≥95%时间成功、丢失/新增原成功样本、轮端与非轮部件受力、数值异常，保存样本0视频。

各组模式观测保持各自训练语义，物理提前失败在独立回放中绕过以提供完整20秒机会；数值异常样本永久排除。各组训练课程继续用自己的原判据，因此评价规则统一不等于暗中替换训练目标。

协调器在每组结束后写`analysis.json`与汇总`analysis.md`，对照同轮数recovery_reference并列出原成功样本的丢失情况。不同配方的训练reward不直接排名；单个seed与64初态只提供本批趋势证据，不能代替长期训练、多seed或部署验收。

相关：[退化审计](../diagnostics/20261012-mjlab-recovery-ground-regression-audit.md)、[本次启动记录](../diagnostics/20261012-mjlab-recovery-factors-training.md)。
