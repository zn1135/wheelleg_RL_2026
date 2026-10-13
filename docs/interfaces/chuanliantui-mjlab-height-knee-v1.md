# A2500 高度门控与膝限位两组续训接口

接口：`chuanliantui-mjlab-height-knee-v1:knee_gate`／`knee_only`。2026-10-13用户授权两组各4096环境、seed42、从同一A2500独立追加1000次PPO更新，累计3500后停止；不扩组、不追加。实现为新增 `mjlab_training/src/ct_mjlab/experiments/height_knee.py`，默认V14、历史接口、资产及冻结内核不变。记录见[训练与评估报告](../diagnostics/20261013-mjlab-height-knee-training.md)。

## 来源和唯一改动

源接口 `chuanliantui-mjlab-recovery-factors-v1:baseline`，`model_2499.pt`，累计2500更新，SHA256 `706c8387cba4815b491be3f5914a5951881c2966edabbd5fbb90d0da9c9d3dbd`。迁移先验证SHA256，再调用 `recovery_factors.validate_checkpoint(saved, 'baseline')`，核对环境、seed、4096环境及累计计数；其他源一律拒绝。

| case | 膝项每秒权重 | 四项速度门控 |
|---|---:|---|
| knee_gate | −0.5 | clip((0.06−abs(height_above_ground−cmd_height))/0.03,0,1) |
| knee_only | −0.5 | A原5～10cm门控 |

膝项直接取原 `reference_training/huananhu_rewards.py` 的 `RewardState`：仅lf1／rf1，距较近XML限位0.15rad内开始惩罚，归一化越限量上限1.5再平方、两膝相加。RewardManager统一乘控制dt0.02；不重复乘dt。四项为 `track_lin_vel_xy`、`track_lin_vel_xy_square`、`track_ang_vel_z`、`track_ang_vel_z_square`。从未门控的 `huananhu_v14.RewardState` 出发只乘一次门控，保留原指数速度公式和原高度奖励。

基础配置逐项来自 `recovery_factors.make_env_cfg('baseline', ...)`。高度命令U(0.27,0.34)，原速度命令、50%航向控制、地面初态前1.5s速度覆盖、dash特殊模式、累计3000后15% spin_low桶均保留；本次无高度课程、不强制零速训练。spin_mid门槛4000，本批不达到。15%为原重采样条件桶概率，不代表实际时间占比。

观测actor35／critic78、动作6，顺序四腿电机在前两轮在后。动作缩放、PD／力矩限幅、气弹簧、2ms物理／20ms控制、XML、终止、50%普通／50%地面初态、噪声延迟和随机化完全沿用A。PPO网络、标准差／熵系数等沿用A；完整恢复actor、critic、Adam，学习率从Adam恢复1e-5且保留自适应，不人为重置。

## 完整模型和预算

模型保存接口manifest、模块指纹、环境、观测／控制契约、迁移历史和实际完成更新数；自身加载再次核验。`model_resume_initial.pt`保存累计2500且actor／critic／Adam逐项相同。实际每完成100次更新保存 `model_2599.pt`…`model_3499.pt`，其中 `model_2999.pt`是累计3000。没有重新编号、重置计数或仅载actor的迁移。

训练入口只接受4096环境／seed42／1000次追加更新，Runner也限制从2500开始恰好1000；没有自动失败重训或追加预算。上游零基保存间隔由新Runner对齐到已完成的100倍数，不更改PPO实现。训练中站立比例下降也保持原预算和参数。

## 检查与评估

学习前两组各16／4096环境64步全有限、数值失败0；同真实qpos／qvel逐项隔离奖励，核对2／4.5／7cm门控；完整迁移、自身重载、每组初始模型各复现A普通64／地面60的配对成功集合。任一项失败不学习。

新增评估脚本采用上一批原协议、确定性actor、名义物理／关闭噪声延迟、零速度、20s。每项恢复A同一普通／地面64份qpos／qvel与origins，六高度0.20／0.22／0.25／0.27／0.30／0.34m。末5秒≥95%帧组合成立为成功；高度跟踪包含验收v2稳定平衡和±3cm。斜率分别拟合全样本与同一模型六条命令都平衡成功的固定样本集合；没有共同成功样本时记不适用。膝反力矩读原生 `mjCNSTR_LIMIT_JOINT` 行及JᵀF，不用执行器力矩代替。

评估累计3000、3500，以及原 `Metrics/standup/upright_fraction` 在100倍数保存点最低的模型；重复则去重，平局取更早。RSL-RL只在有回合结束时写指标，部分保存轮数没有日志值：取该保存点之前最近的真实原值，明确记录指标实际迭代／累计更新数，不补造测量；另提供该100更新窗口全部已记录值的均值，均值不改变最低点选择。每100更新的站立比例、高度误差、膝奖励与完整原始事件序列记录在training-curve.json。原训练高度误差按20s归一化，短回合可能偏低，不能代替回放高度MAE。

原冻结协调器在读取缺失的2599日志值时退出1，未启动评估；冻结源码、失败status和日志保留。仅评估恢复包装脚本位于 `artifacts/height-knee-training-20261013/assessment-only-remote.py`，只读已完成模型和事件，补齐曲线后调用同一冻结评估／渲染脚本；不调用Runner.learn，不改变训练接口或预算。

0.30m按 `wheel_balance_assessment.paired_acceptance` 与A验收v2配对：丢失任一原成功样本即退化、验收不通过，新增成功不得抵消。最终3500模型仅当至少一种初态0.30m高度跟踪≥32/64时追加256环境、0.27／0.30／0.34m的训练同等随机化复核，不学习；否则明确记录未触发。报告首次原终止条件／数值失败出现的时间，按5s分段统计，不用自动复位替换摔倒样本。正常训练噪声、延迟和间隔推扰保留，只有命令固定为零速和指定高度。

每检查点0.30m两种初态样本0视频由原始GPU qpos渲染，显示时间、实际高度和目标，不再积分。非零速度行走能力本次**未评估**，不作为部署交付。

## 解释边界

wheelleg v11是改法参考，结果不是本仓预期：两个XML被动膝damping0.01／0.03，本仓还有frictionloss0.015；具体因果未消融。单seed与64份配对初态只说明本批趋势。报告分开“已验证”“推断”“未能验证”，门禁通过、训练完成不代表策略通过。
