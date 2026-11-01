# 已批准保留的训练经验

2026-11-01由用户逐条审批。只维护SuperMimic、BC与SMP；批准记录见[审批清单](../maintenance/20261101-retain-678-inventory.json)。本次没有训练或真机验证。E02、E03、E05、E06、E08未获采纳，不纳入经验规则。

## E01：分别判断存活、平衡和运动

存活只说明没有触发终止；双轮平衡需另查承重和姿态；运动需分别查速度、转向、停车和稳定俯仰。不能用一个成功率代替全部能力。

证据摘要：SuperMimic V13 A2200固定96例全部存活，但最终双轮平衡0/96；前进请求+0.5m/s时实际约−0.493m/s。旧稳站v2在同32例中可以均满足末1秒平衡判据，但稳定俯仰随迭代变化。完整来源见[SuperMimic续训报告](../../artifacts/supermimic-v13-continuation-20261101-r2/README.md)；旧稳站专项记录按用户要求清理，其数值摘要保留于审批稿和源码保全记录中。

## E04：对齐时间关系和复位历史

当前策略100Hz，即10ms一次；物理／PD为500Hz，即2ms一次。BC观测和教师标签必须对应动作前同一状态；延迟与动作队列分别核对。局部复位只清对应环境的历史，不能混入上一回合或干扰其他环境。历史接口频率按其原合同确认，不能套用当前值。

证据：[100Hz合同](../interfaces/20261023-mjlab-policy100hz.md)、[共享参考合同](../interfaces/20261025-bc-shared-reference.md)，已有同拍重读和局部复位检查。代码入口：`ct_mjlab/reference_contract.py`、`teacher/reference.py`、`envs/chuanliantui/pd_action.py`。

## E07：示范经过学生实际控制链路验证

LQR直接输出力矩的成功，不等于换成学生PD目标、延迟和限幅后仍成功。BC／SMP使用的示范应通过学生实际动作格式和控制链路采集或验证。

证据：[教师说明](../mjlab-lqr-vmc-teacher.md)、[BC接口](../interfaces/20261025-bc-actor-init.md)；已有独立学生与教师经学生PD接管对照。不因此修改H7控制实现。

## E09：区分数据用途并保留独立验证

学生失败状态上的教师查询动作可以用于BC纠偏，但该失败运动不自动成为SMP成功示范。按初始姿态分组保留不参与训练的验证集合，用于检查新姿态下的闭环效果。

证据：[BC接口](../interfaces/20261025-bc-actor-init.md)、[SMP说明](../mjlab-smp-prior.md)；数据来源与采样核对分别由`bc.ActionDataset`和`smp.dataset.MotionDataset`执行。

## E10：区分初始化与完整续训

初始化明确复制哪些网络、哪些训练状态新建。完整续训核对网络、归一化、优化器、课程、随机数与更新计数；真实环境或回放重建时，不称逐轨迹精确续接。

证据：[BC初始化接口](../interfaces/20261025-bc-actor-init.md)、[SuperMimic续训报告](../../artifacts/supermimic-v13-continuation-20261101-r2/README.md)。BC仅复制actor，不冒充PPO完整续训；SuperMimic恢复四个优化器及网络／归一化／计数。

## E11：分别验证预测能力与控制能力

预测或评分的训练误差降低后，仍需用真实仿真中的策略表现验证。按速度、姿态、关节等物理量及不同预测时长拆开误差，避免总误差掩盖局部退化。

证据：[V12重训报告](../../artifacts/supermimic-v12-retrain-20261031-r1/README.md)：部分速度／姿态预测改善，联合筛选仍失败；V13后续策略仍发生运动退化。这不证明预测误差是唯一原因，也不说明继续训练必定有效或无效。

## E12：分别记录真实更新次数，计数使用整数

各网络分别统计实际参数更新，不把循环次数等同策略更新次数。新计数使用整数；历史浮点累计值只允许有明确推导的舍入误差，不放宽其他拒绝条件。

证据：[V12重训报告](../../artifacts/supermimic-v12-retrain-20261031-r1/README.md)：float32大计数的精确5倍检查曾误拒正常训练；修正限定在精确区间、允许舍入及超界拒绝。检查入口为`maintenance.check_script_merges`的计数检查。
