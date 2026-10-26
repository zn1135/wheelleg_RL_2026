# 新39维观测中的左右轮角位移

新v2已增加首次站稳确认清零，旧v1定义保留；见[新清零接口](20261025-wheel-angle-handover-reset.md)。本页下方不清零的定义适用于历史v1。

后续旧技能保留训练已完成：显式旧39→轮角39迁移，首层14、15列置零，其余旧参数保留，叠加训练侧起立／反馈软约束。最终5000次模型平衡29/40，拒绝采用；全部保存点及候选见[本轮报告](../../artifacts/bc-preserved-wheel39-train-20261025-v1/README.md)。轮角公式及默认PPO／H7接口未改；[迁移与保留合同](20261025-bc-skill-preservation.md)。下方为此前阶段记录。

后续零更新诊断：旧稳定模型在内存忽略两轮角后，原40组起立／平衡40/40、停车0/40；新轮角10000的倾角轮端局部反馈仅为旧稳定模型约18.3%。未生成新模型或启动训练，不改变新模型此前失败结论。[证据与边界](../../artifacts/bc-wheel-angle39-feedback-20261025-v1/README.md)。

最新诊断及纠偏新增本地5000次已完成，轮角39累计10000；原40组独立起立降至0/40、最终平衡／停车仍0/40，退化拒绝采用。固定验证动作拟合改善没有带来闭环稳定，教师先站起再交接也失败。未重训SMP或启动PPO，不替换旧模型。[本轮报告](../../artifacts/bc-wheel-angle39-correction-train-20261025-v1/README.md)。下文保留此前阶段记录。

后续阶段已接入BC采集／训练／回放，并按新授权本地从头5000次；起立2/40、最终平衡0/40、停车0/40，行为未通过。默认PPO／SMP PPO、CPU和H7仍未适配新轮角39接口。下文“不启动训练／未接BC”为观测准备阶段的历史状态。[本轮模型与证据](../model-deliveries/20261025-bc-wheel-angle39-local5000.md)。

2026-10-25，Codex；用户先选择复用第15、16维轮角占位，随后要求删除第40维前向速度估计，当前actor为39维、critic为82维。不启动训练。当前detached HEAD为`2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`，工作区有既存改动，本次不切分支、不提交。

## 定义与参考

Isaac Gym的`wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc.py:compute_proprioception_observations`分别输入两轮`dof_pos[:, [2, 5]]`与轮速，角度缩放为1。这里参考其“左右轮位置分别输入”的布局，按本次确认改为相对回合复位时的连续角位移；不是完全照搬其绝对角零点。

当前Isaac串联腿`chuanliantui.py`为25维，已经删除轮位置，不能把它描述成仍含左右轮位移。

| 字段 | 数组索引（从0开始） | 新含义 |
|---|---|---|
| 左轮角位移 | actor/critic 14 | `q_lfwheel - q_lfwheel_at_reset`，rad |
| 右轮角位移 | actor/critic 15 | `q_rfwheel - q_rfwheel_at_reset`，rad |
| 左右轮转速 | actor/critic 20、21 | 原生关节速度乘0.1，保持 |
| 原第40维前向速度估计 | 原actor 39 | 已删除；新接口不再创建LQR速度旁路观察器 |

两轮独立输入，缩放1，沿用位置槽的±100 rad限幅；连续多圈角差不按2π回绕，不乘轮半径，不取左右轮平均。方向沿用各自XML关节轴，与原轮速一致，因此机体向前滚动时左轮通常为正、右轮通常为负。它们不是准确的世界机身位移，也不是停车目标误差。

每环境独立保存两轮复位角；局部复位只清本环境的角位移和位置组延迟历史。起立交接、教师接管或换命令不重新锁定轮角起点。读取只算角差，不推进积分，不写物理状态或动作FIFO。

actor新字段进入原六关节位置传感器组，共用该组的延迟和噪声；训练角噪声上限±0.025 rad。critic使用当前无噪声轮角差。其余原37个actor通道及80个critic通道不变，actor当前39维、critic仍82维，动作仍6维。

## 启用与兼容

权威实现：`multimode_observations.configure_wheel_angle_observations`启用`V14ObservationCfg.wheel_angle_displacement`，同时将配置传给actor、critic、复位与子步采样，防止首次初始化走错接口。actor直接输出39维，不再拼接估计前向速度。

新观测标识为`chuanliantui-huananhu-wheel-angle-39-82-v1`；对应配对schema为`chuanliantui_actor39_pd6_prestep_wheel_angle_v1`、模型格式为`ct_bc_actor39_pd6_wheel_angle_v1`，记录于`ct_mjlab/wheel_angles.py`。此前的估计速度40维接口及其独立检查入口已移除，历史源码和报告仍保存在上轮产物快照中。

本版与原PPO同为39维，但第15、16维的含义不同，不能仅按网络形状混用模型或示范。也不能直接加载此前40维权重。

普通PPO/SMP PPO仍使用原39/82接口（轮角占位置零）；旧通用轮式里程40/83与共享参考41/84入口不调用本配置，保持原定义。本次仅删除刚才新接口中的第40维，不删除旧数据集、模型或其他历史接口。本次没有把新轮角39维接入PPO训练、BC训练/数据转换、导出或CPU部署入口，未生成新数据集或模型。

H7源码未修改；板端需适配连续轮角累计、各环境对应的上电/模式复位参考、噪声/时序及新输入定义后才能交付。H7负责人核对、板端数值、真实时序及真机行为均待确认。

## 验证

当前正式脚本与结果位于`artifacts/wheel-angle-observation39-20261025-v1/`；此前40维检查记录保留于`artifacts/wheel-angle-observation-20261025-v1/`，不代表当前接口：

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  scripts/agent/check_wheel_angle_observations.py \
  --output artifacts/<新检查目录>/check.json

scripts/mjlab.sh check --num-envs 2 --steps 8
```

20环境检查通过：正反转/原地转向/10圈、原生轴符号、±100限幅、局部及整体复位、延迟历史清空、噪声/延迟、critic即时轮角、轮速和其他观测保持、世界平移不变、重复读取不写物理/PD队列。真实物理步进0.5秒，逐拍核对39/82形状、角位移和轮速，确认没有第40维、没有创建LQR参考观察器，启动前0.2秒主动电机力矩为零。

原39/82 PPO检查为2环境8步。这些是有限数值和接口检查，不是起立、停车、训练效果或真机验收。优化更新0次，没有启动TensorBoard。

首次轮角检查在只读核对时，检查器将mjlab的TorchArray直接与Tensor比较而报类型错误；改为克隆成Tensor后重跑通过，没有修改观测公式来绕过失败。
