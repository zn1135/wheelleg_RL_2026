# 轮角39／82：首次站稳清零v2

后续已接入[SMP PPO v4](20261025-wheel39-smp-ppo-v4.md)的BC初始化、保存／恢复及各运行入口；轮角清零v2定义不改。普通PPO默认、独立CPU／ONNX／H7未改，本轮PPO零更新；下方“PPO仍为原接口”描述此前状态。

后续已将LQR位置重锁改为同一共享事件；轮角观测v2保持，教师标签另用新参考合同。见[共享位置参考](20261025-wheel-lqr-position-reference.md)。下方0.54～0.55秒差描述此前未对齐模式。

2026-10-25，Codex执行；独立成员复查和H7负责人核对待确认。训练仓`wheelleg_RL_2026`，detached HEAD `2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`加既有未提交改动；本轮[源码及差异](../../artifacts/wheel-angle-handover-reset-20261025-v1/source-hashes.json)为运行依据，无提交／推送。机器人为共享XML闭链串联腿，H7代码未改。

旧接口见[原轮角39](20261025-wheel-angle-observations.md)；新标识`chuanliantui-huananhu-wheel-angle-39-82-handover-v2`。权威实现为`ct_mjlab/wheel_angles.py`的HANDOVER常量和`multimode_observations.MultimodeObservationState.synchronize_wheel_handover`。

| 项目 | 旧v1 | 新v2 |
|---|---|---|
| 轮角参考 | 仅回合复位 | 回合复位＋正常趴地首次站稳确认 |
| 触发／重启 | 无起立事件 | `ground_start && recovered`首次为真；仅回合复位重新允许 |
| 字段／单位 | actor／critic14、15，左右轮native角差，rad、缩放1、±100、不回绕 | 保持 |
| 延迟／噪声 | 原位置组历史／噪声 | 清零只冲洗轮位置两列历史，其余历史及噪声不变 |
| 读时序 | 读取角差 | 首次actor或critic或子步处理事件，随后重复读不再清零 |
| 维度／控制／物理 | 39／82／6，100Hz策略、500Hz物理／PD，2／4ms延迟、0.2秒启动等待 | 保持；XML／关节顺序／根初态／增益／动作含义不改 |
| 模型格式 | `ct_bc_actor39_pd6_wheel_angle_v1` | `ct_bc_actor39_pd6_wheel_angle_handover_v2` |
| 配对schema | `chuanliantui_actor39_pd6_prestep_wheel_angle_v1` | `chuanliantui_actor39_pd6_prestep_wheel_angle_handover_v2` |

v2数据保存动作前`wheel_angle_origin`、`wheel_angle_handover_latched`、`wheel_angle_recovered`，加载时从真实轮角逐拍核对原点切换与观测。采集以`--wheel-angle-slots --wheel-angle-handover-reset`启用，独立回放按模型合同选择规则。旧模型／数据继续按v1读取；不按相同39维形状混用或静默转换，不改写旧文件。新版训练策略尚未产生，需另行数据与训练授权；旧技能保留训练的v1配对不能自动转作v2。

默认PPO／SMP PPO仍原零轮角39维，CPU／ONNX／H7未适配新接口；H7需同步首次站稳确认、轮角原点及位置组缓存清理，负责人待确认，本次无部署文件或模型。新清零与LQR内部位置积分的交接仍相差约0.54～0.55秒，并未完全统一两者参考。

20环境算术／有限物理、20条真实教师事件、数据／有限梯度、跨合同拒绝及旧v1检查通过。[报告及逐条证据](../../artifacts/wheel-angle-handover-reset-20261025-v1/README.md)。20条教师预检不等于新学生停车通过；无新增训练、CPU或真机验收，无新模型交付。
