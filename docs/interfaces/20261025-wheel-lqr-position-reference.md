# 轮角39／82：共享LQR位置参考标签

后续已按用户本地新增5000次预算完成新参考BC：原40组起立／平衡均40/40，停车0/40，未替换默认策略，H7仍待适配及负责人确认。[本轮模型记录](../model-deliveries/20261025-bc-aligned-reference-local5000.md)。下方“未训练”描述本接口首次零更新检查阶段。

2026-10-25，Codex执行；独立成员复查／H7负责人核对待确认。训练仓`wheelleg_RL_2026`，detached HEAD `2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`加未提交修改，[实际源快照](../../artifacts/wheel-lqr-reference-aligned-20261025-v1/source-hashes.json)为准。仅仿真侧改动，无提交／推送，H7源码未改。

旧[轮角清零v2](20261025-wheel-angle-handover-reset.md)使用环境站稳确认清轮角，但教师位置仍在内部DONE时清零，有约0.54～0.55秒差。新配对模式让二者都使用首次`ground_start && recovered`：事件前冻结LQR位置误差，事件当拍归零后才计算反馈，后续连续积分；不因内部DONE／离地／接管阶段同步额外锁位置。

观测保持`chuanliantui-huananhu-wheel-angle-39-82-handover-v2`，actor／critic14、15，左右轮连续native角差、rad／缩放1／±100／不回绕，39／82／6维不变。历史仍只清两轮位置列；其余通道、噪声／延迟、关节映射、动作含义、2ms物理／PD、100Hz策略、2／4ms等待、0.2秒启动及XML／初态保持。

教师标签参考改用`ct_bc_shared_wheel_position_reference_v2`，权威定义`bc.ALIGNED_LABEL_REFERENCE`。配对保存`lqr_position_reference_event`、`lqr_position_error_used`、`lqr_position_integrating`、`lqr_velocity_error_used`及`lqr_position_contribution`；数据加载检查同拍清零和后续积分。轮角本身仍不是LQR的补偿位移或速度误差积分。

`collect_reference_bc.py --wheel-angle-slots --wheel-angle-handover-reset`自动启用；`BridgeCfg.shared_wheel_position_reference`默认False，普通教师／历史v1与v2数据按原规则读取。新旧参考标签不得混入同一BC训练，旧记录不改写；当前旧4500候选不能称已适配新配对，没有新模型交付。默认PPO／SMP PPO、CPU／ONNX和H7未接入新规则；H7需同步共享重锁事件及位置积分门控，负责人待适配／确认。

实际20环境清零时差和当拍位置反馈项均0；教师起立／平衡20/20，但完整停车18/20，拒收2条漂移超限。默认6000样本逐值对照和H7原C数值检查通过，本轮BC／SMP／PPO更新0。[完整报告与证据](../../artifacts/wheel-lqr-reference-aligned-20261025-v1/README.md)。尚未验证新学生、移动、10ms压力、CPU或真机行为。
