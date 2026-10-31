# 稳站v2：跟踪激活期间主速度奖励保持权重1.0

| 项目 | 内容 |
|---|---|
| 日期／作者／复查 | 2026-10-29；Codex实现与自查，未独立复查 |
| 机器人／任务 | chuanliantui_stand_v2；硬件、XML不变 |
| 旧接口 | [v6速度课程](20261029-stand-v2-speed-slew.md)，主跟踪奖励recipe `stand-v2-upright-first-tracking-weight1-v4` |
| 新奖励recipe | `stand-v2-upright-first-tracking-active-weight1-v5` |
| profile | 仍`chuanliantui-stand-v2-isaac32-speed-slew-v6` |
| command contract | 仍`stand-v2-post-stand-half-vx15-slew-v2` |
| 训练仓／分支／基点 | Wheel-Legged-Gym；26_wheelleg；952e4260aedf04e28fdbf366d5cb1c5562c75263＋原脏工作区；代码与差异见[本轮证据](../../artifacts/stand-v2-active-weight-20261029/)；未提交推送，远程URL本轮未重新核对 |
| 部署仓／SHA／负责人 | 本轮未读取或修改H7控制代码；版本及负责人核对仍**待确认** |
| 权威实现 | stand_v2的mdp.py／rewards.py／runner.py／export.py；实验续训入口experiment_stand_v2_phase_mix.py |

仅实现与检查，无训练。本记录中“v5”指新奖励recipe；3999的旧profile本来也叫v5，不能只看版本尾号。

## 唯一奖励改动

`StandV2CommandCfg.tracking_weight_while_active`默认False。基础RewardManager权重仍0.2：

```python
boosted = state.tracking_active if state.cfg.tracking_weight_while_active else state.command[:, 0] != 0.
values['tracking_lin_vel'] *= torch.where(boosted, 5., 1.)
```

| 实际使用状态 | 旧recipe v4 | 新recipe v5 |
|---|---:|---:|
| 跟踪已激活，u非零（含爬坡） | 1.0 | 1.0 |
| 跟踪已激活，u=0（急停后，或采样到保持零目标） | 0.2 | **1.0** |
| 跟踪回合未就绪、恢复中 | 0.2 | 0.2 |
| 50%整回合零速角色 | 0.2 | 0.2 |

核分母0.25、项限幅、总奖励乘dt、准备期屏蔽及其他18项奖励全部保持。用实际命令u，不用隐藏target。动机是检验急停归零后权重变弱的推断；本轮不声称这个原因已被训练对照证明，也不声称刹停效果改善。

## 同拍关系

`StandV2Command._update_command`在同一次更新中写入tracking_active与command。mjlab ManagerBasedRlEnv.step先执行物理子步，再计算failure和奖励，之后才更新下一拍命令/观测。因此奖励使用的是本物理步实际执行的u和对应active，不是更新后的下一拍值；检查在每个奖励调用处比对步前快照，确认二者没有错位。

失稳在当前物理步末检测到后，命令和active在本次命令更新一起清零；下一物理步恢复0.2。检测发生的那一拍仍对实际执行过的命令计奖励，不改环境既有先后顺序。未就绪/恢复时没有额外强奖励；0.2秒电机保持期仍全部奖励为0。

旧固定命令v4中，目标幅值始终0.1～0.3且未激活时u=0，因此实际合法路径上`u != 0`与tracking_active逐拍相同，开关两种算法的奖励一致。只对v6可出现的active且u=0路径产生差异。

## 版本与续载

不新增profile或command契约：新旧命令、观测、动作及控制执行完全相同；变的是训练奖励语义。runner的task_contracts第三参数active_weight决定recipe_version，并严格校验。新旧配置原生互载会因recipe或profile不匹配而拒绝；开关默认False，不默认为旧模型升级奖励。

显式实验入口须同时指定`--post-stand-tracking --speed-curriculum --tracking-weight-while-active`：

- 3999（固定命令）或4999（v6命令、旧奖励）先以active开关False完整加载Actor/Critic、两个优化器及原课程，再执行既有速度课程迁移，最后只打开奖励开关。
- 新recipe v5父模型在加载前根据recipe打开开关；漏传CLI开关拒绝继续。新v5保留加载到的速度课程状态。
- 新模型保存完整recipe v5。run名自动追加`_activew1`，包括显式run-name；训练命令中的TensorBoard目录须包含该后缀。
- 新recipe训练轮数由用户显式提供正整数；旧入口原有固定预算规则不变。check-only仍只能iterations=0，不创建训练run，不进行优化器更新。

play加三个开关后才能原生载入新模型；导出按recipe自动识别，图形状和数值路径不改；瞬态与零速评估按checkpoint元数据选择开关，仍是冻结推理。检查中保存的interface-only-untrained文件只是旧权重的接口往返产物，不是新训练模型。

## 部署接口保持

Actor32／历史160／Critic89／动作6，观测6:9仍实际vx/yaw/高度×[2,.25,5]；五帧旧到新、首帧填满、500Hz物理/PD与100Hz策略、实体电机顺序、动作/PD/限幅、初态/接管/终止/随机化全部保持。ONNX仍observation_history[1,160]→actions[1,6]、velocity[1,3]，float32、opset13。

该奖励条件在模型图外，不增加H7观测、动作或协议字段，本轮不改H7代码、固件或物理参数。原v6历史/FIFO、模式与真实时序适配的待确认状态继续保留；未来新训练权重仍须配套交付和负责人确认，不能把奖励检查写成板端或真机通过。

## 验证依据与交付

[本轮检查报告、所有命令/退出码、训练与训练后配对命令](../../artifacts/stand-v2-active-weight-20261029/README.md)。覆盖旧奖励逐拍位相同、v4开关等价、新active零速段主项5倍、其他项/零速角色不变、实际物理步时序、新旧原生加载拒绝、两个优化器/课程往返和ONNX。

原正式检查及速度检查继续运行；从3999、4999和interface-only v5走实际迁移/续载入口均只冻结检查。只有训练后才能用3999/4999/新训练模型按相同seed、初态、物理随机化、a=1和a=0.5比较两次急停及零速v3.1；本轮没有这项行为结论。
