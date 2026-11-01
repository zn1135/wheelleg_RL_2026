# 构建与验证

SuperMimic V13 r2已按实际更新数完成：从A700完整恢复后恰新增1500次actor Adam到A2200，无用户时长上限，不因摔倒或平衡丢失早停；精确封存A1200/A1700，训练结束后才串行评估。正式训练/完成审计、三次固定seed43/96例评估及pipeline均退出0；累计W9056/E/A/C各2200，新增W6000/E/A/C各1500、新399360/累计642048转移，学习循环4509.199秒/训练进程4522.847秒。A1200/A1700均存活0/96、平衡0/96，阶段128/384与112/384；A2200存活96/96、平衡仍0/96、384/384完整，数值异常均0。A2200前进vx −.493～−.492/MAE .992～.993m/s，后退vx −.684～−.683/MAE .183～.184m/s；停车水平mean .553～.589/peak .795～.850m/s、高度.303～.305m。末1秒每轮/总承重持续条件均96例失败，机身离地/姿态/角速度各0例失败；不能写成整段无接触。仅存活和部分跟踪恢复，平衡未恢复，不推断更多更新必然有效或无效。[r2实际命令、退出与结果](../../artifacts/supermimic-v13-continuation-20261101-r2/README.md)。101生产源/算法/物理不改；固定入口及Recorder兼容副本恢复另存属于目录迁移维护，不作算法证据。

按任务选择正式入口；检查输出必须注明模型、配方和覆盖范围。旧实验的详细命令保留在[历史专项检查登记](script-check-reference.md)，脚本旧路径统一查[迁移表](../../scripts/agent/MIGRATION.md)。

## 环境与入口

- mjlab 使用 `scripts/mjlab.sh` 或 `mjlab_training/.venv/bin/python`。直接运行辅助模块前设置 `PYTHONPATH=mjlab_training/src`。
- Isaac 使用 `.env.local` 的 `WHEELLEGGED_PYTHON`；先导入 isaacgym 再导入 torch，不安装新依赖或混用 mjlab 环境，见[旧任务规则](isaac-legacy.md)。
- 仓库没有单元测试框架或 CI。下表是正式程序检查／行为测量入口，不统一称作单测。

| 范围 | 程序检查模块（前缀均为 `scripts.agent.`） | 行为验收与命令 |
|---|---|---|
| V13 v5非零延迟/随机化 | `supermimic.check_v13_v5_training_conditions --conditions delay_only或training`；`supermimic.report_v13_v5_training_conditions` | 冻结双模型，真实PD FIFO和推力调度快照，0～10ms分桶、近零/虚响应及首回合失效记录；严格完整窗口复算、实际训练参数覆盖。0次更新，未覆盖活跃推力时必须注明 |
| V13 v5直接速度约束验证 | `supermimic.check_v13_v5_direct_response`；`supermimic.score_v13_v5_repair_horizons` | 独立MuJoCo原生Jacobian核对真实LINK vx、终态积分/公开step等价、双精度数值梯度；固定数据/尺度比较10/80/240ms自由预测。检查0次更新，通过不代表控制恢复 |
| V13 v5真实轮响应与修正候选 | `supermimic.diagnose_v13_v5_response`；`supermimic.repair_v13_v5_response --check-only`；`supermimic.report_v13_v5_response_repair` | 同状态原生GPU正负轮动作10ms分支与预热/最终模型比较；真实配对监督、直接LINK速度响应、原回放约束、独立seed留出及冻结/梯度检查。实际修正须另获明确更新及时间预算，候选不自动部署；旧300次执行源码按SHA另存，不能用新版损失冒充旧结果 |
| V13 v5限额试训审计 | `supermimic.audit_v13_v5_trial --run <本轮目录> --output <新JSON>`；`supermimic.report_v13_v5_trial --experiment <本轮实验目录>` | 核对SM615初始化、4000世界预热隔离、实际策略300次上限、7项网络/优化器与课程/CPU RNG恢复、源码及96例初始/最终测量；CUDA RNG是否验证单列，不把恢复检查当行为通过 |
| V13动作与失败时序 | `supermimic.audit_v13_action_failure_timing --output <新目录>` | 只读10模型各96例原始轨迹，哈希/首回合/100Hz去重核对，完整和统一时窗、左右轮/腿超限及失败前时序；不执行仿真，不据相关性确定阈值 |
| SuperMimic动作±3 | `supermimic.check_supermimic_action_clamp --integration`；参数见[接口记录](../interfaces/20261101-supermimic-action-clamp.md) | 验证噪声后限幅、执行/回放/奖励/下一观测动作相等、想象梯度、非有限停止、ONNX及新旧保存恢复语义；零优化器更新，不代表行为改善 |
| SuperMimic想象状态夹紧 | `supermimic.check_supermimic_imagination_clamp`；`supermimic.check_supermimic_no_history --integration` | 检查本机XML限位、机体系速度边界、范围内恒等/超界梯度、非有限保留、仅想象调用及真实标签不变；零优化器更新，不代表行为改善 |
| SuperMimic稳定性配方核查 | `supermimic.audit_supermimic_stability_controls` | 只读比较参考代码的状态夹紧、动作限幅和余弦调度；统计封存轨迹动作超界/关节限位和实际优化器学习率，不训练、不改配方 |
| SuperMimic单步终态 | `supermimic.check_supermimic_control_endpoint`；`supermimic.check_supermimic_no_history --integration` | 验证129→46、真实10ms终点监督/归一化、单端点路径、直接终态观测奖励、162维价值输入与有限反传；真实仿真5子步记录保持；零优化器更新 |
| SuperMimic航向无关输入 | `supermimic.check_supermimic_heading_inputs` | 对照参考47维特征、检查整体yaw/平移不变性、外力坐标、预测路径等变、四元数符号和梯度；同为129维的旧坐标合同也拒绝续训；零优化器更新 |
| SuperMimic无动作历史世界模型 | `supermimic.check_supermimic_no_history`（可加`--integration`） | 验证129维输入、历史扰动/梯度独立、当前动作梯度、归一化、旧包拒绝/旧actor继承和新包往返；零优化器更新，不代表行为通过 |
| SuperMimic | `supermimic.check_chuanliantui_supermimic`、`supermimic.check_supermimic_alternating` | [配方与验证](../mjlab-chuanliantui-supermimic.md)；旧配方检查不能代替当前算法检查 |
| SuperMimic预测误差 | `supermimic.record_supermimic_physical_prediction`、`supermimic.score_supermimic_prediction_details` | 冻结策略采集24环境/seed55诊断，再离线测10/80/240ms运动、双腿几何、观测与奖励误差；参数零更新，不代替96例行为验收。历史对照输入由 `supermimic.score_supermimic_prediction` 生成 |
| SuperMimic预测进步对照 | `supermimic.compare_supermimic_prediction_progress` | A700/A1808/A2200在相同当前及历史窗口比较；整体误差统一用A2200归一化尺度，避免尺度变化造成虚假改善；零训练、零新仿真 |
| SuperMimic塌落诊断 | `supermimic.diagnose_v13_collapse`、`supermimic.report_v13_collapse` | A4200/A5200估速器与控制网络交叉，`--parts-only`固定A5200估速、分别替换腿/轮动作；2秒、seed43/96例冻结诊断，同初态/物理逐项核对，不代替30秒行为验收 |
| SuperMimic轮动作方向诊断 | `supermimic.diagnose_v13_wheel_response`、`supermimic.report_v13_wheel_response` | A5200在0.2/0.4/0.6秒恢复原生GPU状态与PD历史，测左右轮两幅度正负扰动的10ms响应；以重复/原生基线误差筛选可判定样本，并核对自动微分方向；无训练，不代表完整240ms策略目标梯度 |
| SuperMimic左右轮输入核对 | `supermimic.check_v13_wheel_inputs` | 对已封存3×96状态检查141维输入的动作/历史/轮速槽、归一化、物理轴向及响应；CPU零训练，不把局部映射通过解释为全部训练分布或梯度正确 |
| SuperMimic错误响应机制 | `supermimic.diagnose_v13_wheel_response --history-only`、`supermimic.analyze_v13_response_mechanism` | 零延迟下固定物理状态/当前动作，仅改历史FIFO及模型上一拍动作；结合A4200/A5200同状态的速度/旋转贡献分解，保留噪声不明确样本，不等同完整策略损失归因 |
| SuperMimic历史数据分析 | `supermimic.analyze_v13_history_data` | 只读原训练初始延迟分布、原源码与诊断录制；核对动作历史标签和相关性，明确缺少完整在线回放时不能追溯具体训练批次 |
| SuperMimic世界模型修正审计 | `supermimic.check_v13_world_repair --experiment <修正目录>` | 审计300次世界模型更新、冻结网络/归一化、训练数据哈希及独立验证门槛；审计通过不等于候选采纳或机器人行为恢复 |
| LQR／BC／SMP | `imitation.check_lqr_vmc`、`imitation.check_bc`、`imitation.check_smp` | [teacher](../mjlab-lqr-vmc-teacher.md)、[BC](../mjlab-behavior-cloning.md)、[SMP](../mjlab-smp-prior.md) |
| 实采／CPU镜像 | `real2sim.check_real2sim_huananhu`，其他按 [real2sim目录](../../scripts/agent/real2sim/) | [实采命令](../commands/real2sim.md)、[sim2sim约定](sim2sim.md) |
| Isaac | [isaac目录](../../scripts/agent/isaac/)及原 `sim2sim/` 入口 | [训练→回放→sim2sim](../commands/isaac.md)；保留旧环境和接口限制 |

模型、参数和必需输入查表中任务文档；不要只凭模块名运行。`check` 有限数值检查、`--selfcheck` 形状检查和地面起立统计都不能单独代表行为通过。串联腿正式验收遵循[双轮平衡标准](../interfaces/chuanliantui-wheel-balance-acceptance-v2.md)，原成功样本任一失效即判退化。

2026-11-01确认[统一运动标准 v1](../interfaces/chuanliantui-motion-acceptance-v1.md)：前后速度平均绝对误差≤0.10 m/s、转向平均绝对误差≤0.10 rad/s、停车稳定阶段水平速度峰值≤0.05 m/s、稳定巡航及停车阶段俯仰绝对值峰值≤5°。起立暂不纳入本次统一。当前保留SuperMimic的共用运动判定；旧平地跟踪（对应旧训练线已清理）的0.30门槛不能代表新标准通过；刹停时限等完整协议仍待统一。

## 保留训练线的正式检查

```bash
export PYTHONPATH="$PWD/mjlab_training/src${PYTHONPATH:+:$PYTHONPATH}"
mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_retained_training
mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_script_layout
mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_script_merges
mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_script_entries
```

第一项检查五个旧task拒绝、普通PPO训练拒绝、⑥⑦⑧导入、共用环境源码／MuJoCo保护哈希、共用函数AST和实际删除清单。后续检查覆盖迁移表、内部导入、保存的SuperMimic预测／阶段指标、计数舍入与入口参数；均不训练、不推进物理，不证明行为通过。

统一运动复算：`python -m scripts.agent.motion.assess --measurement <SuperMimic测量目录> --output <新结果.json>`；算术检查用`--selfcheck`。原stand_v2专属审计入口随任务清理删除。完整范围见[保留线清理记录](../maintenance/20261101-retain-678-review.md)。
