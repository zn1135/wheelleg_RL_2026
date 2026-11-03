# 构建与验证

复旦原并联模型零速诊断：`scripts.agent.fudan_ppo_walk.probe_reference_zero check --output <新目录>`仅核对原脚本观测/历史/动作和资产指纹；`run`须另获600秒授权，使用原默认ONNX和控制函数，Warp GPU物理，不与本仓串联策略混载。原场景在30cm平台上，窗口内用`report_reference_zero --recording <目录>`对已保存姿态核对支撑面、生成高度修正的派生记录，再用`scripts/mjlab.sh accept --task fudan_ppo_walk --recording <目录>/platform-relative --output <新评分目录>`统一评分。原世界坐标、原评分和视频保留；实际控制周期、证据及复现命令见[本次记录](../../artifacts/fudan-original-parallel-zero-20261103/run/README.md)。

复旦显式v2保留续训：`check_preserve_v2 --checkpoint <原7000模型> --output <新JSON>`验证24项实际CPU/CUDA学习状态/RNG恢复、预检合同与采样拒绝、默认v3保留及真实循环7000→8000边界（采样/更新替代，0物理/NN/Adam）；`precheck --history-version v2 --viewer viser ...`才生成本次匹配的真实20环境预检。训练入口显式`--preserve-history-v2`，与`--migrate-history-v2`互斥。完成后`audit_resume_run`核对1000轮、v2→v2、源快照、16项日志与两Adam计数。[实际命令及授权](../../artifacts/fudan-parking-resume7000-to8000-20261102/RUNNING.md)。

复旦停车诊断：`fudan_ppo_walk.diagnose_parking --check-only`只检查替换倍率和作用行；动态入口须另获最多300秒预算，记录同初态原模型／停车前向真值／三轴真值三组，冻结权重。`report_parking --experiment <记录> --output <新报告目录>`仅离线CSV／绘图。`check_parking_commands --output <新JSON>`检查实际复位／步进中的10%保持＋10%停车采样、时间边界、关闭回退及保存恢复，物理／前向／更新0；[接口和命令](../interfaces/20261102-fudan-parking-commands.md)。

最终验收现统一为GPU、共用运动门槛及报告，见[统一GPU验收](../interfaces/20261102-unified-gpu-acceptance.md)。主线统一入口为`scripts/mjlab.sh accept --task <任务> --recording <GPU记录> --output <新目录>`；起立单列，旧CPU流程仅保留历史含义。

复旦当前最终验证仅GPU：[约定](../interfaces/20261102-fudan-gpu-validation.md)。`check_gpu_validation --migration artifacts/fudan-gpu-only-validation-20261102 --output <新JSON>`检查默认GPU、拒绝CPU参数、几何预检及导出依赖，0物理/NN/更新。历史CPU只读报告不再提供动态入口。

复旦延迟／推扰开关：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.check_disturbances --output <新JSON>`，覆盖实际PD前动作队列、传感槽位／Critic、局部复位、单拍推力、CPU/CUDA一致性、GPU夹具条件、checkpoint及预检条件核对；全部真实物理／网络前向／优化器更新为0。动态预检与行为另行验收，见[配置和时序](../interfaces/20261102-fudan-disturbance-switches.md)。

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

复旦PPO新增正式入口：`scripts.agent.fudan_ppo_walk.check`（静态几何/接口/梯度路由，0物理步/0优化器更新）、`check_scoring`（合成轨迹评分）、`precheck --viewer viser`（20环境可视物理预检）、`probe`（并行规模）、`evaluate [gpu]`、`score`、`render`。物理入口均计本轮3600秒，命令及门槛见[规格](../mjlab-fudan-ppo-walk.md)，程序检查通过不能代表行为通过。

复旦v2正式零物理入口：`scripts.agent.fudan_ppo_walk.check_reset_geometry`（含float32写回）与`check_reset_v2`（真实控制流/替代物理、计划热步1次但实际0）；`check_initialization`当时指向后者，当前默认检查已升history v3。动态20环境预检尚未运行，见[交付](../../artifacts/fudan-reset-v2-20261101/README.md)。

复旦验收v2（独立于reset/history版本）：`fudan_ppo_walk.check_scoring_v2 --output <新JSON>`合成70项边界与拒绝检查；`score_v2 --recording <既有目录> --output <新JSON>`；`compare_scoring_v2 --before <6000录制> --after <7000录制> --height030 <可选另列录制> --protected-source-manifest <本次源SHA清单> --output <新目录>`。只读既有记录，无物理/策略前向/优化器更新，不代表新行为或实机验收；[独立协议](../interfaces/20261102-fudan-walk-acceptance-v2.md)。

复旦history v3正式零物理入口：`fudan_ppo_walk.check_history_v3 --output <新JSON>`（当前27项真实输入/复位/CLI控制流，替代后端，无NN前向）；`check_history_migration --checkpoint <7000模型> --baseline-source <修改前fudan_ppo_walk目录> --output <新JSON>`（CPU/CUDA模型/两Adam/学习率/RNG/计数，受限显式v2→v3迁入及拒绝检查，0更新/0前向）。当前`check_initialization`指向v3；历史`check_reset_v2`仍可验证旧语义。无物理夹具不批准真实预检或训练，详见[接口](../interfaces/20261102-fudan-history-rollback-v3.md)。

复旦原生sim2sim只读对照：`fudan_ppo_walk.compare_native_replay --experiment <配对录制目录> --gpu-score <GPU-v2分数> --output <新JSON>`核对同15例/初态并列指标；`compare_native_video --experiment <配对目录> --output <新目录>`只处理已编码GPU/CPU视频帧，0物理/0NN/0网格渲染。`check_scoring_v2`增至71项，补不完整数值终止的标志保留检查，不改门槛。[本次实际两组15例及预算](../../artifacts/fudan-sim2sim-7000-h030-20261102/README.md)。

复旦后退起步只读分析：`fudan_ppo_walk.diagnose_reverse_startup --experiment <既有原生配对目录> --output <新目录>`核对旧轨迹/源码哈希、倾斜与触地先后、姿态罚饱和及CPU/GPU forward/力记录相位；不导入策略或推进物理，不以时间先后代替唯一因果证明。[此次结果](../../artifacts/fudan-sim2sim-7000-h030-20261102/reverse-startup-analysis-r3/README.md)。

复旦历史刷新时序诊断的动态入口`probe_refresh_timing`、`check_refresh_probe`、`audit_refresh_repeat`已随CPU后端撤销；原源码保存在[清理前快照](../../artifacts/fudan-gpu-only-validation-20261102/before/)，只读`report_refresh_probe`与[原结果](../../artifacts/fudan-refresh-timing-7000-20261102/README.md)保留。

复旦估速只读核查：`fudan_ppo_walk.diagnose_estimator --experiment <既有refresh探针目录> --output <新目录>`检查7000归档的标签/倍率/真实storage随机批配对、初始速度可观测性、25维打包及125维v3重建；复用已保存预测，不执行模型前向或新物理。[证据](../../artifacts/fudan-refresh-timing-7000-20261102/estimator-analysis-r2/README.md)。

复旦估速输入归因：`fudan_ppo_walk.attribute_estimator_inputs --check-only`做12项分组/替换/线性与交互归因合成检查，0物理/NN；`--budget <新窗口JSON> --output <新结果目录>`从原7000归档构造冻结encoder，仅对既有历史做同CPU复算和7组128子集离线替换，禁止复用过期窗口；`--report-only <已有实验根目录>`只读预测绘图，0新增NN。归因解释网络输出差，不能代替新物理闭环因果实验。[此次结果](../../artifacts/fudan-estimator-input-cause-7000-20261102/README.md)。

复旦CPU/GPU分叉追溯：`fudan_ppo_walk.diagnose_engine_divergence --experiment <已有refresh探针目录> --output <新目录>`只读取5条状态/动作/实际力矩记录及7000封存弹簧公式，加载mjb仅取关节/选项元数据；核对初态、同GPU动作、早期转移与GPU重复性，0物理步/0NN/0更新。不以相位未对齐的接触力判断先触地，不代表完成同状态单步物理干预。[本次结果](../../artifacts/fudan-engine-divergence-7000-20261102/README.md)。

复旦历史同状态单2ms动态入口`probe_single_step`已撤销，原源码见[清理前快照](../../artifacts/fudan-gpu-only-validation-20261102/before/)；`report_single_step`仍可只读已有记录，0新增物理/NN。[历史结果](../../artifacts/fudan-single-step-7000-20261102/report-final/README.md)保持，不作为当前CPU入口。

统一验收正式程序检查：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.motion.check_acceptance --baseline artifacts/unified-gpu-acceptance-20261102 --output <新JSON>`，核对四项边界、CPU拒绝、丢样本/缺字段、任务适配及同初态退化否决，0物理/NN/更新。Isaac仅检查新增GPU守卫AST，未运行旧环境动态。旧评分v2检查仍单独保留。
复旦8000停车积分：`fudan_ppo_walk.check_parking_integral --output <新目录>/checks.json`做门控/20ms均值/限幅/局部复位/非有限/真实v2历史推进、135归档初态替代恢复、严格差值留证、步后反馈时序及失败报告的无物理检查；`run_parking_integral --output <同目录>`仅在当次授权内执行135条GPU三分支、统一评分和三份并排视频。默认关闭，不加功能CLI，不训练；单一600秒及360秒回放硬截止，具体反馈相位、固定窗口和边界见[独立协议](../interfaces/20261103-fudan-parking-integral.md)。

复旦结构整理正式入口：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.check_structure_suite --output <新检查目录>`。外层禁止真实物理环境构造、积分及Adam更新；覆盖注册链路、配置消费者、迁移前后逐值比较、冻结8000模型、PPO梯度/顺序、两优化器恢复、v2/v3/预算/日志、积分r2及GPU入口。源迁移用明确哈希链核对，不能关闭来源校验；[说明与证据](../maintenance/20261103-fudan-structure-alignment.md)。
