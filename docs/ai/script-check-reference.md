# 历次专项检查参考

本页保留脚本整理前的检查登记；包含固定旧配方和已结束实验。旧脚本路径查[迁移表](../../scripts/agent/MIGRATION.md)，归档工具查[复核说明](../../scripts/archive/20261101-script-cleanup/README.md)。不得将历史通过记录当作当前版本通过。

# 构建与测试

V13续训的新入口为`scripts/agent/train_supermimic_v13_continuation.py`；完整来源/运行审计为`check_supermimic_v13_continuation.py --source <原A368 final.pt> --output <新JSON>`或`--run <新run> --exit-record <实际退出JSON> --output <新JSON>`，只使用mjlab虚拟环境并前置`PYTHONPATH=mjlab_training/src`。固定96例仍用`assess_chuanliantui_supermimic.py`，`report_supermimic_v13_continuation.py`核三套原成功集合、真实29数组和阶段完整性；早停仅数值异常或原存活/平衡ID丢失，方向/停车/pitch不单独早停。外部调度保守把进程启动/保存计入最多3600秒训练墙钟，单独Final预留600秒；预算代数35组有限检查退出0，不能当行为证据。

新预算2026-11-01 00:36:27～02:06:27总90分钟，从A368完整恢复网络、四Adam、归一化及计数；不改生产算法/超参，不重复预热。A700真实训练/审计退出0，学习1021.385秒/进程1034.272秒，累计W3056/E/A/C各700，新增W1328/E/A/C各332及100352转移，累计242688转移。固定96例测量有效，但存活0/96、平衡0/96、数值异常0，原96成功ID全失，完整阶段160/384；巡航和停车均无完整样本，指标须保留N/A。已按原规则停训、不执行1100/1500；同权重独立Final复测实际退出0，仍为0/96存活/平衡、0数值、160/384完整。pipeline于01:10:55.874正常退出0，原02:06:27截止及剩余预算均不追加；不宣称所有长训必定无效。[实际命令、退出和最终结果](../../artifacts/supermimic-v13-continuation-20261101-r1/README.md)。

SuperMimic V13交替训练检查使用`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_supermimic_alternating.py --device cpu --output <新CPU.json>`和同入口`--device cuda:0 --output <新GPU.json>`；正式run审计加`--run <run目录> --exit-record <实际退出.json>`。最终CPU-r3实际11.884秒退出0，夹具真实W8/E/A/C各1仅证明world4/真实encoder/actor/critic阶段，不能计入试训更新；GPU-r2实际15.341秒退出0，640真实转移/3200子步、实际reset0仅为接口证据。检查覆盖无直接扭矩/critic旁路的未来motion梯度与数值差分、done后NaN不变、各窗口尾bootstrap/λ目标、actor期间参数和统计冻结、324维critic删除量不变及恢复拒绝无mutation。课程/RNG晚验经红例退出1证实并修复，坏课程/坏RNG现均在恢复前拒绝且状态不变；锁定源码下已完成实际训练及另行96例行为评估。

该轮2026-10-31 23:21:09～2026-11-01 00:21:09预算内结束，stage1/2实际600.299/601.513秒，累计W1728/E/A/C各368/142336转移；来源world6992次不计入本轮优化器。两次96例均384阶段完整、96存活/96平衡/160pitch原成功ID保留、数值失败0，29初态数组一致；跨旧reward正式比较仍not_comparable。中点方向/停车已退步，但预声明门槛只要求原成功保留，未触发停训，按原规则进入第二段，不追溯改规则。最终+.5请求实际−.4451～−.4457m/s、停车峰.8967～1.0428m/s，完整任务失败，不建议同配方追加。pipeline及只读补充报告均实际退出0，全部模型计算于00:00:06完成，正常1倍速视频由原保存轨迹绘制、零额外策略/积分；新seed/全包线/CPU物理/硬件未验证。[完整结果、实际退出和视频](../../artifacts/supermimic-v13-alternating-20261031-r1/README.md)。

以下V12一小时及更早工具均锁定历史源：其world-only审计、旧完整critic和冻结全部actor等断言不适用于V13，复现应使用对应封存快照，不能直接套当前生产源执行。

V12一小时重训使用入口`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_supermimic_motion_retrain.py --device cpu --steps 32 --output <新CPU.json>`及同入口`--device cuda:0 --steps 32 --output <新GPU.json>`。该轮开始前已是13项reward v7/辅助配方v2，生产只增加固定SM615 SHA的奖励v5→v7 actor-only迁移，原18项来源元数据保留；运动loss/物理/PD/softterm不改。CPU10.632秒/GPU15.041秒均真实退出0、Adam0，除原运动接口/梯度检查外，核完整615及拒绝非奖励元数据变更、保留奖励权重变更、旧整包续训，拒绝前状态/RNG不变。GPU640转移/3200子步，实际reset0；回合边界仅为人工done mask检查，不是物理复位验收。

该轮预算北京时间2026-10-31 22:08:57～23:08:57已结束，world实际2400.618秒/W6992/462848转移，完整actor含估速器/critic及三opt冻结；固定首发布W1792/W3088和真实final独立评分。评分35.073秒/独立307158字段复算7.828秒均实际退出0，所有模型计算在原截止前完成；H24 LINK .049213/.101868m/s和姿态3.42722/18.89501°较旧V12 r2改善，但关节/轮速等仍退步、八项联合筛选失败，不是同数据/同更新单因素对照。第一次完成审计及独立reader的误拒记录保留，最终修正仅针对float32计数容差和元数据序列化，未改模型或重推断。[该轮完整结果与实际退出](../../artifacts/supermimic-v12-retrain-20261031-r1/README.md)。

新评分入口`score_supermimic_motion_retrain.py --checkpoint <标签> <PT> --checkpoint <另一标签> <PT> --output <新目录>`逐模型核当前101源、固定配方及实际检查点来源，重复`--checkpoint`报告锁定更新点和final。两份旧V10采集/V9控制轨迹仍按真实旧v5的18项封存函数重建原reward并核原标签，绝不重贴13项标签；共同运动评分不重新前向旧world。评分和归一化0更新，不用预测承重评价motion world；旧工具继续锁定其历史源码。以下旧轮记录保留，不适用于当前源直接重跑。

V12上一轮r2运动状态对照使用`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_supermimic_motion.py --device cpu --steps 32 --output <新CPU.json>`及同入口`--device cuda:0 --steps 32 --output <新GPU.json>`。CPU核47/46/141/230维接口、删除槽NaN扰动不影响输入/输出/loss、梯度、完整SM615及零优化器；回合边界是人工done输入的ReplayBuffer mask检查，不是实际复位验收。GPU20环境×32拍实际640转移/3200子步，核Actor25/历史125、原采集reward/done/failure不变和运动world反向链路；本次没有真实reset，不能宣称已测真实复位边界。高度奖励改为显式`exp(-(height-target)^2/.0025) * upright_gate`、完全删除contact factor后，r2两进程重跑实际10.033/14.941秒退出0、Adam0；r1原检查记录保留。模型内奖励/终止已简化，不再要求其与原真实任务逐项一致；真实双轮承重行为要求保留。

本轮另获2026-10-31 21:22:33～21:52:33总预算，包含实现/检查/600秒拟合目标及评分；完整SM615策略含估速器冻结、价值网络不更新、world冷启。为显式删除高度项contact factor，r1主动中断（实际退出−2、进程墙时181.741秒、最后记录W480，无final），模型与记录保留且不作最终评分。r2重新冷启world，旧尝试耗时仍计入同一预算，截止不延长。评分仅复用两份原轨迹和固定窗口，以H1/H8/H24共同运动量的真实单位比较V10/V11/V12，封存预测后独立复算；不评价新模型预测承重，不以loss下降或冻结策略存活判有效。实际命令、原始退出及结果见[本轮r2记录](../../artifacts/supermimic-v12-motion-only-20261031-r2/README.md)。

V12最终评分入口为`scripts/agent/score_supermimic_motion.py`，本轮实际参数：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/score_supermimic_motion.py --checkpoint logs/mjlab/chuanliantui_supermimic_walk/20261031_v12_motiononly_worldonly_r2/final.pt --output artifacts/supermimic-v12-motion-only-20261031-r2/score`。仅接受真实退出0的final及一致101源/固定配方，核对原两份V10采集/V9控制器记录与V10/V11封存预测的相同窗口、真值和mask；新模型评分不更新参数/归一化，不重新前向旧模型。r2学习600.690秒/W1708/124928转移，完成审计退出0；评分实际17.847秒退出0、独立90038字段复算退出0，预声明联合筛选失败，暂不进入策略短训。H1/H8/H24真实单位、完整分母及局限见[最终报告](../../artifacts/supermimic-v12-motion-only-20261031-r2/README.md)；现有输出已封存，复现须另选新输出目录。

以下V10/V11程序与结果是历史入口，复现须用对应封存源码；其中对完整物理路径、原奖励/失败等价性的断言不适用于当前V12，不能直接套用当前生产代码执行。

SuperMimic V11冻结策略试训使用专用`check_supermimic_v11_world_only.py`，沿用实际物理、完整actor/critic冻结及真实退出核验，另核精确V11合同与loss配方；旧`check_supermimic_world_only.py`保留V10身份。`score_supermimic_v11_paired_prediction.py --trial <原V10 world-only artifact> --reference <原单拍/连续诊断artifact> --checkpoint early <固定早期PT> --checkpoint final <实际退出后的final PT> --output <新目录> --batch 64`严格区分V9控制器、V10采集和V11世界模型来源，在原360窗作两模型×两轨迹×连续/真状态单拍预测。原输入/目标/mask及训练配方/source SHA均校验，先封存再评分，评分过程零优化器和norm更新；训练期间world norm会更新。早期点不按误差挑选、不替代最终点，预声明门槛与独立复算见[本次30分钟结果与实际退出](../../artifacts/supermimic-v11-world-only-20261031-r1/README.md)。

SuperMimic V11历史 loss专项：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_supermimic_loss_scaling.py --trial <原V10 world-only artifact> --output <新目录>`验证4轴定点公式/59轴逐位不变、独立Huber分段值和导数、固定5N不被极端方差稀释、全mask0梯度、保存/提前拒错配方，以及真实旧world两批64行H1解码不变。Adam0；只是配方与程序检查，不是预测改善。下面V10录制/评分/漂移及world-only审计入口仍锁历史原源码；V11配套核验使用上面的专用入口，不直接套旧runner。[V11全部程序检查、来源与实际退出](../../artifacts/supermimic-v11-loss-scaling-20261031/README.md)。

SuperMimic V10单拍/连续预测诊断：`mjlab_training/.venv/bin/python scripts/agent/diagnose_supermimic_rollout_drift.py --trial <原world-only artifact> --output <新目录> --batch 64`只在CPU冻结最终world，对两份原360窗分别每拍重新输入真state254或连续自回归24拍，其他原始输入相同。必须H1全7字段逐位一致、连续全7字段逐位复现原封存final、世界权重/norm及99源/数据/hash不变；先封存再评分，零优化器/归一化更新/新物理。输出分开16hinge弧度与2slider米的误差，并列上一拍真实力保持基线；每拍真状态末拍仍是一拍条件预测，不能把MAE差称作可加因果或将诊断当策略行为通过。[已完成结果及实际退出](../../artifacts/supermimic-v10-one-step-vs-rollout-20261031-r3/README.md)。

SuperMimic V11历史沿用接触专项：`mjlab_training/.venv/bin/python scripts/agent/check_supermimic_contact_inputs.py --v9-checkpoint <真实V9 PT> --v10-checkpoint <真实V10 PT> --output <新目录>`做CPU物理参数换算、固定网络输入两列、数值梯度和真实V9/V10 actor兼容/拒整包续载检查。真实20×256集成同时核实5个默认接触/15个原随机接触环境、逐拍raw参数和真实路径、非contact参数/RNG不变及两分区覆盖。实际Adam0；临时fixture norm、mock与人工优化器槽单列，不是训练。[当前正式结果与退出](../../artifacts/supermimic-v11-loss-scaling-20261031/README.md)。

SuperMimic V10物理预测诊断：`record_supermimic_physical_prediction.py --checkpoint <PT> --output <新目录> --device cuda:0`冻结actor新录24环境完整五子步物理与原奖励标签；分别核对当前V10采集身份及V10/已验证旧V9 actor-only控制身份，使用外层采集控制器保存真实退出和初态配对，不能用旧部分qpos轨迹补造qvel/接触。`score_supermimic_physical_prediction.py --recording <中途目录> --recording <最终目录> --checkpoint midpoint <V10 PT> --checkpoint final <V10 PT> --output <新目录> --batch 64`仅CPU显式加载V10保存world，先封存四组360窗预测再按H1/H8/H24评分；采集源99项与world源严格核对。旧V9 world只能由`score_supermimic_v9_paired_baseline.py`加载完整校验的旧源码快照，在同一新数据上作配对参照，不迁入V10。各入口在`scripts/agent/`，只用mjlab虚拟环境，不学习。已完成V10拟合后的配对精度评分，速度改善但右轮承重退步，联合判据失败；不代表新策略行为。历史V9的均值替换probe仅在一致V9源码环境使用，当前V10明确拒绝，不能给新编码塞假raw参数。[本轮结果与实际退出](../../artifacts/supermimic-v10-world-only-20261031-r1/README.md)。

SuperMimic世界模型单训核验：`check_supermimic_world_only.py --run <run目录> --parent-actor <SM615 PT> --budget-record <budget.json> --checkpoint <PT> --exit-record <真实退出.json> --output <新JSON>`核对实际初始物理/材料分区、源快照、world更新、完整actor与critic含norm逐位冻结及另外三优化器为空；启动时使用`--startup`，不可用配置值替代实际更新/退出证明。正式训练审计与专用冻结审计分别保存；训练会更新world归一化，不能笼统说所有归一化零更新。只读检查通过不授权继续训练。

SuperMimic V11历史正式程序检查：`mjlab_training/.venv/bin/python scripts/agent/check_supermimic_bptt.py --output <新JSON>`做CPU解析/反向链路；`mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_supermimic.py --device cuda:0 --steps 256 --output <新目录>`做20环境真实5子步采集、Actor25/125/原18奖励/物理失败重建、四优化器零更新、恢复和ONNX。仅接口/梯度证据，不能代表模型行为通过。`check_supermimic_training.py`识别V11及历史合同，V11另核固定loss recipe及继承接触数据配方/编码假设，禁止用旧整包续训。检查通过不授权训练；CPU型入口前置`PYTHONPATH=mjlab_training/src`。

行走估速的独立闭环诊断（2026-10-31另行明确授权）：旧离线失败及门槛保留，`record_walk_estimator_trial.py`仅替换候选估速、冻结控制网络，使用候选自身传感与动作历史；原seed43初态和48/49保存基线逐项核对，沿原窗口完成三组，不因某组行为失败跳过其余组。`report_walk_estimator_closedloop.py --comparison 43 <JSON> --comparison 48 <JSON> --comparison 49 <JSON> --offline-result <旧最终JSON> --output <新目录>`只读汇总，分别报告旧离线结论、闭环行为与最终目标。`audit_walk_estimator_closedloop.py --artifact <本轮目录> --output <新JSON>`独立重算500Hz真实指标、t=0腿窗、持续反向及所有失败；`diagnose_walk_estimator_closedloop.py`只对已有三组成对轨迹作一次冻结估速交叉推理，比较同输入的网络变化与各自闭环历史变化，不新增物理步、拟合或修改门槛。实际命令、退出回执、适用范围和结果见[本轮诊断](../../artifacts/walk25-estimator-closedloop-20261031-r1/README.md)。本次授权不代表采纳候选、部署或继续训练。

行走估速单训独立入口：`train_walk_estimator_only.py --parent <原2999PT> --data-card <完整运行审核JSON> --execution-card <明确授权卡> --output <新目录> --tensorboard-dir <实际logs子目录> --device cpu --check-only`只做0更新真实数据/原模型/冻结与评分检查；去掉check-only的GPU训练必须另有本次明确预算授权，同一执行卡一次性claim防重开。`audit_walk_estimator_only_result.py`核对1000预算、逐字段冻结、曲线/采样与门槛；`evaluate_walk_estimator_final.py`只对两个新完整原控制轨迹核对唯一终点并推理、预测封存后才开标签评分，`audit_walk_estimator_final_result.py`独立重算封存预测指标。最终失败不运行候选闭环；`record_walk_estimator_trial.py`和`compare_walk_estimator_trial.py`仅在全部离线门槛放行且用户已授权时作候选实际反馈/父窗配对。本次完整命令、实际TensorBoard日志路径、退出回执、程序/离线/未执行行为范围见[1000更新结果](../../artifacts/walk25-estimator-only-20261031-r1/README.md)。

行走估速单训的数据准备（不训练）：`audit_walk_estimator_training_data.py --output <新目录>`仅盘点已有2999完整运行、原生25/125/COM/复位字段及分阶段有效帧；同目录加`--verify-legacy-history`只用44/45精确Actor25重建历史并复现原2999输出，不打开真速度评分、不拟合或仿真。按完整run划分44+46训练、45+47开发，另留新最终测试；本脚本通过不授权训练。预算、离线/闭环标准及尚未实现入口见[估速单训方案](../../artifacts/walk25-estimator-only-plan-20261031-r1/plan.md)。

行走影子估速延迟：`record_walk_shadow_sensors.py --training-artifact <冻结训练产物> --checkpoint <2999PT> --output <新目录> --seed 46 --schedule study --device cuda:0`只由原encoder控制，记录500Hz原始18槽及原生25/125输入；独立复核seed47/schedule validation。`analyze_walk_shadow_delay.py --study <目录> --validation <目录> --output <新目录> --device cuda:0`穷举四组3/5/8步全部81组合、命令/上一动作不延迟，先通过无延迟原生复现并封存两批预测，再读取同拍COM评分。逐环境保留初始化/停车/前后匀速/起停、所有组合及未知滞后，不移动标签或挑组合。`audit_walk_shadow_method.py`仅核对源实现及CPU哨兵；`audit_walk_shadow_result.py --artifact <本轮目录> --training-artifact <冻结训练产物>`只读封存预测重算指标/门槛及固定抽样滞后，并检查源码/物理/模型SHA，不重新推理或仿真。以上不代表行为验收；门槛不支持即结束，不自动接入控制或训练。[固定协议、结果及真实回执](../../artifacts/walk25-shadow-delay-20261031-r1/README.md)。

行走传感运动学补偿：`diagnose_walk_velocity_compensation.py --training-artifact <冻结训练产物> --checkpoint <2999PT> --output <新目录> --seed 44 --schedule discovery --feedback encoder`记录精确Actor25传感帧及同拍评分标签；独立验证用seed45/schedule validation。`walk_sensor_velocity.py`仅用IMU/实体电机和固定几何解无滑移闭链，不读真被动膝或真速度。`analyze_walk_velocity_compensation.py --discovery <目录> --validation <目录> --output <新目录>`在两批传感预测保存后才打开评分标签，逐环境分初始化/停车/匀速/起停报告三法误差、偏差、滞后和事前门槛。`audit_walk_compensation_startup.py`只读原始边界与几何复核初始化尖峰，无物理步进。不得用整体MAE改善掩盖初始化峰值退化，离线门槛不通过不进入补偿闭环。[本轮记录与真实凭据](../../artifacts/walk25-sensor-compensation-20261031-r1/README.md)。

行走实验先按[单假设迭代与行为决策](../interfaces/20261030-chuanliantui-two-policy.md#行走迭代与行为决策)写实验卡，分别报告程序检查、训练指标和实际行为；固定配对中原成功丢失即判退化，局部误差或奖励改善不能代替完整任务验收。

行走固定检查点对照：`diagnose_walk_v8_regression.py`支持重复`--extra-checkpoint <标签> <PT>`，在同一冻结环境增加检查点；新NPZ保存`fixed_*`初态、电机参数、质量与摩擦数组。`compare_walk_checkpoints.py --measurement <标签> <JSON/NPZ路径前缀> ... --baseline <标签> --output <新JSON>`仅CPU逐位核对实际保存数组和协议，复算逐条件/逐环境存活、末2秒速度误差及原能力保留；失败留在分母，旧产物物理数组缺项标未复核。使用mjlab虚拟环境，不学习；没有完整roll/pitch/角速度记录时不能判正式双轮平衡通过。

行走高度命令对照：`diagnose_walk_height.py --training-artifact <冻结训练产物> --checkpoint <PT> --output <新目录>`用冻结生产源码、seed43、同16初态/物理参数，在固定0.26m、20秒/2秒预热下仅改变零速/−1m/s请求；500Hz记录实际目标、关节、腿长与承重，禁止Adam，逐环境比较末2秒高度误差。请求前GPU轨迹数值差异另报，不能把行为关联写成唯一奖励原因。[实测与边界](../../artifacts/walk25-height-zero-backward-20261030-r1/README.md)。

行走摆腿离线审查：`summarize_walk_leg_swing.py`只读固定500Hz轨迹及名义视频保存的qpos，报告关节/虚拟腿摆角/腿长峰峰值、目标单拍变化、频谱、承重与逐环境1秒窗口；视频几何只用CPU正向运动学，不推进物理。至少区分初始、切换、斜率完成后和请求保持≥1.5秒的平台；位置差分腿速不能冒充真实qvel或训练奖励。旧轨迹未保存物理数组时明确比较边界；统计成功不能代替行为验收。[完整复现命令和实测](../../artifacts/walk25-leg-swing-audit-20261030-r1/README.md)。

行走估速反馈单轴隔离：`diagnose_walk_v8_regression.py --velocity-source oracle-vx`仅替换Actor的vx latent为同拍COM真速度×2，vy/vz保留原encoder；仍用冻结源码与同协议首回合、禁止Adam。`compare_walk_feedback.py --measurement <标签> <路径前缀> ... --baseline <标签> --output <新JSON>`仅CPU核对同一模型SHA、除velocity_source外相同协议及全部已保存物理数组，逐环境比较存活、速度、机高和15～20秒电机摆幅。缺物理数组或提前终止均明确记录，真速度诊断不当可部署修复。[单轴实测与复现](../../artifacts/walk25-estimator-vx-isolation-20261030-r1/README.md)。

getup等待释放诊断：`mjlab_training/.venv/bin/python scripts/agent/diagnose_getup_release.py --training-artifact <本轮训练产物> --output <新目录> [--recording <原4初态录像目录>] [--num-envs 64] [--mode zero-leg|legs-off|motors-off]`默认2秒，冻结训练源码、校验物理、严格加载999，记录500Hz实际PD/目标误差/实际力矩与接触，禁止Adam更新。`summarize_getup_release.py --root <五组诊断根> --recording <原录像>`核对公式/真实力矩/初态/首回合掩码并作图。仅诊断，关闭电机未失败不能当起立成功，不保证GPU逐位同轨迹。[五组命令与证据](../../artifacts/getup25-release-diagnosis-20261030/README.md)。

行走v8训练预算核对：`check_walk_training.py --training <本轮产物目录> --output <新JSON> --iterations <本轮新增预算> [--startup]`使用本轮冻结契约及源码哈希，检查2048环境/seed42、18项奖励、前后向独立课程、模型有限参数和两个优化器步数；从头和续训均支持，续训另核对源模型哈希、绝对轮次/物理计数及课程未倒退。完成时再核对实际退出0、按50轮保存的全部模型及连续loss事件。startup只查首模型，不代表训练完成或行为通过。检查只用CPU，不学习。

行走真实续训预检：`check_walk_resume.py --checkpoint <本角色真实PT> --output <新目录>`核对方向采样/晋级边界、Actor/Critic/两个优化器逐项恢复、自适应学习率、物理计数和课程状态，再用训练配置20环境256步随机动作做有限检查，禁止Adam更新；不代表行为通过。

当前getup恢复原EMA80、起立前力矩罚生效：`check_getup_torques.py --before-shared <本次修改前共享奖励> --before-contracts <本次修改前JSON> --previous-checkpoint <原EMA80千轮PT> --output <新目录>`检查16组合成幅度/阶段、轮力矩、真实RewardManager单次weight/dt、等待/异常屏蔽、其余18项及旧stand_v2公式、walk契约和旧recipe拒载。`probe_getup_torque_rewards.py`只读旧EMA80 Actor/课程，在原4初态前2秒比较新旧实际扣分与正奖励，禁止Adam更新；另跑原控制、20环境课程/ONNX/play及随机化32拍检查。[命令、全部范围与结果](../../artifacts/getup25-prestand-torques-20261030/README.md)。

行走模型名义诊断：`assess_chuanliantui_walk.py --checkpoint <本轮walk PT> --output <新目录>`先核对当前源码与训练snapshot逐份一致，再运行seed43/64近站环境/20秒首回合，禁止Adam更新，记录命令/速度/姿态/停车/失败及预选4样本原速视频。仅当前0.30m初态任务，报告100Hz名义数据，不代替500Hz正式双轮平衡、训练随机化、CPU或真机验收。[v6最终结果与范围](../../artifacts/walk25-v6-final-evaluation-20261030-r1/README.md)。

getup从头训练预算核对：`check_getup_height_training.py --training <本轮产物> --output <新JSON> [--startup] [--iterations 1000]`默认仍核对2000轮；本次1000轮需显式参数，严格检查0～999、21个保存点、两优化器最终各20000步及1000个loss记录，不从现场模型数猜预算。startup只核对首模型/源码/配方，不代表完成或行为通过。[本轮启动证据](../../artifacts/getup25-ema80-fresh1000-20261030-150851/startup-check.json)。

行走0.30m复位：双策略正式检查新增20环境真实几何与初态预览，核对名义高度0.30、2mm最低碰撞地隙、六闭链残差、关节限位、零初速度、局部复位不动其他环境/其目标高度、PD标称角保持；保留随机高度/速度采样和有限128步、两优化器/ONNX/play检查。原v5 model_999另检查完整恢复及32步有限推理；旧默认0.27m求解结果与修改前逐位相同。不是新初态下的行为通过。[证据](../../artifacts/walk25-reset30-20261030-150035/README.md)。

历史100ms接管检查`check_getup_handover.py`需对应100ms版本源码；当前已恢复EMA80，不在当前源码运行该入口。历史检查范围与证据见[已撤回候选接口](../interfaces/20261030-getup-measured-angle-handover.md)，不覆盖当前无接管控制。

当前getup平滑为80/20：正式动作检查在真实地面等待/释放路径核对0.2输入前三拍0.16/0.192/0.1984及实际PD4.8/5.76/5.952N·m，5子步保持、轮子/raw、局部复位和原FIFO保持；旧100ms控制配方拒载。[当前证据](../../artifacts/getup25-prestand-torques-20261030/control-check-ground-final/result.json)，奖励recipe变化由独立力矩检查核对，不把控制检查当行为改善。

getup四腿动作平滑：`check_getup_leg_smoothing.py --before-contracts <修改前契约JSON> --previous-checkpoint <未平滑getup PT> --output <新目录>`通过真实PD检查100Hz一次递归混合/5子步保持、实际目标与力矩、原始动作和轮子不变、20拍等待与第21拍边界、局部复位、原2子步延迟及旧/错配方拒载；再复用起立20环境有限检查及ONNX/有限play，CPU仅核对未适配时拒载。优化器更新0，不以目标变平滑代替行为验收。[范围与证据](../../artifacts/getup25-leg-smoothing-20261030-143250/README.md)。

行走v5速度下限：双策略正式检查新增两档128批停车/直行/原地转采样，验证首档±1.0、次档幅值[1.0,1.5]及双向覆盖、无组合请求、原256段课程晋级；核对非法下限、缺失/不同下限与旧profile/命令/课程拒载。命令夹具继续验证小于1.0的斜率过渡与停车P纠偏，起立与修改前固定物理输入下128拍命令/RNG逐位相同。保存恢复/两优化器/ONNX/play/独立CPU有限检查均不代表新速度行为通过。[本次记录](../../artifacts/walk25-speed-min1-20261030-142949/README.md)。

历史力矩权重加倍的检查与证据见[原torque2记录](../../artifacts/getup25-torque2-20261030-140050/README.md)，当时保留首次站稳门控；当前`check_getup_torques.py`已扩展为取消getup起立阶段门控、权重保持−0.0002的检查，当前参数与结果见本页顶部，不把旧证据当新配方验证。

getup倒置/腾空终止对齐Isaac：`check_getup_height_curriculum.py`复用9组各45拍合成输入，直接执行Isaac原`check_termination`张量函数逐拍对照，覆盖未触地、首次触地倒置、重力z=0/负数、站稳前后连续19/20拍腾空、单轮恢复清零和支撑力恰为1N；核对同拍缓存及旧终止配方拒载。保留起立20环境有限推演/续载/ONNX/play检查；不代表起立行为通过。[证据](../../artifacts/getup25-isaac-termination-20261030-124700/README.md)。

起立成功高度对齐Isaac：复用`check_getup_height_curriculum.py`，新增未解锁0.20/0.279/0.280/0.281m与已解锁0.199/0.200/0.201m的真实终止函数合成输入检查，要求连续50拍才记站稳，并核对旧0.20m门槛拒载；其余姿态/承重条件固定合格，不能视作物理行为验收。[证据](../../artifacts/getup25-success-height-20261030-122904/README.md)。

getup全强度对称惩罚：`check_getup_nominal_state.py --original <原1999奖励快照> --before <修改前共享奖励快照> --runner-before <修改前runner> --previous-checkpoint <旧缩放getup PT> --output <新目录>`以24组小/大腿角差、首次站稳前后和计时边界核对19项与原1999公式一致、其余18项/独立stand_v2缩放及walk契约不变，并检查旧recipe拒载；复用起立课程正式检查验证有限推演/续载/ONNX/play。无优化器更新，不代表学习效果。[本次证据](../../artifacts/getup25-full-nominal-20261030-120612/README.md)。

行走v4电机奖励：双策略正式检查独立扰动lf00/rf00电机速度及lf1/rf1被动膝，验证腿速集合、真实有限软限位、无限位电机不产生虚构约束/NaN、膝余量独立；验证姿态−100加权后[-1,0]（0/5/10/90/180°）和yaw主项1.0，拒载旧v3/profile及reward recipe，并记录未训练随机策略有限步进中的各项加权奖励贡献。[本次证据](../../artifacts/walk25-motor-rewards-20261030-024313/README.md)。

起立高度课程v2：`check_getup_height_curriculum.py --old-checkpoint <旧1999> --old-getup-checkpoint <旧固定高度999> --output <新目录>`独立验证起立，复用双策略检查新增0.30m未解锁64步、4095/4096完成回合与1228/1229成功边界、局部复位/未解锁完整恢复、旧v1/不同窗口及门槛拒载、CPU未解锁拒载；保留两角色20环境有限推演/两个优化器恢复/ONNX/有限play。`check_getup_height_training.py --training <本轮产物> --output <新JSON> [--startup]`核对从头2000轮预算、源码快照、两个优化器及实际退出码；startup不代表完成或行为通过。[接口与命令](../interfaces/20261030-getup-height-curriculum.md)。

行走奖励v3：双策略正式检查验证17项实际注册/窄核分发、删3项、同物理输入下起立阶段不影响奖励、首次站稳前惩罚生效、固定高度精度、旧奖励配方/权重/profile拒载；起立19项逐项仍调用旧公式，原999契约接受。另核对两优化器续载、ONNX及play有限步进和独立CPU联调。[本次证据](../../artifacts/walk25-reward-simplify-20261030-021050/README.md)。

行走v2随机高度与三类请求复用双策略正式检查，额外检查每回合高度保持/局部和自动复位、Actor槽8/Critic槽3及高度奖励、三类运动请求、旧walk v1/高度范围拒载；可配对`--getup-oracle <启动快照mdp.py> --getup-checkpoint <起立v1 PT>`核对同一物理输入下起立命令/RNG不变及旧契约兼容。独立CPU检查覆盖角色高度交接和越界/组合请求拒绝。[本次证据](../../artifacts/walk25-random-height-20261030-013939/README.md)。

双独立任务检查：`check_chuanliantui_two_policy.py --old-checkpoint <原1999> --output <新目录>`做两角色25/125/71形状、有限冻结步进、严格旧模型/错角色/参数拒载、Actor/Critic和两优化器恢复、两图ONNX及单任务play有限步进；`--commands-only`仅用模拟就绪状态核对命令/P外环门控。`check_chuanliantui_two_policy_sim2sim.py --fixtures <接口检查目录> --output <新目录>`核对CPU观测、独立32步和隔离物理的切换/FIFO/清基准。均用项目虚拟环境及`PYTHONPATH=mjlab_training/src`；禁止optimizer.step，不代表训练行为通过。[接口与范围](../interfaces/20261030-chuanliantui-two-policy.md)。

稳站v2 pitch阶段退化定位：`diagnose_stand_v2_pitch_progression.py measure --checkpoint <模型> --output <新目录> --num-envs 32 --seed 43`复用40秒/500Hz名义首回合、严格按原契约加载后在测量端关闭命令外环，禁止Adam更新；`summarize --measurement <标签> <测量目录> ... --output <新目录>`检查所有初始数组相同，并只在共同站起/双轮支撑/机身离地样本中比较末1秒平均绝对pitch。使用mjlab虚拟环境及`PYTHONPATH=mjlab_training/src`。35°平衡门槛不代表pitch精度达标；这是诊断，不替代行为验收。[1999→6299实际结果](../../artifacts/stand-v2-pitch-from1999-20261029-233840/README.md)。

稳站v2停车外环：`check_stand_v2_position_hold.py --checkpoint <B5999> --output <新目录>`核对关闭时yaw开/关与RNG等价、连续低速锁定/清除、死区/投影/限斜率、原地转/静止角色奖励与课程隔离、配置/版本拒载、ONNX及真实play有限步进。`assess_stand_v2_yaw_control.py`增加`--probe-position-hold`及`--position-source truth|wheel-odometry`；旧5999须同时`--probe-yaw`，仅探测。停车段保存全段/末1秒前向/横向误差、回拉峰值和过零统计，未锁定窗口不给零误差。原正式检查保留，均用仓库mjlab解释器与`PYTHONPATH=mjlab_training/src`。[配置与范围](../interfaces/20261029-stand-v2-position-hold.md)、[本次证据](../../artifacts/stand-v2-position-hold-20261029-214533/README.md)。

稳站v2转向与航向保持：`check_stand_v2_yaw_control.py --checkpoint <B5999> --output <新目录>`覆盖旧关闭路径/RNG、门控、松杆锁向、角度回绕、恢复与局部复位、两优化器续载、配置拒载、冻结推理、ONNX及play有限步进。`assess_stand_v2_yaw_control.py --checkpoint <yaw模型> --output <新目录> --conditions nominal --seed 43 --num-envs 60`测量15组直行/原地转/弯行及保持；旧模型须显式`--probe-yaw`，仅探测能力。正式原检查仍保留；均用仓库mjlab解释器与`PYTHONPATH=mjlab_training/src`，`--selfcheck`不能代替行为验收。[定义与命令](../interfaces/20261029-stand-v2-yaw-heading-hold.md)、[本次证据](../../artifacts/stand-v2-yaw-hold-20261029-q3YOzI/README.md)。

稳站v2首次起立缩放的续训核对：`check_stand_v2_nominal_training.py --training <本轮artifact> --output <新.json>`检查原1999→新2999恰好1000轮、两优化器40000→60000、21个保存点有限参数/配方/课程、启动/当前/训练快照一致及实际进程退出码；`--startup`只查首2000。`evaluate_stand_v2_phase_mix.py --record-physical`可选保存实际500Hz倒置/承重/姿态、首回合active/finite掩码及场景，关闭时仍仅原汇总。`compare_stand_v2_nominal_inversion.py --root <本轮evaluation> --output <新.json> --seed 43`独立复算逐环境倒置、末连续1秒v3.1、14初始数组配对、三分支与旧成功保留；首倒置时间按(index+1)×2ms。均使用仓库mjlab解释器；范围和实际结果见[本轮训练核对](../../artifacts/stand-v2-nominal-from1999-1000-20261029/training-check.json)及[配对统计](../../artifacts/stand-v2-nominal-from1999-1000-20261029/inversion-comparison.json)，统计通过不代表行为通过。

稳站v2首次起立姿态惩罚：`check_chuanliantui_stand_v2.py`追加`nominal_state`先限幅再缩放的0/125/250/490/500/1000ms边界、大角差饱和、小角差原平方、起立中断、首次站稳锁存及同拍缓存检查。其余14项原Isaac公式继续逐项对照；另保留已有4项改动的检查。使用本仓mjlab解释器和`PYTHONPATH=mjlab_training/src`，[本次有限检查](../../artifacts/stand-v2-nominal-first-stand-20261029/check/result.json)退出0、优化器更新0，不代表倒置或行为改善。

稳站v2 B训练完成核对：`check_stand_v2_tight_training.py --training <B训练artifact> --output <新.json>`核对实际预算、两个优化器步数、所有保存点配方/有限参数/课程、启动源码快照和TensorBoard迭代/窄核日志，另列当前源码差异；无仿真或学习，不替代行为验收。三模型两斜率冻结测量后，`summarize_stand_v2_tight_effect.py --root <3999/A/B测量根目录> --output <新.json>`以同ID样本比较保持段误差、急停与原零速v3.1成功保留，并核对零速命令/激活门控为0；角速度超限和真实翻倒分开记录。统计核对退出0不表示行为达标。[本轮结果和精确命令](../../artifacts/stand-v2-tight-effect-20261029/README.md)。

稳站v2 B窄核检查：`check_stand_v2_tight_term.py --checkpoint <3999> --speed-checkpoint <旧速度4999> --active-checkpoint <A4999> --oracle <只读A奖励快照> --output <新目录>`。覆盖默认19项/日志键、v4/速度v6/A关闭B逐位相同、新项公式和非激活零值、实际manager加权、四版本16种原生加载组合、核分母/权重校验及非默认参数、两个优化器与课程往返、ONNX与真实play参数/加载后16步冻结推理（替代无限viewer）。实际迁移入口对3999/旧4999/A/B用`--tracking-tight-term --tracking-weight-while-active --post-stand-tracking --speed-curriculum --check-only --iterations 0`；保持原正式、速度及A检查。验收新增保持段MAE≤0.10比例及按速度的稳态误差，课程门槛保持。[实际命令/退出码与范围](../../artifacts/stand-v2-tight-term-20261029/README.md)。

稳站v2训练效果录像：`assess_stand_v2_speed_transients.py --record-qpos`可选保存本次100Hz姿态及首回合结束时刻，不改变验收口径；`render_stand_v2_speed_comparison.py --before <3999测量目录> --after <新模型测量目录> --output <新目录>`读取匹配初态和保存姿态，按最小共同首段门控完整且40秒存活ID选择±0.5样本，提前结束则冻结并标注，不重新积分。输出两段40秒原速视频、源模型/姿态哈希及逐帧解码结果，例子不替代批次统计。均前置`PYTHONPATH=mjlab_training/src`并使用仓库mjlab解释器。完成核对器`check_stand_v2_speed_training.py`支持active权重标识；[本次三模型两斜率结果及录像](../../artifacts/stand-v2-active-effect-video-20261029/README.md)。

稳站v2激活期奖励检查：`check_stand_v2_active_weight.py --checkpoint <3999> --speed-checkpoint <4999> --oracle <旧rewards.py> --output <新目录>`对实际奖励调用逐拍核对旧实现位相同、固定命令开关等价、active零速段主项×5及其他项保持、同拍命令/active快照、5种错误版本拒载、优化器/课程往返和ONNX。保留原正式/速度检查；实际实验入口以`--tracking-weight-while-active --check-only --iterations 0`核对迁移/原生续载无学习无run。`compare_stand_v2_active_weight.py --root <三模型两斜率测量根目录> --output <新.json>`核对同初态并重算零速v3.1，报告两次急停达标数和门控完整者停稳确认率，训练范围±0.5与全速度分列；`--selfcheck`只测统计边界。全部前置PYTHONPATH并使用仓库mjlab解释器，[命令、结果和局限](../../artifacts/stand-v2-active-weight-20261029/README.md)。

稳站v2速度课程训练效果核对：`check_stand_v2_speed_training.py --training <1000轮artifact> --output <新.json>`核对3999→4999预算、两优化器80000→100000、源快照和各保存点速度课程。`summarize_stand_v2_speed_effect.py --root <本次配对目录> --output <新.json>`独立核对两种子零速14类初态/物理参数、冻结与平衡统计，以及a=1瞬态13类初始数组/重量、各段结果和共同完整保持段误差；未测量不写0，旧成功丢失/新增分别列出。零速评估器支持v6加载后显式关闭新命令，瞬态评估保持外部目标与生产门控。命令和局限见[效果记录](../../artifacts/stand-v2-speed-effect-20261029/README.md)，本轮无新增训练。

稳站v2速度课程：`check_stand_v2_speed_commands.py --checkpoint <3999> --output <新目录>`覆盖限斜率、49/50拍门控、四类转换、0.49/0.50秒测量边界、课程晋级/失败和配置/优化器checkpoint往返、同物理状态零速逐项奖励/观测相等、单输入ONNX、128环境40秒冻结随机化rollout及60米平面射线。`assess_stand_v2_speed_transients.py --checkpoint <2999/3999/v6> --output <新目录> --conditions training --acceleration 1`执行128环境40秒起立/急走/保持/急停/换向，500Hz统计MAE、25°倾角、刹停时间/路径、接触与倒置；50%零速组单独按v3.1验收，未起立/未完成不剔除。`--selfcheck`仅核对统计边界。均使用本仓mjlab解释器及`PYTHONPATH=mjlab_training/src`，不学习；有限检查通过不代表速度行为达标。[配方、命令与范围](../interfaces/20261029-stand-v2-speed-slew.md)。

稳站v2保持复查：`check_stand_v2_retention_seeds.py --root <六组新种子测量目录> --output <新.json>`核对seed43/44/45的14类初始参数、回合统计和共同曾起样本的成对差异。`diagnose_stand_v2_late_balance.py --checkpoint <3999> --original <原seed42测量> --output <新目录>`复用40秒评估入口，对原十例记录500Hz姿态/支撑/真实接触点滑移/外推力及100Hz姿态，不训练；`check_stand_v2_late_balance.py --recording <复跑目录> --original <原测量> --output <新.json>`复算接触运动学和判据；`render_stand_v2_late_balance.py --recording <复跑目录> --output <新目录>`输出十张末5秒曲线和完整40秒原速视频，逐帧解码。均用仓库虚拟环境、前置`PYTHONPATH=mjlab_training/src`，不把复跑冒充原轨迹、不更改验收标准。[本次范围与结果](../../artifacts/stand-v2-retention-20261029/README.md)。

稳站v2跟踪效果汇总：`summarize_stand_v2_post_stand_tracking.py --run <本次artifact> --output <新.json>`联合已通过的训练完成检查、零速配对和速度测量检查，只在共同就绪且完整的样本上比较速度误差，并单列未就绪、倒置及旧成功丢失；不新增验收规则。`render_stand_v2_tracking.py --measurement <速度测量目录> --output <新目录>`从保存qpos渲染完整22秒原速，优先选训练范围内的站稳与失败样本，提前结束时冻结首回合末姿态，逐帧解码检查。结果范围见[本次评估](../../artifacts/stand-v2-tracking-20261028/README.md)。

稳站v2站稳后速度配方：`check_stand_v2_post_stand_tracking.py --checkpoint <原2999> --output <新目录>`检查49/50拍门控、失稳清零、局部复位、零速奖励隔离、非零有效权重1.0、新旧profile及保存导出，并冻结128环境40秒推演，无参数更新。`check_stand_v2_post_stand_training.py --training <training目录> --output <新.json>`只核对授权1000轮、完整续载/优化器步数、门控/保持/std及来源，不代表行为验收。均前置`PYTHONPATH=mjlab_training/src`并使用仓库虚拟环境，见[配方](../interfaces/20261028-stand-v2-post-stand-tracking.md)。

稳站v2 CPU数值异常补查：`diagnose_stand_v2_cpu_launch.py`复现初态17的原2ms积分、关闭电机、关闭全部驱动、0.5ms积分四组；`diagnose_stand_v2_launch_reference.py`在真实训练引擎复跑同初态，不学习。`check_stand_v2_cpu_launch_stability.py`用当前入口检查128初态各前5秒的100Hz状态有限、高度≤1m、|竖直速度|≤5m/s；不裁剪状态，不作起立验收。接口检查新增每10ms策略步5次PD、20次原生积分断言。原v1初态17峰值5.234m；当前细分积分筛查通过，但初态17 CPU仍未站起，行为未通过，见[命令与证据](../../artifacts/stand-v2-cpu-launch-20261028/README.md)。

稳站v2原生CPU sim2sim：`scripts/agent/check_stand_v2_cpu_sim2sim.py --checkpoint <模型> --onnx <单输入160维图> --initial-states <名义initial.npz> --output <新目录>`（前置`PYTHONPATH=mjlab_training/src`，使用mjlab虚拟环境）对照真实任务的32维构造、标称角、PD/保持、五帧FIFO及102拍承重恢复/成功计时，参数更新0。`sim2sim/mj_sim2sim_chuanliantui_stand_v2.py`用原生CPU物理执行ONNX，`--seconds 40 --report <新.json>`作有限单例，`--viewer --port 8093`提供浏览器回放；均须同checkpoint/onnx/initial-states。当前只支持已解锁0.22m名义零噪声/零时延，范围和命令见[本次记录](../../artifacts/stand-v2-cpu-sim2sim-20261028/README.md)，不代替完整多初态或真机验收。

稳站v2探索下限续训：实验入口支持`--min-std 0.05`，使用原生Gaussian采样范围而不改raw参数或entropy0.005；编号从父checkpoint继续。`check_stand_v2_exploration_resume.py`核对2499→2999恰好500轮、两个优化器50000→60000、全程采样下限和真实复位时序、生产源码未变及确定性输出不加噪声。`compare_stand_v2_phase_mix.py --seeds 42`可单独核对本次有限行为配对，不与旧三种子汇总混比；有旧成功丢失时写出`regressed_rejected`并退出2，输入/统计核对通过不代表行为通过。[命令和范围](../../artifacts/stand-v2-phase-stdmin-20261028/README.md)。

稳站v2打散复位实验：`scripts/agent/experiment_stand_v2_phase_mix.py`用真实40秒预运行和局部复位形成回合年龄分布，不改生产配方或伪造回合时钟。`--check-only --iterations 0`检查保持力矩、历史局部复位、参数/优化器未变及后续41秒阶段分布；训练只接受本次授权的500轮。`evaluate_stand_v2_phase_mix.py`以40秒/500Hz、nominal/training两条件和三个种子作首回合配对测量，`compare_stand_v2_phase_mix.py`核对初态/物理/模型哈希并汇总起立和平衡，漂移单列；多个评估种子不等于重复训练，也不替代既有20秒验收。[命令与本次范围](../../artifacts/stand-v2-phase-mix-20261028/README.md)。

稳站v2速度命令探测：`PYTHONPATH=mjlab_training/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mjlab_training/.venv/bin/python scripts/agent/assess_stand_v2_tracking.py --checkpoint <模型> --seed 42 --output <新目录>`。原地面初态先零速起立8秒，仅已站起且NORMAL者接收0/±0.1/±0.3/±0.5m/s命令10秒，跳过响应前2秒统计，最后4秒归零；64环境每档9或10例，速度与双轮平衡分开报告，不套用multimode的±0.3m/s粗筛容差。测试使用独立计时，不能用检查点恢复的累计训练步数计时；每拍核对Actor命令槽和历史末帧。`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_stand_v2_tracking.py --measurement <测量目录> --initial-reference <原验收/physical.npz> --output <新检查.json>`核对命令实际送达、统计复算、500Hz支撑、初态/质量及模型源码哈希。stand_v2只训练零速，非零为未训练能力探测，不能当作训练随机化或真机验收。

稳站v2起立/漂移分开验收：`PYTHONPATH=mjlab_training/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mjlab_training/.venv/bin/python scripts/agent/assess_chuanliantui_stand_v2.py --checkpoint <本版模型> --num-envs 64 --seed 42 --output artifacts/<新目录>`。真实500Hz子步检查20秒末连续1秒双轮平衡v3.1，并要求曾满足任务起立条件；速度、位移和高度仅单列，不套用旧静止门槛。保留首回合冻结，核对actor/encoder参数不变，保存全部100Hz姿态以制作对应视频；有限名义种子不能代表真机或训练随机化成绩。定义见[稳站v2](../mjlab-chuanliantui-stand-v2.md)。

同口径模型配对：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/compare_stand_v2_acceptance.py --before <旧验收/result.json> --after <新验收/result.json> --output <新比较.json>`；独立复算末500拍组合条件、核对冻结掩码、全部初态/质量及模型/源码哈希；有旧成功丢失时退出2并记录`regressed_rejected`。对应完整视频：`mjlab_training/.venv/bin/python scripts/agent/render_stand_v2_acceptance.py --acceptance <验收目录> --output <新视频目录>`，只读取该次保存轨迹，原速并排成功/失败样本，不另跑策略。

`chuanliantui_stand_v2`正式有限检查：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_stand_v2.py --output artifacts/<新目录>`；对照保留的15项Isaac奖励和站起后终止，另检查直立门控、未饱和姿态扣分、起立弱阻尼、当前50拍站稳/精确高度/失稳回退、新成功课程，以及宽限/超时/奖励屏蔽、32/160/89/6、500/100Hz、历史局部复位、梯度隔离、旧配方拒载、续载和ONNX；优化器step被拦截，参数更新0。范围和命令见[稳站v2](../mjlab-chuanliantui-stand-v2.md)。

接触诊断追加1N边界、同拍NORMAL与接触的交集、真实回合均值、局部复位统计及离线汇总核对；统计不反馈控制。已有`diagnose_chuanliantui_stand_v2.py`结果可用`--from-report <result.json> --output <新目录>`复算接触统计，不重新仿真，详见任务说明。

首回合诊断在样本提前结束后冻结该样本结果，仅复位已结束环境以继续其余样本；复位后的轨迹不计统计。2026-10-27 model2500重测覆盖23例提前结束，已核对active终止后全False及最终成功计数为0；[证据与限制](../../artifacts/stand-v2-resume2000-20261027/README.md)。

起立翻转诊断：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/diagnose_stand_v2_inversion.py --report <已有result.json> --output <新目录> --seconds 5`重放同初态，采集500Hz电机需求和100Hz动作/奖励。`--zero-actions`保留PD默认目标，`--motors-off`仅在诊断中关闭六电机、保留弹簧；对照用1秒独立目录。无学习，非行为验收；[本次原始对照及采样限制](../diagnostics/20261027-stand-v2-inversion.md)。

平滑接管仅诊断验证：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/validate_stand_v2_handover.py --report <已有40秒result.json> --output <新目录> --handover-seconds <0或候选秒数>`；复用相同初态及原平衡判据，核对参数不变，完整40秒及首秒500Hz采集，不修改生产动作类。当前0.05/0.10/0.20秒均丢失原成功，未采纳，[结果与限制](../diagnostics/20261027-stand-v2-handover.md)。

NORMAL轮端承重检查追加每轮10%/5N、合计80%边界、随机化质量、仅取向上分量、连续50拍、旧恢复规则拒载/显式诊断及保存导出版本。行为另用`compare_stand_v2_wheel_normal.py --before <旧result.json> --after <新result.json> --output <新比较.json>`核对完全相同初态/质量/模型；丢失旧平衡样本退出2。来源与已发现的旧模型退化见[轮端NORMAL接口](../interfaces/20261027-stand-v2-wheel-load-normal.md)。

独立稳站任务`chuanliantui_stand`：`mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_stand.py --output artifacts/<新目录>/finite-check.json`检查XML、500/100 Hz、39/82/6、复位、课程续载和500 Hz验收采集器。有限检查不替代20秒行为验收，完整命令、标准和TensorBoard入口见[稳站任务](../mjlab-chuanliantui-stand.md)。

训练结束续接工具：`mjlab_training/.venv/bin/python scripts/agent/check_wait_training_completion.py --output artifacts/<新目录>/check.json`以真实短进程检查退出事件、PID复用、等待中断隔离和结果不能覆盖；不训练、不调用其他模型。当前代理等待及使用命令见[稳站任务](../mjlab-chuanliantui-stand.md#训练结束后自动续接当前代理)。

2026-10-23 USB大机目标腿长→训练机体高度：实际C拨轮/模式/观测/恢复及当前XML直立参考检查入口如下；设备状态使用替身，不证明实机几何、调度或行为。两机各180组模式回归、低位恢复计时与大小机正式构建证据见[目标换算接口](../interfaces/20261023-h7-leg-to-body-height.md)。`--before-rc`指向本轮修改前已保存的原RC源码，用于模式回归。

```bash
CC=/home/zn/miniforge3/envs/fudan_env/bin/x86_64-conda-linux-gnu-gcc \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 mjlab_training/.venv/bin/python \
  scripts/agent/check_h7_height_command.py --h7-root deployment/H7_RL_ros \
  --before-rc artifacts/h7-height-align-20261023/before/imcalib/task/rc_command.c \
  --out artifacts/<新目录>/height-check.json
```

2026-10-23 H7 USB导入13842：`CC=<已有宿主gcc> mjlab_training/.venv/bin/python scripts/agent/check_h7_policy_timing.py --h7-root deployment/H7_RL_ros --out artifacts/<新目录>/policy-timing.json`核对实际分频函数（外设替身）及实际恢复C：大机100Hz策略/500Hz PD分频，小机100Hz/1kHz；200ms趴地保持、500ms稳定与退出清零。只证明有限软件计时，不证明板端调度或完整执行链。两机正式编译命令保持下文`check_real2sim_huananhu.py --build-only`，交付见[13842记录](../model-deliveries/20261023-model13842-h7-import.md)。

2026-10-23策略提频检查复用`check_h7_simulation_latency.py`：mjlab训练／play与独立CPU均为2 ms物理、10 ms策略，实际每动作调用5次PD，趴地保持仍为0.2秒；CPU新动作首用对应2/12 ms。`check_multimode_cpu_sim2sim.py`同步检查20步零力矩保持、21步policy_active浮点边界、20 ms腾空确认与0.5秒恢复确认。`scripts/mjlab.sh check`报告实际物理／策略步长，`eval`默认按环境周期换算20秒步数。命令、证据和行为范围见[100 Hz策略试验](../interfaces/20261023-mjlab-policy100hz.md)。

2026-10-22默认腿角偏置检查复用下方`check_h7_simulation_latency.py`：新增训练四腿±5°范围、PD/观测同基准、复位固定与play零偏置检查。范围见[默认腿角随机化](../interfaces/20261022-huananhu-nominal-leg-offset.md)。

2026-10-22新增电机响应候选有限检查：`mjlab_training/.venv/bin/python scripts/agent/check_motor_response.py --out artifacts/<新目录>/motor-response.json`。覆盖非实测4ms时间常数的CPU/GPU真实PD响应、指数卷积、部分复位及禁用清零、四环境32步；`--baseline-cpu <改动前源码快照>`可另核对默认0时间常数的CPU逐值等价。默认候选关闭，不学习、不辨识真实参数、不替代行为验收。见[接口说明](../interfaces/20261022-h7-simulation-timing.md#电机响应候选未实测默认关闭)。

2026-10-22新增实测H7时序有限检查：`mjlab_training/.venv/bin/python scripts/agent/check_h7_simulation_latency.py --out artifacts/<新目录>/timing-check.json`。覆盖训练6/8/10 ms随机等待、play/CPU一拍等待、真实PD首用与旧动作保持、部分复位与39/82维有限状态，不训练、不代表行为验收。校准范围与精度限制见[2 ms精度时序接入](../interfaces/20261022-h7-simulation-timing.md)。

串联腿新任务使用独立 Python 3.12 / mjlab 环境，入口、有限检查及待训练行为验收见 [mjlab 串联腿迁移](../mjlab-chuanliantui.md)。下文 Python 3.8 / Isaac Gym 约定适用于旧任务；新包不得安装进旧环境。

稳站v2奖励/估速诊断：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/diagnose_stand_v2_rewards_velocity.py --checkpoint <模型> --num-envs 64 --seconds 40 --seed 42 --velocity-mode baseline --output artifacts/<新目录>`；记录实际加权奖励、动作前真/估速度与动作后电机状态，并核对参数不变。`oracle-after10`仅对10秒后已站稳者用真速度替换三维估速输入；`oracle-all-after10`对全部存活环境替换，均为诊断，不可用于训练或部署结论。保留首回合冻结与已结束环境复位规则。对照须核对初态，并将干预前已经不同的样本分开；[model4999实证及限制](../../artifacts/stand-v2-reward-velocity-20261027/README.md)。

## mjlab 检查

完整输出采集（H7OUT001，离线，不打开串口）的正式有限检查：

```bash
# CC需指向已有宿主C编译器；缺少cc时显式设置，不安装到Python环境。
CC=<已有宿主gcc路径> ARM_CC=/usr/bin/arm-none-eabi-gcc \
  mjlab_training/.venv/bin/python scripts/agent/check_h7_output_capture.py \
  --build-h7 --out artifacts/<新的完整采集检查目录>
```

检查实际C队列/USB/驱动、652组插桩前后报文字节与HAL返回、模拟500/1000Hz输出/40ms后台停顿、会话CSV和损坏流拒绝，可选大小机编译链接。需要保留本次`artifacts/h7-output-capture-20261021/source-baseline/`作为原驱动对照。结果`final/report.json`、USB/UART回归和未验证的板端时序范围见[完整输出接口](../interfaces/20261021-h7-output-capture-v1.md)。软件通过不代表2ms真实控制截止时间或物理输入已验证。

2026-10-21轮端试验补偿：`--h7-root deployment/H7_RL_ros --build-h7`增加实际`wheel_response.c`与`rl_torque.c`的108组宿主C/Python对照，验证大机已确认补偿、小机原控制、腿部保持、非有限结果拒绝及最终限幅。大小机各107来源编译链接通过，报告`artifacts/real2sim-wheel-response-h7-20261021/check-final.json`；新旧版本/元数据及动力学拒绝检查见同目录`check-profile-complete.json`，旧完整动力学回归见`check-legacy-dynamics.json`。槽158控制版本、156/157复位计数及159CRC均覆盖。软件通过不等于行为通过：同目录`sim-approved-coefficients.json`的30秒闭环复核仍持续移动且跟踪失败，未烧录，详见项目实验记录顶部。

2026-10-21：选择`--h7-root deployment/H7_RL_ros`时，同一入口另检查单轮台架真实C状态机、RC解码、许可与最终分发；最新源码为准备无人扶持零速采集恢复原RL投入，正常LQR档位及台架独占保留。检查覆盖独占切档、释放后RL投入、超时/超速/失联/反馈过期/时间回绕/CAN提交失败。不会控制真机；`--build-h7`同时编译大小机。最新软件和构建结果分别为`artifacts/real2sim-zero-stand-20261021/check-usb-software.json`与`check-arm-build.json`，板端未更新。历史暂停RL结果仍保留在`artifacts/wheel-bench-20261021/final/check.json`。

H7→Jetson只读镜像的软件检查（不训练、不连接或控制真机）：

实测输入的 MuJoCo 动力学对照另有 `--dynamics-only`，不依赖宿主 C 或 ARM 编译器：

```bash
mjlab_training/.venv/bin/python scripts/agent/check_real2sim_huananhu.py \
  --dynamics-only --out artifacts/<新名称>/check-dynamics.json
# 对完整运行：核对全部保存终点／统计，并抽样重新步进。
mjlab_training/.venv/bin/python scripts/agent/check_real2sim_huananhu.py \
  --dynamics-only --dynamics-run artifacts/<对照目录> \
  --out artifacts/<新名称>/check-dynamics-results.json
```

它检查原始流 CRC、时间回绕、重启、缺口、断连、过期与截断拒绝，固定台架的姿态烘焙／零漂移、`mj_step` 与单轮方程一致性、预测不读取后续实测，以及独立候选 XML 只允许两个轮关节的附加惯量／阻尼／干摩擦变化。完整动力学运行命令和输入假设见 [USB 镜像接口的离线动力学对照](../interfaces/20261020-h7-ros-usb-mirror.md#离线mujoco动力学对照)。这些是有限软件检查，不构成真实力矩、物理标定或 gap 消除的验收。

轮端惯量分配诊断使用独立入口，不改变上述六字段候选的约束：

```bash
mjlab_training/.venv/bin/python sim2sim/real2sim_coupling_huananhu.py \
  --dynamics-run artifacts/<完整动力学对照目录> --out artifacts/<新分配对照目录>
mjlab_training/.venv/bin/python scripts/agent/check_real2sim_huananhu.py \
  --coupling-only --coupling-run artifacts/<分配对照目录> \
  --out artifacts/<新名称>/check-coupling.json
```

诊断保持固定台架总轴惯量和损耗，只在独立XML中按0/5/10/15%将附加惯量移到轮本体，拒绝超出既有质量/包络半径必要上界的值。使用旧运行中逐位相同的初态，不重新拟合；覆盖台架响应不变、机身耦合变化、全部保存终点/统计、旧基线复现及重构数值敏感性。必要几何上界不等于实测或物理分布可实现性证明。已有完整验证见`artifacts/real2sim-coupling-20261021/check-fixed-initials.json`。

CPU入口的`--model`／默认关闭的`--wheel-friction-compensation`用于轮端模型和实验控制器对照，运行命令见上述接口。原`scripts/agent/check_multimode_cpu_sim2sim.py`检查训练默认语义保持，并增加补偿过零、正反速度、腿控制保持、统一±3.9Nm限幅与趴地保持检查。最新结果`artifacts/real2sim-wheel-closed-loop-20261021/check-cpu-compensation.json`；它不验证实机补偿或零位移。

```bash
# 训练仓本机（需宿主cc及已有Matplotlib）：主工程默认UART软件回归。
mjlab_training/.venv/bin/python scripts/agent/check_real2sim_huananhu.py \
  --out artifacts/<新名称>/check-software.json
# USB镜像工程：真实打包/发送源码、机器映射、模型及录制/离线分析检查；CDC由替身模拟。
mjlab_training/.venv/bin/python scripts/agent/check_real2sim_huananhu.py \
  --h7-root deployment/H7_RL_ros --out artifacts/<新名称>/check-real2sim-recording.json
# 训练仓：从当前eIDE实际存在的来源编译、链接大小机；需要完整ARM GCC/Newlib
ARM_CC=/usr/bin/arm-none-eabi-gcc mjlab_training/.venv/bin/python \
  scripts/agent/check_real2sim_huananhu.py --build-only --out artifacts/<新名称>/check-h7-build.json
# 同一入口独立编译USB工程，不能拿主工程结果代替。
ARM_CC=/usr/bin/arm-none-eabi-gcc mjlab_training/.venv/bin/python \
  scripts/agent/check_real2sim_huananhu.py --h7-root deployment/H7_RL_ros \
  --build-only --out artifacts/<新名称>/check-usb-build.json
```

镜像不依赖PyTorch，单独依赖见`sim2sim/requirements-real2sim.txt`。`--build-h7`可在同时具备宿主C编译器
和ARM工具链的训练主机合并执行；构建产物全部写到报告目录，不污染eIDE增量构建。过时且不存在的
虚拟来源列入报告，不修改工程清单。检查覆盖CRC／伪尾／分包、序号回绕、DMA忙、单设备失效、200ms
断流，以及模型闭链重构；不验证实际串口、物理极性／零点／IMU轴向或装配分支。操作见
[部署接口](../deployment-contract.md#host-debug-plan)。

只读镜像的模型范围越界采用显示并警告：实际趴地反馈检查覆盖双腿更新、黄色警告、200ms过期变红、回到范围内恢复原色及XML限位保持；闭合失败和坏反馈仍冻结。该显示规则不改变训练或H7控制。

real2sim录制／回放／分析沿用同一检查入口，检查原始字节分包／损坏流、连接事件、单调时间、会话完整／错误状态、模型与机器配置哈希核对，以及离线与直播的姿态／过期／范围警告一致性。完整软件检查会调用离线分析，须使用本机已有Matplotlib的mjlab环境；分析生成会话`analysis/trajectory.csv`、`report.json`、`signals.png`。Jetson基础依赖不增加Matplotlib，直播／录制／回放无需绘图库，完整检查不要求在Jetson执行。

USB源码镜像v2扩为160路／644字节，增加实际配对obs39／action6、命令、PD目标／反馈、门控后力矩请求和来源接收年龄；旧v1录制仍可读，缺失新列留空。新增检查覆盖v1／v2混流、真实C扩展打包与644字节缓冲保护，以及扩展字段录制／回放／CSV一致性。v2源码与板端版本、软件及台架验收须分别记录，见[接口记录](../interfaces/20261020-h7-ros-usb-mirror.md#只读对照信号v2)。
本轮正式USB软件检查见`artifacts/real2sim-record-replay-20261020/check-usb-final-v3.json`，结果通过；新增窗口初始化不占采样预算、关窗耗时不误报最终来源过期的回归。默认主工程UART最终回归见同目录`check-uart-final.json`，通过。
Jetson最终1072有效帧完整会话的CRC拒绝／序号缺口／重复／重启／乱序均0，原速离线回放1072帧、最终姿态最大差0，本机分析三文件已生成；证据见同目录`jetson-capture-final.json`、`jetson-replay-final.json`和`verification.json`。首次774帧采样预算／收尾状态问题已修，旧产物保留。实收会话不作为受控静止或逐关节方向／零点验收，动态配合仍待台架；不能用软件和USB会话通过替代物理映射验收。操作命令、证据与已知趴地超限记录见[USB镜像接口](../interfaces/20261020-h7-ros-usb-mirror.md#real2sim录制与离线诊断)。

倒地恢复机制检查（不训练）：训练配置不主动施加绕轴击倒，手动冲量只在机制检查中使用。

```bash
# 未注册主动击倒事件，恢复窗口仍存在；12秒有限步进
mjlab_training/.venv/bin/python scripts/agent/check_multimode_recovery.py --training-config
# 手动击倒后恢复通道重启、恢复期接触奖励门控与窗口内不提前终止
mjlab_training/.venv/bin/python scripts/agent/check_multimode_recovery.py
```

最终平衡 v3 的判据检查及既有轨迹离线重算（不运行物理、不重跑策略、不改旧产物）：

```bash
# NumPy / Torch 数值路径，覆盖倾角、承重、接触、角速度、时长及提前终止
mjlab_training/.venv/bin/python scripts/agent/check_multimode_final_balance.py
# 仅重算已录下的 7 个样本；输出目录须使用新名称
mjlab_training/.venv/bin/python scripts/agent/check_multimode_final_balance.py \
  --recording-dir logs/mjlab/evaluation/kd-failure-videos-4850 \
  --output artifacts/<新重算名>/result.json
```

新判据看结束前连续 1 秒平衡，允许滚动、转向与少量腿部接触，仍检查双轮主要承重、机身不撑地与横滚/俯仰稳定；历史 `last5_*` 保留原义。离线重算仅覆盖已录制样本，不替代完整模型配对、独立 sim2sim 或真机验收，标准见[当前验收 v3](../interfaces/chuanliantui-wheel-balance-acceptance-v2.md)。

在仓库根运行，不启动训练或 TensorBoard，不写根 `logs/`：

```bash
# 有限检查：加载 checkpoint，核对 actor 39／critic 82／动作 6，数值有限
scripts/mjlab.sh check --num-envs 2 --steps 16 \
  --checkpoint artifacts/multimode-v1-replay-20261014/models/model_3023.pt
# 基线回放：64 个强制地面初态、20 秒
scripts/mjlab.sh eval --checkpoint artifacts/multimode-v1-replay-20261014/models/model_3023.pt \
  --num-envs 64 --seeds 42 --start-mode force --terrain flat --output /tmp/mjlab-eval/result.json
```

平地速度／角度跟踪测量与判据检查（不训练；口径见[跟踪测量](../interfaces/chuanliantui-huananhu-multimode-tracking-v1.md)）：

```bash
# 判据算术、边界与容差一致性的离线检查（不建环境、不加载模型、不跑物理）
mjlab_training/.venv/bin/python scripts/agent/check_multimode_tracking.py
# 平地固定命令网格的跟踪测量（play 配置、每环境单回合、12 s、前 2 s 为稳定窗口）
# 脚本每步断言 actor 命令槽等于网格：中途抛 AssertionError 表示命令未被钉住，不是策略结论
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_tracking.py \
  --checkpoint <model_*.pt> --output artifacts/<新名称>/result.json
```

静止站立腿角运动测量与 CPU 入口一致性检查（不训练；2026-10-17 站立腿摆修复的证据工具）：

```bash
# 入口一致性与恢复高度检查（静态 + 有限步力矩／接触／模式探针，不做长时行为验收；不一致即非零退出）
mjlab_training/.venv/bin/python scripts/agent/check_multimode_cpu_sim2sim.py \
  --out artifacts/<新名称>/check-cpu-entry.json
# 静止站立腿角运动测量（50 Hz 控制拍；诊断性测量，不设通过阈值）
# --backend cpu 用独立 CPU sim2sim 入口；--backend mjlab 用训练后端（play 配置、GPU），
# --reset stance 复刻 CPU 入口初态，--reset bank 用训练姿态库初态
mjlab_training/.venv/bin/python scripts/agent/measure_multimode_leg_motion.py \
  --backend cpu --checkpoint <model_*.pt> --output artifacts/<新名称>/leg-motion-cpu.json
mjlab_training/.venv/bin/python scripts/agent/measure_multimode_leg_motion.py \
  --backend mjlab --checkpoint <model_*.pt> --device cuda:0 --output artifacts/<新名称>/leg-motion-mjlab.json
```

一致性检查覆盖：入口与训练模型 qpos0 及闭链站立解一致；观测腿槽与腿 PD 目标以 `q_nominal` 为中心、
限幅以 `qpos0` 为中心（q＝nominal、零动作时腿力矩为零）；腿增益 0.5／60／2、轮增益 0.1／0、
轮速 `clip(10a, ±150)`、力矩限幅 40／3.9；气弹簧逐点一致；动作顺序与观测排列、NORMAL 模式／标志、
报告容差与跟踪测量脚本一致；机器人 geom 优先级（训练场景 `priority=1` 补丁）与机器人-地面接触摩擦等于训练值。腿角运动测量记录六电机 q／qd、原始动作、观测腿槽与机身状态，并给出
腿摆峰峰值、主导频率与观测偏置；它只覆盖单模型、单命令、单次运行的读数，不是验收。

2026-10-18新增CPU恢复高度检查：使用两腿机体系 `L*cos(theta)` 均值，覆盖整机平移／旋转不变性、
左右腿不对称，以及抬高机身但保持折腿时不误判恢复。该高度条件有意采用H7量的定义，
不再与训练的世界坐标高度相同；`height_m` 与原跟踪／静站判据保留原义。
检查通过不表示H7腿角标定已验证，也不表示完整H7恢复时序或真实恢复行为验收通过。

恢复高度改动前后的CPU配对诊断（相同模型、初态和命令；默认7组normal命令与16个固定seed地面初态，
每侧20秒；`--before`必须是改动前入口的完整源码快照）：

```bash
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_recovery_height.py \
  --before <改动前入口.py> --checkpoint <model_*.pt> \
  --output artifacts/<新名称>/result.json
```

输出逐样本模式切换时间、动作／位置／速度的最大差、末段连续1秒双轮平衡及原成功样本是否丢失，
同时保存初态与逐步轨迹。末段平衡使用v3.1门槛并读取原生MuJoCo接触力；该初态分布和运行流程
不等于训练侧完整验收，也不代替速度／高度跟踪测量。当前实测结论见[项目实验记录](project-log.md)。

包线边界与引擎差异的审计跟进测量（不训练；2026-10-17 接触参数审计的证据工具）：

```bash
# 训练引擎（warp）在训练接触材料随机化下的包线复测；--material randomized 复原训练抽签
# （轮滑动 min(U(.5,1.2), U(.4,1.))、基座 U(.01,.1)、训练 solref），solref/xml 为对照，
# --material pinned --wheel-slides 为逐环境固定轮摩擦的阈值扫描；--envs-per-combo N 即起点分布扫描
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_envelope_friction.py \
  --checkpoint <model_*.pt> --output artifacts/<新名称>/grid-randomized.json \
  --material randomized --envs-per-combo 8
# CPU 入口包线诊断：--mode starts 训练姿态库 normal 分支起点分布（--friction training 叠加
# 训练接触抽签）；--mode sweep 轮滑动摩擦扫描；--mode trace 与训练引擎的 50 Hz 轨迹分叉
# （--twin-perturb EPS 额外测入口自身对初值扰动的敏感度）
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_cpu_envelope.py \
  --mode starts --checkpoint <model_*.pt> --starts 64 \
  --output artifacts/<新名称>/cpu-starts.json
```

迁移一致性检查（用冻结快照；`--snapshot` 缺省时逐候选目录核对 23 个模块哈希后自动选定；
`--gpu` 会加载并完整续训读取 model_3023，但不调用 learn）：

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_mjlab_multimode.py \
  --gpu --out /tmp/mjlab-multimode-check.json
```

它检查：每个模块等于「冻结快照 + 已记录偏离」逐字节重构的结果（未记录的差异一律失败），偏离表在
`scripts/agent/multimode_snapshot_deviations.json`（源偏离 6 个模块；配置偏离 train 0、play=False 8、
play=True 5，均来自 2026-10-15/16 恢复工作与 2026-10-17 平地跟踪阶段）；XML 与姿态库哈希及缓存签名；
环境和 PPO 配置与快照工厂的输出在该偏离表之外完全相同；没有代码再引用已删除的模块；另外检查续训
课程进度 7525、保存后读回参数不变、导出用 actor 与 runner 策略输出一致、39/82 维、有限步进。

`eval` 在**平地协议**下的新参考基线：`model_3023` 与 `model_4850` 签名同为 `4a0b867588ff…`，
严格起立 64/64、存活 64/64、末态双轮承重 64/64、数值失败 0（`logs/mjlab/evaluation/model3023-flat-gpu64/`、`model4850-flat-gpu64/`）；
平地跟踪续训产物 `model_5849`（SHA256 `d4be0bc9…`）在同一套初态上严格起立 63/64、存活 63/64（`env 27` 在 2.64 s 物理终止），产物在 `logs/mjlab/evaluation/model5849-flat-gpu64/`。历史混合地形基线 `458f15a3…`（平子集 31/64）只作历史记录，不能与新签名配对。以上不等于行为验收，
不替代双轮平衡验收 v3、平地跟踪测量或真机验证。

## 环境

项目源码为纯 Python 包，`pip install -e .` 一次即可；Isaac Gym 依赖的
`gymtorch` 在导入时可能通过 PyTorch JIT 编译 C++ 扩展，需要可用的主机 C++ 编译器。

本仓库必须用**装了 Isaac Gym 的 Python 3.8 conda 环境**，不能用系统 python。本机该环境的路径见 [COMMANDS.md](../../COMMANDS.md) 开头，下文 `python` 均指它。

按 `COMMANDS.md` 完整执行 `conda activate`，使编译器 `CXX` 和动态库路径的激活脚本生效。
只指定环境内的 Python 绝对路径不等于激活环境；`which c++` 返回非零时先核对这一点。

已验证可用的版本组合：Python 3.8.20 / torch 2.1.0（CUDA 可用）/ mujoco 3.2.2 / Isaac Gym Preview 4。

MuJoCo 固定版本安装命令（保留现有满足依赖的 NumPy/Torch）：

```bash
python -m pip install --upgrade --upgrade-strategy only-if-needed 'mujoco==3.2.2'
```

自检一句话：

```bash
<你的环境>/bin/python -c "import isaacgym; import torch; import mujoco; print(mujoco.__version__)"
```

`import isaacgym` 必须在 `import torch` 之前，否则报错。

## 正式流程

策略、控制和机器人资产改动按对应任务执行训练、Isaac 回放、MuJoCo 行为验证，区分“策略未训练好”和“sim2sim 有差异”。以下为 imcawl 示例，chuanliantui 使用 [COMMANDS.md](../../COMMANDS.md) 的对应命令。纯文档或本机路径示例修改按 [协作指南](../../CONTRIBUTING.md) 检查差异、链接和配置语法，并明确未运行行为验证。

每项结果记录实际代码 SHA、工作区差异、模型版本和执行证据；模型交付使用 [交付模板](../templates/model-handoff.md)。

### 1. 训练

```bash
python wheel_legged_gym/scripts/train.py --task=mini_wheel_legged --headless

# 续训
python wheel_legged_gym/scripts/train.py --task=mini_wheel_legged --headless \
    --resume --load_run Jul20_12-42-47_
```

常用参数：`--max_iterations` / `--num_envs`（显存不够调小）/ `--checkpoint` / `--run_name`。
续训时 `--max_iterations` 表示本次**新增**轮数，例如从 `model_3000.pt` 续训 3000 轮会保存到 `model_6000.pt`，应使用新的 run 名保留原权重。
runner 恢复 PPO 优化器后会同步自适应学习率；新 checkpoint 也保存 encoder 优化器。旧 checkpoint 不含 encoder 优化器状态时会明确提示，并使用新建的 encoder 优化器，不能视为完整训练状态的逐步重放。
`scripts/agent/check_training_resume.py` 可在上述 Python 3.8 环境做 CPU 续训状态检查（新旧 checkpoint、仅加载权重和无 encoder 分支），不创建仿真，也不替代可见预检、正式训练或行为验收。
输出：`logs/mini_wheel_legged/<日期时间_run_name>/model_*.pt`。
在另一终端监控：`tensorboard --logdir=logs/mini_wheel_legged --port=8080`。

### 2. Isaac 内回放

```bash
python wheel_legged_gym/scripts/play.py --task=mini_wheel_legged
python wheel_legged_gym/scripts/play.py --task=mini_wheel_legged \
    --load_run Jul20_12-42-47_ --checkpoint 6900
```

必须先在 Isaac 里确认策略本身能站能走，再进 sim2sim。

### 3. MuJoCo sim2sim

用完整 checkpoint `model_*.pt`（含 encoder）。导出的 `policy_1.pt` 只有 actor、缺 encoder，**不可用**。

```bash
# 3a 形状自检（不跑仿真，几秒）
python sim2sim/mj_sim2sim.py --selfcheck \
    --checkpoint logs/mini_wheel_legged/Jul21_14-09-24_/model_4000.pt

# 3b 站立（vx=0）
python sim2sim/mj_sim2sim.py --render \
    --checkpoint <ckpt> --cmd_vx 0 --cmd_height 0.30 --init_height 0.30

# 3c 行走
python sim2sim/mj_sim2sim.py --render --checkpoint <ckpt> --cmd_vx 1.0 --init_height 0.30
```

判读终端每 100 策略步的 `x/z/vx/|a|max` 行：`z` 稳定在 `cmd_height` 附近、`vx` 跟上 `cmd_vx`、`|a|max` 若持续接近 100（饱和）说明策略发散。

其他参数见 [COMMANDS.md](../../COMMANDS.md)（`--cmd_yaw` / `--sim_time` / `--no_realtime` / `--friction` / `--teleop` 等）。

### 改了 XML 或 URDF

```bash
python sim2sim/check_model.py     # 碰撞对 + 轮圆柱位置自检
```

### 失稳时的对照组

```bash
python sim2sim/eval_isaac.py --load_run <run> --checkpoint <n> --cmd_vx 0.5
```

Isaac 也失稳 → 策略问题，回去调训练。Isaac 稳 → 引擎 gap，调 MJCF 接触参数或加强域随机化。

#### chuanliantui 30 秒静站与共同推扰

按 [命令速查](../../COMMANDS.md#30-秒静站与共同推扰计划) 先做同 checkpoint、零速度、
0.22 m、150 N/侧气弹簧、摩擦 0.5 的 30 秒无推扰对照，再给两端追加同一
`--push_schedule /tmp/standing-pushes.json`。用
`python sim2sim/standing_push_schedule.py --output /tmp/standing-pushes.json` 生成默认
30 秒、seed 42 的计划；`python scripts/agent/check_standing_push_schedule.py` 仅检查
2 ms 边界、重叠拒绝及计划可复现性，不替代抗扰行为验收。

Isaac 评估自动延长回合期限到至少 `sim_time + 1` 秒，保留失稳终止；训练期限不变。
取消轮目标裁剪的 H7 对照需显式传 `--h7_no_wheel_target_clip`，电机力矩限幅仍保留。
固定计划在两端每 2 ms 向基座质心施加同一世界系力，重置不重播；Isaac 不能同时启用
`--pushes` / `--domain_rand`。轨迹中的真速度和 encoder 估速应按同一推理时刻比较。
100 Hz 推力采样相位不同（MuJoCo 为策略步起点、Isaac 为末个物理子步），不可直接积分
核对；检查 metadata 的 `scheduled_push_impulse_world_ns` 与计划总冲量。
分别报告位移、速度、高度、姿态、重置及推后恢复；原本持续漂移的模型恢复运动状态，
不等于恢复原地稳站。命令和静态检查通过不代表本轮行为验收通过。

## 没有单元测试

### chuanliantui 气弹簧

`python scripts/agent/check_chuanliantui_gas_spring.py` 在 CPU 对比 Torch 的姿态相关
膝力矩、MuJoCo tendon 广义力和长度差分，检查左右符号、零推力、刚体姿态旋转、
电机/被动力分离与随机推力合成。它是数值和接口检查，不替代行为回放。
`check_chuanliantui_train_proxy.py` 同时检查代理的六电机、两气弹簧及 CAD 安装点。
气弹簧默认 150 N/侧；历史无弹簧行为对照需在两种回放中显式传 `--gas_spring_force 0`。

`python scripts/agent/check_gas_spring_asset_geometry.py` 检查从原始/训练 URDF
定位件读取端点、任意关节轴及安装 RPY，使用独立 MuJoCo tendon 和长度差分作
数值参照；覆盖安装点变更、左右改名、joint/link 不同名、两侧不同推力及无效资产。
CUDA 可用时同时检查批量 GPU 数学路径，不创建 Isaac 物理仿真。
2026-10-08 的执行范围与结果见[气弹簧资产几何验证](../diagnostics/20261008-gas-spring-asset-geometry.md)。

### chuanliantui ONNX 导出检查

`scripts/agent/export_chuanliantui_onnx.py` 从完整 25 维 checkpoint 导出 encoder + actor，执行 ONNX checker 和参考执行器/PyTorch 数值比较。可用 `--fixed-batch --opset 13` 生成面向 STM32H723 的静态 batch=1 模型；默认仍为动态 batch、opset 17。运行方式和 Cube.AI 验证步骤见 [ONNX 导出说明](../../ONNX导出说明.md)。该检查不替代训练、回放、sim2sim 或板端验证。

本仓库没有 pytest、没有 CI。「测试」= `--selfcheck` + 上面三步人眼判读。不要声称跑过单测。新增可自动判读的检查请放 `scripts/agent/`。


## 完整输入会话校准的有限离线核验

本次独立worktree新增`scripts/agent/check_h7_capture_calibration.py`，检查原始会话哈希/完整性/划分、实际优化预算、候选编译保护字段、自由积分可复现/不依赖未来实测速度及±1%参数敏感性；有旧RL输出时复核保存终点并每模型重新积分一窗。只代表这些有限检查，不是整机行为验收或真实参数辨识。本次报告/命令见[固定300ms完整输入校准](../diagnostics/20261021-fixed300-complete-input-calibration.md)。已有产物可只读复核：

```bash
mjlab_training/.venv/bin/python scripts/agent/check_h7_capture_calibration.py --data-root artifacts/fixed-h7-fixed300-data-20261021 --fit-root artifacts/fixed-h7-fixed300-data-20261021/fit-budget80 --out artifacts/fixed-h7-fixed300-data-20261021/fit-budget80/recheck.json
```


## 连续驱动/滑行候选的有限检查

新增`scripts/agent/check_h7_drive_response.py`，仅离线核对原始固件/机器/正式模型哈希、候选物理字段与非轮执行器不变、同一惯量、实际预算、完整悬空轮验证曲线复算/重复一致及±1%驱动参数敏感性。user-gain原生回调与独立Python逐步实现对照由`sim2sim/probe_h7_drive_response.py`先执行；源码`sim2sim/h7_drive_response.c`仅为离线MuJoCo回调，不接H7。候选XML需要该回调，单独XML不构成完整模型。范围、失败导出留档、命令与本轮悬空轮结果（整机未验证）见[检验报告](../diagnostics/20261021-h7-drive-coast-response.md)。


## 当前悬空轮端候选回放

用户选用轮速拟合最好的候选；独立选择记录为`artifacts/h7-wheel-best-20261021/selection.json`。`replay_selected.py`读取该选择，在悬空夹具中自由积分，不重新拟合、不运行RL或接硬件；结果必须复现选定RMSE。需要XML与同一增益回调配套。联合拟合仍由现有`check_h7_drive_response.py`有限核验；本次范围、局部退化和复现见[候选选择报告](../diagnostics/20261021-h7-wheel-best-selection.md)。


## 共享华南虎XML来源检查

check_multimode_cpu_sim2sim.py现同时验证训练/CPU/real2sim引用同一XML、旧模型路径为兼容链接且CPU姿态库位置保留。2026-10-21来源合并前后30项编译物理字段完全一致，有限检查通过；未执行训练、硬件或新行为验收。证据及复现见[共享XML说明](../diagnostics/20261021-shared-huananhu-xml.md)。


## 已接入共享XML的轮端响应

新增`scripts/agent/check_shared_wheel_response.py`核对当前XML等于选定轮端候选、部件/非轮/硬件保护项不变、训练与CPU力矩计算等价且G只乘一次、原录制严格归档模型、活动XML连续回放与候选一致，并计算2ms步长误差。CPU加载使用已有C回调的源码/版本缓存；没有cc/gcc时设置CC。已运行GPU2环境16步检查，仅检查有限数值/接口，不代表行为验收，不调用学习。命令和范围见[接入报告](../diagnostics/20261021-h7-wheel-response-integration.md)。

### 已录H7 RL故障时序的离线复核

`mjlab_training/.venv/bin/python scripts/agent/analyze_h7_rl_capture.py --session <完整会话目录> --seconds 30 --checkpoint <对应model_*.pt> --out <尚不存在的分析目录>` 验证原始流哈希/元数据后，输出首30秒档位、在线/故障、来源年龄与请求曲线，并复算已有PD快照和至多64个已录策略输入。可选`--comparison-checkpoint`以同一实测输入核对其他权重。只用于故障时序与算术检查，不打开串口、不生成另一套驱动输入，不验证真实力矩、位移、计圈真实性或故障物理根因；当前覆盖见[最新前30秒分析](../diagnostics/20261021-model6848-first30-fault-analysis.md)。

### RL初始角度分支的定向回归

`CC=<已有宿主C编译器> mjlab_training/.venv/bin/python scripts/agent/check_h7_rl_angle_branch.py --session <完整v2会话> --out <新目录>` 使用实际H7映射/观测/PD源码及现有大机配置，检查首次分支锁存、多圈/跨界连续、失效保持、obs/PD同源及前30秒实录角度映射。只做离线软件验证，不证明真实姿态、力矩或失联原因；大小机完整固件链接另见[候选验证记录](../interfaces/20261021-h7-rl-angle-branch-fix.md)。

## H7LAT001新增时序旁带的有限软件检查

`check_h7_latency_capture.py`编译真实采集/CAN及USB源码，检查逐buffer配对、复用/失败/IRQ、队列/CRC/分包、首次PD配对/旧动作保持，并复用原生产者回归；`--build-h7`在工程外分别链接大小机。宿主C编译器和完整ARM工具链需先设置CC/ARM_CC；不连接硬件、不证明板端开销/调度或行为。输出目录用新名称。

```bash
CC="${CC:?请设置已有宿主C编译器}" ARM_CC="${ARM_CC:?请设置完整ARM工具链}" \
  mjlab_training/.venv/bin/python scripts/agent/check_h7_latency_capture.py \
  --build-h7 --out artifacts/h7-timing-check-new
```

当前候选、验证与新实测步骤见[时序采集接口](../interfaces/20261022-h7-latency-capture-v1.md)。旧录制没有H7LAT001时解码明确拒绝，不能回填新增测点。
