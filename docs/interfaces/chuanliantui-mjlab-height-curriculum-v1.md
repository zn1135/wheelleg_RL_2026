# A2500高度课程两组续训接口

2026-10-12用户授权从A2500独立续训两组，各500次PPO更新、4096环境、seed42，远端GPU并行；预算累计3000轮，到此停止，不扩组、不追加。源SHA256仅接受 `706c8387cba4815b491be3f5914a5951881c2966edabbd5fbb90d0da9c9d3dbd`，用原 `recovery_factors.validate_checkpoint(saved,'baseline')` 验证。对照是已有A2500，不另训练基线。

## 配方与兼容性

新入口 `ct_mjlab.experiments.height_curriculum`，接口 `chuanliantui-mjlab-height-curriculum-v1:<case>`。不修改历史模块、注册入口或默认V14。

| case | 相对A的变更 |
|---|---|
| curriculum | 零速度命令、整批高度课程、宽高度指数奖励 |
| curriculum_knee | 上述全部，另启用原被动膝限位公式，权重−0.5 |

基础是 `recovery_factors.make_env_cfg('baseline',...)`。其余奖励、动作、PD、观测、初态、随机化、终止、气弹簧、资产及PPO参数保留；actor35、critic78、动作四腿在前两轮在后。旧V14实际奖励管理器不注册权重0的knee项，第二组显式添加同名项，调用 `reference_training.huananhu_rewards.RewardState` 的原公式：两膝 lf1/rf1，近边界距离、0.15 rad软区、1.5截断后的平方和。第一组没有膝奖励贡献。

`zero_velocity_commands=True` 同时关闭特殊高速、航向反馈和地面初态前1.5秒速度覆盖；速度奖励原5～10 cm高度门控不变。宽指数为 `exp(-abs(height_above_ground-command_height)/0.12)`，权重+1；高度平方公式与−1权重不变。高度取A原口径，不另换质心或轮轴高度。

## 整批课程

L=0.21 m，每级+0.01至0.34；命令重采样取 `U(max(0.20,L-0.02),L)`，最终L=0.34时取 `U(0.27,0.34)`。级别变化不强制复位或立即重采样，新的范围从后续原5～15秒重采样生效。

升级同时满足：当前级已完成至少50次实际PPO更新、至少2/3非数值失败环境满足轮端承重＋近水平＋目标高度±3 cm、此比例连续保持2秒。连续时间独立于50更新门槛积累，比例不足清零；全部数值失败时比例为0。轮端承重和近水平直接复用 `wheel_balance_assessment.components` 的 `wheel_load_bearing & near_level & target_height`，不是将完整静止平衡门槛偷偷加入训练课程。

计数在原 `alg.update()` 成功返回后递增，不用rollout步数估算已完成更新。课程在每次控制转移的原终止／奖励阶段后、自动复位前读取非数值失败分母，记录全批等级与达标比例；不改终止函数、接触传感器或物理步进。每次升级打印 `HEIGHT_LEVEL`，含前后等级、控制步、累计完成更新和达标比例。每控制步日志：`HeightCurriculum/level_m`、`HeightCurriculum/ready_fraction`、`HeightCurriculum/completed_updates`。

L、当前等级开始更新数、累计更新数、连续达标时间和升级历史完整保存并恢复。0.21到0.34需要13次升级，仅50更新最低驻留就需650次；本批500更新预算不保证达到上限，结束报告实际等级，不能自动延长预算。

## 完整模型迁移与预算

actor、critic、Adam状态逐张量与源相等，学习率从Adam恢复为1e-5，保留原自适应更新、动作探索std和熵系数。累计计数从2500开始，课程L=0.21。保存 `model_resume_initial.pt` 及逐项迁移核验；自己的checkpoint仅接受相同case和完整manifest，并恢复课程。

每100轮保存，评估 `model_2600.pt`（累计2601）和最终 `model_2999.pt`（累计3000）。专用训练入口只允许2500+500，不自动重启失败训练、不隐式跨case、不增加预算。新接口未注册进历史CLI，应用专用脚本，不可当作部署已验收策略。

## 训练前门禁与评估

两组各16和4096环境各64步检查全部观测／奖励有限、数值失败0；同一真实状态逐项奖励隔离，以及4096环境的完整迁移和自身checkpoint重载。未全部通过不启动学习。

评估使用相同64份普通／地面初态、20秒、六高度命令、确定性actor、零速度、名义物理、无噪声／延迟。课程在评估中不升级；物理/观测沿用A baseline回放配置，手动命令。保存末5秒高度、共同成功样本斜率、完整验收v2平衡与高度跟踪数、原生膝 `mjCNSTR_LIMIT_JOINT` 激活比例及JᵀF反力矩。

0.30 m用 `paired_acceptance` 对A普通64／地面60逐样本否决：任何原成功样本丢失即“退化，验收不通过”，新增成功不抵消。每个时间点保存普通／地面样本0的20秒视频，显示实际和目标高度；没有六组拼屏。

冻结源码完整哈希快照及新run保留。仅增加新模型配方；H7部署接口/代码未同步，本任务没有ONNX、板端或真机验收。单seed、64初态仅说明本批趋势。运行记录见 [训练与评估记录](../diagnostics/20261012-mjlab-height-curriculum-training.md)。
