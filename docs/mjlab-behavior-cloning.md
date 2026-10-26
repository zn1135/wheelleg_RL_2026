# 学生PD动作示范、行为克隆与PPO初始化

当前轮角39维BC已接通[SMP PPO v4初始化和恢复](interfaces/20261025-wheel39-smp-ppo-v4.md)：入口按观测定义自动配置，旧零轮角39维继续v3。仅复制actor，critic和PPO优化器新建；本轮零更新检查，不追加BC或PPO训练。后续以任务奖励改善停车、SMP作辅助，不要求BC完全复制LQR停车后才能接PPO。检查、回放和视频见[本轮报告](../artifacts/ppo-wheel39-interface-20261025-v1/README.md)，下方“尚未接通”为历史状态。

最新零更新停车诊断：学生独立30秒0/156，LQR经学生PD接管17/156停车通过，补34000帧实际纠偏，未训练。用户目标是PPO超过LQR，后续不以BC完全复制停车为硬门槛；先核对轮角39维PPO合同和任务奖励，再另定更新预算。[诊断和视频](../artifacts/bc-aligned-braking-diagnostic-20261025-v1/README.md)。下方为此前阶段记录。

共享参考轮角39维BC已本地新增5000次：原40组起立／平衡40/40，停车0/40；末5秒速度均值0.0109m/s、漂移均值5.17cm。只留阶段候选，预算用完，不自动接PPO或SMP。[本轮报告与视频](../artifacts/bc-aligned-reference-20261025-v1/README.md)、[模型记录](model-deliveries/20261025-bc-aligned-reference-local5000.md)。下方为此前阶段记录。

已对齐轮角清零与教师LQR位置重锁，使用共享站稳确认事件；新旧参考标签禁止混用，未训练新策略。[接口](interfaces/20261025-wheel-lqr-position-reference.md)、[报告](../artifacts/wheel-lqr-reference-aligned-20261025-v1/README.md)。

已接通首次站稳清零v2，旧v1模型和数据保留原规则；新版尚未训练。见[接口](interfaces/20261025-wheel-angle-handover-reset.md)、[检查报告](../artifacts/wheel-angle-handover-reset-20261025-v1/README.md)。

本轮旧技能保留已新增本地5000次：最终起立40／平衡29／停车0，退化不采用；已逐个验收本轮10个保存点。新增4500次版本保住原40组平衡及平滑，停车0/40；仅记为阶段候选，不自动导出、部署或接PPO。原40组已用于模型选择，停车尚未通过，SMP／PPO未启动。[方法](interfaces/20261025-bc-skill-preservation.md)、[报告及视频](../artifacts/bc-preserved-wheel39-train-20261025-v1/README.md)。

后续零更新诊断：旧稳定模型在内存忽略两轮角后，原40组起立／平衡40/40、停车0/40；新轮角10000的倾角轮端局部反馈仅为旧稳定模型约18.3%。未生成新模型或启动训练，不改变新模型此前失败结论。[证据与边界](../artifacts/bc-wheel-angle39-feedback-20261025-v1/README.md)。

最新诊断及纠偏新增本地5000次已完成，轮角39累计10000；原40组独立起立降至0/40、最终平衡／停车仍0/40，退化拒绝采用。固定验证动作拟合改善没有带来闭环稳定，教师先站起再交接也失败。未重训SMP或启动PPO，不替换旧模型。[本轮报告](../artifacts/bc-wheel-angle39-correction-train-20261025-v1/README.md)。下文保留此前阶段记录。

当前新增轮角39／82／6接口已接通教师采集、BC及独立回放，原索引14、15输入左右轮连续角位移，格式与旧占位为零39维严格区分。本地全新网络5000次已完成，起立2/40、平衡0/40、停车0/40，行为失败，不替换旧模型或自动接PPO。[本轮报告与视频](../artifacts/bc-wheel-angle39-train-20261025-v1/README.md)。下方40／41维内容保留为历史实验；默认PPO／SMP PPO仍为原接口。

当前新BC使用41维actor／84维critic／6维PD动作：在原39维后保留通用轮式里程，再追加教师同拍的位置参考误差。学生独立运行同一只读估计器，标签和观测共用同一拍缓存；不混入教师动作。[41维定义、更新／重锁时机和部署边界](interfaces/20261025-bc-shared-reference.md)。普通PPO／SMP PPO及H7仍39／82维。

`collect_reference_bc.py`重采原教师示范、首轮失败学生的旁路查询，以及累计10000／15000次学生运行后教师接管两类纠偏。缺少原始参考状态的历史纠偏不事后补造；`prepare_reference_bc.py`检查同拍递推、真实执行归属、原39／10组隔离和完整来源。旁路查询只作动作纠偏，不能作为SMP成功轨迹；实际教师接管仍须通过完整筛选。

新训练显式`--init <旧模型> --expand-observations`保留旧39／40维所有参数，仅首层新增输入列置0，使用新Adam和新预算。支持迁移不表示已经学会利用新增输入；先做零更新独立回放，再按单独授权训练。未显式选择迁移仍拒绝维度不匹配。41维不能直接送入当前39维PPO初始化或板端网络。

本轮已在本地从旧累计15000次模型新增5000次，累计20000；原40组起立19、最终平衡15、停车0，判退化，不替换旧模型。参考／实际动作核对通过不代表行为成功。[本轮完整记录和视频](../artifacts/bc-reference41-train-20261025-v1/README.md)。

## 历史40维版本与命令

当前BC输入为40维actor／83维critic／6维原始PD动作：在原39／82维末尾追加通用有符号轮式里程，回合复位锁定编码器起点，起立交接、停车、换命令均不清零。它不是停车目标误差，也不是教师内部位置积分。2ms物理／PD、100Hz策略、2／4ms延迟和0.2秒启动门控保持。[新接口与验证](interfaces/20261025-bc-wheel-odometry.md)。旧39维模型继续按原数据回放；普通PPO／SMP PPO及H7仍为39维，40维策略不能直接混用。

统一清单在`artifacts/bc-wheel-reference-20261025-v2`：171条起立、69＋121条教师实际接管段，656000训练帧／66000验证帧。保留原前39维、动作和验证划分，只补入可从原始编码器准确重算的里程。旧180212帧旁路查询留档，排除出新清单。教师正常起立交接或实测支撑后的接管重锁一次，随后连续控制；该标签参考规则不改变轮式里程的回合起点。不能据此保证新增输入完全消除教师隐藏状态。

新训练入口要求统一参考版本，拒绝用旧39维actor初始化40维，不静默补权重。40维从头5000次已按用户预算在本地完成；原40组留出起立9/40、最终平衡5/40、综合停车0/40，验收未通过，预算用完，不替换旧模型、不进入PPO。[模型、检查及视频](model-deliveries/20261025-bc-wheel-local5000.md)。下面命令仅供后续新预算使用，TensorBoard未启动：

```bash
scripts/mjlab.sh train-bc \
  --dataset artifacts/bc-wheel-reference-20261025-v2/paired/dataset.json \
  --corrections artifacts/bc-wheel-reference-20261025-v2/braking69 \
  --corrections artifacts/bc-wheel-reference-20261025-v2/braking121 \
  --action-sampling rise-brake-turn-stop-v2 \
  --iterations "${CT_BC_STEPS:?先设置本次授权的BC更新次数}" \
  --output logs/mjlab/chuanliantui_bc/<新的40维运行名>

mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_bc --port 8081
```

新采集`collect_lqr_student.py`默认输出40维；`bc.py`按清单构建39或40维网络，`eval_bc.py`按模型自动选择对应观测。新模型格式`ct_bc_actor40_pd6_wheel_v1`，网络40→256→128→64→6，其他训练定义保持。以下39维命令、旧纠偏和结果为历史记录；未版本化旧清单只允许回放，旧训练命令现在会被入口拒绝。

## 配对采集

`scripts/agent/collect_lqr_student.py`在原SMP回合之外保存`observation`、`source_motor_q/dq`、`nominal_motor_positions`、`observation_time`和`action_time`。每拍先保存实际39维学生观测，再由实时LQR生成归一化6维PD目标，然后经过原PD／FIFO执行；状态和SMP特征仍在步进后记录。故`observation_time = action_time = time - 0.01`，不能把步进后状态和本拍动作直接配对。手动指定完整初态后重新构建观测缓存，避免首拍沿用另一个随机姿态。

已有170条SMP示范没有这组完整配对，保持原文件；新采集另外输出。接收仍采用完整20秒、严格起立／最终平衡、平滑、停车及高度等原筛选。数据分组按姿态库行，同姿态不同延迟整体划分，不跨训练／验证。当前新数据为171条／49个初态，138条／39初态训练、33条／10初态验证；[本轮数据与结果](../artifacts/bc-lqr-20261025/README.md)。

`scripts/agent/check_bc.py`独立重算步进前关节、角速度、重力方向和上一拍动作，核对时戳、文件哈希、分组及有限梯度；优化器更新0。该数据使用名义无噪声／无观测延迟的39维输入，不代表已适应正式训练随机化。

## 行为克隆训练

`ct_mjlab/bc.py`直接构建现有RSL `MLPModel`（39→256→128→64→6，ELU，无输入归一化），不增加历史编码器或额外反馈。输出就是原始6维策略动作。仅训练MLP，按起立前2秒／后续站立等概率取样；损失按训练集各动作标准差缩放，下限0.1，验证数据不参与尺度统计。这个尺度只影响监督损失，不增加部署侧输入／输出变换。

新增`scripts/mjlab.sh train-bc`入口，强制显式更新预算；输出BC模型／Adam状态／数据指纹／RNG／日志／源码。动作分布标准差固定保存在检查点为0.05，模仿阶段不优化它；未来PPO初始化会一并加载，避免立即恢复到原随机actor的标准差1.0。

制动纠偏可显式选`--action-sampling rise-brake-turn-stop-v2`：起立50%，旧站立／过渡、新制动、新转向纠正、新停稳各12.5%。新增三类按动作生成前的真实速度／偏航／腿速和姿态分区，训练类别互斥且完整，验证仍按原起立／站立等概率抽样。默认旧采样及其数据指纹保持兼容；新定义随checkpoint保存，回放和拟合检查自动按模型定义核对数据。此选项仅用于动作模仿，不改变SMP／PPO或部署接口。

```bash
scripts/mjlab.sh train-bc \
  --dataset artifacts/bc-lqr-20261025/paired/dataset.json \
  --iterations "${CT_BC_STEPS:?先设置本次授权的BC更新次数}" \
  --output logs/mjlab/chuanliantui_bc/<新的运行名>

mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_bc --port 8081
```

首轮远程已按授权完成5000次，单帧验证误差下降，但独立起立验收未通过，不能以低拟合误差代替闭环成功。

随后按新增5000次预算加入180212帧纠偏训练，累计10000次；原40组留出起立／最终平衡40/40，腿部平滑指标40/40，但零速下持续移动和转向，停车0/40。尚未通过完整验收，PPO未启动。[纠偏训练报告、同初态视频与来源](../artifacts/bc-correction-20261025-v1/README.md)。

再按新增5000次预算加入69条合格制动，并采用上述五类采样，累计15000次；原留出起立／最终平衡仍40/40，偏航和腿速门槛40/40通过，平均速度约0.207m/s、末5秒漂移约1.03m，停车仍0/40。预算用完，PPO继续暂缓，不以转向改善代替停车。[本轮报告、完整视频和复现](../artifacts/bc-braking-train-20261025-v1/README.md)，[模型交付记录](model-deliveries/20261025-bc-braking5000.md)。

随后零更新接管诊断：仅39个训练初态×四延迟，学生0/156、保留参考教师55/156、重锁参考教师121/156综合停车；500Hz执行链核对通过，同实测状态的位置／朝向误差项明显影响教师轮目标。已导出121条／242000帧合格教师纠偏段至`artifacts/bc-braking-takeover-20261025-v2/corrections`，原验证张量及固定抽样保持，尚未训练。各分支前缀不是精确同状态，不能把组间差异全部归因于重锁；当前0.207m/s漂移实际是后退。[诊断、限制和完整成功／失败视频](diagnostics/20261025-bc-braking-takeover.md)。

## 独立回放和纠偏

`scripts/agent/eval_bc.py`默认只用验证组不同初态，各覆盖四种2／4ms延迟，学生独立出动作；没有教师力矩混入。报告完整20秒、严格起立、最终双轮平衡、末5秒腿速RMS／速度／漂移。`scripts/agent/render_bc.py`从真实记录生成完整20秒60fps的同初态、同延迟教师／学生视频；提前终止后明确标记定格，不能当20秒物理存活。

停车／转向的零更新接管对照另见[专项诊断](diagnostics/20261025-bc-parking.md)，入口`check_bc_parking.py`／`assess_bc_parking.py`；它检查教师阶段、隐藏停车参考和训练覆盖，不导出训练标签。本次严格短程数值恢复失败与补充恢复控制分开记录，不能将产物核对当作数值复现通过。

```bash
PYTHONPATH=mjlab_training/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  mjlab_training/.venv/bin/python scripts/agent/eval_bc.py \
  --dataset artifacts/bc-lqr-20261025/paired/dataset.json \
  --checkpoint logs/mjlab/chuanliantui_bc/20261025_remote5000/bc_5000.pt \
  --output artifacts/<新的独立验收目录>
```

失败后，仅用`--split train --collect-corrections`采集学生真实经过的状态和教师查询动作，排除教师失败／终止后的帧。教师只查询标签，原学生动作继续经原PD执行；查询记录不作为新的成功运动示范加入SMP。验证初态禁止生成训练纠偏数据。

采集端现已增加阶段同步：每回合首次实际连续双轮支撑满1秒后，将旁路教师同步到平衡阶段并重锁停车参考；局部复位只清本环境的同步历史，不更改学生动作或FIFO。另用`collect_bc_braking.py`从训练初态采集“学生前10秒自然漂移→教师后20秒制动”，156条中69条通过完整筛选，新增138000帧有效动作，只保留合格轨迹的教师段。原39／10组划分和验证张量保持，尚未用新数据训练。[数据、完整对照视频、正式检查与命令](../artifacts/bc-braking-20261025-v1/README.md)，[阶段同步与筛选说明](diagnostics/20261025-bc-braking.md)。

训练组纠偏标签共180212帧，验证帧保持不变；检查实际动作严格等于同GPU精度／批次下的独立网络输出，教师目标没有被目标限幅裁剪。`ActionDataset`只将这些标签加入训练侧；`train-bc --corrections <目录> --init <bc.pt>`从已有actor出发，启用新的Adam状态和新的显式预算，不是恢复旧预算。下面命令已完成一轮新增5000次，重跑仍需新的预算和输出目录。

```bash
scripts/mjlab.sh train-bc \
  --dataset artifacts/bc-lqr-20261025/paired/dataset.json \
  --corrections artifacts/bc-lqr-20261025/corrections \
  --init logs/mjlab/chuanliantui_bc/20261025_remote5000/bc_5000.pt \
  --iterations "${CT_BC_STEPS:?先设置本轮新增BC预算}" \
  --output logs/mjlab/chuanliantui_bc/<新的纠偏运行名>

mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_bc --port 8081
```

纠偏模型回放时也指定相同`--corrections`，以核对训练数据指纹。不能拿验证状态补完数据后继续称其为留出验收。

## PPO接入边界

`train --smp-prior <prior.pt> --bc-init <bc.pt>`仅在新SMP PPO运行中加载actor（含0.05动作标准差），critic和PPO优化器保持新状态；与`--resume`互斥。检查XML、39维观测、学生动作实现哈希及网络结构，保存BC来源SHA／数据指纹／步数并随PPO续训恢复。PPO输入和实体执行接口保持原样。

代码接入与零更新往返检查已经完成；只有独立行为验收通过且用户另行授权PPO预算后才启动。当前首轮BC没有通过，因此未训练PPO。模型与运行结果详见[本轮报告](../artifacts/bc-lqr-20261025/README.md)，不能将其作为新的可用控制策略或真机模型。
