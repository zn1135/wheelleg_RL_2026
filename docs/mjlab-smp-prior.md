# 示范采集、SMP 与 PPO 接入

2026-10-25：新增[轮角39／82 SMP PPO v4](interfaces/20261025-wheel39-smp-ppo-v4.md)，从当前首次站稳清零BC初始化PPO，并在保存／恢复和各运行入口检查对应观测定义。旧v3及普通PPO默认保持；本轮不更新先验、不训练策略，先验仍为学生170条的10000次版本。零更新回放、奖励明细和视频见[接入报告](../artifacts/ppo-wheel39-interface-20261025-v1/README.md)。

2026-10-24 清理状态：按用户要求，已删除本地立即出力起立示范 `artifacts/smp-rise-move-20261024/ground-integral-030`、`prior_2000.pt` 及其同权重的 `smp_prior.pt` 副本，并移除三份本地归档包中的对应内容。新 0.2 秒等待起立示范、静站和移动示范保留；远程文件未修改。下文涉及旧数据或 prior2000 的路径仅为历史记录，原命令当前不能直接运行；清理后的归档包也不再对应历史包哈希。新先验仍需另行训练。

第二阶段交付 `collect`、`train-prior` 和运动评分 API；第三阶段通过可选 `--smp-prior` 接入 PPO。仍只有 `chuanliantui` 注册任务，不传该参数时保持原行为。39/82维观测、6维动作、2ms物理／PD和10ms策略接口不变。实验结果统一见[项目实验记录](ai/project-log.md)顶部。

新0.2秒等待示范的SMP检查点与来源见[2026-10-25本地prior2000交付](model-deliveries/20261025-smp-hold-prior2000.md)；原立即出力数据的[历史prior2000](model-deliveries/20261024-smp-prior2000.md)本地已清理。训练后可用`scripts/agent/assess_smp_prior.py`核对预算、连续日志、优化器状态及固定验证窗口去噪误差；该脚本不更新参数。

SMP（Score-Matching Motion Priors）用去噪模型判断一段运动像不像示范。它学习连续状态窗口，**不是把六路教师力矩直接回归成 PPO 动作**。力矩仍随原始状态保存，用于核对控制输出。

动作模仿另用[行为克隆入口](mjlab-behavior-cloning.md)：采集同拍39维观测与6维学生PD目标，训练现有actor，独立验收后才考虑`train --smp-prior ... --bc-init ...`。首轮5000次的独立行为验收失败，PPO没有启动；纠偏查询标签不能作为SMP成功运动示范。

2026-10-25已用170条学生PD示范在远程从头完成10000次更新，新模型及预算检查见[交付记录](model-deliveries/20261025-smp-student170-prior10000.md)。新增正式`scripts/agent/check_smp_student_scores.py`：`passive`从指定完整初态运行真实主动零出力物理；`score`以相同留出姿态比较平滑成功、10ms轮延迟振荡和落稳趴地，独立重算特征、固定噪声、核对评分不改参数／统计。单位尺度的本轮排序通过，示范误差缩放仅作诊断，未改生产PPO奖励或训练PPO。[报告、三类分数和复现命令](../artifacts/smp-ranking-student170-20261025/README.md)。

## 当前起立扩采：学生PD接口

2026-10-25新增独立采集脚本`scripts/agent/collect_lqr_student.py`：实时LQR每10ms读取真实反馈并反解学生目标，原500Hz PD/FIFO执行；使用当前训练的腿／轮2/2、2/4、4/2、4/4ms，前0.2秒实际主动电机零出力。它不加载PPO模型，也不启用预览补偿，不修改下方默认1ms直接力矩采集入口。10ms仅作独立压力测试，失败轨迹不加入正示范。

```bash
PYTHONPATH=mjlab_training/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  mjlab_training/.venv/bin/python scripts/agent/collect_lqr_student.py \
  --seeds 4101,4102 --pose-seed 4100 --poses-per-seed 32 \
  --output artifacts/<新扩采目录>
```

从1024行正常趴地姿态库无放回选择64个不同初态，每个初态覆盖四种延迟，共256条20秒候选；不倒置、不发移动命令，腿长0.30m。接收除完整回合／严格起立／末段双轮平衡外，还要求无控制器故障、2秒后机身无接触且最大倾角≤25°、平均高度误差≤3cm，以及末5秒腿速RMS≤0.02rad/s、平均水平速度≤0.01m/s、相对该窗口首帧位置变化≤2cm。后面三项是本批数据的预设筛选门槛，不是v3.1平衡定义。

输出仍为相同`dataset.json`／accepted／rejected格式，记录100Hz在线特征、六路PD目标／请求和实际执行器力、接管标志及完整初态。按姿态库行分组，同初态的不同延迟副本整体进入训练或验证，避免互相泄漏；失败轨迹只列入`report.json`。新数据只提供起立和后续静站窗口，不能冒充移动示范。结果与视频见[本轮扩采报告](../artifacts/smp-student-rise-steps12-20261025-v3/README.md)；未自动重训SMP或PPO。

本轮完整清单为`artifacts/smp-student-rise-steps12-20261025-v3/dataset.json`：256候选接收170、覆盖49个初态，训练／验证137／33条；86条停车不合格的回合另存。`check_smp.py --dataset <清单> --require-behaviors standup,standing --output <新报告>`检查通过，优化器更新0。v1为启动失败、v2为补录字段前主动中断，不能用它们替代完整v3。

## 原1ms直接力矩采集与资格检查

```bash
# 站立初态；两个种子，每个种子两个独立物理环境；20秒仿真可能需要数分钟
scripts/mjlab.sh collect --initial standing --num-envs 2 --seeds 42,43 \
  --seconds 20 --vx 0 --yaw 0 --leg-length 0.30 \
  --output artifacts/smp-standing-030

# 正常趴地起立：先等待0.2秒再主动出力，与PPO门控对齐
scripts/mjlab.sh collect --initial ground --num-envs 2 --seeds 42,43 \
  --seconds 20 --leg-length 0.30 --velocity-integral --ground-start-hold 0.2 \
  --output artifacts/smp-ground-030

# 前进／后退±0.5m/s、左转／右转±0.5rad/s，每环境固定一条命令
scripts/mjlab.sh collect --num-envs 4 --seeds 42,43 --seconds 20 --velocity-integral \
  --commands mjlab_training/configs/smp_moving_commands.json --output artifacts/smp-moving-030
```

`--vx/--yaw/--leg-length/--sequence` 沿用 teacher 的单位和限幅。`--commands`读取固定命令数组，每项只有`vx/yaw/leg_length`，环境按数组顺序循环分配，环境数必须是命令条数的整数倍；不能与`--sequence`混用。每个回合保存其实际命令，加载时逐帧核对，不把一个环境的命令标给另一个环境。

默认采集0.30m腿长静站，teacher默认0.15m不变。`--velocity-integral`是可选仿真扩展：将速度误差累计到原LQR位置反馈槽，限制在±3.5m，准备和腾空时暂停；用于消除原参考模式持续欠速。H7增益、补偿、物理参数和固件不改，原H7参考模式仍为默认。没有倒置恢复采集选项。

`collect --initial ground`现在默认`--ground-start-hold 0.2`：前200个1ms物理步主动六电机力矩为零，重力与实体弹簧照常作用；从第201步（t=0.200s）起，控制器从实际落地状态启动，不把等待时间计入起立阶段或积累积分/滤波历史。站立初态不施加该等待。独立`teacher`默认仍立即控制，预览新示范须显式加`--ground-start-hold 0.2`。等待时长写入回合元数据、训练数据描述与新先验检查点，轨迹另记`control_active`；加载器拒绝混用等待0秒和0.2秒的起立回合。

旧`ground-integral-030`及`prior_2000.pt`来自立即出力的教师，本地副本后来按用户要求清理；不能给旧轨迹补零或改时间戳来冒充新示范。数据指纹改变，旧先验的完整续训会拒绝。新合格起立数据位于`artifacts/smp-hold-020-20261024/ground-integral-030/dataset.json`，检查与接口边界见[启动时序报告](../artifacts/smp-hold-020-20261024/README.md)。2026-10-24时序修改阶段只采集/检查；后续用户另行授权2000次更新并改为本地执行，已于2026-10-25完成[新先验训练](../artifacts/smp-hold-local-2000-20261025-023450/README.md)，没有追加PPO训练。

每个完整回合必须同时满足：完整20秒、无控制器故障、最后连续1秒符合双轮平衡v3.1、2秒稳定窗口后速度／偏航平均绝对误差≤0.30、高度误差≤0.03m、测量段倾角≤25°且机身无接触。高度命令仍明确换算为腿长加0.03648m。正常趴地允许起立前触地，但起立后漂移也会拒收；另要求记录到准备阶段转入DONE，并保存首次交接时刻，不能把始终处于平衡阶段的静站冒充起立。

输出目录：

| 文件／目录 | 用途 |
|---|---|
| `dataset.json` | 仅列合格的完整回合，训练入口读取此文件 |
| `accepted/*.npz` | 100Hz完整仿真qpos/qvel、根状态、命令、请求和实际六路力矩、接触、起立阶段、运动特征与元数据 |
| `rejected/*.npz` | 被拒绝的原始回合，供排查，不列入训练清单 |
| `report.json` | 接收／拒收数量、拒收原因、已完成种子和异常状态 |
| `rollout-<seed>/` | 教师原始批量轨迹与逐环境验收结果 |
| `snapshot/` | 源码、工作区差异、XML和姿态库快照 |

每个回合记录种子、环境索引、初态、行为标签、实际命令、控制模式、XML哈希、源码哈希、资格判据与文件哈希。中途异常保留此前完成的回合，并把整次采集标为未完成；不把半个回合写成完整示范。序列可以记录用于诊断，但尚无正式序列行为协议，**序列回合目前全部拒收**；不能用固定命令的检查冒充腿长变化验收。

## 运动特征与回合划分

同一个 `smp/features.py::motion_features` 用于在线读取和离线重算，30维顺序如下：

| 维度 | 内容 |
|---|---|
| 0 | 机身离地高度m |
| 1～3 | 机体系重力方向 |
| 4～6 | 机体系线速度m/s |
| 7～9 | 机体系角速度rad/s |
| 10～13、14～17 | 四髋角度的sin、cos，顺序LF前／后、RF前／后 |
| 18～23 | 六电机速度，顺序`lf0,lf00,lfwheel,rf0,rf00,rfwheel` |
| 24～29 | 左右轮轴相对机身的位置，机体系xyz、单位m |

排除累计轮角、世界水平位置、绝对航向和命令。状态先按100Hz记录，再组成连续10帧窗口（相邻首末帧相隔0.09秒）。根四元数为wxyz，线／角速度转换方向明确；完整qpos/qvel保留闭链被动关节。

同一行的仿真状态和特征是步进后的测量；六路力矩驱动了此前1ms物理步，不能直接把该行解释为策略的状态—动作监督对。

训练／验证先按完整回合划分；同一初态、命令、控制选项、源码与种子的多环境副本进一步归入一组，整组只进一侧。每种行为至少需要两个合格的种子／命令组，并分别划出验证组，保证起立、前进、后退和转向等类别在两侧都有数据。然后在各自回合内部取窗口，禁止跨回合。重复回合、错误文件哈希、采样间隔不符、命令标签不一致、特征重算不一致或不同XML版本会报错。

起立回合中，10帧内包含准备阶段或前2秒稳定窗口的片段归为`standup`，其余交接后窗口归为`standing`。DONE只是把控制交给LQR，不能把之后伸腿抬高、落稳的过程丢掉。采样先等概率选择运动类别，再在该类别内随机取窗口，避免短起立被后续十几秒静站淹没。清单顺序改变不影响划分与采样；采样规则`uniform_motion_classes_v2`记入指纹，旧规则检查点不能冒充完整续训。特征均值和标准差仍只用训练回合，每帧计一次，标准差下限0.2。

## SMP 模型与训练命令

实现参考[SMP论文](https://arxiv.org/abs/2512.03028v3)和[MimicKit](https://github.com/xbpeng/MimicKit/tree/fbe1a36e878f036d6bf5521defb8eb3b8ab4170d)。保留2层DiT、4个64维注意力头、50个余弦扩散步、epsilon预测、L1损失、AdamW默认2e-4、梯度范数上限1、EMA0.995／每10步更新／5000步前复制规则。纯PyTorch实现，无新增运行依赖；移植范围、上游固定提交和Apache许可证见[第三方说明](../third_party/mimickit_smp/NOTICE.md)。

先提供20环境可视预检命令；它不替代数据资格检查：

```bash
scripts/mjlab.sh teacher --num-envs 20 --leg-length 0.30 --viewer viser --seconds 20
```

正式训练需要明确更新步数，入口没有默认训练预算。先把 `CT_SMP_STEPS` 设置为本次确认的步数，再运行：

```bash
scripts/mjlab.sh train-prior --dataset artifacts/smp-standing-030/dataset.json \
  --dataset artifacts/smp-ground-030/dataset.json --dataset artifacts/smp-moving-030/dataset.json \
  --iterations "${CT_SMP_STEPS:?先设置本次确认的SMP更新步数}" --batch-size 512

# 另一终端：指向本次SMP输出的日志根目录
mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_smp_prior --port 8081
```

多个采集目录可重复提供 `--dataset`，例如分别采集不同种子。输出为 `logs/mjlab/chuanliantui_smp_prior/<时间>/`，每2000步及预算最后一步保存 `prior_<累计步数>.pt`；不覆盖旧模型。TensorBoard包含训练L1、验证L1和EMA验证L1。验证使用固定输入噪声，不参与更新。

```bash
# 完整续训：iterations仍是本次新增步数，lr须与原运行相同
scripts/mjlab.sh train-prior --dataset artifacts/smp-standing-030/dataset.json \
  --dataset artifacts/smp-ground-030/dataset.json --dataset artifacts/smp-moving-030/dataset.json \
  --resume "${CT_SMP_RESUME:?先设置已有prior_N.pt的路径}" --lr 0.0002 \
  --iterations "${CT_SMP_STEPS:?先设置本次确认的新增步数}"

mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_smp_prior --port 8081
```

检查点保存网络和EMA、特征与评分定义、训练／验证分组指纹、优化器、步数、采样器和PyTorch随机状态、误差归一化状态及源码哈希；数据／划分不一致时拒绝完整续训。当前机器人输入与公开人形模型不同，不能加载人形SMP权重。零步数的检查夹具也不能作为已训练模型加载。

评分使用EMA网络，在噪声等级`[22,15,8]`计算预测噪声均方误差，再计算`exp(-6 * mean(误差 / 各噪声等级历史绝对均值))`。误差归一化不减均值；`record()`与`update()`分开。`prior, metadata = load_prior(path)`加载后可调用`prior.freeze()`冻结模型参数，`prior.score()`返回分数。没有生成初态功能，避免生成状态破坏机器人闭链。

## 第三阶段：可选 SMP PPO

实现为 `smp/ppo.py`，沿用现有 PPO，不叠加教师力矩。新增奖励率 `1.0 * SMP分数` 与原任务各项相加，由 RewardManager 统一乘 `dt=0.01` 一次。当前v3固定三个噪声级误差尺度为`[1,1,1]`，分数为`exp(-6*mean(原始误差))`，训练／回放相同；不累计随机策略误差，不作在线record/update，也不采用示范均值缩放。SMP网络、EMA和特征统计固定。v1／v2前100万个窗口在线估计尺度的实现仅为历史版本。独立随机数发生器产生评分噪声，不消耗PPO动作采样的随机数流。当前仅支持单GPU、24步PPO批次。

每10ms、在自动复位之前刷新物理派生量，调用与示范相同的30维特征函数。连续10帧后才评分，前9步为零；终止步使用终止前的窗口，随后只清空已复位环境的历史。没有跨回合拼接。每步日志包含 `SMP/reward`、`SMP/raw_error`（原始误差）、`SMP/error_scale`（固定1）、`SMP/valid_fraction`、`SMP/normalizer_count`（固定0），回合日志包含 `Episode_Reward/smp_motion`。实际奖励占比用`scripts/agent/check_smp_reward_scale.py`读取RewardManager全部加权项，再乘一次dt核对总奖励，区分正奖励占比和包含负惩罚的绝对值占比。

**初态为正常趴地落地启动**：与教师 `--initial ground` 相同，双腿从姿态库同一行的 `negative` 分支采样，左右配对，机身四元数 `[1,0,0,0]`、根高度0.25m，全部初速度为零。0时刻可能尚未触地，先自由落下；沿用原PPO前0.2秒主动电机力矩门控，物理弹簧仍生效。当前按用户要求只学正常起立和零速站稳，速度／转向命令整个回合均为零；站稳、起立失败和重采样均不放开移动。目标机身高度0.33648m（教师腿长0.30m），所有回合启用站稳后的零速保持奖励。保留原训练物理随机化及推扰；不传 `--smp-prior` 时原初态、命令与奖励保持。

倒置（重力投影z>0）立即终止并自动复位，不训练倒置恢复。PPO的正常趴地起立仍使用原recover模式输入；它与倒置恢复是不同范围。当前配置标识为`chuanliantui-smp-ppo-normal-ground-stand-unit-v3`，见[固定单位尺度接口](interfaces/20261025-smp-unit-reward-v3.md)；初态及全程零速沿用[零速阶段接口](interfaces/20261024-smp-stand-v2.md)。原300轮对应[历史v1接口](interfaces/20261024-smp-ppo.md)，它的移动命令配置已经停止使用；旧v1／v2的自适应尺度检查点不能作为v3完整续训／直接回放检查点，需在对应冻结源码下重放。

先检查20环境初态（没有 `--checkpoint` 时使用零动作，只用于初态预览）：

```bash
# 170条学生PD示范已完成10000次SMP更新；本命令只预览，不训练PPO
export CT_SMP_PRIOR=logs/mjlab/chuanliantui_smp_prior/20261025_student170_remote10000-v2/prior_10000.pt
scripts/mjlab.sh play --smp-prior "${CT_SMP_PRIOR:?先设置现存先验路径}" --num-envs 20 --viewer viser
```

接入交付时只做有限检查，没有启动PPO。之后用户另行授权的首轮300轮已完成，实际起立结果与交付边界见[本轮记录](model-deliveries/20261024-smp-ppo300.md)，不视为可用策略。后续在拥有本次明确PPO预算的终端运行：

```bash
scripts/mjlab.sh train --smp-prior "${CT_SMP_PRIOR:?先设置prior路径}" \
  --num-envs 2048 --iterations "${CT_PPO_STEPS:?先设置本次授权的PPO轮数}" --run-name smp-stand-first

mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_huananhu_multimode_smp_stand_unit_v3 --port 8081
```

输出独立保存到 `logs/mjlab/chuanliantui_huananhu_multimode_smp_stand_unit_v3/<时间>/`，包含actor/critic/PPO优化器、SMP文件哈希、固定单位尺度状态、评分随机状态；另复制 `smp_prior.pt` 和源码快照。续训须同时指定 `--smp-prior` 和 `--resume model_*.pt`，只接受同一先验／奖励定义的SMP PPO模型，拒绝普通PPO、旧自适应尺度profile或非单位统计的检查点。回放、有限check和起立eval同样支持 `--smp-prior`；eval的force/sample均使用该正常趴地分布，不与原姿态分布结果混比。ONNX导出仍仅导出39→6 actor，SMP不进入部署推理图；当前v3未训练，没有新策略交付；原v1的300轮检查点起立失败。有限检查保存的未训练往返夹具不作为策略交付。

```bash
mjlab_training/.venv/bin/python scripts/agent/check_smp_ppo.py \
  --prior "${CT_SMP_PRIOR:?先设置prior路径}" --render-initial \
  --dataset artifacts/smp-stage2-20261024/standing-030/dataset.json \
  --dataset artifacts/smp-stage2-20261024/standing-030-seed43/dataset.json \
  --dataset artifacts/smp-rise-move-20261024/ground-integral-030/dataset.json \
  --dataset artifacts/smp-rise-move-20261024/moving-integral-030-v2/dataset.json \
  --output artifacts/smp-ppo-integration-20261024/check.json
```

20环境实际采样/returns、冻结、局部复位、倒置终止、评分尺度与随机状态保存重载已通过；优化器更新0次。384个验证窗口的合成倒置方向／极端速度探测均明显降分，但时间打乱仅65.9%配对降分，不能冒充真实异常轨迹、运动节奏或策略行为验收。24步起立初期探测中，dt缩放后SMP均值约4.63e-5，原任务奖励约-0.07976；默认权重1尚未按已训练策略校准，后续应看上述日志的奖励比例与实际起立／跟踪指标。完整证据、20环境初态图与未验证范围见[接入报告](../artifacts/smp-ppo-integration-20261024/README.md)。

## 真实运动评分补充检查

新增`scripts/agent/compare_smp_motion_scores.py`。成功样本只取原资格检查通过、且属于验证组的正常起立回合；被动趴地样本是真实零电机力矩、保留弹簧和重力的自由仿真，筛选高度<0.20m且速度低的连续窗口；失败样本由原300轮冻结源码和模型重新运行64环境，初态签名与原检查一致，禁止用当前零速v2配置改写历史失败对照。所有组用同一EMA模型、相同评分噪声、固定评分尺度比较，记录原始误差，绝不更新先验参数或统计。

```bash
mjlab_training/.venv/bin/python scripts/agent/compare_smp_motion_scores.py \
  --prior logs/mjlab/chuanliantui_smp_prior/20261024-191610_first2000/prior_2000.pt \
  --checkpoint logs/mjlab/chuanliantui_huananhu_multimode_smp_v1/2026-10-24_20-21-41_smp-first300/model_299.pt \
  --policy-source logs/mjlab/chuanliantui_huananhu_multimode_smp_v1/2026-10-24_20-21-41_smp-first300/snapshot \
  --legacy-evaluation artifacts/smp-ppo300-20261024-201649/remote-results/migration/evaluation.json \
  --dataset artifacts/smp-stage2-20261024/standing-030/dataset.json \
  --dataset artifacts/smp-stage2-20261024/standing-030-seed43/dataset.json \
  --dataset artifacts/smp-rise-move-20261024/ground-integral-030/dataset.json \
  --dataset artifacts/smp-rise-move-20261024/moving-integral-030-v2/dataset.json \
  --output artifacts/smp-real-score-check-new

mjlab_training/.venv/bin/python scripts/agent/check_teacher_pd_interface.py \
  --dataset artifacts/smp-rise-move-20261024/ground-integral-030/dataset.json \
  --output artifacts/teacher-pd-interface-new.json
```

对照输出包含各组均值/中位数/分位数和排序AUC：AUC=1表示这批样本中每个成功窗口得分都高于失败窗口，0.5接近随机排序；它不是起立成功率。逐帧教师→PD检查只证明相同状态下力矩请求在学生接口可表达，不证明1ms教师改为2ms物理、100Hz动作、0.2秒门控和延迟后的动态复现。实际结果与限制见[本次检查报告](../artifacts/smp-stand-check-20261024/README.md)。

## 新先验与学生接口来源验收（2026-10-25）

已用新0.2秒等待prior2000对真实教师、被动趴地、失败策略和学生成功起立重评分，并用冻结model3023检查当前学生接口完整腿/轮延迟矩阵。结果与命令集中在[评分、来源与视频报告](../artifacts/smp-source-score-20261025/README.md)：学生早期成功起立评分仍低，288/320来源联合合格但全部有漂移，只列起立段候选，未新训练或修改实际奖励。

新增来源检查入口可同时保存代表样本视频：

```bash
env MUJOCO_GL=egl PYTHONPATH=mjlab_training/src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  mjlab_training/.venv/bin/python scripts/agent/check_smp_policy_source.py \
  --checkpoint artifacts/multimode-v1-replay-20261014/models/model_3023.pt \
  --num-envs 160 --seeds 42,43 --output artifacts/smp-source-matrix-new --video

# 已有验收状态直接补导视频，不重新运行策略或物理
env MUJOCO_GL=egl PYTHONPATH=mjlab_training/src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  mjlab_training/.venv/bin/python scripts/agent/check_smp_policy_source.py \
  --render-only artifacts/smp-source-score-20261025/model3023-delay-matrix
```

成功判据：验收目录包含`source-check.json`、真实轨迹`policy-*.npz`及`acceptance-seed42.mp4/.json`；视频覆盖首个种子的代表回合，有同初态失败时并排显示，保存前核对轨迹哈希、保存后核对帧数与首末帧可解码。视频不替代逐环境数值判据，不代表已查看全部环境或已通过定点站稳。

## 正式检查

```bash
mjlab_training/.venv/bin/python scripts/agent/check_smp.py \
  --output artifacts/smp-check.json

# 加真实数据检查：可重复--dataset合并多个采集目录
mjlab_training/.venv/bin/python scripts/agent/check_smp.py \
  --dataset artifacts/smp-standing-030/dataset.json \
  --dataset artifacts/smp-ground-030/dataset.json --dataset artifacts/smp-moving-030/dataset.json \
  --require-behaviors standing,standup,forward,backward,turn_left,turn_right \
  --output artifacts/smp-data-check.json
```

检查覆盖特征的平移／航向／轮累计角不变性、10帧有效窗口、局部复位、完整回合分组与窗口边界、训练专用归一化、漂移回合拒收、原公开函数固定输入／噪声评分对照、原DiT计算路径对照、EMA与误差归一化、CPU/GPU评分一致性、检查点数值和优化器／随机状态保存。只做有限反向传播与序列化检查，**不调用优化器更新，不构成正式训练或行为验收**。合成数据只在临时目录用于检查，随后删除，不加入示范库。

尚未覆盖正式预训练收敛、真实异常运动评分、PPO收敛、奖励权重有效性和完整策略验收；这些不能由有限检查通过推断。首轮训练预算/日志/检查点可用 `scripts/agent/assess_smp_ppo.py --checkpoint <model_299.pt> --prior <prior_2000.pt> --expected-iterations 300 --output <报告.json>` 核对，该脚本不更新参数，不能替代起立或移动行为检查。
