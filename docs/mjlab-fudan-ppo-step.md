# fudan_ppo_step：无视觉单级平台候选

4200轮已结束并冻结GPU评估：200mm完整登台停车20→24/27，但丢1个旧成功；原47例47/47保留。0.35m原地转向0→2/6，±2rad/s已能跟随，+0.5rad/s仍几乎不转，不能称转向修好。完整矩阵/下台/场地未验，不再自动续训。[最新结果与正常速度视频](../artifacts/fudan_ppo_step/20261118-v8-effect4200-01/README.md)。

当前默认配方为`fixed200-turn-rehearsal-v8`：只用200mm台阶（40%上台、40%下台）和20%平地，不再升降台阶高度。侧障碍保持；台阶命令仍为正向0.3～2.5m/s、yaw0、站高0.35m。平地保留一半行驶、四分之一站立、四分之一行驶后停车；一半高度抽0.35m，另一半沿用0.22～0.30m；普通行驶中一半改为原地转向（双向0.5～2rad/s），其余保留行驶转弯。固定3000教师仅约束≤0.30m平地指令，高站姿允许自行探索。奖励、限矩、双延迟、25/125/6接口和真实77点高度采样保持。

跨配方不直接`--resume`。`scripts.agent.fudan_ppo_step.prepare_fixed200_turn --output <新目录>`显式保存3200轮派生输入与前后配方，台阶等级映射为200mm；逐字段确认全网络/std、双Adam、RNG和学习计数精确继承。随后重新完成20环境零学习GPU预检及4096环境完整rollout/反向容量检查，才可在用户预算内续训。当前本机授权新增1000轮、3200→4200；运行与命令见[本轮记录](../artifacts/fudan_ppo_step/20261117-fixed200-turn-resume3200-4200-01/README.md)。预检通过不等于转向或登台通过。

历史v7 `contact-assisted-platform-exploration-v7`远程200→3200续训已按预算结束；模型、GPU行为结论及远程关机状态统一见[项目实验记录](ai/project-log.md)。[续训输入、恢复审计与训练/TensorBoard命令](../artifacts/fudan_ppo_step/20261117-v7-resume200-to3200-01/README.md)。下述建设、启动及阶段评估按当时状态保留；后文多高度课程描述为v7历史，不是v8配置。

2026-11-17远程v7的200轮冻结评估完成：同源正向20/50/200mm各27例，完整登台停车13/6/0例；原3000同工况均0例完整成功，曾短暂上200mm的3例最终失败。原47例平地47→44，丢失高站姿两项高速前进和一项原地转向，按原约定判退化。没有追加训练；优先解决50mm过沿停车与3个平地失效，再升80mm，不据200轮奖励直接续800轮。[完整报告、原始GPU证据与正常速度视频](../artifacts/fudan_ppo_step/20261117-v7-effect200-01/README.md)。下方启动/建设状态按其当时记录保留。

历史启动记录：2026-11-17用户指定远程SSH训练，v7在L40上启动200轮/4096环境，新的独立源码与日志目录不覆盖远程原项目。远程预检/容量/初始化继承通过，按200轮自动停止和审计；启动时尚未验证登台或平地保留。[运行位置、训练与TensorBoard命令及启动证据](../artifacts/fudan_ppo_step/20261116-v7-remote200-01/README.md)。下文“尚未训练”指v7建设完成时的历史状态。

**v7建设完成、尚未训练时的历史记录。** 用户明确允许短暂搭沿并要求保留探索空间：台阶过程接触只统计、不扣碰撞奖励或立即终止；过沿前暂停高度/对称/竖直速度/俯仰约束，参考动作只给弱奖励。速度直接覆盖0.3～2.5m/s，平地恢复完整指令并增加原3000动作模仿保留项。最终仍须解除非轮接触、双轮停车2秒；物理限矩/关节限位与Actor25/125/6保持。20环境2500物理步、真实PPO反向及保留回调均在优化器边界拦截，0更新通过；这不是登台或平地行为通过。[实现、公式、参考指纹与最终预检](../artifacts/fudan_ppo_step/20261116-contact-exploration-v7-01/README.md)。

后续支撑诊断新增35个有限GPU样本：严格15个均早停；允许继续观察但保留接触失败标记的20个中，从棱边假设起步10个有8个出现短暂双轮台面支撑，下层5个仍0。全部无撑沿判据失败，且棱边初态已预先获得高度，不代表完整200mm登台。当前触发逻辑不能识别支撑面，实录未绑定精确固件版本，仍需核对触发前半段；未改验收条件或开训。[支撑假设与撑沿后段视频](../artifacts/fudan_ppo_step/20261116-traditional-support-hypotheses-01/README.md)。

补充：用户提供10次传统上台阶实录并确认高度200mm，现有四髋映射/闭链内部几何核对通过；在原PD/限矩下的80台阶＋10平地GPU参考跟踪未复现登台。实录缺轮地接触和机身平移，触发初态仍是假设，不能据此否定实机动作或当作合格训练示范。未启动新训练，见[量测、逐步核验与视频](../artifacts/fudan_ppo_step/20261116-traditional-replay-02/README.md)。

补充：20mm“抬轮—前移—落轮”已做无学习GPU参考动作尝试，当前未通过；静态可解不等于动态可达，不启动新训练。结果与脚本见[108工况、判据和视频](../artifacts/fudan_ppo_step/20261116-20mm-keyframe-probe-01/README.md)。

目标是RMUC2027公路区200mm单级入口；Actor25维、五帧历史、六动作保持，当前没有通过行为验收的台阶模型。

历史已完成配方`clearlab-single-platform-low-random-v6`。用户要求分析失败、试其他开源并减少随机化，轮数交由助手决定，本次200轮/4096环境已完成，从原3000初始化；冻结GPU台阶0/81、原平地47→10退化。[最终结果与视频](../artifacts/fudan_ppo_step/20261115-clearlab-v6-effect200-01/RESULTS.md)。主参考切为南科大`clearlab-sustech/Wheel-Legged-Gym`固定c354431e；采用其速度跟踪1/增强1、动作变化/平滑−.01、腿姿态差−.1、高度指数幅值1等定义。速度从.3～1m/s起步，跟踪达标才逐步扩展上限，最高2.5；高度仍固定.35m。台阶类型最低20mm，失败不退0mm，平地回练单独保留。多数参数/噪声幅度为v5的五分之一，实际轮地摩擦收为.7～.9；双10～20ms延迟保持源3000接口，不宣称所有随机量都减少。

[失败分析、开源比较与来源](../artifacts/fudan_ppo_step/20261115-low-random-open-source-01/REPORT.md)、[本次200轮配置/检查/运行与TensorBoard命令](../artifacts/fudan_ppo_step/20261115-clearlab-low-random-200-01/README.md)。当前200轮已按预算停止，预检/预算通过，但登台行为未通过，不自动追加。

已完成的v3.3/v4各500轮及复旦适配v5的1000轮均未通过。v5冻结GPU台阶0/81、原47平地47→4；末50轮20mm过沿8/293、50mm0/15，最终1613/1638上台环境退回0mm。保留全部模型/日志，不续用这些退化模型。本次仍继承原3000全部网络/std，新优化器与任务计数从零，不能跨配方resume。[v5最终结果与视频](../artifacts/fudan_ppo_step/20261115-upstairs3-v5-effect1000-01/RESULTS.md)。

上版固定2.5指令/权重1.5/逐项±dt截断组合使实际0～.843m/s的增强项均为−.01，主速度项近0；这是明确的即时奖励弱区，但没有证明它是唯一原因。原复旦也有相同截断，不能把截断本身称为本机独有错误。新试验保留截断，采用南科大权重1及速度课程；不混入Flamingo位置命令或Jump网络，也不改变25维接口。完整开源实现与本机适配有明确边界，不称其完整复现。

## 量测与边界

原版场地固定`ffd1eb7d1d1a2d1654125d8f316fdb0fdac6d001`。最吻合截图的红方台阶约为`(-5.76,-5.405)`，朝向`-Y`；蓝方为中心对称位置。规则手册V1.0.0（20261015）印刷页57图4-35标注200mm高、1148mm宽、2090mm深；页29规定该尺寸级别±5%公差。

本机轮碰撞半径约60mm，名义质量13.395kg。水平机身实际根高为.22/.26/.30m时，机身碰撞净空约.075/.115/.155m。这是静态几何条件，不是高度指令跟踪成绩。200mm是轮半径约3.33倍；固定腿姿硬滚不成立。26.3J势能增量对应的1.98m/s只是理想能量等效值，不是建议或最低接近速度。主动抬轮/跃台的动力学可行性仍需验证。

正式量测脚本只读取原版缓存，输出全场栅格、5mm剖面、0.1mm边沿采样、轮径、净空与来源指纹，不调用`mj_step`：

```bash
PYTHONPATH=mjlab_training/src:. mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_step.measure_arena_step \
  --output artifacts/fudan_ppo_step/新实验/measurement
```

## 配方与模型迁移

入口为`mjlab.sh train → task_registry → StepEnv/StepRunner`。配置在`mjlab_training/src/ct_mjlab/envs/fudan_ppo_step/config.py`。物理控制复用原实体闭链；新增task不会改平地类的默认配置。

- Actor25维、五帧125历史、六动作、原观测缩放`[2,.25,5]`保持；Critic环境141维加估速3维。77个高度槽改成真实向下射线，平地输入等价；扫描排除机器人组，并从机身/轮轴上方发射。
- 腿P20/D1、轮P.1/D0，1ms积分、5ms PD、10ms策略；动作和传感延迟各10～20ms；不启用推扰或持续外力。
- 质量、质心、增益、零位、电机强度和观测噪声缩为v5半幅配方的五分之一；实际地面摩擦专门采用.7～.9，双延迟保持10～20ms。实际范围随`training_recipe`保存，不称为原复旦并联机器人完整配方复现。
- `--ppo-init`默认读取双延迟3000源（SHA256 `dfb221c09648d278d5c5e5c7b78e4d729b7a7a5a56458762b7313dc50831c4f8`），继承Actor、encoder、Critic、std，优化器和任务计数从零。std顺序lf0/lf00/lfwheel/rf0/rf00/rfwheel，值约`.290114/.274964/1.216986/.291316/.273997/1.190875`，没有重置或另设下限。
- 当前没有重置Critic的开关；若以后加入，必须同时实现冻结Actor/encoder/std的Critic预热及漂移记录。维度相同不表示旧价值函数已适应新奖励。
- `--resume`只接受同任务、同配方和相同环境数，恢复网络、双优化器、RNG、课程等级/地形类型；物理回合重新开始。零更新的初始化PT用于推理检查，不是已完成训练的恢复点。
- 物理`contract`保持兼容，新增`task`和`training_recipe`用于训练身份；`load_policy`仍可加载台阶候选。H7及实机不在本任务范围。

地形高度库为0、20、50、80、120、160、180、200、210、220mm；台阶类型从20mm开始且最低保持20mm，后三档加入两侧障碍。固定环境类型约20%平地、40%上台、40%下台（整数分配取整）；这是环境/采样比例，回合长度不同，不能称为精确完成回合比例。平地一半普通行驶、四分之一站立、四分之一行驶后停车。各环境保留自身类型，只在该类型内按进度升级：双轮到有效目标区域支撑至少.1秒、按当前配方允许临时接触、无超速违规的已结束台阶回合升一级，其余降一级；最高级通过后随机回练20～220mm已有等级，台阶类型不回0mm；平地类型的0高度地面独立保留。课程升级不等于严格任务成功；首回合不随机截短。

上台出生距边沿1.2m，下台.6m；随等级逐渐扩大到±20°斜向，横向出生偏移使直线对准通道中部。下台出生仍在平整台面。倒车回放时机身转180°且vx取负，当前训练不抽取倒车。v6台阶命令为正向.3～当前速度上限、yaw0、机身高度.35m；上限初始1m/s，复位满足20秒策略时钟边界及线速/偏航跟踪阈值后加.1，最高2.5m/s。该上限沿用本机全局跟踪课程，区别于南科大逐环境/地形的速度上限；不根据台阶真值改写高度指令。v6平地单独用前后.2～.5m/s、±.5rad/s及.22～.30m高度。v6两轮越沿.45m且正确支撑连续.1秒后发停车命令，稳定2秒才成功；实际接近超过3m/s失败，非轮接触限制和越界判据保持。旧v3低速配方的.8m/s失败线按历史保留。

## 奖励、接触与摩擦

下表为v6基底。v7在越障过沿前暂停`base_height/base_height_enhance/nominal_state/lin_vel_z`，`orientation`仅罚横滚；上/下台碰撞扣分为0。新增`reference_motion=.2`与`target_support=1`均乘dt，分别是无固定时序的最近实录运动状态提示、正确目标面上的实际轮支撑。过沿判据净空由.45放到.15m，之后恢复站高/姿态目标。原接触计时器仍记录旧规则违反，但v7仅平地应用其终止；台阶成功不追溯否决先前搭沿，停车窗口仍要求无非轮接触。完整新配方见顶部链接。

复旦[“上台阶3”](https://github.com/yly-true/fudan_rl_wheel_leg/tree/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/logs/wheel_legged/上台阶3)来自固定`8204e853`快照，配置高度是.23～.33m，“上台阶3_angz+”是.17～.33m；.10～.20属于另一份plane默认配置。下方v6基底奖励表采用[南科大配置](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/c354431e5633eb98d9663fa3fbe444201d2de383/wheel_legged_gym/envs/base/legged_robot_config.py)，下文复旦来源作为历史核对。上游P15/D1、默认角、命令缩放以及TOF触发Jump网络不能整套照搬；公开ONNX未与checkpoint绑定，见[已有来源审计](20261002-复旦训练与部署对应分析.md)。

| 项 | 本任务 |
|---|---|
| tracking_lin_vel / enhance | 1 / 1，南科大原版定义，增强项不再因1.5倍权重过早触及负限幅 |
| tracking_ang_vel / enhance | 1 / 0，与南科大默认启用列表一致 |
| base_height | scale1、sigma.001、函数幅值1；参考双轮下方地面，抬轮不清零 |
| base_height_enhance | 0，按南科大默认关闭 |
| nominal_state | −.1；上/下台距边沿.55m内乘0，不要求先失去双轮支撑；其余保持 |
| lin_vel_z / ang_vel_xy | −2 / −.05；距沿.55m内暂停lin_vel_z，允许主动向上运动 |
| orientation | −10，重力投影XY平方和；不再使用分开的roll/pitch角度平方惩罚 |
| dof_vel / dof_acc / torques | −5e−5 / −2.5e−7 / −1e−4，使用实体电机量 |
| action_rate / action_smooth | −.01 / −.01，按南科大默认配方 |
| collision | −1，按每策略步内是否出现非轮接触计罚 |
| dof_pos_limits | −1，沿用本机闭链关节限位 |

沿用复旦也存在的逐项dt缩放/±dt裁剪，不把取消截断当作本次适配。姿态按实际重力投影公式计算：纯5°倾斜每10ms约−.0007596，取消旧pitch额外弱罚。距沿.55m内继续关闭对称和竖直速度惩罚，允许左右轮先后登台，也允许同步；此项是本机适配差异。高度项全程连续，以局部双轮下方地面为参考，不改成复旦77点平均参考，也不暗改高度指令。正常双轮独立支撑要求两轮有向上支撑接触、没有轮侧或非轮接触且支撑面高度差小于2cm，只用于严格停车条件。停车用整数策略拍计时，连续200拍为2秒；roll与pitch分别检查5°。在线轮支撑在策略步端判定，诊断的每1ms持续轮支撑是更严格口径，须分别报告。

机身/腿杆分别用每部件最大接触力，避免合力抵消；每1ms累计，采样对应该物理步求解的接触（与积分后的qpos有一个物理步时序差）。超过1N计入接触。v6及v7平地应用接触惩罚；v6的非轮连续接触≥.20s，或超过实际重量10%的接触持续≥.10s，终止失败。停车2秒内不允许超过1N的非轮接触，水平速度≤.05m/s、roll/pitch≤5°。这些阈值是候选规格，不是已验证的机械安全阈值。

TensorBoard从第一轮合并已结束和在途回合：接触率记录这些回合各自时间占比的平均值（不是全局按时间加权占比），接触力和持续时间取最大值；有接触时记录峰值世界位置。失败复位不会清掉已结束回合的证据。`episodes.jsonl`保留每回合原始指标、峰值世界位置及成败。GPU验收另存每1ms的机身/腿杆力和位置，用原始轨迹复算，不能只相信环境success标记。

摩擦按实际`terrain`静态体的所有碰撞geom设置，并同时写机器人侧，覆盖priority选择。平地仍走原单geom逻辑。检查实际Warp接触摩擦，而不只看配置值。模型和初始化历史不覆盖原日志。

## 检查、无学习吞吐与训练命令

本机倒车方向诊断使用`scripts.agent.fudan_ppo_step.diagnose_reverse`：`geometry`重定向参考髋轮轨迹，`stance`检查着地站姿净空，`run`有限GPU回放，`audit`复算原始记录，`render`离线正常速度录像；均不更新网络。默认单次76例、最长16秒，名义参数、固定.8摩擦和双15ms延迟，关闭侧障碍；不能替代完整验收或证明机构绝对不可达。[本轮结果、边界与命令](../artifacts/fudan_ppo_step/20261115-reverse-diagnostic-01/README.md)。

v4高速效果使用显式`run --high-speed --speeds 2.5 --command-height .35 --directions forward --frictions .4 .8 1.2`，直接给复位指令、保留配置侧障碍，单批最多128例、每例最长20秒；协议明确3m/s高速界限，旧低速评分默认.8不改。`inspect_training.py training --run ...`只统计指定运行，续训多段须显式给`--episode-runs`，防止混入旧实验。[高速效果、真实速度、视频与原平地回归](../artifacts/fudan_ppo_step/20261115-fast-effect500-01/RESULTS.md)。

所有输出必须使用新目录。GPU预检含零动作物理检查和48步PPO采样/回报接口检查，不反传、不调用优化器更新，并逐值确认全部网络仍等于源模型。以下是交付命令，不表示已经授权运行训练。

```bash
STEP_CHECK="artifacts/fudan_ppo_step/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_precheck"
PYTHONPATH=mjlab_training/src:. mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_step.check \
  --gpu --viewer --num-envs 20 --steps 200 --deadline "$(( $(date +%s) + 240 ))" --output "$STEP_CHECK"

# 对.4和1.2端点分别使用新目录；此类端点检查不能替代完整随机摩擦配方预检。
PYTHONPATH=mjlab_training/src:. mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_step.check \
  --gpu --friction .4 --deadline "$(( $(date +%s) + 240 ))" --output artifacts/fudan_ppo_step/新实验/friction04

# 分配真实PPO缓存并检查小批次反向显存，但不更新权重；另测采样吞吐。
PYTHONPATH=mjlab_training/src:. mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_step.benchmark \
  --num-envs 4096 --steps 100 --ppo-memory-check --deadline "$(( $(date +%s) + 900 ))" --output artifacts/fudan_ppo_step/新实验/benchmark4096
```

训练前明确本次轮数，再执行下面配对命令。当前默认4096，已完成该规模的采样和PPO显存检查；采样吞吐不能当PPO训练吞吐。改变环境数须显式迁移课程数组并记录来源，不能重置计数或静默丢弃Critic。

```bash
: "${STEP_ITERATIONS:?先填写用户本次授权的训练轮数}"
: "${STEP_CHECK:?先完成上面的当前源码20环境预检}"
STEP_RUN="logs/mjlab/fudan_ppo_step/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_upstairs"
scripts/mjlab.sh train --task fudan_ppo_step --num-envs 4096 \
  --iterations "$STEP_ITERATIONS" --precheck "$STEP_CHECK/report.json" --log-dir "$STEP_RUN"

# 另一个终端使用同一个STEP_RUN的实际值；不指向整个logs根目录。
mjlab_training/.venv/bin/python -m tensorboard.main --logdir "$STEP_RUN" --port 6021
```

初始化默认来自冻结3000；显式`--ppo-init`仍检查物理控制、时序、轮模型、复位及延迟。继续本任务用`--resume`替代初始化。新目录存在即拒绝，防止覆盖旧产物。

## GPU矩阵与原场地复核

旧v3低速正式矩阵540项：上/下×.20/.21/.22m×.15/.30/.50m/s×0/±10/±20°×.4/.8/1.2摩擦×正向/倒车，固定高度指令.35m，每项20秒，实际速度>.8m/s仍不计低速成功。该矩阵不能作为v4高速成绩；高速行为需另按实际速度、登台支撑/非轮接触和停车记录，不冒充低速通过。终止/缺测留在分母，保护其他世界的复位不能复活失败样本。`--limit`或缩短`--seconds`只是诊断子集。

```bash
PYTHONPATH=mjlab_training/src:. mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_step.evaluate \
  --checkpoint "$STEP_RUN/model_final.pt" --batch-size 20 --deadline "$STEP_EVAL_DEADLINE" \
  --output artifacts/fudan_ppo_step/新实验/gpu-evaluation
scripts/mjlab.sh accept --task fudan_ppo_step --recording artifacts/fudan_ppo_step/新实验/gpu-evaluation \
  --output artifacts/fudan_ppo_step/新实验/acceptance
```

台阶报告把尚未提供的平地回归列为缺测；原成功样本失效单列退化。用`scripts.agent.fudan_ppo_step.walk_regression --checkpoint ... --output ... --deadline ...`运行原平地冻结矩阵并记录模型指纹，再在上述accept命令中同时提供`--flat-recording <候选平地GPU目录> --flat-baseline <基线平地GPU目录>`。评分会核对模型身份、解码后的初态及样本集合；诊断子集不能替代正式矩阵。完整任务不能仅凭台阶通过就宣称验收通过。起立能力和真机均未覆盖。

原场地入口增加`--spawn-yaw-deg`和`--command-sequence`。序列JSON是非空数组，每项`{"duration_s":秒数,"command":[vx,yaw,height]}`，时长需整策略拍；仅headless使用，总时长由序列确定或与`--seconds`严格一致。下例先确认位置；正式登台须在有限序列中包含停车，不用方向键的历史2m/s默认值做验收。

```bash
scripts/rmuc2027.sh --checkpoint "$STEP_RUN/model_final.pt" \
  --spawn -5.76 -4.20 --spawn-yaw-deg -90 --viewer viser --command 0 0 .30
scripts/rmuc2027.sh --checkpoint "$STEP_RUN/model_final.pt" \
  --spawn -5.76 -4.20 --spawn-yaw-deg -90 --viewer none \
  --command-sequence artifacts/fudan_ppo_step/新实验/commands.json \
  --output artifacts/fudan_ppo_step/新实验/arena-replay
PYTHONPATH=mjlab_training/src:. mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_step.check_arena_replay \
  --recording artifacts/fudan_ppo_step/新实验/arena-replay --output artifacts/fudan_ppo_step/新实验/arena-replay-check
```

台阶模型的原生键盘前进默认.5m/s、转向.5rad/s，历史平地模型保持2m/s和2rad/s；可显式给`--keyboard-speed`调整前进速度。完整场地仍拒绝含推扰/外力的模型，原生回放不是GPU正式验收。未来若选择跃台，先冻结新奖励与边界，再改变配方版本，不能沿用本候选低速通过标准冒充跃台成绩。
