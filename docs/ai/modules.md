# 模块职责

V13 r2外部有限入口新增真正count-only模式：完整恢复A700的网络、四Adam、归一化及计数，同一训练进程精确保存A1200/A1700并继续至A2200，保留每60秒保存/异常现场；仅新增1500次实际actor更新，没有旧时限或行为早停，评估全部在训练后串行进行。训练/审计、三评估及pipeline已实际退出0，累计W9056/E/A/C各2200、新增6000/1500/1500/1500及399360转移、累计642048；循环4509.199秒/进程4522.847秒。A1200/A1700均0/96存活/平衡、128/384与112/384阶段完整；A2200恢复96/96存活、384/384完整，但平衡0/96，数值异常均0。A2200前进实际−.493～−.492m/s仍反向，后退−.684～−.683m/s；停车水平mean .553～.589/peak .795～.850m/s，高度.303～.305m。末1秒持续每轮及总承重条件均96例未满足，机身离地/姿态/角速度各0例失败，不能外推为整段无接触。相比A368仅后退/停车/高度改善、前进略恶化，属于部分恢复，不能称站稳、持续全面改善或推断长训必定有效/无效。101生产源码、算法/超参/物理保持；目录迁移期间固定六入口及Recorder兼容副本的恢复/另存属于工具路径维护，不是算法变化。[r2结果](../../artifacts/supermimic-v13-continuation-20261101-r2/README.md)。

上一轮r1使用独立外部入口`scripts/agent/train_supermimic_v13_continuation.py`（历史固定轮次工具，现列入[归档／运行中保护清单](../../scripts/agent/MIGRATION.md)），只为原训练循环添加累计actor计数/真实UTC停止边界、异常现场保存和有限数值检查；生产训练/更新顺序与超参不改。该轮预算2026-11-01 00:36:27～02:06:27，从上一轮A368完整恢复全部网络、四Adam、归一化及计数，不重复预热；环境/回放仍按原main重建，不能称轨迹精确续接。A700实际学习1021.385秒/进程1034.272秒，W3056/E/A/C各700，新增100352转移，训练/审计退出0。固定96例存活0/96、平衡0/96、数值异常0，原成功ID全失，160/384阶段完整；按原规则停止，1100/1500未执行。同权重独立Final复测已完成且结果一致，pipeline于01:10:55.874退出0，余量不追加。该轮规则不套用于本轮count-only试验。[r1结果](../../artifacts/supermimic-v13-continuation-20261101-r1/README.md)。

`mjlab_training/src/ct_mjlab/supermimic/`当前V13使用预热后world4→真实估速监督→actor BPTT→critic交替更新，无PPO。`learner.py`仍导入`motion.py::WorldModel`：删除轮角9/14及49:64实际力矩/接触力，学习47维、增量46维、输入141/输出230，预测5个2ms子步；16个非轮关节位置及全部18个关节速度（含轮速）保留。`state.py`的254/64维仍为真实采集和兼容容器；`networks.py`新增324维`MotionCritic`，只读47motion/162传感历史/18噪声/1年龄/75参数/21上下文，历史actor、Critic和完整物理world保留。

`motion_adapter.py`负责PD力矩估计、轮角补0几何、延迟传感和软终止；不读取真实力矩/接触/承重，10槽记忆仅年龄有效。当前13项奖励v7/辅助配方v2、高度`height_tracking * upright_gate`无contact factor及PD/软终止在V13不改变，模型与真实奖励/失败仍不逐项等价。`kinematics.py`是纯Torch树FK（不解闭链），`buffer.py`负责含终止当前拍、排除后续回合的mask；learner只对有效行前向，encoder仅监督真实history对应的COM机体速度×2。actor阶段冻结world/critic/encoder参数和normalizer但保留输入梯度，critic阶段才更新统计、重算detach的λ目标。

`__main__.py`提供`--init-world`，仅与完整actor初始化组合并严格核正式V12来源；初次V13来源6992次与当地Adam0分列，critic/四优化器当时冷启。原615固定SHA的18→13窄桥保留；resume先核超参、配方、真实计数/阶段、课程和RNG再恢复，真实环境和回放重建。256次初始world-only后W260首次E/A/C，prewarm/postwarm锁点，不按loss挑模型。上一轮有限试验已完成两段600.299/601.513秒，累计W1728/E/A/C各368/142336转移，正式检查和96例评估实际退出0。原成功集合保留，但前进反向、停车变差、目标.26m仍偏高，候选不采纳；随后用户另批90分钟续训，阶段结果见顶部。代码与旧模型保留，未改真实控制/失败/随机化、XML/25/125六动作/PPO/H7。结果及未验证项见[适配说明](../mjlab-chuanliantui-supermimic.md)。

行走初态名义高度0.30m：共享standing_pose增加可选高度，默认0.27m不变；split env_cfg仅覆盖walk的初始关节及root位置，reset_walk_pose使用相同0.30m几何/原倾斜和地隙。目标高度仍每回合0.22～0.30m，PD标称角与模型契约不改，getup/旧任务保持原路径。[初态记录](../../artifacts/walk25-reset30-20261030-150035/README.md)。

2026-10-30 getup当前恢复`chuanliantui_split/action.py::GetupPdAction`的原EMA80：100Hz四腿递归混合、raw观测保留、轮子不滤波，等待期/复位清缓存；100ms接管及其共享目标方法已撤回，历史实现保存在训练快照和artifacts。`chuanliantui_stand_v2/rewards.py::RewardState.compute`仅role=getup分支取消torques的has_stood门控，权重−0.0002/屏蔽/限幅不变，其余18项及walk/独立stand_v2保持；split runner保存v4奖励配方与原v2滤波契约，CPU未适配EMA时拒载。[当前接口与检查](../interfaces/20261030-getup-prestand-torque.md)。

当前行走profile v5仅改变非零直行速度采样下限1.0m/s和课程上限(1.0,1.5)：`SplitCommandCfg.walking_speed_min`由mdp采样使用，env_cfg定义两档，runner严格保存核对下限及新版本。奖励沿用v4，getup/旧stand_v2不变；停车和斜率小指令保留。[参数与范围](../interfaces/20261030-chuanliantui-two-policy.md)。

2026-10-30新增两个独立mjlab任务`chuanliantui_getup`、`chuanliantui_walk`，共享实现位于`envs/chuanliantui_split/`，单帧25/历史125/Critic71/动作6、无模式观测；`env_cfg.py`分开配方、`rewards.py`分开角色权重，行走v4独立计算17项无起立阶段的奖励，腿速/软限位使用四实体腿电机且保留XML无限位语义、起立保留原公式，`runner.py`严格分角色续载。训练不加载另一策略，双模型切换只在新独立CPU sim2sim入口。旧任务与模型保留，详见[双策略接口](../interfaces/20261030-chuanliantui-two-policy.md)。

串联腿 mjlab 包含 `chuanliantui` 华南虎 multimode 与 `chuanliantui_stand` 独立稳站任务，均使用 39/82 维观测，代码在 `mjlab_training/src/ct_mjlab/`，见 [multimode说明](../mjlab-chuanliantui.md)与[稳站说明](../mjlab-chuanliantui-stand.md)。旧 `wheel_legged_gym` 任务和内嵌算法继续服务 Isaac 历史模型。

以下任务和目录索引以 `26_wheelleg` 代码为基准；其他维护线的差异见下文。开始工作时以实际检出的分支与任务配置为准。

## mjlab 任务入口

独立`chuanliantui_stand_v2`在`envs/chuanliantui_stand_v2/`：32维单帧＋160维历史、89维Critic、6动作，`network.py`负责速度编码器及其独立优化器，`export.py`导出encoder＋actor。共享闭链模型/PD，独立起立终止、恢复观测及checkpoint版本。见[稳站v2](../mjlab-chuanliantui-stand-v2.md)。

`mjlab_training/src/ct_mjlab/` 按与 Isaac 侧相同的职责划分：`envs/__init__.py` 注册 `chuanliantui`；`envs/chuanliantui/` 放 model_3023 冻结快照里的 23 个模块；`utils/task_registry.py` 提供 `get_cfgs`、`make_env`、`make_alg_runner`；`scripts/` 提供 train、play、eval、export、check；`cli.py` 与 `scripts/mjlab.sh` 只负责把子命令转发到对应脚本。详见 [mjlab README](../../mjlab_training/README.md)。

## 顶层

| 路径 | 职责 |
|---|---|
| `wheel_legged_gym/envs/` | 环境实现与配置，任务注册 |
| `wheel_legged_gym/rsl_rl/` | 内嵌的 rsl_rl（PPO / ActorCritic / runner / storage） |
| `wheel_legged_gym/scripts/` | `train.py`、`play.py` 入口 |
| `wheel_legged_gym/utils/` | `task_registry`、`helpers`（参数解析）、`terrain`、`logger`、`math` |
| `sim2sim/` | MuJoCo 部署验证与对照工具 |
| `resources/robots/` | URDF / mesh / 关节配置 |
| `logs/` | 训练输出（权重 + tensorboard）。**已 gitignore，且不可删** — 见下 |
| `docs/ai/` | 本文档目录 |

## 环境层

| 类 | 文件 | 说明 |
|---|---|---|
| `BaseTask` | `envs/base/base_task.py` | gym 接口骨架 |
| `LeggedRobot` | `envs/base/legged_robot.py` | 主力基类：仿真步进、观测构造、奖励、域随机化、地形 |
| `LeggedRobotVMC` | `envs/wheel_legged_vmc/wheel_legged_vmc.py` | 用 VMC 统一开链/闭链机构的运动控制 |
| `MiniWheelLegged` | `envs/mini_wheel_legged/mini_wheel_legged.py` | imcawl 机器人。override 前向轴（+y）与倾角终止阈值 |

## 已注册任务

在 `envs/__init__.py` 里注册；新任务必须加到这里才能用 `--task` 调用。

| `--task` 值 | 环境类 | 机器人 | 说明 |
|---|---|---|---|
| `mini_wheel_legged` | `MiniWheelLegged` | imcawl | `experiment_name = mini_wheel_legged` |
| `wheel_legged` | `LeggedRobot` | wl | 端到端开链，多地形 |
| `wheel_legged_vmc` | `LeggedRobotVMC` | wl | VMC，可迁移到闭链实机 |
| `wheel_legged_vmc_flat` | `LeggedRobotVMC` | wl | 平地，显存需求低 |
| `chuanliantui` | `Chuanliantui` | chuanliantui | 串联腿平地站立/行走 |
| `chuanliantui_standup` | `ChuanliantuiStandup` | chuanliantui | 0.15 m 地面后摆起立；与站立任务共享 25 维观测 |

## 机器人资产

| 名称 | 路径 | 状态 |
|---|---|---|
| imcawl | `resources/robots/imcawl/urdf/imcawl.urdf` | MuJoCo 侧对应 `sim2sim/imcawl.xml`，两者必须手工保持一致 |
| wl | `resources/robots/wl/urdf/wl.urdf` | **在用**。`wheel_legged` 任务直接引用；`wheel_legged_vmc` / `wheel_legged_vmc_flat` 继承其 `asset` 段，也用它 |
| xwl | `resources/robots/xwl/urdf/xwl.urdf` | **在用，但 env config 在 `XML` 分支上**，见下节 |

imcawl 关键参数（`mini_wheel_legged_config.py`）：

- DOF 顺序 `[lf0, lf1, l_wheel, rf0, rf1, r_wheel]`
- 默认关节角 `[0.9, -1.62, 0, -0.9, 1.62, 0]`，对应站高约 0.30 m
- 连杆 `l1 = 0.21`（髋→膝）、`l2 = 0.25`（膝→轮心）；改 URDF 必须同步这两个值
- 腿 Kp=60 / Kd=2，轮 Kp=0 / Kd=0.5（轮是纯速度控制）
- 力矩上限 `[30, 30, 5, 30, 30, 5]` N·m
- 站高命令范围 `[0.20, 0.40]` m
- `fail_tilt_pg_z = -0.85`（约 32°）：基类 -0.1 太松，策略曾学会 ~40° 前倾斜靠停机作弊

## 分支分叉与维护线

下表在 2026-09-28 按本地分支代码核对，是维护线用途索引，不表示远程默认分支。开始任务和提交 PR 时都要再次核对实际版本及目标分支。

| 维护分支 | 机器人／任务范围 | `mini_wheel_legged` 使用的资产 |
|---|---|---|
| `26_wheelleg` | 含 `chuanliantui`、`chuanliantui_standup`，同时保留 imcawl 与 wl 任务 | `resources/robots/imcawl/urdf/imcawl.urdf` |
| `大腿` | imcawl 维护线，同时保留 wl 任务 | `resources/robots/imcawl/urdf/imcawl.urdf` |
| `XML` | xwl 维护线，同时保留 wl 任务 | `resources/robots/xwl/urdf/xwl.urdf` |

`大腿` 与 `XML` 的 `mini_wheel_legged` 同名但机器人不同；xwl 的配置和历史权重属于 `XML` 这条线，不应误用 imcawl 的接口或资产。chuanliantui 的 25 维观测也不兼容旧 27 维 checkpoint。

基类、奖励、reset 和观测实现也可能随维护线分叉。跨分支引用代码或权重前，对比具体 SHA 的实现并读取该分支的仓库说明；不使用“当前分支”或固定提交数量代替版本记录。团队的功能分支与 PR 流程见 [CONTRIBUTING.md](../../CONTRIBUTING.md)。

## sim2sim 工具

`mj_sim2sim_chuanliantui_stand_v2.py`为32/160维稳站v2原生CPU入口，复用multimode物理助手、独立构造本任务观测/历史/恢复；ONNX零速0.22m名义回放，支持保存初态与浏览器查看。有限范围见[接口记录](../interfaces/20261028-stand-v2-native-cpu.md)，不能与旧39维入口或H7互套。

| 文件 | 职责 |
|---|---|
| `mj_sim2sim.py` | imcawl 的 MuJoCo 部署验证脚本，文件头注释是该机器人的部署契约权威来源 |
| `mj_sim2sim_ct.py` | chuanliantui 串联训练代理回放；闭链模式用于差异诊断，不能据此声称真机已通过 |
| `eval_isaac.py` | Isaac 干净环境对照组，分辨「策略问题」vs「引擎 gap」 |
| `check_model.py` | `imcawl.xml` 碰撞对与轮几何自检，改完 XML 跑一次 |

## 两个叫 logs 的目录，别搞混

| 路径 | 是什么 | 能不能动 |
|---|---|---|
| `logs/`（仓库根） | 训练产物：`model_*.pt` 权重 + tensorboard 记录，按 `<experiment_name>/<日期时间_run_name>/` 组织 | **只读。不可删、不可移动、不可覆盖** |
| `wheel_legged_gym/logs/` | `envs/` 的陈旧 `.py` 代码副本（2026-01-13 快照），不含任何训练产物 | 不要 import、不要改、不要参考 |

`wheel_legged_gym/logs/` 不是训练产物目录：它之所以没进 Git，是因为 `.gitignore` 里 `logs` 那条规则不带路径前缀，把它一起连带忽略了。它是 `envs/` 在 2026-01-13 的快照，与当前 `envs/` 已分叉（`legged_robot_config.py` 差 162 行、`legged_robot.py` 差 18 行）且缺 `mini_wheel_legged`。全仓搜索确认**没有任何代码 import 它**，但它自带 `__init__.py` 并在其中调 `task_registry.register`，误 import 会注册出一套过时配置。真实环境代码只在 `wheel_legged_gym/envs/`。
