# sim2sim 与部署契约

整机 S2R1 采集的上位机策略复算与闭链 MuJoCo 离线复刻见
[S2R1 离线对照](s2r-offline.md)。该工具不连接电机；真实数据、关节映射和初态需另行提供。

关节 USB 台架采集后的固定机身闭链对照使用 `sim2sim/fit_joint_usb.py`，数据协议及安全门见 H7 仓库 `md/joint-usb-sysid.md`。它只拟合实测力矩输入下的模型响应，不修改训练策略、部署物理极性或现有 XML。

采集到独立的拟合 run 和留出 run、并由台架作者确认四实体电机的关节对应、符号与零点后，在训练仓用已配置的 `WHEELLEGGED_PYTHON` 执行：

```bash
"$WHEELLEGGED_PYTHON" sim2sim/fit_joint_usb.py \
  --run <拟合采集目录> --holdout <留出采集目录> \
  --mapping <已确认映射.json> --out <新结果目录>
```

脚本拒绝未批准映射、相同的拟合/留出 CSV、故障、坏帧或丢帧超过 1% 的 run；输出参数、四实体关节及左右腿几何的实测/仿真对照。`delay_s` 从主控 CAN 排队时刻估计，不能解释为电机内部生效延迟；几何角度约定须先做静态核对，参数 Jacobian 秩不足时不能断言所有参数都已辨识。

改动观测构造、控制时序、URDF/XML、或准备真机部署时读这篇。imcawl 的权威来源是 `sim2sim/mj_sim2sim.py` 的文件头注释；chuanliantui 见本文后面的对应小节及 `sim2sim/mj_sim2sim_ct.py`、训练实现。跨仓库接口版本与模型交付要求见 [部署接口约定](../deployment-contract.md)。

## imcawl 部署契约

imcawl 的部署端（MuJoCo 脚本、真机驱动）必须逐条对齐训练实现；本表不适用于 chuanliantui：

| 项 | 值 |
|---|---|
| 策略 | `ActorCriticSequence`，`action = actor(cat(obs_27, encoder(history_135)))` |
| 权重 | 用 `model_*.pt`；`policy_1.pt` 缺 encoder 不可用 |
| DOF 顺序 | `[lf0, lf1, l_wheel, rf0, rf1, r_wheel]` |
| 观测 27 维 | `base_ang_vel*0.25(3)`, `projected_gravity(3)`, `cmd*[2.0,0.25,5.0](3)`, `(dof_pos-default)*1.0(6)`, `dof_vel*0.05(6)`, `last_action(6)`，裁剪 ±100 |
| 历史 135 | 27×5 FIFO，最旧在前、最新在末；上电用首帧重复 5 次填充 |
| `dof_vel` | 位置差分 `wrap_to_pi(Δdof_pos)/sim_dt`，每个 sim 子步更新一次（**不是**读速度传感器） |
| 动作→力矩 | 腿位置控制 Kp=60/Kd=2；轮速度控制 Kp=0/Kd=0.5 |
| 时序 | `sim_dt=0.005`，`decimation=2` → 策略 100 Hz，PD 内环 200 Hz |
| 力矩上限 | `[30, 30, 5, 30, 30, 5]` N·m |
| 链路延迟 | 训练随机化 0~10 ms。真机通信+执行延迟必须 ≤10 ms，超出即出分布 |
| 前进方向 | 机体 **+y**。`cmd_vx` 是「前向速度命令」（通道 0），对应机体系 vy |
| yaw 通道 | **是外环反馈，不是常数**：`cmd[1] = 1.5*(目标航向 - 当前航向)` 剪 ±5、每步刷新 |

最后一条最容易漏：训练时 `heading_command=True`。部署端喂恒定 yaw 值会导致偏航漂移无人纠正，巡航数秒后摔车。真机必须实现同样的航向保持外环。

## MuJoCo 侧三个已修的坑

`sim2sim/mj_sim2sim.py` 里已全部修复，**勿回退**：

1. **`mj_objectVelocity` 必须用 `mjOBJ_XBODY`**（机体系）。`mjOBJ_BODY` 返回惯性主轴系，base 惯量特征值降序导致轴置换，陀螺仪三分量整个换位。症状：站立勉强、一加速就翻。
2. **`mj_step` 后派生量（`cvel` 等）滞后一个子步**，读状态前必须 `mj_forward` 刷新。
3. **PD 内环顺序必须是「算力矩 → 步进 → 差分更新 dof_vel」**，对齐 `legged_robot.py`。顺序错了 obs 里的 `dof_vel` 滞后 5 ms。

真机同构风险：IMU 角速度的坐标系定义与采样时序必须与策略推理同拍。5 ms 级延迟对静态站立无感、对加速瞬态致命。验证方法是双源交叉（位置差分 vs 速度读数）。

## 摩擦

MuJoCo 摩擦 = 两 geom 逐元素 **max**；PhysX 取**平均**。训练等效摩擦区间约 [0.4, 1.0]，等效均值约 0.75。MJCF 里 0.5 踩在下沿，`mj_sim2sim.py --friction` 可覆盖。

XML 已用 `cone=elliptic` + `impratio=10`（轮式标准配置）。

## Isaac 对照评估

`eval_isaac.py` 必须设 `train_cfg.runner.resume = True`（对齐 `play.py`），否则 `make_alg_runner` **静默**跑随机初始化网络。症状：换 checkpoint 输出逐字节不变。自检习惯：换权重必须换行为。

## 当前性能上限：encoder 速度估计正偏置

截至 2026-07-21，基准权重 `logs/mini_wheel_legged/Jul21_14-09-24_/model_4000.pt` 在两引擎站立/加速/巡航零摔倒，sim2sim 通过。

余下三个现象同一个根源——latent 前 3 维（`base_lin_vel*2` 的估计，见 `rsl_rl/algorithms/ppo.py`）**系统性估高 0.15~0.25**：

- 稳态欠速：真实速度恒低于命令约 0.2 m/s
- MuJoCo 低速后拉：偏置 +0.1 时，调 v̂ 到 0 对应真实 -0.1
- 高速 ≥1.8 m/s 刹车瞬态 encoder 跟丢 → 发散翻车

结构性原因是 `obs_history_length = 5`，encoder 只有 50 ms 窗口。

**已接受现状，安全包线 ±1.5 m/s**（遥操作已限幅）。若将来要更高速：`obs_history_length` 5→20 全重训，或加大 encoder 损失权重续训作廉价改良。

诊断手段：`mj_sim2sim.py` 与 `eval_isaac.py` 日志里的 `v̂` 列（策略内部速度估计）对比真实 `v_fwd`。

## URDF / XML 同步

`resources/robots/imcawl/urdf/imcawl.urdf`（Isaac 侧）与 `sim2sim/imcawl.xml`（MuJoCo 侧）是**两份手工维护的模型**。改一边必须改另一边，然后跑 `python sim2sim/check_model.py`。连杆长度变了还要同步 `mini_wheel_legged_config.py` 的 `asset.l1 / l2`。

## chuanliantui 串联训练代理回放

`sim2sim/mj_sim2sim_ct.py` 默认加载 `chuanliantui_train_proxy.xml`。它由
`resources/robots/chuanliantui_new_1/urdf/chuanliantui_train.urdf` 生成，保留训练端
6-DOF 串联拓扑、固定后支链、地面、根 free joint 及六个同名力矩电机；模型必须保持
`neq=0`。改训练 URDF 后运行
`scripts/agent/generate_chuanliantui_train_proxy_mjcf.py`，再用
`scripts/agent/check_chuanliantui_train_proxy.py` 检查该契约。

chuanliantui actor 观测为 25 维：机体系角速度(3)、重力投影(3)、命令(3)、四个腿关节
`[lf0, lf1, rf0, rf1]` 的位置(4)、六个关节速度(6)、上次动作(6)。连续轮的绝对位置
不进入 actor 或 encoder；历史为 `25×5=125`。这改变了网络的 encoder、actor 和 critic
输入形状，所有旧 27 维 chuanliantui checkpoint 均不能加载，必须重新训练。
提交观测变更前可运行 `python scripts/agent/check_chuanliantui_observation.py`，核对训练端布局、
噪声、critic 维度、历史 FIFO 和 MuJoCo 构造是否一致。

训练与 MuJoCo 回放均用 500 Hz 的物理/PD 内环（`sim_dt=0.002`），每 5 个内环步推理一次，
因此策略、观测和历史 FIFO 保持 100 Hz（`0.002×5=0.01 s`）。策略输出在中间 4 个内环步零阶保持；
延迟后的 PD 目标按下文动作 FIFO 时序生效，可在策略周期中途切换。
气弹簧力在每个 2 ms 内环步重写。

`chuanliantui` 和继承它的 `chuanliantui_standup` 开启动作目标延迟随机化，
范围为 0–10 ms。每个环境在初始化时连续均匀采样、四舍五入到物理子步，
得到 0/2/4/6/8/10 ms；一个环境的延迟随后固定，并非逐拍抖动或等概率离散采样。
FIFO 每个物理子步入队一次，容量包含当前动作和最大延迟对应的历史动作；
回合重置会清空该环境的动作及 FIFO，防止上一回合的目标继续生效。
延迟发生在动作目标进入 PD 之前，PD 使用当前状态；它不单独模拟观测延迟或
电机力矩响应。已有 checkpoint 需续训或重训才能学习该随机化。
`play.py` 和默认 `eval_isaac.py` 会关闭动作延迟，属于无延迟对照；
`eval_isaac.py --domain_rand` 保留配置中的域随机化，但同时包含其他随机化项。
单独固定动作延迟使用 `--action_delay_ms <毫秒>`，实际值按物理子步量化并打印；不传 `--domain_rand` 时其他随机化仍关闭。
对比起立模型可用 `--standup_post_unlock` 在加载后统一解锁阶段，避免 checkpoint 的课程状态造成成功判据不同；`--metrics_report <新 JSON 路径>` 记录逐控制步轨迹和 5 秒之后的指标，重置步不参与连续指标，失败次数单列。
MuJoCo 零速诊断可在有限时长、无渲染回放上追加 `--diagnostic_report <新 JSON 路径>`，记录同一推理时刻的状态、原始 encoder 估计、动作、关节和接触数据。
`--diagnostic_true_vx --diagnostic_true_vx_after 5` 仅从第 5 秒起把送入 actor 的前向速度 latent 替换为仿真真值，按训练相同的 10 ms 根位置差分、当前姿态转机体系计算；其余输入不变，报告仍保留原估计。这是消融诊断，不是部署输入或通过验收的策略。
排查时同时核对实际关节角与限位：MuJoCo 软约束允许少量越界，不能仅凭两边 `range` 相同就认定限位执行一致；关节限位刚度、轮地接触刚度和控制 PD 的 Kd 是不同参数。
串联训练代理的 `lf1/rf1` 单独设置 `solreflimit="0.004 1"`，生成器和代理检查同步该值。
此调整只改变两膝限位约束的响应，保留 `lf1=[-0.12,0.77]`、`rf1=[-0.77,0.12]` rad
的物理范围，不改变全局接触参数、关节耗散或控制 PD；不能把该设置等同于实体闭链或真机验证。
队列上界、实际子步延后时长和部分回合重置可用
`python scripts/agent/check_action_delay.py` 做 CPU 检查；这不替代正式行为验收。

### chuanliantui 气弹簧支撑

训练环境与串联 MuJoCo 代理现默认加入每侧 150 N 恒定伸张气弹簧。
两端安装点与 CAD 闭链一致，训练按物理膝角求方向和力臂，每 2 ms 重算；
对上下连杆施加等大反向膝轴扭矩，与串联理想铰链上的端点力具有相同广义力。
它与水平推力合并后每子步只提交一次，不经过电机限矩、倍率或动作延迟，
`self.torques` 和电机能耗奖励仍仅含主动电机力矩。起立接管前也生效。
MuJoCo 代理通过独立的两个 tendon motor 表示弹簧，主动电机仍为六路。

`eval_isaac.py`、`mj_sim2sim_ct.py` 的 `--gas_spring_force 0` 可恢复无弹簧对照；
训练配置为 `ChuanliantuiCfg.gas_spring.force_n`。旧配置快照会继承当前父类，
仅加载旧快照不会自动关闭弹簧。旧 checkpoint 形状兼容但行为需重新评估/续训。
数值检查：`python scripts/agent/check_chuanliantui_gas_spring.py`；
正式起立、站立回放仍按 [命令速查](../../COMMANDS.md) 执行。
本轮没有把约 0.03 rad 的闭链虚拟/物理膝差写成补偿，也没有改变关节反馈。
150 N 是现有仿真假设，真机弹簧推力待测量核实。

该模式用于对齐 Isaac 串联训练资产与 MuJoCo 回放资产，**不是**真实闭链或
真机 sim2sim。默认会拒绝任何含 equality/connect 约束的模型，防止两条链路混用。

仅在闭链差异诊断时，可显式传入 `--closed_chain`。该选项加载本机
`chuanliantui.xml`（`neq=4`），实体前、后输入轴分别是
`lf0/lf00` 与 `rf0/rf00`。模型用两个 spatial tendon 模拟左右气弹簧；端点由本机
CAD 定位件换算到前支路，默认每侧恒定伸张推力 150 N，可用 `--gas_spring_force 0`
关闭做对照。复旦 XML 仅作为单独的模型参考，不参与本机策略回放。该恒力模型不含
真实气弹簧的力—长度曲线、阻尼或行程，也不能作为真机验证通过的证据。

闭链回放默认使用 `H7ClosedChainAdapter`，以用户指定的 H7 仓库源码复现五连杆解算、
反馈速度、环绕角差 PD、轮目标 ±20 rad/s、虚功映射和十拍预热。
CAD 机构、物理限位与气弹簧保留；状态估算不读取被动膝状态。后轴到 H7 坐标的默认值
来自 CAD 输入杆方向，只是名义几何注册，需共同姿态数据才能验证实机编码器对应。
H7 软件气弹簧补偿当前关闭，模型物理气弹簧仍保留。详见
[H7 闭链回放接口](../interfaces/chuanliantui-h7-closed-replay.md)。

加 `--closed_chain_controller cad` 保留原 `ClosedChainAdapter` 对照：它从本机 MJCF 的前/后
支路尺寸、零位装配分支和实体电机 `lf0/lf00`、`rf0/rf00` 的角度解两次平面圆交点，
得到训练策略所需的虚拟膝角；闭链几何方程的解析 Jacobian 把电机速度变为虚拟膝速度，
并按虚功关系把虚拟膝力矩映射到两个实体电机。轮力矩仍直通。这沿用复旦部署的
“电机状态→虚拟膝状态、虚拟力矩→电机力矩”路径，但使用 chuanliantui 自己的
CAD 几何，不复制复旦的连杆长度或偏置。几何不可达或 Jacobian 奇异会显式报错。
该 CAD 对照的虚拟膝速度来自电机速度和几何 Jacobian，其余 DOF 仍按 2 ms 子步差分。
H7 路径则全部从主动轴反馈速度构造，MuJoCo 用 `qvel` 表示反馈，不模拟 CAN 量化与异步到达。

用 `python sim2sim/mj_sim2sim_ct.py --gas_spring_view` 打开气弹簧专用视图（自动开启
闭链和渲染）：半透明机构中，左侧青色、右侧橙色，圆点为安装端点，粗线为气弹簧轴线。
显示加粗仅影响渲染，不改变气弹簧力或碰撞几何；每侧默认仍为 150 N。

如需固定机身、手动拖动腿部观察气弹簧，运行
`python sim2sim/view_gas_spring_ct.py`。该独立演示在内存中移除基座自由关节，
将机身固定于 0.8 m，不加载策略，六个实体电机输出始终为零，只保留每侧 150 N
气弹簧和重力。双击腿部后 Ctrl+右键拖动施力；空格暂停/继续，Home 复位，F5
开关气弹簧。`--headless --duration 10` 可运行无窗口检查；原始 XML 不被修改。
机身上方实时显示左右气弹簧对 `lf1/rf1` 的直接关节力矩（N·m）、轴向力（N）和
有效力臂（mm）。力矩按 `actuator_force * actuator_moment` 计算，符号遵循各自
关节轴，不包含重力、鼠标外力和闭链约束反力，也不是主动电机轴的等效补偿力矩。

复旦原腿的固定机身对照：`python sim2sim/view_gas_spring_fudan.py`，使用 `--xml`
指定本机参考仓库中的原 XML，不依赖脚本内某位成员的默认路径，
每侧使用 150 N 便于同力对照；`--gas_spring_force 300` 恢复原脚本的力值。
该视图无策略、无实体电机驱动，保留原模型的限位、阻尼和闭链锚点；在内存中把
`connect site1/site2` 转为 MuJoCo 3.2.2 支持的两刚体局部锚点，不修改源 XML。
复旦模型的 `l20/r20` 不限位；当前闭链 XML 的 `lf00/rf00` 也已改为
`limited="false"`，允许后输入轴连续旋转，避免原 ±3.14 rad 限位阻挡髋部转过一圈。
这是 MuJoCo 模型对原始 CAD URDF 限位的显式修正；闭链生成器已同步该规则和气弹簧定义，
重新生成后运行 `python scripts/agent/check_chuanliantui_closed.py`，检查左右膝限位、
端点连接、每侧 0/150 N 施力、±360° 闭合姿态以及原有的 5 秒无控制仿真。
用户确认实机膝关节行程以 `chuanliantui_train.urdf` 为准：
`lf1=[-0.12,0.77]`、`rf1=[-0.77,0.12]` rad。原始 `chuanliantui.urdf`
的这两处限位曾与实机/训练资产相反，现已修正。闭链 XML 中的膝限位已同步；
生成器也直接读取修正后的源 URDF。其余被动关节仍保留原始 CAD 范围。
原始四对 site 存在约 10 mm 的横向错位，加载后的演示保留此偏差，不能视为闭合精度验证通过。

## chuanliantui 站高精度与膝余量续训

2026-10-03 的站立续训配置采用解锁后目标 **0.22 m**、成功高度门槛 **0.20 m**，
连续满足高度、`pg_z<=-0.90` 和机身离地条件 0.5 s 后记为已起立。解锁前仍使用
0.30 m 目标、0.28 m 成功门槛。初始位置与后摆关节姿态不变。
当前模型在近零 pitch、两膝距硬端点至少 0.05 rad 的几何条件下，最低站高约为
0.218 m；因此 0.20 m 站高与这两项约束不能同时满足，当前选用 0.22 m 目标。
该几何估算不等于策略已经能达到目标。

训练仍按 `dt=0.01 s` 缩放奖励，并将每项裁剪到 `[-dt,dt]`。站立任务的调整为：

| 项目 | 当前计算与作用 |
|---|---|
| 高度项 | 权重从 2 改为 1，原始值为 `exp(-(h-h_cmd)^2/sigma) * gate`；`gate` 随机身接触力由 1 降至 0.2，因此目标附近不再被单项裁剪成相同得分。 |
| 高度容差 | 解锁前 `sigma=0.01 m²`，解锁后 `sigma=0.0025 m²`；解锁后误差 5 cm 时原始指数项为 `exp(-1)`。解锁前保留宽容差，但高度权重同样使用 1。 |
| 膝端点余量 | 从 URDF 原始 DOF 属性保存两膝硬限位，在两端各设 `m=0.05 rad` 奖励区；按 `d=max(L+m-q,q-U+m,0)/m`，取两膝 `d²` 的均值，权重 `-0.3`。 |
| 起立门控 | 新膝余量项仅在 `has_stood=True` 后激活；本回合曾连续站稳 0.5 s 后保持激活，到 reset 清除。旧 `dof_pos_limits` 项继续保留。 |

两膝恰在硬端点时，新余量项每策略步为 `-0.003`；端点外的小超限继续增罚。
它是奖励安全区，不修改物理 `range` 或裁剪 PD 位置目标，也不能保证策略一定留足余量。
单项裁剪仍存在：两膝同时越过硬端点约 0.0413 rad 后，该新项会达到裁剪下限。
姿态奖励仍乘高度进度门控；例如目标 0.22 m、实际高度 0.208 m 时门控约为 0.829，
因此应同时检查实际高度、pitch 和两膝到硬端点的距离，不能只看恢复率。

CPU 回归检查调用真实的奖励准备和计算路径，覆盖实际裁剪后的高度区分、接触门控、
硬限位保存、膝余量的对称性与起立门控：

```bash
python scripts/agent/check_standup_precision_rewards.py
```

该检查不创建仿真或训练产物，不能代替正式训练、Isaac 回放和 MuJoCo 行为验收。
续训命令见 [命令速查](../../COMMANDS.md#当前推荐022-m-站高与-005-rad-膝余量续训)：
从 `Oct03_03-28-58_standup_action_delay_0_10ms/model_9000.pt` 额外训练 3000 轮，
当前叠加下述推扰，使用新的 `standup_h022_knee005_push` run 名及日志目录；保留旧模型做同条件对照。
恢复该已解锁 checkpoint 后，课程命令按当前配置切换为 0.22 m，成功高度为 0.20 m。

此次没有改变观测、动作、PD 或模型输入输出形状，旧 25 维 checkpoint 仍可加载，
但旧模型没有因配置更新而自动学会新高度与余量要求；当前推荐回放显式使用
`--cmd_vx 0 --cmd_height 0.22`，历史 0.20 m 对照应另行标注。
本次未自动启动训练，尚不能据此宣称新目标训练成功或真机问题已修复；H7 代码未修改。

### 接地后分档随机推力

`chuanliantui_standup` 开启专用水平推扰，其他机器人继续使用各自配置与基类实现。
首次任一轮接地后，独立的每环境计时开始；初始自由下落阶段不推。每次脉冲开始时：

- 当前满足 `has_stood` 且连续站稳时间达到 0.5 s：合力模长均匀采样 5–15 N。
- 尚未站稳或当前失稳：合力模长均匀采样 1–5 N。
- 环境坐标系水平面方向均匀采样，竖直力与额外外力矩为零；力施加于 `base_link` 质心。
- 恒力持续 0.10 s，脉冲内不随姿态或稳站状态重采样。每次结束后留出 3–5 s 随机空档。
- 调度按 10 ms 策略步推进，力在每个 2 ms 物理子步重提交；不直接设置根速度。
  单次冲量模长分别为 0.1–0.5 / 0.5–1.5 N·s。回合重置仅清除对应环境状态。

参数均在起立配置的 `domain_rand.standup_push_*` 中。这些是首轮训练强度，未由真机
手推测量标定。站高、膝余量、动作延迟及观测/动作接口保持当前设置；本轮没有把
H7 的轮速目标 ±20 rad/s 裁剪加入训练，推扰不能替代这项控制一致性工作。

`python scripts/agent/check_standup_pushes.py` 检查分档、时序、实际提交张量和重置。
正式行为检查使用 `eval_isaac.py --pushes`，可叠加 `--action_delay_ms 6` 单独比较；
`--domain_rand` 则保留全部训练随机化。默认回放关闭推扰。`--metrics_report` 保存
实际推力和每回合触发次数，非零施力时长按 100 Hz 采样估算并排除重置步。

`standup_push_count` 记录每环境本回合的实际触发次数，回合重置清空，
通过评估报告逐步检查覆盖和推后状态。训练 `recovered_rate` 仍只表示曾起立，
不把它或重置批次平均值当作逐次抗扰成功率。
