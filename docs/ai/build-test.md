# 构建与测试

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
