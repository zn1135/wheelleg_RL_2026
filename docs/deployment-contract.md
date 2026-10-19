# 训练与部署接口约定

H7部署工程作为普通目录保存在[deployment/H7_RL](../deployment/H7_RL/md/file-guide.md)，与训练代码统一管理。2026-10-18已同步独立H7仓的 `e2a3339519add73805df531e6247233a9540fbf0`；生成网络为 `model_5849`，actor39/action6、500Hz实体PD、50Hz推理，并接入平地遥控与倒地恢复。同步范围、LQR整定、构建记录和待复查项见[部署同步记录](interfaces/20261018-h7-model5849-deployment-sync.md)；部署规则见[H7协作规范](../deployment/H7_RL/md/AGENTS.md)。源码同步不表示独立复查或真机验收通过。

## 新 mjlab 实体关节接口

mjlab 训练侧只保留华南虎 multimode：actor39／critic82／action6，2 ms 物理、20 ms／50 Hz 策略，动作顺序 `[lf0,lf00,lfwheel,rf0,rf00,rfwheel]`，模式槽 `28:35`、标志槽 `35:39`。观测打包与 ONNX 契约见 [multimode 接口](interfaces/chuanliantui-huananhu-multimode-replay-v1.md)、[model_3023 说明](onnx-huananhu-multimode-model3023.md)及 [model_5849 说明](onnx-huananhu-multimode-model5849.md)；当前基线、模型采纳与验收结论以[项目实验记录顶部](ai/project-log.md)为准。

当前H7生成网络为 `model_5849`，输入39、输出6，不需要encoder历史输入；默认大机为500Hz采样／PD／正常下发、50Hz板载推理。命令复用公共 `rc.vel/yaw/len`，在RL观测入口限制到±1.5m/s、±1.5rad/s、0.27～0.34m；公共腿长范围可能进一步限制可达高度命令，不能把模型包线当成遥控全行程。

当前已实现平地NORMAL／RECOVER；趴地投入保持200ms，连续稳定500ms转NORMAL，连续倒地严格超过2000ms锁定失败，须退出投入重试。髋轮投影间距代替仿真绝对机身高度／接触测量，airborne／step_active仍固定0，未实现跳跃、台阶、坡地模式。H7负责人对板端数值、真实时序、物理映射和恢复行为的核对状态仍为**待确认**；更换模型须成套核对权重与接口，不能只替换尺寸宏。

| 数据 | 顺序／用途 |
| --- | --- |
| H7内部反馈、观测电机段及PD诊断 | `[lf0, lf00, rf0, rf00, lfwheel, rfwheel]` |
| 网络原始动作及上一拍动作缓存 | `[lf0, lf00, lfwheel, rf0, rf00, rfwheel]`；写入观测时按 `[0,1,3,4,2,5]` 重排 |
| `q_nominal` | 观测减去的默认角，也是腿目标中心 |
| `q_reference` | XML的 `qpos0`，用于目标限幅中心；不是实机编码器零点 |

<a id="host-debug-plan"></a>
## 实机调试与上位机（只读镜像已实现，待联机）

2026-10-18用户批准实施只读镜像：H7上传公共电机／IMU反馈，Jetson重构闭链姿态；不依赖ROS 2，
不回发电机命令。本轮固件与接收／显示软件已经实现并通过软件检查，H7尚未烧录、真实串口与物理映射
仍待台架确认。旁路策略复算、同指令独立仿真及电脑接管继续属于后续规划。

Jetson环境准备（2026-10-18）：已通过SSH在Orin Nano／Ubuntu 24.04／Python 3.12上创建独立
`~/real2sim/.venv`，安装MuJoCo 3.11.0、NumPy 2.5.3、SciPy 1.18.1与pyserial 3.5；复制当前multimode XML、
15个网格、姿态库、当前大机机器配置及镜像程序，保留仓库相对目录。模型哈希、加载与NVIDIA OpenGL
窗口显示通过；先前`check_viewer.py`只是静态环境检查，实际入口改为`sim2sim/real2sim_huananhu.py`。
当次未发现USB串口设备，真实联机仍待确认。

```bash
# Jetson终端；设备名应取插入H7串口转换器后实际出现的by-id路径
cd ~/real2sim
.venv/bin/python sim2sim/real2sim_huananhu.py --port /dev/serial/by-id/<实际设备>
# 只读串口与模型检查：Jetson需已有cc，输出须使用新文件名
.venv/bin/python scripts/agent/check_real2sim_huananhu.py --out artifacts/<新名称>/check.json
```

默认1152000波特、8N1。大机UART8的TX源码为PE1，接转换器RX并共地，电平／板上接口现场核对；
首期不接H7 RX，也不连接转换器DTR／RTS作复位。屏幕位置和高度固定，均不是实测；闭链残差门槛1mm。
被动关节即使满足闭合残差，超出XML行程仍判无效并回滚，避免显示错误装配分支。
任一部位失效保留最后有效姿态并明确标记，200ms无合法新帧标STALE；红色初态不是收到实机数据。
串口仍有新帧但IMU／电机软件发布年龄达到200ms、未发布或非法时，对应来源同样冻结并标记过期。
镜像帧与兼容性见[本次接口记录](interfaces/20261018-h7-jetson-mirror-v1.md)。

| 阶段 | 拟完成内容 | 可观察的成功判据 |
| --- | --- | --- |
| 只读采集与姿态镜像 | 软件已实现；公共原始反馈、时间、序号、在线状态与闭链重构；实际串口／方向待台架 | 软件故障检查通过，联机后须逐电机手动核对动作／单位／方向 |
| 旁路策略复算（尚未实现） | 上传同一次推理配对的obs39／action6，电脑用同权重模型复算 | 有效配对数据的动作误差达到交付数值阈值 |
| 同指令独立仿真 | 实机和MuJoCo各自闭环运行策略，从零速静站开始比较关节、倾角、轮速和动作 | 同一记录可重复分析；分别报告误差与实验条件，不强求长时间轨迹重合 |
| 电脑接管策略（后续候选） | 电脑50 Hz输出动作，H7保留500 Hz电机控制与唯一出力门控 | 验证延迟、动作过期、控制段切换、断线和重新投入；通过前不作为首期功能 |

遥测沿用后台固定布局、一次发送方式；当前48通道改为镜像v1，旧LQR／RL布局完整保留行注释，
不按模式分页。当前有四元数／原始反馈／序号／时间／CRC，但没有obs39或action6，不得把镜像帧当作
网络输入／动作配对证据。后续复算仍须在同一次推理完成后配对保存，不能用发送序号补救异步发布。

不同对比回答不同问题：

- 同一obs输入的网络对比只核对推理数值，不证明实机观测的缩放、方向或零点正确。观测构造对比须使用同一份未缩放原始输入。
- PD对比须使用同一个控制子步的角度、速度和动作；不能拿50 Hz观测反缩放冒充500 Hz当拍反馈。
- 实机镜像按测量更新模型姿势；新增求解器固定lf0/lf00、rf0/rf00，重构被动关节与弹簧，以前一姿态初始化。
  原站姿解算仍只服务sim2sim，不改它；`mj_forward`不会自动求闭链。镜像仅用运动学，不运行物理步进。
  驱动已处理公共极性／零点和轮减速比，Jetson仅追加既有RL模型坐标映射一次，物理正确性仍待台架。
- 录制action后回放是开环动作序列诊断；实机与仿真分别读取各自状态运行策略，才是同指令独立闭环。两者都不能反推物理映射已正确。

电脑接管还须接通收包、明确唯一动作来源、处理上一动作与复位，并拒绝旧控制段的在途动作。现有出力许可和动作超时仍依赖板载 `policy.ready`，不能从后台直接写 `action_state` 绕过现有门控。UART8可双向收发，不强制另占UART；未来控制收包须独立定义定长字段、校验与时效，当前JustFloat遥测帧不能直接作为控制协议。

当前串口布局、未接通的接收链和时序边界只在 [H7总览的VOFA节](../deployment/H7_RL/md/file-guide.md#vofa)维护。旧S2R1／HPI1工具见[历史上位机说明](ai/s2r-offline.md)，不能直接套用到当前39维固件。

## Isaac旧接口（历史）

以下表格与左右映射仅描述旧Isaac／串联代理版本，不定义当前mjlab与H7 RL39的顺序。旧模型与历史交付记录继续按各自版本解释。

2026-10-08 更正CAD左右命名：原 rf→当前 lf（+y实际左），原 lf→当前 rf（−y实际右）。旧Isaac／H7数字通道及极性保留，当前名称分别为 `[rf0, rf1, rfwheel, lf0, lf1, lfwheel]` 和 `[rf0, rf00, rfwheel, lf0, lf00, lfwheel]`；下面历史版本中的名称按其记录版本解释。

本页按机器人索引接口，供训练仓与 [H7_RL](https://github.com/zn1135/H7_RL.git) 协作使用。下表依据训练仓 `26_wheelleg` 的 `f98b8bfcc70b4be7ec1fe82850fad0f3159f2afc` 代码核对，只描述训练与 MuJoCo 接口，不表示 H7 固件已对齐或真机验证通过。

imcawl 的权威来源是 [mj_sim2sim.py](../sim2sim/mj_sim2sim.py) 文件头部署契约及训练实现；chuanliantui 使用 [mj_sim2sim_ct.py](../sim2sim/mj_sim2sim_ct.py)、[训练配置](../wheel_legged_gym/envs/chuanliantui/chuanliantui_config.py) 和 [观测实现](../wheel_legged_gym/envs/chuanliantui/chuanliantui.py)。说明与代码不一致时先核实版本并记录差异，不静默采用另一机器人的参数。

### 旧接口索引

接口标识用于交付记录，由团队维护；运行程序目前不会自动检查这些标识。

| 项目 | imcawl | chuanliantui 串联训练代理 |
|---|---|---|
| 接口标识 | `imcawl-27d-100hz-r1` | `chuanliantui-25d-100hz-r1` |
| 任务 | `mini_wheel_legged` | `chuanliantui`、`chuanliantui_standup` |
| 前向轴 | 机体 +y | 机体 +x |
| 命令通道 | 前向速度、航向保持外环输出、目标高度 | 前向速度、偏航角速度、目标高度 |
| yaw 处理 | `heading_command=True`；航向误差经回绕、乘 1.5、限幅 ±5，每策略步更新 | `heading_command=False`；直接使用偏航角速度命令 |
| 当前观测／历史／latent／动作维度 | 27／135／3／6 | 25／125／3／6 |
| 关节及动作顺序 | `[lf0_joint, lf1_joint, l_wheel_joint, rf0_joint, rf1_joint, r_wheel_joint]` | `[lf0, lf1, lfwheel, rf0, rf1, rfwheel]` |
| 零动作默认角（rad） | `[0.9, -1.62, 0, -0.9, 1.62, 0]` | `[-0.06, 0.10, 0, 0.06, -0.10, 0]` |
| 策略／PD 频率 | 100 Hz／200 Hz（dt=0.005、decimation=2） | 100 Hz／500 Hz（dt=0.002、decimation=5） |
| 腿 Kp／Kd；轮 Kp／Kd | 60／2；0／0.5 | 10／1；0／0.1 |
| 力矩限幅（N·m） | `[30, 30, 5, 30, 30, 5]` | `[40, 40, 3.9, 40, 40, 3.9]` |

其他任务（包括 `XML` 分支的 xwl、wl 的 VMC 任务）不能直接复用此表，首次交付时按 [接口模板](templates/interface-change.md) 核对各自实现。

### chuanliantui 模型与实物左右

chuanliantui 的机体系为 **+X 向前、+Y 向机器人自身左侧、+Z 向上**。但[训练 URDF](../resources/robots/chuanliantui_new_1/urdf/chuanliantui_train.urdf)中，`lf` 腿根位于 −Y，`rf` 腿根位于 +Y。因此，模型的 `lf/rf` 命名与按机体系定义的实物右/左相反：

| 机器人自身方位 | 训练模型侧 | 当前记录的实体 DM 槽位 |
|---|---|---|
| 左侧（+Y） | `rf` | DM0、DM1 |
| 右侧（−Y） | `lf` | DM2、DM3 |

DM 槽位的左右归属来自现场确认，仍需在电机失能状态下按实物位置复核。此表只说明左右对应；电机前/后输入轴、转向符号与零点须分别验证，不能从 `lf/rf` 名称推断。

## 观测顺序和历史

以下为从 0 开始的半开区间。角速度单位 rad/s，关节位置 rad，关节速度 rad/s，前向速度 m/s，高度 m；投影重力为机体系中的单位重力方向。

| 内容 | imcawl 索引 | chuanliantui 索引 | 处理 |
|---|---|---|---|
| 机体系角速度 | `0:3` | `0:3` | ×0.25 |
| 投影重力 | `3:6` | `3:6` | 不缩放 |
| 三通道命令 | `6:9` | `6:9` | ×`[2.0, 0.25, 5.0]`，yaw 含义见上表 |
| 相对默认角的位置 | `9:15`，六关节 | `9:13`，仅 `[lf0, lf1, rf0, rf1]` | ×1.0；chuanliantui 不输入轮绝对位置 |
| 六关节速度 | `15:21` | `13:19` | ×0.05 |
| 上一次动作 | `21:27` | `19:25` | 控制尺度缩放前，随整体观测裁剪至 ±100 |

观测整体裁剪至 ±100。每个策略步构造当前观测，移除最旧帧并追加当前帧，再推理；历史最后一帧等于本次观测。初始化时首帧重复五次，复位时清空旧动作与历史后重新填充。网络不维护历史状态。

## 动作、时序与模型

两个已记录接口均先将动作裁剪至 ±100。腿索引为 0、1、3、4，目标角为 `default + action × 0.5`；轮索引为 2、5，目标角速度为 `action × 10.0`。PD 根据上表增益计算力矩后再限幅，不能将网络输出直接当成电机力矩。

串联代理内环顺序为“算力矩 → 仿真步进 → 位置差分更新 dof_vel”；速度差分为 `wrap_to_pi(Δq)/sim_dt`。策略步之间保持动作。imcawl 的 MuJoCo 状态读取修复与延迟约束见 [sim2sim 说明](ai/sim2sim.md)，变更采样方式或控制频率要同时更新接口记录。

chuanliantui 起立任务与站立任务共享张量布局，但初态和接管条件不同，不能据此认为模型行为可互换。`--standup` 使用 0.15 m 后摆初态；首次轮接地前保持零策略动作并继续 PD 与历史更新，从接地后的下一策略步开始推理。真实闭链还需要实体电机到虚拟膝状态、虚拟力矩到实体电机的映射，详见 [闭链适配器](../sim2sim/chuanliantui_closed_adapter.py) 和对应 sim2sim 说明；串联代理通过不代表该映射或真机通过。

上述首次接地门控用于串联代理及原 CAD 控制对照。当前 `--closed_chain` 默认在 CAD
机构上使用 H7 五连杆状态解算、反馈速度、环绕 PD、轮目标限幅和十拍预热；
`--closed_chain_controller cad` 保留旧行为。H7 默认高度为 0.20 m，0.22 m 是显式测试覆盖。
角度注册、失效处理、版本与验证结果见[H7 闭链回放接口](interfaces/chuanliantui-h7-closed-replay.md)。

2026-10-01 的电脑推理／H7 执行候选使用独立 USB `2901` 会话：25/125 浮点观测历史以100 Hz由 H7 发送，电脑载入完整 `model_6000.pt` 运行 encoder+actor，H7 在2 ms循环计算 PD 与闭链虚功力矩。用户确认 B 机械身份：DM1/DM3 驱动各自 CAD 前长输入，DM0/DM2 驱动 CAD 后短输入；实物左侧 DM0/1 对训练 `rf`，实物右侧 DM2/3 对训练 `lf`。原 `dm_sign`/`dm_zero` 逐槽保持；同姿态 SolidWorks 右髋到轮轴尺寸拟合出的 CAD 输入角候选偏置为奇数槽0.8138714045 rad、偶数槽2.3227884081 rad。该单姿态 CAD 拟合不验证实体方向、左侧零点或力矩；源码中的执行资格和六路实体限值均锁定，不能将本段视为已部署。详见[设计规格](superpowers/specs/2026-10-01-chuanliantui-host-policy-h7-standup-design.md)及[模型使用记录](model-deliveries/20261001-chuanliantui-h7-policy-execution.md)。

训练与现有 MuJoCo 脚本使用包含 encoder 的完整 `model_*.pt`；`policy_1.pt` 缺 encoder，不能作为部署输入。chuanliantui 的板端导出使用包含 encoder + actor 的 ONNX，输入 `observations`（25）和 `observation_history`（125），输出 `actions`（6）和 `latent`（3），float32。H723 的固定 batch=1、opset 13 导出方式及工具兼容范围见 [ONNX 导出说明](../ONNX导出说明.md)。模型图不包含观测预处理、历史、PD 或起立接管逻辑。

## 接口变更与双仓库交付

1. 使用 [接口记录模板](templates/interface-change.md) 创建新记录，列出旧、新接口标识与每项差异；旧定义保留用于历史权重。形状不变但顺序、单位、缩放、坐标、动作或时序改变，也算兼容性变化。
2. 明确 checkpoint 能否加载、策略行为是否仍适用、是否需要重新训练或导出；旧 27 维 chuanliantui checkpoint 不兼容本页 25 维接口。
3. 在训练仓与 H7_RL 的 PR 中互相链接，分别列出需同步文件、目标版本、负责人及合入顺序。未核实的板端情况写“待核实”。
4. 将代码 SHA、接口标识、模型校验值及各阶段结果写入 [模型交付记录](templates/model-handoff.md)。形状检查、数值对齐、仿真行为、板端时序与真机行为分别给出结论。
