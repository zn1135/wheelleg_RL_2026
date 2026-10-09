# 训练与部署接口约定

H7 部署工程已作为普通目录纳入 [deployment/H7_RL](../deployment/H7_RL/README.md)，与训练代码统一提交。导入的是旧25/125维、100Hz策略部署工程，内置网络仍为 `model_15000_h723`；纳入同仓不表示已适配下面的新 mjlab 实体接口。源码指纹及eIDE引用问题见部署目录说明，原有接口历史继续保留。

## 新 mjlab 实体关节接口

`chuanliantui-mjlab-huananhu-v14-flat-standup-r1` 使用 actor35／critic78／action6，实体顺序 `[lf0, lf00, rf0, rf00, lfwheel, rfwheel]`；四腿位置PD、两轮速度控制，50Hz。完整字段、缩放、延迟、哈希校验和导出见 [mjlab接口](mjlab-chuanliantui.md)，兼容性与同步见 [独立接口记录](interfaces/chuanliantui-mjlab-huananhu-v14-flat-standup-r1.md)。本次没有训练模型交付、没有H7固件适配或真机验收；新ONNX不可直接替换旧虚拟关节模型。导出时生成同名JSON记录checkpoint和ONNX SHA256、opset及数值误差。

2026-10-08 更正CAD左右命名：原 rf→当前 lf（+y实际左），原 lf→当前 rf（−y实际右）。旧Isaac／H7数字通道及极性保留，当前名称分别为 `[rf0, rf1, rfwheel, lf0, lf1, lfwheel]` 和 `[rf0, rf00, rfwheel, lf0, lf00, lfwheel]`；下面历史版本中的名称按其记录版本解释。

本页按机器人索引接口，供训练仓与 [H7_RL](https://github.com/zn1135/H7_RL.git) 协作使用。下表依据训练仓 `26_wheelleg` 的 `f98b8bfcc70b4be7ec1fe82850fad0f3159f2afc` 代码核对，只描述训练与 MuJoCo 接口，不表示 H7 固件已对齐或真机验证通过。

imcawl 的权威来源是 [mj_sim2sim.py](../sim2sim/mj_sim2sim.py) 文件头部署契约及训练实现；chuanliantui 使用 [mj_sim2sim_ct.py](../sim2sim/mj_sim2sim_ct.py)、[训练配置](../wheel_legged_gym/envs/chuanliantui/chuanliantui_config.py) 和 [观测实现](../wheel_legged_gym/envs/chuanliantui/chuanliantui.py)。说明与代码不一致时先核实版本并记录差异，不静默采用另一机器人的参数。

## 已记录接口

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
