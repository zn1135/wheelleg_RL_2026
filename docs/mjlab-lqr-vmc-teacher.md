# mjlab LQR+VMC teacher

第一阶段仅运行传统控制器：平地站立、速度／偏航角速度／腿长命令及 H7 正常起立。不含倒置恢复、SMP、PPO 训练或固件修改。验收结果见[实验记录](ai/project-log.md)顶部。

## 学生PD接口闭环实验（2026-10-25）

新增独立`scripts/agent/check_lqr_student_closed_loop.py`，按真实反馈每10ms更新LQR反解位置/轮速目标，由原2ms学生PD/FIFO执行，不重放固定目标、不训练。原teacher默认仍1ms直接力矩；控制器新增可选dt仅用于显式周期换算。无延迟和2/2ms各4/4通过末段平衡，完整6/8/10ms延迟网格24/36，10ms轮延迟持续抖动，尚不能作为全面合格示范采集入口。平衡、末段停车及腿部平滑分开报告。[实验定义](interfaces/20261025-lqr-student-closed-loop.md)，[实际结果、视频及命令](../artifacts/lqr-student-closed-loop-20261025/README.md)。

后续可选`--compensation wheel-lqr --high-rate`以当前状态和FIFO在独立影子模型预览下一拍LQR轮输出，保留真实延迟和增益；本轮4初态完整延迟组合两次均44/44平衡、实际训练网格36/36，原成功无丢失。只预测轮速的方案退化，未采纳。补偿尚未默认接入teacher/collect/train，也不表示PPO权重已改善或全姿态/随机化已验收。[补偿实现、视频、证据与限制](../artifacts/lqr-delay-compensation-20261025/README.md)。

## 使用

从仓库根目录运行；使用 `scripts/mjlab.sh` 指向的独立 mjlab 环境。

```bash
# 默认：单环境、可视化、站立初态、零速度、H7默认腿长0.15m、20秒
scripts/mjlab.sh teacher

# 20环境可视预检；打开终端打印的viser地址查看
scripts/mjlab.sh teacher --num-envs 20 --viewer viser --seconds 20

# 正常趴地起立（同侧向的闭链姿态库初态，不包含倒置）
scripts/mjlab.sh teacher --initial ground --num-envs 20 --viewer viser --seconds 20

# 仿真速度误差积分扩展：正常起立与运动示范使用，不是原H7逐拍等价模式
scripts/mjlab.sh teacher --initial ground --num-envs 20 --leg-length 0.30 --velocity-integral
scripts/mjlab.sh teacher --vx 0.5 --leg-length 0.30 --velocity-integral

# 对齐SMP采集/PPO：前0.2秒仅重力与弹簧作用，然后启动正常起立
scripts/mjlab.sh teacher --initial ground --num-envs 20 --leg-length 0.30 \
  --velocity-integral --ground-start-hold 0.2

# 固定命令：前向m/s、偏航rad/s、髋轴到轮轴的腿长m
scripts/mjlab.sh teacher --headless --vx 0.5 --yaw 0 --leg-length 0.22 \
  --seconds 12 --seed 42 --output artifacts/teacher-forward

# 预设：静站→前进→后退→左转→右转→伸腿→收腿
scripts/mjlab.sh teacher --sequence mjlab_training/configs/teacher_motion.json --seconds 28
```

`--device` 默认 `cuda:0`。`--viewer auto` 有桌面时用 native，否则用 viser；提前关窗会记为未完成。每次输出目录必须不存在。没有 `--output` 时写入 `logs/mjlab/teacher/<时间>`。时长按仿真时间计算，1ms Python 控制循环目前慢于实时。

序列是 JSON 数组，每行必须有 `time/vx/yaw/leg_length`；从 `time=0` 开始且严格递增。命令保持到下一时间点。vx、yaw 均限 ±1.5；腿长命令限 0.13～0.34m，不代表机器人能达到整个范围。手动复位清除控制状态，报告会标记复位，不将多回合拼成一次通过。

## 接入与坐标

| 内容 | 实现 |
|---|---|
| 增益、五连杆几何、VMC（把腿端力换成电机力矩）、补偿 | `mjlab_training/src/ct_mjlab/teacher/gains.py`、`mechanics.py` |
| LQR、腿长／横滚控制、速度和离地估计 | 同目录 `controller.py`、`estimators.py` |
| 正常起立状态机 | 同目录 `standup.py` |
| 独立环境、六电机直接力矩动作、复位 | 同目录 `env.py` |
| 命令、viewer、记录与判据 | `mjlab_training/src/ct_mjlab/scripts/teacher.py` |

参考源为 **USB 大机** `deployment/H7_RL_ros/imcalib/`：`Algorithm/lqr_balance.c`、`lqr_gain_big.c`、`leg_solver.c`、`leg_balance.c`、`gas_spring.c`、`gravity_comp.c`、`standup.c`、`slip.c`、`air_detection.c`，参数来自 `config/machine_config.c`。正常输出组合参考 `task/task_actuation.c`，前髋偏置参考 `task/task_comm.c`。

仿真电机顺序为 `[lf0, lf00, lfwheel, rf0, rf00, rfwheel]`；H7 内部顺序为四髋再双轮。先用索引 `[0,1,3,4,2,5]` 重排，再乘 `[1,1,-1,-1,1,-1]`。仅前髋位置加 π，速度不加。输出力矩使用同一符号逆变换，保持关节功率一致。这里使用已标定坐标，不再次叠加原始编码器零点或驱动极性。

仿真前向为机体 +x，俯仰为绕 +y，偏航绕 +z，角度统一 rad。H7 的 `ATTITUDE_ROLL` 槽实际上供 LQR 作俯仰读取；移植后直接使用对应物理轴，不按枚举名称猜轴。离地检测使用原力反馈算法；接触传感器仅用于验收，不参与控制。

物理与控制均为 0.001s，`decimation=1`。H7 的 PD 阻尼使用逐拍误差差值，**不除以 dt**。腿长轨迹首次从实测播种，此后沿上一拍给定推进；不在每拍重置来削弱控制刚度。所有历史都是按环境分开的张量，支持局部复位。

正常起立沿 `EXTEND→REAR→RETRACT→SWING→DONE`，到位会跳过已完成阶段；DONE 才交给 LQR。保留正常阶段超时与最多三次重试。机体上方向与世界上方向点积 ≤0 时锁停六电机，故障为 `inverted`，只允许显式复位；没有 SETTLE/TUCK/FLIP 倒置恢复阶段。数值无效也锁停对应环境。

`--ground-start-hold`默认0，只有正常趴地初态才生效。设置0.2时，前200个物理步主动电机请求为零，实体弹簧不关闭；等待时清空控制器历史，第201步根据当前真实状态接管，局部复位只重新等待相应环境。`control_active`记录每拍是否已到接管时间；控制器故障是否停机仍看`fault`，两者含义不同。用于SMP的`collect`入口默认等待0.2秒。

## 与 H7 的明确差异

- **停车目标**：仿真零速保持停车原点，位置目标为 0；原 H7 的固定位置目标为 0.4m。增益、几何、补偿和限幅数值对照使用相同输入目标，目标选择差异单独记录，不能宣称整个执行任务逐拍等价。
- **仿真测量**：使用 MuJoCo 理想姿态／角速度，前向加速度由世界速度差分并投影得到，再进入 H7 原速度滤波器。未复现 HI229 采样噪声、板端通信与真实反馈延迟。
- **可选速度积分**：`--velocity-integral`把`∫(估计速度−速度命令)dt`放入原LQR位置反馈槽，限±3.5m，仅在平衡且未腾空时更新，交接、落地与复位清空对应历史。它消除移动命令下参考实现反复清零位置槽造成的持续偏差；不新增拟合偏置、不改增益或机器人参数。默认关闭，原H7数值对照仍针对关闭模式，扩展模式另验积分、限幅、并行复位与仿真行为。
- **物理保持**：共享 XML 质量约13.394584kg、轮半径0.06m；H7 支撑前馈采用20.534kg，速度估计半径0.0525m。保留这些差异，不修改机器人参数迎合控制器。原 XML 最短闭链腿长约0.17163m，H7 默认0.15m无法精确达到。
- **力矩边界**：先按 H7 四髋54Nm、双轮4.84375Nm限幅，再按共享 XML 四髋40Nm、双轮请求3.9Nm限幅。轮端速度响应使用现有 `wheel_response_input`，XML固定增益G只施加一次，实际执行器力矩另行记录。
- **弹簧**：保留共享 XML 两路实体弹簧力；H7 弹簧抵消和腿自重补偿各加一次。原控制器的弹簧模型与实体弹簧模型是两种用途，不能把抵消项当成第二根实体弹簧。AIR 释放轴向主动控制及弹簧抵消，保留重力补偿。

teacher 单独构造环境，不注册训练策略、不调用 runner、不更改共享 XML 或 PPO 配置。CLI 的 `train/play/eval/export/check` 分支保持原有入口。

## 记录与正式检查

`result.json` 记录来源哈希、命令、运行时长、每环境故障、末段平衡与跟踪指标。`trace.npz` 按100Hz记录姿态／速度、关节状态、H7腿长／XML实测腿长、LQR状态和输出、分阶段力／补偿、限幅前请求／H7限幅后／XML限幅后／实际执行器六路力矩、轮支撑、机身接触、AIR、起立阶段。物理和控制仍是1000Hz，验收累计每拍进行。

同一记录行中 `root/motor_q/motor_dq/xml_length` 是步进后状态；`lqr_state/length/theta` 和控制请求来自该步步进前的测量，二者相差1ms。本阶段轨迹用于控制诊断，尚未构造训练所需的观测／动作配对数据集。

```bash
# 数值：编译原H7 C，只有CMSIS三角函数／开方替换为宿主libm
mjlab_training/.venv/bin/python scripts/agent/check_lqr_vmc.py \
  --output artifacts/teacher-numerical.json
# 若本机cc不在PATH，追加 --cc <已有宿主gcc路径>

# 20秒静站；12秒十一组固定命令；20秒二十组正常趴地；20环境局部复位
mjlab_training/.venv/bin/python scripts/agent/assess_lqr_vmc.py --case standing --output artifacts/teacher-standing
mjlab_training/.venv/bin/python scripts/agent/assess_lqr_vmc.py --case motion --output artifacts/teacher-motion
mjlab_training/.venv/bin/python scripts/agent/assess_lqr_vmc.py --case ground --output artifacts/teacher-ground
mjlab_training/.venv/bin/python scripts/agent/assess_lqr_vmc.py --case reset --output artifacts/teacher-reset
# 七段序列压缩为70ms，只核对命令分发和轨迹字段，不判行为通过
mjlab_training/.venv/bin/python scripts/agent/assess_lqr_vmc.py --case sequence --output artifacts/teacher-sequence
```

数值检查覆盖增益／状态反馈、几何／雅可比／虚功、腿长与横滚PD／补偿／限幅、正常起立切换与超时重试、速度融合／离地落地、20环境独立历史与倒置零输出；另外用当前 XML 的闭合姿态核对坐标。不代表真实物理起立或板端控制通过。

平衡采用[现有v3.1](interfaces/chuanliantui-wheel-balance-acceptance-v2.md)末尾连续1秒判据：倾角≤35°、机身接触≤1N、每轮≥max(10%重量,5N)、双轮总承重80%～120%、横滚／俯仰角速度模长≤0.5rad/s；`final_balance_v31`还要求完整20秒、无故障和手动复位，短回放仅报告`final_window_balance`。

跟踪采用[现有平地阈值](interfaces/chuanliantui-huananhu-multimode-tracking-v1.md)：固定命令至少12秒，跳过前2秒，速度／偏航平均绝对误差≤0.30，机身高度误差≤0.03m，测量段最大倾角≤25°、无机身接触、全程无故障。高度目标明确使用 `腿长命令+0.03648m`；另报实际腿长误差。十一组检查不是原协议全部196格命令网格，序列回放也不算固定命令跟踪通过。
