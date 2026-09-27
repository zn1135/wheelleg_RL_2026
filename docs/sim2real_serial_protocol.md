# 整机 sim2real 串口诊断数据协议 v1

日期：2026-09-26。状态：**待固件实现的发送契约**，不是当前串口已经支持的协议。

适用：chuanliantui、25 维观测、125 维历史、6 维动作；本轮对照模型为 `model_6000_h723.onnx`。目标是定位观测、策略、控制、执行及整机动力学链路中的第一处差异。

固件参考：`CtrBoard-H7_ALL`，本次核对的提交为 `3c0af0118e9108986f535ba1db346c65c2c4fd99`。实际烧录版本必须通过 META 报告，不以模型文件名代替固件版本。

## 1. 实现范围与优先级

固件负责：同拍快照、板端对时、明确运行阶段、发送数据及报告丢包。上位机负责：保存原始字节、过滤有效片段、复算网络/控制器、驱动仿真和画对照曲线。

优先完成以下六种帧：

| 类型 | ID | 作用 | 默认频率 |
|---|---:|---|---:|
| META | 0x01 | 模型、固件、坐标、增益和映射参数 | 启动/配置改变/新会话；待机每 5 秒重发 |
| EVENT | 0x02 | 策略开始、停止、故障、历史重置等事件 | 事件触发 |
| POLICY | 0x03 | 精确的网络输入及对应输出 | 每次推理，目标约 100 Hz |
| CONTROL | 0x04 | 实际使用的动作编号、PD、映射和电机反馈 | 从控制内环抽取 100 Hz |
| IMU | 0x05 | 原始与变换后的姿态、角速度、加速度 | 50 Hz |
| HEALTH | 0x06 | 真实周期、超时、队列和丢帧计数 | 10 Hz；待机可降到 1 Hz |
| HISTORY | 0x07 | 精确的 125 维网络历史输入 | 首次推理、历史重置后首次推理、每 0.5 秒 |

HISTORY 也是准确复算策略所必需的同步机制。首次实现应一并提供。

不改控制频率来迁就串口。CONTROL 的 100 Hz 是遥测频率；实际内环频率通过 META 和 HEALTH 报告。诊断 1～2 ms 级执行动态时，另做短时高频采集，见第 10 节。

## 2. 策略开始标志与数据分段（必须实现）

### 2.1 三个不同的状态

- `POLICY_STARTED`：本会话已经成功发布过至少一拍网络动作。发布第一拍时置 1，本会话内保持 1；结束会话后清 0。
- `POLICY_ACTIVE`：当前 RL 控制有效，执行层正在使用本会话成功发布的网络动作。它是持续电平；退出 RL、失能、故障门控或退回预热时清 0。
- `POLICY_START_EDGE`：第一拍网络动作成功发布时的开始事件。它是单次边沿；同时发送 `POLICY_START` EVENT。

**网络初始化成功、模型加载完成、观测预览、历史预热、零动作 PD 运行，都不代表策略已开始。**

成功发布并不等于已送到电机。CONTROL 的 `used_policy_seq` 和发送状态用于找到网络动作第一次实际进入执行链的时刻。禁止把“策略开始”直接当作“电机已响应”。

### 2.2 会话编号

- `boot_id`：每次 MCU 启动的新 64 位标识，可用持久化启动计数或可靠随机数；同一次启动所有帧相同。不能每次上电固定为 0。
- `session_id`：本次启动内递增；0 表示待机。进入一次新的 RL 预热/控制流程时分配新编号，从 1 开始。
- 退出 RL/失能后结束会话；重新投入分配新编号。运行中的模型/配置切换、要求重新初始化策略的历史清空，也应结束旧会话、建立新会话。
- 短暂无效但未重启策略流程时可以保留会话号，必须清除 `POLICY_ACTIVE` 并记录故障/恢复事件；该间隙不能作为连续有效轨迹。
- 待机阶段反复清理预览缓存不创建新会话。

每个会话中：`policy_seq` 从 1 起，每次推理尝试递增，失败也占号；0 表示没有网络动作。`control_seq`、`obs_seq`、`history_seq` 分别按真实控制执行、观测构建、历史更新递增。各编号均使用整数，不转换成 float。

### 2.3 推荐状态位 `flags`（每帧头部都有）

| bit | 名称 | 置 1 的含义 |
|---:|---|---|
| 0 | POLICY_STARTED | 本会话已成功发布第一拍网络动作 |
| 1 | POLICY_START_EDGE | 本帧是开始事件；仅 POLICY_START EVENT 置位 |
| 2 | POLICY_ACTIVE | RL 动作当前有效且进入执行控制流程 |
| 3 | OBS_VALID | 对应快照的策略观测有效 |
| 4 | HISTORY_VALID | 对应快照的历史可用于推理 |
| 5 | INFER_OK | 最近一次推理尝试成功；POLICY 帧指本帧这次尝试 |
| 6 | OUTPUT_ENABLED | 电机输出门控允许；预热时可能为 1 |
| 7 | ALL_REQUIRED_ONLINE | 必需 IMU/电机在线 |
| 8 | FALLEN | 跌倒判据触发 |
| 9 | FAULT | 当前存在控制故障 |
| 10 | WARMUP | 正处于预热 |
| 11 | SNAPSHOT_VALID | 本帧载荷已完成一致性快照 |
| 12 | ANY_CLAMP | 本次快照对应控制周期有任一限幅 |
| 13～31 | RESERVED | v1 写 0 |

不能用浮点动作是否等于 0 判断策略开始；网络可能合法输出零动作。

### 2.4 上位机过滤规则

1. 先验证帧头、版本、长度、CRC；错误字节不进入数值分析，原始接收文件可保留。
2. 按 `(boot_id, session_id)` 分组，`session_id=0` 仅供设备状态查看。
3. 分别保留预热段、有效策略段、停止/故障段。默认对照图显示 `POLICY_STARTED=1 && POLICY_ACTIVE=1` 的有效片段；故障前后数据仍存档。
4. 同时检查观测、历史、推理和快照有效位；不把失败帧中的旧动作当作新动作。
5. 中途接入串口时，即使错过开始边沿，也能通过持续状态位和会话号识别当前运行。精确策略复算须等到 META 完整且收到有效 HISTORY，同步前标记“历史未同步”。
6. 丢包、历史断号、会话切换、设备重启处断开曲线；禁止插值后宣称通过网络逐帧一致性验证。

## 3. 字节格式、时间与有效值

### 3.1 推荐独立二进制协议

现有 `Vofa_Send()` 最大 32 通道，不能承载本协议。建议在诊断模式下，用以下协议替换同一 UART 上的普通 VOFA 输出；文本 printf、32 路 JustFloat 和本协议不要交错发送。若要保留 VOFA，使用另一串口。

诊断串口设置为 **961200 baud、8N1**，固件与上位机均使用这个数值；固件当前 `VOFA_PORT` 默认是 UART8，实际接线及端口由 META 报告。本协议所有整数和 IEEE-754 float32 使用小端；按字段显式序列化，不直接发送有编译器填充的 C struct。

固定头部 40 字节，随后载荷，最后 4 字节 CRC：

| 偏移 | 字段 | 类型 | 含义 |
|---:|---|---|---|
| 0 | magic | uint8[4] | 固定字节 `53 32 52 31`，ASCII `S2R1` |
| 4 | version | uint8 | 1 |
| 5 | frame_type | uint8 | 第 1 节的类型 ID |
| 6 | header_len | uint16 | 40 |
| 8 | payload_len | uint16 | 载荷字节数，v1 最大 1024 |
| 10 | reserved | uint16 | 0 |
| 12 | packet_seq | uint32 | 生成一帧时分配；队列丢弃也占号，全类型共用 |
| 16 | boot_id | uint64 | 启动标识 |
| 24 | session_id | uint32 | 控制会话编号 |
| 28 | t_us | uint64 | 载荷快照发生时的板端单调时间，单位 μs |
| 36 | flags | uint32 | 第 2.3 节状态位 |
| 40 | payload | byte[payload_len] | 按帧类型解释 |
| 40+payload_len | crc32 | uint32 | 头部与载荷全部字节的 CRC |

CRC 使用 CRC-32/ISO-HDLC：poly=`0x04C11DB7`，反射实现 poly=`0xEDB88320`，init=`0xFFFFFFFF`，refin/refout=true，xorout=`0xFFFFFFFF`；`123456789` 的检查值为 `0xCBF43926`。CRC 本身不参与计算，小端发送。

接收侧缓存半包，处理粘包；magic/长度/CRC 失败时前移一个字节重寻帧头。合法帧总长为 `44 + payload_len`。不能只凭 magic 接受一帧。

### 3.2 时间与同步

- 全部时间戳来自同一个 MCU 单调时钟；用整数 uint64 微秒，不能将累计微秒塞进 float32。
- `t_us` 是信号被采样/快照的时刻，不是 UART DMA 启动时刻。排队时间不能加进控制器延迟。
- CAN/IMU 收包回调记录各自接收时刻。接收时刻只反映到达 MCU 的时间；若设备有源采样时间，也保存并声明时钟关系。
- 电脑接收时刻另存 `host_rx_time`，只用于链路诊断。没有时钟同步时不能直接与 MCU 时间相减求物理延迟。
- CAN“提交成功”表示成功入发送队列，不代表驱动器已经执行，也不等同于硬件发送完成。

无效、缺失或无法测量的浮点量填 quiet NaN，并清对应有效位；真实零值保持 0。缺失时间戳填 0，关联有效位必须清除。禁止把请求力矩复制成反馈力矩、把策略估速伪装成真实速度。

## 4. 坐标、顺序与单位

| 代号 | 固定顺序 |
|---|---|
| V6 | `[lf0, lf1, lfwheel, rf0, rf1, rfwheel]`，虚拟关节 |
| L4 | `[lf0, lf1, rf0, rf1]`，虚拟腿关节 |
| M6 | `[左前腿电机, 左后腿电机, 右前腿电机, 右后腿电机, 物理左轮, 物理右轮]` |
| W2 | `[物理左轮, 物理右轮]` |

- POLICY 的动作、观测、latent 均为**训练/网络接口空间**，直接截取实际网络输入输出。
- CONTROL 中带 `_fw` 的量为**固件控制器实际使用的坐标**，不能误标成训练空间。META 发送 `q_train = sign * wrap(q_fw - zero)` 的 sign/zero，以及动作映射规则。
- M6 使用固件逻辑电机正方向，发送/反馈都按同一极性校正，并明确角度/速度/力矩是在电机轴还是减速器输出轴。CAN ID、左右交叉接线和减速比写入 META。
- 四元数统一序列 `[w,x,y,z]`。IMU 原始空间保留设备定义；变换后的 `quat_body` 定义为机体到世界旋转，世界 +z 向上。若源接口方向相反，先转换并在 META 记录转换规则。
- 本机器人训练前向为机体 **+x**。不要套用 imcawl 的 +y 约定。
- 角度 rad，角速度 rad/s，力矩 N·m，电流 A，加速度 m/s²，时间 μs；编码器原始值、DJI raw 电流单独声明换算，不与 SI 值混写。

## 5. POLICY：网络输入、动作、估速（0x03）

在推理调用前冻结输入，推理后补齐输出，作为同一份快照入队。失败尝试也发送，清 INFER_OK，输出填 NaN。不得发送下一拍输入配上一拍输出。

载荷按下表顺序，**212 字节**：

| 字段 | 类型 | 内容 |
|---|---|---|
| policy_seq | uint32 | 本次推理尝试编号 |
| obs_seq | uint32 | 本次实际使用的观测编号 |
| history_seq | uint32 | 本次实际使用的历史更新编号 |
| history_epoch | uint32 | 每次历史重置递增 |
| t_obs_us | uint64 | 对应观测构建时刻 |
| t_infer_start_us | uint64 | 推理开始时刻 |
| t_infer_end_us | uint64 | 推理完成时刻 |
| obs | float32[25] | 最终缩放、裁剪后的实际输入 |
| action_raw | float32[6] | 网络原始输出，V6 训练空间 |
| action_published | float32[6] | 动作裁剪等处理后发布的值，仍用训练空间表示 |
| latent | float32[3] | encoder 原始输出，保持网络缩放；当前估速换算 `/2` |
| command | float32[3] | 构建本次观测使用的 `[vx, yaw_rate, height]`，未缩放物理值 |

`obs` 布局必须保持：

| 下标 | 内容 |
|---|---|
| 0～2 | 机体角速度 ×0.25 |
| 3～5 | 机体系重力投影 |
| 6～8 | 指令 `[vx×2, yaw_rate×0.25, height×5]` |
| 9～12 | L4 的训练角度减默认角 |
| 13～18 | V6 的训练关节速度 ×0.05 |
| 19～24 | 观测中实际使用的上一拍动作 |

默认角为 `[-0.06, 0.10, 0, 0.06, -0.10, 0]`。实际值和缩放以 META 为准。`obs[19:25]` 不能代替本拍 `action_published`。

### HISTORY（0x07）

载荷为 `policy_seq:uint32, history_seq:uint32, history_epoch:uint32, history:float32[125]`，共 **512 字节**。`history` 必须是对应 POLICY 真正送入网络的数组；顺序为最旧在前、最新在后。是否包含当前 obs、首次填充/预热方式在 META 写明。

首次成功推理必须配一帧 HISTORY；定期再次同步。没有推理的预热 HISTORY 可令 `policy_seq=0`。上位机遇到历史断号后，使用下一帧完整 HISTORY 恢复；若用连续 POLICY 重建历史，必须先确认每次历史更新均有对应观测，且顺序与 META 一致。

## 6. CONTROL：PD、映射、执行（0x04）

从真实内环冻结快照，以 100 Hz 发送。先确定本周期使用的动作和反馈，再记录控制计算结果、输出门控后的请求及提交结果；禁止在串口任务中逐个读取不断变化的全局变量拼帧。

载荷按下表顺序，**452 字节**：

| 字段 | 类型 | 内容 |
|---|---|---|
| control_seq | uint32 | 实际控制内环编号，抽样后允许自然跨号 |
| used_policy_seq | uint32 | 本周期真正使用的网络动作编号；预热/待机为 0 |
| state_seq | uint32 | 本周期冻结的关节状态快照编号 |
| imu_seq | uint32 | 当时关联的 IMU 接收样本编号 |
| control_dt_us | uint32 | 相邻真实内环开始时刻差 |
| control_exec_us | uint32 | 本周期控制计算和提交请求的耗时 |
| t_dm_enqueue_us | uint64 | 本周期 DM 发送批次开始提交的时刻；各帧差异需专项高频采集 |
| t_dji_enqueue_us | uint64 | 本周期 DJI 提交时刻 |
| motor_rx_us | uint64[6] | 本周期实际消费的 M6 反馈各自接收时刻 |
| motor_feedback_valid_mask | uint32 | M6 的总反馈有效位，bit i 对应电机 i |
| motor_send_ok_mask | uint32 | M6 本周期请求提交成功；不是“电机已执行” |
| motor_clamp_mask | uint32 | M6 本周期发生输出限幅 |
| virtual_clamp_mask | uint32 | V6 本周期发生虚拟力矩限幅 |
| pd_wrap_mask | uint32 | L4 本周期实际执行了误差 wrap，bit 0～3 |
| motor_field_valid | uint32[6] | 各电机字段有效位：bit0 q，1 dq，2 τ反馈，3 电流，4 rx时间 |
| q_virtual_fw | float32[6] | 控制器实际使用的虚拟角度，V6；未使用的轮绝对角填 NaN |
| dq_virtual_fw | float32[6] | 控制器实际使用的虚拟速度，V6 |
| q_target_fw | float32[4] | L4 实际位置目标 |
| wheel_speed_target_fw | float32[2] | 两轮实际速度目标 |
| error_before_wrap_fw | float32[4] | L4 在 PD 内实际计算出的目标减反馈 |
| error_after_wrap_fw | float32[4] | L4 实际用于 P 项的误差；未 wrap 时与前项相同 |
| tau_virtual_raw_fw | float32[6] | PD 输出，虚拟力矩限幅前 |
| tau_virtual_limited_fw | float32[6] | 虚拟力矩限幅后、气弹簧补偿前 |
| gas_tau_shank_fw | float32[2] | 实际额外加入左右虚拟小腿的气弹簧补偿力矩；关闭为 0 |
| shank_jacobian | float32[4] | `[左∂q膝/∂q前, 左∂q膝/∂q后, 右∂q膝/∂q前, 右∂q膝/∂q后]`，固件坐标 |
| tau_motor_unclipped | float32[6] | 机构映射及补偿后、实体输出限幅前，M6 |
| tau_motor_request | float32[6] | 全部限幅和使能门控后，最终逻辑电机请求，M6 |
| tau_motor_feedback | float32[6] | 驱动器反馈或标定电流换算的力矩，M6；来源在 META 声明 |
| q_motor | float32[6] | 实体电机角度，M6，保留连续/多圈定义 |
| dq_motor | float32[6] | 实体电机角速度，M6 |
| current_motor | float32[6] | 实际反馈电流 A，M6；只有 raw 且无可靠换算时填 NaN |

对无反馈或离线电机，相关有效位清零。发送失败时保留“本来想发送的请求”，同时清对应提交成功位；不能把它当成已施加力矩。

`tau_virtual_raw_fw` 需要在限幅前新增快照。当前 `state->virtual_torque` 是**虚拟限幅后、气弹簧补偿前**，只对应 `tau_virtual_limited_fw`。

当前 `vshank_jac[0]` 对后电机、`vshank_jac[1]` 对前电机；填本协议数组时注意顺序。虚拟力矩与实体电机力矩分开比较，不能直接逐元素相减。

## 7. IMU：观测源与姿态变换（0x05）

载荷按下表顺序，**88 字节**：

| 字段 | 类型 | 内容 |
|---|---|---|
| imu_seq | uint32 | 源样本编号 |
| t_imu_rx_us | uint64 | 此样本到达 MCU 的时间 |
| valid_mask | uint32 | bit0 原始quat，1 原始gyro，2 原始accel，3 body quat，4 body gyro |
| filter_mask | uint32 | bit0 quat经过滤波，bit1 gyro经过滤波，bit2 accel经过滤波；细节见 META |
| quat_sensor | float32[4] | 设备空间四元数，按 `[w,x,y,z]` 排列 |
| gyro_sensor | float32[3] | 设备空间原始角速度，rad/s |
| accel_sensor | float32[3] | 设备空间加速度，m/s²；是否含重力写入 META |
| quat_body | float32[4] | 实际用于观测构建的机体姿态，转换到第 4 节定义 |
| gyro_body | float32[3] | 实际用于观测构建的机体角速度，未乘观测缩放 |

IMU 已校准仍发送这些数据，用于追查不同任务采样时刻和滤波延迟。BODY 数据与网络观测通过时间/样本编号关联；50 Hz IMU 帧不保证包含每个 100 Hz POLICY 的源样本，精确观测复算需要临时提升 IMU 频率并检查预算。

## 8. META、EVENT、HEALTH

### 8.1 META（0x01）：配置清单

使用 UTF-8 JSON。允许分片，每片载荷为 `meta_id:uint32, chunk_index:uint16, chunk_count:uint16, total_json_bytes:uint32, json_chunk:byte[]`，每片 JSON 最多 512 字节，序号从 0 开始。同一 meta_id 内容不可变化；收齐再解析。运行中每 5 秒可重发同一清单，限制发送速率；控制数据优先。

必须包含：

- `protocol_version`、固件 git commit/dirty 状态/构建 ID、机器型号、UART 端口、波特率、各帧目标发送率。
- `model_name`、模型内容哈希、哈希算法、输入输出形状、推理后端/量化配置。当前 ONNX 文件 MD5 为 `4899195601babb22a7c0b46cc151ad84`，实际板端模型必须据真实生成产物报告。
- 实际配置的策略周期、PD 周期、反馈频率、预热步数；时钟单位和来源。
- 25 维观测排列、全部缩放/裁剪、125 维历史排列/填充/更新方式、last_action 更新时机。
- V6/L4/M6 名称、CAN ID、原始反馈索引、物理左右映射、符号、零点、减速比、电流/力矩换算常数及测量来源。
- PD 实际 Kp/Kd/Ki、P/D 实现、位置/速度动作比例、默认角、wrap、死区、滤波、各层限幅和斜坡参数。若存在控制器内部状态，记录初始/重置规则。
- IMU 轴映射、四元数方向、安装旋转、滤波参数及加速度是否含重力。
- 气弹簧补偿启用状态、参数、符号；几何/Jacobian 参数或可取得的配置文件哈希。
- 机器人总质量、载荷配置、轮半径、机构/资产版本；未知项写 null。轮地摩擦是待辨识项，不能把仿真假设填作测量值。
- `config_id` 与配置内容哈希。运行中配置改变必须通知上位机并划分新会话。

大 JSON 不在实时任务中生成。启动时预生成或使用静态常量，在低优先级任务分片发送。

### 8.2 EVENT（0x02）

载荷 24 字节：`event_seq:uint32, event_code:uint16, reason_code:uint16, policy_seq:uint32, control_seq:uint32, detail0:uint32, detail1:uint32`。

| event_code | 名称 | 含义 |
|---:|---|---|
| 1 | SESSION_BEGIN | 新会话建立，进入预热 |
| 2 | POLICY_START | 第一拍网络动作成功发布；头部 START_EDGE=1 |
| 3 | POLICY_STOP | 策略流程停止；放在旧 session 中发送 |
| 4 | FAULT_RAISED | 故障出现 |
| 5 | FAULT_CLEARED | 故障恢复 |
| 6 | HISTORY_RESET | 历史清空；detail0 为新 history_epoch |
| 7 | CONFIG_CHANGED | detail0 为新 config_id，随后发送 META |
| 8 | SYNC_MARK | 外部同步标记，detail0 为标记编号，可配合同步 LED/视频 |

`reason_code`：0正常、1用户退出、2失能、3源数据失效/离线、4观测失效、5推理失败、6跌倒、7超时、8配置变更；其它代码须在 META 列出。

开始/停止事件建议连续发送 3 次，`event_seq` 和原事件 `t_us` 不变，各次 `packet_seq` 不同，上位机按 event_seq 去重。重复开始事件不能重复清空分析缓存。

### 8.3 HEALTH（0x06）

载荷按下列顺序，全部 uint32，共 **100 字节**：

```text
window_us,
policy_attempt_count, control_count,
policy_dt_min_us, policy_dt_max_us, policy_dt_sum_us, policy_dt_count,
control_dt_min_us, control_dt_max_us, control_dt_sum_us, control_dt_count,
infer_exec_max_us, control_exec_max_us,
policy_overrun_count, control_overrun_count,
imu_age_max_us, motor_age_max_us,
can_submit_fail_count, infer_fail_count,
telemetry_drop_total, uart_busy_total, telemetry_queue_peak_bytes,
motor_clamp_or_mask, virtual_clamp_or_mask, telemetry_generated_total
```

除带 `_total` 的启动累计量外，其余统计每个 HEALTH 窗口重置。周期求和、计数来自实际相邻执行时刻，用 `sum/count` 算平均；无有效周期时 count=0、min/max/sum=0。超时判定阈值写 META。

年龄由本周期时刻减真正消费的接收时刻得到，统计窗口内最大值。计数与限幅 OR mask 在每个真实控制内环更新，不能只统计串口抽样点。`uart_busy_total` 表示发送泵遇忙次数，遇忙应保留队列，不能直接当作丢帧；实际丢弃必须增加 `telemetry_drop_total`。

## 9. 带宽与快照实现

8N1 每字节占 10 bit，961200 baud 的理论载荷能力为 **96120 B/s**。

按固定帧头+CRC 44 字节计算：

| 帧 | 每帧总字节 | 频率 | 字节/秒 |
|---|---:|---:|---:|
| POLICY | 256 | 100 Hz | 25,600 |
| CONTROL | 496 | 100 Hz | 49,600 |
| IMU | 132 | 50 Hz | 6,600 |
| HEALTH | 144 | 10 Hz | 1,440 |
| HISTORY | 556 | 2 Hz | 1,112 |
| 合计 | — | — | **84,352，约 87.8%** |

理论余量为 **11,768 B/s，约 12.2%**，用于 META/EVENT、抖动和接收侧延迟。运行中 META 重发限速至不超过 2,000 B/s；启动清单可在待机阶段优先发完。若发送队列持续增长，可将 CONTROL 遥测降到 80 Hz，此时基础占用为 74,432 B/s、约 77.4%，并在 META 报告实际帧率。若真实推理频率/帧率更高，必须重新计算。不能再叠加原有 32 通道高频 VOFA 帧。

实现要求：

1. 控制任务只拷贝固定快照到预分配缓冲，不 printf、不等待 UART、不做 JSON 编码。
2. 用双缓冲/队列或序列锁保证整帧一致；IMU/CAN 状态也用原子快照，避免半帧更新。
3. 推理输入、对应历史和输出绑定 policy_seq；执行层快照绑定真正使用的 used_policy_seq。
4. 低优先级发送任务从队列启动 UART DMA；DMA 使用中的内存不能重写。缓存一致性沿用固件已有维护方式。
5. 队列满时丢遥测，不阻塞控制；保留最新状态，累加丢弃数。事件和 HISTORY 优先级高于可降频的普通状态帧。
6. 增加帧发送分频器；HEALTH 保存全部内环的峰值和计数，避免抽样隐藏瞬态限幅。
7. `packet_seq` 的间隙用于发现遥测缺失；CONTROL.control_seq 因抽样跨号是正常现象。不能把两者混用。

## 10. 整机预测所需的补充与能力边界

### 10.1 执行延迟和高速关节响应

默认 100 Hz CONTROL 适合整机筛查，不能凭它排除 1～2 ms 级动态差异。发现可疑关节后，在板端按真实内环速率缓存一个短窗口，保存本协议 CONTROL 的完整快照及准确时间；采集结束后慢速导出。缓存满即停采并报告，不能影响控制。也可定义精简高频帧，但须升级/补充协议后再接收，不能复用旧帧 ID 改布局。

### 10.2 位移、速度与接触

精确初始化整机仿真还需要机身位置、线速度、姿态、角速度、关节状态及外力/接触信息。当前串口可以提供其中部分信息。

- 机身位置和真实线速度：同步视频、外部定位或独立传感器；通过 SYNC_MARK 对时。若使用里程计，必须标明“估计”，不能用轮速估计作为检验轮打滑的独立真值。
- 轮地接触：有触地/力传感器就另行定义可识别的新类型帧；仅有电流/姿态推断时，记录来源和估计状态。
- 地面材料、坡度、附加载荷、气弹簧实物状态、视频文件名记录在本次采集说明中。
- 仅将两套策略各自长时间运行后比较是否跌倒，不能定位第一处差异。优先同输入复算，再用多段短时动力学预测检查运动响应。

缺少上述外部真值时，仍可排查网络、控制律、执行及部分姿态动态，但不宣称已排除全部接触/整机动力学 gap。

## 11. 当前固件取数位置与接入注意点

以下路径相对部署仓库根目录：

| 位置 | 建议取数 |
|---|---|
| `imcalib/task/task_policy.c` | 观测构建、历史更新、真实推理前后、动作发布；记录阶段和 policy_seq |
| `imcalib/Algorithm/rl_observation.h` | 核对 obs 25 维布局及 history 125 维定义 |
| `imcalib/Algorithm/rl_policy.h` | 原始动作、latent、运行状态；run_us 只代表推理耗时 |
| `imcalib/Algorithm/rl_torque.c` | PD误差、目标、限幅前后虚拟力矩、补偿和Jacobian |
| `imcalib/task/task_actuation.c` | 真正使用的动作编号、输出门控、最终请求、提交结果 |
| 电机/IMU 接收入口 | 每个源的接收序号和板端时间戳 |
| `imcalib/task/task_comm.c` | 低优先级发送；诊断模式停发旧 VOFA 帧 |
| `imcalib/user-lib/Vofa_send.c` | 当前是32路JustFloat，需独立实现本协议的缓冲与DMA发送 |

当前旧帧 ch23～28 是上一拍动作，ch3～6 是实体电机请求；二者不能直接充当 POLICY.action_published 和 CONTROL.tau_virtual_raw_fw。

本协议是**被动诊断遥测**，不需要启用旧自动 Sysid 激励模式。当前 Sysid 控制入口为 `#if 0`，发送泵也未接入；本次不要把启用激励测试当成串口采集的前提。

## 12. 修改后交付与接入检查

交付材料：固件 commit/构建 ID、实际使用模型哈希、完整 META 示例、各类型帧一份十六进制样例、采样频率与串口设置。首次记录建议覆盖：待机 → 预热 → 策略开始 → 起立/站稳 → 退出，约 20 秒；动作按现有操作流程执行。

接入时逐项确认：

- [ ] 开机无关字节和坏包能过滤，待机不被识别为策略运行。
- [ ] 首次有效动作发布有 POLICY_START；持续 POLICY_STARTED/ACTIVE 和 session_id 正确。
- [ ] 预热 PD 输出与网络动作执行可区分，used_policy_seq 能关联到准确 POLICY。
- [ ] 停止/失能及时清 ACTIVE，再次投入建立新 session；重启后 boot_id 改变。
- [ ] POLICY 输入/历史/动作一一对应；中途接入及丢包后能重新同步 HISTORY。
- [ ] 单位、左右、符号、虚拟/实体空间通过 META 明确，缺失值不会伪装为零。
- [ ] UART/CAN提交失败、遥测丢帧、控制超时可见；时间戳不依赖电脑收包间隔。
- [ ] 发送诊断数据后，实际控制周期与执行耗时没有明显恶化。

本文件仅定义待实现接口；尚未修改、编译或烧录部署固件，也未验证串口接收。
