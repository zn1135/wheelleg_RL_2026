# 模型使用记录：2026-10-01 chuanliantui 电脑推理／H7 执行候选

## 当前结论

监督试验已真实运行完整2秒，100Hz观测／动作和500Hz输出、遥控停止及2秒自动停机已验证；用户确认未起立，现象为轮子空转／打滑、腿几乎不动。第04次会话已取得40/40组电机反馈，DM电流推算力矩跟随目标，但腿仍停滞；当前保持10/1 N·m限幅，继续核查实物负载、气弹簧／机构约束与实体映射。软件协议通过不代表起立验收通过。当前镜像与本轮记录见文末。

## 任务与接口

| 项目 | 内容 |
| --- | --- |
| 日期、执行者、接收者、独立复查人 | 2026-10-01，Codex／用户当前工作区；独立代码复查已提出输出资格前的阻断项 |
| 机器人与任务 | chuanliantui_standup，大轮腿，STM32H723；物理机构版本待核实 |
| 用途与验收目标 | 电脑 CPU 运行完整模型，H7 100 Hz 观测／500 Hz PD，地面后摆起立闭环 |
| 接口 | `chuanliantui-25d-100hz-r1` + 候选执行会话 `2901`；[接口约定](../deployment-contract.md)、[设计规格](../superpowers/specs/2026-10-01-chuanliantui-host-policy-h7-standup-design.md) |
| 关联 PR | 无；当前没有提交、推送或创建 PR |

## 代码与训练来源

| 项目 | 训练仓 | 部署工作区 |
| --- | --- | --- |
| 仓库／分支 | Wheel-Legged-Gym，`26_wheelleg`，HEAD `bb56dd40964dc569b4e2a406983b03fa19d943a1`；工作区改动未提交 | `$H7_REPO_PATH`，无 Git 元数据 |
| 工作区状态 | 有既存与本任务改动，不以 HEAD 代表运行版本 | 有既存与本任务改动；限时监督试验版本已烧录并读回；2901完整策略执行2秒、自动停机通过，但未起立 |
| 差异／源码快照 | 离线实现时点相关源码包 `captures/20261001-policy-implementation-final-offline-01/train-source.tar.gz`，SHA-256 `eb26728ec32b9c0a5347092687ade484e0cff92ed11ca9d96496f469085081b2`；保留既存工作区改动 | 实施前 `captures/20261001-policy-implementation-baseline-01/h7-source.tar.gz`，SHA-256 `d3df0b4b0310f763d47e39204891b96626a182267db22f9ce37eac056bc08323`；离线实现时点 `captures/20261001-policy-implementation-final-offline-01/h7-source.tar.gz`，SHA-256 `67fa53a31892cc67d34b0b88bf446aa54bd33b9f01cb9a393fd5909192a7ca23`；现场 DM2/DM3 工具和位置包线修改后的 H7 源码快照 `captures/20261001-policy-after-onsite-dm2dm3-01/h7-source.tar.gz`，SHA-256 `117ac5b955879505fe8468deaba2fa24a399d12711e0e4722fe0a3cc5e5334c8`，此版本尚未烧录 |
| 配置与凭据 | `.env.local` 不入快照；训练根 `logs/` 只读 | USB by-id 接口已现场使用；串口权限由用户临时授权 |

模型使用与训练来源承接[失能诊断记录](20261001-chuanliantui-disabled-policy-diagnostic.md)：
只读使用 `logs/chuanliantui_standup/Sep22_16-26-11_standup_no_legangle_resume/model_6000.pt`
（1,199,938 字节，SHA-256 `8070977e19114cb9bbf358a090bd78cdbd1be568a76e02d417c369e828bc8f36`），
checkpoint 内含 encoder 和 actor。本次不训练、不恢复优化器、不导出或运行 ONNX；
历史训练命令、seed、实际配置和 Isaac／MuJoCo 行为结果沿用原记录中“待独立核实”的状态。
电脑端指定 Isaac Gym Python 3.8、CPU 单线程，先 `import isaacgym` 再 `torch`。
板端 Debug 构建使用本机 ARM GCC 工具链；此前141116字节镜像为新增2901失能演练入口的锁定输出版本，
ELF SHA-256 `5d0368e9ae4bbe5624b2048d45b8f26eab64f6cc9ec55d5212994fd35fbfa2c8`，
141,116 B Flash 读回与构建 bin 逐字节相同。USB 权限恢复后，2901 失能会话已完成板端核验。

## 分阶段验证

| 阶段 | 状态与证据 |
| --- | --- |
| 完整模型加载、CPU 100 Hz 失能数据链 | 先前通过，见[失能诊断记录](20261001-chuanliantui-disabled-policy-diagnostic.md)；不代表执行模式通过 |
| B 轴身份与观测数值 | 用户确认 B 和双腿同姿态镜像；C／训练侧1000观测、2000闭链几何、四槽来源隔离通过；右侧 SolidWorks 轮轴与输入臂角度校核通过，左侧按镜像条件逐槽求偏置后数值闭合。新固件现场失能 201 帧与独立 CAD／DM 复算最大角差 `7.13e-7 rad`，旧固件负对照约 `0.09 rad`；证据 `H7_clion/captures/20261001-policy-bilateral-zero-firmware-01/`。实体输入臂力矩方向仍待核验 |
| 500 Hz PD／虚功映射纯数值 | 本机 C 测试与1000组独立训练侧差分通过；`H7_clion/captures/20261001-policy-control-check-01/comparison.json`。这不是实体力矩验收 |
| 2901 状态机、主机协议与异常清理 | 原生测试、合成 USB 帧、接管边界、丢样本／主机中断与全局 STOP 兜底通过；新增失能板端演练 100 帧逐帧核对与漏发动作故障检查，证据见下 |
| H7 Debug 编译 | 当前已烧录锁定 ELF SHA-256 `5d0368e9ae4bbe5624b2048d45b8f26eab64f6cc9ec55d5212994fd35fbfa2c8`，FLASH 141,116 B，SWD 读回一致。用户取消的位置和速度阈值门控已撤掉；当前源码与镜像仍保持 `LOWER_RUN_OUTPUT_QUALIFIED=0`、`LOWER_OBS_MAPPING_VERIFIED=0`、六路试验力矩限值全0。 |
| MuJoCo 起立行为 | 串联代理和闭链模型各运行4秒 `--standup --cmd_vx 0 --cmd_height 0.20 --friction 0.75 --no_realtime` 无渲染对照；分别在第8/9个100Hz步首次轮接地，接管后采样高度约0.20m维持至第350步，终端正常退出。日志 `captures/20261001-solidworks-zero-geometry-01/sim2sim-standup-headless.log` 与 `sim2sim-standup-closed-chain-headless.log`；未做渲染目视判读，不等同实机起立 |
| 2901 板端时序与失能 | 新固件已烧录并逐字节读回；策略输出三重锁仍生效。`0x35 DRY_START` 失能会话收到100帧，全为10ms采样间隔，动作回执为2ms，主机逐帧校验25/125维观测与历史。第5帧自动注入的演练接管使其经历 PRECONTACT→CONTACT_EDGE→ACTIVE；STOP 后独立状态为 DM 失能、六目标零。另一会话故意漏发第30帧 ACTION，板端回 DEADLINE fault=2，STOP 后仍失能零目标。证据 `H7_clion/captures/20261001-policy-dry-normal-01/`、`-dry-drop-01/`。此入口不执行500Hz力矩计算或发送，不能代表真实带力矩闭环。 |
| 本轮 2901 锁定路径与起点 | 100 次只读 QUERY 的 99 个间隔各为 10 ms/5 tick；正确格式 START 被锁定镜像拒绝，仍为 OFF/session0、DM使能0、六目标零。会话 STOP 在 OFF 被拒绝，全局 STOP 回执及独立 STATUS 确认失能零目标，未发 ACTION/CONTACT/ARM/力矩；`H7_clion/captures/20261001-policy-locked-session-negative-01/`。用户恢复 SolidWorks 后摆姿态后，50 次只读 STATUS 的四路均值 `[0.54425,1.96016,0.48520,1.98029] rad`，相对旧同姿态参考最大差 0.02167 rad，且遥控2/2、四DM失能、六目标零；`H7_clion/captures/20261001-policy-restored-start-pose-01/`。执行输出仍锁定。 |
| 失能全程摆动与实测起点 MuJoCo | 用户确认最远摆动位置就是计划直立姿态且无异常；1501 次只读 STATUS 覆盖 DM0 `[-0.998,0.549]`、DM1 `[0.426,1.960]`、DM2 `[-0.692,0.531]`、DM3 `[0.806,2.022]` rad，始终失能零目标，候选闭链几何 1501 帧全有效。后摆实测虚拟髋约 ±1.17 rad，不等于训练起点 ±1.566 rad；以实测起点补充的闭链 MuJoCo 4秒模拟最终机身高度约0.204m，但前移约0.53m。2000个2ms仿真步的六路峰值绝对模型力矩 `[11.44,15.79,3.9,17.86,12.51,3.9]` Nm，不能视为实物安全限值或 H7 500Hz 实测；证据 `H7_clion/captures/20261001-policy-manual-travel-01/`。按用户最新要求，位置和速度仅用于监测记录，不做 2901 数值阈值门控；尚未烧录的源码门控已撤掉。 |
| 2701 单槽 DM2 极性试验 | 用户要求直接对单槽输出 `+5 N·m`、`0.5 s`，以遥控触发。第二次现场试验发出50个 `[0,0,5,0,0,0]` 目标，历时 `0.5021 s`；首个 STOP 前 DM2/DM3 位移约 `+0.871/+0.858 rad`，DM0/DM1 仅 `−0.006/−0.008 rad`。约60秒等待遥控期间四台失能电机已被动位移约 `0.52–0.68 rad`，不可算作脉冲响应。第三次试验 50 包同样只作用于 DM2，触发前至 STOP DM2 增加 `0.969 rad`。两次独立 STATUS 均确认四台失能、六目标零。用户报告从机器人右侧看 DM2 轴端顺时针，并确认近照中白臂直接固定于 DM2 输出轴；据刚性连接关系，**该直连臂从同一侧看也顺时针**。先前将广角图的 DM3 白臂误认作 DM2 已纠正；这一误认不影响上述轴端到直连臂的方向推论。广角图仍未独立证明 DM2 对应 CAD 前长／后短哪根臂。证据 `H7_clion/captures/20261001-polarity-dm2-plus5-05s-02/` 与 `-03/` |
| 2701 单槽 DM3 极性试验 | 前两次因遥控右拨杆板端读数不在下档而基线拒绝，零力矩、STOP 确认。恢复板端 2/2 后，第三次发出50个 `[0,0,0,5,0,0]` 目标、全部回读一致，历时 `0.5022 s`；DM3 增加 `0.993 rad`，DM2 同侧被动增加 `0.987 rad`。独立 STATUS 确认失能和零目标。用户明确指出视频蓝圈中的白色链轮旁臂属于 DM3；它在脉冲期间从右下转到近乎正下，**右侧视角顺时针**。对该臂的外观长短与 CAD 前／后命名仍待机械身份核对；证据 `H7_clion/captures/20261001-polarity-dm3-plus5-05s-03/` |
| DM0/DM1 左侧极性 | 用户确认左右机械镜像，并要求从右侧结果推算，不执行左侧脉冲。代码中左槽 `dm_sign=+1`、右槽 `dm_sign=-1`，逻辑 `+5 N·m` 发给原生驱动的符号相反；因此左侧镜像方向是**配置与对称性推算**，不是现场实测。随后只读板端确认遥控 2/2、在线 63、使能 0、六路目标 0。该推算不能单独解锁整机策略输出。 |
| 外部部署参考 | 用户提供 [XYEGA RM2026 WheelLeg Infantry RLdeploy](https://github.com/chushanxiaodaoshi/XYEGA_RM2026_WheelLeg_Infatry_RLdeploy)，按 `fcfdd3959be5c9b00893ec0701a591ddbf55b830` 查阅。其 25/125 观测与历史、100/500 Hz 分频、训练索引与物理电机索引分离值得用于接口复核；其板端 CubeAI、物理通道顺序、16.33 轮减速比、增益／零点／气弹簧补偿属于另一台机器人，不复制数值。 |
| 旧固件现场失能零位复核 | 只读2701连续201帧、100Hz、最终失能零目标；另取100次四路 STATUS 反馈。板上原 Flash 132,580 B 与保存的旧 ELF 二进制逐字节一致。左侧 CAD 偏置后来按用户确认的镜像姿态求得。证据 `H7_clion/captures/20261001-policy-disabled-on-site-zero-01/` |
| 整机起立闭环与独立高度观测 | 未执行；无现场起立结果 |

当前交付范围为**板上零位观测、DM2/DM3 单槽输出和 2901 失能会话已验证、策略带力矩执行尚锁定的候选实现**。只有在四台 DM 失能反馈、实体零点和方向、
虚功方向、实测力矩限值、烧录后带输出的500 Hz时序与故障停机依次核验，并解决独立复查发现的命令停机集成测试与 500 Hz 追踪后，
才能配置版本化试验限值、解除执行资格并尝试地面起立。零策略动作在板端仍经过 PD，
可能产生非零实体力矩。完整实机验收记录与回退固件身份留待对应试验完成后填写。

## 2026-10-01 完整策略监督试验

上文锁定输出结论对应历史镜像。本次用户要求直接部署完整策略，取消新增单槽微脉冲；采用独立的限时试验配置，不把尚未独立实测的实体映射／左侧极性标成已验收。用户已确认地面后摆、双轮接地、前方留空、防跌保护、遥控双下及现场操作员就绪。

- 板端试验配置：DM0～3 每路 ±10 N·m，双轮每路 ±1 N·m；从接受 START 起 2000 ms 自动 STOP／失能。不设位置、速度阈值门控。遥控、USB、动作期限、反馈有效性、CAN、IMU、看门狗继续有效。
- 完整 checkpoint 不变；电脑 100 Hz 推理，H7 500 Hz 输出。预先确认双轮接地后，host 首样本只发送一次人工 CONTACT，随后进入 ACTIVE。
- 所选限幅的闭链模型4秒对照到达0.204 m，但前移0.606 m；2/0.5和5/0.5 N·m未起立。这是模型结果，不能当作实物安全力矩证明。结果目录 `H7_clion/captures/20261001-policy-direct-cap-sim-01/`。
- 修复真实 START 参数准备检查、运行中 DM 掉使能停机、反馈时间戳与 tick 的读取竞态。新增板端500Hz目标力矩／CAN发送返回值记录；STOP后清理D-cache以供SWD读回。记录不是实测轴端力矩。
- 构建快照 `H7_clion/captures/20261001-policy-direct-firmware-01/`：ELF `c32746f4460fb724d87590fa1249392b4d11728e8a0bda6da837edfc2a35b886`；bin142572 B，`6ce84a19ab33fd0e0164cafcc7607481ecdc36d256bce558a7d2f3618bde26dd`；H7源码包 `b22982700efba594d2f86a499458d8cb2536f661008201fd1f13225fc8e01fa5`；主机源码包 `297c1d657b54baec8cdedaf93087f506686cdff6aede5c0acb636ae77d09f250`。
- 本轮原生H7测试组、77个Python测试、主机2901合成数据检查及ARM构建通过；烧录／读回及实机执行结果待补。不能据此宣称实机起立成功。

刷写更新：试验镜像142572字节已通过SWD逐字节读回核验，哈希与上述bin相同。USB重枚举后等待用户恢复临时ACL；参数预检与真实策略START尚未执行。trace符号在该ELF为`0x24040440`，长度40976字节。

### 首次真实执行结果

参数核对和新版30帧失能演练通过后，`captures/20261001-policy-direct-run-01/` 首次执行了完整策略。65帧观测间隔均10ms，328条电机输出间隔均2ms；人工左杆拨下触发REMOTE停机，用户已确认是主动操作。输出历时655ms，峰值 `[10,10,10,9.09447,1,1]` N·m，没有超过所设限值，HAL发送均成功。故障时刻325291ms发送零目标和失能，325296ms独立反馈确认四DM失能、六目标零；随后只读遥控为双下。STOP应答先回enabled8、2ms后回0，触发了主机过早清理错误；原始日志保留，正在修复主机等待反馈的判定。

这是完整策略真实执行及人工遥控停机的证据，尚不能证明实机起立，2秒自动停机尚未走到。腿输出持续饱和、腿角变化小，轮速不对称；用户要求重复同限值，重试前正在确认接地和机身稳定。分析文件 `reviewed-result.json`，500Hz trace `trace.bin`（SHA-256 `c0dbf2db7e77bcf3423daa57d50d3febeaae5c6556f070418213845542e15c14`）；trace为目标与发送返回值，不是实测轴端力矩。

### 第二次真实执行：完整2秒，未起立

用户要求同限值重试并再次确认接地、稳定和双下。`captures/20261001-policy-direct-run-02/`记录200帧观测、1000拍输出+1拍终止，全部2ms/1tick、零丢记录，板端从543986ms至545986ms恰好2000ms后自动OFF；545992ms独立STATUS确认失能和六目标零。主机`protocol_pass`、无故障，用户明确没有起立，轮子空转／打滑、腿几乎不动。峰值目标`[10,10,10,9.26296,1,1]` N·m；ACTIVE时DM0/DM2均为−10、DM1有98.88%拍为+10。相同实测状态的训练侧独立复算吻合H7目标，199个边界最大差1.30e-5 N·m，说明当前软件控制合同一致；不能据此证明B实体映射正确或实际电机兑现了目标力矩。

清理等待修复版host SHA-256 `14776d374f6f61386d302aac3cabd5f1b53c9578368dd44526673407d3c72aef`；源码归档`host-source.tar.gz`为`f863bf169132fd46195f0f2f3116dd81cc077b20ea8b8a02bd11de383e3bb44c`。trace SHA-256 `cae12be3d156868d1c2d98b73a02d2f44d1a1c86e305792df146bf09df90392d`。下一步采已有DM位置／力矩估计及DJI电流反馈，区分实际出力、负载／约束与映射问题；尚未提高限值。

### 电机反馈实采（第04次会话）

`captures/20261001-policy-direct-run-04-feedback/`完成200帧策略、40/40组电机反馈和2000ms自动停止；1000拍输出均2ms，无缺失／故障。用户再次描述轮子打滑、腿几乎不动，并单独确认没有顶住机身／地面／防跌支撑。当前全失能、遥控双下。

0.3s后，目标均值`[-10,+10,-10,+1.49365]`、驱动估计力矩均值`[-10.00957,+9.98242,-9.97388,+1.50640]` N·m，最大绝对误差各≤0.247 N·m；末0.5s四轴最大位置跨度0.18°。驱动电流推算支持目标已经兑现到电流环，仍非独立轴端测力。当前没有确定软件计算／传输错误，实物负载、气弹簧／机构约束及独立轴映射继续待核。模型整机质量13.1487427kg、每侧150N弹簧，实际值已向用户询问。未增加限幅。

完整统计见`analysis/analysis.json`；trace SHA-256 `90fe6ca29fc3982cbc0898cc59c447f94a5e69293c6669dcc891f12809c7aeb4`，源码／原始记录摘要见同目录`artifact-sha256.json`。本轮是反馈与停机验证通过，起立验收仍未通过。

### wheelbipe_ros2_sim2sim 参考对照

用户提供 [scutrobotlab/wheelbipe_ros2_sim2sim](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim)。本次固定审阅 `dd367bf78c7e393d811595edeb4affe253156c95`，读取控制器、real_bridge、MuJoCo 执行器及模型源码；未安装或执行外部项目，未更改本机控制代码、固件和试验限幅，也未启动新实机试验。

| 项目 | 参考源码 | 本机当前合同／结论 |
| --- | --- | --- |
| 策略接口 | 35维单输入、6动作，50Hz推理／500Hz控制；normal-only控制尾部固定 | 25维观测＋125维encoder历史，100Hz推理／500Hz板端PD；模型与观测不能替换 |
| 腿动作 | 四个实体front1/rear1的位置目标，腿Kp=60、Kd=2 | 虚拟髋／膝PD，Kp=10、Kd=1，再经闭链Jᵀ映射；MIT发送kp=kd=0是因为PD已在板端计算，不是漏掉PD |
| 起始状态 | 独立PREPARE状态把四腿插值到0rad，目标插值上限1rad/s；FSM也允许直接进入RL | 本机执行地面后摆起立策略；参考PREPARE不是同任务的起立验收证据，也不能改写本机零点 |
| 气弹簧 | 力模型650→450N，粘性阻尼500N·s/m；MJCF另有弹簧滑块与端点闭链约束 | 本机闭链仿真恒150N tendon，没有对应滑块行程／阻尼；需要本机实测参数，不能复制参考数值 |
| 执行器 | 二阶响应、变化率和随转速降额，分别保存requested/applied effort | 本机MuJoCo直接施加映射力矩；现有500Hz目标trace与20Hz驱动反馈可用于辨识差异 |
| 真机桥接 | 按数组透传六路关节状态及五参数命令；未包含可审计的本机DM零点／CAN极性标定 | 无法用其left/right/front/rear名称直接证明或否定本机B映射 |

来源：[策略接口](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/PARAMETERS.md)、[PREPARE实现](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/src/controllers/template_ros2_controller/src/fsm/states/state_prepare.cpp#L22)、[FSM转移](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/src/controllers/template_ros2_controller/src/fsm/state_machine.cpp#L64)、[弹簧及执行器配置](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/src/resources/robot_descriptions/wheelbipe_V14/xacro/ros2control.xacro#L4)、[执行器响应](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/src/interfaces/mujoco_bridge/src/mujoco_system.cpp#L474)、[真机状态透传](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/src/interfaces/real_bridge/src/real_bridge.cpp#L334)。

**实体辨识方法纠正：**参考MJCF的front1→front2轴间XZ投影约113.4mm，rear1→rear2为210mm；本机CAD“前”长臂为210mm、“后”短臂为113.4mm，命名不可直接对号。尤其本机210mm长臂上还有距髋约113.4mm的中间闭链销轴，因此此前向用户提出的“输出轴到第一个销轴长度”不足以区分两输入臂。应以整根刚性臂、远端连接与完整闭链拓扑核对，保留用户确认的B作为当前候选，不由参考名称擅自换槽。证据：[参考轴间连接](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/src/resources/robot_descriptions/wheelbipeV14_2/mjcf/wheelbipeV14_2.xml#L64)，本机`sim2sim/chuanliantui.xml`第51/54/75/78/100/126行。

**静态模型差异：**逐项求和XML显式inertial质量，参考MJCF为25.10227007kg、本机为13.14874270kg；参考description.xacro自身为21.77948481kg，与其MJCF不同，不能混用。参考短输入轴frictionloss=1.5N·m，本机为0.015N·m。参考弹簧力插值区间为[-0.005,0.07]m，但MJCF实际slide关节范围为[-0.015,0.065]m，二者含义不同。该比较未加载模型运行，不能当作实物称重或行程测量。来源：[参考MJCF](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/src/resources/robot_descriptions/wheelbipeV14_2/mjcf/wheelbipeV14_2.xml#L48)、[参考Xacro](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/src/resources/robot_descriptions/wheelbipe_V14/xacro/description.xacro)。

**限幅需看完整路径：**参考控制器对完整PD＋前馈限幅，再调整前馈项，不能误认为硬件PD完全绕过限幅；其轮控制器配置±9.99N·m，仿真硬件接口又限至±5N·m。公开real_bridge不能证明其H7最终限幅。参考的腿±50.9N·m、Kp=60/Kd=2以及执行器54N·m等均不构成本机加力依据。来源：[完整PD限幅](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/src/controllers/template_ros2_controller/src/template_ros2_controller.cpp#L724)、[仿真接口限幅](https://github.com/scutrobotlab/wheelbipe_ros2_sim2sim/blob/dd367bf78c7e393d811595edeb4affe253156c95/src/interfaces/mujoco_bridge/src/mujoco_system.cpp#L160)。

本次未发现已证明的本机新符号／缩放错误。结合第04次会话，下一项有辨别力的核查是实物总质量、气弹簧额定推力及停滞姿态的弹簧行程，并按完整连杆拓扑核对输入轴。用户已排除外部碰撞／卡住；这仍不能单独排除弹簧内行程端点、静态负载差异或映射问题。整机起立尚未验收。

### 复旦训练／sim2sim 与 XYEGA 部署联合对照

用户再次提供两个关联参考，本次固定版本：

- `yly-true/fudan_rl_wheel_leg`：`8204e853dfd2ed06d85a322e1a998c3d20a3be2c`；仅提取文本源码和模型描述，不执行外部脚本。
- `chushanxiaodaoshi/XYEGA_RM2026_WheelLeg_Infatry_RLdeploy`：`fcfdd3959be5c9b00893ec0701a591ddbf55b830`；远端HEAD仍与前次相同，复查实车源码快照和伪代码。其[说明](https://github.com/chushanxiaodaoshi/XYEGA_RM2026_WheelLeg_Infatry_RLdeploy/blob/fcfdd3959be5c9b00893ec0701a591ddbf55b830/readme.md)明确运动学来自前者的sim2sim；该快照不是可独立编译的完整H7工程。

#### 已对齐的部分与不能混用的参数

双方使用25维观测、5帧125维历史、6维交错动作`[lf0,lf1,lwheel,rf0,rf1,rwheel]`；陀螺仪×0.25、速度×0.05、四个腿角减默认值、上一原始动作进入观测，历史最旧在前。腿位置动作×0.5、轮速度动作×10，虚拟PD后转换到实体输入轴。与本机现有结构一致，未由此发现新的观测维数／重复缩放错误。来源：[复旦观测及历史](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot.py#L341)、[XYEGA观测及历史](https://github.com/chushanxiaodaoshi/XYEGA_RM2026_WheelLeg_Infatry_RLdeploy/blob/fcfdd3959be5c9b00893ec0701a591ddbf55b830/pseudocode/math_core.cpp#L134)。

XYEGA执行500Hz／推理100Hz；复旦当前plane训练入口继承dt=0.005、decimation=2，即200Hz物理／100Hz策略，其sim2sim默认同为5ms/10ms。不能概括成所有参考路径都是500Hz。复旦当前plane配置的腿Kp=20、轮Kd=0.2，与sim2sim plane预设和XYEGA Stable的15/1/0.1也不同；应按具体模型绑定配置。本机当前checkpoint仍用本机默认角`[-0.06,0.10,0,0.06,-0.10,0]`、腿10/1、轮0.1，不因接口维数相同替换参数。来源：[复旦当前训练配置](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/wheel_legged/wheel_legged_config.py#L39)、[基类时序](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot_config.py#L116)、[XYEGA策略参数](https://github.com/chushanxiaodaoshi/XYEGA_RM2026_WheelLeg_Infatry_RLdeploy/blob/fcfdd3959be5c9b00893ec0701a591ddbf55b830/pseudocode/policy_params_design.hpp#L49)。

#### 新的明确差异：物理弹簧与软件补偿分开处理

| 路径 | 物理弹簧 | 控制中的修正 |
| --- | --- | --- |
| 复旦`onnx_mj_binglian.py` | 独立tendon每侧施加300N | 映射后反算虚拟腿径向力F／转矩T，左F减`369*l0`、右F加`369*l0`，再映回电机，最后限幅 |
| XYEGA实车快照 | README描述实体300N气弹簧 | 同样的F/T域修正；默认系数370.1，Spin左270.1、右300.1 |
| 本机闭链MuJoCo＋H7 | MuJoCo恒150N tendon；实物标称推力待核 | 当前H7仅虚拟PD→Jᵀ→逐槽限幅，没有上述F/T补偿 |

来源：[复旦控制修正](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/mujoco/python_tools/onnx_mj_binglian.py#L719)、[复旦物理施力](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/mujoco/python_tools/onnx_mj_binglian.py#L758)、[XYEGA实车力矩路径](https://github.com/chushanxiaodaoshi/XYEGA_RM2026_WheelLeg_Infatry_RLdeploy/blob/fcfdd3959be5c9b00893ec0701a591ddbf55b830/source_code/User/Application/chassis/chassis.cpp#L2220)。

`l0`是髋到轮的虚拟腿长，单位m，不是气弹簧伸缩长度。由F/T矩阵量纲推断，369／370.1等是N/m的等效补偿系数，源码未给标定曲线；不能称为369N／370N的气弹簧推力，也不能认为它就是实体弹簧刚度。举例l0=0.30m时，370.1对应径向修正111.03N。该修正的左右符号依赖参考坐标系。本机不能把150N直接填入这一系数，也不能只复制加减号。来源：[XYEGA几何及F/T矩阵](https://github.com/chushanxiaodaoshi/XYEGA_RM2026_WheelLeg_Infatry_RLdeploy/blob/fcfdd3959be5c9b00893ec0701a591ddbf55b830/source_code/User/Application/chassis/chassis.cpp#L1538)。

#### 机构、索引及起立路径限制

- 复旦参考几何参数l1/l2为0.175/0.208m；本机为0.21/0.25m，且CAD偏置与弹簧锚点不同。虚拟膝、Jacobians和补偿均须用本机几何，不按名称套槽。
- XYEGA物理数组为`[左大腿,右大腿,右小腿,左小腿,右轮,左轮]`，与本机DM0～3＋两轮不同；训练顺序与输出顺序必须分别处理。[物理枚举](https://github.com/chushanxiaodaoshi/XYEGA_RM2026_WheelLeg_Infatry_RLdeploy/blob/fcfdd3959be5c9b00893ec0701a591ddbf55b830/source_code/User/Application/chassis/chassis.h#L101)
- XYEGA的LargeRecover分支会清空RL力矩、改由专用恢复控制计算输出；不能把其大角度自救当成本机后摆起立模型的同任务验证。[最终仲裁](https://github.com/chushanxiaodaoshi/XYEGA_RM2026_WheelLeg_Infatry_RLdeploy/blob/fcfdd3959be5c9b00893ec0701a591ddbf55b830/source_code/User/Application/chassis/chassis.cpp#L1454)
- 复旦`onnx_mj_chuanlian.py`与`binglian.py`驱动对象不同：当前chuanlian默认同样加载闭链XML，但直接驱动虚拟膝，并将弹簧输出清零；其行为不能作为实体输入轴执行验证。[控制写入](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/mujoco/python_tools/onnx_mj_chuanlian.py#L748)

本次得到的是可验证的新假设：训练串联代理与实物弹簧之间的等效力矩差异可能影响当前受限起立。已有“150N物理弹簧＋无软件补偿”的本机闭链模拟能起立，所以仅凭别人包含补偿不能确认本机故障。下一步用本机实测姿态和弹簧锚点量化等效力矩，并取得实际弹簧力／行程信息后再决定补偿方式。未调整实机限幅、未烧录或启动新试验。

#### 本机静态弹簧力矩量化

随后使用第04次会话的首帧与末帧观测、本机当前XML及`ClosedChainAdapter`重建闭链姿态，只执行`mj_forward`和几何中心差分，不运行动态步进、策略推理或硬件。每侧150N模型弹簧折算到当前B／USB逻辑槽：

| 记录帧 | DM0 | DM1 | DM2 | DM3 |
| --- | ---: | ---: | ---: | ---: |
| seq0，起点 | +5.851 | −5.851 | +5.832 | −5.832 |
| seq199，1.99s停滞姿态 | +5.905 | −5.905 | +5.854 | −5.854 |

单位N·m。这是弹簧在机构上的模型等效作用，不是应直接发送的补偿命令；抵消作用还须取反并核对本机坐标、实际参数和输出限幅。其绝对值约为当前10N·m试验限幅的58%～59%，支持优先核查弹簧作用。MuJoCo广义力与长度中心差分残差小于7e-9N·m，仅证明计算内部一致。

可复算记录：`H7_clion/captures/20261001-policy-direct-run-04-feedback/analysis/reference-spring-check/{check.py,result.json,README.md}`。XML SHA-256为`66e894257b50f9e78c70733c270a0beeb3cb678f76f3c26def25437917e9a2a0`，adapter为`7a9f573e382ec10b5fbfa89a9d0d156734438f56678c0f9962be3b133b01d3ae`；原始run04记录SHA与上文一致。run04的host源码包没有归档XML，因此本次严格称为“当前XML在历史实测观测姿态下的估计”。计算未包括实物弹簧变化、重力、地面作用、摩擦或加速度，也没有独立验证B实体轴方向；不能据此断言故障根因或增大输出。

### 用户提出极性假设后的复查

用户怀疑电机极性导致未起立，本轮优先检查该假设，不将弹簧差异认定为根因。只读对比确认9个关键H7源文件与已执行ELF配套源码快照一致，包括观测、控制、几何、runtime、安全层、DM/DJI驱动和machine_config；没有代码或实机状态修改。

| 槽 | 控制CAN ID | 原生反馈→逻辑反馈 | 逻辑角→模型输入角增量 | 当前模型对应（B候选） |
| --- | --- | ---: | ---: | --- |
| DM0 | 0x01 | + | + | 实物左／模型rf后输入 |
| DM1 | 0x03 | + | + | 实物左／模型rf前输入 |
| DM2 | 0x02 | − | − | 实物右／模型lf后输入 |
| DM3 | 0x04 | − | − | 实物右／模型lf前输入 |

驱动输出符号与反馈符号成对，均为`[+,+,−,−]`；控制端DM力矩恰为观测速度变换的转置乘虚拟力矩，物理限幅前满足虚功／功率一致性。右侧出现两层负号是原生→逻辑→模型的坐标链，未发现重复取反缺陷。两轮CAN201/202的符号及最终`motor[5],motor[4]`发送顺序也相配。这仅证明软件内部一致，不能证明配置与实体轴完全一致。

右侧方向的独立推导：站在实物右侧，即模型−y侧朝+y看，XML的lf0/lf00正轴−y指向观察者，CAD正角对应逆时针。用户／视频记录中的右视顺时针对应CAD角减小，而两次单槽+5N·m期间逻辑反馈角均增加；当前右侧`Δq_model=−Δq_logical`与已有证据一致。该推导不依赖长短臂身份，但也不能独立验证B绑定、绝对零偏或Jacobians。左侧当前符号预测“从左侧看，逻辑反馈增加为逆时针”，仍只有用户此前要求的镜像推断，没有左侧独立脉冲证据。

历史原始USB重解码：DM2、DM3各50个CRC有效SET_TORQUE包，只有目标槽+5、其余五槽零；匹配STATUS均回同一目标。按已核对驱动符号，右槽原生MIT力矩指令应为−5N·m（12位编码1857），这是固件路径推导，不是独立CAN抓包。脉冲前最后状态至末个+5应答的逻辑位移为DM2 +0.945425rad、DM3 +0.969394rad，50帧速度均为正、位置单调增加。上文历史+0.96949/+0.99317rad使用随后STOP时刻的状态，包含额外时间，不能混用终点。

本轮结论：当前证据支持右侧两轴旋向转换，未发现软件符号缺陷；整机实体极性仍不能全部排除，尤其左侧独立带力矩证据不足。用户随后明确怀疑来自“轮打滑、腿不动”，未补充某台电机实际反向的观察。因此该现象保持为症状，不能单独确定极性故障。既有失能手动后摆→直立采集可检查观测随姿态的变化，但不能独立验证左侧输出极性。任何修正须先区分下发符号、反馈符号、坐标转换与轴绑定；若改位置符号，须连同速度／力矩变换及同一实体零位的偏置一起核对。未直接翻转任何槽，未启动新的带力矩试验。

原始脉冲重解码与时界报告：`H7_clion/captures/20261001-polarity-dm23-plus5-analysis-01/{README.md,report.json,analyze.py}`。该目录为新增离线分析，不覆盖任何原始采集。
