# 训练与部署接口约定

2026-10-29：[稳站v2 model5999的H7_ros编译候选](interfaces/20261029-h7ros-stand-v2-model5999.md)已完成ONNX、160维历史/双输出SDK、AIR反馈估力和实机恢复代理适配及大小机编译；用户确认使用AIR估力，明确不烧录。USB v3当前帧32与旧v1/v2兼容解析已同步电脑端；板端真实时序、估力/接触代理及真机和H7负责人核对待确认，不将编译候选当作真机通过。[交付与证据](model-deliveries/20261029-stand-v2-model5999-h7ros.md)。

2026-10-25后续：[轮角39／82 SMP PPO v4](interfaces/20261025-wheel39-smp-ppo-v4.md)已接通当前BC初始化及PPO保存／恢复、play／eval／check，按检查点定义区分旧零轮角39维与首次站稳清零39维。本轮仅零更新检查和回放，动作、物理、奖励、初态与时序保持；普通PPO默认不变。新轮角CPU／ONNX／H7未适配，旧导出器拒绝，部署负责人核对待确认。

2026-10-25后续：[BC共享位置参考41／84](interfaces/20261025-bc-shared-reference.md)保留通用轮式里程，再追加同拍教师位置误差，显式保留旧39维权重后新增本地5000次；独立起立19/40、平衡15/40、停车0/40，行为未通过。普通PPO／SMP PPO、CPU和H7继续39维，41维不直接部署；H7负责人核对待确认。下方40维描述为前一阶段记录。

2026-10-25：[BC通用轮式里程](interfaces/20261025-bc-wheel-odometry.md)在BC采集／训练／独立回放追加actor第40维、critic第83维，使用回合起点的左右轮连续编码器角差，适用于起立、站立和移动，不绑定停车目标。普通PPO／SMP PPO、CPU和H7保持39维；新40维BC不能直接接入旧网络或部署。本轮无新策略训练／导出／固件同步，H7负责人核对待确认。

2026-10-23：mjlab与独立CPU为100 Hz策略／500 Hz内环；USB工程`deployment/H7_RL_ros`已导入model13842并将大机名义策略周期同步为10ms／100Hz，PD仍500Hz，编译通过，并于本次动力断开确认后烧录、校验、复位成功；板端真实时序/行为待测。UART工程仍50Hz。时序接口、模型/固件哈希及板端待确认项见[13842交付](model-deliveries/20261023-model13842-h7-import.md)与[100Hz接口](interfaces/20261023-mjlab-policy100hz.md)。

2026-10-22：multimode训练启用四腿默认角初始化随机偏置，作者最新将范围从±0.03 rad扩大到±5°（±0.08726646 rad）；PD目标与相对角观测共用，play/CPU保持原默认角，39/82/6及动作映射不变。仅训练随机化，不修改板端零点；新增1000轮训练预算、行为及H7负责人核对待确认，见[默认腿角偏置记录](interfaces/20261022-huananhu-nominal-leg-offset.md)。

2026-10-22：[H7LAT001本地闭环时序旁带](interfaces/20261022-h7-latency-capture-v1.md)已完成烧录、吞吐修复与正常RL实采；按作者最新要求，训练动作等待为每回合腿/轮独立随机6、8、10 ms，play及独立CPU回放保留实测量化的一拍2 ms，见[仿真时序记录](interfaces/20261022-h7-simulation-timing.md)。39/82/6、模型、标定和控制频率保持；旧模型加载兼容，闭环行为需重新评估，H7负责人核对待确认。

2026-10-21作者已要求将joint_sharp等效轮端响应写入共享XML并同步仿真执行链。训练与CPU均保留obs39/action6、原名义轮速PD/±3.9Nm输入限幅及策略/控制节拍；只改变仿真指令到响应模型，G/V/p同源于XML、增益只乘一次。H7固件/物理标定/生成网络均未改，未开始训练或交付新权重。部署负责人核对状态待确认；检查结果/随机化保留/未验证工况见[接入报告](diagnostics/20261021-h7-wheel-response-integration.md)。

2026-10-21固定300ms台架固件已烧录并完成有动力左右正反脉冲：两份独立120秒会话各4次，完整记录各240000项，缺口/新增丢弃/CRC/截包/HAL失败0。原RL纯P版本0、权重/机器标定/普通RL与LQR保持；bench_profile及固件SHA单独区分旧保护版。新增等效轮端候选改善总体轮速，但加速阶段及旧RL短时俯仰有退化，整机校准未通过，不改正式XML、不训练或同步环境。新完整500Hz输入RL验证及H7部署负责人复查仍待确认，见[完整输入校准报告](diagnostics/20261021-fixed300-complete-input-calibration.md)。

2026-10-21独立worktree新增`h7-mirror-v2+output-capture-v1`采集准备：原644字节镜像前缀后，同次USB提交附加每拍控制/真实DJI CAN请求队列，含序号/时间/原接收年龄/台架原因/入队状态。原纯P控制版本0、39/6接口、权重和标定保持；软件及GCC构建通过；大机已烧录/校验/复位，断动力30秒实际USB观察59980项记录，无缺口/新增丢弃/CRC拒绝。非RL投入时序或机械响应验收；H7负责人、RL板端节拍与长期吞吐仍待确认。旧录制不能补出此数据，详见[完整输出接口](interfaces/20261021-h7-output-capture-v1.md)。

2026-10-21 USB工程新增作者确认的轮端补偿试验源码，尚未烧录：大机`[0.11604,0.06147]Nm*tanh(轮速/0.5rad/s)`叠加原轮速比例输出后统一±3.9Nm限幅，小机保持原控制。obs39/action6、权重、时序、物理标定及正式XML保持；该试验改变输出控制行为，不能宣称与原训练控制器等价或无需行为复核。USB v2/644字节保持，槽158新增控制版本，录制归档对应系数，离线预测须匹配实际控制器。H7负责人核对和台架待确认；已确认系数的离线闭环复核未通过，当前不得标为已消除gap，证据见[实验记录](ai/project-log.md#当前状态2026-10-21)。

`deployment/H7_RL_ros`为独立USB诊断工程，最新只读v2新增配对obs39／action6及输出诊断；2026-10-21大机已烧录／校验／复位，Jetson十秒1000帧v2、CRC拒绝／序号缺口0。随后10:34动态会话3001帧包含1501个实际推理配对／2876个有效PD样本，已核对计算与电脑一致；有人扶持／支架约束，物理映射及同条件动力学对照仍待台架。接口与验收见[USB镜像记录](interfaces/20261020-h7-ros-usb-mirror.md#只读对照信号v2)。下文主工程`H7_RL`的UART v1及其历史结论不代表USB v2行为验收。

H7部署工程作为普通目录保存在[deployment/H7_RL](../deployment/H7_RL/md/file-guide.md)，与训练代码统一管理。2026-10-18已同步独立H7仓的 `e2a3339519add73805df531e6247233a9540fbf0`；生成网络为 `model_5849`，actor39/action6、500Hz实体PD、50Hz推理，并接入平地遥控与倒地恢复。同步范围、LQR整定、构建记录和待复查项见[部署同步记录](interfaces/20261018-h7-model5849-deployment-sync.md)；部署规则见[H7协作规范](../deployment/H7_RL/md/AGENTS.md)。源码同步不表示独立复查或真机验收通过。

## 新 mjlab 实体关节接口

2026-10-21当前USB大机固件已写入[RL初始角度分支锁存修复](interfaces/20261021-h7-rl-angle-branch-fix.md)：首次有效RL反馈锁存整数周期，观测与PD共用连续模型角，退出RL才重新选择。模型仍6848，标定/增益/驱动/消息通道不改；板端实际角度和失联问题尚待实测，负责人待确认。

mjlab 训练侧只保留华南虎 multimode：actor39／critic82／action6，2 ms 物理、20 ms／50 Hz 策略，动作顺序 `[lf0,lf00,lfwheel,rf0,rf00,rfwheel]`，模式槽 `28:35`、标志槽 `35:39`。观测打包与 ONNX 契约见 [multimode 接口](interfaces/chuanliantui-huananhu-multimode-replay-v1.md)、[model_3023 说明](onnx-huananhu-multimode-model3023.md)及 [model_5849 说明](onnx-huananhu-multimode-model5849.md)；当前基线、模型采纳与验收结论以[项目实验记录顶部](ai/project-log.md)为准。

当前主UART工程`deployment/H7_RL`的生成网络仍为 `model_5849`；实机USB工程`deployment/H7_RL_ros`已按2026-10-23授权导入并烧录 `model_13842`。两者输入39、输出6，不需要encoder历史输入；默认大机采样／PD／正常下发500Hz，UART策略50Hz、USB策略100Hz。命令复用公共 `rc.vel/yaw/len`，速度／转向仍限制到±1.5m/s、±1.5rad/s。USB大机最新源码将腿长目标按`h=clamp(rc.len+0.03648,0.2165,0.3664)`换算训练机体高度，RL拨轮腿长范围0.18002～0.32992m，低位恢复门槛配套调整；训练参考偏移不改变实机标定。**该换算已于2026-10-23烧录大机、读回校验并复位**；UART工程仍为腿长直接裁剪0.27～0.34m的旧逻辑。见[目标换算接口](interfaces/20261023-h7-leg-to-body-height.md)。

2026-10-22[起立后零速保持训练准备](interfaces/20261022-huananhu-zero-hold-training.md)：作者指定腿长0.18～0.33m，训练/CPU高度命令换算为0.2165～0.3664m；39/82/6接口保持，H7未同步范围及低位恢复门槛、未生成新模型、未训练或烧录。该训练目标不代表原固件已支持，部署负责人核对待确认。

2026-10-21新增[model6848模型文件交付](model-deliveries/20261021-model6848-h7-export.md)：完整PT、opset13 ONNX及参考向量存于USB工程`deployment/H7_RL_ros/models/model6848-20261021/`，39/6接口不变、导出及零速ONNX回放通过。随后按独立授权完成C转换、64组真实生成C宿主数值检查、大机构建和实际写入/校验/复位；[本次烧录证据](interfaces/20261021-h7-model6848-flash.md)。H7负责人对6848板端数值/时序及实机核对仍待确认，未运行带动力试验。

当前已实现平地NORMAL／RECOVER；趴地投入保持200ms，连续稳定500ms转NORMAL，连续倒地严格超过2000ms锁定失败，须退出投入重试。髋轮投影间距代替仿真绝对机身高度／接触测量，airborne／step_active仍固定0，未实现跳跃、台阶、坡地模式。H7负责人对板端数值、真实时序、物理映射和恢复行为的核对状态仍为**待确认**；更换模型须成套核对权重与接口，不能只替换尺寸宏。

| 数据 | 顺序／用途 |
| --- | --- |
| H7内部反馈、观测电机段及PD诊断 | `[lf0, lf00, rf0, rf00, lfwheel, rfwheel]` |
| 网络原始动作及上一拍动作缓存 | `[lf0, lf00, lfwheel, rf0, rf00, rfwheel]`；写入观测时按 `[0,1,3,4,2,5]` 重排 |
| `q_nominal` | 观测减去的默认角，也是腿目标中心 |
| `q_reference` | XML的 `qpos0`，用于目标限幅中心；不是实机编码器零点 |

<a id="host-debug-plan"></a>
## 实机调试与上位机（UART10已烧录，首次双向通过）

2026-10-18用户批准实施只读镜像：H7上传公共电机／IMU反馈，Jetson重构闭链姿态；不依赖ROS 2，
不回发电机命令。固件与接收／显示软件已经实现并通过软件检查；此前UART8版已烧录并实测上行，
2026-10-20 UART9版已烧录、校验并复位；随后按用户确认改为UART7通信测试，本次断开电机动力确认后已烧录并通过FUNC_UART双向探测。之后用户要求源码改回UART8复测，恢复UART7上的HI229、继续暂停DR16；UART8复测版已在本次断电确认及明确重烧要求后写入／校验／复位成功，随后UART8与FUNC_UART首次双向探测通过，但紧接着第二次复测下行无回报，不能据首次结果认定稳定。随后用户要求完整互换UART7／8，完成源码后选择先不烧录；最新要求撤销该互换，源码已恢复UART7接HI229，并将Jetson链路改到H7 UART10；用户最新要求恢复遥控，源码已恢复UART9 DR16接收。用户本次确认断电及接线后，恢复遥控的UART10版已烧录／校验／复位，并通过首次十秒双向探测。传感器／电机物理映射及机器人行为仍待台架确认。旁路策略复算、同指令独立仿真及电脑接管继续属于后续规划。

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

默认1152000波特、8N1。当前大机源码改用UART10的PE3/TX，接转换器RX并共地；HI229恢复UART7，921600波特。电平／板上接口现场核对；
仅运行只读镜像时不需要接H7 RX，也不连接转换器DTR／RTS作复位；双向探测接法见下一节。屏幕位置和高度固定，均不是实测；闭链残差门槛1mm。
被动关节即使满足闭合残差，超出XML行程仍判无效并回滚，避免显示错误装配分支。
任一部位失效保留最后有效姿态并明确标记，200ms无合法新帧标STALE；红色初态不是收到实机数据。
串口仍有新帧但IMU／电机软件发布年龄达到200ms、未发布或非法时，对应来源同样冻结并标记过期。
镜像帧与兼容性见[本次接口记录](interfaces/20261018-h7-jetson-mirror-v1.md)。

### H7与Jetson双向串口探测（2026-10-20，UART10已烧录，首次双向通过）

2026-10-20复查后源码已修复RX DMA错误恢复及主机迟到回报判断：保留RX的TC中断完成HAL异步终止；主机先检查250ms期限再接受ACK，到期即超时。修复版软件检查及大小机编译链接通过，证据见`artifacts/uart10-review-fix-20261020/check.json`。用户随后要求完成修复，修复版已实际烧录／校验／复位，Jetson常用脚本已更新并备份旧版；三次TX BREAK错误注入后，H7接收重启计数依次0→1→2→3，每次恢复后26发／26回、超时0，证据见`artifacts/uart10-dma-recovery-20261020/error-recovery.json`及同目录`flash.json`。以下首次十秒实测仍对应修复前版本；本次覆盖BREAK引起的接收错误恢复，不代表长期稳定性或机器人控制验收，H7负责人核对仍待确认。线序、协议及控制接口不变。

本阶段验证收发链路，不迁移推理、不接收action。当前大机源码为UART10接Jetson、UART7接HI229、UART8保留VOFA原LQR 48路布局；HI229初始化／解析保持启用，按用户最新要求恢复UART9 DR16初始化／接收解析，保留IMU／遥控离线故障与既有出力门控。当前验证仍限软件与通信，真实遥控／机器人控制待台架。H7继续约100Hz在UART10上传196字节镜像，
`jetson_uart.c`在通信任务轮询同一UART的环形RX DMA，只接受14字节CRC探测包；回报复用镜像预留槽，
不增加发送分支或RTOS任务。上位机`sim2sim/h7_serial.py`仅依赖pyserial，不依赖MuJoCo或策略模型。

当前UART10配置接线按信号名核对（均为3.3V串口信号）；Jetson通信线接H7板上UART10接口，HI229保留原UART7：

| H7源码引脚 | WeAct N002功能串口FUNC_UART |
|---|---|
| UART10 TX / PE3 | RX |
| UART10 RX / PE2 | TX |
| GND | GND |
| 电源脚 | 不连接两板的3.3V／5V电源脚 |

HI229接法恢复为：传感器TX→H7 UART7 RX／PE7，传感器RX←H7 UART7 TX／PE8，共地，保持原921600波特率与原供电方式；Jetson与H7电源脚仍不互连。UART10硬件名为USART10，PE2/RX使用AF4、PE3/TX使用AF11，已按[STM32H723数据手册DS13313表8](https://www.st.com/resource/en/datasheet/stm32h723ve.pdf)分别配置。

DR16遥控接收已在当前UART10源码中恢复：接H7 UART9 RX／PD14，100000／8E1，沿用原供电与接收线序；离线保护和原拨杆许可不变。恢复版已烧录，真实遥控接收待确认。

N002使用照片中五个白色插座的中间第3个；实际板上接口、H7电平和引脚仍须核对。不要接DBG_UART。
N002厂家教程把Orin Nano/NX的FUNC_UART映射为JetPack5的`/dev/ttyTHS0`、JetPack6/7的`/dev/ttyTHS1`，
最终以实际系统为准；USB转串口则用稳定的`/dev/serial/by-id/...`。波特率沿用1152000、8N1、无流控。

```bash
# Jetson终端：先查看实际串口，再将下条命令的设备名替换为对应接口。
cd ~/real2sim
.venv/bin/python sim2sim/h7_serial.py --list-ports
ls -l /dev/ttyTHS*
.venv/bin/python sim2sim/h7_serial.py --port /dev/ttyTHS1 --seconds 10
```

2026-10-19用户当次授权烧录并确认电机动力断开，助手已编译、烧录、校验并复位H7；固件已包含探测模块。Jetson已通过网线SSH认证并部署检查脚本。接线调整后及2026-10-20双方重新上电后，ttyTHS1十秒均收到1000帧、校验拒绝0；39个下行探测均未获得回报。重新上电后的只读SWD检查显示H7 RX DMA已启用、剩余计数256、接收缓冲区全0、合法包及CRC错误数均0，说明下行字节尚未进入H7 DMA；仍需检查Jetson TX至H7 RX路径，双向通信未通过。
上述失败记录属于此前UART8固件。换底板后，Jetson FUNC_UART曾通过1152000波特率、三次重开串口、15包／3840字节随机二进制外部回环；恢复原因和重新上电后的保持情况未确认，不能据此认定旧底板损坏。接回H7 UART8仍只通上行。用户随后要求改UART9：源码已切换，RX沿用DMA1 Stream4，TX在CubeMX用户区补DMA1 Stream0及IRQ，UART9 IRQ转交HAL，避免误入未初始化的DR16 IDLE缓冲。大小机完整编译／链接与协议软件检查通过；UART9版已在用户本次断开电机动力确认后烧录／校验／复位成功，随后只读检查确认程序、UART9 DMA和引脚配置运行，DR16离线故障有效。正式十秒联机收到1000帧、校验拒绝0，但38个下行探测均超时；用户已确认黄线接回，双向尚未通过。证据见`artifacts/serial-uart9-20261020/flash.json`与`uart9-link-1.json`；H7负责人核对仍待确认。
此前UART7版已按用户本次断电确认烧录／校验／复位成功，用户确认接线完成后，FUNC_UART正式十秒检查收到1000帧、校验拒绝0，92个探测全部回报、超时0，平均往返9.98ms、最大10.13ms。只读镜像同时确认IMU／DR16离线及对应故障位置位；这是通信测试配置，不能用于机器人控制。证据见`artifacts/serial-uart7-20261020/uart7-func-link.json`与`offline-protection.json`。UART7与FUNC_UART本次双向通过，不据此认定此前UART8／9失败的根因；H7负责人及真实控制时序／行为核对仍待确认。
此前UART8复测版已烧录：首次1000kHz下载在写入期间检测到外部复位并失败；用户明确要求重新烧录后，以200kHz SWD下载成功，写入／校验／复位标记齐全。固件内容不变，失败根因未确定。用户对移线要求回复确认后，UART8与FUNC_UART正式十秒复测收到1000帧、校验拒绝0，92个探测全部回报、超时0，平均往返9.99ms、最大10.13ms，双向通过；见`artifacts/serial-uart8-retest-20261020/flash.json`及`uart8-func-link.json`。此结果不说明之前下行失败的根因，也不替代HI229、电机及策略行为验收。
作者要求再次测试后，使用同一固件／端口重新打开串口测试十秒：收到1000个有效镜像帧、校验拒绝0，39探测回报0、超时39。本次只证明H7上传正常，下行未通过；与首次通过结果不同，期间是否有物理接线变化及根因均未确认。证据见`artifacts/serial-uart8-retest-20261020/uart8-func-link-2.json`；本次未烧录或修改代码／串口配置。
作者再次要求测试，第四次十秒收到1000有效帧、校验拒绝0，93探测全部回报、超时0，平均往返10.00ms、最大10.48ms，本轮双向通过。前两次下行失败仍保留，不能由本轮通过认定稳定或确定根因；本次未改配置／代码或烧录。证据见`artifacts/serial-uart8-retest-20261020/uart8-func-link-4.json`。
本次确认电机断电及UART10接线后，前两次OpenOCD在CMD_INFO阶段失败，均未开始写入；确认下载器未被占用后重置该PowerDebugger TX的USB连接，随后以200kHz成功写入／校验／复位同一HEX。UART10／FUNC_UART正式十秒收到999帧、校验拒绝0，93探测全部回报、超时0，平均往返9.99ms、最大10.13ms，首次双向通过；本次未测试真实遥控、传感器或电机动作，长期稳定性待确认。证据见`artifacts/serial-uart10-rc-20261020/flash.json`和`uart10-func-link.json`。
不要与姿态镜像程序或VOFA同时打开同一串口。看到`probe_firmware: true`、`probe_ack`持续增加、
最终`ok: true`且`timeouts: 0`才算本次双向探测通过；只看到`mirror_frames`增加只证明上传。
旧固件没有探测扩展时只接收、不发探测，最终失败；串口断开、写入失败或探测超时也返回非零退出码。
`rtt_ms_*`是同一主机时钟测得的探测往返时间，包含等待后台镜像发送的时间，不是策略控制延迟。
后续推理接管仍需独立定义观测／动作协议、唯一动作来源、模式切换、时效和断线处理。

| 阶段 | 拟完成内容 | 可观察的成功判据 |
| --- | --- | --- |
| 只读采集与姿态镜像 | 软件已实现；公共原始反馈、时间、序号、在线状态与闭链重构；实际串口／方向待台架 | 软件故障检查通过，联机后须逐电机手动核对动作／单位／方向 |
| 旁路策略复算（尚未实现） | 上传同一次推理配对的obs39／action6，电脑用同权重模型复算 | 有效配对数据的动作误差达到交付数值阈值 |
| 同指令独立仿真 | 实机和MuJoCo各自闭环运行策略，从零速静站开始比较关节、倾角、轮速和动作 | 同一记录可重复分析；分别报告误差与实验条件，不强求长时间轨迹重合 |
| 电脑接管策略（后续候选） | 电脑50 Hz输出动作，H7保留500 Hz电机控制与唯一出力门控 | 验证延迟、动作过期、控制段切换、断线和重新投入；通过前不作为首期功能 |

遥测分两路且都沿用后台固定布局、一次发送方式：UART8的VOFA保持原48通道LQR布局，UART10的Jetson镜像是镜像v1（四元数／原始反馈／序号／时间／CRC），两者不共用端口、缓冲或发送函数，也不按模式分页。镜像帧没有obs39或action6，不得当作网络输入／动作配对证据。后续复算仍须在同一次推理完成后配对保存，不能用发送序号补救异步发布。

不同对比回答不同问题：

- 同一obs输入的网络对比只核对推理数值，不证明实机观测的缩放、方向或零点正确。观测构造对比须使用同一份未缩放原始输入。
- PD对比须使用同一个控制子步的角度、速度和动作；不能拿50 Hz观测反缩放冒充500 Hz当拍反馈。
- 实机镜像按测量更新模型姿势；新增求解器固定lf0/lf00、rf0/rf00，重构被动关节与弹簧，以前一姿态初始化。
  原站姿解算仍只服务sim2sim，不改它；`mj_forward`不会自动求闭链。镜像仅用运动学，不运行物理步进。
  驱动已处理公共极性／零点和轮减速比，Jetson仅追加既有RL模型坐标映射一次，物理正确性仍待台架。
- 录制action后回放是开环动作序列诊断；实机与仿真分别读取各自状态运行策略，才是同指令独立闭环。两者都不能反推物理映射已正确。

电脑接管还须接通收包、明确唯一动作来源、处理上一动作与复位，并拒绝旧控制段的在途动作。现有出力许可和动作超时仍依赖板载 `policy.ready`，不能从后台直接写 `action_state` 绕过现有门控。UART8可双向收发，不强制另占UART；未来控制收包须独立定义定长字段、校验与时效，当前JustFloat遥测帧不能直接作为控制协议。

当前镜像布局、探测接收链和时序边界见 [H7总览的Jetson节](../deployment/H7_RL/md/file-guide.md#jetson)，VOFA布局见同页[VOFA节](../deployment/H7_RL/md/file-guide.md#vofa)。旧S2R1／HPI1工具见[历史上位机说明](ai/s2r-offline.md)，不能直接套用到当前39维固件。

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
