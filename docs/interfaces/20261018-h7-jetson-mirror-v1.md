# 接口记录：H7→Jetson只读姿态镜像v1

| 项目 | 内容 |
|---|---|
| 日期、作者、复查 | 2026-10-18，Codex按用户批准方案实施；Claude独立代码审查；H7负责人／台架核对待确认 |
| 对象 | 大轮腿，H7＋Jetson Orin Nano；沿用当前multimode XML，不修改物理参数 |
| 旧接口 | UART JustFloat固定48路LQR诊断，见H7总览历史；不是obs39或控制协议 |
| 新接口 | `h7-jetson-kinematic-mirror-v1`，单向只读；标识7101、版本1、机型0 |
| 代码版本 | 同一仓库`26_wheelleg`，HEAD `ca6cdabf0b61d2ce0e13e15ae7b93bfa9aed333a`＋本轮未提交改动；源码哈希清单在本机证据目录 |
| 权威实现 | H7 `vofa_debug.c::Vofa_Debug_Send`；上位机`sim2sim/real2sim_huananhu.py`；字段表只维护于[H7总览](../../deployment/H7_RL/md/file-guide.md#vofa) |

## 接口定义与兼容性

沿用48个小端float32加4字节JustFloat尾，共196字节。正文含原始驱动电机角／速、四元数wxyz／角速度、
逐来源有效位、软件快照时间、发布年龄及提交序号；序号与毫秒各以两个精确16位float编码。
CRC16/CCITT-FALSE（poly1021／initFFFF／不反射／xorout0）校验前188字节，结果以float数值存于第47槽。
100Hz按后台实际经过时间尝试，DMA忙跳过；序号仅成功提交后增加。发布时间不等于采样时间，
提交序号缺口不等于CAN丢帧，也不跨H7／Jetson时钟直接测端到端延迟。

H7保留原控制、驱动校准和网络；新协议不发送obs39、action6，不接收控制命令，不需要重训或重新导出。
旧VOFA软件仍可读取浮点数组，但曲线含义变化；旧LQR布局保留整段行注释，手动启用后Jetson会拒绝该帧格式。
首期接收器拒绝机型1，不能拿大机模型显示小机。历史S2R1／HPI1脚本不兼容本协议。

## 映射、运动学与失效行为

传输角为H7公共驱动已校准的连续角，轮角／速已除减速比。Jetson从本次交付的`machine_config.c`
读取大机RL `motor_sign/motor_zero`，只做这一次额外模型坐标转换；不重复驱动符号、零点或减速比。
这份配置须与烧录固件一同交付，后续改映射时同步Jetson副本。
物理正确性仍必须逐电机和IMU三轴手动核对，不能从网络数值或模型残差反推标定正确。

固定实测lf0/lf00与rf0/rf00，以上一姿态为初值求各侧四个被动关节，再重构弹簧；保持驱动角不变。
只运行运动学，未调用物理步进、策略推理或串口write。连接残差超过1mm、被动关节超出XML行程或求解失败则回滚该腿；
IMU、左右腿和两轮分别保留最后有效姿态并标记异常。200ms无合法新帧为STALE；初始红色模型为NO DATA。
即使串口仍有新帧，IMU或电机软件发布年龄达到200ms、未发布或年龄非法，也冻结对应来源并标记过期。
机身平移保持XML初值，屏幕明确标明位置／高度非实测；装配分支和实机运动尚待台架。

## 同步与验证

| 部分 | 状态与依据 |
|---|---|
| H7源码／构建 | 大小机各104来源ARM GCC编译并链接通过；旧BMI088 packed告警保留；15条已不存在的eIDE虚拟引用单列，未改工程清单 |
| Jetson环境 | Ubuntu24.04／Python3.12；MuJoCo3.11.0、NumPy2.5.3、SciPy1.18.1、pyserial3.5；模型与源文件按哈希核对 |
| 软件与模型 | 实际C打包代码（硬件替身）、损坏流／伪尾／分包、计数边界、伪串口断线与零输出、已知模型姿态与失效回滚检查；证据在`artifacts/real2sim-20261018/` |
| 烧录／串口／真机 | 未烧录H7，未接通真实数据；方向、轴向、零点、实际节拍及装配分支未验收 |
| H7负责人核对 | 待确认；AI软件审查不代替另一位维护者签认 |

Claude独立复查重跑大小机编译／链接、CRC与伪串口检查，未发现P1阻断问题；发现的两处P2已修复并复测：
拒绝闭合但超出被动关节行程的错误分支；宿主检查直接核对真实`Vofa_send.c/h`的帧尾与48路容量。
IMU有限性由上游`Attitude_Update()`校验后发布online，Jetson另检查四元数；轮位置沿用轮速度的符号转换，实物方向仍待手转确认。

可执行命令见[实机镜像操作](../deployment-contract.md#host-debug-plan)及[构建检查](../ai/build-test.md)。
本轮没有提交或推送，既有工作区改动保留；后续交付模型或电脑接管须另记接口，不复用此只读帧作控制协议。

## 2026-10-19：增加双向串口连通性探测

| 项目 | 内容 |
|---|---|
| 请求与范围 | 用户要求先实现串口通信；补双向收发与校验，不迁移推理、不接收动作或改控制许可 |
| 工作区 | 同仓当前`26_wheelleg`与未提交改动；既有镜像／快照改动保留，无提交或推送 |
| 下行探测 | `J7P1`四字节包头＋小端uint32序号＋uint32随机数＋uint16 CRC，共14字节；CRC16/CCITT-FALSE校验前12字节 |
| 上行回报 | 原196字节镜像不变，32槽为探测扩展版本1，33～34序号、35～36随机数、37～38合法包数；uint32仍拆两个16位float；39～46仍为0 |
| 收发职责 | main初始化RX DMA；`jetson_uart`由comm每拍轮询解析，只更新通信状态；后台短临界区复制回报到原镜像，无新增发送路径或任务 |
| 兼容性 | 原姿态镜像解析器忽略新增预留槽，仍保持只读；探测客户端发现32槽不是1时不发送，失败退出；旧LQR／RL布局仍不兼容 |
| 模型影响 | obs39、action6、控制周期、物理映射、生成网络和出力门控均未改变，无需重训或导出 |
| 主机工具 | `sim2sim/h7_serial.py`，仅pyserial；最多10Hz、一个在途请求、250ms超时；序号与随机数配对，不把旧回报算成本次响应 |
| 接线 | H7大机UART8 PE1/TX→N002 FUNC_UART RX，PE0/RX←FUNC_UART TX，共地，双方独立供电；实际接口／3.3V电平／波特率仍待现场核对 |
| 时序边界 | RX DMA 256字节，消费间不得写满一整圈；仅为低速探测。主机往返耗时包括后台镜像等待，不等于推理或控制延迟 |
| 软件验证 | 实际C解析／打包、分包、噪声、坏CRC、DMA环尾／重启、计数回绕、模拟串口双向／丢请求／旧固件，以及原镜像检查通过；大小机各105来源ARM GCC编译／链接通过，见`artifacts/serial-20261019/` |
| 部署与复查 | 2026-10-19用户确认本次电机动力断开后，助手完成实际编译／烧录／校验／复位，HEX SHA256为`07f824d06d9c5a4341d1a226f745e5568cf47a7261f3d0117debd3a38b152683`；Jetson已登录并部署脚本，实际ttyTHS1十秒检查0帧；Jetson单包发出14字节，H7接收0包；H7镜像提交计数持续增加，待核对TX／RX／GND，尚未通信通过；H7负责人待确认，没有独立复查 |

MDK工程已登记新源，eIDE目录自动发现；固定`builder.params`是来源清单快照，必须重新生成／纳入
`imcalib/drivers/jetson_uart.c`后才能使用，不能拿漏源的旧参数构建。操作和成功判据见
[双向串口探测](../deployment-contract.md#host-debug-plan)。后续电脑推理接管仍须独立的控制协议与验收。

### 2026-10-20：重新上电复测

接线调整后的2026-10-19检查及双方重新上电后的本次检查，ttyTHS1均在十秒内收到1000个有效镜像帧，
校验拒绝0；发出39个探测、回报0、超时39。上行约100Hz已实测，下行尚未通过。
本次只读SWD检查未暂停、复位或写入固件：UART8接收及DMA请求已使能，DMA1 Stream5已启用，
NDTR仍为256、接收缓冲区全0、合法包／CRC错误／重启计数均0，未见下行字节进入DMA。
这些证据不能区分Jetson TX未实际输出与中间接线故障，下一步检查Jetson TX→H7 RX路径。
证据见`artifacts/serial-20261020/jetson-uart8-power-cycle.json`及`h7-rx-runtime.json`；H7负责人核对仍待确认。

用户确认的H7线色为绿3.3V、黄RX、黑TX、红GND；本次三线连接为H7红↔Jetson黑（GND）、
H7黑→Jetson绿（TX→RX）、H7黄←Jetson黄（RX←TX），双方电源脚不连接。

### 2026-10-20：切换UART9通信测试（已烧录，上行通过，下行未通过）

- 用户明确要求改UART9，并确认暂停DR16遥控、只测通信、保留遥控离线保护。大机`MACHINE_VOFA_PORT=9`，TX为PD15、RX为PD14，1152000／8N1；Jetson仍用FUNC_UART／ttyTHS1。
- UART9初始化USER CODE区覆盖原DR16参数；RX DMA1 Stream4不变，补TX DMA1 Stream0及IRQ／反初始化。UART9 IRQ按所选端口交HAL或原DR16 IDLE路径；Core生成区字节未改，ioc保留基础DR16配置。后续CubeMX生成TX DMA前须处理用户区同名句柄／IRQ，避免重复定义。
- 大机不调用DR16_Init／DR16_Process，快照保持离线；DR16_Online、FAULT_RC、模式和最终出力门控保留，该测试配置不能遥控出力。小机仍为USART1通信、UART9接DR16。obs39／action6、模型、控制频率及物理标定不变。
- 新底板FUNC_UART回环曾通过：1152000波特率重开三次，随机二进制15包／3840字节均原样返回；恢复原因及重新上电后是否保持未确认。接回H7 UART8仍为999帧／10秒、39探测均超时。电压／接线观察不证明转接板RX到PE0连通，未以推测修改物理量。
- 现有正式协议检查通过，大小机各105来源ARM GCC编译／链接通过；编译后调用和向量符号确认大机huart9、DMA0强定义、DR16初始化／接收调用已排除，小机原路径保留，两机均保留遥控离线检查。证据：`artifacts/serial-uart9-20261020/check-final.json`、`compiled-port-routing.json`；此前联机记录在`artifacts/serial-20261020/`。
- 用户在本次明确确认电机动力已断开并要求现在烧录，助手执行写入／校验／复位，成功标记齐全。HEX SHA256为`0991be8ae0e576a73c27ecee46a7ed8aa979ab2f3c4cb277022796b4fca07d97`。随后只读SWD确认tick和镜像提交计数增加，UART9为8N1、BRR=0x77、RX DMA4／TX DMA0及PD14／PD15 AF11正确，DR16快照全0、FAULT_RC置位。
- UART9正式十秒联机收到1000帧、校验拒绝0；38个下行探测均超时。14／16／32／256字节探测长度对照均无回报，不据此认定芯片损坏。用户已确认黄线接回，双向未通过。证据见`artifacts/serial-uart9-20261020/flash.json`与`uart9-link-1.json`。板端控制时序、物理映射、行为及H7负责人核对仍待确认；未提交或推送。

用户确认黄线已接回后，三秒Jetson TX BREAK请求期间三次只读采样的PD14均为高，RX DMA NDTR仍256，合法包／CRC错误／重启仍0。该次未同步实测连接状态下的Jetson TX电压，不能区分转接板走线与输出冲突；下一步绕过上层转接板核对主控板PD14输入。证据见`artifacts/serial-uart9-20261020/rx-pin-during-tx-low.json`。

### 2026-10-20：DBG_UART二进制回环（当前协议不能直接迁移）

- 用户要求Jetson改用DBG_UART，随后确认H7端TX／RX断开、Jetson侧TX／RX短接。该口为ttyTCU0，当前物理速率115200；临时停止登录服务后以原始串口模式测试，结束已恢复服务，未改启动配置或H7固件。
- ASCII、CR、NUL和FF单项原样返回；LF由`0A`变为`0D 0A`，全字节序列256字节返回257字节。先前ASCII回环通过不能作为二进制协议透明传输的证据。结果见`artifacts/serial-20261020/jetson-dbg-binary-loopback.json`。
- 现有196字节／100Hz镜像需19600字节／秒，超过115200／8N1的11520字节／秒上限。继续迁移需选择编码与较低镜像频率；当前UART9固件仍为1152000，尚未完成DBG与H7联机，不据此声称双向通过。

### 2026-10-20：UART7＋FUNC_UART（已烧录，双向通过）

- 用户要求H7改为UART7、Jetson改回FUNC_UART，并在说明UART7被HI229占用后确认修改。本次暂时停用HI229，延续暂停DR16接收；IMU／遥控离线故障及最终出力门控保留，该配置仅用于通信测试。小机HI229／DR16路径不变。
- 大机选择`MACHINE_VOFA_PORT=7`，PE8/TX、PE7/RX，1152000／8N1；UART7初始化USER CODE区覆盖原HI229波特率，TX沿用DMA1 Stream1并改为NORMAL单帧发送，RX沿用Stream2环形DMA。UART7 IRQ交HAL，避免访问未初始化的HI229 IDLE缓存；main／imu任务不启动／解析HI229。
- Jetson使用FUNC_UART／`/dev/ttyTHS1`，接线前解除此前TX／RX短接、拔下H7 UART7原HI229。H7 TX→Jetson RX、H7 RX←Jetson TX、GND互连，双方电源脚不连；具体板上插座线序仍按信号核对。协议、196字节镜像、约100Hz发送、模型与物理标定不变，无需重训。
- 既有正式检查通过实际C打包／收包、模拟串口正常／损坏／旧固件路径；大小机各105来源完整ARM GCC编译／链接通过。编译后核对大机收包句柄为huart7、HI229／DR16初始化及解析均未调用、UART7交HAL，两机仍有HI229_Online／DR16_Online保护，小机原接收路径保持。Core生成区与本轮修改前字节一致。
- eIDE完整构建成功，仅既有BMI088 packed告警。待烧录HEX SHA256为`2d52aeaf2bd29fb1fbd57e4aca9e0c5bc8d6f0e24f7747a586ac600e59dcda32`；证据见`artifacts/serial-uart7-20261020/check.json`、`compiled-port-routing.json`及`firmware-build.json`。用户本次明确确认电机动力断开后已编程／校验／复位成功，随后确认UART7与FUNC_UART接线完成。正式十秒收到1000帧、校验拒绝0，92探测全部回报、超时0，平均往返9.98ms、最大10.13ms；只读镜像确认IMU／DR16离线及对应故障置位。证据见同目录`flash.json`、`uart7-func-link.json`和`offline-protection.json`。只读SWD连接尝试未返回寄存器值，不据此声称运行寄存器核对通过；本次双向探测通过，实际控制时序／行为及H7负责人核对仍待确认，之前UART8／9失败根因未确定；未提交或推送。

### 2026-10-20：改回UART8复测（已烧录，多轮结果不一致）

- 用户要求H7代码改回UART8后再试，Jetson仍用FUNC_UART／ttyTHS1。大机选8，PE1/TX、PE0/RX、1152000／8N1，沿用RX DMA1 Stream5与TX Stream7；原UART7条件分支恢复HI229接收，DR16延续暂停，离线保护和出力门控保留，小机路径不变。
- 协议、196字节镜像及约100Hz频率、模型和物理标定均未改，无需重训。现有正式协议检查、大小机各105来源编译／链接、编译后huart8／HI229恢复／DR16暂停选择及eIDE完整构建通过，仅既有BMI088 packed告警；Core生成区未改。
- 证据见`artifacts/serial-uart8-retest-20261020/check.json`、`compiled-port-routing.json`和`firmware-build.json`。待烧录HEX SHA256为`ab14d9007c537d2adebfe10b3d3d2346747b9454756e2df1f6c87a26e8719906`；本次确认电机动力断开后，首次1000kHz写入因外部复位／等待目标超时失败，未完成校验；用户明确要求重新烧录后，以200kHz下载同一HEX，写入／校验／复位标记齐全，UART8版烧录成功。证据见同目录`flash-attempt-1.json`与`flash.json`，失败根因未确定。用户随后对移线要求回复确认，UART8与FUNC_UART正式十秒复测收到1000帧、校验拒绝0，92探测全部回报、超时0，平均往返9.99ms、最大10.13ms，本次双向通过；证据见同目录`uart8-func-link.json`。之前UART8下行失败的根因仍未确定，HI229实际输入及策略／电机控制未由本次通信探测验证。H7负责人、真实时序和行为核对待确认；未提交或推送。

- 作者要求再次测试，使用同一UART8固件、FUNC_UART／ttyTHS1、1152000重新开口十秒收到1000帧、校验拒绝0，39探测回报0、超时39；本次下行未通过，不能据首次通过认定通信稳定。证据见`artifacts/serial-uart8-retest-20261020/uart8-func-link-2.json`。本次未改代码／串口配置或烧录；根因及期间物理接线变化未确认。

- 作者再次要求复测，第三次UART8／FUNC_UART十秒检查仍为1000有效帧、校验拒绝0、39探测回报0／超时39，与第二次一致；证据见`artifacts/serial-uart8-retest-20261020/uart8-func-link-3.json`。连续两轮下行失败，未改配置或烧录。

- 作者再次要求测试，第四次十秒收到1000有效帧、校验拒绝0，93探测全部回报、超时0，平均往返10.00ms、最大10.48ms，本轮双向通过。前两次下行失败仍保留，不能由本轮通过认定稳定或确定根因；本次未改配置／代码或烧录。证据见`artifacts/serial-uart8-retest-20261020/uart8-func-link-4.json`。

### 2026-10-20：UART7／UART8完整互换（已编译，未烧录）

- 用户要求互换两个串口功能。大机Jetson链路改为UART7 PE8/TX、PE7/RX、1152000／8N1，HI229改为UART8 PE1/TX、PE0/RX、921600／8N1。Jetson仍为FUNC_UART／ttyTHS1；HI229保持原供电、协议和波特率，数据线按TX交叉RX并共地。两板电源脚不互连。
- HI229_Init改选huart8与hdma_uart8_rx，UART8初始化USER CODE覆盖波特率，UART8 IRQ交HI229 IDLE缓存；UART7 TX DMA1 Stream1为NORMAL、RX Stream2交Jetson轮询，IRQ交HAL。main及imu任务均启用HI229初始化／解析，不沿用先前UART7测试版暂停IMU的行为。DR16仍暂停，IMU／遥控离线保护及电机门控保留；小机USART1通信、UART7 HI229和UART9 DR16路径不变。
- 镜像／探测协议、100Hz镜像尝试频率、模型、观测／动作、控制周期及物理标定不变，无需重训。现有正式协议检查、大小机各105来源编译／链接、编译后接收UART／DMA与IRQ归属及HI229调用检查、Core生成区字节核对、eIDE完整构建通过，仅既有BMI088 packed告警。
- 证据见`artifacts/serial-uart7-uart8-swap-20261020/check.json`、`compiled-port-routing.json`、`generated-regions.json`和`firmware-build.json`；待烧录HEX SHA256为`5eab71270f354c65c2a9c3588bdedfb96563a32b15e003fac4cb8a92afbdc9b1`。互换版未烧录或真实联机，HI229真实输入及H7负责人核对待确认，不沿用旧UART7／UART8测试结果作为本版验收；未提交或推送。

### 2026-10-20：撤销互换，Jetson改H7 UART10（已编译，未烧录）

- 用户要求恢复原配置，再将Jetson通信改到UART10。按H7端UART10实施：撤销尚未烧录的UART7／8互换，HI229恢复UART7 PE8/TX、PE7/RX、921600／8N1；UART8仅初始化、不承担业务收发。大机Jetson改到USART10（板上UART10）PE3/TX、PE2/RX、1152000／8N1，Jetson仍用FUNC_UART／ttyTHS1。DR16继续暂停，IMU／遥控离线保护及出力门控保留；小机路径不变。
- 根据[ST STM32H723数据手册DS13313表8](https://www.st.com/resource/en/datasheet/stm32h723ve.pdf)，PE2/RX为AF4、PE3/TX为AF11，两脚分别配置。Core用户区新增硬件初始化和启动调用，使用空闲DMA2 Stream0／请求118作环形RX、Stream1／请求119作NORMAL TX，并补DMA2及USART10中断；保留现有DMA1资源。生成区字节未变，ioc未加入USART10，后续CubeMX启用前须处理用户区同名定义。
- 不改协议、196字节镜像、约100Hz发送频率、模型、观测／动作、控制节拍或物理标定，无需重训。正式协议检查、大小机各105来源完整编译／链接、编译后接收句柄／HI229与DR16调用／真实向量表绑定、Core生成区字节比较和eIDE完整构建通过，仅既有BMI088 packed告警。
- 证据见`artifacts/serial-uart10-20261020/check.json`、`compiled-port-routing.json`、`generated-regions.json`和`firmware-build.json`。待烧录HEX SHA256为`1fba3849f2544a3fc159e22079c04f5ae368c6532e6771dd7a702e82ec135b8d`；本次仅代码与编译，延续用户先不烧录的选择，UART10硬件接收／波形／时序及H7负责人核对待确认，不沿用此前UART7／8结果作为本版通过依据；未提交或推送。

### 2026-10-20：当前UART10配置恢复遥控（已烧录，首次串口双向通过）

- 用户明确要求恢复遥控。main及comm在当前选10时重新调用DR16_Init／DR16_Process；UART9为100000／8E1、PD14/RX、DMA1 Stream4，原IDLE处理保留。DR16_Online／FAULT_RC、拨杆许可和最终出力保护不变。Jetson仍UART10、HI229仍UART7；旧端口测试条件及小机路径不变。
- 大小机各105来源完整编译／链接、编译后两机DR16初始化／解析／离线检查调用、UART9 RX／DMA／IRQ归属、Core生成区字节核对及eIDE完整构建通过，仅既有BMI088 packed告警。本轮没有修改协议、观测／动作、模型、节拍或物理标定，无需重训；未重跑不相关协议／模型数值检查。
- 证据见`artifacts/serial-uart10-rc-20261020/check.json`、`compiled-port-routing.json`和`firmware-build.json`。待烧录HEX SHA256为`577d184012abcc29022feeb31f57ecd92d1b60b1cb258367018bd1caafb5edab`；用户随后要求测试UART10，并确认本次电机动力断开及UART10／FUNC_UART接线；前两次下载器CMD_INFO阶段失败均未写入，重置该PowerDebugger TX USB连接后以200kHz完成写入／校验／复位，同目录`flash.json`记录三次尝试。正式十秒收到999帧、校验拒绝0，93探测全部回报、超时0，平均往返9.99ms、最大10.13ms，首次双向通过；证据见`uart10-func-link.json`。本次仅验证串口，真实遥控接收／传感器／控制行为及H7负责人核对待确认，不据此认定长期稳定；未提交或推送。

### 2026-10-20：复查修复DMA错误恢复与ACK期限（未烧录）

- RX环形DMA只禁用HT，保留TC，使UART错误后的HAL异步终止完成并解锁，再由comm重启；主机先处理250ms到期，再匹配回报，迟到ACK不再算成功。线上14／196字节格式、端口分配与控制接口不变。
- 既有正式入口增加模拟DMA状态／锁／TC恢复和249／250／300ms回报边界；实际C解析、伪终端正常／损坏／旧固件与大小机各105来源完整编译链接通过，见`artifacts/uart10-review-fix-20261020/check.json`。DMA状态为测试桩，未执行真实HAL中断或板上错误注入。
- 当前修复只在本机源码及构建产物中，未烧录／部署Jetson，前述十秒联机结果对应修复前固件；真实错误恢复与稳定性、H7负责人核对仍待确认。

### 2026-10-20：修复版烧录与实际接收错误恢复（通过）

- 用户要求完成DMA错误恢复板测。工程外eIDE全量编译104个C来源及1个汇编通过，以200kHz写入／校验／复位修复版；HEX SHA256为`2a3d5a3530add09e40dc030f16dfc0bedf6564f9d4ccda97120474ade1c0bf0a`，Jetson常用脚本同步修复版并备份旧版。协议／接线／控制接口不变。
- Jetson TX BREAK请求20ms连续注入三次，H7内存中的rx_restarts为0→1→2→3；每次错误后3秒均收到300镜像帧、探测26发／26回、超时0，DMA恢复BUSY且TC开启。实际错误恢复通过，见`artifacts/uart10-dma-recovery-20261020/error-recovery.json`、`flash.json`及`firmware-build.json`。调试连接已关闭。
- 本次只验收BREAK诱发错误后的接收恢复；未验证任意噪声／溢出、长期稳定性、遥控／IMU和策略／电机行为，H7负责人核对仍待确认。
