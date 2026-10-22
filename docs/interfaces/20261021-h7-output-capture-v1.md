# H7完整输出记录v1（大机已烧录，断动力30秒USB采集通过）

2026-10-22最终吞吐修复已烧录并完成断动力实采：48条/批、等价快速CRC和USB忙重试，预热后连续约55秒两队列无新增丢弃/缺号/CRC拒绝；开头输出2/时序6条丢弃和截断仍保留。[最终证据](../../artifacts/h7-capture-throughput-fix-20261022/report.md)。下段为首版候选阶段。

2026-10-22吞吐修复候选：实测约78Hz镜像下，24条/包不足以导出禁用档约2000条/秒；已改48条/包并更新解码器，128项队列/记录布局/控制保持，软件与大小机构建通过，未烧录。[本轮状态](20261022-h7-latency-capture-v1.md)。下文首版24条/包及其验证为历史状态。

最新固定300ms台架固件已于17:18烧录大机，写入校验/复位及10秒断动力零输出采集通过；正在另建会话验证有动力300ms响应，详情见文末。

2026-10-21，Codex；H7负责人复查待确认。按作者“添加”要求，准备完整输入记录，首次交付时未烧录；后续作者另行授权烧录和测试，实际状态见文末。仅在独立worktree `fixed-h7-real2sim` 实现；HEAD `d7c857f10c24751f8bb2e525438bf10bad499312`，未提交、未推送。

## 兼容性与修改范围

旧接口是 [USB镜像v2/644字节](20261020-h7-ros-usb-mirror.md#只读对照信号v2)；新传输为 `h7-mirror-v2+output-capture-v1`：保留原v2镜像前缀，**同一次**USB提交追加一个 `H7OUT001` 二进制批次。后台仍只调用原 `Vofa_Debug_Send` → `Vofa_Send`，不增加RTOS任务、第二发送分支或控制收包。

当前以作者已恢复的原控制源码为基线：轮0.1速度比例环、±3.9Nm限幅，原LQR/台架/门控不变，控制版本0。开始时worktree内旧补偿快照先存入 `artifacts/h7-output-capture-20261021/before-rebase/`，再核对复制原工作区的原控制版本；来源及哈希见同目录 `baseline.json`，原工作区没有回写。

不修改观测39/动作6、模型、策略、控制增益、时序配置、物理极性/零点/减速比/力矩换算、XML或质量惯量。不需要重训/重导出。记录功能会增加少量执行开销和USB负载，**尚未在板端证明不影响2ms节拍**；不能把软件通过写成调度/行为通过。大机已安装新增记录固件；断动力30秒USB采集已通过，RL实际节拍仍待验证，见文末。

## 两类记录及调用位置

- **CONTROL（kind=1）**：`task_actuation.c: Mirror_Output_Publish`保存每个实际分发拍，包含原门控后四腿两轮Nm请求、该拍真实使用的公共轮速/原接收年龄、输出序号/标志、故障、控制模式、台架phase/reason。RL通常500Hz；台架/LQR的实际周期按记录时刻检查，不把所有记录都叫500Hz。
- **CAN（kind=2）**：`dji.c: Dji_Send_Current`在原 `Can_Bus_Transmit` 返回后，读取**原本送入该调用的8字节局部报文**，记录四个真实电流码槽、报文ID、HAL返回值及调用开始/结束时间。覆盖执行任务、通信任务停机、台架发送失败后的再次清零；不重新计算电流，不读取可能已被别的任务改写的旧 `wheel_current`。

请求力矩仍是固件公共方向/名义Nm，反馈仍是既有驱动已处理标定与减速比的轮轴rad/s；CAN电流码是电气报文槽位，尚未转换为模型Nm。离线使用时只按对应版本映射一次。两者都不是独立实测轴端力矩。

CAN的HAL返回0只表示入队成功，不证明总线送达或电机执行。失败调用的电流码是**尝试值**。begin/end是调用时间区间，不是机械延迟；多个调用区间可能因任务抢占而重叠，记录入队顺序也不能强行解释成物理发送顺序。CONTROL的时间开始在对应分发前，结束在下发及镜像发布后；因此不能要求所有不同kind的begin时间按队列顺序递增。

## 缓存和传输

`imcalib/bsp/output_capture.c/.h`维护128项×88字节队列（11264字节），执行/通信任务只在很短的IRQ保护区复制固定大小记录；CRC和批量复制在后台。队满只增加丢弃数，**不阻塞、不等待USB、不修改电机输出**。序号对每次入队尝试递增，因此溢出会在后续数据里留下缺口。临界区保存并恢复原PRIMASK，支持两个生产任务。

每次100Hz镜像尝试最多附24项记录。USB未就绪、忙或提交失败时不释放队列；提交成功后才出队，异步传输完成前原静态缓冲不得覆盖。满批次总长度为644+40+24×88+2=2798字节，名义100Hz约280kB/s，是所需带宽，**不是已测吞吐**。若后台/USB长期不足，控制继续运行，数据明确标为不完整。

原v2解析器仍能读取镜像前缀；旧VOFA客户端是否接受额外二进制内容未实测。现有real2sim接收线程继续把全部原始字节保存到一个会话，姿态显示仍按v2；新增解码器离线读取同一原始会话，不另开串口。

## 线上布局

全部小端。头40字节：magic `H7OUT001` 8字节；version/record_size/count/reserved 各uint16（1/88/0～24/0）；packet_sequence/produced/dropped/queued_before/high_water/reserved 各uint32（最后为0）。尾部uint16 CRC16/CCITT-FALSE，覆盖头和全部记录。

每项88字节，与头文件 `output_capture_record_t` 一致：

| 字节偏移 | 类型 | 字段/解释 |
|---:|---|---|
| 0/4 | uint32 | 全局尝试序号 / kind=1控制或2CAN |
| 8/16 | uint64 | begin_ns / end_ns，同一板端单调时基 |
| 24/28 | uint32 | id：控制拍序号或CAN ID / CONTROL故障码 |
| 32 | float32×6 | CONTROL门控后请求，四腿两轮固件顺序，Nm |
| 56 | float32×2 | CONTROL当拍公共左右轮轴速度，rad/s |
| 64 | uint32×2 | CONTROL原接收年龄μs，未知UINT32_MAX |
| 72 | int16×4 | CAN实际尝试的四槽电流码，保持发送极性/槽位 |
| 80 | uint16 | CONTROL原镜像输出flags；CAN为HAL返回值 |
| 82～87 | uint8×6 | CONTROL策略/bench phase/reason、机型、轮online mask、控制版本0 |

不适用于该kind的字段按零初始化；CSV会输出空白，不能将CONTROL的零电流槽当实际发了零电流，也不能将CAN的零请求字段当六路零矩。全局序号自然回绕可读；复位/重连、CRC失败、缺口与截断必须报告。

## 录制、版本与离线命令

新增 `--firmware-manifest` 仅把预期构建JSON及其SHA256归档到会话；**不是读回板端固件或启动证明**。本次检查产物及后续待确认固件清单在 `artifacts/h7-output-capture-20261021/`。XML、机器配置和接收程序哈希仍由原录制器保存。

未来完成独立烧录/接线/动力安排后，沿用一个real2sim进程录制（本次没有执行）：

```bash
mjlab_training/.venv/bin/python sim2sim/real2sim_huananhu.py \
  --port /dev/serial/by-id/<实际H7设备> \
  --machine-config deployment/H7_RL_ros/imcalib/config/machine_config.c \
  --firmware-manifest artifacts/h7-output-capture-20261021/firmware-machine-0.json \
  --record artifacts/<新的完整会话名> --seconds 30
```

离线解码不会打开串口：

```bash
mjlab_training/.venv/bin/python sim2sim/h7_output_capture.py \
  --session artifacts/<完整会话名> --out artifacts/<新的输出分析目录>
```

输出 `outputs.csv`、`packets.jsonl`、`report.json`。查看 `complete_observed_span`、`event_sequence_gaps`、`queue_drop_increase`、`partial_packets_at_connection_end`、`failed_CAN_attempts` 和控制周期统计。缺口/丢弃/截断/损坏时返回2；旧固件没有H7OUT001时明确拒绝。`complete_observed_span`只评价已导出的观察区间，不覆盖录制开始前/结束后，不能等同物理输入已测全或无故障。

首次接通USB时可能导出旧缓存；累计dropped可能来自录制前，要同时检查本次增量、记录缺口和队列积压，不能只看最后总数。首尾未导出条数单独保存；不得让采集跨缺口后仍声称整个试验完整。旧动力学拟合入口目前仍按100Hz镜像请求处理，**不会自动采用新旁带**；新数据取得后必须显式接入新输出表，不能称旧分析已完成500Hz输入验证。

## 首次软件交付验证与同步状态（历史：当时未烧录）

- `final/report.json`：实际C驱动652组命令/停机/返回值与插桩前基线逐字节一致；队列满、重试、IRQ状态、索引/序号回绕、CRC、17字节分包、USB缓冲保护通过；虚拟500/1000Hz调度下6000项记录、40ms后台停顿不丢失。模拟调度不是H7性能测试。
- C生成二进制→原始会话→CSV共655条；模拟CRC错误、截断、溢出及旧固件缺少批次均被识别。合成数据中故意包含489个非HAL_OK结果，不是实机故障。
- `regression-usb-final.json`：现有USB打包/录制/回放/分析、策略配对、原轮矩公式、门控及台架保护通过；`regression-uart.json`：默认主工程兼容检查通过。
- 大小机各107来源GCC编译/链接通过，仅既有BMI088 packed告警；AXI/USB/DTCM均由链接器检查范围。大机text/data/bss=385016/2744/75456，小机381224/2744/75072字节。Keil项目登记已更新，未实际运行Keil构建。
- 未验证：板端实际控制周期/截止时间、最坏记录开销、USB长期吞吐、断连恢复期间真实积压、CAN电气送达与物理响应；未烧录、未接电机、未训练。

训练侧解码器与H7源码在本worktree准备，大机已烧录；Jetson新增解码器尚未同步，本次用SSH只读字节流传回本worktree解码，不改Jetson源码；H7负责人对接口/真实时序/采集结果核对**待确认**。实际采集前必须配套安装解码器及含此记录功能的固件，并验证控制节拍；不更换策略或物理标定。

## 2026-10-21 16:28 大机烧录与断动力检查

作者明确要求“烧录”“并测试”，随后确认“本次大机电机动力已断开，H7 和 RX 保持供电”。独立worktree的大机HEX经PowerDebugger/OpenOCD、200kHz SWD写入，日志同时出现Programming Finished、Verified OK、Resetting Target，退出0。HEX SHA256 `c7c54a36ccd7a8e42f0166ddf43770403157f9d84f5081bdc03d60d3a3f2f95b`；来源93项及受保护10项哈希与构建清单一致。没有主机电机控制命令、动力测试、训练、提交或推送。

复位后通过不暂停/不复位/不写内存的SWD读取检查：系统tick递增，记录生产持续推进，两次当前fault均0。SRAM保留128项记录（CONTROL/CAN各64），六路请求与四槽电流码均零；启动样本曾含IMU/遥控未就绪故障1/3，不能当作全程无故障。两个连续序号小段内控制周期范围0.996755～1.003593ms，仅是未投入RL的启动小样本，不证明500Hz RL截止时间。两个小段之间缺798项，400ms记录时间跳变跨过丢失数据，不解释为控制停顿。

USB静态发送缓冲包含1个CRC正确的H7OUT001批次，但提交计数停在1、队列128项满且丢弃持续增加；**持续USB接收未通过/待定位**。可能缺少主机读取，也可能链路卡住，未确定原因。电脑只见下载器CDC，H7数据USB在Jetson，需要Jetson端继续只读收包及完整性检查。不能用SRAM包校验代替USB端到端通过。证据及可重复命令：`artifacts/h7-output-capture-20261021/flash-20261021-162829/flash-result.json`、`openocd.log`、`runtime-memory-read-command.json`、`runtime-dump-command.json`。预期构建清单保留原始未烧录快照，以本次flash-result作为实际写板证据；H7负责人复查待确认。

## Jetson连接持久化与30秒实际USB验收

作者补充“imca，并写入项目中”。核实imca是用户名，主机返回localhost.localdomain，原有有线地址可用、SSH密钥认证通过。实际SSH目标、远端项目目录、STM32 USB by-id路径已写入本worktree的Git忽略文件 `.env.local`（JETSON_SSH_TARGET/JETSON_PROJECT_DIR/H7_USB_PORT）；公共 `.env.example` 保留占位符，[COMMANDS.md](../../COMMANDS.md#jetson只读usb采集连接)记录读取方式。原工作区配置和Jetson源码未修改。

故障复核：Jetson当时没有接收进程。打开只读USB接收后，未重新烧录/复位即恢复持续传输；停读时USB提交停住、队列满并丢记录。支持本次停顿由未持续读取引起，不证明所有断连故障都能自行恢复。10秒预检及首次30秒录制保留了开头积压和缺口，两段完整性均为false，不能计入无丢失通过样本。

随后同一USB连接先录5秒warmup；只有完整H7OUT001包已收完且队列排空时，才切换到新的测量会话，全程原始字节仍保存。正式观察约30秒：

| 指标 | 实测 |
|---|---:|
| H7OUT001批次 / 记录 | 2999 / 59980 |
| CONTROL / CAN记录 | 29990 / 29990 |
| CRC或格式拒绝 / 记录缺口 / 批次缺口 | 0 / 0 / 0 |
| 新增队列丢弃 / 重连重启 / 结尾截包 | 0 / 0 / 0 |
| 完整观察区间 | true |
| 镜像v2帧 / 镜像序号缺口 | 2999 / 0 |
| CONTROL周期min / median / max | 990.036 / 999.945 / 1010.524微秒 |
| 输出分发记录区间median / max | 12.517 / 24.598微秒 |
| 实际接收字节 / 平均吞吐 | 7335555 / 约244523字节每秒 |

fault全0、机器0/原控制版本0、六路请求和四槽电流码全0；镜像状态12，无电机使能或RL投入。旧镜像解析器在二进制旁带内拒绝3个非镜像尾候选（last_error=not mirror v1/v2），实际镜像序号连续；不将候选拒绝数称作3个CRC坏包。累计历史丢弃1063704保持不增，来自本观察前的停读，不隐去累计值。完整性只覆盖保存的观察区间，不覆盖前后未录时段。

这是电机动力断开、未投入RL时的1000Hz空输出采集检验。不能证明RL 500Hz最坏截止时间、长期吞吐、轴端实际力矩或整机俯仰预测通过。约30秒全零请求也不能取代下一步有动力的独立轮矩脉冲和RL实测。H7负责人复查仍待确认，不执行训练或环境同步。

本次直接将Jetson只读数据通过SSH传入本worktree的原会话记录器；事件时间是Jetson读返回的单调时钟，SSH到达延迟不用于推断电机延迟。Jetson端只用已有虚拟环境和pyserial读取，没有write、DTR/RTS复位、控制指令或远端文件产物。接收结束后USB可能再次积压；后续采集必须持续读取并先确认排空/序号连续，再开始实际试验。

所有原始流、warmup、负结果、哈希及统计在 `artifacts/h7-output-capture-20261021/jetson-usb/`；汇总 `result.json`，正式会话 `steady-30s/`，标准解码 `steady-analysis/report.json`。任务专用可重复命令（新目录名；运行前保持本次动力安排，不能与其他接收器同时占用USB）：

```bash
source ./.env.local
CAPTURE_SESSION="artifacts/h7-output-capture-20261021/jetson-usb/usb-$(date +%Y%m%d-%H%M%S)"
mjlab_training/.venv/bin/python   artifacts/h7-output-capture-20261021/jetson-usb/record_jetson_usb.py   --warmup 5 --seconds 30 --session "$CAPTURE_SESSION"
mjlab_training/.venv/bin/python sim2sim/h7_output_capture.py   --session "$CAPTURE_SESSION" --out "${CAPTURE_SESSION}-analysis"
```

成功判据为标准解码 `complete_observed_span=true`，缺口/新增丢弃/CRC拒绝/截包均0；未达到时保留原数据并停止将该段作为完整输入。任务专用SSH采集脚本是本地实验产物，不是已部署到Jetson的正式接收入口。

## 2026-10-21 固定300ms台架候选（源码/构建通过，未烧录）

作者在采得一次新悬空脉冲后明确选择“取消全部提前停止，只保留300ms结束”。本次仅在独立worktree改USB工程的台架RUN：成功准备并手动触发后，锁存轮子/正负3Nm指令；运行中忽略轮速、遥控离档/离线、反馈、故障、总输出开关、发送失败及执行间隔变化，按原HAL毫秒时基到300ms结束。执行策略在该窗口固定为轮台架，通信失能不抢先发轮零矩；四髋仍失能/零请求。触发前的静止、有效反馈、故障/遥控与四髋失能检查保留；上电不自动发脉冲，持续上/上不连发，每次须重新准备。

改变的是台架刺激过程，正常RL/LQR、轮速比例环、幅值、驱动极性/零点/换算和模型不改。`controller_mode=0`仍表示原RL轮P算法，不能据此把新台架固件当作旧固件；另用固件SHA和 `bench_profile=fixed300_no_early_stop_v1`区分。旧保护版实录继续保留，不改写其元数据。

实际C状态机/最终分发/策略选择/通信停机仲裁检查通过，覆盖300ms边界、uint32回绕、原准备互锁、RUN中超速/NaN反馈/遥控离线及离档/反馈故障/发送失败/执行间隔变化、到时清零和四髋零请求。现有USB完整回归、输出采集652组驱动/缓存/CRC检查通过；大小机各107来源GCC编译链接成功，仅既有BMI088 packed告警。回归中两个已不再使用的测试替身产生unused提示，移除后单独重跑实际输出发布检查通过。未使用硬件故障注入，Keil未构建。

候选与报告在 `artifacts/h7-bench-fixed300-20261021/`：`check/report.json`、`native-bench.json`、`output-producer-final.json`、`capture-check/report.json`、`firmware-machine-0.json`和`candidate-source/`。大机HEX SHA256 `c6bef66b74e45318bbaaef6566ff9c2f34c5a09d1365985e32e930876d3334e8`；源文件及ELF/HEX/BIN哈希在预期构建清单。未烧录此候选；板上仍是此前已验证USB的提前撤销版。当前作者已确认电机动力上电，下一次写板必须另获本次断动力确认。

300ms是正常固件调度下的请求窗口，不是CAN断路、MCU停调度或断电时的物理力矩保证；名义3Nm不是独立测得轴矩。新固件尚未实测完整300ms脉冲、最高轮速、实际调度或机械行为。H7负责人复查待确认，不训练、不改正式XML、不提交/推送。

本次新采集保存在 `artifacts/fixed-h7-new-data-20261021/`：预检10秒完整、全零请求；`first-pulse`含完整299960项记录，记录/批次缺口及新增丢弃0，左轮+3Nm请求58.998869ms后归零，归零轮速30.50077rad/s，峰值33.56041rad/s，四髋全零、fault0。reason=GATED（1），源码在许可函数中先汇总速度/反馈等条件，再由状态机写此原因；本次越过30rad/s与此路径一致，但原因码不独立证明唯一触发项。初次拟合集 `fit-session`未请求新脉冲，改为暂停/静态证据，不用于拟合；尚无完整新拟合集/验证集。用户要求变更后停止后续触发，`split.json`不把未完成会话冒充训练/验证数据。

后续烧录和零输出复核通过后，采集需显式使用新预期清单，不能沿用旧清单：

```bash
source ./.env.local
mjlab_training/.venv/bin/python artifacts/fixed-h7-new-data-20261021/record_bench.py --warmup 5 --seconds 120 --session artifacts/fixed-h7-new-data-20261021/new-profile-session --firmware-manifest artifacts/h7-bench-fixed300-20261021/firmware-machine-0.json
```

该命令只读，不自动触发电机；必须新会话名、实际板端版本与动力/台架安排相符，并先验证完整性再做拟合。旧版“左下/右下立即提前取消RUN”的操作不再适用于这个新候选，触发后要等300ms窗口结束。

## 2026-10-21 17:18 固定300ms固件实际烧录与断动力复核

作者再次明确确认本次电机动力断开、H7和RX供电。新profile大机HEX由PowerDebugger/OpenOCD 200kHz SWD写入；Programming Finished、Verified OK、Resetting Target齐备，退出0，HEX SHA256 `c6bef66b74e45318bbaaef6566ff9c2f34c5a09d1365985e32e930876d3334e8`。实际证据 `artifacts/h7-bench-fixed300-20261021/flash-20261021-171749/`；此前编译段的“未烧录”是当时状态。预期构建清单及交付快照保留烧录前内容，不能用其中flashed=false代替实际烧录结论；当前状态另存current-status.json。

复位后Jetson USB重新枚举；先持续读取5秒排空，再记录10秒：999批次/19980记录，完整观察区间true，CRC拒绝/记录缺口/新增丢弃/截包/HAL失败均0；CONTROL fault0、bench phase0，六路请求和四槽电流码全零，镜像999帧且序号无缺口、无电机使能或RL投入。完整数据和报告为postflash-10s/、postflash-analysis/、postflash-summary.json。证明程序启动及断动力记录链路正常，不证明300ms带动力响应或500Hz RL截止时间。

按作者原采集请求继续使用独立数据目录artifacts/fixed-h7-fixed300-data-20261021/，与旧提前撤销版分开；先做1次完整输入采集检查，再安排按完整会话划分的拟合/验证。用户接通动力及手动触发状态按本次对话另行记录，主机不发电机控制命令。未训练、提交、推送或修改物理标定/正式XML；H7负责人复查待确认。


## 2026-10-21 固定300ms有动力完整会话验收

按用户确认的固定悬空条件完成小样本1次及独立拟合/验证各4次，使用新profile预期清单，不混入旧59ms样本。正式会话分别120秒、12000批次/240000记录，CRC/记录及批次缺口/新增队列丢弃/截包/HAL失败0；每次300条非零CONTROL与CAN提交、约300ms、到时DONE/reason0，四髋全0、fault0。只证明记录窗口内完整指令及悬空响应，不证明轴端实际力矩、RL500Hz截止时间或自由整机行为。

本地record_bench.py已修复任意字节收尾会截断批次的问题：在计划时间后继续到完整CRC有效批次边界，warmup/正式切换保持同一串口，原始数据全保留。最初截尾253字节的fit-session整份排除，validation-session零脉冲尝试排除；旧失败数据和完整性false不改写。新完整会话哈希及会话划分锁定于artifacts/fixed-h7-fixed300-data-20261021/split.json，实际触发间隔并非全部10秒、温度未测，如实报告。

独立等预算轮端拟合及旧RL俯仰检查已完成，整机校准未通过；正式XML/机器标定/策略保持，不同步环境或训练。完整参数、搜索依据、左右/方向/速度/加减速误差、曲线、数值核验及复现见[完整输入校准报告](../diagnostics/20261021-fixed300-complete-input-calibration.md)。结束用户报告左下/右下，已告知断动力，未收到最终断动力确认；H7负责人复查仍待确认。
