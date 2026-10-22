# H7 RL初始角度分支修复与烧录（2026-10-21）

## 最新状态：已烧录、校验并复位

2026-10-22断动力只读补验通过：两次独立录制均捕获禁用→RL，四路偏移分别为±3圈/±2圈并正确消除，近似同姿态的策略角最大差0.018696rad，小于预设0.03rad；请求全0。中间USB断连未拼接，失败段保留，恢复后另录；带动力PD、左腿失联及站立仍未验证。[数据、曲线与边界](../diagnostics/20261022-h7-angle-reentry-validation.md)。以下为此前烧录及离线阶段历史。

2026-10-21本次用户明确要求烧录并确认电机动力已断开、H7/RX保持供电。当前源码重新完整编译退出0，HEX本次生成；与已验证候选仅CubeAI生成网络的编译时间字符串3字节不同，其余全部609496字节Flash地址/内容一致。随后PowerDebugger/OpenOCD SWD1000kHz实际写入，Programming Finished、Verified OK、Resetting Target齐备，退出0；到此停止。当前大机HEX SHA256 `57c2a6813b071a6d1a249e29e7ec86463f5f4904531e8aa85a0a55ac90ba726e`，模型仍6848，初始角度分支锁存修复已写入。尚未做板端只读角度核对或动力响应试验，不声称左腿失联/饱和已经消除。H7负责人待确认。证据：`artifacts/h7-rl-angle-branch-fix-20261021/flash-01/flash-result.json`。

以下“未烧录”为本次写板前的离线候选阶段记录，保留验证边界。最新状态以本节及current-status.json为准。

源码修复和离线/完整构建验证已完成；未烧录，实机仍运行之前的6848固件。本文不是实机恢复正常的验收。用户要求修复前30秒发现的异常多圈角，并验证；修改全部位于原独立worktree，未提交/推送。

## 改法与边界

RL_Motor_Map新增由执行任务持有的rl_motor_map_state_t。进入RL、首次源反馈有效时，先用既有motor_sign/motor_zero换算，再按原驱动周期2*dm_pos_max选择最靠近q_reference的等价初始表示，锁存四个整周期偏移。运行期间始终减去同一偏移，既不逐帧取模，也不把当前位置重新置零；因此保留后续连续角、速度和跨编码器边界运动。退出RL才清initialized，反馈故障及其恢复过程中保持偏移，不因有效性抖动偷偷更换角度坐标。

观测和PD继续共用control_frame.rl_motor.pos。只修观测而让PD仍读-18rad不能解决满幅请求。PD角误差不环绕，原sign/zero/量程/力矩换算/增益/限幅/网络权重/控制节拍不变；原始连续反馈及USB通道10～13继续保留原计圈量，修复后的RL坐标可从实际obs与PD pos看到。LQR、bench300ms撤销逻辑、公共驱动及所有故障门控未改变。故障恢复后自动再次投入的现有行为也未修改；需要另行实机验证左腿失联原因，不能只靠角度变换宣称已消除反复冲击。

模型事实更正：复核XML后，前驱动关节range约q_reference±100rad，后驱动关节limited=false，并非四驱动关节都有有限硬限位。此前分析把它们称为有限行程关节不准确。q_reference±3.14是训练PD的目标限制，不是机械真实限位。因此本修复仅在初始化时选分支，运行中允许连续角跨过该区间，不用逐拍wrap掩盖真实运动。

修改位置：rl_observation.c/.h的映射和状态；task_actuation.c新增执行任务独占状态、退出RL清初始化、映射调用传入原驱动周期；rl_types.h和machine_config.h仅同步角度定义注释。相对本任务开始的准确小补丁在artifacts/h7-rl-angle-branch-fix-20261021/candidate.patch，避免把该工作区此前所有改动误算成本任务。

## 已完成验证

- scripts/agent/check_h7_rl_angle_branch.py编译实际rl_observation.c与rl_torque.c，O0、Wall/Wextra/Werror，52组功能样本通过：已有表示、多圈±1/±3、正负运动、跨编码器及参考分支切口、无效反馈、NaN/Inf和退出后重新初始化。多圈表示误差最大6.55651e-7rad；速度符号不变。
- 实际C观测构造与PD共享同一位置，观测位置槽等于新位置减既有nominal，PD位置逐值相同；独立零action样本的PD输出符合原增益/符号。没有用新策略生成另一套输入冒充原录制的预测或实机改善。
- 最新实录前30秒仅复算角度映射：796个有效RL样本通过，265个源反馈失效样本仍无效。首组模型角约[0.413481,1.982977,-0.421247,-1.980004]rad；四腿范围分别约[0.369,0.608]、[1.938,2.122]、[-1.260,-0.405]、[-2.819,-1.961]rad。这只是同一已录反馈的表示修复，不是新控制后的实机轨迹。
- 当前姿态库normal/positive/negative左右两腿原生初始角全部距离q_reference小于π，因此正常初始表示保持；范围快照在reset-pose-representation.json。此检查不证明实机姿态/零点已经正确。
- 原eIDE/unify_builder配置独立完整重编大小机，两者退出0、build successfully、ELF/HEX/BIN均生成，当前O0不变。只见原BMI088 packed告警。首次调用因新输出目录尚未创建而工具在.lock处退出134；保留日志，建立目录后成功，不是源码编译错误。
- machine_config.c、dm.c、dji.c、rl_torque.c、6848模型/ONNX、XML哈希保持，原始录制不改。来源、产物哈希及命令见同目录result.json和commands.txt。

大机候选HEX：artifacts/h7-rl-angle-branch-fix-20261021/build-machine-0/CtrBoard-H7_ALL.hex，SHA256 5b7175773ead3fc194e3f3c6bb64e1a2a90ea5642c31ce5253064f3b43e351ca。

小机仅做兼容编译，不授权替换小机模型或上板测试。

## 未验证与下一步

未写入硬件、未发送电机指令、未重启H7、未做带动力实验。缺少原始单圈角/计圈整数/DM错误码与电压数据，因此不能宣称计圈产生原因、电机保护或左腿断反馈原因已经修好。不能宣称轮速、俯仰、位移或站立改善。

下一步上板必须先保持电机动力断开，烧录后只读核对：同一固定姿态下obs/PD的角度一致、没有整圈偏移，切换RL不制造跳变；确认后再安排独立动力测试。H7部署负责人核对状态：待确认。
