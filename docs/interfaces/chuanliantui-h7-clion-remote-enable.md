# 接口记录：chuanliantui H7_clion DR16 使能互锁

| 项目 | 内容 |
| --- | --- |
| 日期、作者、复查 | 2026-10-01，Codex；独立只读审计提出重复 ARM／DMA 内存／过期帧风险，最终板测见部署证据 |
| 机器人／硬件 | chuanliantui，大轮腿、STM32H723、DR16 UART9 |
| 旧接口 | H7_clion USB v1，STATUS/ARM/SET_TORQUE/STOP；无遥控门禁 |
| 新接口标识 | h7-clion-remote-gate-2602；USB外层v1保持 |
| 训练仓版本 | 分支26_wheelleg，HEAD bb56dd40964dc569b4e2a406983b03fa19d943a1，有既存未提交改动；本次不改训练实现 |
| 部署仓版本 | `.env.local` 的 H7_REPO_PATH，当前 H7_clion 无Git元数据；以固件及源码快照SHA标识 |
| 权威实现 | 部署端 App/lower_remote.c、lower_remote_uart.c、lower_safety.c、lower_app.c；docs/remote-enable.md、docs/protocol.md |

## 定义与兼容性

按用户确认沿用旧H7的USB调试组合：左拨杆上、右拨杆下才允许使能。上电、STOP、掉线或故障后，必须先见左下再切允许组合。左下经中档到上档的机械过渡有效；已经使能后离开组合即停机。

使能还要求板载USB连接以及六台电机/CAN就绪。使能后为零力矩待机，首次合法SET_TORQUE开始输出；之后100ms超时停机。遥控50ms无有效帧、坏帧或UART错误亦停机。USB ARM不能绕过遥控许可；重复ARM不刷新力矩看门狗。

新增REMOTE_STATUS命令0x05、响应0x82（64字节，layout2602），包含原始档位、接收时刻、有效/坏帧计数、故障代数、许可、ARM及DM反馈使能掩码。旧132字节STATUS和SET_TORQUE的字节布局不变；STATUS新增bit8–10和停止原因5/6。旧主机可查询状态，但无法再绕开遥控直接ARM。

25维观测、5帧历史的训练语义、关节映射、IMU轴向、动作尺度、闭链几何、控制增益及模型权重均未修改。不需为这一互锁改动重训或重导出。`LOWER_OBS_MAPPING_VERIFIED`仍为0；真实策略动作和历史时序诊断尚待后续接入，不能把本次遥控检查当作策略链验收。

## 版本与验证

部署端证据目录：`captures/20261001-remote-enable-firmware/`，含改前源码、运行源码、ELF/bin、构建／主机测试／烧录／Flash读回及板端状态记录。

| 产物 | SHA-256 |
| --- | --- |
| H7_clion.bin | 980d0d3b2227a1ab737581d9a0b758f6832f8708ac973ac4a92ca8c12cf2f184 |
| H7_clion.elf | 45279478b9f2b320646d1ee3218595ec230bc54b583e7498e47e3e43b3b02b6b |
| source-snapshot.tar.gz | fd29bad1602ea946ffea764d2c978996967788944f79a45411c0ebb72504f87d |

ARM GCC Debug构建通过；部署仓4组C检查（含8组安全场景）和27项Python检查通过。此为部署仓检查，不是训练仓单元测试。DMA缓冲位于0x24000000 RAM_D1。板端验收结果以该目录verification.json和board状态记录为准；未记录的动作或延迟不视为通过。

本次无模型交付。训练、Isaac回放、MuJoCo行为验证未执行：没有更改策略、仿真控制或机器人资产，这些仿真也不包含新增DR16/USB硬件门禁。真机非零力矩、策略运动闭环未执行。旧观测方向证据在映射未改范围内仍有效。
