# 接口记录：chuanliantui 失能策略数据链 2701

| 项目 | 内容 |
| --- | --- |
| 日期／作者／复查 | 2026-10-01，Codex；独立子任务复查USB缓冲、时间戳、会话和遥控互锁，已修正提出的四处边界 |
| 机器人／硬件 | chuanliantui_standup，大轮腿，STM32H723，板载USB CDC，DR16 UART9 |
| 旧接口 | [25维预览](chuanliantui-h7-clion-vofa25-preview.md)及[遥控2602](chuanliantui-h7-clion-remote-enable.md) |
| 新接口 | h7-policy-disabled-diag-2701；USB外层A55A v1，payload上限768字节 |
| 训练版本 | 26_wheelleg；bb56dd40964dc569b4e2a406983b03fa19d943a1，已有未提交改动，另附运行源码快照 |
| 部署版本 | H7_REPO_PATH指定H7_clion，无Git元数据；以captures/20261001-policy-runtime-firmware的ELF/bin/源码SHA标识 |
| 权威实现 | 训练sim2sim/host_policy_diag.py；部署App/lower_policy_diag.c、lower_app.c及docs/policy-runtime-diagnostic.md |

## 定义与差异

观测仍为25维，原物理映射与缩放不变。普通预览last_action为0；诊断时obs[19:25]改为
前一个sample的真实encoder+actor输出裁剪±100后原始六维float32动作，不乘腿/轮控制尺度，
不使用Nm。固定命令[0,0,0.20]。机体+x前向、机体系IMU及直接yaw角速度命令均沿用旧接口。

历史125维、5帧，最旧在前、当前在末；第0帧动作全0并重复首帧5次，此后每10ms移位并
追加当前obs，再发送完整obs/history。首尾帧跨度40ms。电脑推理a(n)，板端只接受同会话、
来源sample=n、且应用解析接受时间早于sample_ms+10的动作；下一帧回显实际采用身份和数值。

新增0x10 START、0x11 ACTION、0x12 STOP、0x13 QUERY，响应0x90 SAMPLE（664B）及
0x91 STATUS（64B）；所有长度、字段及错误码见部署端协议文档。时间戳为HAL毫秒分辨率；
action_rx_ms表示主任务接受时刻，包含USB接收后的排队时间，tx_ms在构包时采集。

首帧、复位、超时和断线均清动作/历史。错session、错序、重复、非有限动作、缺拍、采样
周期不为10ms或USB背压时进入FAULT；不补造重复帧。FAULT保留模式占用，必须明确STOP
退出。遥控必须在线左下，任何有效帧离开左下也锁存FAULT。RUNNING/FAULT均禁止遥控
使能和USB ARM/SET_TORQUE，电机保持失能。旧132B STATUS保持，并增加bit11诊断占用。

VOFA仍为63路，诊断时布局标记改2701，普通预览仍2501。旧2501工具对诊断数据应拒绝
固定零动作断言。此次没有部署动作→目标→PD→力矩执行链，未更改关节零点、几何、轮速比例、
动作控制尺度或模型权重。

## 兼容与边界

沿用完整model_6000.pt，形状25/125→6/3；不需要为数据链诊断重新训练或导出。
旧HPI1主机脚本不兼容本协议，使用新host_policy_diag.py。训练reset只清last_actions但
观测读取self.actions，以及训练history先于外层obs裁剪，这两项与清零部署reset/饱和边界
不同。本次没有修改训练实现，不宣称这些边界完全对齐。

关联[模型使用记录](../model-deliveries/20261001-chuanliantui-disabled-policy-diagnostic.md)。
主机离线26项检查和部署5组C/27项Python检查通过；正式板测以证据目录的运行summary.json
和verification.json为准。未完成项不能由构建或离线检查替代；不证明策略运动或所有标定正确。

板测已完成：正常及故障后重启各1001帧通过，板端间隔均10ms，动作应用接受延迟最大4ms，
动作与FIFO逐位一致；漏发第120拍后在下一10ms边界触发预期FAULT，三次session和历史
初始化互相隔离。全过程失能，退出均确认OFF和目标0。第一次主机START检查过严已在r2
修正，失败原始记录保留。运行版本、每段SHA及结论见部署证据目录manifest/verification。
