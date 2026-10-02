# 接口记录：chuanliantui 动作执行前复算

| 项目 | 内容 |
| --- | --- |
| 日期、执行者、独立复查 | 2026-10-01，Codex；独立只读子任务复查虚拟 PD 工具及实际 2002 帧复算，非人工运动验收 |
| 机器人／任务 | 大轮腿，chuanliantui_standup |
| 既有接口 | [chuanliantui 25d](../deployment-contract.md)、[2701 失能诊断](chuanliantui-h7-policy-diag2701.md) |
| 本次变化 | 无线协议／控制行为不变；增加纯离线动作→虚拟 PD 检查、准备单轮台架工具 |
| 训练版本 | 26_wheelleg，bb56dd40964dc569b4e2a406983b03fa19d943a1，脏工作区；逐文件 SHA 在部署 captures/20261001-policy-torque-preview-01/report.json |
| 部署版本 | H7_REPO_PATH，H7_clion 无 Git；继续使用 2701 固件，未因本次复算烧录 |
| 权威实现 | legged_robot.py::_compute_torques，chuanliantui_config.py，mj_sim2sim_ct.py::compute_torques，H7 App/lower_observation.c、lower_geometry.c、lower_app.c 和 dm/dji 驱动 |

## 定义与兼容性

观测、history、坐标、零点、25/125/6 张量、500 Hz PD／100 Hz 策略契约、模型格式均不改变。
腿目标 `default+0.5*action`、轮速目标 `10*action`，名义腿 Kp/Kd=10/1、轮=0/0.1；
虚拟力矩限幅 `[40,40,3.9,40,40,3.9]`。新脚本只评估本拍观测与该拍发布动作对应的
假想首个 PD 步，不模拟机械响应，不把 previous action 当成这一步的新输出。

模型 lf 是实体右，rf 是实体左；腿实体 USB 槽为 rf→0/1、lf→2/3，轮为 rf→4、lf→5。
闭链执行需要 Jacobian 转置及与观测配对的符号；部署端完整公式与单轮台架条件见
`H7_REPO_PATH/docs/action-output-validation.md`。没有把该公式接入板端策略执行。

沿用原 model_6000.pt，无需因离线工具重新训练／导出；仅能复用其张量与推理接口，
**完整策略的真实机械行为仍未验收**。不根据本次输出范围更改训练尺度、关节限位或增益。
两个仓库无需切换版本合入；没有 commit、PR 或发布，旧诊断工具继续可用。

## 验证证据

新工具 `sim2sim/preview_policy_torques.py` 用项目 Python3.8、Isaac Gym 先于 torch 导入。
normal-02 与 reset-01 共 2002 帧重新验证数据链通过；虚拟 PD 与独立简化公式逐帧一致。
结果在 `H7_REPO_PATH/captures/20261001-policy-torque-preview-01/`，保存输入与源码哈希。
双轮每帧达到各自 ±3.9 N·m，腿虚拟力矩最大约 35.7 N·m；这是静止失能状态下的假想输出，
不能证明闭环失稳、观测错误或实体电机确实受到该力矩。

本次没有同拍实体 DM 输入角，未报告精确实体闭链力矩；未运行训练、Isaac 回放或 MuJoCo
行为验证，因为未修改训练、策略、动力学或执行控制器。新增主机台架程序的离线检查与非零
实机运动验收分别记录在部署端文档；完整执行链、500 Hz 实际内环及物理标定仍待完成。

模型使用来源见[既有模型记录](../model-deliveries/20261001-chuanliantui-disabled-policy-diagnostic.md)。

## 后续单轮台架结果（2026-10-01）

用户确认支撑条件后，以独立于策略的单轮小脉冲验证了两轮：实体右轮 USB5、实体左轮 USB4
正力矩均使胎顶朝机头，STATUS 轮速度均为正。模型 lfwheel 的负号、rfwheel 的正号保留。
用户指定峰值 0.50 N·m；右轮实际指令达到 0.50、左轮约 0.4004 时已触发主机速度停止条件，
分别约 102/82 ms 提前切零失能，没有完成全长 300 ms。不是硬限速或立即刹停。
随后两次独立只读复查均确认静止、全零和失能；四腿非零输出及完整策略仍未测试。
部署端完整原始记录、失败的左轮等待尝试、工具版本与检查见
`H7_REPO_PATH/docs/wheel-output-validation.md`。轮速比例、力矩幅值、零点待验项不变。
