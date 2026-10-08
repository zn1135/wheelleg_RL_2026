# chuanliantui：CAD 机构上的 H7 闭链控制回放

## 范围与版本

| 项目 | 记录 |
|---|---|
| 日期、实施、复查 | 2026-10-03；Codex 实施，独立子 Agent 复查 H7 源码和回放集成 |
| 用户目标 | 保留 CAD 闭链机构，状态解算和控制对齐用户指定的 H7；仅验收起立、零速站立 |
| 旧接口 | `chuanliantui-25d-100hz-r1`，原闭链 CAD 适配器 |
| 本次回放标识 | `chuanliantui-h7-closed-replay-r1`；不是新固件接口发布 |
| 训练仓 | `zn1135/wheelleg_RL_2026`，`26_wheelleg`，HEAD `3e5b1d88c0825784b4b25a5a5d7f1f0dfc6d02c5` 加未提交改动 |
| H7 来源 | 用户指定的 H7 克隆，`main`，HEAD `0af5620eca540c2e547ca31ac14ad8298e256992`；本轮不用 H7_clion |
| H7 工作区 | 存在工程配置、遥测指纹、VOFA 打包和文档改动；本次引用的解算、PD、机器表未修改 |
| H7 权威实现 | `imcalib/Algorithm/leg_solver.c`、`rl_torque.c`、`rl_observation.c`、`rl_policy.h`；`task_policy.c`、`task_actuation.c`；`user-lib/machine_config.c/h`、`pid.c` |
| 当前同步状态 | 只修改训练仓仿真适配及说明；H7 源码、固件和实体电机均未操作，无 PR |

## 接口定义

策略仍使用完整 `model_*.pt` 的 encoder+actor：观测 25、历史 125、latent 3、动作 6。
观测字段、缩放、默认训练关节角见[部署约定](../deployment-contract.md)。+x 前向，yaw 命令是偏航角速度。
MuJoCo 的机体状态作为已校正 IMU 输入，不重新套用硬件 IMU 安装变换。

| 环节 | 原 CAD 控制 `cad` | 当前默认 H7 控制 `h7` |
|---|---|---|
| 物理机构 | `chuanliantui.xml`，四个 connect、六电机、两气弹簧 | 同一 CAD 机构；XML、质量、惯量、被动关节和气弹簧不因本次控制对齐而改变 |
| 虚拟膝 | CAD 两次圆交点和解析 Jacobian | H7 同轴五连杆，`lu=.21`、`lg=.25`，固定圆交点分支；`vshank=wrap(phi_a-qf-pi/2)` |
| 策略关节序 | `[lf0,lf1,lfwheel,rf0,rf1,rfwheel]` | 同序；四腿角按 H7 `rl.sign/zero` 转训练空间，轮绝对角不输入策略 |
| 速度 | 虚拟膝用 Jacobian，其余 DOF 位置差分 | 主动轴 `qvel` 表示反馈速度；虚拟膝用 H7 Jacobian；不模拟 CAN 量化、滤波或异步到达 |
| 腿控制 | Kp=10、Kd=1，普通位置差 | 同增益，位置误差按 H7 环绕到 `[-pi,pi]` |
| 轮控制 | Kv=.1，默认不限目标速度 | `clip(action*10,±20 rad/s)`，Kv=.1 |
| 动作和限矩 | 动作 ±100；虚拟腿40/轮3.9，映射后相同 | 相同数值；在固件坐标计算 PD、按 H7 Jacobian 映射，再裁实体力矩并转 CAD 轴向 |
| 气弹簧 | 每侧物理推力默认150 N | 同上；H7 `gas_comp_sign={0,0}`，不额外叠加软件补偿 |
| 周期 | 2 ms 物理/PD、10 ms策略 | 同上；反馈与控制理想同步，不代表 H7 任务调度实测 |
| 接管 | `--standup` 首次轮接地后的下一拍 | 模拟 RL 已投入；前10个有效策略步零动作但允许PD，第11拍推理，无接地门控 |
| 无效状态 | CAD 几何失败报错 | 无效观测清空历史、关闭电机输出、重新预热；推理非有限输出关闭当拍电机输出 |
| 初态 | `--standup` 为0.15 m后摆，CAD求闭合初态 | 相同物理初态；该初态不等于实机投入时实际姿态 |
| 默认高度 | .32 m，可显式覆盖 | .20 m，来自当前 H7 命令；本轮新训练目标 .22 m 必须显式覆盖并记录 |

## 角度注册及几何核查

H7 第一逻辑腿对应模型 `lf`，第二逻辑腿对应模型 `rf`，按策略槽位记录。
这不证明实物左右、编码器或接线已标定。模型 `lf/rf` 与实物左右命名问题见部署约定。
仿真已直接读取具名轮关节，H7 驱动层的左右轮槽位交叉不再次叠加。

在 H7 平面 `(x,y)=(CAD x,-CAD z)` 中，使用：

```text
qf = front_zero + CAD_axis_y * q_front
qb = rear_zero  + CAD_axis_y * q_rear
front_zero = atan2(-front_rod.z, front_rod.x) ≈ 2.476872099134 rad
rear_zero  = atan2(-rear_rod.z,  rear_rod.x)  ≈ 0.667594581547 rad
```

前轴方向与 H7 `rl.zero_thigh=2.476872` 一致；后轴值只是 CAD 名义几何注册。
H7 的 `dm_zero` 是原始编码器校准量，不能直接替代此值；`rl.zero_shank=3.086386`
也不是后轴零点。有共同姿态实测依据后可用 `--h7_rear_zero <LF> <RF>` 覆盖。

虽然物理 CAD 杆件数更多，其后支链 `.1134/.135` 是 `.21/.25` 的 .54 缩比联动。
真实 H7 C 解算与 CAD 解算在三个初态和40组膝行程/髋跨环绕网格上的最大膝差为
`5.28e-7 rad`、Jacobian元素差 `2.50e-7`。该名义运动学核查不支持“换五连杆公式即可修复 gap”的结论。
结果采用 C float/libm；未验证 MCU CMSIS 查表的位级一致性。

## 使用与兼容性

```bash
# 代码数值对照：实际 H7 C 解算、PD 和机器表，仅在 /tmp 编译 host 参考库
python scripts/agent/check_chuanliantui_h7_control.py --h7-repo <H7目录>
python scripts/agent/check_h7_policy_warmup.py

# 起立及零速站立；使用完整 checkpoint，0.22 m 是显式命令覆盖
python sim2sim/mj_sim2sim_ct.py --closed_chain --standup --render \
  --checkpoint <完整model_*.pt> --cmd_vx 0 --cmd_yaw 0 --cmd_height 0.22
```

加 `--closed_chain_controller cad` 可回到原 CAD 控制。未传 `--closed_chain` 的串联训练代理
保持原控制行为。旧25维 checkpoint 能加载，但不保证 H7控制路径的行为；此次未训练、未导出ONNX。
本轮不验收行走。观察实际高度、pitch、位移、被动膝余量和受扰恢复，不只看 H7 估算状态。

诊断报告分别记录策略虚拟状态、真实被动膝状态、实体主动轴状态及实体力矩。
`physical_dof_names` 与 `motor_names` 显式给出两个不同的6维顺序，不能把被动膝角和后轴力矩配对。

## 验证记录

本机数值/回放记录位于 `/tmp/ct-h7-closed-alignment-20261003/`；几何探针及文件SHA-256
位于 `/tmp/ct-h7-legsolver-20261003/`。交付时需将本机临时记录复制到团队产物位置。
当前验证采用旧 `Oct03_03-28-58_standup_action_delay_0_10ms/model_9000.pt`，SHA-256：
`54e4e5712e8e45e69824e2cad40d41706c332f76ff439cf0d98c813204f05d49`。
历史模型与 logs 不修改。相关运行源码和 H7 数学模块的快照与哈希保存在
`/tmp/ct-h7-closed-alignment-20261003/source-snapshot/`，包含本次运行依赖的已有未提交改动。

数值检查：实际 H7 C 与 Python 的解算/Jacobian/PD/力矩输出对照通过，预热/历史恢复检查通过；
完整 checkpoint 形状自检通过（不验证行为）。回放均为10秒、零前进/转向指令、默认XML摩擦0.5、旧9000权重，退出0且数据有限：

| 控制条件 | 命令高度 | 5–10秒实际高度 | 5–10秒 pitch | 5–10秒前向速度 | 站立结论 |
|---|---:|---:|---:|---:|---|
| 原CAD、不裁轮目标 | .22 m | .20390 m | −.99° | 约0 m/s | 起立后基本静止；仍未达到新站高/膝余量 |
| H7完整控制 | .22 m | .15788 m | +6.20° | +.856 m/s | 持续前跑，零速站立未通过 |
| H7完整控制 | .20 m | .15546 m | +6.88° | +.860 m/s | 持续前跑，零速站立未通过 |
| 原CAD、仅加轮目标±20 | .22 m | 约.158 m | 约+6.2° | 约+.86 m/s | 同样持续前跑，零速站立未通过 |

保留的 CAD 对照路径与修改前1000个采样的19个共有字段逐值一致。
两组 H7 回放在0.10秒首次推理，几何一直有效；轮目标限幅各触发990/1000个策略步。
H7 .22组10秒内世界x位移约+8.17 m，实体力矩在100Hz记录点未触顶。
仅增加轮目标裁剪就足以在此模型、权重和初态下复现持续前跑，支持继续排查该训练/部署差异；
它不等于已经证明真机唯一根因。未为通过站立而改变 H7 对齐参数，未额外训练或验收行走。
单因素对照在0.10秒时状态、观测和动作完全相同，仅轮力矩因裁剪而不同；
0.11秒起状态才分叉，详见 `cad-wheel20-first-divergence.json`。
`behavior-summary.json`、`cad-compare.json` 和对应日志保留完整结果。
板端时序、实际烧录版本、编码器标定与真机行为均未验证。
