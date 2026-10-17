# 平地速度与角度跟踪测量（multimode 39 维，v1）

## 适用范围

mjlab 串联腿 multimode 单线（任务 `chuanliantui`，39 维 actor／82 维 critic／6 动作，观测契约
`chuanliantui-huananhu-multimode-39-82-v1`、实验 profile 同名，2 ms 物理、20 ms／50 Hz 策略）
在**平地**上对**前向速度**与**偏航角速度**命令的跟踪测量。

2026-10-17 起地形由生成地形（flat 50%／slope 20%／step 15%／stair 15%）改为无限平地，命令包线相应
收窄为 vx ±1.5 m/s、yaw ±1.5 rad/s；模式槽、倒地恢复机制、奖励权重、恢复窗口与 39/82 契约不变。

本口径只覆盖「平地、固定命令、单模型、单次测量」。它**不替代**双轮平衡／起身验收
（[双轮平衡配对验收 v3](chuanliantui-wheel-balance-acceptance-v2.md)），不等于 CPU MuJoCo sim2sim
通过，也不等于真机可用；H7 对 39 维接口、50 Hz 调度与板端数值的核对状态仍是「待确认」。

## 测量协议

- 配置：`play`（无域随机化、无观测噪声与延迟、无推扰事件）。
- 初态：只用姿态库的普通（站立）初态并强制 `ground_start=False`；地面初态会让稳定窗口用于起立而非跟踪。
- 命令：每个环境在整段测量中**固定一条命令**（不重采样），网格为
  vx ∈ {0, ±0.5, ±1.0, ±1.5} m/s × yaw ∈ {0, ±0.5, ±1.0, ±1.5} rad/s × height ∈ {0.27, 0.34} m，
  每点默认 1 个环境、seed 42。
- 时长：12 s，其中前 2 s 为稳定窗口（不参与统计）。每环境**单回合**：一旦终止即冻结该环境，
  不重启、不补跑，因此成功率不会被重跑抬高（批次内该环境仍会被 reset 以便继续步进，
  但累计量已冻结，不再进入统计；在稳定窗口内就结束的环境记为 `no_measured_steps` 失败而不是让报告中断）。
- 命令在测量期不重采样（重采样时间区间钉为极大），命令项自己的 `_resample_command` 也被替换成写入该网格的实现，
  使复位时的抽签同样落在网格上；每一步都断言传入策略的观测里命令槽等于网格，一旦漂移立即报错，不做静默容忍。
- 首拍另用一步零动作步刷新管理器的缓存观测（观测在策略步外重算会返回缓存），该步落在 2 s 稳定窗口内、不计入统计。
- 跳跃请求在测量期被抑制（`jump_cooldown` 钉高），并报告 `mode_jump` 时长，避免「被静默忽略」当成通过。
- 速度定义：训练侧的 `root_link_lin_vel_b`／`root_link_ang_vel_b`（机体系）。
- 实现：`scripts/agent/assess_multimode_tracking.py`；判据常量是本脚本的模块常量，为唯一真源。

## 判据（单环境，全部满足才算通过）

| 条件 | 门槛 |
|---|---|
| `tracked_vx` | 稳定窗口后 mean \|vx_cmd − vx\| ≤ 0.30 m/s |
| `tracked_yaw` | mean \|yaw_cmd − ω_z\| ≤ 0.30 rad/s |
| `tracked_height` | mean \|h_cmd − h_terrain\| ≤ 0.03 m |
| `upright` | 过程中最大倾角 ≤ 25° |
| `no_base_contact` | 机身接触力 > 1 N 的时长为 0 |
| `survived` | 12 s 窗口内没有物理或数值终止 |
| `numerical_ok` | 无数值异常 |

报告另外给出 `vx_bias`、`yaw_bias`（实测 − 命令，用于识别系统性欠速／超速）与 `vy_abs`（横向漂移诊断，
不参与判据），并按 `all`／静止（vx=yaw=0）／运动／各 \|vx\| 档／各 \|yaw\| 档分组汇总。
阈值由 `scripts/agent/check_multimode_tracking.py` 离线核对；文档、脚本常量与离线检查不一致时，
以脚本常量与离线检查为准。

## 命令

```bash
# 测量（输出目录每次使用新名称）
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_tracking.py \
  --checkpoint <model_*.pt> --output artifacts/<新名称>/result.json
# 把成功率当门槛（默认只报告，退出码 0）
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_tracking.py \
  --checkpoint <model_*.pt> --output artifacts/<新名称>/result.json --min-success-rate 1.0
# 判据算术与容差一致性的离线检查（不建环境、不加载模型）
mjlab_training/.venv/bin/python scripts/agent/check_multimode_tracking.py
```

## 已知边界

- 只测平地与固定命令；不覆盖命令段切换（训练侧 5～15 s 重采样）、坡面／台阶／楼梯、推扰抗性、
  中途倒地恢复与真机。
- 判据只覆盖跟踪本身：不检查双轮承重分配、非轮支撑与横向漂移（`vy_abs` 只作诊断），
  因此不能代替[双轮平衡验收 v3](chuanliantui-wheel-balance-acceptance-v2.md)。
- 训练侧奖励对 \|命令\|<0.1 的通道归零、线速度还乘 `cos(pitch)`，本测量直接比较机体系速度，
  两者不是同一个量；阈值是粗筛，不证明精确跟踪。
- 12 s 窗口短于任务回合 20 s，因此 `survived` 表示「窗口内没有终止」，不是「撑满回合」。
  出界半径 45 m（`multimode_mdp.OUT_OF_PATCH_RADIUS_M`）高于一个回合的最坏直线位移 30 m（20 s × 1.5 m/s）
  以及约 25% 超速时的 37 m，12 s 窗口内更不会因出界提前终止；该半径只是失控保护，
  不承担分隔环境的职责（mjlab 每个环境是独立物理世界）。
- 机身触地用接触传感器 8 拍力历史（与奖励里 `undesired_contact` 同口径），约覆盖 16 ms，短于 20 ms 策略周期：
  策略周期最开始约 4 ms 内的瞬时接触可能被漏记。
- `sim2sim/mj_sim2sim_huananhu_multimode.py` 用同一组容差，但协议不同（窗口长度可调、渲染中可按 Home/R 复位、
  触地只看策略时刻），因此两边是「容差一致」，不是「判据等价」。
- 地形改为平地后，`eval` 的 64 强制初态**初态签名**（含地形列）随之改变，旧的
  `model3023-baseline-gpu64` 配对协议不可直接沿用，需要按平地协议重录参考基线；
  恢复验收 v3.1 的口径本身不变。
- 单次实现的读数是抽样，不是策略的稳定属性：该线已记录过「同权重、同种子、同 GPU 的两次运行会
  出现样本翻转」。跟踪能力的刻画需要重复运行或用本测量覆盖网格。
- 训练前参考测量与本轮结论见[项目实验记录](../ai/project-log.md)。
