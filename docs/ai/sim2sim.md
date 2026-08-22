# sim2sim 与部署契约

改动观测构造、控制时序、URDF/XML、或准备真机部署时读这篇。权威来源是 `sim2sim/mj_sim2sim.py` 的文件头注释，本文是它的索引与排查经验。

## 部署契约

任何部署端（MuJoCo 脚本、真机驱动）必须逐条对齐 `wheel_legged_gym/envs/base/legged_robot.py`：

| 项 | 值 |
|---|---|
| 策略 | `ActorCriticSequence`，`action = actor(cat(obs_27, encoder(history_135)))` |
| 权重 | 用 `model_*.pt`；`policy_1.pt` 缺 encoder 不可用 |
| DOF 顺序 | `[lf0, lf1, l_wheel, rf0, rf1, r_wheel]` |
| 观测 27 维 | `base_ang_vel*0.25(3)`, `projected_gravity(3)`, `cmd*[2.0,0.25,5.0](3)`, `(dof_pos-default)*1.0(6)`, `dof_vel*0.05(6)`, `last_action(6)`，裁剪 ±100 |
| 历史 135 | 27×5 FIFO，最旧在前、最新在末；上电用首帧重复 5 次填充 |
| `dof_vel` | 位置差分 `wrap_to_pi(Δdof_pos)/sim_dt`，每个 sim 子步更新一次（**不是**读速度传感器） |
| 动作→力矩 | 腿位置控制 Kp=60/Kd=2；轮速度控制 Kp=0/Kd=0.5 |
| 时序 | `sim_dt=0.005`，`decimation=2` → 策略 100 Hz，PD 内环 200 Hz |
| 力矩上限 | `[30, 30, 5, 30, 30, 5]` N·m |
| 链路延迟 | 训练随机化 0~10 ms。真机通信+执行延迟必须 ≤10 ms，超出即出分布 |
| 前进方向 | 机体 **+y**。`cmd_vx` 是「前向速度命令」（通道 0），对应机体系 vy |
| yaw 通道 | **是外环反馈，不是常数**：`cmd[1] = 1.5*(目标航向 - 当前航向)` 剪 ±5、每步刷新 |

最后一条最容易漏：训练时 `heading_command=True`。部署端喂恒定 yaw 值会导致偏航漂移无人纠正，巡航数秒后摔车。真机必须实现同样的航向保持外环。

## MuJoCo 侧三个已修的坑

`sim2sim/mj_sim2sim.py` 里已全部修复，**勿回退**：

1. **`mj_objectVelocity` 必须用 `mjOBJ_XBODY`**（机体系）。`mjOBJ_BODY` 返回惯性主轴系，base 惯量特征值降序导致轴置换，陀螺仪三分量整个换位。症状：站立勉强、一加速就翻。
2. **`mj_step` 后派生量（`cvel` 等）滞后一个子步**，读状态前必须 `mj_forward` 刷新。
3. **PD 内环顺序必须是「算力矩 → 步进 → 差分更新 dof_vel」**，对齐 `legged_robot.py`。顺序错了 obs 里的 `dof_vel` 滞后 5 ms。

真机同构风险：IMU 角速度的坐标系定义与采样时序必须与策略推理同拍。5 ms 级延迟对静态站立无感、对加速瞬态致命。验证方法是双源交叉（位置差分 vs 速度读数）。

## 摩擦

MuJoCo 摩擦 = 两 geom 逐元素 **max**；PhysX 取**平均**。训练等效摩擦区间约 [0.4, 1.0]，等效均值约 0.75。MJCF 里 0.5 踩在下沿，`mj_sim2sim.py --friction` 可覆盖。

XML 已用 `cone=elliptic` + `impratio=10`（轮式标准配置）。

## Isaac 对照评估

`eval_isaac.py` 必须设 `train_cfg.runner.resume = True`（对齐 `play.py`），否则 `make_alg_runner` **静默**跑随机初始化网络。症状：换 checkpoint 输出逐字节不变。自检习惯：换权重必须换行为。

## 当前性能上限：encoder 速度估计正偏置

截至 2026-07-21，基准权重 `logs/mini_wheel_legged/Jul21_14-09-24_/model_4000.pt` 在两引擎站立/加速/巡航零摔倒，sim2sim 通过。

余下三个现象同一个根源——latent 前 3 维（`base_lin_vel*2` 的估计，见 `rsl_rl/algorithms/ppo.py`）**系统性估高 0.15~0.25**：

- 稳态欠速：真实速度恒低于命令约 0.2 m/s
- MuJoCo 低速后拉：偏置 +0.1 时，调 v̂ 到 0 对应真实 -0.1
- 高速 ≥1.8 m/s 刹车瞬态 encoder 跟丢 → 发散翻车

结构性原因是 `obs_history_length = 5`，encoder 只有 50 ms 窗口。

**已接受现状，安全包线 ±1.5 m/s**（遥操作已限幅）。若将来要更高速：`obs_history_length` 5→20 全重训，或加大 encoder 损失权重续训作廉价改良。

诊断手段：`mj_sim2sim.py` 与 `eval_isaac.py` 日志里的 `v̂` 列（策略内部速度估计）对比真实 `v_fwd`。

## URDF / XML 同步

`resources/robots/imcawl/urdf/imcawl.urdf`（Isaac 侧）与 `sim2sim/imcawl.xml`（MuJoCo 侧）是**两份手工维护的模型**。改一边必须改另一边，然后跑 `python sim2sim/check_model.py`。连杆长度变了还要同步 `mini_wheel_legged_config.py` 的 `asset.l1 / l2`。
