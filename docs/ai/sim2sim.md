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

## chuanliantui 串联训练代理回放

`sim2sim/mj_sim2sim_ct.py` 默认加载 `chuanliantui_train_proxy.xml`。它由
`resources/robots/chuanliantui_new_1/urdf/chuanliantui_train.urdf` 生成，保留训练端
6-DOF 串联拓扑、固定后支链、地面、根 free joint 及六个同名力矩电机；模型必须保持
`neq=0`。改训练 URDF 后运行
`scripts/agent/generate_chuanliantui_train_proxy_mjcf.py`，再用
`scripts/agent/check_chuanliantui_train_proxy.py` 检查该契约。

chuanliantui actor 观测为 25 维：机体系角速度(3)、重力投影(3)、命令(3)、四个腿关节
`[lf0, lf1, rf0, rf1]` 的位置(4)、六个关节速度(6)、上次动作(6)。连续轮的绝对位置
不进入 actor 或 encoder；历史为 `25×5=125`。这改变了网络的 encoder、actor 和 critic
输入形状，所有旧 27 维 chuanliantui checkpoint 均不能加载，必须重新训练。
提交观测变更前可运行 `python scripts/agent/check_chuanliantui_observation.py`，核对训练端布局、
噪声、critic 维度、历史 FIFO 和 MuJoCo 构造是否一致。

训练与 MuJoCo 回放均用 500 Hz 的物理/PD 内环（`sim_dt=0.002`），每 5 个内环步推理一次，
因此策略、观测和历史 FIFO 保持 100 Hz（`0.002×5=0.01 s`）。动作在中间 4 个内环步零阶保持；
气弹簧力在每个 2 ms 内环步重写。

该模式只消除 Isaac 串联训练资产与 MuJoCo 回放资产的结构差异，**不是**真实闭链或
真机 sim2sim。默认会拒绝任何含 equality/connect 约束的模型，防止两条链路混用。

仅在闭链差异诊断时，可显式传入 `--closed_chain`。该选项改用
`chuanliantui.xml`（`neq=4`）和 `ClosedChainAdapter` 的闭链姿态求解/力矩映射；它还以
两个 spatial tendon 模拟左右气弹簧。端点由 CAD 定位件 `rf001/rf002` 与
`lf001/lf002` 换算后附着在实际前支路 body 上，默认每侧恒定伸张推力 150 N；可用
`--gas_spring_force 0` 关闭以做对照。该恒力模型不含真实气弹簧的力—长度曲线、阻尼或
行程，不能作为串联训练策略已完成 sim2sim 或可上真机的证据。

用 `python sim2sim/mj_sim2sim_ct.py --gas_spring_view` 打开气弹簧专用视图（自动开启
闭链和渲染）：半透明机构中，左侧青色、右侧橙色，圆点为安装端点，粗线为气弹簧轴线。
显示加粗仅影响渲染，不改变气弹簧力或碰撞几何；每侧默认仍为 150 N。

如需固定机身、手动拖动腿部观察气弹簧，运行
`python sim2sim/view_gas_spring_ct.py`。该独立演示在内存中移除基座自由关节，
将机身固定于 0.8 m，不加载策略，六个实体电机输出始终为零，只保留每侧 150 N
气弹簧和重力。双击腿部后 Ctrl+右键拖动施力；空格暂停/继续，Home 复位，F5
开关气弹簧。`--headless --duration 10` 可运行无窗口检查；原始 XML 不被修改。
机身上方实时显示左右气弹簧对 `lf1/rf1` 的直接关节力矩（N·m）、轴向力（N）和
有效力臂（mm）。力矩按 `actuator_force * actuator_moment` 计算，符号遵循各自
关节轴，不包含重力、鼠标外力和闭链约束反力，也不是主动电机轴的等效补偿力矩。

复旦原腿的固定机身对照：`python sim2sim/view_gas_spring_fudan.py`，默认从本机
`/home/zn/文档/fudan_rl_wheel_leg-main` 加载原 XML（可用 `--xml` 指定路径），
每侧使用 150 N 便于同力对照；`--gas_spring_force 300` 恢复原脚本的力值。
该视图无策略、无实体电机驱动，保留原模型的限位、阻尼和闭链锚点；在内存中把
`connect site1/site2` 转为 MuJoCo 3.2.2 支持的两刚体局部锚点，不修改源 XML。
复旦模型的 `l20/r20` 不限位；当前闭链 XML 的 `lf00/rf00` 也已改为
`limited="false"`，允许后输入轴连续旋转，避免原 ±3.14 rad 限位阻挡髋部转过一圈。
这是 MuJoCo 模型对原始 CAD URDF 限位的显式修正；闭链生成器已同步该规则和气弹簧定义，
重新生成后运行 `python scripts/agent/check_chuanliantui_closed.py`，检查端点连接、
每侧 0/150 N 施力、±360° 闭合姿态以及原有的 5 秒无控制仿真。
前支路膝关节与其余被动关节仍保留原有范围。
原始四对 site 存在约 10 mm 的横向错位，加载后的演示保留此偏差，不能视为闭合精度验证通过。
