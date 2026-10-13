# r15 v3 两组实际训练参数全表

## 版本与原始配置

适用本次「移除默认角惩罚」的从头2000轮组、4000→5000轮续训组，接口 `chuanliantui-mjlab-getup-curriculum-r15-assisted-v3`。记录的是启动配置与代码实际行为，不是历史r14默认值，也不是评估时关闭随机化的配置。本次仅整理参数，没有改变训练。

2026-10-11从远端两组run取回完整JSON。两组 `environment.json` 完全一致，SHA256均为 `df4ee22e9d5f7e55576e34480c094ea4c7946e199368bd08a1620a146bf1337b`。本地90个源码/资产文件全部通过训练快照SHA256核对。

原始配置保留所有管理器、求解器、场景、传感器、查看器默认字段：

- [从头 environment.json](../../artifacts/getup-r15-v3-training-parameters-20261011/scratch/environment.json)、[runner.json](../../artifacts/getup-r15-v3-training-parameters-20261011/scratch/runner.json)、[interface.json](../../artifacts/getup-r15-v3-training-parameters-20261011/scratch/interface.json)、[source-state.json](../../artifacts/getup-r15-v3-training-parameters-20261011/scratch/source-state.json)。
- [续训 environment.json](../../artifacts/getup-r15-v3-training-parameters-20261011/resume/environment.json)、[runner.json](../../artifacts/getup-r15-v3-training-parameters-20261011/resume/runner.json)、[interface.json](../../artifacts/getup-r15-v3-training-parameters-20261011/resume/interface.json)、[source-state.json](../../artifacts/getup-r15-v3-training-parameters-20261011/resume/source-state.json)。
- [原始配置ZIP](../../artifacts/getup-r15-v3-training-parameters-20261011.zip)、[两组启动及恢复记录](20261011-mjlab-getup-r15-v3-two-groups-start.json)。

`interface.json` 的 `base` 包含历史r14奖励/宽门控说明；当前生效奖励以外层r15 `weights` 和下文为准。`environment.json` 中奖励管理器权重全部是1，阶段权重在奖励函数内部应用。

## 1. 两组启动、保存与恢复

| 参数 | 从头组 | 续训组 |
|---|---|---|
| num_envs / seed / device | 4096 / 42 / cuda:0 | 相同 |
| 本次新增PPO更新 | 2000 | 1000 |
| 起始累计更新 | 0 | 4000 |
| 目标累计更新 | 2000 | 5000 |
| 网络起点 | 随机权重 | v2的model_3999.pt |
| 初始Adam状态 | 空 | 完整恢复 |
| 实际起始学习率 | 0.0001 | 0.000015 |
| 初始课程 | 扶正，stage_steps=0 | 扶正，stage_steps=96000 |
| 初始ready_steps / history | 0 / [] | 0 / [] |
| 辅助力累计进度 | 从0开始 | 从4000开始，因此全程0 |
| 本次配置max_iterations | 2000 | 1000，表示新增而非累计 |
| 显式迁移标志 | 无 | --resume-pose-bias-fix |

续训恢复actor、critic、Adam、课程与累计更新；不重新初始化探索标准差或辅助力进度。两组 `runner.json` 的learning_rate都写初始化默认0.0001，续训恢复优化器后实际覆盖为0.000015。随机种子相同不代表两组的网络、优化器或采样轨迹相同。

- 每50轮保存，另有初始/最终模型；logger=TensorBoard，upload_model=false；本次没有启动TensorBoard服务。
- `init_at_random_ep_len=false`；auto_reset=true；is_finite_horizon=false。
- experiment_name=`chuanliantui_getup_r15`，runner run_name=`getup_r15`；输出目录另含用户run标签。
- 实际训练和TensorBoard命令见[启动说明](20261011-mjlab-getup-r15-v3-pose-bias-removed-training.md)，不应重复执行现有训练。

## 2. PPO与网络

| 参数 | 值 |
|---|---|
| actor / critic / action维数 | 35 / 78 / 6 |
| actor / critic隐藏层 | 都为[256,128,64] |
| 模型 / 激活 | MLPModel / ELU，无CNN/RNN/encoder |
| actor / critic观测归一化 | 都关闭 |
| 分布 | GaussianDistribution，std_type=scalar，init_std=1.0 |
| 算法 / 优化器 | PPO / Adam |
| 每环境采样步数 | 24，即0.48秒物理时间 |
| 每轮总样本数 | 4096×24=98304 |
| epochs / mini_batches | 5 / 4，每batch24576样本 |
| 学习率调度 | adaptive，desired_kl=0.01 |
| 自适应学习率范围 | 1e-5～1e-2 |
| 自适应条件 | KL>0.02：LR÷1.5；0<KL<0.005：LR×1.5，按上述上下界截断 |
| gamma / GAE lambda | 0.99 / 0.95 |
| PPO clip_param | 0.2 |
| value_loss_coef | 4.0 |
| use_clipped_value_loss | true |
| entropy_coef | 0.005 |
| max_grad_norm | 1.0 |
| normalize_advantage_per_mini_batch | false |
| clip_actions | null；原始策略输出不截断，执行器目标和力矩另有限制 |
| share_cnn_encoders | false |

学习率调度已读当前远端rsl_rl的PPO实现核实。完整runner附带load_run、load_checkpoint、logger等字段，见原始JSON；实际续训通过自定义runner显式加载checkpoint，不由runner.json中的默认resume=false决定。

## 3. 仿真、场景、闭链

| 参数 | 值 |
|---|---|
| physics timestep / frequency | 0.002 s / 500 Hz |
| decimation / policy dt | 10 / 0.02 s（50 Hz） |
| 回合长度 | 20 s =1000控制步 |
| 地形 / 环境间距 | plane / 3 m |
| gravity | [0,0,-9.81] m/s² |
| integrator / solver / jacobian | implicitfast / newton / auto |
| cone / impratio | elliptic / 10 |
| iterations / tolerance | 100 / 1e-10 |
| ls_iterations / ls_tolerance | 20 / 0.01 |
| ccd_iterations | 50 |
| nconmax / njmax / contact_sensor_maxmatch | 128 / 2000 / 64 |
| disableflags / enableflags | 都为空 |
| mjlab nan_guard | disabled；任务自己的数值异常检查仍启用 |
| 地面摩擦 | sliding=1、torsional=0.005、rolling=0.0001 |
| 机器人接触priority | 1；地面默认0，使用机器人侧摩擦 |
| 接触传感器 | wheel_contact、base_contact、undesired_contact；found+force，netforce，history_length=8 |
| 闭链connect数量 | 6，左右腿各2、左右气弹簧各1 |
| 闭链solref | [0.005,1] |
| 闭链solimp | [0.95,0.99,0.001,0.5,2] |

机器人资产为 [chuanliantui.xml](../../resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml)，包含完整质量、惯量、碰撞、关节轴向、ref、被动限位、气弹簧端点。原模型总质量13.3945841005 kg、重量131.400870026 N；训练每环境随机质量后重量不同。名义姿态两轮水平间距0.4265059511 m。

被动膝物理限位仍存在：lf1 [-0.77,0.12] rad，rf1 [-0.12,0.77] rad；`knee_limit`奖励项不在r15奖励管理器中。其他全部物理字段以资产为准，未因删除pose_bias而改变。

## 4. 实体电机控制

| 参数 | 值 |
|---|---|
| 策略及观测顺序 | [lf0,lf00,rf0,rf00,lfwheel,rfwheel] |
| 内部物理顺序 | [lf0,lf00,lfwheel,rf0,rf00,rfwheel] |
| 四腿名义目标角，策略顺序 | [-0.6047205545,0.6083241726,0.6047205545,-0.6083241726] rad |
| 腿位置scale | 0.5 rad/action |
| 腿Kp / Kd | 60 / 2 |
| 腿目标限幅 | 各电机XML qpos0参考角±3.14 rad |
| 四腿XML参考角 | [-0.6647205545,0.6675945815,0.6647205545,-0.6675945815] rad |
| 轮速度scale / 限幅 | 10 rad/s/action / ±150 rad/s |
| 轮速度误差增益 / 额外阻尼 | 0.1 / 0 |
| 腿/轮力矩限幅 | ±40 / ±3.9 N·m |
| 地面初态电机静默 | 前0.2 s；电机力矩和向上辅助力为0，气弹簧物理力仍在 |
| landing_gate / standup_pushes | false / false；一般随机推扰仍启用 |
| 动作延迟 | 腿、轮独立抽取5或10 ms，向上取整成3或5物理步，实际6或10 ms |

控制公式：腿 `tau=strength*(60*kp_scale*(q_target-q)-2*kd_scale*qdot)`；轮 `tau=strength*kd_scale*0.1*(v_target-qdot)`；最后力矩截断。名义角只用于控制和观测，pose_bias移除不删除名义姿态。

## 5. 气弹簧与辅助力

### 气弹簧物理力

每侧一根真实滑动关节，沿资产轴向施力，通过闭链传播到腿；不是直接给腿电机添加固定补偿矩。

| 参数 | 值 |
|---|---|
| initial_compression | 0.01 m |
| force_min / force_max_reference | 0 / 150 N |
| reference_stroke | 0.09 m |
| 等效线性刚度 | 150/0.09=1666.6667 N/m |
| nominal damping | 50 N·s/m |
| damping随机范围 | 25～75 N·s/m |
| force_bias随机范围 | -50～50 N，每侧独立 |
| 最终force_limit | [0,150] N；只推不拉 |

公式：`compression=max(0,0.01-slide_q)`；`elastic=clip(1666.6667*compression,0,150)`；`F=clip(elastic+bias-damping*slide_qdot,0,150)`。零位是模型滑动关节零位，不是世界坐标长度。bias每次reset重采样；damping按电机参数重采样门槛更新。

### 质心向上辅助力

| 参数 | 值 |
|---|---|
| 施加部位 / 方向 | base_link质心 / 世界+z |
| 辅助力矩 | 无 |
| 适用阶段 | 扶正、站起；行走关闭 |
| 最大比例 | 1倍当前随机化后整机重量 |
| 退火终点 | 累计3000次PPO更新 |
| 课程升级允许辅助力 | ≤当前整机重量的5% |
| 地面初态前0.2s / 回放 | 关闭 / 关闭 |

`F_aux = W * clip((h_target-h)/h_target,0,1) * clip(1-u/3000,0,1)`，`u=恢复的累计PPO更新+当前run控制步数/24`。与随机外力叠加，每物理步重算，不累积。3000是累计轮数；从头训练2000轮末预算仍有1/3，但达到目标高度时实际力为0；4000轮续训起点预算已经为0。

## 6. 初态与重置

| 参数 | 值 |
|---|---|
| 普通 / 地面初态比例 | 50% / 50% |
| 普通root高度 | 基准0.38 m，必要时按碰撞界抬高 |
| 普通root roll / pitch / yaw | 各U(-0.15,0.15)、U(-0.15,0.15)、U(-π,π) rad |
| 地面root高度 / 姿态 | 0.25 m / 单位四元数，roll=pitch=yaw=0 |
| 地面机体附加随机化概率 | 0，倾斜/朝向/高度offset均0 |
| 地面腿正/负姿态分支 | 左右独立抽取，60% / 40% |
| 四腿电机初始速度 | 各U(-π/2,π/2) rad/s |
| 轮初始位置 / 速度 | 各U(-2π,2π) rad / U(-50,50) rad/s |
| 机体线/角初速 | 都0 |
| 被动关节初速 | 0 |
| 碰撞留量 | 0.002 m |
| 初态库 | 每模式每侧1024行，左右独立抽取；normal/positive/negative |

闭链位置来自离线求解的姿态库。电机速度单独随机化，未投影到闭链速度切空间。姿态库不是简单的名义角加噪声。[姿态库元数据](../../mjlab_training/src/ct_mjlab/reset_poses.json)、[完整库](../../mjlab_training/src/ct_mjlab/reset_poses.npz)、[采样实现](../../mjlab_training/src/ct_mjlab/reference_training/huananhu_poses.py)。

源腿长范围普通0.13～0.32 m、地面0.14～0.36 m；映射后的本机腿长范围0.1716367925～0.3367438954 m。`poses.json`中的历史reset高度0.15 m不是本轮实际地面初态，当前由采样器覆盖为0.25 m。

## 7. 命令与整批课程

| 参数 | 扶正 | 站起 | 行走 |
|---|---:|---:|---:|
| stage | 0 | 1 | 2 |
| height_target | 0.27 m | 0.30 m | 0.30 m |
| 前进命令 | 0 | 0 | U(-2.7,2.7) m/s |
| 偏航角速度 | 0 | 0 | U(-2π,2π) rad/s或航向控制 |
| 辅助力 | 开启且按累计轮数退火 | 同左 | 关闭 |
| 因触地/倾倒提前终止 | 否 | 否 | 连续20控制步后终止 |

- 命令重采样间隔U(5,15)秒。侧向速度命令固定0。
- 行走50%航向控制，目标航向U(-π,π)，增益5，输出截断至±2π；普通环境两次10%站立抽样，合计19%零速度。
- 行走地面初态前1.5秒有速度覆盖：vx/yaw各U(-1,1)。扶正/站起由课程在最终输出处重新清零。
- special_modes_enabled=false，不启用额外高速/旋转课程。
- height_range启动记录为[0.27,0.27]；阶段命令更新会覆盖为该阶段0.27/0.30/0.30，不能只读这个字段推断全程高度。

### 课程晋级门槛

整批单向扶正→站起→行走；按每控制步更新，不按个体normal/recover标志切换阶段。

| 参数 | 值 |
|---|---|
| ready_fraction | ≥2/3环境 |
| 连续保持 | 2 s =100控制步，比例不达标即重新计时 |
| 每阶段最少控制步 | 24×100=2400，对应100次PPO更新 |
| 高度误差 | ≤0.03 m |
| 总倾角 | ≤15°，依据投影重力z，不是两个欧拉角各15° |
| 双轮向上支撑 | 每轮>5 N，代码使用clamp(-Fz,0) |
| 机体接触 | ≤1 N |
| 辅助力 | ≤随机化后整机重量5% |
| 数值状态 | 有限、未标记numerical_failure |

阶段0和1允许整个20秒回合尝试恢复；阶段2不会再升级。课程ready_fraction与训练strict_recovered、回放持续站稳是不同指标。

## 8. 三阶段全部奖励

以下是奖励函数内部的实际权重；0表示该阶段不使用。最终每步送PPO的reward为 `0.02 * sum(weight * raw_term)`，不做正奖励截断。

| 项目 | 扶正 | 站起 | 行走 |
|---|---:|---:|---:|
| termination | -200 | -100 | -500 |
| orientation | -2.6 | -2 | -15 |
| inverted | -0.6 | -1 | 0 |
| height | 4 | 6 | 5 |
| linear_tracking | 0 | 0 | 7 |
| yaw_tracking | 0 | 0 | 4 |
| torque | 0 | -1e-6 | -1e-5 |
| acceleration | -2.5e-8 | -2.5e-6 | -2.5e-6 |
| leg_velocity | 0 | 0 | -0.004 |
| action_rate | -0.06 | -0.02 | -0.1 |
| symmetry | -0.02 | -0.02 | 0 |
| no_fly | -2 | 0 | -0.2 |
| wheel_load | 4 | 0 | 0 |
| wheel_distance | 0 | 0 | -2 |
| velocity_band | 0 | 0 | 1 |
| motor_limits | 0 | 0 | -5 |
| torque_limits | 0 | 0 | -0.05 |
| action_smoothness | 0 | 0 | -0.02 |
| power | 0 | 0 | -1e-4 |
| pose_bias | 已移除 | 已移除 | 已移除 |
| 被动knee_limit | 未启用 | 未启用 | 未启用 |

左右腿对称奖励只在扶正和站起阶段使用；行走的motor_limits是另一个针对实体电机关节的项，不能与被动knee_limit混同。

### 全部原始奖励公式与阈值

符号：h为root高度，ht为命令高度，g为机体系投影重力，xy=gx²+gy²，a为原始请求动作，tau为控制器电机力矩，M为随机化后总质量，F为双轮非负向上支撑。求和均遍历该项涉及的关节或动作。

| 项目 | raw_term |
|---|---|
| termination | reset_terminated，超时不包含在内 |
| height | exp(-abs(h-ht)/0.12)，绝对误差指数，不是误差平方高斯 |
| orientation | 0.02*xy + 2*(gz+1)² - clamp(exp(-xy/0.25),0,0.1) |
| inverted | 指示(-gz < -0.2) |
| symmetry | 左右腿相对默认角差的平方和；右腿先乘-1统一坐标符号 |
| no_fly | 指示没有同时满足双轮支撑各>1 N |
| wheel_load | exp(-abs(sum(F)-M*10)*0.05)，此项参考重量使用M*10而非M*9.81 |
| linear_tracking | exp(-((vx_cmd-vx)²+vy²)/0.12) * 指示(h>0.24 m) |
| yaw_tracking | exp(-(yaw_cmd-wz)²/0.12)，没有高度门控 |
| torque | sum(tau²) |
| acceleration | sum(((qdot-qdot_previous)/0.02)²) |
| leg_velocity | 四腿电机sum(qdot²) |
| action_rate | sum((a-a_previous)²) |
| action_smoothness | sum((a-2*a_previous+a_previous_previous)²) |
| wheel_distance | clamp(nominal_spacing-0.03-spacing,0,1)+clamp(spacing-nominal_spacing-0.03,0,1)；spacing为水平距离，nominal_spacing=0.4265059511 m |
| velocity_band | vx处于命令的0.8～1.1倍区间返回+1，否则-1；负命令反转区间不等号 |
| motor_limits | 四腿q超出default_joint_pos_limits中心±45%总区间后的超出量之和，软区间=90%硬区间；实际范围来自mjlab实体关节限制 |
| torque_limits | sum(max(abs(tau)-0.8*limit,0))，腿阈值32、轮3.12 N·m |
| power | sum(abs(qdot)*abs(tau))*指示(h<0.28 m) |

**实际门控：r15未使用历史r14的5～10 cm宽高度门控。只有行走linear_tracking在h>0.24 m时打开，yaw_tracking未加高度门控。**

**实际失败扣分：扶正/站起只因数值异常或20秒超时结束；触地/倾倒没有termination。数值异常又会把所有raw_term清零，超时不是reset_terminated，因此这两阶段的-200/-100不会对上述失败产生罚分。行走物理失败若触发，termination单项每步贡献-500×0.02=-10，数值异常仍清零。** 这描述当前实现，不代表已修正；本次不改代码。

全部公式见 [getup_curriculum.py](../../mjlab_training/src/ct_mjlab/experiments/getup_curriculum.py)。实际runtime RewardManager.compute已核实对包括termination在内的所有项乘dt。

## 9. 域随机化与扰动

| 项目 | 随机范围 / 时机 |
|---|---|
| 基座质量比例 | U(0.9,1.3)，启动采样 |
| 其他刚体质量比例 | U(0.9,1.1)，启动采样 |
| 刚体惯量 | 按对应质量比例同步缩放；没有额外独立惯量随机事件 |
| 基座COM偏移xyz | ±[0.04,0.02,0.02] m，启动采样 |
| 基座静/动摩擦抽样 | 各U(0.01,0.1)，64桶；有效MuJoCo滑动系数=min(静抽样,动抽样) |
| 轮静/动摩擦抽样 | U(0.5,1.2) / U(0.4,1.0)，64桶；有效滑动系数同上 |
| 恢复系数抽样 | U(0.02,0.2)，记录在桶中；没有作为MuJoCo反弹系数施加，接触solref设[0.004,1] |
| 腿电机附加dry / viscous | U(0.25,1) / U(0.05,0.2)，加到XML基值 |
| 轮附加dry / viscous | U(0.05,0.25) / U(0,0.01)，加到XML基值 |
| 其他被动关节附加dry / viscous | U(0.05,0.1) / U(0.01,0.025)，加到XML基值 |
| 气弹簧滑动关节 | 排除上述通用阻尼随机化，由气弹簧力模型处理 |
| Kp、Kd比例 | 各U(0.75,1.25) |
| 腿电机strength | U(0.8,1.1) |
| 轮strength | U(0.9,1.1) |
| 电机参数更新 | 只在reset且距上次≥720控制步时采样；14.4 s门槛，不是每14.4 s强制中途更新 |
| 动作延迟 | 每次reset，腿/轮独立抽取5/10 ms |
| 气弹簧bias | 每次reset，±50 N |
| 气弹簧damping | 电机参数更新时U(25,75) |
| 根节点速度推扰 | 每U(5,10)s，把世界vx/vy设置为各U(-0.25,0.25) m/s，不是增量 |
| 基座持续局部外力 | 每U(5,10)s，各xyz U(-10,10) N |
| 基座持续局部外力矩 | 同事件，各xyz U(-1,1) N·m |
| 外力持续规则 | 保持到下次采样或reset；每物理步由机体系旋转至世界系；reset清零 |

普通/地面比例、初速以及动作/观测延迟都是额外的随机来源。默认角没有±0.03偏移随机化；历史父类该路径被关闭。详见 [huananhu_randomization.py](../../mjlab_training/src/ct_mjlab/reference_training/huananhu_randomization.py)、[huananhu_action.py](../../mjlab_training/src/ct_mjlab/reference_training/huananhu_action.py)。

## 10. 观测、噪声、延迟与动态模式

### Actor 35维（索引从0开始，右端不含）

| 索引 | 内容 | 缩放 / 限幅 |
|---|---|---|
| 0:3 | vx命令、固定0的vy命令、yaw命令 | clamp ±100 |
| 3:4 | 高度命令 | clamp[0,1]，×5 |
| 4:7 | 机体系角速度 | clamp±100，×0.5 |
| 7:10 | 投影重力 | clamp±100，不另缩放 |
| 10:16 | 四腿相对名义角、两个轮位置 | clamp±100；原始轮位置为0，随后噪声仍包含这两槽 |
| 16:22 | 四腿电机速度、两轮速度 | clamp±200，×0.1 |
| 22:28 | 上拍原始请求动作，四腿两轮 | clamp±100，控制缩放前 |
| 28:35 | normal/stair/slope/recover/jump/height_target/state_time | 动态normal/recover，其余模式标志0 |

没有观测历史堆叠。actor读取延迟/带噪声的传感器；critic读当前无噪声无延迟的状态。

传感器原单位加U(-bound,bound)噪声，然后按上述缩放：角速度±0.25 rad/s，投影重力±0.05，六个相对关节位置槽±0.025 rad，四腿速度±0.5 rad/s，两轮速度±1 rad/s。命令、动作、模式槽不加这组噪声。

四组独立观测延迟：角速度、重力、位置、速度；每回合独立抽取5/10/15 ms，按最近物理步取整为3/5/8步，实际6/10/16 ms。每物理子步缓存状态和重采样噪声，重复读取观测不重复抽样。

### Critic 78维

| 索引 | 内容 |
|---|---|
| 0:28 | 与actor基本槽相同，但传感器当前、无噪声/延迟 |
| 28:31 | 机体系root COM线速度，clamp±100 |
| 31:32 | root高度，clamp±10后×5 |
| 32:39 | 动态模式7槽 |
| 39:45 / 45:51 | 六电机Kp / Kd，clamp±100 |
| 51:57 | 六电机实际actuator力矩，clamp±100后×0.05 |
| 57:61 | 四组传感器延迟，物理步数 |
| 61:63 | 腿/轮动作延迟，物理步数 |
| 63:69 | 双轮相对COM速度，机体系，侧向分量固定0，clamp±100 |
| 69:71 | 双轮接触历史峰值norm>5 N标志 |
| 71:72 | 基座质量比，clamp±10 |
| 72:78 | 两轮原生sliding/torsional/rolling摩擦，clamp±100 |

### normal / recover个体标志

- 地面初态从recover开始；普通初态从normal开始。
- 进入recover：机体接触>1 N，或|roll|/|pitch|>40°，或高度<0.16 m，连续0.1 s。
- 返回normal：高度误差≤3 cm、|roll|/|pitch|各≤15°、双轮接触历史峰值norm各>5 N、机体≤1 N、有效且未物理失败，连续0.5 s。
- recover时index31=1、index33=高度命令×5、index34=恢复状态秒数；normal时index28=1，目标/时间槽0。
- 这是策略观测状态机；不直接决定整批奖励阶段。

## 11. 终止与训练诊断

| 数值异常条件 | 阈值 |
|---|---|
| qpos / qvel | 任意非有限值 |
| 任一六电机速度绝对值 | >500 rad/s |
| 任一机体系线速度绝对值 | >100 m/s |
| 任一机体系角速度绝对值 | >200 rad/s |

行走物理失败：机体接触力norm>1 N或roll/pitch任一>40°，连续20控制步=0.4 s；扶正/站起会更新这些计数和诊断，但不据此提前终止。20秒超时time_out=true，有限时域开关false，算法按超时语义处理。

训练strict_recovered：回合内曾连续0.5秒满足高度±3cm、roll/pitch各≤15°、双轮接触历史峰值norm>5N、无机体接触/数值/延迟物理失败；成功标志保持至reset。strict_stand_fraction记录全回合达标时间比例。整批课程晋级使用当前双轮向上支撑和总倾角，条件见第7节。无辅助力回放的最后5秒持续站稳是另一个评估指标，不能互换。

## 12. 参数来源与本次核对

- 冻结源码：`/root/gpufree-data/wheelleg-remote/releases/20261011T083634Z-getup-r15-pose-bias-removed`。
- 远端运行时：`/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python`；Python3.12、mjlab1.6、Torch2.14cu130、MuJoCo3.11、RTX4090。
- 从头run：`20261011_164219_836198_getup_r15_pose_bias_removed_scratch_2000`；续训run：`20261011_164219_970332_getup_r15_pose_bias_removed_resume_4000_to_5000`，均在远端 `logs/mjlab/chuanliantui/`。
- 原始配置来自两个run保存的JSON；启动状态来自已有恢复验证。额外读取安装版RewardManager/PPO实现及静态MuJoCo模型，未创建训练环境、未进行策略更新。
- 本次未改变任何训练参数，未启动新训练或TensorBoard。完整资产与姿态库保留原路径，未删除或覆盖历史logs。
