# 常用命令速查

先按 [协作指南](CONTRIBUTING.md) 创建并填写 `.env.local`。在仓库根目录执行以下准备，完整激活对应 Python 3.8 conda 环境（`conda` 命令需已在 PATH 中）：

```bash
source ./.env.local
: "${WHEELLEGGED_PYTHON:?请配置装有 Isaac Gym 的 Python 3.8 环境}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$(dirname "$(dirname "$WHEELLEGGED_PYTHON")")"
command -v python
command -v "${CXX:-c++}"
```

确认 `python` 指向 `WHEELLEGGED_PYTHON`，下面的 `python` 和 `tensorboard` 均指该环境中的程序，不能使用系统 Python。本机路径仅保存在 Git 忽略的 `.env.local` 中。

仅执行 `source .env.local`、指定 Python 绝对路径或前置环境的 `bin/`，都不会执行
Conda 的激活脚本。Isaac Gym 的 `gymtorch` 在导入时可能编译 C++ 扩展，
依赖激活脚本设置的 `CXX` 和动态库路径。若报 `which c++` 返回非零，
先按上面完整激活环境并检查编译器；已安装 Conda 编译器时无需再安装系统编译器。
保留环境激活后生成的 `LD_LIBRARY_PATH`。

Sim2Sim 使用同一环境内固定的 `mujoco==3.2.2`；重建环境时执行：

```bash
python -m pip install --upgrade --upgrade-strategy only-if-needed 'mujoco==3.2.2'
```

这不替代 Isaac Gym：仍必须使用 Python 3.8，且 `import isaacgym` 必须先于 `import torch`。

## 训练（Isaac Gym）

```bash
# 从头训练。--headless 不开渲染窗口，速度快得多
python wheel_legged_gym/scripts/train.py --task=mini_wheel_legged --headless

# 从某次 run 的最新 checkpoint 续训
python wheel_legged_gym/scripts/train.py --task=mini_wheel_legged --headless \
    --resume --load_run Jul20_12-42-47_

# 常用可选参数
#   --max_iterations 2000     总迭代数（覆盖 config）
#   --num_envs 4096           并行环境数（显存不够就调小）
#   --checkpoint 6900         配合 --resume，指定从 model_6900.pt 续训（默认最新）
#   --run_name xxx            给本次 run 起名，方便区分日志目录
```

日志和模型存在 `logs/mini_wheel_legged/<日期时间_run_name>/model_*.pt`。

在另一个启用相同环境的终端监控：

```bash
tensorboard --logdir=logs/mini_wheel_legged --port=8080
```

## 回放（Isaac Gym 里看策略效果）

```bash
# 加载最新 run 的最新模型，开窗口回放
python wheel_legged_gym/scripts/play.py --task=mini_wheel_legged

# 指定 run 和 checkpoint
python wheel_legged_gym/scripts/play.py --task=mini_wheel_legged \
    --load_run Jul20_12-42-47_ --checkpoint 6900
```

训练和 sim2sim 之间先用 play 确认策略在 Isaac 里本身是好的，排除"训练没练好"和"sim2sim 有 gap"两种问题的混淆。

## Sim2Sim（MuJoCo 部署验证）

用完整 checkpoint `model_*.pt`（含 encoder），不能用导出的 `policy_1.pt`。

```bash
# 第一步：策略形状自检（不跑仿真，几秒钟）
python sim2sim/mj_sim2sim.py --selfcheck \
    --checkpoint logs/mini_wheel_legged/Jul21_14-09-24_/model_4000.pt

# 第二步：站立测试（vx=0，只保持平衡和高度）
python sim2sim/mj_sim2sim.py --render \
    --checkpoint logs/mini_wheel_legged/Jul21_14-09-24_/model_4000.pt \
    --cmd_vx 0 --cmd_height 0.30 --init_height 0.30

# 第三步：行走测试（默认先站 3 秒再 1 秒爬升到目标速度）
python sim2sim/mj_sim2sim.py --render \
    --checkpoint logs/mini_wheel_legged/Jul21_14-09-24_/model_4000.pt \
    --cmd_vx 1.0 --init_height 0.30

# 常用可选参数
#   --cmd_yaw 0.5        目标航向角 [rad]（heading 外环，默认开）
#   --cmd_height 0.20~0.40   目标机身高度（训练命令范围）
#   --sim_time 20        仿真时长 [s]
#   --no_realtime        不按真实时间节流，全速跑（无渲染批量测试用）
#   --no_hold            结束后不保持窗口
#   --friction 0.75      覆盖轮地滑动摩擦（训练等效均值约 0.75）

# 键盘遥操作（先点击 MuJoCo 窗口获得焦点；字母键是 viewer 内置渲染快捷键，勿用）
#   ↑/↓ 加减速 0.1 m/s（限幅 ±1.5 = 当前策略安全包线，更高速会瞬态发散翻车）
#   ←/→ 左右转向 | PgUp/PgDn 升降高度 | 空格/回车 急停
#   Home 摔倒后复位（回初始位姿，速度/航向清零） | 关窗退出
python sim2sim/mj_sim2sim.py --render --teleop \
    --checkpoint logs/mini_wheel_legged/Jul21_14-09-24_/model_4000.pt --init_height 0.30
```

### chuanliantui

#### 当前推荐：加入每侧 150 N 气弹簧后续训

`chuanliantui` 与 `chuanliantui_standup` 现在默认每侧恒定伸张力 150 N。
安装点沿用 CAD 闭链，每个 2 ms 子步按当前膝角计算力矩，以上下连杆等大反向
扭矩施加，独立于电机限幅、动作延迟和推扰门控。串联 MuJoCo 代理同步相同弹簧。
这是支撑力的对齐，不表示后支路惯量、闭链反馈或真机动力学全部对齐。

旧 25 维 checkpoint 可加载，但旧训练没有弹簧，需要续训；以下保留 0.22 m 站高、
0.05 rad 膝余量奖励、随机推力及 0–10 ms 动作延迟，从 12000 轮模型新增 3000 轮。
初始位置未改，可直接 headless；默认只交付命令，不自动启动训练。

```bash
python wheel_legged_gym/scripts/train.py --task=chuanliantui_standup --headless \
    --resume --load_run Oct03_14-27-59_standup_h022_knee005_push --checkpoint 12000 \
    --max_iterations 3000 --run_name standup_h022_gas150_push
```

另一终端加载本文件开头相同环境后：

```bash
tensorboard --logdir=logs/chuanliantui_standup --port=8080
```

数值及施力合成检查：`python scripts/agent/check_chuanliantui_gas_spring.py`。
回放新模型时，`eval_isaac.py` 与 `mj_sim2sim_ct.py` 均默认每侧 150 N；
复现原无弹簧串联训练条件，两者显式传 `--gas_spring_force 0`。
训练端对应 `ChuanliantuiCfg.gas_spring.force_n`，不随 `--domain_rand` 开关改变。
旧起立配置快照仍会继承当前父类弹簧配置，历史对照需显式关闭。
详见[气弹簧物理条件记录](docs/interfaces/chuanliantui-gas-spring-training.md)。

#### 前一轮：0.22 m 站高与 0.05 rad 膝余量续训

本节保留前一轮命令；在当前代码执行同样命令也会启用上面新增的气弹簧。

本轮用户目标仅为起立和原地站稳，线速度与偏航角速度命令均为零。验收关注站高、
pitch、零速位移与转动、膝余量及受扰恢复；行走能力暂不纳入本轮验收。

当前 `chuanliantui_standup` 解锁后目标高度为 **0.22 m**，连续站稳的高度门槛为
**0.20 m**；膝端点奖励区宽度为 **0.05 rad**。在当前模型几何下，近零 pitch 且两膝
保留该余量时最低站高约为 0.218 m，不能同时要求 0.20 m 站高、近零 pitch 和该膝余量。
奖励区不改变物理关节 `range`，也不裁剪 PD 位置目标；本次不修改 H7。
奖励公式与 MuJoCo 膝限位设置见 [sim2sim 说明](docs/ai/sim2sim.md#chuanliantui-站高精度与膝余量续训)。

先检查实际奖励缩放、单项裁剪、膝硬限位与起立门控：

```bash
python scripts/agent/check_standup_precision_rewards.py
```

这是不创建仿真的 CPU 检查，不能代替训练后的 Isaac 回放和 MuJoCo 行为验收。
以下从已有的 25 维 `Oct03_03-28-58_standup_action_delay_0_10ms/model_9000.pt`
续训**额外 3000 轮**，输出到新的时间戳 run，保留原模型和日志：

当前同时启用随机水平推力：首次轮接地后开始计时，每隔 3–5 s 施加 0.1 s 的恒力；
尚未站稳时合力模长 1–5 N，当前连续站稳 0.5 s 时为 5–15 N，方向在水平面均匀采样。
间隔指上一脉冲结束到下一脉冲开始的空档；强度在脉冲开始时选定，脉冲内保持不变。
力作用于 `base_link` 质心，初始自由下落阶段不推；重置清空本回合推力。
0–10 ms 动作延迟、零速度命令和以上站高/膝余量目标同时保留。

```bash
python wheel_legged_gym/scripts/train.py --task=chuanliantui_standup --headless \
    --resume --load_run Oct03_03-28-58_standup_action_delay_0_10ms --checkpoint 9000 \
    --max_iterations 3000 --run_name standup_h022_knee005_push
```

另一个相同环境的终端：

```bash
tensorboard --logdir=logs/chuanliantui_standup --port=8080
```

此次初始位置不变，可直接 headless。旧 25 维模型接口仍可加载，但需要续训学习新目标与
奖励；仅把回放命令改成 0.22 m 不等于已经学会。以上仅交付手动命令，未自动启动训练，
也没有据此宣称新策略训练成功或真机问题已修复。

推力调度 CPU 检查：`python scripts/agent/check_standup_pushes.py`。
默认 `play.py` / `eval_isaac.py` 关闭推扰；单独验收推扰可用：

```bash
python sim2sim/eval_isaac.py --task chuanliantui_standup \
    --load_run <新训练run> --checkpoint <新checkpoint编号> \
    --cmd_vx 0 --cmd_yaw 0 --cmd_height 0.22 --standup_post_unlock \
    --pushes --action_delay_ms 6 --sim_time 20 --metrics_report /tmp/<新推扰报告>.json
```

加入 `--domain_rand` 可同时保留其余训练随机化。报告记录推力、持续时长和回合内次数；
应结合重置、推后高度/pitch/零速恢复判断。`recovered_rate` 只表示该回合曾站起，
不能直接当成抗推成功率。随机推力训练也不能消除训练控制与 H7 轮目标裁剪的差异。

#### 动作延迟随机化续训（历史示例）

当前 `chuanliantui` 与 `chuanliantui_standup` 开启 0–10 ms 动作目标延迟，
按 2 ms 内环量化；每个环境初始化时采样一次。先做队列边界与重置检查：

```bash
python scripts/agent/check_action_delay.py
```

此检查不创建仿真，不替代训练、Isaac 回放和 MuJoCo 行为验收。
以下保留原延迟续训命令；复现历史实验须使用对应代码和配置。在当前代码执行它也会
使用当前站高与膝余量奖励；当前推荐使用上面的 9000 轮 checkpoint 续训示例。
下面以已有 `Sep22_16-26-11_standup_no_legangle_resume/model_6000.pt` 为续训基点；
若实际部署使用其他 checkpoint，替换 `--load_run` 和 `--checkpoint`。
`--max_iterations 3000` 表示额外训练 3000 轮；输出为带新时间戳的独立 run，
保留原模型做对照。此次不修改初始位置，可直接 headless；命令仅供手动执行。

```bash
python wheel_legged_gym/scripts/train.py --task=chuanliantui_standup --headless \
    --resume --load_run Sep22_16-26-11_standup_no_legangle_resume --checkpoint 6000 \
    --max_iterations 3000 --run_name standup_action_delay_0_10ms
```

另一个相同环境的终端：

```bash
tensorboard --logdir=logs/chuanliantui_standup --port=8080
```

普通 `play.py` 和默认 `eval_isaac.py` 会关闭动作延迟；
`eval_isaac.py --domain_rand` 才保留配置中的延迟及其他域随机化项，不能作为单独的延迟消融。
单独固定延迟可用 `--action_delay_ms 0`、`6` 或 `10`；配合 `--standup_post_unlock --metrics_report /tmp/<新报告>.json`，统一起立判据并保存逐步轨迹和 5 秒之后的指标。

#### 回放与诊断

零速度诊断可使用现有 Isaac 对照脚本，显式选择起立任务和同一 checkpoint。
下例用旧 9000 轮模型检查新命令下的基线；验收续训结果时替换为新 run 与 checkpoint：

```bash
python sim2sim/eval_isaac.py --task chuanliantui_standup \
    --load_run Oct03_03-28-58_standup_action_delay_0_10ms --checkpoint 9000 \
    --cmd_vx 0 --cmd_yaw 0 --cmd_height 0.22 --sim_time 15
```

该脚本按 chuanliantui 的 +x 前向记录真实速度和 encoder 估速，固定起立课程高度，
并区分超时重置与其他重置。应同时检查位移、高度、姿态及 `has_stood`；
没有重置不代表已经起立或保持原地稳定。它是诊断对照，不替代可视行为验收。

##### 30 秒静站与共同推扰计划

通过接口自检后，先用同一 12000 轮模型做 **30 秒无推扰** 对照：零速度、0.22 m、
每侧气弹簧 150 N、接触摩擦 0.5，动作延迟均为零。下列旧模型尚未学习新增气弹簧条件。

```bash
python sim2sim/eval_isaac.py --task chuanliantui_standup \
    --load_run Oct03_14-27-59_standup_h022_knee005_push --checkpoint 12000 \
    --cmd_vx 0 --cmd_yaw 0 --cmd_height 0.22 --standup_post_unlock \
    --gas_spring_force 150 --contact_friction 0.5 --action_delay_ms 0 \
    --sim_time 30 --metrics_report /tmp/isaac-standing-30s.json

python sim2sim/mj_sim2sim_ct.py --closed_chain --standup --h7_no_wheel_target_clip \
    --checkpoint logs/chuanliantui_standup/Oct03_14-27-59_standup_h022_knee005_push/model_12000.pt \
    --cmd_vx 0 --cmd_yaw 0 --cmd_height 0.22 --gas_spring_force 150 --friction 0.5 \
    --sim_time 30 --no_realtime --diagnostic_report /tmp/mujoco-standing-30s.json
```

`--h7_no_wheel_target_clip` 显式关闭 H7 回放的轮目标速度裁剪，保留电机力矩限幅；
只允许 H7 闭链路径，不能与 `--wheel_vel_limit` 合用。不加该开关时仍默认 ±20 rad/s。
Isaac 有限时长评估会把回合期限延长到至少 `sim_time + 1` 秒，保留失稳终止，
避免训练的 20 秒超时打断 30 秒观察；这不修改训练回合长度。

再生成一次固定计划，两端共用同一个 JSON：

```bash
python scripts/agent/check_standing_push_schedule.py
python sim2sim/standing_push_schedule.py --duration_s 30 --seed 42 \
    --output /tmp/standing-pushes.json
```

推扰阶段重复上面两条评估命令，均追加 `--push_schedule /tmp/standing-pushes.json`，
并把报告改为新的文件名，例如 `/tmp/isaac-pushed-30s.json`、`/tmp/mujoco-pushed-30s.json`。
计划和报告均拒绝覆盖已有文件。Isaac 的固定计划不能与 `--pushes` 或 `--domain_rand` 合用。
默认首次推力在 5 秒，持续 0.1 秒，水平幅值 5–15 N；以后每次结束后留 3–5 秒空档。
两端均每 2 ms 向 `base_link` 质心施加同一世界系力，使用整次评估时钟，重置不重播计划。

Isaac 记录推理时刻的 `policy_t_s`、真速度、encoder 估速及误差；MuJoCo 同时记录
`velocity_pos_diff` 与 `estimated_velocity`，比较时只取策略已激活的同一时刻。
两端均以 100 Hz 保存轨迹，但 MuJoCo 的推力字段取策略步起点，Isaac 取末个物理子步，
**不能用这些采样行积分核对冲量**；应比较 metadata 的 `scheduled_push_impulse_world_ns`
与计划的 `sum(force_world_n * duration_s)`，容许 float32 舍入误差。
若无推扰时已经持续漂移，后续只能报告推后是否恢复原运动状态，不能称为恢复原地稳站。

chuanliantui 的 actor 观测已改为 25 维、历史为 125；轮绝对位置不再输入策略。
此前所有 27 维 chuanliantui `model_*.pt` 与当前网络接口不兼容，以下回放命令均应替换为
新训练产生的 25 维 checkpoint。
训练与 MuJoCo 回放采用 500 Hz 物理/PD 内环和 100 Hz 策略推理（每 5 个内环步更新一次动作）。

`mj_sim2sim_ct.py` 默认使用 `sim2sim/chuanliantui_train_proxy.xml`：它由 Isaac Gym
使用的 `chuanliantui_train.urdf` 生成，含相同的 6 个训练 DOF、固定后支链和对应的
六个力矩电机，**没有** `connect` 闭链约束。该路径只做串联训练代理一致性回放，
不是实体闭链或真机验证。

```bash
# 改 chuanliantui_train.urdf 后重新生成并检查串联代理。
python scripts/agent/generate_chuanliantui_train_proxy_mjcf.py
python scripts/agent/check_chuanliantui_train_proxy.py
python scripts/agent/check_chuanliantui_observation.py

# 策略接口自检：只加载网络并验证 actor/encoder 输出形状，不运行 MuJoCo。
# --selfcheck：启用接口检查模式。
# --checkpoint：完整 model_*.pt 路径；不能使用缺 encoder 的 policy_1.pt。
python sim2sim/mj_sim2sim_ct.py --selfcheck \
    --checkpoint logs/chuanliantui/<new_25d_run>/model_<checkpoint>.pt

# 训练一致的地面后摆起立回放。--standup 固定使用基座 0.15 m 和关节
# [-1.566,0,0,1.566,0,0]；首次轮触地后下一策略步接管。
# 点击 MuJoCo 窗口后按 Ctrl+R，可恢复后摆初态并清空速度/航向及观测历史。
python sim2sim/mj_sim2sim_ct.py --standup --render \
    --checkpoint logs/chuanliantui_standup/<run>/model_<checkpoint>.pt \
    --cmd_vx 0 --cmd_height 0.22 --friction 0.75

# CAD 闭链机构 + H7 五连杆状态解算、PD、限幅、十拍预热。
# 0.22 m 是本轮站立测试命令覆盖；H7 源码当前默认 0.20 m。
python sim2sim/mj_sim2sim_ct.py --closed_chain --standup --render \
    --checkpoint logs/chuanliantui_standup/<run>/model_<checkpoint>.pt \
    --cmd_vx 0 --cmd_height 0.22 --friction 0.75

# 串联训练代理站立一致性回放。
# --render：打开 MuJoCo viewer 并启用方向键遥操作；命令参数为初始值，关闭窗口退出。
# --checkpoint：要验证的完整策略权重；--cmd_vx：前向速度命令，0 表示原地站立。
# --cmd_height：策略观测中的目标机身高度；--init_height：复位时基座初始高度。
python sim2sim/mj_sim2sim_ct.py --render \
    --checkpoint logs/chuanliantui/<new_25d_run>/model_<checkpoint>.pt \
    --cmd_vx 0 --cmd_height 0.32 --init_height 0.33

# 串联训练代理行走一致性回放。
# --render：打开 viewer 并启用方向键遥操作；命令参数为初始值，关闭窗口退出。
# --checkpoint：完整策略权重；--cmd_vx 1.0：请求 1.0 m/s 前向行走。
# --cmd_height 0.32：策略观测的目标高度；--init_height 0.33：初始基座高度。
python sim2sim/mj_sim2sim_ct.py --render \
    --checkpoint logs/chuanliantui/<new_25d_run>/model_<checkpoint>.pt \
    --cmd_vx 1.0 --cmd_height 0.32 --init_height 0.33

# 若另行修改初始位置，先不带 --headless、--num_envs 20 观察新环境；
# 此次仅调整目标与奖励，初始位置不变，可使用上面的 headless 续训命令。
# --task=chuanliantui_standup：选择 0.15 m 地面后摆起立任务；--num_envs 20：只创建 20 个并行环境，便于人工观察。
python wheel_legged_gym/scripts/train.py --task=chuanliantui_standup --num_envs 20

# 另一个启用相同环境的终端监控训练输出。
tensorboard --logdir=logs/chuanliantui_standup --port=8080

# 新起立权重的串联训练代理 MuJoCo 回放：将 <run> 和 <checkpoint> 替换为新训练输出。
# --standup：使用与当前 Isaac 训练一致的 0.15 m 地面后摆初态；首次轮接地后下一控制步才推理策略。
# --render：打开 MuJoCo viewer 并启用方向键遥操作；命令参数为初始值，关闭窗口退出。
# --checkpoint：新训练生成的完整 model_*.pt；--cmd_vx 0：起立阶段不请求前进。
# --cmd_height 0.22：与当前起立训练解锁后的目标高度一致。
python sim2sim/mj_sim2sim_ct.py --standup --render \
    --checkpoint logs/chuanliantui_standup/<run>/model_<checkpoint>.pt \
    --cmd_vx 0 --cmd_height 0.22
```

`--standup` 使用当前任务的 0.15 m 地面后摆初态，只能搭配完整 `model_*.pt`。
串联代理与 `--closed_chain_controller cad` 保留首次轮接地后的下一控制步接管。
`--closed_chain` 默认使用 H7 控制路径：模拟 RL 已投入，前十个有效策略步保持零动作 PD，
随后推理；不使用首次接地门控，无效观测关闭电机输出并重新预热。
原 CAD 解算对照需加 `--closed_chain_controller cad`。闭链后轴角默认以 CAD 杆方向注册，
有共同姿态标定后可传 `--h7_rear_zero <LF> <RF>`（rad）；它不是编码器 `dm_zero`。
两条路径的定义及验证边界见 [H7 闭链回放接口](docs/interfaces/chuanliantui-h7-closed-replay.md)。

渲染模式关闭窗口退出；无渲染可指定 `--sim_time 15 --no_realtime`。
日志 `pg_xyz` 中，前后倾看 `pg_x`，直立时 `pg_z` 接近 -1。
未传 `--closed_chain` 时仍使用串联训练代理。复旦模型仅作独立参考。

## 典型工作流

1. 改 config → `train.py --headless` 训练
2. `play.py` 在 Isaac 里确认能站能走
3. `mj_sim2sim.py` 站立 → 行走，验证跨引擎迁移
4. MuJoCo 里也稳了，再考虑上真机（参照 `sim2sim/mj_sim2sim.py` 头部的部署契约注释）
