# 构建与测试

串联腿新任务使用独立 Python 3.12 / mjlab 环境，入口、有限检查及待训练行为验收见 [mjlab 串联腿迁移](../mjlab-chuanliantui.md)。下文 Python 3.8 / Isaac Gym 约定适用于旧任务；新包不得安装进旧环境。

## mjlab 检查

H7→Jetson只读镜像的软件检查（不训练、不连接或控制真机）：

```bash
# Jetson（~/real2sim，需cc）：实际C打包代码＋损坏流＋模型姿态＋伪串口断线检查
.venv/bin/python scripts/agent/check_real2sim_huananhu.py --out artifacts/<新名称>/check-software.json
# 训练仓：从当前eIDE实际存在的来源编译、链接大小机；需要完整ARM GCC/Newlib
ARM_CC=/usr/bin/arm-none-eabi-gcc mjlab_training/.venv/bin/python \
  scripts/agent/check_real2sim_huananhu.py --build-only --out artifacts/<新名称>/check-h7-build.json
```

镜像不依赖PyTorch，单独依赖见`sim2sim/requirements-real2sim.txt`。`--build-h7`可在同时具备宿主C编译器
和ARM工具链的训练主机合并执行；构建产物全部写到报告目录，不污染eIDE增量构建。过时且不存在的
虚拟来源列入报告，不修改工程清单。检查覆盖CRC／伪尾／分包、序号回绕、DMA忙、单设备失效、200ms
断流，以及模型闭链重构；不验证实际串口、物理极性／零点／IMU轴向或装配分支。操作见
[部署接口](../deployment-contract.md#host-debug-plan)。

倒地恢复机制检查（不训练）：训练配置不主动施加绕轴击倒，手动冲量只在机制检查中使用。

```bash
# 未注册主动击倒事件，恢复窗口仍存在；12秒有限步进
mjlab_training/.venv/bin/python scripts/agent/check_multimode_recovery.py --training-config
# 手动击倒后恢复通道重启、恢复期接触奖励门控与窗口内不提前终止
mjlab_training/.venv/bin/python scripts/agent/check_multimode_recovery.py
```

最终平衡 v3 的判据检查及既有轨迹离线重算（不运行物理、不重跑策略、不改旧产物）：

```bash
# NumPy / Torch 数值路径，覆盖倾角、承重、接触、角速度、时长及提前终止
mjlab_training/.venv/bin/python scripts/agent/check_multimode_final_balance.py
# 仅重算已录下的 7 个样本；输出目录须使用新名称
mjlab_training/.venv/bin/python scripts/agent/check_multimode_final_balance.py \
  --recording-dir logs/mjlab/evaluation/kd-failure-videos-4850 \
  --output artifacts/<新重算名>/result.json
```

新判据看结束前连续 1 秒平衡，允许滚动、转向与少量腿部接触，仍检查双轮主要承重、机身不撑地与横滚/俯仰稳定；历史 `last5_*` 保留原义。离线重算仅覆盖已录制样本，不替代完整模型配对、独立 sim2sim 或真机验收，标准见[当前验收 v3](../interfaces/chuanliantui-wheel-balance-acceptance-v2.md)。

在仓库根运行，不启动训练或 TensorBoard，不写根 `logs/`：

```bash
# 有限检查：加载 checkpoint，核对 actor 39／critic 82／动作 6，数值有限
scripts/mjlab.sh check --num-envs 2 --steps 16 \
  --checkpoint artifacts/multimode-v1-replay-20261014/models/model_3023.pt
# 基线回放：64 个强制地面初态、20 秒
scripts/mjlab.sh eval --checkpoint artifacts/multimode-v1-replay-20261014/models/model_3023.pt \
  --num-envs 64 --seeds 42 --start-mode force --terrain flat --output /tmp/mjlab-eval/result.json
```

平地速度／角度跟踪测量与判据检查（不训练；口径见[跟踪测量](../interfaces/chuanliantui-huananhu-multimode-tracking-v1.md)）：

```bash
# 判据算术、边界与容差一致性的离线检查（不建环境、不加载模型、不跑物理）
mjlab_training/.venv/bin/python scripts/agent/check_multimode_tracking.py
# 平地固定命令网格的跟踪测量（play 配置、每环境单回合、12 s、前 2 s 为稳定窗口）
# 脚本每步断言 actor 命令槽等于网格：中途抛 AssertionError 表示命令未被钉住，不是策略结论
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_tracking.py \
  --checkpoint <model_*.pt> --output artifacts/<新名称>/result.json
```

静止站立腿角运动测量与 CPU 入口一致性检查（不训练；2026-10-17 站立腿摆修复的证据工具）：

```bash
# 入口一致性与恢复高度检查（静态 + 有限步力矩／接触／模式探针，不做长时行为验收；不一致即非零退出）
mjlab_training/.venv/bin/python scripts/agent/check_multimode_cpu_sim2sim.py \
  --out artifacts/<新名称>/check-cpu-entry.json
# 静止站立腿角运动测量（50 Hz 控制拍；诊断性测量，不设通过阈值）
# --backend cpu 用独立 CPU sim2sim 入口；--backend mjlab 用训练后端（play 配置、GPU），
# --reset stance 复刻 CPU 入口初态，--reset bank 用训练姿态库初态
mjlab_training/.venv/bin/python scripts/agent/measure_multimode_leg_motion.py \
  --backend cpu --checkpoint <model_*.pt> --output artifacts/<新名称>/leg-motion-cpu.json
mjlab_training/.venv/bin/python scripts/agent/measure_multimode_leg_motion.py \
  --backend mjlab --checkpoint <model_*.pt> --device cuda:0 --output artifacts/<新名称>/leg-motion-mjlab.json
```

一致性检查覆盖：入口与训练模型 qpos0 及闭链站立解一致；观测腿槽与腿 PD 目标以 `q_nominal` 为中心、
限幅以 `qpos0` 为中心（q＝nominal、零动作时腿力矩为零）；腿增益 0.5／60／2、轮增益 0.1／0、
轮速 `clip(10a, ±150)`、力矩限幅 40／3.9；气弹簧逐点一致；动作顺序与观测排列、NORMAL 模式／标志、
报告容差与跟踪测量脚本一致；机器人 geom 优先级（训练场景 `priority=1` 补丁）与机器人-地面接触摩擦等于训练值。腿角运动测量记录六电机 q／qd、原始动作、观测腿槽与机身状态，并给出
腿摆峰峰值、主导频率与观测偏置；它只覆盖单模型、单命令、单次运行的读数，不是验收。

2026-10-18新增CPU恢复高度检查：使用两腿机体系 `L*cos(theta)` 均值，覆盖整机平移／旋转不变性、
左右腿不对称，以及抬高机身但保持折腿时不误判恢复。该高度条件有意采用H7量的定义，
不再与训练的世界坐标高度相同；`height_m` 与原跟踪／静站判据保留原义。
检查通过不表示H7腿角标定已验证，也不表示完整H7恢复时序或真实恢复行为验收通过。

恢复高度改动前后的CPU配对诊断（相同模型、初态和命令；默认7组normal命令与16个固定seed地面初态，
每侧20秒；`--before`必须是改动前入口的完整源码快照）：

```bash
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_recovery_height.py \
  --before <改动前入口.py> --checkpoint <model_*.pt> \
  --output artifacts/<新名称>/result.json
```

输出逐样本模式切换时间、动作／位置／速度的最大差、末段连续1秒双轮平衡及原成功样本是否丢失，
同时保存初态与逐步轨迹。末段平衡使用v3.1门槛并读取原生MuJoCo接触力；该初态分布和运行流程
不等于训练侧完整验收，也不代替速度／高度跟踪测量。当前实测结论见[项目实验记录](project-log.md)。

包线边界与引擎差异的审计跟进测量（不训练；2026-10-17 接触参数审计的证据工具）：

```bash
# 训练引擎（warp）在训练接触材料随机化下的包线复测；--material randomized 复原训练抽签
# （轮滑动 min(U(.5,1.2), U(.4,1.))、基座 U(.01,.1)、训练 solref），solref/xml 为对照，
# --material pinned --wheel-slides 为逐环境固定轮摩擦的阈值扫描；--envs-per-combo N 即起点分布扫描
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_envelope_friction.py \
  --checkpoint <model_*.pt> --output artifacts/<新名称>/grid-randomized.json \
  --material randomized --envs-per-combo 8
# CPU 入口包线诊断：--mode starts 训练姿态库 normal 分支起点分布（--friction training 叠加
# 训练接触抽签）；--mode sweep 轮滑动摩擦扫描；--mode trace 与训练引擎的 50 Hz 轨迹分叉
# （--twin-perturb EPS 额外测入口自身对初值扰动的敏感度）
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_cpu_envelope.py \
  --mode starts --checkpoint <model_*.pt> --starts 64 \
  --output artifacts/<新名称>/cpu-starts.json
```

迁移一致性检查（用冻结快照；`--snapshot` 缺省时逐候选目录核对 23 个模块哈希后自动选定；
`--gpu` 会加载并完整续训读取 model_3023，但不调用 learn）：

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_mjlab_multimode.py \
  --gpu --out /tmp/mjlab-multimode-check.json
```

它检查：每个模块等于「冻结快照 + 已记录偏离」逐字节重构的结果（未记录的差异一律失败），偏离表在
`scripts/agent/multimode_snapshot_deviations.json`（源偏离 6 个模块；配置偏离 train 0、play=False 8、
play=True 5，均来自 2026-10-15/16 恢复工作与 2026-10-17 平地跟踪阶段）；XML 与姿态库哈希及缓存签名；
环境和 PPO 配置与快照工厂的输出在该偏离表之外完全相同；没有代码再引用已删除的模块；另外检查续训
课程进度 7525、保存后读回参数不变、导出用 actor 与 runner 策略输出一致、39/82 维、有限步进。

`eval` 在**平地协议**下的新参考基线：`model_3023` 与 `model_4850` 签名同为 `4a0b867588ff…`，
严格起立 64/64、存活 64/64、末态双轮承重 64/64、数值失败 0（`logs/mjlab/evaluation/model3023-flat-gpu64/`、`model4850-flat-gpu64/`）；
平地跟踪续训产物 `model_5849`（SHA256 `d4be0bc9…`）在同一套初态上严格起立 63/64、存活 63/64（`env 27` 在 2.64 s 物理终止），产物在 `logs/mjlab/evaluation/model5849-flat-gpu64/`。历史混合地形基线 `458f15a3…`（平子集 31/64）只作历史记录，不能与新签名配对。以上不等于行为验收，
不替代双轮平衡验收 v3、平地跟踪测量或真机验证。

## 环境

项目源码为纯 Python 包，`pip install -e .` 一次即可；Isaac Gym 依赖的
`gymtorch` 在导入时可能通过 PyTorch JIT 编译 C++ 扩展，需要可用的主机 C++ 编译器。

本仓库必须用**装了 Isaac Gym 的 Python 3.8 conda 环境**，不能用系统 python。本机该环境的路径见 [COMMANDS.md](../../COMMANDS.md) 开头，下文 `python` 均指它。

按 `COMMANDS.md` 完整执行 `conda activate`，使编译器 `CXX` 和动态库路径的激活脚本生效。
只指定环境内的 Python 绝对路径不等于激活环境；`which c++` 返回非零时先核对这一点。

已验证可用的版本组合：Python 3.8.20 / torch 2.1.0（CUDA 可用）/ mujoco 3.2.2 / Isaac Gym Preview 4。

MuJoCo 固定版本安装命令（保留现有满足依赖的 NumPy/Torch）：

```bash
python -m pip install --upgrade --upgrade-strategy only-if-needed 'mujoco==3.2.2'
```

自检一句话：

```bash
<你的环境>/bin/python -c "import isaacgym; import torch; import mujoco; print(mujoco.__version__)"
```

`import isaacgym` 必须在 `import torch` 之前，否则报错。

## 正式流程

策略、控制和机器人资产改动按对应任务执行训练、Isaac 回放、MuJoCo 行为验证，区分“策略未训练好”和“sim2sim 有差异”。以下为 imcawl 示例，chuanliantui 使用 [COMMANDS.md](../../COMMANDS.md) 的对应命令。纯文档或本机路径示例修改按 [协作指南](../../CONTRIBUTING.md) 检查差异、链接和配置语法，并明确未运行行为验证。

每项结果记录实际代码 SHA、工作区差异、模型版本和执行证据；模型交付使用 [交付模板](../templates/model-handoff.md)。

### 1. 训练

```bash
python wheel_legged_gym/scripts/train.py --task=mini_wheel_legged --headless

# 续训
python wheel_legged_gym/scripts/train.py --task=mini_wheel_legged --headless \
    --resume --load_run Jul20_12-42-47_
```

常用参数：`--max_iterations` / `--num_envs`（显存不够调小）/ `--checkpoint` / `--run_name`。
续训时 `--max_iterations` 表示本次**新增**轮数，例如从 `model_3000.pt` 续训 3000 轮会保存到 `model_6000.pt`，应使用新的 run 名保留原权重。
runner 恢复 PPO 优化器后会同步自适应学习率；新 checkpoint 也保存 encoder 优化器。旧 checkpoint 不含 encoder 优化器状态时会明确提示，并使用新建的 encoder 优化器，不能视为完整训练状态的逐步重放。
`scripts/agent/check_training_resume.py` 可在上述 Python 3.8 环境做 CPU 续训状态检查（新旧 checkpoint、仅加载权重和无 encoder 分支），不创建仿真，也不替代可见预检、正式训练或行为验收。
输出：`logs/mini_wheel_legged/<日期时间_run_name>/model_*.pt`。
在另一终端监控：`tensorboard --logdir=logs/mini_wheel_legged --port=8080`。

### 2. Isaac 内回放

```bash
python wheel_legged_gym/scripts/play.py --task=mini_wheel_legged
python wheel_legged_gym/scripts/play.py --task=mini_wheel_legged \
    --load_run Jul20_12-42-47_ --checkpoint 6900
```

必须先在 Isaac 里确认策略本身能站能走，再进 sim2sim。

### 3. MuJoCo sim2sim

用完整 checkpoint `model_*.pt`（含 encoder）。导出的 `policy_1.pt` 只有 actor、缺 encoder，**不可用**。

```bash
# 3a 形状自检（不跑仿真，几秒）
python sim2sim/mj_sim2sim.py --selfcheck \
    --checkpoint logs/mini_wheel_legged/Jul21_14-09-24_/model_4000.pt

# 3b 站立（vx=0）
python sim2sim/mj_sim2sim.py --render \
    --checkpoint <ckpt> --cmd_vx 0 --cmd_height 0.30 --init_height 0.30

# 3c 行走
python sim2sim/mj_sim2sim.py --render --checkpoint <ckpt> --cmd_vx 1.0 --init_height 0.30
```

判读终端每 100 策略步的 `x/z/vx/|a|max` 行：`z` 稳定在 `cmd_height` 附近、`vx` 跟上 `cmd_vx`、`|a|max` 若持续接近 100（饱和）说明策略发散。

其他参数见 [COMMANDS.md](../../COMMANDS.md)（`--cmd_yaw` / `--sim_time` / `--no_realtime` / `--friction` / `--teleop` 等）。

### 改了 XML 或 URDF

```bash
python sim2sim/check_model.py     # 碰撞对 + 轮圆柱位置自检
```

### 失稳时的对照组

```bash
python sim2sim/eval_isaac.py --load_run <run> --checkpoint <n> --cmd_vx 0.5
```

Isaac 也失稳 → 策略问题，回去调训练。Isaac 稳 → 引擎 gap，调 MJCF 接触参数或加强域随机化。

#### chuanliantui 30 秒静站与共同推扰

按 [命令速查](../../COMMANDS.md#30-秒静站与共同推扰计划) 先做同 checkpoint、零速度、
0.22 m、150 N/侧气弹簧、摩擦 0.5 的 30 秒无推扰对照，再给两端追加同一
`--push_schedule /tmp/standing-pushes.json`。用
`python sim2sim/standing_push_schedule.py --output /tmp/standing-pushes.json` 生成默认
30 秒、seed 42 的计划；`python scripts/agent/check_standing_push_schedule.py` 仅检查
2 ms 边界、重叠拒绝及计划可复现性，不替代抗扰行为验收。

Isaac 评估自动延长回合期限到至少 `sim_time + 1` 秒，保留失稳终止；训练期限不变。
取消轮目标裁剪的 H7 对照需显式传 `--h7_no_wheel_target_clip`，电机力矩限幅仍保留。
固定计划在两端每 2 ms 向基座质心施加同一世界系力，重置不重播；Isaac 不能同时启用
`--pushes` / `--domain_rand`。轨迹中的真速度和 encoder 估速应按同一推理时刻比较。
100 Hz 推力采样相位不同（MuJoCo 为策略步起点、Isaac 为末个物理子步），不可直接积分
核对；检查 metadata 的 `scheduled_push_impulse_world_ns` 与计划总冲量。
分别报告位移、速度、高度、姿态、重置及推后恢复；原本持续漂移的模型恢复运动状态，
不等于恢复原地稳站。命令和静态检查通过不代表本轮行为验收通过。

## 没有单元测试

### chuanliantui 气弹簧

`python scripts/agent/check_chuanliantui_gas_spring.py` 在 CPU 对比 Torch 的姿态相关
膝力矩、MuJoCo tendon 广义力和长度差分，检查左右符号、零推力、刚体姿态旋转、
电机/被动力分离与随机推力合成。它是数值和接口检查，不替代行为回放。
`check_chuanliantui_train_proxy.py` 同时检查代理的六电机、两气弹簧及 CAD 安装点。
气弹簧默认 150 N/侧；历史无弹簧行为对照需在两种回放中显式传 `--gas_spring_force 0`。

`python scripts/agent/check_gas_spring_asset_geometry.py` 检查从原始/训练 URDF
定位件读取端点、任意关节轴及安装 RPY，使用独立 MuJoCo tendon 和长度差分作
数值参照；覆盖安装点变更、左右改名、joint/link 不同名、两侧不同推力及无效资产。
CUDA 可用时同时检查批量 GPU 数学路径，不创建 Isaac 物理仿真。
2026-10-08 的执行范围与结果见[气弹簧资产几何验证](../diagnostics/20261008-gas-spring-asset-geometry.md)。

### chuanliantui ONNX 导出检查

`scripts/agent/export_chuanliantui_onnx.py` 从完整 25 维 checkpoint 导出 encoder + actor，执行 ONNX checker 和参考执行器/PyTorch 数值比较。可用 `--fixed-batch --opset 13` 生成面向 STM32H723 的静态 batch=1 模型；默认仍为动态 batch、opset 17。运行方式和 Cube.AI 验证步骤见 [ONNX 导出说明](../../ONNX导出说明.md)。该检查不替代训练、回放、sim2sim 或板端验证。

本仓库没有 pytest、没有 CI。「测试」= `--selfcheck` + 上面三步人眼判读。不要声称跑过单测。新增可自动判读的检查请放 `scripts/agent/`。
