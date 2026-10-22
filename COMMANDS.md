# 常用命令速查

2026-10-22 更新。除明确标注“Jetson端”的步骤外，均在电脑端的仓库根目录执行。模型采纳状态与实验结论见[项目实验记录顶部](docs/ai/project-log.md)，本页只维护命令。

| 要做什么 | 入口 |
|---|---|
| 当前串联腿训练、回放、评估与导出 | [mjlab 主线](#mjlab-当前串联腿主线) |
| Jetson连接、实采、实测姿态回放 | [Jetson操作](#jetson操作连接录制与回放) |
| 原UART与USB工程的选择 | [部署目录说明](deployment/README.md) |
| 固件编译与烧录 | 所选工程的[UART说明](deployment/H7_RL/md/file-guide.md#terminal-flash)或[USB说明](deployment/H7_RL_ros/md/file-guide.md#terminal-flash) |
| Isaac历史模型与旧检查脚本 | [Isaac旧环境准备](#isaac旧环境准备) |

## mjlab 当前串联腿主线

使用 `mjlab_training/.venv`（Python 3.12），入口为 `scripts/mjlab.sh`。actor输入39维、critic输入82维、动作6维；物理／控制500 Hz、策略50 Hz。actor负责根据观测产生动作，critic仅在训练中估计状态的收益。与下文旧串联腿25／125维模型不兼容，mjlab参数使用连字符，例如 `--num-envs`，不要套用Isaac的 `--num_envs`。

### 选择模型与检查环境

以下用项目基线演示；测试其他模型时修改 `ct_checkpoint`，新终端需重新设置。运行使用当前共享XML，不等于重现该模型训练时的物理配置；对应来源见[共享XML记录](docs/diagnostics/20261021-shared-huananhu-xml.md)。

```bash
ct_checkpoint="artifacts/multimode-v1-replay-20261014/models/model_3023.pt"
test -f "$ct_checkpoint"
scripts/mjlab.sh check --checkpoint "$ct_checkpoint" --num-envs 2 --steps 16
```

成功判据：退出码0，输出actor形状 `[2,39]`、critic形状 `[2,82]`、动作维数6，无非有限值。此项只检查接口和有限步数值。

已有环境无需重新安装；首次配置见[mjlab说明](docs/mjlab-chuanliantui.md#可执行命令)。

### 训练与续训

以下轮数仅为命令示例，实际使用本次已确定的训练预算。两条训练命令择一执行，脚本默认不创建可视窗口，无需 `--headless`。若修改初始位置或复位分布，先执行 `scripts/mjlab.sh play --num-envs 20` 观察初态。

```bash
# 从零训练。
scripts/mjlab.sh train --num-envs 4096 --iterations 3000 --run-name multimode-new
# 从选定checkpoint续训；--iterations是本次新增轮数。
scripts/mjlab.sh train --num-envs 2048 --iterations 300 --run-name multimode-resume \
  --resume "$ct_checkpoint"
```

另一电脑终端在仓库根目录监控：

```bash
mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_huananhu_multimode_v1 --port 8080
```

成功判据：终端打印新的 `Training output` 目录，目录内生成 `params/`、TensorBoard事件及checkpoint；监控页面出现训练曲线。旧模型与日志保留。

### 可视回放

```bash
scripts/mjlab.sh play --checkpoint "$ct_checkpoint" --num-envs 1
# 手动每5秒击倒已站稳的机器人，观察恢复。
scripts/mjlab.sh play --checkpoint "$ct_checkpoint" --num-envs 1 --knockdown-every 5
# 无桌面时使用网页查看器，8091避开TensorBoard的8080。
scripts/mjlab.sh play --checkpoint "$ct_checkpoint" --num-envs 4 \
  --viewer viser --viser-port 8091
```

成功判据：viewer显示策略动作；原生窗口左上角CMD实时显示当前选中环境的前进速度(m/s)、偏航角速度(rad/s)和目标高度(m)；击倒回放能观察到恢复过程。play关闭随机化及观测噪声/延迟，保留实测校准的一拍2 ms动作等待，不等于训练配置下的验收。

### 地面起立评估与跟踪测量

```bash
# 每次创建新结果目录，避免覆盖历史产物。
ct_eval_dir="$(mktemp -d artifacts/multimode-eval-XXXXXX)"
scripts/mjlab.sh eval --checkpoint "$ct_checkpoint" --num-envs 64 --seeds 42 \
  --start-mode force --terrain flat --steps 1000 --output "$ct_eval_dir/standup.json"
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_tracking.py \
  --checkpoint "$ct_checkpoint" --output "$ct_eval_dir/tracking.json"
```

成功判据：生成JSON报告，分别查看起立成功／存活计数与速度／偏航／高度跟踪误差；命令执行完毕不代表行为全部通过。最终双轮平衡另按[平衡验收](docs/interfaces/chuanliantui-wheel-balance-acceptance-v2.md)执行，跟踪按[跟踪测量标准](docs/interfaces/chuanliantui-huananhu-multimode-tracking-v1.md)判读，两者互不替代。完整击倒验收命令见[mjlab说明](docs/mjlab-chuanliantui.md#可执行命令)。

### ONNX导出与独立CPU回放

ONNX是保存策略网络的文件格式；当前模型固定输入39维、输出6维。使用通用导出入口 `scripts/mjlab.sh export`；旧 `scripts/export_huananhu_multimode_onnx.py` 只适用于model_3023。

```bash
ct_export_dir="$(mktemp -d artifacts/multimode-export-XXXXXX)"
scripts/mjlab.sh export --checkpoint "$ct_checkpoint" \
  --output "$ct_export_dir/policy.onnx" --opset 13
# 相同checkpoint配对其刚导出的ONNX，固定零命令测20秒。
mjlab_training/.venv/bin/python sim2sim/mj_sim2sim_huananhu_multimode.py \
  --checkpoint "$ct_checkpoint" --onnx "$ct_export_dir/policy.onnx" \
  --cmd-vx 0 --cmd-yaw 0 --cmd-height 0.27 --seconds 20 \
  --report "$ct_export_dir/cpu-normal.json"
```

成功判据：生成ONNX、`*_reference_vectors.npz`、`*_export.json`，导出数值检查通过；CPU报告查看 `passed` 和各项误差。图外的观测构造、PD与电机映射仍需单独核对。

看地面起立时追加 `--initial-pose ground` 并使用新的报告名；可视回放追加 `--render`。键盘遥操作使用 `--render --teleop --seconds 0`，同时去掉 `--report`，因为变化命令不能用于固定命令测量。CPU回放默认读取当前共享XML，历史模型验收需配套当时资产；详细定义见[ONNX使用说明](docs/onnx-huananhu-multimode-model5849.md)。

## Jetson操作：连接、录制与回放

当前Jetson只读接收H7反馈，不发送策略动作或电机指令。下面分别标明电脑端和Jetson端；Jetson项目为 `~/real2sim`，使用其 `.venv/bin/python`。2026-10-22已核对Jetson实际脚本的 `--help`；当前不支持 `--firmware-manifest` 或 `--warmup`，不要在下列Jetson命令中加入这两个参数。

<a id="jetson只读usb采集连接"></a>

### 电脑端：连接Jetson

在电脑的本项目根目录运行。本机用户名、SSH地址、Jetson项目目录和H7 USB设备保存在Git忽略的 `.env.local`，字段见 `.env.example`。`imca`是当前Jetson用户名；SSH连接使用配置中的 `JETSON_SSH_TARGET`：

```bash
source ./.env.local
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$JETSON_SSH_TARGET" 'id -un; hostname; ls -l /dev/serial/by-id'
# 进入Jetson终端；后续“Jetson端”命令也可直接在Jetson本机终端运行。
ssh "$JETSON_SSH_TARGET"
```

成功判据：用户名正确，且列出STM32 Virtual ComPort设备；H7数据口不是PowerDebugger的CDC口。SSH密钥或密码不写入项目。

### Jetson端：准备USB设备与实时画面

```bash
cd ~/real2sim
ls -l /dev/serial/by-id/
# 取H7的稳定设备路径；应只输出一个实际存在的STM32 USB口。
h7_usb_port="$(ls /dev/serial/by-id/*STM32_Virtual_ComPort*-if00)"
printf '%s\n' "$h7_usb_port"
fuser "$h7_usb_port"
```

`fuser`没有输出表示当前没有接收进程占用；有进程时先在其终端按Ctrl+C结束。每个新Jetson终端重新设置 `h7_usb_port`，只开一个USB接收进程。

```bash
# Jetson本机桌面：显示实机姿态，Ctrl+C或关窗结束。
.venv/bin/python sim2sim/real2sim_huananhu.py --port "$h7_usb_port"
```

成功判据：窗口姿态随实机变化，来源不持续显示STALE（数据过期）。屏幕位置固定，不表示实测位移。普通SSH终端采用下面的无窗口录制；开画面的命令在Jetson本机桌面执行。

### Jetson端：录制120秒实机RL数据

先结束上面的实时显示进程，再在同一Jetson终端运行：

```bash
rl_record_dir="artifacts/model6848-real-rl-$(date +%Y%m%d-%H%M%S)"
printf '本次录制目录：%s\n' "$rl_record_dir"
.venv/bin/python -u sim2sim/real2sim_huananhu.py \
  --port "$h7_usb_port" \
  --record "$rl_record_dir" \
  --headless --seconds 120
```

成功判据：会话目录生成 `stream.jsonl` 和 `meta.json`，正常结束后 `meta.json` 中 `complete` 为 `true`。录制启动不会自动投入RL；是否投入以遥控档位和记录中的实际投入状态为准。USB断开后应另开新会话，保留中断段，不手工改成完整记录。

当前 `deployment/H7_RL_ros` 遥控档位如下；轮矩测试独占期间的切档另受台架状态机约束。

| 左拨杆／右拨杆 | H7请求的模式 |
|---|---|
| 左下／右下 | 禁用 |
| 左上／右下 | 选择RL，不投入出力 |
| 左上／右中 | 投入RL，实际出力还受反馈和许可条件约束 |
| 左上／右上 | 轮矩脉冲测试档，不是RL投入档 |

### Jetson端：回放录制

在Jetson本机桌面、同一终端中回放刚录的会话；回放不占用USB：

```bash
cd ~/real2sim
.venv/bin/python sim2sim/real2sim_huananhu.py --replay "$rl_record_dir"
```

新终端中先列出会话，再把所选目录赋给变量：

```bash
cd ~/real2sim
ls -dt artifacts/model6848-real-rl-*/
rl_record_dir="artifacts/<要回放的会话目录>"
.venv/bin/python sim2sim/real2sim_huananhu.py --replay "$rl_record_dir"
# 普通SSH下只检查回放，不开窗口。
.venv/bin/python sim2sim/real2sim_huananhu.py --replay "$rl_record_dir" --headless
```

列表按目录修改时间排序，不保证第一项就是最近开始录制的会话。成功判据：能够读完整会话，画面按录制时间变化；这是实测姿态重放，不是MuJoCo自由积分预测。

### 电脑端：取回录制并生成CSV和曲线

回到电脑的本项目根目录；把 `rl_session_name` 换成Jetson上实际的会话名：

```bash
source ./.env.local
rl_session_name="<实际会话名，例如model6848-real-rl-YYYYMMDD-HHMMSS>"
rl_local_dir="artifacts/jetson-import-$(date +%Y%m%d-%H%M%S)/$rl_session_name"
mkdir -p "$rl_local_dir"
rsync -a "$JETSON_SSH_TARGET:$JETSON_PROJECT_DIR/artifacts/$rl_session_name/" "$rl_local_dir/"
mjlab_training/.venv/bin/python sim2sim/real2sim_huananhu.py --analyze "$rl_local_dir"
```

成功判据：本地会话的 `analysis/` 下生成 `trajectory.csv`、`report.json` 和 `signals.png`。分析会核对录制所用模型及机器配置哈希；缺少对应版本时补齐原版本，不跳过核对。完整控制输出的采集与检查另见[完整输出接口](docs/interfaces/20261021-h7-output-capture-v1.md)。

## Isaac旧环境准备

下文命令属于旧 Isaac 环境，不混装依赖。

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
#   --max_iterations 2000     本次新增迭代数（覆盖 config），不是续训后的总轮数
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

## Sim2Sim（Isaac历史模型的MuJoCo验证）

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

### chuanliantui（Isaac旧25维模型）

本节仅用于旧模型维护：actor观测25维、历史125维，物理／PD 500 Hz、策略100 Hz。当前mjlab主线使用上文39维入口。旧脚本的H7闭链适配是25维控制对照，不能当成当前39维固件的验收。

旧起立配置为解锁后目标0.22 m、成功高度0.20 m、膝余量奖励区0.05 rad，默认每侧150 N气弹簧。具体条件见[旧sim2sim说明](docs/ai/sim2sim.md#chuanliantui-站高精度与膝余量续训)；复现旧实验需要同时匹配模型、代码和配置。

#### 气弹簧条件下续训（历史示例）

以下保留旧任务从12000轮模型新增3000轮的命令，不作为当前mjlab训练推荐。修改run和checkpoint时成对替换；输出进入新的时间戳目录。

```bash
python wheel_legged_gym/scripts/train.py --task=chuanliantui_standup --headless \
    --resume --load_run Oct03_14-27-59_standup_h022_knee005_push --checkpoint 12000 \
    --max_iterations 3000 --run_name standup_h022_gas150_push
```

另一相同Isaac环境的终端监控：

```bash
tensorboard --logdir=logs/chuanliantui_standup --port=8080
```

检查气弹簧数值与施力合成：`python scripts/agent/check_chuanliantui_gas_spring.py`。无弹簧历史对照需在回放中显式加 `--gas_spring_force 0`；仅加载旧配置快照不会自动关闭当前父类的弹簧配置。完整条件见[气弹簧记录](docs/interfaces/chuanliantui-gas-spring-training.md)。

#### 前一轮：0.22 m 站高与 0.05 rad 膝余量续训

此标题保留供旧文档定位；历史基点是 `Oct03_03-28-58_standup_action_delay_0_10ms/model_9000.pt`。当时训练条件和结果查[项目实验记录](docs/ai/project-log.md)，不再重复多轮续训命令。

#### 动作延迟随机化续训（历史示例）

历史延迟为0–10 ms，按2 ms内环量化；检查入口为 `python scripts/agent/check_action_delay.py`。普通play／默认Isaac对照回放关闭延迟，`eval_isaac.py --action_delay_ms 0`、`6`、`10` 可单独固定延迟；`--domain_rand` 同时启用其他随机化，不是单独的延迟对照。

#### 回放与诊断

奖励、膝余量和推力调度检查分别为 `check_standup_precision_rewards.py`、`check_standup_pushes.py`（均在 `scripts/agent/`，用已激活的Isaac环境运行）。实际回放同时观察高度、倾角、位移、重置与膝余量，不能仅以 `recovered_rate` 判断站稳或抗推成功。

##### 30 秒静站与共同推扰计划

通过接口自检后，先用同一 12000 轮模型做 **30 秒无推扰** 对照：零速度、0.22 m、
每侧气弹簧 150 N、接触摩擦 0.5，动作延迟均为零。下列旧模型尚未学习新增气弹簧条件。

```bash
isaac_eval_dir="$(mktemp -d /tmp/isaac-standing-XXXXXX)"
python sim2sim/eval_isaac.py --task chuanliantui_standup \
    --load_run Oct03_14-27-59_standup_h022_knee005_push --checkpoint 12000 \
    --cmd_vx 0 --cmd_yaw 0 --cmd_height 0.22 --standup_post_unlock \
    --gas_spring_force 150 --contact_friction 0.5 --action_delay_ms 0 \
    --sim_time 30 --metrics_report "$isaac_eval_dir/isaac-standing-30s.json"

python sim2sim/mj_sim2sim_ct.py --closed_chain --standup --h7_no_wheel_target_clip \
    --checkpoint logs/chuanliantui_standup/Oct03_14-27-59_standup_h022_knee005_push/model_12000.pt \
    --cmd_vx 0 --cmd_yaw 0 --cmd_height 0.22 --gas_spring_force 150 --friction 0.5 \
    --sim_time 30 --no_realtime --diagnostic_report "$isaac_eval_dir/mujoco-standing-30s.json"
```

`--h7_no_wheel_target_clip` 显式关闭 H7 回放的轮目标速度裁剪，保留电机力矩限幅；
只允许 H7 闭链路径，不能与 `--wheel_vel_limit` 合用。不加该开关时仍默认 ±20 rad/s。
Isaac 有限时长评估会把回合期限延长到至少 `sim_time + 1` 秒，保留失稳终止，
避免训练的 20 秒超时打断 30 秒观察；这不修改训练回合长度。

再生成一次固定计划，两端共用同一个 JSON：

```bash
python scripts/agent/check_standing_push_schedule.py
python sim2sim/standing_push_schedule.py --duration_s 30 --seed 42 \
    --output "$isaac_eval_dir/standing-pushes.json"
```

推扰阶段重复上面两条评估命令，均追加 `--push_schedule "$isaac_eval_dir/standing-pushes.json"`，
并把报告改为新的文件名，例如 `"$isaac_eval_dir/isaac-pushed-30s.json"`、`"$isaac_eval_dir/mujoco-pushed-30s.json"`。
计划和报告均拒绝覆盖已有文件。Isaac 的固定计划不能与 `--pushes` 或 `--domain_rand` 合用。
默认首次推力在 5 秒，持续 0.1 秒，水平幅值 5–15 N；以后每次结束后留 3–5 秒空档。
两端均每 2 ms 向 `base_link` 质心施加同一世界系力，使用整次评估时钟，重置不重播计划。

Isaac 记录推理时刻的 `policy_t_s`、真速度、encoder 估速及误差；MuJoCo 同时记录
`velocity_pos_diff` 与 `estimated_velocity`，比较时只取策略已激活的同一时刻。
两端均以 100 Hz 保存轨迹，但 MuJoCo 的推力字段取策略步起点，Isaac 取末个物理子步，
**不能用这些采样行积分核对冲量**；应比较 metadata 的 `scheduled_push_impulse_world_ns`
与计划的 `sum(force_world_n * duration_s)`，容许 float32 舍入误差。
若无推扰时已经持续漂移，后续只能报告推后是否恢复原运动状态，不能称为恢复原地稳站。

##### 串联代理与闭链可视回放

`mj_sim2sim_ct.py` 默认加载 `chuanliantui_train_proxy.xml`（无闭链约束）；加 `--closed_chain` 才加载实体闭链模型。两者仅用于对应旧25维checkpoint，旧27维串联腿模型不能直接加载。

```bash
# 修改训练URDF后，才需要重新生成代理；随后检查结构和观测。
python scripts/agent/generate_chuanliantui_train_proxy_mjcf.py
python scripts/agent/check_chuanliantui_train_proxy.py
python scripts/agent/check_chuanliantui_observation.py

# 替换为实际的完整checkpoint；不能用缺encoder的policy_1.pt。
isaac_ct_checkpoint="logs/chuanliantui_standup/<实际run>/model_<编号>.pt"
python sim2sim/mj_sim2sim_ct.py --selfcheck --checkpoint "$isaac_ct_checkpoint"
# 串联训练代理起立。
python sim2sim/mj_sim2sim_ct.py --standup --render \
    --checkpoint "$isaac_ct_checkpoint" --cmd_vx 0 --cmd_height 0.22
# 旧H7控制的闭链对照；不代表当前39维H7接口。
python sim2sim/mj_sim2sim_ct.py --closed_chain --standup --render \
    --checkpoint "$isaac_ct_checkpoint" --cmd_vx 0 --cmd_height 0.22
```

成功判据：自检仅确认网络形状；可视回放需看到接地后起立，并持续观察高度、姿态与位移。点击viewer取得焦点后，Ctrl+R复位；关窗退出。无窗口回放去掉 `--render`，加 `--sim_time 15 --no_realtime`。

`--standup` 使用旧任务的地面后摆初态。串联代理和 `--closed_chain_controller cad` 在首次轮接地后接管；默认H7闭链路径先做十拍零动作PD预热。后轴名义角度注册、轮目标裁剪及报告定义见[旧H7闭链接口](docs/interfaces/chuanliantui-h7-closed-replay.md)，不用其历史参数替代当前固件定义。

## 典型工作流

| 任务 | 顺序 |
|---|---|
| 当前mjlab主线 | 有限检查 → 预算内训练 → play回放 → 地面起立、平地跟踪及最终平衡评估 → 配对ONNX的独立CPU回放 → 对应版本的板端与实机核对 |
| Isaac历史任务 | 激活Python 3.8环境 → 预算内训练 → Isaac play回放 → 对应旧脚本的MuJoCo自检／站立／行走 |
| Jetson只读实采 | 核对USB与唯一接收进程 → 新目录录制 → 确认会话完整性 → 实测姿态回放 → 取回电脑分析 |

自检、导出或编译通过均不能替代行为验收；模型交付与实机接口按[部署约定](docs/deployment-contract.md)核对。
