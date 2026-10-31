# Isaac 历史任务命令

以下命令均从仓库根目录执行。模型采纳状态以[项目实验记录](../ai/project-log.md)为准，示例模型不代表当前推荐。

## Isaac旧环境准备

下文命令属于旧 Isaac 环境，不混装依赖。

先按 [协作指南](../../CONTRIBUTING.md) 创建并填写 `.env.local`。在仓库根目录执行以下准备，完整激活对应 Python 3.8 conda 环境（`conda` 命令需已在 PATH 中）：

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

旧起立配置为解锁后目标0.22 m、成功高度0.20 m、膝余量奖励区0.05 rad，默认每侧150 N气弹簧。具体条件见[旧sim2sim说明](../../docs/ai/sim2sim.md#chuanliantui-站高精度与膝余量续训)；复现旧实验需要同时匹配模型、代码和配置。

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

检查气弹簧数值与施力合成：`python -m scripts.agent.isaac.check_chuanliantui_gas_spring`。无弹簧历史对照需在回放中显式加 `--gas_spring_force 0`；仅加载旧配置快照不会自动关闭当前父类的弹簧配置。完整条件见[气弹簧记录](../../docs/interfaces/chuanliantui-gas-spring-training.md)。

#### 前一轮：0.22 m 站高与 0.05 rad 膝余量续训

此标题保留供旧文档定位；历史基点是 `Oct03_03-28-58_standup_action_delay_0_10ms/model_9000.pt`。当时训练条件和结果查[项目实验记录](../../docs/ai/project-log.md)，不再重复多轮续训命令。

#### 动作延迟随机化续训（历史示例）

历史延迟为0–10 ms，按2 ms内环量化；检查入口为 `python -m scripts.agent.isaac.check_action_delay`。普通play／默认Isaac对照回放关闭延迟，`eval_isaac.py --action_delay_ms 0`、`6`、`10` 可单独固定延迟；`--domain_rand` 同时启用其他随机化，不是单独的延迟对照。

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
python -m scripts.agent.isaac.check_standing_push_schedule
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
python -m scripts.agent.isaac.generate_chuanliantui_train_proxy_mjcf
python -m scripts.agent.isaac.check_chuanliantui_train_proxy
python -m scripts.agent.isaac.check_chuanliantui_observation

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

`--standup` 使用旧任务的地面后摆初态。串联代理和 `--closed_chain_controller cad` 在首次轮接地后接管；默认H7闭链路径先做十拍零动作PD预热。后轴名义角度注册、轮目标裁剪及报告定义见[旧H7闭链接口](../../docs/interfaces/chuanliantui-h7-closed-replay.md)，不用其历史参数替代当前固件定义。

