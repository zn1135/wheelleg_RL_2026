# 常用命令速查

Python 环境：`/home/zn/miniforge3/envs/wheellegged_py38/bin/python`（下面简写为 `python`）

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
# [11,0,0,-11,0,0]（写入 MuJoCo 前规范到等价 [-pi,pi)）；首次轮触地后下一策略步接管。
python sim2sim/mj_sim2sim_ct.py --standup --render \
    --checkpoint logs/chuanliantui_standup/<run>/model_<checkpoint>.pt \
    --cmd_vx 0 --cmd_height 0.20 --friction 0.75

# 可选：真实闭链差异诊断。--closed_chain 明确启用旧 chuanliantui.xml 与
# ClosedChainAdapter；它不是串联训练代理一致性或真机验证通过的依据。
python sim2sim/mj_sim2sim_ct.py --closed_chain --standup --render \
    --checkpoint logs/chuanliantui_standup/<run>/model_<checkpoint>.pt \
    --cmd_vx 0 --cmd_height 0.20 --friction 0.75

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

# 重新训练地面起立策略：必须先不带 --headless、--num_envs 20 观察新环境，
# 用户确认画面后才可去掉 --num_envs 并加 --headless 做正式训练。
# --task=chuanliantui_standup：选择 0.15 m 地面后摆起立任务；--num_envs 20：只创建 20 个并行环境，便于人工观察。
python wheel_legged_gym/scripts/train.py --task=chuanliantui_standup --num_envs 20

# 新起立权重的串联训练代理 MuJoCo 回放：将 <run> 和 <checkpoint> 替换为新训练输出。
# --standup：使用与当前 Isaac 训练一致的 0.15 m 地面后摆初态；首次轮接地后下一控制步才推理策略。
# --render：打开 MuJoCo viewer 并启用方向键遥操作；命令参数为初始值，关闭窗口退出。
# --checkpoint：新训练生成的完整 model_*.pt；--cmd_vx 0：起立阶段不请求前进。
# --cmd_height 0.20：与起立训练一致的目标高度。
python sim2sim/mj_sim2sim_ct.py --standup --render \
    --checkpoint logs/chuanliantui_standup/<run>/model_<checkpoint>.pt \
    --cmd_vx 0 --cmd_height 0.20
```

`--standup` 使用当前 `chuanliantui_standup` 的 0.15 m 地面后摆初态与配置的后摆关节；落地前使用零策略动作但仍执行 PD 内环；首次轮接地后的下一控制步才调用策略，观测历史在落地前持续更新。它只能搭配完整 `model_*.pt`。渲染模式默认启用键盘遥操作，关闭窗口退出；无渲染模式持续运行，使用 Ctrl+C 退出。默认仍使用串联训练代理；`--closed_chain` 仅显式启用旧真实闭链 XML 与 `ClosedChainAdapter` 进行差异诊断。

## 典型工作流

1. 改 config → `train.py --headless` 训练
2. `play.py` 在 Isaac 里确认能站能走
3. `mj_sim2sim.py` 站立 → 行走，验证跨引擎迁移
4. MuJoCo 里也稳了，再考虑上真机（参照 `sim2sim/mj_sim2sim.py` 头部的部署契约注释）
