# 常用命令速查

Python 环境：`/home/zn1135/miniconda3/envs/wheellegged_py38/bin/python`（下面简写为 `python`）

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

当前 `sim2sim/chuanliantui.xml` 是由 `chuanliantui_new_1` 生成的真实闭链模型：14 个机构关节、6 个实体 motor、4 个 `connect`。`mj_sim2sim_ct.py` 会从被动 `f1` 构造串联策略观测，并用 equality Jacobian 将虚拟 `f0/f1` 力矩映射到实体 `f0/f00`。

```bash
# 从 URDF 重建 XML（改 URDF 后必须重跑）
python scripts/agent/generate_chuanliantui_closed_mjcf.py

# 自动检查：模型维度、四个销轴初始误差、5 秒无控制约束稳定性
python scripts/agent/check_chuanliantui_closed.py

# 仅查看真实闭链结构：不加载策略、不施加电机动作，打开 MuJoCo 原生 viewer。
# --mjcf=sim2sim/chuanliantui.xml：指定要加载的 MJCF 模型；其中包含两条支链和 4 个 connect。
python -m mujoco.viewer --mjcf=sim2sim/chuanliantui.xml

# 策略接口自检：只加载网络并验证 actor/encoder 输出形状，不运行 MuJoCo。
# --selfcheck：启用接口检查模式。
# --checkpoint：完整 model_*.pt 路径；不能使用缺 encoder 的 policy_1.pt。
python sim2sim/mj_sim2sim_ct.py --selfcheck \
    --checkpoint logs/chuanliantui/Sep08_12-56-46_new1_train_proxy_v1_resume/model_3000.pt

# 真实闭链站立验证：在重力、地面接触和实体 f0/f00 电机下运行策略。
# --render：打开 MuJoCo viewer；--no_hold：仿真结束后自动关闭窗口。
# --checkpoint：要验证的完整策略权重；--cmd_vx：前向速度命令，0 表示原地站立。
# --cmd_height：策略观测中的目标机身高度；--init_height：复位时基座初始高度。
# --sim_time：仿真持续秒数。
python sim2sim/mj_sim2sim_ct.py --render --no_hold \
    --checkpoint logs/chuanliantui/Sep08_12-56-46_new1_train_proxy_v1_resume/model_3000.pt \
    --cmd_vx 0 --cmd_height 0.32 --init_height 0.33 --sim_time 20

# 真实闭链行走验证（当前 model_3000.pt 能保持直立，但尚未通过 1.0 m/s 速度跟踪）
# --render：打开 viewer；--no_hold：20 秒结束后自动关闭。
# --checkpoint：完整策略权重；--cmd_vx 1.0：请求 1.0 m/s 前向行走。
# --cmd_height 0.32：目标高度；--init_height 0.33：初始基座高度；--sim_time 20：持续 20 秒。
python sim2sim/mj_sim2sim_ct.py --render --no_hold \
    --checkpoint logs/chuanliantui/Sep08_12-56-46_new1_train_proxy_v1_resume/model_3000.pt \
    --cmd_vx 1.0 --cmd_height 0.32 --init_height 0.33 --sim_time 20

# 重新训练闭链适配的起立策略：必须先不带 --headless、--num_envs 20 观察新环境，
# 用户确认画面后才可去掉 --num_envs 并加 --headless 做正式训练。
# --task=chuanliantui_standup：选择 1 m 高空微蹲自由落地后稳站任务；--num_envs 20：只创建 20 个并行环境，便于人工观察。
python wheel_legged_gym/scripts/train.py --task=chuanliantui_standup --num_envs 20

# 新起立权重的真实闭链 MuJoCo 验证：将 <run> 和 <checkpoint> 替换为新训练输出。
# --standup：使用与当前 Isaac 训练一致的 1 m 高空微蹲初态；首次轮接地后下一控制步才推理策略。
# --render：打开 MuJoCo viewer；--no_hold：仿真结束后自动关闭窗口。
# --checkpoint：新训练生成的完整 model_*.pt；--cmd_vx 0：起立阶段不请求前进。
# --cmd_height 0.3276：起立目标高度；--sim_time 20：最多运行 20 秒。
python sim2sim/mj_sim2sim_ct.py --standup --render --no_hold \
    --checkpoint logs/chuanliantui_standup/<run>/model_<checkpoint>.pt \
    --cmd_vx 0 --cmd_height 0.3276 --sim_time 20
```

`--standup` 使用当前 `chuanliantui_standup` 训练的初态：虚拟关节为微蹲默认值、基座高度为 1 m。落地前使用零策略动作但仍执行 PD 内环；首次轮接地后的下一控制步才调用策略，观测历史在落地前持续更新。它只能搭配 `chuanliantui_standup` 的完整 checkpoint。IMU 和气弹簧锚点暂未写入该 MJCF；它们也不参与当前四个闭环约束。

## 典型工作流

1. 改 config → `train.py --headless` 训练
2. `play.py` 在 Isaac 里确认能站能走
3. `mj_sim2sim.py` 站立 → 行走，验证跨引擎迁移
4. MuJoCo 里也稳了，再考虑上真机（参照 `sim2sim/mj_sim2sim.py` 头部的部署契约注释）
