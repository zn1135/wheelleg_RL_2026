# 构建与测试

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

### chuanliantui ONNX 导出检查

`scripts/agent/export_chuanliantui_onnx.py` 从完整 25 维 checkpoint 导出 encoder + actor，执行 ONNX checker 和参考执行器/PyTorch 数值比较。可用 `--fixed-batch --opset 13` 生成面向 STM32H723 的静态 batch=1 模型；默认仍为动态 batch、opset 17。运行方式和 Cube.AI 验证步骤见 [ONNX 导出说明](../../ONNX导出说明.md)。该检查不替代训练、回放、sim2sim 或板端验证。

本仓库没有 pytest、没有 CI。「测试」= `--selfcheck` + 上面三步人眼判读。不要声称跑过单测。新增可自动判读的检查请放 `scripts/agent/`。
