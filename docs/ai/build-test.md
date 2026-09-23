# 构建与测试

## 环境

无编译步骤，纯 Python 包，`pip install -e .` 一次即可。

本仓库必须用**装了 Isaac Gym 的 Python 3.8 conda 环境**，不能用系统 python。本机该环境的路径见 [COMMANDS.md](../../COMMANDS.md) 开头，下文 `python` 均指它。

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

改完代码按顺序走完这三步，不要跳步——它们分别隔离「训练没练好」和「sim2sim 有 gap」两类问题。

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
监控：`tensorboard --logdir=./ --port=8080`。

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

## 没有单元测试

### chuanliantui ONNX 导出检查

`scripts/agent/export_chuanliantui_onnx.py` 从完整 25 维 checkpoint 导出 encoder + actor，执行 ONNX checker 和参考执行器/PyTorch 数值比较。可用 `--fixed-batch --opset 13` 生成面向 STM32H723 的静态 batch=1 模型；默认仍为动态 batch、opset 17。运行方式和 Cube.AI 验证步骤见 [ONNX 导出说明](../../ONNX导出说明.md)。该检查不替代训练、回放、sim2sim 或板端验证。

本仓库没有 pytest、没有 CI。「测试」= `--selfcheck` + 上面三步人眼判读。不要声称跑过单测。新增可自动判读的检查请放 `scripts/agent/`。
