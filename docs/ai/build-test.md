# 构建与测试

## 环境

无编译步骤，纯 Python 包，`pip install -e .` 一次即可。

本仓库必须用**装了 Isaac Gym 的 Python 3.8 conda 环境**，不能用系统 python。本机该环境的路径见 [COMMANDS.md](../../COMMANDS.md) 开头，下文 `python` 均指它。

已验证可用的版本组合：Python 3.8.20 / torch 2.1.0（CUDA 可用）/ mujoco 2.3.7 / Isaac Gym Preview 4。

自检一句话：

```bash
<你的环境>/bin/python -c "import isaacgym, torch, mujoco; print('ok')"
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

本仓库没有 pytest、没有 CI。「测试」= `--selfcheck` + 上面三步人眼判读。不要声称跑过单测。新增可自动判读的检查请放 `scripts/agent/`。
