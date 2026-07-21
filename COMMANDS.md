# 常用命令速查

Python 环境：`/home/zn1135/miniconda3/envs/wheellegged_py38/bin/python`（下面简写为 `python`）

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
    --checkpoint logs/mini_wheel_legged/Jul20_12-42-47_/model_6900.pt

# 第二步：站立测试（vx=0，只保持平衡和高度）
python sim2sim/mj_sim2sim.py --render \
    --checkpoint logs/mini_wheel_legged/Jul20_12-42-47_/model_6900.pt \
    --cmd_vx 0 --cmd_height 0.30

# 第三步：行走测试
python sim2sim/mj_sim2sim.py --render \
    --checkpoint logs/mini_wheel_legged/Jul20_12-42-47_/model_6900.pt \
    --cmd_vx 1.0

# 常用可选参数
#   --cmd_yaw 0.5        目标偏航角速度 [rad/s]
#   --cmd_height 0.20~0.40   目标机身高度（训练命令范围）
#   --sim_time 20        仿真时长 [s]
#   --no_realtime        不按真实时间节流，全速跑（无渲染批量测试用）
#   --no_hold            结束后不保持窗口
```

终端每 100 策略步打印一行 `x/z/vx/|a|max`：z 应稳定在 cmd_height 附近，vx 应跟上 cmd_vx，|a|max 持续饱和（接近 100）说明策略在发疯。

## 典型工作流

1. 改 config → `train.py --headless` 训练
2. `play.py` 在 Isaac 里确认能站能走
3. `mj_sim2sim.py` 站立 → 行走，验证跨引擎迁移
4. MuJoCo 里也稳了，再考虑上真机（参照 `sim2sim/mj_sim2sim.py` 头部的部署契约注释）
