# 串联腿 mjlab 训练

辅助脚本已按用途迁移；下文固定旧实验保留其历史配方，旧路径查[迁移表](../scripts/agent/MIGRATION.md)，归档复核不代表当前版本验收。


2026-10-30新增`chuanliantui_getup`和`chuanliantui_walk`两个独立任务：25维观测、125维历史、71维Critic、6动作，网络无模式标志。各自训练、续载、单模型play和导出，日志独立；两模型自动切换只在独立CPU sim2sim。原stand_v2及旧模型保留；接口、20环境预检和训练/TensorBoard命令见[双策略说明](../docs/interfaces/20261030-chuanliantui-two-policy.md)。

提供 `chuanliantui` 华南虎 multimode 与 `chuanliantui_stand` 独立稳站任务，均为39/82维观测、6维动作、500 Hz物理／100 Hz策略。稳站Actor保留第15、16维轮角占位，不追加前向、侧向平面速度；历史41维模型另按冻结配方回放。原任务见 [mjlab 串联腿训练](../docs/mjlab-chuanliantui.md)，新任务的固定 0.27 m、高度／机体平速／腿电机角速度验收和独立日志见[稳站说明](../docs/mjlab-chuanliantui-stand.md)。使用独立的 Python 3.12 环境，不装进 Isaac Gym 环境。

## 代码结构

独立 `chuanliantui_stand_v2`：Isaac布局32维＋5帧速度编码器、89维Critic、6动作；全地面起步、起立宽限、准备期奖励屏蔽、init_std=0.5。闭链XML/PD/500Hz物理与100Hz策略保留，旧39维任务模型不能直接续载。完整网络导出及检查见[稳站v2](../docs/mjlab-chuanliantui-stand-v2.md)。

和 Isaac 的 `wheel_legged_gym/` 按同样的职责划分：

```text
src/ct_mjlab/
├── cli.py                    # ct-mjlab / scripts/mjlab.sh：<command> → scripts.<command>
├── envs/
│   ├── __init__.py           # 注册 chuanliantui、chuanliantui_stand 与 chuanliantui_stand_v2
│   └── chuanliantui/         # 冻结快照里的 23 个模块（model_3023 训练时的源码）
│       ├── multimode_env_cfg.py    # 环境配置与 PPO 配置工厂
│       ├── multimode_runner.py     # checkpoint 契约检查与课程进度
│       ├── multimode_mdp.py / multimode_observations.py / multimode_rewards.py / multimode_terrains.py
│       ├── huananhu_*.py           # multimode 基于的平地配方：命令、动作、观测、奖励、随机化、姿态库
│       └── model.py / pd_action.py / gas_spring.py / runner.py / ...
│   └── chuanliantui_stand/   # 独立命令、混合初态、奖励、推扰课程及模型契约
├── scripts/                  # train / play / eval / export / check
└── utils/
    ├── task_registry.py      # register、get_cfgs、make_env、make_alg_runner
    └── helpers.py            # 公共命令行参数
```

`envs/chuanliantui/` 下的文件与快照逐字节相同，只有 `model.py` 改了资源路径（指向 `resources/robots/chuanliantui_new_1/mjlab/`）。checkpoint 的观测、控制契约字符串和姿态库缓存签名都依赖这些源码，改奖励或观测时要同时更新契约名，否则旧模型会被误当作兼容。

## 在脚本中使用

```python
from ct_mjlab.envs import task_registry

env_cfg, train_cfg = task_registry.get_cfgs('chuanliantui')
env, env_cfg = task_registry.make_env('chuanliantui', env_cfg=env_cfg, train_cfg=train_cfg, num_envs=64)
runner, train_cfg = task_registry.make_alg_runner(env, 'chuanliantui', train_cfg=train_cfg, log_dir='default')
runner.learn(train_cfg.max_iterations)
```

`log_dir='default'` 时，日志写到 `logs/mjlab/<experiment_name>/<时间>_<run_name>`。不传 `log_dir` 则不创建目录、不写日志。

## 新增任务

1. 在 `envs/<task>/` 下写好环境配置工厂、PPO 配置工厂和 runner 类。
2. 在 `envs/__init__.py` 里调用 `task_registry.register(...)` 注册。
3. 用 `scripts/mjlab.sh check --task <task>` 跑一次有限检查。

## 独立传统控制入口

`scripts/mjlab.sh teacher` 运行 1ms LQR+VMC，不加载策略、不训练。
正常起立、20环境预检、命令序列、数值与行为检查见
[teacher说明](../docs/mjlab-lqr-vmc-teacher.md)。现有PPO入口保持原行为。

`scripts/mjlab.sh collect` 采集合格示范，`scripts/mjlab.sh train-prior` 按显式预算预训练SMP运动评分模型。
`train/play/check/eval --smp-prior <prior_N.pt>` 可选接入冻结SMP评分；当前只训练正常趴地起立和全程零速站稳；初态与教师一致，39/82/6和未启用SMP的原任务行为保持。PPO训练要求显式轮数；本次只检查未训练，见[接入与初态说明](../docs/mjlab-smp-prior.md#第三阶段可选-smp-ppo)。
数据筛选、回合分组、训练命令和正式检查见[第二阶段说明](../docs/mjlab-smp-prior.md)；尚未接入PPO奖励。
正常起立和前后／转向示范使用可选`--velocity-integral`；`--commands`可在一批环境中分配不同固定命令。
