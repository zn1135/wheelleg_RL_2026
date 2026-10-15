# 串联腿 mjlab 训练

## 代码结构

日常入口按 Isaac `wheel_legged_gym` 的职责组织，mjlab 仍使用独立 Python 3.12 环境：

```text
src/ct_mjlab/
├── envs/
│   ├── __init__.py                   # task_registry 注册任务
│   └── chuanliantui/
│       ├── chuanliantui.py           # 环境/runner 工厂、任务执行
│       ├── chuanliantui_config.py    # 环境与 PPO 配置工厂、配方索引
│       ├── profiles.py              # 配方选择、历史接口分流、迁移入口限制
│       └── commands.py              # recover-r14 与旧版回放/评估/导出流程
├── scripts/                         # train.py / play.py / eval.py / export.py / check.py
├── utils/
│   ├── task_registry.py             # 注册、get_cfgs、make_env、make_alg_runner、run
│   ├── helpers.py                   # 统一命令行参数
│   └── provenance.py                # recover 训练源码快照
├── cli.py                           # 原 ct-mjlab / scripts/mjlab.sh 兼容入口
├── experiments/                     # 保留原位置的版本化实验实现
├── reference_training/              # 保留原位置的底层管理器与历史合同
└── environment.py / recipe.py / …   # 保留 r5 历史实现与公共资产辅助代码
```

`scripts → task_registry → envs/chuanliantui → 对应版本实现` 是统一调用路径。
`task_registry.get_cfgs('chuanliantui', profile=...)` 每次创建独立配置，默认
`huananhu-v14`；返回 mjlab 环境配置和 PPO 字典。`make_env` 返回环境及其实际配置，
`make_alg_runner` 只构造 runner，不开始学习。注册表创建的环境会记录任务和配方，
构造 runner 时自动沿用；显式指定不同任务或配方会在构造前报错，即使提供了自定义 PPO 配置。
注册表之外创建的环境必须显式传入 `profile`，由调用方确保与环境匹配。
这些工厂针对从头创建环境；历史模型回放与续训走脚本入口，由原实现恢复课程阶段和核对合同。

```python
from ct_mjlab.envs import task_registry

env, env_cfg = task_registry.make_env('chuanliantui', profile='recover-r14', num_envs=2)
try:
    runner, train_cfg = task_registry.make_alg_runner(env, 'chuanliantui')  # 自动沿用 recover-r14
finally:
    env.close()
```

新增机器人在 `envs/<robot>/` 放环境、配置及命令执行，并在 `envs/__init__.py` 注册，
通过 `default_profile` 指定该任务的新环境默认配方。
新增串联腿配方在 `chuanliantui_config.py` 登记工厂、在 `profiles.py` 登记分流及迁移限制。
历史实验源码及 `reference_training/` 全部 `.py` 文件的哈希参与 checkpoint 合同，
本轮保留原文和导入路径；修改已有算法/奖励配方仍需显式版本与兼容性设计。
不复制 Isaac 的内嵌 `rsl_rl/`：mjlab 继续使用其独立环境中现有的 RSL-RL。

在仓库根可直接运行独立脚本；参数沿用 mjlab 的连字符形式。下列只查看帮助或做有限检查：

```bash
mjlab_training/.venv/bin/python -m ct_mjlab.scripts.train --help
mjlab_training/.venv/bin/python -m ct_mjlab.scripts.play --help
mjlab_training/.venv/bin/python -m ct_mjlab.scripts.check --num-envs 2 --steps 8
# 原有脚本仍可使用：
scripts/mjlab.sh check --num-envs 2 --steps 8
```

结构检查及其范围见 [构建与测试](../docs/ai/build-test.md#mjlab-结构与入口检查)。

## 配方与历史兼容

当前默认为 `huananhu-v14`，按用户选择恢复最早华南虎V14 Flat-v0配方，见 [恢复说明](../docs/interfaces/chuanliantui-mjlab-scut-v14-flat-v1.md)。入口 `src/ct_mjlab/experiments/huananhu_v14.py`；以下为显式选择 `recover-r14` 的历史说明。r15仍可显式选择，已有模型按原版本续训／回放。

独立 Python3.12 环境；显式r14加入前进速度±2.7 m/s、偏航速度±2π rad/s、50%航向控制，使用5～10 cm速度奖励高度门控，knee_limit权重保持0，当前拒绝隐式跨版本续训，初态沿用r10的50%普通／50%地面、地面机体零姿态。动态normal/recover和四腿两轮动作顺序继续使用，保留本仓实体闭链机器人／气弹簧资产。原生mjlab管理器、Actor35/Critic78、普通PPO。

配置／参数／命令见 [训练说明](../docs/mjlab-chuanliantui.md)，接口见 [r14](../docs/interfaces/chuanliantui-mjlab-wheelleg-recover-r14.md)。根目录使用 `scripts/mjlab.sh`，默认只交付训练命令。r11已完成1000轮；用户授权r12从其model_999追加2000轮，已在远端4090完成续训；r13已按用户授权从r11完成追加1000轮、累计2000轮，见 [运行记录](../docs/diagnostics/20261010-mjlab-r13-cloud-training.md)及 [效果](../docs/diagnostics/20261010-mjlab-r13-cloud-model1999-evaluation.md)。r14尚未训练。

`src/ct_mjlab/reference_training/` 为当前训练实现，SOURCE_IMPORT.json记录参考脏工作区逐文件指纹。`environment.py`/`recipe.py`/`cpu.py`保留r5回放。禁止覆写／删除根logs。

r6～r12历史模型按原动作顺序、mode、膝惩罚、初态分布及高度门控回放；当前检查见 [r13诊断](../docs/diagnostics/20261010-mjlab-r13-wide-height-gate.md)，r10远端1000轮训练及效果见 [评估记录](../docs/diagnostics/20261010-mjlab-r10-cloud-model999-evaluation.md)。
