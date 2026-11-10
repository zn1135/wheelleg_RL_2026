# 复旦PPO结构整理：保持mjlab训练行为

本轮仅整理结构与写法，不训练、不推进物理、不回放、不进行GPU行为评估。当前分支`26_wheelleg`，开始时没有训练/评估/诊断进程，只有读取8000轮日志的TensorBoard；原暂存与未提交改动保留。最新模型仍为8000轮、v2历史，SHA256为`39fa104d176f82a33db8317a75b4db28749362daf6bc4d2e06cb83d2c5ae32c9`。

## 实际链路

参考了`wheel_legged_gym/scripts/train.py`、`utils/task_registry.py`、`envs/wheel_legged/wheel_legged_config.py`、基础任务环境和内嵌`rsl_rl`的实际调用关系，只读，没有导入Isaac或历史logs源码。

整理前：`mjlab.sh fudan-ppo-walk train → fudan_ppo_walk/__main__.py`直接创建`FudanEnv → FudanRunner → PPO`。已有通用注册器只是挂了工厂，复旦训练绕过该注册器。

整理后：`mjlab.sh train --task fudan_ppo_walk → ct_mjlab/scripts/train.py → task_registry.get_cfgs/make_env/make_alg_runner → FudanEnv/FudanWalkCfg/FudanWalkCfgPPO → FudanRunner → PPO`。注册器实际解析任务类并调用构造器；任务环境仍负责自己的状态/动作/观测/复位循环，不套另一个通用PPO环境循环。复旦分支不额外调用通用Torch后端配置函数，保留原入口的设置与RNG顺序。

其他训练线仍沿原分支。SMP的原`train()`主体AST完全相同，通用注册器对未提供专用任务类的任务保留原环境包装及runner创建逻辑。

## 文件迁移表

位置均相对于`mjlab_training/src/ct_mjlab/`：

| 旧位置 | 当前唯一实现 | 职责/处理 | 受影响入口 |
|---|---|---|---|
| `fudan_ppo_walk/env.py` | `envs/fudan_ppo_walk/fudan_ppo_walk.py` | 状态、动作、观测、奖励、命令与复位 | 训练、预检、GPU评估、诊断 |
| `fudan_ppo_walk/config.py` | `envs/fudan_ppo_walk/fudan_ppo_walk_config.py` | `FudanWalkCfg`及`FudanWalkCfgPPO`集中配置，实例配置互不共享可变字段 | 注册、训练、工具 |
| `fudan_ppo_walk/commands.py` | `envs/fudan_ppo_walk/commands.py` | 停车采样/时钟；消费者读取实际任务配置 | 环境、预检、诊断 |
| `fudan_ppo_walk/disturbances.py` | `envs/fudan_ppo_walk/disturbances.py` | 三个开关、队列、推力；旧记录默认值仍属固定版本语义 | 环境、恢复、评估 |
| `fudan_ppo_walk/reset_geometry.py` | `envs/fudan_ppo_walk/reset_geometry.py` | 同一PD中心闭链几何求解，无积分 | 初始化、预检、诊断 |
| `fudan_ppo_walk/parking_integral.py` | `envs/fudan_ppo_walk/parking_integral.py` | 独立组件，默认关闭，不进入训练/Actor/ONNX | 积分诊断及正式检查 |
| `fudan_ppo_walk/runner.py` | `rsl_rl/runners/fudan_on_policy_runner.py` | 采样、调度、日志、计数、预算、保存恢复 | 注册训练、续训 |
| `fudan_ppo_walk/ppo.py` | `rsl_rl/algorithms/fudan_ppo.py` | PPO及Encoder监督，原梯度/更新顺序 | runner |
| `fudan_ppo_walk/network.py` | `rsl_rl/modules/actor_critic_sequence.py` | 原网络、分布、初始化与推理 | PPO、冻结模型、导出 |
| `fudan_ppo_walk/storage.py`、`utils.py` | `rsl_rl/storage/fudan_rollout_storage.py`、`rsl_rl/utils.py` | 原存储、mini-batch与轨迹计算 | PPO |
| 独立`__main__.py`训练实现 | `scripts/train.py`的复旦分支 + `utils/task_registry.py` | 仅一套训练链路，旧命令转入该链路 | 新旧训练命令 |
| `fudan_ppo_walk/contract.py` | 保留 | 接口/版本和旧模型固定配方；网络构造读取集中配置，加载时校验冻结网络合同 | 保存、恢复、模型加载 |
| `measurement.py`、`policy.py`、`export.py`及`protocol.py` | 保留工具位置，依赖改为新生产路径 | 不另造训练环境/算法；协议和历史评分规则保持 | GPU评估、评分、视频、导出 |

旧环境/算法文件仅有导入兼容，不含另一份类或训练函数。旧`fudan-ppo-walk train`命令只转入通用`train.py`，用于现有命令脚本兼容；旧`fudan-ppo-walk export`继续调用同一个导出函数。没有恢复CPU物理入口。未支持把复旦模型送入其他任务的通用play/export模型格式；复旦继续使用专用GPU评估/导出工具和对应合同。

## 日常修改位置

| 修改内容 | 文件/字段 |
|---|---|
| 奖励权重 | `envs/fudan_ppo_walk/fudan_ppo_walk_config.py::FudanWalkCfg.rewards.scales` |
| 奖励公式与计算次序 | `envs/fudan_ppo_walk/fudan_ppo_walk.py::FudanEnv.reward` |
| 命令范围、课程、停车采样 | 配置`commands`；实现为任务环境`resample/reset_idx`与`commands.py` |
| 观测、噪声、裁剪、历史 | 配置`normalization`；实现为环境`frame/observations/reset_idx/step` |
| 随机化与延迟/推扰 | 配置`domain_rand`；环境`randomize`、`PhysicalAction`与`disturbances.py` |
| PD与控制周期 | 配置`control`、`sim`；实际交给`PhysicalActionCfg`与mjlab仿真配置 |
| PPO、学习率、网络参数 | `FudanWalkCfgPPO.algorithm/policy`；算法/网络公式在`rsl_rl`对应文件 |
| 采样长度、存盘、阶段间隔、时间余量 | `FudanWalkCfgPPO.runner` |
| 训练预算、实际更新计数与恢复 | 入口校验预算；runner负责停止、计数和保存，不在PPO里另建训练流程 |
| 独立停车积分 | `FudanWalkCfg.parking_integral`，仅诊断显式启用分支 |

历史`PPO_CONFIG`和`NETWORK_CONFIG_V1`属于已有模型的固定合同，不是当前可调配置。不能把这些校验字典改成按当前配置生成，以免配置漂移后悄悄接受旧模型。25维当前帧、125维历史、6动作、141+3 Critic与Encoder接口保持；v2保留/向v3迁移仍需显式标志，默认新训练v3。

下次训练必须另有明确预算、匹配预检和新log-dir。当前已有日志可用仓库虚拟环境启动TensorBoard（本轮没有启动或重启它）：

```bash
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000 --port 6006
```

## 行为保持与正式验证

- 整理前完整源码保存在独立产物目录，保留工作区已有修改。新旧网络、存储和轨迹工具计算AST一致；PPO只把原更新位置的提交转为显式runner回调，取消运行时替换Adam.step。
- 同输入/状态/RNG逐值比较：16奖励、观测/Critic/历史、初始化、原地/停车命令、局部复位、课程、动作限幅、500Hz关节差分、传感打包、域随机化、初始化网络及8000轮动作/估速/动作采样。
- 执行真实PPO损失、反向传播、mini-batch与Encoder监督，但提交用检查记录器替代，参数始终不变；损失、每次梯度、RNG、学习率及先20个PPO后20个Encoder的顺序逐值一致。这不是训练。
- 注册器检查实际创建环境/runner/PPO，仅替代底层物理后端；入口检查观察真实`train.py → registry → learn_until`调用。配置检查更改代表字段并验证环境/网络/PPO/存储消费者确实读取它。
- 复用历史v2/v3、显式保留/迁移、源版本/配方/部分更新拒绝、两Adam/学习率/CPU及CUDA RNG恢复、追加轮数、墙钟停止和真实TensorBoard事件检查。积分r2的归档恢复、差值留证及步后反馈方法AST相同，并复用其72项无物理检查。
- 全套外层保护禁止构造真实`ManagerBasedRlEnv`、MuJoCo积分和真实Adam.step。允许CPU冻结网络前向与无提交的梯度检查；静态几何不推进时间。
- [逐文件旧/新哈希记录](20261103-fudan-source-migration.json)包括迁移实现、兼容模块、新增文件，以及整理前已有停车/续训改动的明确来源链。旧快照和未知新哈希均严格校验，不能忽略全部差异；旧CPU原始源码只在历史归档保留。

正式入口：`scripts.agent.fudan_ppo_walk.check_structure_suite --output <新检查目录>`。本轮执行结果、计数、原8000模型/归档输入/资源/H7哈希以及各检查日志见[交付证据](../../artifacts/fudan_ppo_walk/fudan-structure-alignment-20261103/README.md)。

实际物理行为、GPU轨迹、实际训练吞吐量和同墙钟预算的实际训练轮数未重测，不能声称训练或停车效果通过。8000模型及积分r2历史结论保持；本轮没有重跑积分、调整增益、修改H7或导出新ONNX，也没有Git提交/推送。
