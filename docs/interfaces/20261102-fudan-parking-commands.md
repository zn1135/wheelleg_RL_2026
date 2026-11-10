# 复旦行走停车采样v1：代码准备，尚未训练

后续状态：用户已授权仅新增1000轮停车采样续训，4096环境、7000→8000。使用显式`--preserve-history-v2`保留源历史；默认新训练仍为v3，不自动迁移。预检须同时匹配历史、停车采样和延迟/推扰配置，旧缺采样记录的预检不能放行。新命令及实际状态见[执行记录](../../artifacts/fudan_ppo_walk/fudan-parking-resume7000-to8000-20261102/RUNNING.md)。本轮不额外训练原采样对照组，不用真值替换Actor，不自动开展策略评估。

2026-11-02，当前分支`26_wheelleg`，基于提交`7e89033`后的工作区修改。用户授权先诊断估速与零速输出，再补停车样本。GPU诊断使用原7000-v2模型；新增采样只影响后续训练，没有用它产生新模型或验收成绩。

## 依据

三组同初态GPU诊断显示：原模型将约−0.051m/s估成−0.023m/s；仅在停车稳定段换成同口径真实前向速度后，5秒后退从约25.5cm降至13.2cm，三轴真值后仍约12.8cm。估速有影响，但真值替换仍有零速残差。因此先补样本，保持原16项奖励。诊断不是部署修复，数据、窗口和重复误差见[报告](../../artifacts/fudan_ppo_walk/fudan-parking-diagnostic-7000-20261102/report/README.md)。

## 代码配置

修改[config.py](../../mjlab_training/src/ct_mjlab/fudan_ppo_walk/config.py)中的`FudanWalkCfg.commands`：

```python
standstill_env_ratio = .1
stop_env_ratio = .1
stop_drive_time_s = [3.,5.]
stop_speed_range = [.25,1.]
parking_heights = [.22,.26,.30]
```

比例按每次真实回合复位采样，不是每个策略拍重新抽取，也不保证每批恰好10%。配置含义：

| 回合类型 | 默认比例 | 指令行为 |
|---|---:|---|
| 普通行走 | 80% | 维持原vx/yaw/高度范围，每5秒重新采样 |
| 原地保持 | 10% | vx=0、yaw=0，一直保持至复位 |
| 行驶后停车 | 10% | 以0.25～1m/s行驶3～5秒，前后方向各半；之后vx=0、yaw=0至复位 |

两种停车回合在0.22/0.26/0.30m中均匀选择一个高度，回合内不被5秒周期重采样覆盖。普通行走高度仍连续均匀采样0.22～0.30m。停车不是关闭控制，原平衡策略和PD始终运行。

将两个比例都设为0，即恢复原连续命令采样；检查验证此时指令和随机数状态与原采样一致。不增加命令行参数。

独立`command_elapsed`计真实经过的10ms策略步，避免首轮随机episode进度把3～5秒驾驶段提前跳过。启动零动作热步计一次；第一轮因episode相位随机化仍可能是缩短回合，后续完整20秒回合通常有15～17秒停车段。提前终止时不保证停满。

## 接口与兼容性

| 项目 | 本次变化 |
|---|---|
| 命令配方标识 | 新`fudan-parking-commands-v1`；旧缺省记录为`fudan-uniform-commands-v1` |
| 用户配置／实现 | `config.py`；`commands.py`负责模式、时钟和采样；`env.py`接入实际复位／步进 |
| 输入／动作 | 仍为当前25、历史125、估速3、动作6；没有新增Actor模式位，零速仍由原命令字段表示 |
| 奖励／网络／优化器 | 16项奖励、PPO、网络、存储、PD／物理和初始姿态均未改 |
| 历史语义 | 未改；原7000仍v2，通用新训练入口仍默认v3，不能把采样变更和历史迁移混称单因素实验 |
| 固定GPU评估 | `resample=False`时不注入停车计划；夹具配置清空训练命令计数，保留原完整测试命令 |
| checkpoint／恢复／导出 | 保存实际`command_sampling`；恢复报告区分源与目标；ONNX JSON记录源模型训练采样条件 |
| 旧模型 | 权重仍可按原合同加载；新增代码不会让7000自动学会停车，也不覆盖旧模型／日志／视频 |
| H7 | 未改；没有新增部署输入。真值替换只在诊断脚本中，不写入正式Actor或固件。负责人核对仍待确认 |

本轮没有新权重，因此没有新的模型交付。实现者自查不代表独立复核或最终用户视频验收。

## 验证与复现

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.check_parking_commands --output /tmp/fudan-parking-commands-new-check.json
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.check_history_v3 --output /tmp/fudan-parking-history-new-check.json
```

输出需为新文件。正式检查覆盖实际复位／步进流程、三种模式、停止时刻、5秒边界、三高度、局部复位、固定评估指令不变、关闭后原随机序列、保存恢复来源及奖励／网络／物理函数不变。真实物理、网络前向和优化器更新均为0；[采样检查](../../artifacts/fudan_ppo_walk/fudan-parking-diagnostic-7000-20261102/parking-commands-height-check.json)、[27项历史检查](../../artifacts/fudan_ppo_walk/fudan-parking-diagnostic-7000-20261102/history-after-sampling-check.json)。

下一步需要独立训练预算。在保持同一历史版本、相同起点和新增轮数的前提下，对照“原采样续训”和“新增停车采样续训”；先不加停车奖励。若以7000-v2为基线，须使用明确支持v2且预检匹配的实验入口，不能自动用当前v3默认配置替代。检查三高度的原地／前后停车漂移、原有行走转向是否退化，最终观看完整视频。原验收标准未在本次代码准备中改变。
