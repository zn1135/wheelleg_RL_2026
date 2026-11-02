# 复旦行走：三个独立延迟／推扰开关

后续变更：按[GPU最终验证约定](20261102-fudan-gpu-validation.md)撤销CPU物理后端；本页Native/两引擎核对记录仅代表当时实现与检查。GPU历史、扰动与模型合同保持。

日期：2026-11-02。用户确认三个独立开关，默认全关。本次只有代码和无物理检查，不启动训练或动态评估，不修改H7。

## 配置与用法

直接修改 [config.py](../../mjlab_training/src/ct_mjlab/fudan_ppo_walk/config.py) 的 `FudanWalkCfg.domain_rand`，与复旦的 `domain_rand.push_robots` 配置写法一致。三个开关互相独立，不再提供这三项的命令行开关。

```python
class FudanWalkCfg:
    class domain_rand:
        randomize_action_delay = False  # 额外动作延迟
        delay_ms_range = [0., 10.]      # 毫秒，复旦参考范围，非本机实测

        randomize_sensor_delay = False # 额外传感数据延迟
        sensor_delay_ms_range = None   # 开启前填写确定的毫秒范围

        push_robots = False            # 主动推扰，与复旦同名
        push_interval_s = 7.           # 推扰间隔，秒
        max_push_vel_xy = 2.           # 冲量速度尺度，m/s
```

将对应项改成 `True` 即启用。传感延迟没有实测默认值，开启前必须填写 `sensor_delay_ms_range`；例如 `[2., 6.]` 只表示写法，不是本机实测结论。相同的两个数表示固定延迟。只修改范围不会自动启用开关。

预检、训练（含续训）和GPU评估都以当前代码配置为准；checkpoint仅记录原训练条件，不覆盖用户现在的配置。修改后重新启动命令生效，不支持在运行中热切换。内部 `DisturbanceConfig` 只负责校验和保存实际值，不是另一个用户配置入口。

续训若当前代码配置与源模型条件不同，恢复报告记录源条件与目标条件；必须先有同条件的物理预检，不能用全关预检批准开启后的训练。直接调用runner.load仍拒绝未声明的条件变更。旧checkpoint缺少记录时，原训练条件始终解释为全关，不受当前配置影响。

打开推扰后，预检至少覆盖第一次推扰及后续清零，仍受给定deadline约束；时间不够时中止，不自动追加预算。预检报告记录实际延迟拍数及各环境推扰次数；零动作有限性预检不是策略抗扰验收。

CPU评估读取GPU夹具保存的配置、实际延迟、推扰种子与力尺度，并核对当前代码配置；不一致时须先按新配置生成GPU夹具。导出JSON始终记录checkpoint的真实训练条件，不会被后来编辑的配置覆盖。

## 实际时序

### 动作延迟

策略仍为10ms一拍，PD／物理仍为2ms一拍。策略动作在每个PD拍进入队列，读取指定拍数前的动作，再计算PD；PD反馈仍使用当前关节状态。延迟以所有六电机共用的每环境单个值表示，没有沿用旧任务的腿／轮分离随机延迟。

每环境初始化采一次连续均匀毫秒值，用 `torch.round(ms/2)` 量化成2ms拍，半整数按偶数舍入；0～10ms的端点桶概率与中间桶不同，并非六个档位等概率。队列容量为最大拍数＋1，包含当前动作；例如10ms延迟时，前五个PD拍使用复位零动作，第六拍使用新动作。局部复位仅清选中环境的动作队列，保留初始化采到的延迟；原始策略动作仍用于动作变化奖励与“上一动作”观测。

### 传感数据延迟

在每个2ms物理拍结束，直接读取当前 `qpos/qvel` 和该拍电机角度差分速度，保存以下16个已经缩放的传感值：角速度3、重力方向3、四腿相对角4、六电机速度6。10ms策略观测从队列取指定数据龄的这16项。

命令3项和上一策略动作6项保持当前值；Critic特权输入、奖励、终止判断和PD反馈仍用当前状态。现有观测噪声在取出延迟传感值后添加，再按模型原有v2/v3规则更新历史与限幅；这里模拟额外数据龄，不声称模拟了传感器内部滤波或真实噪声生成过程。

局部复位用该环境的新鲜复位传感值填满传感队列，其他环境不变。没有为这项功能改变25／125输入、历史帧数、动作维度或网络权重。

### 主动推扰

依据固定源码 `plane/wheel_legged_gym/envs/base/legged_robot.py::_push_robots` 的方式：

1. 每环境从物理复位开始按2ms计数，到达间隔时触发；初始10ms零动作热步也计入，随机首轮episode进度不改变此物理计时。
2. 本机取当前批量机身body质量均值，计算每轴力尺度 `F = mean(base_mass) × max_push_vel / 0.002`，单位N。`max_push_vel` 是冲量的速度尺度，不是持续施力值，也不保证实际速度改变等于它。
3. 在机体系三个方向独立均匀采样 `[-F,F]`，旋转到世界系后将世界竖直分量乘0.5，作用于机身质心，不追加外力矩。
4. 只施加一个2ms物理拍，下一拍明确清零；CPU若用4×0.5ms积分，同一力保持四个积分子步。局部复位清除所选环境外力、计数和旧队列，并进入新的可复现随机序列。

间隔向上量化到2ms拍，7秒即3500拍。共享XML、PD增益、电机限矩、弹簧、奖励16项和PPO保持不变。机身质量使用本机实际批量参数；随机数序列及跨引擎求解器不可能因此称为复旦逐位复现，力的实际动态效果仍待验证。

## 版本、交付与边界

| 项目 | 内容 |
|---|---|
| 代码基线 | `zn1135/wheelleg_RL_2026`，`26_wheelleg`，基线 `b0def9d8267099619e6d32e3eb63ce1569f7ff82`＋本次工作区改动 |
| 配置标识 | `fudan-optional-disturbances-v1`，独立于原动作／reset-history合同 |
| 实现 | `config.py`用户配置；`disturbances.py`队列与推扰；`env.py`实际PD与采样；`native.py`独立CPU控制；`measurement.py`夹具导出 |
| 入口与保存 | `__main__.py`、`runner.py`、`precheck.py`、`evaluate.py`、`export.py`保存和核对实际条件 |
| checkpoint | 新增完整disturbances记录；旧模型缺字段按全关解释；改变开关不自动改变旧权重能力 |
| ONNX | 图结构不变；队列／推扰属于图外仿真，JSON记录训练条件；启用随机延迟后不再把延迟写死为0 |
| H7 | 代码未改；不要在真机人为补一份训练延迟。真实端到端数据龄和动作等待须实测，部署负责人核对待确认 |
| 物理与行为 | 本次没有新动态仿真、训练、行为视频或新模型；原有视频验收与模型结论保持原范围 |

开关用于可复现的仿真对照，不替代[实机延迟测量交付流程](../h7-latency-measurement-handoff.md)。本次不增加模型交付，因为没有训练新权重。

## 检查与验收

仓库根目录执行，输出须为新文件：

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.check_disturbances --output /tmp/fudan-disturbances-new-check.json
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.check_history_v3 --output /tmp/fudan-history-new-check.json
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.check_isolation
scripts/mjlab.sh fudan-ppo-walk train --help
```

已完成：[代码配置开关检查](../../artifacts/fudan-disturbance-switches-20261102/check-code-config.json)，包含CPU/CUDA队列与推力一致性、真实PD路径的0/2/4/6/8/10ms边界、传感槽位／Critic区分、局部复位、单拍外力清零、CPU夹具回放、保存恢复和不匹配预检拒绝。[31项原历史检查](../../artifacts/fudan-disturbance-switches-20261102/history-code-config.json)通过。所有检查实际物理步、网络前向和优化器更新均为0；Native控制流里的积分调用由测试替身替代，不作为物理仿真结果。

下一步优先在另行确认的动态预算内，只开一种机制做20环境可视预检：核对实际延迟拍数／推扰次数与请求一致、复位无残留且状态有限；之后再决定训练或冻结策略对照。不同开关组合分别保存新目录，完整行为与失败按原视频验收规则交用户确认。此建议不启动新预算。
