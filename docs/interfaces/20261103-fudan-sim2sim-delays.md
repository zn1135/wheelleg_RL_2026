# 接口记录：复旦sim2sim独立动作/传感延迟

2026-11-10后续用户指定仅动作改为10ms，传感保持训练2ms/GPU仿真4ms；下文原设置与结果保留历史含义。[当前设置](20261110-fudan-action-delay10ms.md)。

后续训练侧已依据板端首测改为2+2ms，sim2sim仍为4+4ms；下文训练全关只代表本记录原日期。[最新复查](20261103-fudan-measured-training-delays.md)。

后续实际复测已完成：[零延迟/4+4ms九停车对照](../../artifacts/fudan_ppo_walk/fudan-parking-delay-replay-20261103/README.md)，末段均9/9，但4+4ms高.26后退起步新增触地约591N/26°，高.30一例腿RMS增大5.7倍；不认定全面无退化，待用户完整视频验收。下文“未动态验证”保留其准备阶段历史含义。

| 项目 | 内容 |
|---|---|
| 日期、实现、复查 | 2026-11-03；Codex；H7负责人待确认 |
| 任务、模型 | `fudan_ppo_walk`，当前选用8000-v2＋停车+2°；GPU sim2sim |
| 旧条件 | 训练/预检/评估共读`domain_rand`，全关；[原队列定义](20261102-fudan-disturbance-switches.md) |
| 新条件 | 用户选择sim2sim动作固定4ms、传感器固定4ms；只为诊断起点，不是真机实测 |
| 接口标识 | `fudan-optional-disturbances-v1`及25/125/6保持；实际条件记录有变化 |
| 代码 | `26_wheelleg`，HEAD `c3c8fdffe0739aed314fb71c69f2a2d3135b3fb9`＋未提交修改；[执行源指纹](../../artifacts/fudan_ppo_walk/fudan-sim2sim-delays-20261103/source-sha256.json) |
| 权威实现 | `FudanWalkCfg.sim2sim`、`disturbances.from_sim2sim_config`、`scripts.agent.fudan_ppo_walk.evaluate.gpu` |
| 部署 | H7未修改/交付，负责人同步及板端/真机验证待确认 |

## 配置与实际调用

集中修改[任务配置](../../mjlab_training/src/ct_mjlab/envs/fudan_ppo_walk/fudan_ppo_walk_config.py)的`FudanWalkCfg.sim2sim`：

```python
class sim2sim:
    randomize_action_delay = True
    delay_ms_range = [4., 4.]
    randomize_sensor_delay = True
    sensor_delay_ms_range = [4., 4.]
```

上下限相同即固定延迟；两个开关独立。关闭两开关可回到零额外延迟。`domain_rand`仍是训练/预检配置，动作和传感延迟保持关闭；没有根据sim2sim配置偷偷修改训练条件。推扰仍按原`domain_rand.push_robots`配置，本次保持关闭。

GPU评估的默认选用方案、显式检查点两种入口都读取`from_sim2sim_config()`，把实际条件传给`FudanEnv`。模型记录仍反映原训练的零额外延迟，`protocol.json`分别保存`disturbances`和`training_disturbances`；夹具保存实际每环境`action_delay_steps=2`、`sensor_delay_steps=2`。

## 时序和字段

| 项目 | 实际定义 |
|---|---|
| 策略/物理/PD | 原10ms/2ms/2ms保持 |
| 动作 | 新动作在500Hz队列中等待两个物理拍，再进入原PD；关节反馈仍是当前值 |
| 传感 | 策略读取两物理拍前的角速度3、重力3、四腿角4、六电机速度6，共16项 |
| 当前字段 | 命令3项、上次发出的策略动作6项保持当前值；Critic、奖励、终止和PD反馈不加传感延迟 |
| +2°参考 | 先交付延迟传感帧，再按各策略/历史帧时间应用原+2°参考；不多推进历史、不重复延迟动作 |
| 复位 | 所选环境动作队列清零、传感队列以新鲜复位值填满，其他环境保持；固定两拍延迟保留 |
| 模型/物理 | 原8000权重、25/125/6、PD、XML、动作尺度均不变；不重训、不导出新ONNX |

这里只模拟额外执行等待和数据龄，不声称包含传感器滤波、传输抖动或实机完整端到端延迟。已有三维窗口播放保存的旧零延迟轨迹，不会随配置编辑变成新实验；带延迟结果必须重新生成GPU记录。

## 验证与范围

- [30项sim2sim独立检查](../../artifacts/fudan_ppo_walk/fudan-sim2sim-delays-20261103/sim2sim-check.json)：真实PD/传感方法配合合成状态，核对4ms两拍、独立开关、不改训练、+2°对延迟帧的处理、复位以及默认/显式GPU入口读取与记录条件。
- [121项既有队列检查](../../artifacts/fudan_ppo_walk/fudan-sim2sim-delays-20261103/core-check.json)：CPU/CUDA张量、0/2/4/6/8/10ms动作到PD、传感槽位、局部复位和夹具/训练恢复条件。
- 全部真实物理步、网络前向、优化器更新均0。新带延迟停车/平滑性、原45工况回归与正式987尚未执行；之前8000+2°九停车9/9属于零额外延迟，不移作本条件成绩。
- [原模型选用记录](../model-deliveries/20261103-fudan-8000-pitch2-selection.md)仍有效，新增条件动态结果待测。H7不应人为叠加仿真的延迟，真机数据龄/执行等待应另行实测。

下一步在另行授权的动态窗口内生成零延迟/固定4+4ms的同初态记录，比较九停车的偏移、均速、峰速和腿部速度RMS；不以队列检查代替行为验收。
