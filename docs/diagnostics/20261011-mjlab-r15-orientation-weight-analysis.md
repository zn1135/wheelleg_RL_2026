# r15续训姿态权重收紧与倾斜分析

## 用户选择与改动

用户明确选择「只调权重：保留公式，加大orientation权重」。实现独立profile `getup-r15-orientation`，接口 `chuanliantui-mjlab-getup-curriculum-r15-orientation-weight-v1`，默认r15 v3和历史模型回放仍沿用原配置。

| 阶段 | 原orientation权重 | 新权重 |
|---|---:|---:|
| 扶正 | -2.6 | -30 |
| 站起 | -2 | -30 |
| 行走 | -15 | -30 |

奖励公式完全复用原v3 `RewardState.compute`，未添加平方倾角公式、15°额外惩罚或硬终止。其他奖励、PPO、机器人、控制、初态、随机化、课程、normal/recover、辅助力退火和失败判定保持原参数。这里的15°仍是课程晋级/恢复指标门槛，不是新奖励中的分段阈值。

实现：[getup_orientation.py](../../mjlab_training/src/ct_mjlab/experiments/getup_orientation.py)，CLI增加选择与显式迁移标志：[cli.py](../../mjlab_training/src/ct_mjlab/cli.py)。原 [getup_curriculum.py](../../mjlab_training/src/ct_mjlab/experiments/getup_curriculum.py) 未修改，SHA256仍为 `9725ffa5830d9346fb77a491a6a9f312a1dd54d6d88cc628fe6611e63c470e1c`。

## 倾斜现象的直接证据

续训v3累计5000轮、无辅助力、无随机化、确定性actor的64普通/64地面初态回放：持续站立均0/64，地面仅1/64曾短暂严格恢复。阶段仍为扶正。

重新读取已保存GPU回放qpos，不进行仿真积分；样本0最后5秒51帧的总倾角（由root四元数计算，不把roll/pitch相加）：

| 初态 | 总倾角平均 | 平均root高度 | height贡献 | orientation贡献 | 相比完全正立少获得 |
|---|---:|---:|---:|---:|---:|
| 普通 | 25.474° | 0.271316 m | 3.95636 | +0.201227 | 0.058773 |
| 地面 | 20.450° | 0.265690 m | 3.85891 | +0.232980 | 0.027020 |

贡献均为原扶正阶段加权值、尚未乘0.02控制步长。这是视频样本0的测量，不是64环境的平均姿态；完整批次行为见[两组评估](20261011-mjlab-getup-r15-v3-two-groups-evaluation.md)。

## 原公式为何容许这种解

原公式：

```text
xy = gx² + gy²
raw_orientation = 0.02*xy + 2*(gz+1)²
                  - clamp(exp(-xy/0.25), 0, 0.1)
```

当总倾角为theta、投影重力单位化时，gz=-cos(theta)，xy=sin²(theta)。从正立到约49.34°，指数项仍被截断到同一个0.1常数。小角度附近：

```text
raw_orientation ≈ -0.1 + 0.02*theta² + 0.5*theta⁴
```

因此小倾角的二次项系数很小，主要姿态代价到较大角度才增强。判断姿态激励应看相对正立的奖励差，不能仅因倾斜时贡献为正就认定公式直接奖励倾斜。

### 与高度奖励的量级比较

| 总倾角 | 原扶正姿态损失，权重-2.6 | 新姿态损失，权重-30 |
|---|---:|---:|
| 15° | 0.009521 | 0.109855 |
| 20° | 0.024995 | 0.288405 |
| 25° | 0.054934 | 0.633858 |
| 30° | 0.106336 | 1.226952 |

`height=4*exp(-abs(h-0.27)/0.12)`；从5毫米高度误差改善到0能增加0.163242，约为旧25°姿态损失的2.97倍。所有项都乘相同0.02，量级关系不变。这个比较证明原目标函数的姿态与高度激励比例较弱，不证明机器人从倾斜变正立必然损失5毫米高度。

新-30让25°姿态损失提高为0.633858，超过上述5毫米高度收益；在15°仍只有0.109855，所以只加权重不能保证硬性满足15°。不同机械姿态的实际高度、载荷和动作代价也会变化，最终效果必须经续训和无辅助回放确认。

原公式包含常数项，增大权重也把完全正立时的orientation贡献从0.26提高到3.0。新旧训练的总reward绝对值不能直接当成效果改善证据，应比较无辅助力的姿态、持续站稳率和课程晋级。

## 其他助长持续倾斜的条件

1. 扶正阶段高度和轮载荷权重各4，姿态原权重仅-2.6。轮载荷项奖励双轮合计承重接近整机重量，不直接约束姿态；两轮同时接触也不要求正立。因此倾斜但达到高度、承重的姿态仍可能有较高总回报。
2. 课程15°门槛只控制是否升级，没有直接进入PPO奖励。训练可持续停留在扶正阶段；高reward不代表达到了课程门槛。
3. 前两阶段接触或倾倒允许整个20秒探索。原-200/-100 termination对触地/倾倒不生效，数值异常时奖励又清零，超时不是失败；不会额外惩罚这种未达严格站立条件的平衡。
4. 5000轮模型的续训起点学习率为1.5e-5，末轮动作标准差约0.08（训练日志）。它已有稳定的倾斜策略，在较弱姿态激励下继续训练可能只细调原解。未测得此项的独立因果贡献，不能把优化器或探索作为唯一原因。
5. 本次5000轮组辅助力预算和实际辅助力均0，回放也关闭辅助力，所以当前倾斜不能用「仍被辅助力托住」解释。删除pose_bias也未解决持续扶正。

综合证据支持「原奖励允许高度较准的倾斜局部解」这一推断；没有完成接触、关节阻尼、可达姿态或策略动作的因果消融，不能声称机械参数或旧默认角惩罚是唯一根因。

## 显式续训与历史兼容

- v3源模型必须使用 `--training-profile getup-r15-orientation --resume-orientation-weight`；不带标志拒绝跨配方续训。
- 仅接受已知v3模块哈希与完整控制/观测/环境契约，迁移不允许附带其他物理或重置变化。
- 恢复网络、探索分布参数、Adam、学习率、累计更新与课程；辅助力继续累计进度，5000轮续训不会重新打开辅助力。
- 保存新的interface和migration_history。新模型自动使用新profile回放；旧v3模型使用原profile。旧v1/v2迁移标志不能与此迁移混用。
- 尚未执行本次PPO训练。不能把有限仿真检查当成学习效果已改善。

新增有限检查入口：[check_chuanliantui_orientation_weight.py](../../scripts/agent/check_chuanliantui_orientation_weight.py)，覆盖实际三阶段奖励、dt、旧公式、其他项不变和完整状态迁移。

远端检查源码目录：`/root/gpufree-data/wheelleg-remote/releases/20261011T100539Z-getup-r15-orientation-weight`。原训练release和历史logs保留。

## 实际检查结果

RTX4090、既有Python3.12/mjlab运行时，20环境，三个阶段各64步：观测35/78、奖励有限，数值异常均0；旧orientation原始公式、其他阶段权重和一次dt缩放均通过。

从实际v3 `model_4999.pt` 在4096环境进行迁移检查，无PPO更新：actor、critic和Adam逐项完全一致；恢复累计5000轮和原课程，实际学习率为3.375e-5；辅助力预算和实际力均0；新profile保存/再次加载通过；拒绝v3隐式迁移、未知源哈希以及旧profile读取新模型。

恢复后 `common_step_counter=120000`、`training_iteration`偏移=0，二者合成120000/24=5000。检查累计更新必须使用合成值，不能把偏移单独当作已完成轮数。

初次检查暴露两个检查脚本问题：mjlab构造环境后会更新内部场景配置，需在工厂输出层核对唯一配置差异；累计更新应核对偏移与控制步的合成值。修正检查脚本后第三次完整检查退出0，未改变奖励/控制内核。失败日志保留在远端job中。

CLI `check --training-profile getup-r15-orientation --num-envs 20 --steps 1` 退出0、观测/奖励有限、数值异常0。完整结果：[check.json](../../artifacts/getup-r15-orientation-weight-validation-20261011/check.json)、[CLI输出](../../artifacts/getup-r15-orientation-weight-validation-20261011/cli-check.log)、[实际回放姿态贡献](../../artifacts/getup-r15-orientation-weight-validation-20261011/orientation-analysis.json)、[权重曲线](../../artifacts/getup-r15-orientation-weight-validation-20261011/weight-curve.json)、[92个源码/资产哈希](../../artifacts/getup-r15-orientation-weight-validation-20261011/source-state.json)。取回的ZIP、10个payload、本地与远端92个源码/资产文件校验全部一致。

证据ZIP SHA256：`2f76be395d0b592d2f58c590fac28f62bcad863c93b26d7587c0172943fdefd7`。其中包含核验的最终三个代码文件；初始上传patch是检查脚本修正前的版本，最终代码以 `verified-code/` 与源码哈希清单为准。未启动新的训练，未启动TensorBoard，未提交Git。
