# mjlab 100 Hz策略／500 Hz内环试验

## 来源与范围

| 项目 | 内容 |
| --- | --- |
| 日期、实施人与复查 | 2026-10-23，Codex；H7部署负责人及独立复查待确认 |
| 机器人／任务 | 当前华南虎multimode串联腿，`chuanliantui` |
| 用户要求 | 先提交当前状态，再试用旧Isaac串联腿的网络100 Hz／内环500 Hz配置；随后要求独立CPU sim2sim同样提频并回放 |
| 训练仓与分支 | `https://github.com/zn1135/wheelleg_RL_2026.git`，`26_wheelleg` |
| 变更前版本 | `27bbb4d0cd3587fae767fff24a969041af36115d`；85文件备份提交已完成，未推送 |
| 实验版本 | 上述提交加本次未提交差异，完整证据见下方目录 |
| 部署代码版本 | 同仓上述提交，`deployment/H7_RL_ros`本次无改动；板端仍是原50 Hz实现 |
| 旧接口 | [multimode 39/82接口](chuanliantui-huananhu-multimode-replay-v1.md) |
| 试验记录标识 | `mjlab-policy100hz-20261023`；观测／动作契约字符串保持原值，不能仅按字符串判定时序兼容 |
| 权威实现 | `multimode_env_cfg.make_multimode_env_cfg`覆盖`cfg.decimation = 5`；独立CPU入口`DT=.002, DECIMATION=5` |

## 接口差异

| 项目 | 变更前 | 本次试验 |
| --- | --- | --- |
| mjlab物理／实体PD | 2 ms，500 Hz | 保持 |
| mjlab策略 | 20 ms，50 Hz，decimation=10 | 10 ms，100 Hz，decimation=5 |
| 每动作PD子步数 | 10 | 5 |
| 生效入口 | mjlab训练、play、check、eval | 均使用100 Hz |
| 独立CPU | 20 ms策略／2 ms内环 | 后续同步为10 ms／100 Hz策略、2 ms／500 Hz内环 |
| H7 | 20 ms策略／2 ms内环 | 保持50 Hz／500 Hz，本次不修改 |
| 观测、动作及模型张量 | actor39／critic82／action6；float32，固定缩放、模式槽及顺序 | 保持，原网络权重及ONNX未改 |
| 上一动作观测 | 上次策略原始输出 | 含义保持，但两次输出间隔缩为10 ms |
| 坐标、默认角、角度回绕、速度采样 | 模型原生实体反馈，定义见旧接口 | 保持 |
| PD、目标缩放、限幅与XML | 原四腿位置PD／双轮速度比例控制 | 保持，未改物理资产或标定 |
| 动作等待 | 训练腿／轮独立6/8/10 ms；play固定2 ms | 保持；10 ms等待现在跨越一个策略周期 |
| 观测延迟、随机化、电机响应 | 原配置，电机响应候选默认关闭 | 保持 |
| 趴地接管 | 起始0.2秒零力矩 | 保持0.2秒，对应20个新策略步 |
| 起立、恢复、复位与停止 | 按环境实际step_dt计算各时间窗 | 保持原秒数和初态分布 |
| PPO参数和奖励 | 原配置 | 不额外重标定折扣、rollout步数或动作差分奖励；本次未学习 |

策略周期缩短会改变单位物理时间内的观测／动作次数及后续训练采样时长，不保证旧权重变频后更好。旧50 Hz checkpoint可按原张量格式加载进行本次对照，其行为需重新测量；它不会因此变成按100 Hz训练的模型。若后续按100 Hz训练，导出和部署必须交付实际时序，不能直接套用H7当前50 Hz实现或旧50 Hz行为结论。

## 配套检查与同步

`eval.py`原有默认1000步及报告中写死的20 ms已改为读取任务实际周期；默认仍观察20秒，在100 Hz下为2000步。`check.py`和零速保持报告记录实际物理／策略步长。现有时序检查核对mjlab与CPU均为100 Hz，并验证实际每动作5个PD子步、0.2秒接管保持以及原动作延迟／复位行为。

CPU后续提频时同步修正了两处写死的10步趴地门控（PD请求和电机响应状态清零），用0.2秒除以实际策略周期得到20步；不能把零力矩保持缩短到0.1秒。`policy_active`继续按训练的float32时间乘积判断，20步时该乘积略小于0.2，21步置位；整数力矩门控仍在0.2秒结束。腾空确认保留20 ms，恢复稳定确认保留0.5秒。动作等待仍2 ms，前两次网络动作首次进入PD为2/12 ms。回放实例的`h7_timing.simulation.policy_dt_s`记录有效10 ms，共享JSON中的H7原50 Hz实测来源不改。

| 范围 | 本次状态 | 负责人 |
| --- | --- | --- |
| mjlab配置、评估报告、有限检查与文档 | 已修改，验证证据见下方 | RL侧／Codex |
| H7固件、策略节拍、模型生成与烧录 | 本次不修改，不能用于100 Hz同配置验收 | H7部署负责人核对待确认 |
| 独立CPU回放 | 后续同步100 Hz／500 Hz，历史50 Hz证据保留 | RL侧 |

## 验证与回退

输入模型为[已交付model9845](../model-deliveries/20261022-model9845-h7-import.md)；本次只做固定权重回放，不训练、不改模型、不操作硬件。

证据目录：`artifacts/mjlab-policy100hz-20261023-fP6BGy/`。包含50 Hz改动前配置／评估脚本、两种频率各seed42/43的零速保持报告与轨迹、正式有限检查日志、CPU回归以及默认eval四环境2000步／20秒的实际报告。每种频率的零速保持为四高度×普通／趴地×两初态，共32组，按原20秒／最大位移5厘米判据测量。所有命令退出0；本次改动的源文件、差异、命令与校验值保存在证据目录的`source-after/`、`experiment.diff`和`manifest.json`，具体行为结果记录在[项目实验记录顶部](../ai/project-log.md)。

```bash
# 时序、延迟、部分复位与有限状态检查，不学习。
mjlab_training/.venv/bin/python scripts/agent/check_h7_simulation_latency.py \
  --out artifacts/<新目录>/timing-check.json

# 现有模型的有限物理／张量检查，不学习。
scripts/mjlab.sh check --num-envs 4 --steps 32 \
  --checkpoint deployment/H7_RL_ros/models/model9845-20261022/model_9845.pt

# 正式零速保持测量；seed42与43分别保存新文件。
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_zero_hold.py \
  --checkpoint deployment/H7_RL_ros/models/model9845-20261022/model_9845.pt \
  --envs-per-case 2 --seed 42 --output artifacts/<新目录>/hold-100hz-seed42.json
```

CPU后续同步的证据另放`artifacts/cpu-policy100hz-20261023-2NzBG4/`，含提频前CPU／检查脚本快照、正式CPU检查、时序检查、model9845 ONNX原生CPU回放报告和原生窗口日志；行为结论统一见[项目实验记录顶部](../ai/project-log.md)。原生交互回放命令：

```bash
mjlab_training/.venv/bin/python sim2sim/mj_sim2sim_huananhu_multimode.py \
  --checkpoint deployment/H7_RL_ros/models/model9845-20261022/model_9845.pt \
  --onnx deployment/H7_RL_ros/models/model9845-20261022/model_9845_opset13.onnx \
  --initial-pose ground --render --teleop --seconds 0
```

回退需要将mjlab的`cfg.decimation`与CPU的`DECIMATION`一并恢复为10并重新创建环境；CPU趴地保持步数会按0.2秒自动换算。保留的85文件提交用于核对提频前完整版本，不覆盖历史模型、日志或本次诊断产物。回退后应同步恢复检查期望值和文档时序。


## 2026-10-23后续：model13842导入USB H7候选

按用户要求导入最新13842，本次权威实现为`deployment/H7_RL_ros/imcalib/config/machine_config.h`大机MACHINE_POLICY_DIV20→10；生成网络替换为13842，策略100Hz／PD500Hz，与mjlab/CPU名义时序一致。小机原100Hz／1kHz保持，UART工程不改。本节之后的当前状态以[13842交付记录](../model-deliveries/20261023-model13842-h7-import.md)为准；上文“本次不修改H7”只描述先前提频试验。

观测/动作/坐标/缩放/历史含义/默认角/PD/限幅/机械资产及恢复停止条件不变；上一动作观测间隔随策略缩为10ms。板端未人为加入训练随机等待或随机偏置，命令仍0.27～0.34m。旧50Hz权重张量可加载，但行为结论不能跨时序复用；回退须配套还原旧权重与分频20。

导出256组、生成C64组及参数字节、大小机108编译单元/链接、分频与200/500ms恢复有限检查通过；用户本次确认动力断开后，大机于15:53:22烧录/校验/复位成功；未使能或实测控制。H7负责人/板端100Hz最坏时序与真机核对待确认。源码/配置/差异、模型与固件哈希见交付记录和`artifacts/model13842-h7-20261023/`。
