# 轮角39 PPO：移植Isaac串联腿起立奖励

2026-10-26，按用户“将奖励改为Isaac Gym中的chuanliantui起立”新增实验配置。仅移植奖励，训练仍暂停。本轮未执行任何PPO／BC／SMP参数更新。

## 来源与选择

来源为当前工作区的 `wheel_legged_gym/envs/chuanliantui_standup/chuanliantui_standup_config.py`、`chuanliantui_standup.py` 及 `envs/base/legged_robot.py` 中实际继承的奖励。采用**解锁前**配方；不搬入累计4096回合达到30%成功后切换0.22m、orientation权重-10和高度sigma=0.0025的课程。

移植模块：`mjlab_training/src/ct_mjlab/smp/isaac_standup_rewards.py`。`WEIGHTS`列出全部19项，`reward_values`计算原始奖励，`RewardState`读取当前机器人数据并保存各环境的着地／站稳历史，`configure_rewards`替换旧任务奖励。

| 奖励项 | 权重 |
|---|---:|
| 前向速度跟踪／增强 | 各0.2 |
| 偏航速度跟踪／增强 | 各0.2 |
| 高度、连续站稳 | 各1 |
| 高度渐进姿态惩罚 | -1 |
| 左右虚拟腿摆角差 | -3 |
| 关节软限位 | -1 |
| 膝端点0.05rad余量 | -0.3 |
| 机身接地／离地 | -0.3／0.2 |
| 双轮同时无支撑 | -0.3 |
| 竖直速度、横滚／俯仰角速度 | -1／-0.2 |
| 前链髋膝速度、六电机力矩 | -0.00005／-0.0001 |
| 六路动作一阶差分、四腿动作二阶差分 | 各-0.01 |

原任务的termination、旧华南虎奖励和停车位置／减速项均不混入。未新增终止惩罚，源配置也无非零termination奖励。原SMP项独立保留，权重仍可用既有`smp-weight`设为0或1。

与Isaac一致：跟踪、速度／力矩／动作差分及膝余量项由“曾连续站稳0.5秒”启用；持续稳站项依赖当前连续站稳时长，失稳即归零。高度项使用sigma=0.01，机身接触力按0～5N连续门控，接地时最低保留20%高度奖励。各项先加权限幅到±1，再统一乘一次策略dt=0.01；负总奖励不裁零。

## 必须保留的映射差异

- 高度命令仍为当前SMP的0.33648m，不是源任务0.30m。奖励站稳条件沿用“比目标低2cm”的裕量，即0.31648m，重力z≤-0.9、机身接触力≤0.1N、已轮触地。该奖励门控独立于原mjlab恢复状态和正式v3.1验收，不修改两者。
- 初态、前0.2秒电机等待、39／82维观测、100Hz策略／500Hz控制、2／4ms动作延迟不变；前0.2秒任务奖励也不记入，继续保留现有学生流程。未移植Isaac首次轮落地才开始采样的流程。
- 当前闭链XML用实际髋到轮心向量计算机体系虚拟腿摆角，不复制旧URDF的角度零点。关节速度与软限位仍取前链髋膝 `lf0,lf1,rf0,rf1`；97%软限位只用于奖励，不改物理限位。膝端点取当前XML硬限位。
- 力矩项取当前实际六路电机广义力矩，不把旧Isaac代理的膝力矩当成当前后髋电机力矩，也不计气弹簧力矩。轮支撑使用netforce世界系z分量取反后>1N；机身使用合力大小。

因此这是奖励公式／权重的移植，并非两种机器人物理轨迹或完整训练课程等价。

## 入口与兼容

现有训练入口新增 `--bc-finetune isaac-standup`。沿用v2的学习率、BC动作参考容许尺度、小探索噪声、20轮价值预热和KL限制，仅替换任务奖励。日志子目录为 `logs/mjlab/chuanliantui_huananhu_multimode_smp_wheel_handover_v4_bc_finetune_isaac_standup_v1`。

新检查点profile为 `chuanliantui-smp-ppo-wheel39-bc-isaac-standup-v1`，保存独立recipe。play／eval／check／resume按检查点识别新模式；旧v1／v2保留原奖励，不能将旧PPO检查点直接以新奖励recipe完整续训。新实验可由稳定BC初始化，价值网络重新适应新奖励。未替换普通PPO默认配置。

仅奖励变更，模型观测／动作交付接口不变，H7和XML未修改。本轮没有新训练模型或部署交付，效果须经授权训练后按原40组及完整视频验收。

## 检查范围与证据

正式脚本 `scripts/agent/check_isaac_standup_rewards.py`：从Isaac源文件AST提取实际纯张量奖励方法，在128组输入上逐项对照19项公式、权重与加权限幅；不导入Isaac运行时。随后20环境×300策略步、0.2秒零力矩、奖励总和dt、重复读取不重复计时、5环境局部复位隔离、39／82输入及零更新检查点往返检查。

[数值及GPU检查](../../artifacts/isaac-standup-rewards-20261026-v5/result.json)、[真实入口检查](../../artifacts/isaac-standup-rewards-20261026-v5/entries/check.json)。入口检查在learn前拦截，play的viewer边界用4步调用替代；不表示可视预检或行为验收。`untrained-interface.pt`仅检查产物，不能作为训练效果。

保留早期失败日志：v1检查脚本在同一配置上重复调用配置器，v2／v3比较了两次工厂创建的lambda对象地址；修正检查脚本为同一原始配置的独立副本。v4有限检查通过后，将轮接触判定由绝对z力明确改为向上支撑力，v5为当前正式结果。未用放宽奖励公式或行为门槛修复这些检查。
