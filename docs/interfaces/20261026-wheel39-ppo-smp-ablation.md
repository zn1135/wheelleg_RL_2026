# 轮角39 PPO：同条件SMP对照

2026-10-26，用户采纳“两组各300轮，完成后原40组复测及视频”的方案。Codex执行，独立成员复查和H7负责人核对待确认。沿用[轮角39 SMP PPO v4](20261025-wheel39-smp-ppo-v4.md)，当前detached HEAD `2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`加未提交改动；[运行源码快照](../../artifacts/ppo-wheel39-ablation300-20261026-v1/source-manifest.json)才是本次执行依据。无Git提交、推送或部署修改。

两组从同一累计20000次BC actor初始化，critic和PPO优化器新建；seed42、2048环境、每轮24步、每轮5个epoch×4个minibatch，初始学习率0.0001、BC动作标准差0.05。300轮对应7200个策略控制步、6000次Adam更新；每组单独使用300轮预算，不相互追加。

## 唯一实验变量

新增`train --smp-weight 0|1`，未指定时新运行默认1，续训自动读检查点中的实际权重。0为仅任务奖励，1为原任务奖励加SMP。其他控制、随机化、初态、任务奖励权重和观测均保持。

`RewardManager`遇到配置权重0会跳过该项。为保证两组调用顺序一致，两组管理器权重均保留1：先执行同样的`sim.forward`、SMP窗口和冻结先验评分，再在`SmpReward.__call__`返回处乘0或1。评分使用独立RNG，不更新先验或统计。对照组仍有计算开销，但奖励中没有SMP贡献；日志`SMP/reward`及`Episode_Reward/smp_motion`均应为0。

`infos.smp.weight`保存实际贡献权重0或1，初始化、保存和恢复继续沿用v4观测／物理合同；跨奖励权重续训拒绝。`play`、`eval`、`check`自动读取该值。两个实验目录分别为：

- `logs/mjlab/chuanliantui_huananhu_multimode_smp_wheel_handover_v4_taskonly`
- `logs/mjlab/chuanliantui_huananhu_multimode_smp_wheel_handover_v4`

旧权重1的SMP模型默认兼容，普通PPO入口保持。新轮角39／82／6、轮角14／15首次站稳清零、2ms物理／PD、10ms策略、腿轮独立2／4ms延迟、0.2秒零出力、正常趴地与全程零速保持；XML／标定／PD／弹簧补偿和H7均未修改。没有倒置恢复或移动训练。

## 检查与训练

`scripts/agent/check_ppo_smp_ablation.py`在真实20环境逐项比较两组初始actor、critic、姿态、观测、腿轮延迟和CPU／CUDA随机状态；实际采样24步确认任务项＋SMP贡献等于返回奖励，权重0的SMP贡献严格为0，并核对跨权重恢复拒绝。首次配置比较因默认地形lambda对象地址不同而失败；改为比较代码／闭包后完整重跑通过，未改环境配置，失败日志保留。

`scripts/agent/train_ppo_smp_comparison.py`调用真实训练入口，仅在learn前保存实际初始化检查点及物理／观测／RNG证据。没有教师动作插入或BC保留损失。每组结束核对迭代299及7200个策略步；`assess_ppo_smp_comparison.py`再核对两组实际2048环境初始状态、预算、6000次Adam、连续TensorBoard标量和模型有限值。

复现训练入口如下，**本次由已授权的专用检查脚本调用，不能重复执行以下命令当作同一预算**。再次训练需新预算与新run名。

```bash
scripts/mjlab.sh train \
  --smp-prior logs/mjlab/chuanliantui_smp_prior/20261025_student170_remote10000-v2/prior_10000.pt \
  --bc-init logs/mjlab/chuanliantui_bc/20261025_aligned_reference_local5000-v1/bc_5000.pt \
  --smp-weight 0 --seed 42 --num-envs 2048 \
  --iterations "${CT_PPO_STEPS:?先设置本次授权的单组PPO轮数}" --run-name wheel39-taskonly

# 对应TensorBoard；本次未启动新服务
mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_huananhu_multimode_smp_wheel_handover_v4_taskonly --port 8082
```

SMP组将`--smp-weight`设为1并另取run名；对应TensorBoard：

```bash
mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_huananhu_multimode_smp_wheel_handover_v4 --port 8081
```

## 验收与边界

`eval_wheel39_ppo_init.py`新增`--ppo-checkpoint`，用实际训练PPO actor独立回放，`--checkpoint`只提供其BC来源及原姿态／观测定义。保持原10留出初态×四种2／4ms延迟、20秒、原停车和平滑门槛；不把SMP分数当成功率。`assess_bc_closed_loop.py`独立复算PPO动作、轮角、速度和位移；视频明确标注PPO组别和轮数。

本次只做一个种子对，结果不能证明普遍优越性；原40组已参与此前模型选择，不是新增泛化样本。完整训练随机化下的行为、移动、10ms压力、独立CPU／ONNX／H7和真机不在本轮验收范围，H7负责人待确认。最终模型、预算、实际行为、失败及视频统一见[本轮报告](../../artifacts/ppo-wheel39-ablation300-20261026-v1/README.md)和[项目实验记录](../ai/project-log.md)。
