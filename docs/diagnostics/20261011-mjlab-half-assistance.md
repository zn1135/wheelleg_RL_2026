# 新训练辅助力减半

用户选择：新训练上限从整机重量100%降为50%；5000→6000轮续训保持零辅助力。

新建 r15 / orientation 训练通过 CLI 自动选择独立的 half-assistance 配方。
只改 `cfg.actions.pd.max_weight_ratio` 为0.5，原动作、气弹簧、奖励、课程和随机化内核不变。

```text
Fz = 整机重量 × 0.5 × clamp((目标高度−当前高度)/目标高度, 0, 1)
     × clamp(1−累计PPO更新数/3000, 0, 1)
```

仍只在扶正、站起阶段启用；地面电机保持期、行走和回放不施加辅助力。
同一姿态及累计更新数下，新训练辅助力为原先的一半。

新模型保存独立接口、实际动作配置及源码哈希；只能恢复相同half-assistance配置，保留actor/critic/Adam、课程和累计衰减进度。
历史r15 v3和orientation完整模型续训由原入口处理，不重置辅助力计时。
本次5000→6000轮使用已验证的orientation冻结release，三个阶段orientation权重−30；进度已超过3000，因此辅助力全程为零。

实现：`mjlab_training/src/ct_mjlab/experiments/getup_half_assistance.py` 和 CLI 路由。
## 验证与训练

RTX4090、原Python3.12/mjlab环境，普通r15和orientation两种配方各20环境、三个阶段各8步，观测35/78、奖励有限、数值异常0。
工厂配置比较确认只改上限比例。辅助力在累计0/1000/2000/3000/5000/6000更新时均为旧值的一半；行走、回放关闭条件保留。
两种配方的模型保存／加载往返确认actor、critic、Adam、课程与累计5000更新恢复一致，5000更新处实际辅助力为零。
上述有限检查未执行PPO更新。随后按用户要求启动50%配方从头1000轮训练并验证首个更新，行为效果尚未评估，见[新训练启动记录](20261011-mjlab-half-assistance-scratch-1000-training.md)。

检查结果：[result.json](../../artifacts/half-assistance-and-resume-start-20261011/half-assistance/result.json)。
最终CLI源码在两种profile下各20环境、1步检查退出0，数值异常0；最终远端release为 `20261011T104817Z-getup-r15-half-assistance-final`，两个修改文件SHA256与本机一致。证据：[最终入口检查](../../artifacts/half-assistance-and-resume-start-20261011/cli-final-validation.json)。交付ZIP中的code-hashes属于前一版检查入口；最终CLI哈希以此JSON为准，辅助力配方源码未变。
本次零辅助力续训已启动并验证首个更新：[训练启动记录](20261011-mjlab-r15-orientation-resume-5000-to-6000-training.md)。
