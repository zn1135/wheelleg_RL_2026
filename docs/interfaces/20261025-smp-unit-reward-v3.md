# 串联腿SMP PPO：固定单位误差尺度v3

2026-10-25，Codex执行，独立复查未进行。用户授权固定SMP评分尺度并检查实际奖励占比，不授权本轮训练。

| 项目 | 定义 |
|---|---|
| 机器人／任务 | 共享XML串联腿，chuanliantui可选`--smp-prior` |
| 旧定义 | [正常起立与零速站稳v2](20261024-smp-stand-v2.md)，前100万个策略窗口在线估计误差尺度 |
| 新profile | `chuanliantui-smp-ppo-normal-ground-stand-unit-v3` |
| 权威实现 | `mjlab_training/src/ct_mjlab/smp/ppo.py`中的SmpReward／SmpOnPolicyRunner |
| 训练来源 | detached HEAD `2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`及未提交修改；[本轮完整源码及差异哈希](../../artifacts/smp-unit-reward-20261025/source-manifest.json) |
| 部署侧 | 本次未改H7，未交付新PPO模型；部署负责人未来策略核对待确认 |

## 奖励变化与兼容

三个噪声级评分误差尺度固定为`[1,1,1]`，分数`exp(-6*mean(raw_errors))`。加载先验后清空遗留误差计数／待累计统计并设单位尺度；训练／play均不调用record或update，不使用示范均值校准。先验参数、EMA、特征均值／标准差冻结。SMP权重仍1，RewardManager仍只乘一次dt=0.01，其他任务奖励权重保持。

保存新profile、`normalizer_mode=fixed-unit`、`normalizer_scale=[1,1,1]`、`normalizer_samples=0`及评分RNG；续训／回放加载拒绝旧v1／v2、不同先验／奖励合同和非单位统计。新实验目录`logs/mjlab/chuanliantui_huananhu_multimode_smp_stand_unit_v3`，不覆写旧日志。历史审计脚本仍支持旧v1／v2；旧模型实际回放应使用各自冻结源码，不视为v3续训。

actor39／critic82／动作6、SMP连续10×30状态窗口、前9步预热、局部复位、倒置终止保持。2ms物理／PD、100Hz策略、训练腿轮各自随机1／2物理步即2／4ms、前0.2秒主动零出力、正常趴地0.25m／零初速度、全程零速／零偏航及高度0.33648m保持。关节顺序／默认角／物理增益／XML／特征定义和部署张量接口均不变，不向H7同步评分网络。

## 已验证与剩余问题

[本轮报告与复现命令](../../artifacts/smp-unit-reward-20261025/README.md)：

- 20环境真实PPO采样／returns、保存／重载／RNG、门控、局部复位、倒置终止和旧profile／污染尺度拒绝检查通过；另20环境初始随机actor运行2秒，优化器更新0。
- 名义实时LQR→学生PD：10个留出完整初态×四种2／4ms延迟，40环境完整20秒，严格起立／最终平衡40/40。全部奖励明细逐步相加等于实际返回奖励，SMP只乘一次dt；模型／EMA／全部统计不变、计数0。
- SMP占全部正奖励：教师起立0.2～2秒18.815%，末段站稳16.517%；初始随机actor在真实训练随机化下接管后仅约0.000269%，平均每步SMP贡献2.33e-8。两组初态和随机化不同，不能作同分布的策略效果对照。

固定尺度修正了评分标准被随机策略改变的问题，但初始随机策略的SMP奖励近零，仍缺少明显的从头训练引导。本轮保留权重1，不调整评分温度／初态课程／策略初始化，不训练SMP或PPO；不将这些有限检查当作新PPO行为改善或真机通过。
