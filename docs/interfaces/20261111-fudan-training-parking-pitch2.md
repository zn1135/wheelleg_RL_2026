# 固定停车2°奖励实现：已撤回

2026-11-11用户改为roll/pitch分项、仅前倾0～+5°弱惩罚，原固定2°目标模块、配置字段及相应PT字段实现已撤回。当前以[分项奖励约定](20261111-fudan-roll-pitch-soft-band.md)为准，不能再使用`rewards.parking_pitch_deg`。

原方案没有训练新权重；原14000模型不变。既有8000图外补偿属于历史推理方案，未随本次撤回。此前14000图外2°与0°对照[结果保留](../../artifacts/fudan_ppo_walk/20261111-model14000-parking-pitch2-gpu20-01/README.md)。撤回前完整说明与源码已封存于[before快照](../../artifacts/fudan_ppo_walk/20261111-roll-pitch-reward-band-01/before/docs/interfaces/20261111-fudan-training-parking-pitch2.md)。
