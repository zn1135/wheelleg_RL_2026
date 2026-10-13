# r15 辅助力课程1000轮回放评估

## 训练结束

- RTX4090，4096环境，seed42，随机初始化，完成1000次PPO更新，退出码0，训练日志耗时20分24秒。
- 最终模型 `model_999.pt`，目录 `/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_134150_495971_getup_r15_assisted`。
- 整批课程仍在扶正 `righting` 阶段，目标0.27 m；阶段历史为空，没有进入站起／行走。
- 末轮平均reward84.68，平均回合1000步，训练平均高度误差0.659 cm，数值异常0；`getup_ready_fraction=0`。
- 末轮平均实际辅助力1.09 N，预算系数约0.667。该系数是最大力预算，实际力仍受高度差和初态电机保持限制。

## 无辅助力回放

使用记录该模型的原始源码与接口，确定性actor；名义物理，无观测噪声、随机动作延迟或推扰；冻结课程、关闭辅助力。分别64个普通初态和64个地面初态，各20秒，零速度命令、目标0.27 m。物理失败仅记录，允许地面样本完整尝试恢复；数值失败永久排除。

| 指标 | 普通初态 | 地面初态 |
|---|---:|---:|
| 持续0.5秒严格站立 | 0/64 | 0/64 |
| 最后5秒至少95%时间严格站立 | 0/64 | 0/64 |
| 最后5秒平均高度 | 0.26745 m | 0.26094 m |
| 最后5秒平均绝对高度误差 | 0.281 cm | 0.990 cm |
| 最后5秒机身接触时间占比 | 1.56% | 10.03% |
| 数值异常 | 0 | 0 |

严格站立条件：高度误差≤3 cm、横滚／俯仰各≤15°、双轮接触力历史峰值模长各>5 N、机身接触力≤1 N，连续0.5秒。最后5秒均值包括所有有限样本，包括曾触发物理失败的样本。

结论：已学到接近目标高度的支撑姿态，但不能判为完成稳定扶正或起立，更没有训练到行走。视频仅显示每类的样本0，不能代表64个样本成功率。

## 发现的轮接触判定错误

额外使用同一模型、无辅助力、2个名义普通初态，回放5秒以核对实际接触数据。双轮竖直力约为 `[-58.88,-40.30] N` 和 `[-65.42,-40.31] N`，两轮确有承重，机身接触力0。当前mjlab传感器输出方向使承重Fz为负。

但 `getup_curriculum.py` 的 `readiness()` 使用 `Fz>5`；奖励中的 `dual` 使用 `Fz>1`。因此正常双轮承重被判为不接触：

- 阶段推进条件无法通过，与整个训练 `getup_ready_fraction=0`、阶段历史为空一致。
- `no_fly` 持续得到-2，`pose_bias` 受 `dual` 屏蔽为0。轮载荷奖励使用竖直力绝对值，未受同样的符号判断影响。
- 无辅助力严格回放的双轮条件使用模长，独立于上述符号错误。回放严格成功率0仍有姿态原因：两个核对样本横滚约17.8°、俯仰约15.6～15.8°，超出15°门槛。

建议先按传感器方向统一承重竖直力，再对齐双轮判定及课程推进；不能仅放宽成功标准或把reward上涨视为已经站稳。本轮只评估并记录问题，没有修改训练奖励或启动续训。

## 证据与回放

- [完整评估JSON](../../artifacts/getup-r15-assisted-model999-evaluation-20261011/evaluation.json)
- [轮接触核对](../../artifacts/getup-r15-assisted-model999-evaluation-20261011/contact-probe.json)
- [普通初态视频](../../artifacts/getup-r15-assisted-model999-evaluation-20261011/replay-normal.mp4)
- [地面初态视频](../../artifacts/getup-r15-assisted-model999-evaluation-20261011/replay-ground.mp4)
- [0／2／10／20秒图](../../artifacts/getup-r15-assisted-model999-evaluation-20261011/replay-poster.png)
- [渲染说明](../../artifacts/getup-r15-assisted-model999-evaluation-20261011/render-record.json)：来自记录的GPU qpos，仅CPU `mj_forward` 渲染，没有重新积分仿真。

新增可复用评估入口 `scripts/agent/eval_chuanliantui_getup_curriculum.py`，复用原四组实验回放记录器，严格核对r15源码／接口。训练源码快照未改。首次远端执行因缺少共享评估脚本失败，补齐后重跑成功；正式评估退出码0。评估JSON、PNG及两段MP4已逐项核对远端SHA256一致。

本次实际评估命令（远端，现已完成）：

```bash
export PYTHONPATH=/root/gpufree-data/wheelleg-remote/releases/20261011T054119Z-getup-r15-assisted/mjlab_training/src
export MUJOCO_GL=disable
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u /root/gpufree-data/wheelleg-remote/jobs/20261011T054119Z-getup-r15-assisted/evaluation-tools/eval_chuanliantui_getup_curriculum.py --checkpoint /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_134150_495971_getup_r15_assisted/model_999.pt --output /root/gpufree-data/wheelleg-remote/evaluations/20261011T060114Z-getup-r15-model999 --console /root/gpufree-data/wheelleg-remote/jobs/20261011T054119Z-getup-r15-assisted/output.log --num-envs 64 --seconds 20 --seed 42
```

输出目录要求不存在，复用命令时指定新的评估目录。评估工具和临时接触核对脚本位于作业目录，未写入训练logs。结束后GPU空闲（1 MiB、0%）。
