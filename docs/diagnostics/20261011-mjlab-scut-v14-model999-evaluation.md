# 华南虎V14从头1000轮训练及回放效果

2026-10-11，评估本次用户指定的原始华南虎V14配方，不是半辅助力r15续训组。

## 训练完成

- run：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_205306_764251_scut_v14_flat_scut_v14_flat_scratch_1000`。
- 4096环境，从随机权重完成1000次更新，退出码0。末次TensorBoard记录为2026-10-11 21:15:44（Asia/Shanghai），训练日志累计学习耗时22分19秒。
- 模型 `model_999.pt`，SHA256 `0201396ba38378abba8ba986aa8e32405d7ebdb874eecd119911ee354939fda7`；检查点记录completed_updates=1000。
- 末轮mean reward=12.6530，mean episode length=961.90步，高度误差指标0.109893 m，strict_recovered=16.25%，strict_stand_fraction=6.53%，velocity_error=0.82896 m/s，numerical_failure=0，学习率约1e−5。

这些是随机速度／高度命令及随机化训练下的日志指标，不能用训练16.25%严格恢复替代下面固定目标的独立回放成功率。

## 独立回放

普通初态64份、地面初态64份，seed42；各运行20 s，确定性actor、名义物理、无噪声／延迟、无辅助力；固定vx=0、yaw=0、height=0.30 m。保留本次V14固定normal模式和100 rad/s轮速限幅。使用训练冻结发布中的V14工厂，完整检查点接口校验通过，没有用r14/r15奖励、模式或控制配方代替。

物理触地／倾角失败仅记录，不截断20秒恢复机会；数值失败永久排除。严格站立要求高度误差≤3 cm、roll/pitch各≤15°、双轮接触>5 N、基座接触≤1 N连续0.5 s；末5秒持续站立要求该完整时段至少95%的时间满足严格条件。

| 指标 | 普通初态 | 地面初态 |
|---|---:|---:|
| 严格站立成功 | 0/64 | 0/64 |
| 末5秒持续严格站立 | 0/64 | 0/64 |
| 末5秒平均机体高度 | 0.18999 m | 0.18617 m |
| 末5秒平均绝对高度误差 | 11.00 cm | 11.38 cm |
| 末5秒平均水平速度 | 0.05318 m/s | 0.04174 m/s |
| 全程未触发共同物理失败 | 63/64 | 56/64 |
| 末5秒基座触地比例 | 0% | 12.5% |
| 末5秒倒置比例 | 0% | 0% |
| 数值失败 | 0 | 0 |

本次结果是低姿态保持，没有达到0.30 m起立目标。普通初态最终降到约0.19 m，地面初态也没有一份完成严格站立；“未触发物理失败”不等于成功站起。该结论限定于本次固定零速度、0.30 m回放，未评估非零速度行走或完整高度命令范围。

## 视频与证据

- [普通初态视频](../../artifacts/scut-v14-eval-1000-20261011/results/comparison-normal.mp4)
- [地面初态视频](../../artifacts/scut-v14-eval-1000-20261011/results/comparison-ground.mp4)
- [回放JSON](../../artifacts/scut-v14-eval-1000-20261011/results/evaluation.json)
- [训练末轮JSON](../../artifacts/scut-v14-eval-1000-20261011/training-final.json)
- [最终帧](../../artifacts/scut-v14-eval-1000-20261011/results/comparison-final.png)

视频是第0份环境的代表轨迹，不是64环境成功率的可视替代。普通与地面各自的初始qpos/qvel、命令、地形原点与此前半辅助力1000轮评估逐值相同，见 [配对记录](../../artifacts/scut-v14-eval-1000-20261011/paired-initial-state-comparison.json)。两组训练配方不同，不能把差异单独归因于辅助力。

远端job：`/root/gpufree-data/wheelleg-remote/jobs/20261011T132439Z-scut-v14-eval-1000`。评估及渲染均退出码0；无学习。证据ZIP SHA256 `109928ae613957c36b24bbd37ece205c99135a4044b22fcd5e183f38ab75acc9`，下载后核对ZIP和全部载荷指纹通过。训练日志、检查点、冻结训练发布未改写。

## 评估入口改动

`scripts/agent/eval_chuanliantui_getup_curriculum.py` 增加原始V14接口分派，固定0.30 m比较目标并复用既有记录器；原有r15阶段目标和分派保持。公共rollout与渲染脚本没有修改。评估脚本SHA256 `cb438082e0887cb26e165623077cb71f41065aa996de2cb157ae3de998f97f64`，公共rollout SHA256 `d3edcb5dd2475797d98e9fec8db2370441655f51a41c8319a85b523270c80a92`。

在有本仓独立mjlab运行时的Linux仓库中，评估命令为：

```bash
PYTHONPATH=mjlab_training/src:scripts/agent mjlab_training/.venv/bin/python \
  scripts/agent/eval_chuanliantui_getup_curriculum.py \
  --checkpoint logs/mjlab/chuanliantui/<V14_run>/model_999.pt \
  --output <新的输出目录> --num-envs 64 --seconds 20 --seed 42 --device cuda:0
MUJOCO_GL=egl mjlab_training/.venv/bin/python scripts/agent/render_chuanliantui_ablation_replay.py \
  --input <上述输出目录> --groups huananhu_v14 --width 640
```

本次没有修改训练参数，没有追加训练，没有启动TensorBoard；Windows迁移目录仍无Git历史，未提交。
