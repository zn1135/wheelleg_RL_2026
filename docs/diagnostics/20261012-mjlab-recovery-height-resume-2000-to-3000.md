# 恢复与高度课程：2000→3000轮续训启动

**完成状态：2026-10-12北京时间00:33:23累计3000轮正常结束，退出码0。课程停在双轮平衡，尚未启用高度奖励；独立回放普通平衡64/64、地面56/64，0.30 m高度跟踪均0/64。详见[结果记录](20261012-mjlab-recovery-height-model2999-evaluation.md)。以下保留启动时记录。**

用户明确授权下一步训练，并确认4096环境追加1000轮、累计3000轮。北京时间2026-10-12 00:13:17启动，实际进入PPO学习；00:14:33核验时日志到累计迭代索引2048。本记录仅证明启动和模型迁移成功，训练效果待完成后独立回放。

## 参数与起点

- 配方`huananhu-v14-recovery`，整批扶正→双轮平衡→高度跟踪，课程从扶正重新开始。
- 源为已评估2000轮高度门控组`model_1999.pt`，SHA256 `1c3487a7da8d1fe74c418466ec4068f621c162e36406c14786c972a2839e7027`。
- 4096环境、seed42，追加1000次PPO更新；完成后目标`model_2999.pt`，累计3000轮。不是此前退化的stand3000轮组。
- actor35/critic78，四腿两轮；完整恢复actor、critic、探索分布与Adam，初始学习率1e-5，继续V14自适应调节。
- 普通/地面初态各50%，三阶段零速度命令，高度命令0.27～0.34 m。
- 无辅助力、无被动膝限位奖励、无默认角惩罚；物理气弹簧与实体闭链控制保留。
- 恢复与最终高度跟踪使用独立判据，详见[配置说明](../interfaces/chuanliantui-mjlab-v14-recovery-height-v1.md)。

## 启动修复与冻结版本

首次启动因训练分支缺少`ROOT`导入退出，未创建训练run、未进入学习。失败job保留：`jobs/20261011T161151Z-v14-recovery-resume-2000-to-3000`，退出码1。

只补充`from ..config import ROOT`，未调整训练参数。重新冻结为：

`/root/gpufree-data/wheelleg-remote/releases/20261012-v14-recovery-height-launchfix`

模块SHA256：`d7a7808c0bdcb444cf66aeed053ca83501df88237bec519461d701734f012596`，与本地源码一致。之前4096环境有限检查仍记录原v1源码，不能用它声称已覆盖首次训练入口；本次另外验证实际训练入口、完整迁移和PPO更新，补齐这一缺口。两个冻结版本均保留。

## 实际运行与监控

训练PID：`25973`；job：

`/root/gpufree-data/wheelleg-remote/jobs/20261011T161317Z-v14-recovery-resume-2000-to-3000`

新训练run（通过release的logs链接解析后的真实路径）：

`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261012_001331_180042_v14_recovery_height_v14_recovery_resume_2000_to_3000`

已执行的训练命令记录：

```bash
cd /root/gpufree-data/wheelleg-remote/releases/20261012-v14-recovery-height-launchfix
TZ=Asia/Shanghai PYTHONPATH=mjlab_training/src MUJOCO_GL=egl \
  /root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u -m ct_mjlab.cli train \
  --training-profile huananhu-v14-recovery --num-envs 4096 --iterations 1000 \
  --seed 42 --device cuda:0 \
  --resume /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_214628_750246_scut_v14_height_v14_height_resume_1000_to_2000/model_1999.pt \
  --resume-v14-recovery-fix --run-name v14_recovery_resume_2000_to_3000
```

TensorBoard命令（未启动）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/tensorboard \
  --logdir=/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261012_001331_180042_v14_recovery_height_v14_recovery_resume_2000_to_3000 \
  --port=8080
```

`job/output.log`记录运行进度；结束时wrapper写`.exit-code`和`finished.json`。进程通过独立会话后台运行，不依赖SSH连接。未创建后续自动训练/定时任务。

## 实际迁移核验

从真实run读取`model_resume_initial.pt`：actor、critic、Adam与原2000轮checkpoint逐项完全相同，累计计数2000、课程阶段扶正、优化器学习率1e-5，模块指纹匹配修复冻结版本。训练进程存活、日志已进入2001之后的PPO更新。

本地证据：`artifacts/recovery-height-resume-startup-20261012.zip`，SHA256：

`f4e5afec2212bd6fb15b889aade4b195ba55b1712cda61ffe159c008a9722d91`

包内7个文件及ZIP均校验，包含launch、wrapper、实际迁移核验JSON、启动日志、interface/environment/runner配置。历史logs与模型未删除、移动或覆盖。
