# r6完整移植后正式训练启动

用户授权“改为后就开始训练1000轮”，2026-10-10启动4096环境、1000次PPO更新，seed42/cuda:0，无窗口，从随机权重开始，不resume旧r5或外部策略。分支26_wheelleg，基础版本58947e8fabc4683b2025d42bda742b06c84dc566加未提交改动，没有Git提交。

参数与参考wheelleg_RL recover_pose_v2全部训练配置一致，机器人资产保留本仓；完整对照及预检见 [移植检查](20261010-mjlab-reference-training.md)。原生mjlab/PPO、35/78/6、27奖励、高度0.27～0.34m、100%落地初态、固定NORMAL、2ms×10、每50轮保存。原20环境10s窗口预检已正常退出。

已执行（运行中，不要重复启动）：

```bash
scripts/mjlab.sh train --num-envs 4096 --iterations 1000 --run-name wheelleg_recover_r6_4096_1000
```

实际输出：`logs/mjlab/chuanliantui/20261010_003055_713431_wheelleg_recover_r6_4096_1000`。PID79968、监督PID79967，终端日志`/tmp/chuanliantui-mjlab-r6-4096-1000-20261010_003044_720909.log`，退出状态由`/tmp/chuanliantui-mjlab-r6-4096-1000-20261010_003044_720909.json`在结束时更新。运行在独立会话，结束当前聊天不会停止训练。

记录时已观察iteration0～35，model_0.pt和TensorBoard事件已保存，无Traceback/OOM。首次checkpoint SHA256：`17968a3fb16827c25fcaa3b7ee9f90bb72a1184d38a82aa9fff792da1ce5ec89`。已核对interface.json、runner.json（max_iterations=1000/save_interval=50）、environment.json（4096）、source-state.json、source.diff及43份源码/资产指纹与当前工作区一致；源码快照保存在此次新run中。

TensorBoard（仅交付，未启动）：

```bash
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui/20261010_003055_713431_wheelleg_recover_r6_4096_1000 --port 8080
```

[启动JSON](20261010-mjlab-r6-formal-training.json)只证明已启动并执行更新，不表示1000轮已完成、起立成功或策略可部署。完成后需要同接口回放／CPU行为验收及H7适配，历史logs与r5模型保留。

## 训练完成

2026-10-10 01:52:23（北京时间）监督进程记录正常结束，退出码0，训练与监督进程已退出。已完成1000次PPO更新（0～999），98,304,000条transition；训练计时1:21:16。model_999.pt及每50轮的检查点均已保存。最终平均reward−23.43、平均回合1000步、动作std0.19。日志无Traceback、CUDA OOM或RuntimeError。

最终checkpoint SHA256：`250cd269c3bf2b5d7243c2c1c1c8ca4355c9b837aa739aa0d27e0b25baec1c89`。

末轮已结束回合的日志指标：平均绝对高度误差0.0897 m、strict_recovered0.0047、strict_stand_fraction0.0017、numerical_failure0、宽松recovered1.0。严格指标较低，宽松指标不能代表目标高度起立成功；这些是最后一批回合的训练日志，不是全部训练回合的成功率，也不是独立回放验收。训练完成记录时尚未执行r6最终策略行为评估或H7验证。随后20秒CPU行为评估未通过，32环境GPU诊断定位到腿连杆撑地低位姿态，见 [原因分析](20261010-mjlab-r6-low-stance-analysis.md)；H7仍未验证。
