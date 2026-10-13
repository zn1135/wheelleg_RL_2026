# mjlab r5闭链修复后正式训练启动

2026-10-09，用户明确要求启动训练；沿用此前4096环境、1000轮的要求。分支26_wheelleg，基础提交58947e8fabc4683b2025d42bda742b06c84dc566加既有未提交r2～r5修改及闭链修复。本次没有Git提交。

## 运行配置

- mjlab GPU训练，cuda:0，seed42，无窗口，4096环境、1000次PPO更新；从随机权重开始，没有resume或使用外部工程策略。
- r5接口，35维actor／78维critic／6维动作，PD60/2，2 ms物理×10子步，50 Hz策略；六组connect的solref为0.005 1。
- 普通／地面混合随机初态，正常／恢复模式槽，目标高度0.22 m和零速度命令；现有26项华南虎V14奖励保持当前定义。
- 启动前两种seed、32环境×128策略步的有限检查已通过原5 mm门槛，GPU最大全子步销轴偏差2.467 mm；见 [闭链修复](20261009-mjlab-closure-fix.md)。

已执行的命令（运行中，不要重复启动）：

```bash
scripts/mjlab.sh train --num-envs 4096 --iterations 1000 --run-name huananhu_r5_closurefix4096_1000
```

CLI实际learn(1000)；runner.json模板max_iterations=20000不代表本次训练20000轮。实际轮数以source-state.json命令和终端iteration .../1000为准。

输出目录：`logs/mjlab/chuanliantui/20261009_213808_247780_huananhu_r5_closurefix4096_1000`。PID58638，监督进程PID58637；终端输出`/tmp/chuanliantui-mjlab-r5-4096-1000-20261009_213800_761161.log`，实际进程状态与最终退出码由`/tmp/chuanliantui-mjlab-r5-4096-1000-20261009_213800_761161.json`记录。该JSON记录会在进程退出时由监督进程更新；本页仅记录启动时点。

已核对interface.json／runner.json／source-state.json／source.diff及21个源码和资产快照与当前代码SHA一致，TensorBoard事件及model_0.pt已生成。首次checkpoint SHA256为`106f285b094e125acd3c88a6ef8cc341e6593b129ceaff5fe176beeb96f4bdc0`。记录时已输出iteration0～13，没有Traceback／OOM，PPO持续执行。

TensorBoard命令（仅交付，未启动服务）：

```bash
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui/20261009_213808_247780_huananhu_r5_closurefix4096_1000 --port 8080
```

监督启动器、版本和首次模型证据见 [启动JSON](20261009-mjlab-r5-formal-training.json)。训练只创建新的run目录；历史r4模型及日志保留。

本记录只证明训练已启动并执行更新，不代表1000轮已完成、站高达标、策略收敛或H7／真机验收通过。新的训练结果需另行回放及行为评估。

## 本轮训练完成

2026-10-10 00:04:25（北京时间）监督进程记录训练正常结束，退出码0。已完成1000次PPO更新（编号0～999），累计98,304,000条transition，训练计时2:26:03。最终平均reward30.85、回合长度999步。model_999.pt已保存，SHA256为`3f7cacf828d1acb0b9037fbba93774218857ba547a9cb67dc5029ba9af094d16`；model_0.pt和model_500.pt保留。PID58638及监督进程均已退出。日志未发现Traceback、CUDA OOM或NaN／Inf报告。

最后恢复完成日志指标0.0145是该采样窗口指标，不能作为完整恢复成功率。此处尚未执行最终策略回放／行为评估，不据reward判定起立或目标站高达标。
