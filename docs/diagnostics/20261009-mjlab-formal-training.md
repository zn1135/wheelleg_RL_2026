# 串联腿 mjlab 正式训练启动记录

2026-10-09，用户明确要求开始正式训练，并将并行环境改为4096、轮数改为1000。分支 `26_wheelleg`，基础提交 `58947e8fabc4683b2025d42bda742b06c84dc566` 加现有未提交r4修改；本轮未提交代码，未做独立复查。

## 最终运行配置

- 入口：独立Python3.12环境 `scripts/mjlab.sh`，mjlab GPU批量物理与华南虎V14普通PPO。
- 4096并行环境，1000次PPO更新，从随机权重开始，无resume，无窗口；seed42，cuda:0。
- 接口 `chuanliantui-mjlab-huananhu-v14-flat-standup-r4`，actor35／critic78／action6，四实体腿电机位置PD、两轮速度控制，实体闭链与气弹簧。
- 普通／地面混合随机初态、正常／恢复模式输入、静止命令／0.22 m高度、启动0.2 s零电机力矩和现有26项奖励保持当前定义。
- 每轮24策略步，每轮4096×24=98304条transition。每500轮保存，正常完成后runner另存最终checkpoint。
- CLI传入 `--iterations 1000` 并实际调用learn(1000)；runner.json保留模板max_iterations=20000，实际轮数以source-state.json命令和终端 `iteration .../1000` 为准。

已执行的训练命令（不要重复运行启动第二个进程）：

```bash
scripts/mjlab.sh train --num-envs 4096 --iterations 1000 --run-name huananhu_r4_formal4096_1000
```

监控命令（仅交付，TensorBoard服务未启动）：

```bash
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui/20261009_192207_944967_huananhu_r4_formal4096_1000 --port 8080
```

## 输出与启动证据

新输出：`logs/mjlab/chuanliantui/20261009_192207_944967_huananhu_r4_formal4096_1000`。进程PID32961，独立后台session运行，输出重定向到 `/tmp/chuanliantui-mjlab-formal4096-1000-20261009_192200_757954.log`；launcher信息见同名JSON。日志、模型与配置使用新run，不覆盖历史资产。

核对interface／runner／source-state、差异及修改源码／新资产快照齐全，源SHA与当前工作区一致；实际命令为4096环境和1000轮。已保存 `model_0.pt`（含actor、critic与PPO优化器，iter0），SHA256为 `deab612877519bfc968c0363b683725920e346572ac0c1fd0ad57dcef1299d01`，已生成TensorBoard事件。记录时已完成iteration0～16，日志无Traceback或CUDA OOM，训练持续运行；最近GPU查询总显存2080 MiB、利用率91%。初始预计完成时间约1小时，随实际每轮耗时变化。

这仅证明正式训练成功启动并执行PPO更新，不表示1000轮已完成、收敛、起立／30秒静站行为验收通过或H7部署可用。

## 预览与配置调整历史

- 20环境r4可视训练先保存 `model_105.pt` 并正常退出，退出码0；详见 [模式运行记录](20261009-mjlab-control-modes.md)。
- 初始1024环境／20000轮run实际启动，收到用户改为4096的要求后用SIGINT中断，终端最后完整日志为iteration19，出现预期KeyboardInterrupt。其已保存产物保留，不将其称为正常完成。
- 随后4096环境／20000轮在启动期间收到用户改为1000轮的要求，被SIGINT中断；新进程按最终4096／1000配置从随机权重开始。早期run目录和终端日志均保留。
- 当前只保留最终配置的训练进程，未启动TensorBoard，未改机器人资产、奖励／控制实现或H7固件。

## 训练完成与首次效果评估

训练日志完整结束于iteration999/1000，累计98,304,000条transition，用时1:20:30，最终checkpoint `model_999.pt`（SHA256 `5c3c5cb91fb92c3a2620ef61e07717bd9e9b76dd32d78762bf6cf8672a0d50bb`）与 `model_500.pt` 均已保存，原PID32961不再运行，日志无Traceback。启动器未保存最终进程退出码，不能从进程消失单独推断退出码0。效果检查见 [model999评估记录](20261009-mjlab-model999-evaluation.md)。本轮达成1000次PPO更新，尚未达到起立高度／静站行为验收。
