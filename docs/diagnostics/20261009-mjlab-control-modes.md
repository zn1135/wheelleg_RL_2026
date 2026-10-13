# 正常／恢复模式槽检查与20环境可视训练

2026-10-09，`26_wheelleg`，基础提交 `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交r4代码。用户要求利用七个模式槽，定义见 [r4接口](../interfaces/chuanliantui-mjlab-huananhu-v14-flat-standup-r4.md)。独立复查未执行。

## 实现

恢复模式从地面初态开始，连续0.5 s严格站稳后转正常；正常持续失稳0.1 s重新进入恢复。使用原槽28／31做互斥normal／recover，33为恢复高度目标×5，34为本次恢复秒数；actor与critic共用状态、CPU与GPU共用实现。模式每策略步推进一次，读观测与刷新窗口无状态副作用。

参考本机mjlab工程的地面恢复状态和严格站稳判据；原始SCUT flat V14的recover仅为占位，本次扩展明确记录，没有将楼梯／斜坡／跳跃加入平地任务。原26项奖励、普通连续失败终止窗口和启动力矩门控保持当前定义。窗口HUD显示模式、恢复目标／计时和七槽实时数值。

## 有限检查

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_mjlab.py --gpu --out /tmp/chuanliantui-control-mode-check-20261009.json
```

退出码0，结果见 [检查JSON](20261009-mjlab-control-modes-check.json)。

- 25拍站稳阈值、站稳中断重计、双轮接触判据、5拍失稳重新进入、倒置姿态、部分reset和观测读取无副作用检查通过。
- actor28:35与critic32:39完全一致，初始模式匹配ground_start，GPU64步中模式与内部状态一致。
- 26项奖励与源实现最大绝对差9.313226e−10。
- CPU六处闭合最大误差1.610356 mm；GPU4环境64步最大4.559326 mm，状态／奖励有限，无数值异常重置掩盖；实体弹簧推力界检查通过。
- 未训练随机actor ONNX数值误差1.043081e−7，PPO runner构造通过。这些不代表策略行为通过。

## 旧训练停止与兼容性

r3可视训练已在当前PPO更新完成后保存退出，最终iteration445，退出码0。旧run为 `logs/mjlab/chuanliantui/20261009_190253_790595_huananhu_random_pose20`，最终 `model_445.pt` SHA256为 `d589e5441fd619d4d024042f865b19bcf5771b3bc2e86864ef8602d4695181af`。实际读取checkpoint确认接口r3、iter445；r4的加载校验拒绝该接口。

新r4从随机权重开始，不直接恢复旧模型；所有历史日志和checkpoint保留。模式观测含义改变但35／78／6维保持；manifest新增模式阈值和实现SHA。H7模式观测与传感判据尚待适配，未构建固件或做真机验证。

## r4可视预检与实际训练

20环境随机初态可视预检运行10秒，退出码0；随后开启新接口20环境实际PPO训练，从随机权重开始，域随机化启用。窗口HUD显示当前选中环境的模式、目标高度、恢复计时和七槽值。

```bash
DISPLAY=:0 scripts/mjlab.sh play --num-envs 20 --random-reset --render --seconds 10
DISPLAY=:0 scripts/mjlab.sh train --num-envs 20 --render --iterations 20000 --run-name huananhu_modes20
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui/20261009_191746_257934_huananhu_modes20 --port 8080
```

新输出：`logs/mjlab/chuanliantui/20261009_191746_257934_huananhu_modes20`。核对接口为r4，已保存model_0.pt和TensorBoard事件，源码快照包含模式状态机／观测／GPU及CPU回放实现，SHA与此次代码一致。整理记录时已完成iteration0～20，仍在运行；没有完成20000轮或正式起立／静站验收。TensorBoard命令仅交付，未启动服务。终端记录 `/tmp/chuanliantui-mjlab-modes-train20-20261009.log`。

空格暂停、V开关刷新、[／]选环境，关闭窗口会在当前PPO更新完成后保存退出。未提交本轮代码。

## 后续正式训练切换

用户要求开始正式训练后，本20环境预览收到首次SIGINT，完成当前更新并保存 `model_105.pt` 后正常退出，退出码0。实际读取确认接口r4、iter105及optimizer_state_dict，SHA256为 `8dfa8b146c55f885c4567dbc744a0044028d4729bcbe3113c137df8b17669fdc`。新正式训练从随机权重开始，最新4096环境／1000轮配置及运行状态见 [正式训练启动记录](20261009-mjlab-formal-training.md)。
