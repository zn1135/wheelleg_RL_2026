# mjlab 20 环境可视初训

日期：2026-10-09。用户明确授权开启20环境训练并观察实时窗口。分支 `26_wheelleg`，基础提交 `58947e8fabc4683b2025d42bda742b06c84dc566` 加r2气弹簧和实时查看器未提交修改。

## 执行与结果

```bash
DISPLAY=:0 scripts/mjlab.sh train --num-envs 20 --render --iterations 20000 --run-name physical_spring_preview20
```

使用独立Python 3.12 / mjlab环境、RTX4060、接口 `chuanliantui-mjlab-huananhu-v14-flat-standup-r2`，从随机初始权重开始，无resume。20个训练环境开启域随机化，实际PPO采样环境在窗口中网格显示；不是独立回放副本。

设定上限20000轮；窗口关闭后完成当前更新并正常退出，退出码0。实际完成iteration0～28，共29次PPO更新。保存 `model_0.pt` 和 `model_28.pt`，TensorBoard事件已写入。本次仅短时可视初训，未进行正式起立／静站验收，不能作为已训练可部署策略交付。

新输出目录：`logs/mjlab/chuanliantui/20261009_185346_185674_physical_spring_preview20/`。未覆盖历史run。配置在 `interface.json`、`runner.json`，基础版本、实际命令及修改文件SHA在 `source-state.json`；对应修改源码、新模型和清理网格副本在 `source-snapshot/`，已跟踪源码差异在 `source.diff`。终端输出保存在 `/tmp/chuanliantui-mjlab-train20-20261009-01.log`。

TensorBoard命令（仅交付，未启动）：

```bash
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui/20261009_185346_185674_physical_spring_preview20 --port 8080
```

## 查看器行为

查看器实现从用户指定的本机 `mjlab/wheelleg_RL` 实时训练查看器适配到当前direct环境。代码在 `mjlab_training/src/ct_mjlab/training_viewer.py`；`train --render` 启用，默认30FPS，显示实际训练环境数。

- 查看器使用独立host模型及数据，同步每环境状态与随机化模型字段，不修改共享训练模型、不额外调用仿真step。
- 空格暂停采样，V停止／恢复刷新但继续训练，[／]选择起始环境。
- 关窗或首次Ctrl+C请求在当前完整PPO更新后保存停止，避免把未完成更新当成恢复点。

本次实际验证了图形窗口启动、完整PPO更新、TensorBoard写入，以及关窗后保存并正常退出。没有启动TensorBoard、ONNX导出、正式策略回放或固件验证。
