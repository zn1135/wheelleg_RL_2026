# mjlab 基础命令（multimode 示例）

以下命令均从仓库根目录执行。模型采纳状态以[项目实验记录](../ai/project-log.md)为准，示例模型不代表当前推荐。

## multimode 基础操作

使用 `mjlab_training/.venv`（Python 3.12），入口为 `scripts/mjlab.sh`。actor输入39维、critic输入82维、动作6维；物理／控制500 Hz、策略100 Hz。actor负责根据观测产生动作，critic仅在训练中估计状态的收益。与下文旧串联腿25／125维模型不兼容，mjlab参数使用连字符，例如 `--num-envs`，不要套用Isaac的 `--num_envs`。

### 选择模型与检查环境

以下用项目基线演示；测试其他模型时修改 `ct_checkpoint`，新终端需重新设置。运行使用当前共享XML，不等于重现该模型训练时的物理配置；对应来源见[共享XML记录](../../docs/diagnostics/20261021-shared-huananhu-xml.md)。

```bash
ct_checkpoint="artifacts/multimode-v1-replay-20261014/models/model_3023.pt"
test -f "$ct_checkpoint"
scripts/mjlab.sh check --checkpoint "$ct_checkpoint" --num-envs 2 --steps 16
```

成功判据：退出码0，输出actor形状 `[2,39]`、critic形状 `[2,82]`、动作维数6，无非有限值。此项只检查接口和有限步数值。

已有环境无需重新安装；首次配置见[mjlab说明](../../docs/mjlab-chuanliantui.md#可执行命令)。

### 训练与续训

以下轮数仅为命令示例，实际使用本次已确定的训练预算。两条训练命令择一执行，脚本默认不创建可视窗口，无需 `--headless`。若修改初始位置或复位分布，先执行 `scripts/mjlab.sh play --num-envs 20` 观察初态。

```bash
# 从零训练。
scripts/mjlab.sh train --num-envs 4096 --iterations 3000 --run-name multimode-new
# 从选定checkpoint续训；--iterations是本次新增轮数。
scripts/mjlab.sh train --num-envs 2048 --iterations 300 --run-name multimode-resume \
  --resume "$ct_checkpoint"
```

另一电脑终端在仓库根目录监控：

```bash
mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_huananhu_multimode_v1 --port 8080
```

成功判据：终端打印新的 `Training output` 目录，目录内生成 `params/`、TensorBoard事件及checkpoint；监控页面出现训练曲线。旧模型与日志保留。

### 可视回放

```bash
scripts/mjlab.sh play --checkpoint "$ct_checkpoint" --num-envs 1
# 手动每5秒击倒已站稳的机器人，观察恢复。
scripts/mjlab.sh play --checkpoint "$ct_checkpoint" --num-envs 1 --knockdown-every 5
# 无桌面时使用网页查看器，8091避开TensorBoard的8080。
scripts/mjlab.sh play --checkpoint "$ct_checkpoint" --num-envs 4 \
  --viewer viser --viser-port 8091
```

成功判据：viewer显示策略动作；原生窗口左上角CMD实时显示当前选中环境的前进速度(m/s)、偏航角速度(rad/s)和目标高度(m)；击倒回放能观察到恢复过程。play关闭随机化及观测噪声/延迟，保留实测校准的一拍2 ms动作等待，不等于训练配置下的验收。

### 地面起立评估与跟踪测量

```bash
# 每次创建新结果目录，避免覆盖历史产物。
ct_eval_dir="$(mktemp -d artifacts/multimode-eval-XXXXXX)"
scripts/mjlab.sh eval --checkpoint "$ct_checkpoint" --num-envs 64 --seeds 42 \
  --start-mode force --terrain flat --steps 1000 --output "$ct_eval_dir/standup.json"
mjlab_training/.venv/bin/python -m scripts.agent.multimode.assess_multimode_tracking \
  --checkpoint "$ct_checkpoint" --output "$ct_eval_dir/tracking.json"
```

成功判据：生成JSON报告，分别查看起立成功／存活计数与速度／偏航／高度跟踪误差；命令执行完毕不代表行为全部通过。最终双轮平衡另按[平衡验收](../../docs/interfaces/chuanliantui-wheel-balance-acceptance-v2.md)执行，跟踪按[跟踪测量标准](../../docs/interfaces/chuanliantui-huananhu-multimode-tracking-v1.md)判读，两者互不替代。完整击倒验收命令见[mjlab说明](../../docs/mjlab-chuanliantui.md#可执行命令)。

### ONNX导出与独立CPU回放

ONNX是保存策略网络的文件格式；当前模型固定输入39维、输出6维。使用通用导出入口 `scripts/mjlab.sh export`；旧 `scripts/export_huananhu_multimode_onnx.py` 只适用于model_3023。

```bash
ct_export_dir="$(mktemp -d artifacts/multimode-export-XXXXXX)"
scripts/mjlab.sh export --checkpoint "$ct_checkpoint" \
  --output "$ct_export_dir/policy.onnx" --opset 13
# 相同checkpoint配对其刚导出的ONNX，固定零命令测20秒。
mjlab_training/.venv/bin/python sim2sim/mj_sim2sim_huananhu_multimode.py \
  --checkpoint "$ct_checkpoint" --onnx "$ct_export_dir/policy.onnx" \
  --cmd-vx 0 --cmd-yaw 0 --cmd-height 0.27 --seconds 20 \
  --report "$ct_export_dir/cpu-normal.json"
```

成功判据：生成ONNX、`*_reference_vectors.npz`、`*_export.json`，导出数值检查通过；CPU报告查看 `passed` 和各项误差。图外的观测构造、PD与电机映射仍需单独核对。

看地面起立时追加 `--initial-pose ground` 并使用新的报告名；可视回放追加 `--render`。键盘遥操作使用 `--render --teleop --seconds 0`，同时去掉 `--report`，因为变化命令不能用于固定命令测量。CPU回放默认读取当前共享XML，历史模型验收需配套当时资产；详细定义见[ONNX使用说明](../../docs/onnx-huananhu-multimode-model5849.md)。

