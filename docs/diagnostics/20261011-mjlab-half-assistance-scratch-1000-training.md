# 50%辅助力，从头训练1000轮

2026-10-11 18:51:32（北京时间）在原RTX4090远端启动；新的随机网络与优化器，4096环境，1000次PPO更新，seed=42。

- 使用最终release `20261011T104817Z-getup-r15-half-assistance-final`。
- CLI选择 `getup-r15-orientation`，新训练实际接口为 `chuanliantui-mjlab-getup-curriculum-r15-orientation-half-assistance-v1`。
- orientation三阶段−30，原公式保留；pose_bias和knee_limit均关闭。
- 辅助力上限0.5倍整机重量，按高度缺口缩放，累计3000更新衰减到零。1000更新结束时预算上限为1/3倍整机重量，实际力可能更小或为零；不在本次1000轮末强制清零。
- 初始模型累计更新0，课程扶正阶段、stage_steps=0、历史为空；Adam状态为空，学习率0.0001，无续训标志或外部模型初始化。
- 首个新模型 `model_0.pt` 记录累计更新1，actor已经变化，actor/critic参数有限。
- 18:52:23核对时日志迭代16/1000，进程运行，无退出码。
- 原5000→6000轮零辅助力续训组仍运行，共享同一GPU，单组吞吐会受影响。

证据：[启动检查JSON](20261011-mjlab-half-assistance-scratch-1000-start.json)。本次确认实际启动和首个PPO更新；新策略行为效果尚未评估。

## 实际训练命令

以下命令已经后台执行，请勿重复启动该组。

```bash
cd /root/gpufree-data/wheelleg-remote/releases/20261011T104817Z-getup-r15-half-assistance-final
PYTHONPATH="$PWD/mjlab_training/src" MUJOCO_GL=disable \
  /root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u -m ct_mjlab.cli train \
  --training-profile getup-r15-orientation --num-envs 4096 --iterations 1000 \
  --seed 42 --device cuda:0 --run-name half_assistance_orientation_scratch_1000
```

TensorBoard命令（未启动）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/tensorboard \
  --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_185145_808090_getup_r15_half_assistance_orientation_half_assistance_orientation_scratch_1000 \
  --port 8080
```

后台任务及输出：
`/root/gpufree-data/wheelleg-remote/jobs/20261011T105132Z-half-assistance-orientation-scratch-1000/`。

输出新建于独立日志目录，目标末模型 `model_999.pt`；历史日志保留。

后续结果：1000次更新正常完成，退出0，已进行最终模型的无辅助配对回放，见[1000轮效果](20261011-mjlab-half-assistance-eval-1000.md)。
