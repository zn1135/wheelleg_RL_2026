# 华南虎 V14 从头训练1000轮

**后续完成状态：1000轮已正常完成，退出码0。实际高度未达标，独立回放与视频见 [model999评估](20261011-mjlab-scut-v14-model999-evaluation.md)。以下保留启动时记录。**

用户在恢复最早华南虎 V14 Flat-v0 配方后明确要求训练1000轮。本次从随机权重开始，没有加载 r14/r15 模型。

## 运行配置

- 开始：2026-10-11 20:52:53（Asia/Shanghai）。
- 配方：`huananhu-v14`，接口 `chuanliantui-mjlab-scut-v14-flat-v1`。
- 4096环境、1000次PPO更新、seed42、CUDA:0，远端RTX4090。
- 固定V14奖励、指数速度跟踪、无高度门控、固定normal模式、高度命令0.20～0.42 m；没有三阶段课程或向上辅助力。实体气弹簧仍工作。
- 使用已检查发布 `/root/gpufree-data/wheelleg-remote/releases/20261011-v14-original-parameters-final`，没有修改机器人或冻结源码。
- job：`/root/gpufree-data/wheelleg-remote/jobs/20261011T125253Z-scut-v14-scratch-1000`；wrapper PID21079，训练PID21080。
- 实际输出：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_205306_764251_scut_v14_flat_scut_v14_flat_scratch_1000`。
- 目标最终模型：上述目录的 `model_999.pt`，本记录创建时尚未完成。

## 启动证据

源文件与上一轮已检查发布的指纹一致，旧续训组退出码0，启动前GPU空闲。实际run中接口、环境数、高度范围、模式配置与V14一致。`model_initial.pt` 完成更新计数0；`model_0.pt` 计数1，actor权重已改变且全部有限，证明已实际执行PPO学习。

启动证明采样于20:53:46：日志27/1000、GPU82%、显存2620 MiB，未退出；收尾复查日志86/1000，训练PID仍运行。早期日志数值异常0，初始ETA约25分钟，仅是当时估计。训练输出尚不作为策略行为验收。

完整证据 [启动JSON](20261011-mjlab-scut-v14-scratch-1000-start.json)，SHA256 `2d6189abb8354061b419d787eae532f9d1447f4475cbd3f77ce259b315fcac13`，从远端下载后校验一致；记录中的本地现存源码指纹核对未发现差异。历史logs／模型未删除、移动或覆盖。

## 实际命令与监控

已在远端通过后台wrapper执行下列命令；不用重复启动：

```bash
cd /root/gpufree-data/wheelleg-remote/releases/20261011-v14-original-parameters-final
PYTHONPATH=/root/gpufree-data/wheelleg-remote/releases/20261011-v14-original-parameters-final/mjlab_training/src \
  /root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u -m ct_mjlab.cli train \
  --training-profile huananhu-v14 --num-envs 4096 --iterations 1000 \
  --seed 42 --device cuda:0 --run-name scut_v14_flat_scratch_1000

# 另一个远端终端监控；本次未启动TensorBoard。
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m tensorboard.main \
  --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_205306_764251_scut_v14_flat_scut_v14_flat_scratch_1000 \
  --port 8080
```

后台进程不依赖SSH会话，退出码保存在job的 `.exit-code`，输出在 `output.log`。后续查看进度或评估应使用本run，勿误用已经完成的半辅助力续训组。
