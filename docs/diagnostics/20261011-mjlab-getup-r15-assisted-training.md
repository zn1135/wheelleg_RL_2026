# r15 辅助力课程训练

用户授权“开始训练”，本次单组4096环境、1000次PPO更新、seed=42，从随机权重开始。使用已验证的 `chuanliantui-mjlab-getup-curriculum-r15-assisted-v1`，按整批扶正→站起→行走推进。当前阶段目标0.27 m；后两阶段0.30 m。被动knee_limit保持关闭。

扶正／站起阶段的世界向上辅助力按整机重量与高度差缩放，累计3000次PPO更新后退为零；行走及默认回放关闭。本次1000轮并不保证辅助力预算已退为零，也不保证课程完成。

## 启动证据

- [启动记录](20261011-mjlab-getup-r15-assisted-start.json)：远端RTX4090，训练PID9528，后台启动器PID9527；初始checkpoint完成更新数0、扶正阶段0，首次保存模型完成更新数1，actor权重已改变、Adam状态非空，actor／critic参数有限。
- 核对时控制台迭代索引60/1000，约1.20 s/轮，GPU占用2610 MiB、利用率81%。最新回合统计在索引41：数值异常0、扶正阶段0；这些启动统计不代表最终行为效果。
- [有限验证](../../artifacts/getup-r15-assistance-validation-20261011/check.json)中的四个源码SHA256与训练版本一致。
- 不可变训练源码目录：`/root/gpufree-data/wheelleg-remote/releases/20261011T054119Z-getup-r15-assisted`。
- 实际产物目录：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_134150_495971_getup_r15_assisted`。该目录保存初始模型、后续模型、配置、TensorBoard事件和源码快照。
- 控制台与退出码目录：`/root/gpufree-data/wheelleg-remote/jobs/20261011T054119Z-getup-r15-assisted`，其中 `output.log` 为日志，`.exit-code` 在退出后生成。

目录时间戳来自服务器时钟，与本机时钟有偏差；启动记录的UTC标签来自本机时钟。后台进程独立于SSH连接。

## 已执行命令

以下是本次已执行命令的复现记录，请勿重复启动本次任务。

```bash
cd /root/gpufree-data/wheelleg-remote/releases/20261011T054119Z-getup-r15-assisted
export PYTHONPATH="$PWD/mjlab_training/src"
export MUJOCO_GL=disable
unset WHEELLEG_HUANANHU_INIT_CHECKPOINT
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u -m ct_mjlab.cli train --training-profile getup-r15 --num-envs 4096 --iterations 1000 --seed 42 --device cuda:0 --run-name assisted
```

TensorBoard命令（远端另开终端，本次未启动）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m tensorboard.main --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_134150_495971_getup_r15_assisted --host 127.0.0.1 --port 8080
```

本机转发端口后访问 `http://127.0.0.1:8080`：

```bash
ssh -p 30702 -N -L 8080:127.0.0.1:8080 root@183.147.142.40
```

最终效果需要在训练完成后关闭辅助力回放评估，核对严格起立率、站高误差、稳定时间和数值异常。
