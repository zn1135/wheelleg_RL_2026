# r15 v2从头训练2000轮

用户授权“再加一个从头开始训练2000轮”。新增独立训练组，4096环境、seed42、2000次PPO更新，从随机权重开始，使用已修正轮承重力方向的r15辅助v2完整配置。与原1000→2000轮续训组同时运行在远端RTX4090上。

## 配置与启动证据

- 使用同一冻结源码 `/root/gpufree-data/wheelleg-remote/releases/20261011T061841Z-getup-r15-contact-fix`，启动前逐文件核对源码／资产SHA256，没有修改训练配置。
- 新命令不含resume参数，清除环境中的预训练模型入口。初始检查点完成更新数0，Adam状态为空，初始学习率1e-4，六维动作标准差均为1。
- 初始课程为扶正，目标0.27 m，阶段内步数0、达标计时0、阶段历史为空。辅助力退坡从0轮开始，沿用累计3000更新退出规则。
- 首次保存 `model_0.pt` 完成更新数1，actor已变化、actor／critic参数有限。
- 核对时日志迭代索引23/2000，约2.21秒／轮，两组总显存5215 MiB、GPU利用率100%。原续训PID11122仍存活，新训练PID11408。两组共享GPU会影响速度。
- [完整启动记录](20261011-mjlab-getup-r15-v2-scratch-2000-start.json)包含源码哈希、初始模型核对、实际命令及并行状态。

这是训练启动检查，尚未完成2000轮或评估最终行为。后续比较时需分别记录起点：续训组前1000轮使用有接触判定错误的v1；本组全部2000轮使用修复后的v2。即使累计轮数和seed相同，两组训练经历也不同。

## 目录

- 新组模型／TensorBoard／源码快照：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_142949_678298_getup_r15_contact_fix_scratch_2000`。
- 新作业控制台／启动／退出码：`/root/gpufree-data/wheelleg-remote/jobs/20261011T062851Z-getup-r15-v2-scratch-2000`。日志 `output.log`，退出后生成 `.exit-code`。
- 初始模型 `model_initial.pt`，最终预计模型 `model_1999.pt`。
- 现有续训组的独立记录见 [轮承重修复后续训](20261011-mjlab-getup-r15-contact-fix-training.md)。新组不覆盖它的产物，也不恢复其优化器或课程进度。

## 已执行训练命令

以下为复现记录，不要重复启动本次作业。

```bash
cd /root/gpufree-data/wheelleg-remote/releases/20261011T061841Z-getup-r15-contact-fix
export PYTHONPATH="$PWD/mjlab_training/src"
export MUJOCO_GL=disable
unset WHEELLEG_HUANANHU_INIT_CHECKPOINT
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u -m ct_mjlab.cli train --training-profile getup-r15 --num-envs 4096 --iterations 2000 --seed 42 --device cuda:0 --run-name contact_fix_scratch_2000
```

TensorBoard（远端另开终端，本次未启动）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m tensorboard.main --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_142949_678298_getup_r15_contact_fix_scratch_2000 --host 127.0.0.1 --port 8080
```

本机转发后访问 `http://127.0.0.1:8080`：

```bash
ssh -p 30702 -N -L 8080:127.0.0.1:8080 root@183.147.142.40
```

后台作业与SSH连接分离。没有启动TensorBoard、修改部署代码或执行Git提交。
