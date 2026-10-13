# r11 RTX4090 远端训练启动记录

2026-10-10，用户在恢复限位惩罚后要求“开始训练”。沿用上一轮远端实例 vajnykxo-tp49pgnv 的4096环境／1000轮预算。当前本机训练源码来自 `/home/zn/文档/Wheel-Legged-Gym`，26_wheelleg、HEAD `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交r11改动。

## 实际配置与证据

- RTX4090 24GB，独立Python3.12／mjlab环境。
- 新快照 `20261010T083032Z-58947e8-7dMYGm55E7`；120个训练源码／资产文件上传后的SHA256及实际run的source-state快照与本机一致。
- 原生任务r11；实体被动膝lf1/rf1限位惩罚权重-0.5、余量0.15 rad，其余训练配置沿用r10，见 [接口](../interfaces/chuanliantui-mjlab-wheelleg-recover-r11.md)。
- Actor/Critic/优化器随机初始化；没有resume或init_source；不是接着r10策略续训。
- seed42，headless，4096环境、1000轮；20s回合；50%普通／50%地面、地面机体零姿态；动态normal/recover、四腿两轮顺序、零速度命令与0.27～0.34m目标高度。
- 后台控制PID `2290`；截至 `2026-10-10T16:34:37.946608+08:00` 进程存活，尚无退出码。
- 已观察到日志索引 `48/1000`，即前49轮PPO更新；首个 `model_0.pt` 已保存，checkpoint接口r11、累计训练进度1、SHA256 `abba8cf8e78e46c4e6ea0910278cbf6792bfc360a425f5cbedec36b5fd47b40c`。
- 当时日志限位惩罚贡献-0.1506，数值异常0；每轮约1.37秒、剩余约20分32秒，均为启动时样本，不能作为最终效果或固定ETA。

## 命令与日志

实际生效训练命令（已启动，避免重复启动）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m ct_mjlab.cli train --num-envs 4096 --iterations 1000 --seed 42 --device cuda:0 --run-name wheelleg_recover_r11_knee_limit_cloud_4096_1000
```

运行目录：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_163329_854336_wheelleg_recover_r11_knee_limit_cloud_4096_1000`。
控制台：`/root/gpufree-data/wheelleg-remote/releases/20261010T083032Z-58947e8-7dMYGm55E7/.remote-job/output.log`。
退出码：同目录`exit-code`，训练结束由后台脚本生成。

TensorBoard命令，远端另开终端执行；本次未启动TensorBoard：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m tensorboard.main --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_163329_854336_wheelleg_recover_r11_knee_limit_cloud_4096_1000 --host 127.0.0.1 --port 8080
```

本机端口转发：

```bash
ssh -p 30702 -i /home/zn/.ssh/id_ed25519 -N -L 8080:127.0.0.1:8080 root@183.147.142.40
```

## 检查与边界

本机配置／奖励公式与总奖励差／GPU有限物理／历史模型兼容检查通过，见 [r11检查](20261010-mjlab-r11-knee-limit.md)。新快照远端32环境、128步CLI预检退出码0。后台与检查均设置MUJOCO_GL=disable，启动器清除WHEELLEG_HUANANHU_INIT_CHECKPOINT。

复用原指定实例的SSH配置和上传脚本，临时启动器仅将源码根目录指向当前工作区；未使用旧worktree训练版本。默认1024环境／20000轮由参数尾部4096／1000覆盖，并核对了实际environment.json与runner.json。

完整启动快照见 [JSON](20261010-mjlab-r11-cloud-training.json)。本次记录确认训练正在更新，尚未完成1000轮或行为评估。未执行Git提交，未改动H7，未覆盖本机／远端历史logs，训练结束不自动关机。

## 完成与效果更新

2026-10-10 16:59:03（Asia/Shanghai）正常完成1000轮，退出码0，训练耗时25分12秒。最终model_999已下载到本机独立目录并校验一致。CPU和GPU独立回放均未通过严格起立／站高，主要低姿态15.3～15.9cm并伴随平面移动，详见 [效果评估](20261010-mjlab-r11-cloud-model999-evaluation.md)。上文49轮等状态为启动时历史快照。
