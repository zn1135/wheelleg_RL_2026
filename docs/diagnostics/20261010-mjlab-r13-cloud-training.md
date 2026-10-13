# r13宽高度门控，从1000轮追加1000轮

2026-10-10，用户授权“从1000轮续训”，并明确选择“追加1000轮，累计2000轮”。本机分支 `26_wheelleg`，HEAD `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交r13改动；沿用原远端RTX4090实例。

## 配置与恢复证据

- 四项速度奖励高度门控3～6 cm→5～10 cm，膝惩罚-0.5→0；其余训练配置沿用r11。动态normal/recover、四腿两轮动作／观测顺序、50%普通／50%地面初态；严格起立验收仍±3 cm，见 [接口](../interfaces/chuanliantui-mjlab-wheelleg-recover-r13.md)。
- 输入r11 `model_999.pt`，已完成1000次PPO更新，iter999；SHA256 `6c8b4a39ffa812ded73056bee56673ea82f0180065e5aa03c797048d1ea70634`。输入路径：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_163329_854336_wheelleg_recover_r11_knee_limit_cloud_4096_1000/model_999.pt`。
- 通过 `--resume-wide-height-gate` 显式迁移actor／critic／Adam／学习率／计数；修改门控和膝惩罚的两项记录写入checkpoint。普通旧版本续训和旧膝惩罚单项入口仍拒绝。没有使用r12退化后的3000轮策略。
- 4096环境，seed42，无头；源快照 `20261010T123852Z-58947e8-r13-ad109e`，123个运行源码／资产文件SHA256已核对。
- 本机和远端各32环境、16步有限检查通过；完整actor、critic、Adam逐项相等，学习率1e-5，计数1000，数值异常0。见 [本机检查](20261010-mjlab-r13-wide-resume-check.json)及 [远端预检](20261010-mjlab-r13-cloud-preflight.json)。
- 正式run更新前 `resume_state.pt` 再次与源模型逐项核对相等：actor／critic／Adam全部恢复；next_iteration=1000、effective_completed_updates=1000、学习率1e-5、门控[0.05,0.10]、knee_limit=0。
- 仿真状态、回合／延迟缓存重新初始化；完整PPO状态恢复不等于物理轨迹或随机数流精确重放。

## 启动状态

截至 `2026-10-10T20:41:10.561926+08:00`，后台PID3684存活，无退出码。已观察到日志索引1013/2000和 `model_1000.pt`，当前仍在训练。首个新迭代1000，预计最终model_1999，累计完成2000次PPO更新。启动阶段日志数值异常0、knee_limit贡献0，ETA约24分钟；不能据此判定最终策略效果。

运行目录：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_204040_465892_wheelleg_recover_r13_wide_resume1000_plus1000_cloud_4096`。
控制台：`/root/gpufree-data/wheelleg-remote/releases/20261010T123852Z-58947e8-r13-ad109e/.remote-job/output.log`。
退出码和完成时间：同目录 `exit-code`／`completed-at`，训练结束时由后台脚本生成。

## 实际命令

已启动，不要重复执行。以下在远端快照根、`PYTHONPATH=mjlab_training/src`、`MUJOCO_GL=disable`下执行：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m ct_mjlab.cli train --num-envs 4096 --iterations 1000 --resume /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_163329_854336_wheelleg_recover_r11_knee_limit_cloud_4096_1000/model_999.pt --resume-wide-height-gate --run-name wheelleg_recover_r13_wide_resume1000_plus1000_cloud_4096
```

TensorBoard（远端另开终端，本次未启动）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m tensorboard.main --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_204040_465892_wheelleg_recover_r13_wide_resume1000_plus1000_cloud_4096 --host 127.0.0.1 --port 8080
```

本机转发，然后访问 `http://127.0.0.1:8080`：

```bash
ssh -p 30702 -i /home/zn/.ssh/id_ed25519 -N -L 8080:127.0.0.1:8080 root@183.147.142.40
```

完整版本、实际恢复核对及当时日志见 [JSON](20261010-mjlab-r13-cloud-training.json)。历史logs／源checkpoint保留；本次未执行Git提交、H7改动或实例自动关机设置变更。

## 完成与效果更新

2026-10-10 21:08:42正常结束，退出码0，追加1000轮、累计2000轮，耗时27分40秒。最终model_1999已下载并校验SHA256、iter1999和有效累计更新2000。上文启动状态为历史记录，当前训练已完成。

末轮平均reward -13.02，平均回合61.41步约1.23s，严格恢复0；GPU64环境全部在0.58～2.00s内物理失败，严格起立和20s存活均0/64，数值异常0。CPU单例0.76s物理失败。放宽高度门控未解决稳定起立问题，详见 [最终效果](20261010-mjlab-r13-cloud-model1999-evaluation.md)。TensorBoard未启动，实例未自动关机。
