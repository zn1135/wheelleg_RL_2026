# r12关闭knee_limit，从1000轮追加2000轮

2026-10-10，用户明确授权“关闭knee_limit从1000轮续训2000轮”。本机26_wheelleg，HEAD `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交r12改动；使用原远端RTX4090实例vajnykxo-tp49pgnv。

## 配置与恢复证据

- knee_limit权重-0.5→0，其余训练参数、观测／动作、机器人和气弹簧沿用r11。新接口r12，见 [接口](../interfaces/chuanliantui-mjlab-wheelleg-recover-r12.md)。
- 输入 `/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_163329_854336_wheelleg_recover_r11_knee_limit_cloud_4096_1000/model_999.pt`；SHA256 `6c8b4a39ffa812ded73056bee56673ea82f0180065e5aa03c797048d1ea70634`，已完成1000轮，iter999。
- 恢复actor、critic、Adam全部moment／step和学习率1e-5；没有随机重新训练。仿真回合／历史缓存重新初始化，正常续训不承诺物理轨迹或随机数流精确重放。
- next_iteration=1000，追加2000轮、目标累计3000轮；首个新model_1000，最终model_2999。
- 4096环境，seed42，headless。源快照 `20261010T091548Z-58947e8-us83OS6Mqj`；上传后及实际run快照的121个运行文件SHA256与本机一致。
- 本机32环境／128步有限检查及状态逐项恢复通过；远端32环境／32步已恢复策略预检通过，数值异常0，knee_limit贡献0，详见 [检查](20261010-mjlab-r12-resume.md)。
- 真实正式run保存更新前resume_state.pt；与输入源逐项torch.equal确认actor／critic／Adam完全一致，学习率1e-5，已完成更新1000，下一轮1000。模型不是新随机初始化。

## 启动状态

截至 `2026-10-10T17:20:18.704514+08:00`，后台PID3004存活，无退出码。已观察到日志索引1028/3000，即累计完成1029轮，本次新增29轮。model_1000已保存，effective_completed_updates1001，SHA256 `eb4a6b89c7634ff2a3adfd645d380001025ff84daa7e56a0f7f435a0e0d7899e`，迁移来源写入每个checkpoint。

当时每轮约1.42秒，ETA约46分20秒；日志数值异常0、knee_limit贡献0。启动阶段完成回合样本尚不代表最终效果。没有观察到训练完成。

运行目录：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_171930_580570_wheelleg_recover_r12_no_knee_resume1000_plus2000_cloud_4096`。
控制台：`/root/gpufree-data/wheelleg-remote/releases/20261010T091548Z-58947e8-us83OS6Mqj/.remote-job/output.log`。
退出码：同目录exit-code，训练结束由后台脚本生成。

## 实际命令

已启动，不要重复执行：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m ct_mjlab.cli train --num-envs 4096 --iterations 2000 --seed 42 --device cuda:0 --resume /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_163329_854336_wheelleg_recover_r11_knee_limit_cloud_4096_1000/model_999.pt --resume-disable-knee-limit --run-name wheelleg_recover_r12_no_knee_resume1000_plus2000_cloud_4096
```

TensorBoard（远端另开终端，本次未启动）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m tensorboard.main --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_171930_580570_wheelleg_recover_r12_no_knee_resume1000_plus2000_cloud_4096 --host 127.0.0.1 --port 8080
```

本机端口转发：

```bash
ssh -p 30702 -i /home/zn/.ssh/id_ed25519 -N -L 8080:127.0.0.1:8080 root@183.147.142.40
```

完整版本／恢复校验／初始检查点／当时日志见 [JSON](20261010-mjlab-r12-cloud-training.json)。新run与r11源logs分开，未覆盖或删除历史模型；未执行Git提交或H7改动。TensorBoard未启动，实例按原设置不自动关机。

## 完成与效果更新

2026-10-10 18:14:30正常完成新增2000轮，累计3000轮，退出码0，耗时54分19秒。最终model_2999已下载并核对SHA256与r12接口。上述启动状态为历史记录，当前训练已完成。

末轮平均回合54.35步约1.09s。独立CPU回放1.58s物理失败；GPU32×2组全部在0.58～1.68s物理失败，严格起立0/64、连续20s存活0/64、数值异常0，续训后较r11退化。训练height_error按固定20s归一化，短回合0.0068不能解释为实际误差0.68cm。详见 [最终效果](20261010-mjlab-r12-cloud-model2999-evaluation.md)。实例未自动关机。
