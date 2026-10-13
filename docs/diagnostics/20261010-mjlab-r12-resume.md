# r12关闭knee_limit并完整续训检查

2026-10-10，26_wheelleg、HEAD58947e8加未提交改动。用户明确要求从r11完成1000轮模型追加2000轮。输入SHA256 `6c8b4a39ffa812ded73056bee56673ea82f0180065e5aa03c797048d1ea70634`。

本机独立Python3.12环境执行既有参考检查：32GPU环境、128控制步、1280物理子步、seed42，退出码0，不执行PPO更新。配置／其他26项奖励和PPO与参考核对通过；knee_limit贡献精确0，总奖励差公式误差2.68919393e-08；最大闭链偏差2.3040mm、小于5mm；数值异常0。

实际r11 actor／critic／Adam全部状态逐项torch.equal通过；学习率1e-5、下一轮索引1000、有效累计进度1000；普通跨版本load拒绝，错误control／observation／experiment合同和缺优化器的输入拒绝；迁移历史保存／加载通过。旧r5～r11推理保留。

[完整结果](20261010-mjlab-r12-resume-check.json)、[实际执行版本](20261010-mjlab-r12-resume-check-script.txt)。执行后检查脚本仅将临时路径查找替换为可复现的`--resume-checkpoint`参数；生产训练逻辑未变化。

复现命令（替换为实际r11模型路径）：

```bash
mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_reference_training.py --num-envs 32 --steps 128 --seed 42 --resume-checkpoint <r11-model_999.pt> --out /tmp/chuanliantui-r12-resume-check.json
```

有限检查不代表续训后起立效果改善。远端检查与正式启动将在核实后追加。

远端121个运行文件SHA256一致；32环境／32步加载后策略有限检查退出码0、数值异常0、膝惩罚贡献0，actor／critic／Adam完全恢复。见 [远端预检](20261010-mjlab-r12-cloud-preflight.json)。正式新run的更新前resume_state.pt也与源逐项一致，已进入PPO续训，见 [运行记录](20261010-mjlab-r12-cloud-training.md)。
