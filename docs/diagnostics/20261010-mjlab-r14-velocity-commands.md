# r14速度／航向命令有限检查

2026-10-10，本机，分支 `26_wheelleg`，HEAD `58947e8fabc4683b2025d42bda742b06c84dc566` 加工作区改动。本次不启动学习，不运行TensorBoard。配置见 [r14接口](../interfaces/chuanliantui-mjlab-wheelleg-recover-r14.md)。

## 执行及结果

1. 独立Python3.12，`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_velocity_commands.py --historical-checkpoint <r11 model_999.pt> --historical-checkpoint <r13 model_1999.pt> --out docs/diagnostics/20261010-mjlab-r14-velocity-commands-check.json`，seed42，退出码0。实际checkpoint完整路径在 [JSON](20261010-mjlab-r14-velocity-commands-check.json)。8192份命令采样覆盖指定范围，航向比例50.3784%；环绕／限幅、部分环境更新、地面1.5 s过渡、动态模式保留和关闭后期特殊命令检查通过。两个历史actor加载成功，按模型保存配置重建后速度命令保持0。
2. `scripts/mjlab.sh check --num-envs 20 --steps 32 --device cuda:0`，本机RTX4060 Laptop，退出码0。Actor35／Critic78、奖励有限，CUDA物理运行32控制步成功。见 [有限检查摘要](20261010-mjlab-r14-finite-check.json)。
3. `git diff --check`通过。

有限检查不证明策略在非零速度下的跟踪、稳定性或恢复效果。r14尚无训练结果，原零速度模型没有非零命令行为验收；未进行部署端或真机验证。旧r11／r13回放仍用其原命令配置，不会自动换成新的随机速度命令。
