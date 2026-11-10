# 历史 PPO400 示例与旧实验入口

以下命令均从仓库根目录执行。模型采纳状态以[项目实验记录](../ai/project-log.md)为准，示例模型不代表当前推荐。

## 2026-10-26 的 PPO400 回放示例

在本目录执行以下命令，加载已采纳的400轮模型及其SMP配方；`--smp-prior`同时选择配套轮角39维观测、正常趴地初态、0.2秒出力等待及零速命令。旧multimode和独立稳站任务的显式入口继续保留。

```bash
ct_checkpoint="logs/mjlab/chuanliantui_huananhu_multimode_smp_wheel_handover_v4_bc_finetune_isaac_standup_v3/2026-10-26_18-56-31_wheel39-phase-mix-bcfinetune-smp-400/model_399.pt"
ct_smp_prior="logs/mjlab/chuanliantui_smp_prior/20261025_student170_remote10000-v2/prior_10000.pt"
scripts/mjlab.sh play --checkpoint "$ct_checkpoint" --smp-prior "$ct_smp_prior" --num-envs 1
```

模型来源、备份和本目录验证见[同步记录](../../artifacts/legacy_mjlab/ppo400-sync-20261026-v1/README.md)。该轮停车未通过，不能据采纳决定写成超过LQR。

