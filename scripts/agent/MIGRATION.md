# 辅助脚本迁移表

旧路径不再提供包装入口；其他运行中任务使用的7个原文件暂留，见[runtime-protection.json](runtime-protection.json)。所有命令从仓库根目录执行，使用表中环境；mjlab 工具先设置 `export PYTHONPATH="$PWD/mjlab_training/src${PYTHONPATH:+:$PYTHONPATH}"`。

归档条目通过[原源码恢复工具](maintenance/restore_script_archive.py)在仓库外复核，见[归档说明](../archive/20261101-script-cleanup/README.md)。源码保全不等于旧实验已重新通过。

## 多模式任务

| 旧文件 | 新入口／状态 | 环境 | 用途与原因 |
|---|---|---|---|
| `assess_multimode_cpu_envelope.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `assess_multimode_cpu_zero_hold.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `assess_multimode_envelope_friction.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `assess_multimode_knockdown_recovery.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `assess_multimode_recovery_height.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `assess_multimode_tracking.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `assess_multimode_zero_hold.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_mjlab_multimode.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_multimode_cpu_sim2sim.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_multimode_final_balance.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_multimode_ground_ratio.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_multimode_recovery.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_multimode_tracking.py` | `-m scripts.agent.motion.assess --selfcheck`（已合并） | mjlab_training/.venv/bin/python | Offline check of the flat tracking criteria used by assess_multimode_tracking.py.；已合并：统一运动判据／离线审计；--selfcheck |
| `compare_6848_leg_oscillation.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `compare_multimode_eval.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `compare_multimode_knockdown.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `inspect_multimode_leg_length.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `measure_multimode_leg_motion.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `record_multimode_knockdown_failures.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `render_multimode_leg_length.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `render_multimode_minimum_leg.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `multimode_snapshot_deviations.json` | 已退役／删除 | 非Python资产 | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |

## 独立稳站

| 旧文件 | 新入口／状态 | 环境 | 用途与原因 |
|---|---|---|---|
| `analyze_chuanliantui_stand.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `assess_chuanliantui_stand.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `assess_ground_chuanliantui_stand.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_chuanliantui_stand.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_chuanliantui_stand_regressions.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_chuanliantui_stand_reset_migration.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_chuanliantui_stand_velocity.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `migrate_chuanliantui_stand_velocity.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `probe_chuanliantui_stand_rewards.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `record_chuanliantui_stand.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `render_chuanliantui_stand.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |

## 稳站v2

| 旧文件 | 新入口／状态 | 环境 | 用途与原因 |
|---|---|---|---|
| `assess_chuanliantui_stand_v2.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `assess_stand_v2_speed_transients.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `assess_stand_v2_tracking.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `assess_stand_v2_yaw_control.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_chuanliantui_stand_v2.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_h7_stand_v2.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_stand_v2_active_weight.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_stand_v2_cpu_launch_stability.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_stand_v2_cpu_sim2sim.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_stand_v2_exploration_resume.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_stand_v2_late_balance.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_stand_v2_nominal_training.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Check the authorized1999 continuation and source snapshots; no simulation or learning.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `check_stand_v2_position_hold.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_stand_v2_post_stand_tracking.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_stand_v2_post_stand_training.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Verify the authorized tracking1000 continuation and saved recipe; no simulation.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `check_stand_v2_retention_seeds.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_stand_v2_speed_commands.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_stand_v2_speed_training.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Verify speed-v6 continuation budget, optimizer progress, source and curriculum.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `check_stand_v2_tight_term.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_stand_v2_tight_training.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Verify a completed B continuation from its launch snapshot; never simulates or learns.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `check_stand_v2_tracking.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_stand_v2_yaw_control.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `compare_stand_v2_acceptance.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `compare_stand_v2_active_weight.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `compare_stand_v2_nominal_inversion.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `compare_stand_v2_phase_mix.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `compare_stand_v2_wheel_normal.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_chuanliantui_stand_v2.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_stand_v2_cpu_launch.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_stand_v2_inversion.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_stand_v2_late_balance.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_stand_v2_launch_reference.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_stand_v2_mode_input.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_stand_v2_pitch_progression.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_stand_v2_rewards_velocity.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `evaluate_stand_v2_phase_mix.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `experiment_stand_v2_phase_mix.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `record_stand_v2_sample.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `render_stand_v2_acceptance.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `render_stand_v2_late_balance.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `render_stand_v2_sample.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `render_stand_v2_speed_comparison.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `render_stand_v2_tracking.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `summarize_stand_v2_hold_trial.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Summarize a bounded yaw/parking trial and paired frozen probes; never trains.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `summarize_stand_v2_post_stand_tracking.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Join verified rise/balance and matched velocity probes; do not impose a new acceptance rule.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `summarize_stand_v2_speed_effect.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Verify paired zero/transient measurements and summarize speed-v6 training effects.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `summarize_stand_v2_split_inversion.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `summarize_stand_v2_tight_effect.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Summarize 3999/A/B paired precision, braking and retained zero-v3.1 cases.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `validate_stand_v2_handover.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |

## 起立／行走双策略

| 旧文件 | 新入口／状态 | 环境 | 用途与原因 |
|---|---|---|---|
| `analyze_walk_shadow_delay.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `analyze_walk_velocity_compensation.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `assess_chuanliantui_walk.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `assess_getup_target_slew.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `audit_walk_compensation_startup.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `audit_walk_estimator_closedloop.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `audit_walk_estimator_final_result.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Audit saved final-test predictions, phase metrics and gates without inference.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `audit_walk_estimator_only_result.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Read-only saved-state audit; no network construction, inference or optimizer.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `audit_walk_estimator_timing.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `audit_walk_estimator_training_data.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `audit_walk_goal_feedback.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `audit_walk_shadow_method.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `audit_walk_shadow_result.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Independently verify sealed shadow-delay scores; never rerun policy or physics.；固定旧实验结果审计依赖冻结源码哈希；使用原源码归档复核，不改旧锁值 |
| `check_chuanliantui_two_policy.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_chuanliantui_two_policy_sim2sim.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_getup_handover.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_getup_height_curriculum.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_getup_height_training.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Verify an authorized fresh getup training budget and saved recipe; no learning.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `check_getup_leg_smoothing.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_getup_nominal_state.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_getup_target_slew.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_getup_target_slew_state.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_getup_torques.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_walk_command_readiness.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_walk_direction_curriculum.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_walk_leg_symmetry.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_walk_resume.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_walk_tracking_economy.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `check_walk_training.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Verify a fresh or resumed walk-v8 budget against launch artifacts; CPU only, no learning.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `check_walk_vertical_reward.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `compare_walk_checkpoints.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `compare_walk_estimator_trial.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `compare_walk_feedback.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `compare_walk_height_response.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_getup_release.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_getup_start.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_walk_estimator_closedloop.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_walk_forward_failure.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_walk_goal_feedback.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_walk_height.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_walk_v8_regression.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `diagnose_walk_velocity_compensation.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `evaluate_walk_estimator_final.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `probe_getup_torque_rewards.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `record_walk_estimator_trial.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `record_walk_shadow_sensors.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `render_getup_target_slew.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `report_walk_estimator_closedloop.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Read-only three-seed report of sealed estimator-in-control comparisons.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `report_walk_estimator_only.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Plot sealed encoder-only metric files and the optimizer journal, read-only.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `summarize_getup_release.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Audit saved getup release traces and plot the release mechanism; no simulation.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `summarize_walk_goal_feedback.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `summarize_walk_leg_swing.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | CPU-only audit of saved walk leg motion; no physics step, policy or learning.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `summarize_walk_v8_regression.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Read saved fixed-request traces; audit support-trigger timing and plot comparisons.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `train_walk_estimator_only.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `walk_estimator_offline.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `walk_sensor_velocity.py` | 已退役／删除 | mjlab_training/.venv/bin/python | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |

## SuperMimic

| 旧文件 | 新入口／状态 | 环境 | 用途与原因 |
|---|---|---|---|
| `assess_chuanliantui_supermimic.py`（运行中原入口暂留） | `-m scripts.agent.supermimic.assess_chuanliantui_supermimic` | mjlab_training/.venv/bin/python | Measure a frozen SuperMimic policy or original PPO reference under one protocol.；保留现有能力；未据名称或无引用推定废弃；另一任务正在续训及验收，原字节入口暂留，禁止本次移除 |
| `check_chuanliantui_supermimic.py` | `-m scripts.agent.supermimic.check_chuanliantui_supermimic` | mjlab_training/.venv/bin/python | V11 true contact mixture/five-substep path, BPTT, checkpoint and ONNX; zero updates.；保留现有能力；未据名称或无引用推定废弃 |
| `check_supermimic_alternating.py` | `-m scripts.agent.supermimic.check_supermimic_alternating` | mjlab_training/.venv/bin/python | Finite V13 gradient, stage, episode-boundary and saved-run checks.；保留现有能力；未据名称或无引用推定废弃 |
| `check_supermimic_analytic_auxiliaries.py` | `-m scripts.agent.supermimic.check_supermimic_analytic_auxiliaries` | mjlab_training/.venv/bin/python | Isolated recorded-data check of analytic motor torques and integrated wheel q.；保留现有能力；未据名称或无引用推定废弃 |
| `check_supermimic_bptt.py` | `-m scripts.agent.supermimic.check_supermimic_bptt` | mjlab_training/.venv/bin/python | V11 physical-path adapters, BPTT/tail oracles and mocked updates; CPU only.；保留现有能力；未据名称或无引用推定废弃 |
| `check_supermimic_contact_inputs.py` | `-m scripts.agent.supermimic.check_supermimic_contact_inputs` | mjlab_training/.venv/bin/python | V11 CPU contact encoding, fixed inputs and real V9/V10 actor compatibility; zero updates.；保留现有能力；未据名称或无引用推定废弃 |
| `check_supermimic_loss_scaling.py` | `-m scripts.agent.supermimic.check_supermimic_loss_scaling` | mjlab_training/.venv/bin/python | V11 four-coordinate loss, gradients and recipe guards; zero optimizer updates.；保留现有能力；未据名称或无引用推定废弃 |
| `check_supermimic_motion.py` | `-m scripts.agent.supermimic.check_supermimic_motion` | mjlab_training/.venv/bin/python | V12 motion-only zero-update checks or read-only frozen-run audit.；保留现有能力；未据名称或无引用推定废弃 |
| `check_supermimic_motion_retrain.py` | `-m scripts.agent.supermimic.check_supermimic_motion_retrain` | mjlab_training/.venv/bin/python | V12 thirteen-reward cold retraining checks and frozen-run evidence audit.；保留现有能力；未据名称或无引用推定废弃 |
| `check_supermimic_motion_retrain_counts.py` | `-m scripts.agent.supermimic.check_supermimic_motion_retrain`（已合并） | mjlab_training/.venv/bin/python | V12 thirteen-reward cold retraining checks and frozen-run evidence audit.；并入修正浮点计数误判的统一重训检查 |
| `check_supermimic_training.py` | `-m scripts.agent.supermimic.check_supermimic_training` | mjlab_training/.venv/bin/python | Read-only CPU audit of a SuperMimic run; never starts training or simulation.；保留现有能力；未据名称或无引用推定废弃 |
| `check_supermimic_v11_world_only.py` | `-m scripts.agent.supermimic.check_supermimic_v11_world_only` | mjlab_training/.venv/bin/python | Read-only CPU evidence audit for frozen-actor V11 world fitting, never training.；保留现有能力；未据名称或无引用推定废弃 |
| `check_supermimic_v13_continuation.py`（运行中原入口暂留） | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Finite CPU checks of a full V13 continuation source and completed runs.；已结束实验的固定轮次检查或结果报告；在原源码树复核；另一任务正在续训及验收，原字节入口暂留，禁止本次移除 |
| `check_supermimic_world_only.py` | `-m scripts.agent.supermimic.check_supermimic_world_only` | mjlab_training/.venv/bin/python | Read-only CPU evidence audit for frozen-actor V10 world fitting, never training.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_supermimic_behavior.py`（运行中原入口暂留） | `-m scripts.agent.supermimic.compare_supermimic_behavior` | mjlab_training/.venv/bin/python | Compare paired saved walk measurements on CPU; never run a policy or physics.；保留现有能力；未据名称或无引用推定废弃；另一任务正在续训及验收，原字节入口暂留，禁止本次移除 |
| `diagnose_supermimic_rollout_drift.py` | `-m scripts.agent.supermimic.diagnose_supermimic_rollout_drift` | mjlab_training/.venv/bin/python | Frozen V10: true-state one-tick prediction versus a 24-tick rollout on recorded actions.；保留现有能力；未据名称或无引用推定废弃 |
| `inspect_supermimic_motion_milestone.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | CPU-only inspection of an already locked V12 milestone during training.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `probe_supermimic_action_gradient.py` | `-m scripts.agent.supermimic.probe_supermimic_action_gradient` | mjlab_training/.venv/bin/python | Frozen-policy action-gradient probe using native resets and verified prefix replay.；保留现有能力；未据名称或无引用推定废弃 |
| `probe_supermimic_contact_encoding.py` | `-m scripts.agent.supermimic.probe_supermimic_contact_encoding` | mjlab_training/.venv/bin/python | Frozen SOLREF-input sensitivity on already sealed V9 real-action windows.；保留现有能力；未据名称或无引用推定废弃 |
| `record_supermimic_physical_prediction.py` | `-m scripts.agent.supermimic.record_supermimic_physical_prediction` | mjlab_training/.venv/bin/python | Record complete frozen V10 transitions for fixed-action physical prediction checks.；保留现有能力；未据名称或无引用推定废弃 |
| `render_chuanliantui_supermimic.py` | `-m scripts.agent.supermimic.render_chuanliantui_supermimic` | mjlab_training/.venv/bin/python | Draw saved first-episode physics poses as a 30 s, real-time four-panel video.；保留现有能力；未据名称或无引用推定废弃 |
| `report_supermimic_v13_continuation.py`（运行中原入口暂留） | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | CPU saved-behavior comparisons; only numerical/survival/balance losses stop continuation.；已结束实验的固定轮次检查或结果报告；在原源码树复核；另一任务正在续训及验收，原字节入口暂留，禁止本次移除 |
| `report_supermimic_v13_count_budget.py`（运行中原入口暂留） | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | CPU saved-behavior report for an update-count trial; never controls training.；已结束实验的固定轮次检查或结果报告；在原源码树复核；另一任务正在续训及验收，原字节入口暂留，禁止本次移除 |
| `score_supermimic_motion.py` | `-m scripts.agent.supermimic.score_supermimic_motion` | mjlab_training/.venv/bin/python | V12 motion-only forecasts versus sealed V10/V11 common motion quantities.；保留现有能力；未据名称或无引用推定废弃 |
| `score_supermimic_motion_retrain.py` | `-m scripts.agent.supermimic.score_supermimic_motion_retrain` | mjlab_training/.venv/bin/python | Current-reward V12 retraining versus three sealed historical motion models.；保留现有能力；未据名称或无引用推定废弃 |
| `score_supermimic_physical_prediction.py` | `-m scripts.agent.supermimic.score_supermimic_physical_prediction` | mjlab_training/.venv/bin/python | CPU-only frozen V10 recorded-action prediction, not policy or behavior scoring.；保留现有能力；未据名称或无引用推定废弃 |
| `score_supermimic_v11_paired_prediction.py` | `-m scripts.agent.supermimic.score_supermimic_v11_paired_prediction` | mjlab_training/.venv/bin/python | Frozen V11 worlds on the original V10 recordings and V9 controllers.；保留现有能力；未据名称或无引用推定废弃 |
| `score_supermimic_v13_prediction.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Frozen CPU V13 forecasts on the original sealed V12 comparison windows.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `score_supermimic_v9_paired_baseline.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Frozen original V9 final world on the two new V10-capture action datasets.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `train_supermimic_v13_continuation.py`（运行中原入口暂留） | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | V13 full resumes with actor-count-only or explicitly timed stopping guards.；已结束实验的固定轮次检查或结果报告；在原源码树复核；另一任务正在续训及验收，原字节入口暂留，禁止本次移除 |

## LQR示范／行为克隆／SMP

| 旧文件 | 新入口／状态 | 环境 | 用途与原因 |
|---|---|---|---|
| `analyze_ppo_bc_anchor.py` | `-m scripts.agent.imitation.analyze_ppo_bc_anchor` | mjlab_training/.venv/bin/python | Measure trained action changes on fixed recorded BC states; no optimization.；保留现有能力；未据名称或无引用推定废弃 |
| `analyze_ppo_reward_regression.py` | `-m scripts.agent.imitation.analyze_ppo_reward_regression` | mjlab_training/.venv/bin/python | Read saved PPO/BC trajectories and checkpoints; no simulation or optimization.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_aligned_bc_braking.py` | `-m scripts.agent.imitation.assess_aligned_bc_braking` | mjlab_training/.venv/bin/python | Recompute shared-reference braking evidence and accept only real train-pose labels.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_bc.py` | `-m scripts.agent.imitation.assess_bc` | mjlab_training/.venv/bin/python | Audit the authorized BC budget and heldout prediction errors; never update.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_bc_braking_takeover.py` | `-m scripts.agent.imitation.assess_bc_braking_takeover` | mjlab_training/.venv/bin/python | Audit live braking branches, 500Hz PD/force/FIFO and hidden-reference wheel requests.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_bc_closed_loop.py` | `-m scripts.agent.imitation.assess_bc_closed_loop` | mjlab_training/.venv/bin/python | Recompute heldout BC motion metrics and compare immutable initial states.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_bc_local_feedback.py` | `-m scripts.agent.imitation.assess_bc_local_feedback` | mjlab_training/.venv/bin/python | Replay saved local feedback probes from frozen controller histories; no physics.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_bc_parking.py` | `-m scripts.agent.imitation.assess_bc_parking` | mjlab_training/.venv/bin/python | Assess exact-switch parking branches, teacher validity and training coverage; zero updates.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_fast_bc_screen.py` | `-m scripts.agent.imitation.assess_fast_bc_screen` | mjlab_training/.venv/bin/python | Check cached poses, frozen actions and motion statistics of the fast screen.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_frozen_bc_randomization.py` | `-m scripts.agent.imitation.assess_frozen_bc_randomization` | mjlab_training/.venv/bin/python | Independently recompute frozen-BC action/noise and motion evidence; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_lqr_delay_compensation.py` | `-m scripts.agent.imitation.assess_lqr_delay_compensation` | mjlab_training/.venv/bin/python | Audit exact 500Hz FIFO/PD algebra, preview error and paired balance outcomes.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_lqr_ppo_benchmark.py` | `-m scripts.agent.imitation.assess_lqr_ppo_benchmark` | mjlab_training/.venv/bin/python | Independent contact/trajectory metrics for cached-pose LQR and PPO runs.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_lqr_student_motion.py` | `-m scripts.agent.imitation.assess_lqr_student_motion` | mjlab_training/.venv/bin/python | Compare saved student-PD motions with matched LQR demonstrations; no simulation.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_lqr_vmc.py` | `-m scripts.agent.imitation.assess_lqr_vmc` | mjlab_training/.venv/bin/python | Finite teacher behavior assessment; no training, policy or automatic resets.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_ppo400_import.py` | `-m scripts.agent.imitation.assess_ppo400_import` | mjlab_training/.venv/bin/python | Audit copied files, protected local changes, model assets and destination checks.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_ppo_bc_finetune.py` | `-m scripts.agent.imitation.assess_ppo_bc_finetune` | mjlab_training/.venv/bin/python | Recompute frozen refinement replay metrics and checkpoint provenance; no simulation.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_ppo_smp_comparison.py` | `-m scripts.agent.imitation.assess_ppo_smp_comparison` | mjlab_training/.venv/bin/python | Audit matched initial states and both explicitly budgeted PPO runs; no training.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_ppo_wheel_credit.py` | `-m scripts.agent.imitation.assess_ppo_wheel_credit` | mjlab_training/.venv/bin/python | Recompute frozen wheel-pulse returns and actual PPO GAE directions.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_preserved_bc_feedback.py` | `-m scripts.agent.imitation.assess_preserved_bc_feedback` | mjlab_training/.venv/bin/python | Measure retained local feedback on untouched heldout standing anchors.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_smp_ppo.py` | `-m scripts.agent.imitation.assess_smp_ppo` | mjlab_training/.venv/bin/python | Audit a first SMP PPO run's budget/checkpoint/logs; no optimizer updates.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_smp_prior.py` | `-m scripts.agent.imitation.assess_smp_prior` | mjlab_training/.venv/bin/python | Assess a saved SMP prior and its training log; never updates model parameters.；保留现有能力；未据名称或无引用推定废弃 |
| `check_bc.py` | `-m scripts.agent.imitation.check_bc` | mjlab_training/.venv/bin/python | Check BC timing, grouped split, network parity and gradients without updates.；保留现有能力；未据名称或无引用推定废弃 |
| `check_bc_braking.py` | `-m scripts.agent.imitation.check_bc_braking` | mjlab_training/.venv/bin/python | Audit qualified braking labels, measured stage entry and immutable heldout data.；保留现有能力；未据名称或无引用推定废弃 |
| `check_bc_braking_takeover.py` | `-m scripts.agent.imitation.check_bc_braking_takeover` | mjlab_training/.venv/bin/python | Train-pose student/teacher braking diagnosis with live duplicate-world prefixes.；保留现有能力；未据名称或无引用推定废弃 |
| `check_bc_corrections.py` | `-m scripts.agent.imitation.check_bc_corrections` | mjlab_training/.venv/bin/python | Audit teacher queries on student trajectories and exclude heldout/truncated labels.；保留现有能力；未据名称或无引用推定废弃 |
| `check_bc_init.py` | `-m scripts.agent.imitation.check_bc_init` | mjlab_training/.venv/bin/python | Check real BC actor initialization into fresh SMP PPO; no parameter updates.；保留现有能力；未据名称或无引用推定废弃 |
| `check_bc_local_feedback.py` | `-m scripts.agent.imitation.check_bc_local_feedback` | mjlab_training/.venv/bin/python | Compare one-step teacher/actor feedback at identical live standing states.；保留现有能力；未据名称或无引用推定废弃 |
| `check_bc_parking.py` | `-m scripts.agent.imitation.check_bc_parking` | mjlab_training/.venv/bin/python | Zero-update parking diagnosis: teacher queries and delayed PD takeover at 10s.；保留现有能力；未据名称或无引用推定废弃 |
| `check_bc_parking_restore.py` | `-m scripts.agent.imitation.check_bc_parking_restore` | mjlab_training/.venv/bin/python | Restore canonical10s state into a fresh env; unchanged student must reproduce next5 ticks.；保留现有能力；未据名称或无引用推定废弃 |
| `check_bc_preservation.py` | `-m scripts.agent.imitation.check_bc_preservation` | mjlab_training/.venv/bin/python | Check migration, train-only retained skills and the actual preserved sampler.；保留现有能力；未据名称或无引用推定废弃 |
| `check_bc_query_sync.py` | `-m scripts.agent.imitation.check_bc_query_sync` | mjlab_training/.venv/bin/python | Check real QueryAction boundary masks and partial reset without physics or learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_bc_sampling.py` | `-m scripts.agent.imitation.check_bc_sampling` | mjlab_training/.venv/bin/python | Check braking sampling, legacy compatibility and gradients without optimizer updates.；保留现有能力；未据名称或无引用推定废弃 |
| `check_bc_wheel_dataset.py` | `-m scripts.agent.imitation.check_bc_wheel_dataset` | mjlab_training/.venv/bin/python | Check new40D BC data, finite gradients and legacy replay/fail-closed training.；保留现有能力；未据名称或无引用推定废弃 |
| `check_frozen_bc_randomization.py` | `-m scripts.agent.imitation.check_frozen_bc_randomization` | mjlab_training/.venv/bin/python | Four matched40 frozen-BC replays; training conditions and action sampling toggled separately.；保留现有能力；未据名称或无引用推定废弃 |
| `check_lqr_control_clock.py` | `-m scripts.agent.imitation.check_lqr_control_clock` | mjlab_training/.venv/bin/python | Finite analytic checks for retimed H7 derivative, ramp and standup clocks.；保留现有能力；未据名称或无引用推定废弃 |
| `check_lqr_student_closed_loop.py` | `-m scripts.agent.imitation.check_lqr_student_closed_loop` | mjlab_training/.venv/bin/python | Live LQR-to-student-PD closed-loop diagnostics; no policy or optimizer is created.；保留现有能力；未据名称或无引用推定废弃 |
| `check_lqr_vmc.py` | `-m scripts.agent.imitation.check_lqr_vmc` | mjlab_training/.venv/bin/python | Formal H7 numerical checks; run with mjlab_training/.venv/bin/python.；保留现有能力；未据名称或无引用推定废弃 |
| `check_ppo400_import.py` | `-m scripts.agent.imitation.check_ppo400_import` | mjlab_training/.venv/bin/python | Check the imported trained PPO through real train/play/check dispatch; never learns.；保留现有能力；未据名称或无引用推定废弃 |
| `check_ppo_bc_finetune.py` | `-m scripts.agent.imitation.check_ppo_bc_finetune` | mjlab_training/.venv/bin/python | Finite PPO refinement checks and original40 sampled replay; optimizer steps forbidden.；保留现有能力；未据名称或无引用推定废弃 |
| `check_ppo_exploration.py` | `-m scripts.agent.imitation.check_ppo_exploration` | mjlab_training/.venv/bin/python | Check wheel-only variance changes and saved recipe compatibility, no updates.；保留现有能力；未据名称或无引用推定废弃 |
| `check_ppo_finetune_resume.py` | `-m scripts.agent.imitation.check_ppo_finetune_resume` | mjlab_training/.venv/bin/python | Verify nonempty Adam/BC reference restoration and next iteration; never learns.；保留现有能力；未据名称或无引用推定废弃 |
| `check_ppo_phase_mixing.py` | `-m scripts.agent.imitation.check_ppo_phase_mixing` | mjlab_training/.venv/bin/python | Check real staggered starts, motor gating and checkpoint state without updates.；保留现有能力；未据名称或无引用推定废弃 |
| `check_ppo_smp_ablation.py` | `-m scripts.agent.imitation.check_ppo_smp_ablation` | mjlab_training/.venv/bin/python | Matched initialization and actual zero-SMP reward checks; never optimize.；保留现有能力；未据名称或无引用推定废弃 |
| `check_shared_reference.py` | `-m scripts.agent.imitation.check_shared_reference` | mjlab_training/.venv/bin/python | Check shared reference clock, readonly student observation and partial reset.；保留现有能力；未据名称或无引用推定废弃 |
| `check_smp.py` | `-m scripts.agent.imitation.check_smp` | mjlab_training/.venv/bin/python | Formal finite SMP checks; no training loop or optimizer step is run.；保留现有能力；未据名称或无引用推定废弃 |
| `check_smp_policy_source.py` | `-m scripts.agent.imitation.check_smp_policy_source` | mjlab_training/.venv/bin/python | Screen a frozen policy as a standup demonstration source on the current student interface.；保留现有能力；未据名称或无引用推定废弃 |
| `check_smp_ppo.py` | `-m scripts.agent.imitation.check_smp_ppo` | mjlab_training/.venv/bin/python | Finite SMP/PPO integration and score probes; never learn/update/optimizer.step.；保留现有能力；未据名称或无引用推定废弃 |
| `check_smp_reward_scale.py` | `-m scripts.agent.imitation.check_smp_reward_scale` | mjlab_training/.venv/bin/python | Measure actual reward contributions on live student-PD teacher motion; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_smp_student_scores.py` | `-m scripts.agent.imitation.check_smp_student_scores` | mjlab_training/.venv/bin/python | Matched-pose physical negatives and frozen SMP ranking; never trains a policy.；保留现有能力；未据名称或无引用推定废弃 |
| `check_teacher_pd_interface.py` | `-m scripts.agent.imitation.check_teacher_pd_interface` | mjlab_training/.venv/bin/python | Check real teacher torque requests against the student's physical PD interface.；保留现有能力；未据名称或无引用推定废弃 |
| `check_teacher_startup.py` | `-m scripts.agent.imitation.check_teacher_startup` | mjlab_training/.venv/bin/python | Check 200ms teacher motor hold, fresh takeover and independent reset; no training.；保留现有能力；未据名称或无引用推定废弃 |
| `check_wheel39_ppo.py` | `-m scripts.agent.imitation.check_wheel39_ppo` | mjlab_training/.venv/bin/python | Real wheel39 SMP/PPO initialization, sampling and checkpoint checks; no updates.；保留现有能力；未据名称或无引用推定废弃 |
| `check_wheel39_ppo_entries.py` | `-m scripts.agent.imitation.check_wheel39_ppo_entries` | mjlab_training/.venv/bin/python | Exercise actual wheel39 PPO entry dispatch; intercept learn before any rollout/update.；保留现有能力；未据名称或无引用推定废弃 |
| `check_wheel_angle_bc_baseline.py` | `-m scripts.agent.imitation.check_wheel_angle_bc_baseline` | mjlab_training/.venv/bin/python | Zero-update control: preserve stable legacy39 control while ignoring wheel slots.；保留现有能力；未据名称或无引用推定废弃 |
| `check_wheel_angle_bc_corrections.py` | `-m scripts.agent.imitation.check_wheel_angle_bc_corrections` | mjlab_training/.venv/bin/python | Audit actual teacher recovery labels for wheel-angle BC; performs no update.；保留现有能力；未据名称或无引用推定废弃 |
| `check_wheel_angle_handover.py` | `-m scripts.agent.imitation.check_wheel_angle_handover` | mjlab_training/.venv/bin/python | Check first-rise wheel-angle clearing, history isolation and legacy compatibility; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_wheel_angle_handover_data.py` | `-m scripts.agent.imitation.check_wheel_angle_handover_data` | mjlab_training/.venv/bin/python | Audit actual first-recovery wheel clearing in fresh teacher records; no training.；保留现有能力；未据名称或无引用推定废弃 |
| `check_wheel_angle_observations.py` | `-m scripts.agent.imitation.check_wheel_angle_observations` | mjlab_training/.venv/bin/python | Check reset-relative wheel-angle slots in the 39D profile; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_wheel_lqr_reference.py` | `-m scripts.agent.imitation.check_wheel_lqr_reference` | mjlab_training/.venv/bin/python | Check shared wheel/LQR position resets and unchanged default controller; no training.；保留现有能力；未据名称或无引用推定废弃 |
| `collect_bc_braking.py` | `-m scripts.agent.imitation.collect_bc_braking` | mjlab_training/.venv/bin/python | Collect train-pose, zero-command teacher braking through the original student PD.；保留现有能力；未据名称或无引用推定废弃 |
| `collect_lqr_student.py` | `-m scripts.agent.imitation.collect_lqr_student` | mjlab_training/.venv/bin/python | Collect screened LQR demonstrations through the unchanged student PD interface.；保留现有能力；未据名称或无引用推定废弃 |
| `collect_reference_bc.py` | `-m scripts.agent.imitation.collect_reference_bc` | mjlab_training/.venv/bin/python | Recollect all BC source classes with the same pre-step shared reference input.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_lqr_policy_motion.py` | `-m scripts.agent.imitation.compare_lqr_policy_motion` | mjlab_training/.venv/bin/python | Compare late-episode leg motion and parking from recorded teacher/student traces.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_smp_motion_scores.py` | `-m scripts.agent.imitation.compare_smp_motion_scores` | mjlab_training/.venv/bin/python | Compare real teacher, passive and failed-policy motion; no parameter updates.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_aligned_bc_braking.py` | `-m scripts.agent.imitation.diagnose_aligned_bc_braking` | mjlab_training/.venv/bin/python | Train-only live student/teacher braking probes and actual takeover; no optimizer.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_bc_wheel_input.py` | `-m scripts.agent.imitation.diagnose_bc_wheel_input` | mjlab_training/.venv/bin/python | Zero-update diagnostic: retain a legacy actor and append a zero input column.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_ppo_sampling.py` | `-m scripts.agent.imitation.diagnose_ppo_sampling` | mjlab_training/.venv/bin/python | Frozen-policy rollout audit: episode phases and long-return value estimates.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_ppo_wheel_credit.py` | `-m scripts.agent.imitation.diagnose_ppo_wheel_credit` | mjlab_training/.venv/bin/python | Frozen 60s continuation and same-state wheel pulse/critic comparison.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_smp_score_noise.py` | `-m scripts.agent.imitation.diagnose_smp_score_noise` | mjlab_training/.venv/bin/python | Measure score randomness on fixed late windows versus adjacent-state changes.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_wheel_angle_bc.py` | `-m scripts.agent.imitation.diagnose_wheel_angle_bc` | mjlab_training/.venv/bin/python | Zero-update diagnosis of teacher fitting versus independent student state coverage.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_wheel_angle_bc_handover.py` | `-m scripts.agent.imitation.diagnose_wheel_angle_bc_handover` | mjlab_training/.venv/bin/python | Zero-update diagnostic: teacher starts the robot, then the BC actor takes over.；保留现有能力；未据名称或无引用推定废弃 |
| `eval_bc.py` | `-m scripts.agent.imitation.eval_bc` | mjlab_training/.venv/bin/python | Independent BC control and optional train-only teacher queries; no optimizer.；保留现有能力；未据名称或无引用推定废弃 |
| `eval_wheel39_ppo_init.py` | `-m scripts.agent.imitation.eval_wheel39_ppo_init` | mjlab_training/.venv/bin/python | Original40 nominal replay through the real PPO actor initialized from wheel39 BC; no updates.；保留现有能力；未据名称或无引用推定废弃 |
| `evaluate_lqr_ppo.py` | `-m scripts.agent.imitation.evaluate_lqr_ppo` | mjlab_training/.venv/bin/python | Paired cached-pose nominal20s benchmark: native1ms LQR, BC or real PPO actor.；保留现有能力；未据名称或无引用推定废弃 |
| `evaluate_ppo_exploration.py` | `-m scripts.agent.imitation.evaluate_ppo_exploration` | mjlab_training/.venv/bin/python | Original40 sampled-policy benchmark, changing wheel exploration only.；保留现有能力；未据名称或无引用推定废弃 |
| `export_bc_braking_takeover.py` | `-m scripts.agent.imitation.export_bc_braking_takeover` | mjlab_training/.venv/bin/python | Export only qualified relocked teacher segments and audit BC loader compatibility.；保留现有能力；未据名称或无引用推定废弃 |
| `generate_chuanliantui_closed_chain_references.py` | `-m scripts.agent.imitation.generate_chuanliantui_closed_chain_references` | mjlab_training/.venv/bin/python | 生成 chuanliantui 的模型闭链位置映射检查轨迹。；保留现有能力；未据名称或无引用推定废弃 |
| `lqr_delay_predictor.py` | `-m scripts.agent.imitation.lqr_delay_predictor` | mjlab_training/.venv/bin/python | Nominal-model preview using current state and pending PD commands only.；保留现有能力；未据名称或无引用推定废弃 |
| `prepare_bc_preservation.py` | `-m scripts.agent.imitation.prepare_bc_preservation` | mjlab_training/.venv/bin/python | Generate train-only successful-actor rise and local-feedback retention pairs.；保留现有能力；未据名称或无引用推定废弃 |
| `prepare_bc_wheel_dataset.py` | `-m scripts.agent.imitation.prepare_bc_wheel_dataset` | mjlab_training/.venv/bin/python | Audit historical teacher ownership, then derive encoder travel without relabeling actions.；保留现有能力；未据名称或无引用推定废弃 |
| `prepare_reference_bc.py` | `-m scripts.agent.imitation.prepare_reference_bc` | mjlab_training/.venv/bin/python | Audit recollected shared-reference pairs and assemble all original BC source classes.；保留现有能力；未据名称或无引用推定废弃 |
| `prepare_wheel_angle_bc.py` | `-m scripts.agent.imitation.prepare_wheel_angle_bc` | mjlab_training/.venv/bin/python | Audit fresh wheel-angle actor39 teacher pairs and assemble fixed pose splits.；保留现有能力；未据名称或无引用推定废弃 |
| `prepare_wheel_angle_bc_recovery.py` | `-m scripts.agent.imitation.prepare_wheel_angle_bc_recovery` | mjlab_training/.venv/bin/python | Keep full parking-qualified recoveries and separately screened early rise labels.；保留现有能力；未据名称或无引用推定废弃 |
| `render_aligned_bc_braking.py` | `-m scripts.agent.imitation.render_aligned_bc_braking` | mjlab_training/.venv/bin/python | Render actual 30s independent-student/teacher-takeover cases and source plots.；保留现有能力；未据名称或无引用推定废弃 |
| `render_bc.py` | `-m scripts.agent.imitation.render_bc` | mjlab_training/.venv/bin/python | Full-length paired teacher/BC videos from saved states; never resimulates.；保留现有能力；未据名称或无引用推定废弃 |
| `render_bc_braking.py` | `-m scripts.agent.imitation.render_bc_braking` | mjlab_training/.venv/bin/python | Render full accepted/rejected braking traces without re-integrating physics.；保留现有能力；未据名称或无引用推定废弃 |
| `render_bc_braking_takeover.py` | `-m scripts.agent.imitation.render_bc_braking_takeover` | mjlab_training/.venv/bin/python | Render actual full student/teacher histories; never splice or restore prefixes.；保留现有能力；未据名称或无引用推定废弃 |
| `render_bc_parking.py` | `-m scripts.agent.imitation.render_bc_parking` | mjlab_training/.venv/bin/python | Render the common student prefix and four real parking continuations at60fps.；保留现有能力；未据名称或无引用推定废弃 |
| `render_fast_bc_screen.py` | `-m scripts.agent.imitation.render_fast_bc_screen` | mjlab_training/.venv/bin/python | Recorded20s baseline versus low-noise frozen BC; no simulation or learning.；保留现有能力；未据名称或无引用推定废弃 |
| `render_frozen_bc_randomization.py` | `-m scripts.agent.imitation.render_frozen_bc_randomization` | mjlab_training/.venv/bin/python | Render fixed case0 in all four frozen-BC conditions from recorded states.；保留现有能力；未据名称或无引用推定废弃 |
| `render_lqr_delay_compensation.py` | `-m scripts.agent.imitation.render_lqr_delay_compensation` | mjlab_training/.venv/bin/python | Render original teacher and paired before/after 10ms-delay states; no simulation.；保留现有能力；未据名称或无引用推定废弃 |
| `render_lqr_ppo_benchmark.py` | `-m scripts.agent.imitation.render_lqr_ppo_benchmark` | mjlab_training/.venv/bin/python | Uncut20s native-LQR / frozen-BC / trained-PPO recorded-state video.；保留现有能力；未据名称或无引用推定废弃 |
| `render_lqr_student_comparison.py` | `-m scripts.agent.imitation.render_lqr_student_comparison` | mjlab_training/.venv/bin/python | Render matched recorded LQR/direct and live student-PD cases without resimulation.；保留现有能力；未据名称或无引用推定废弃 |
| `render_lqr_vmc_demo.py` | `-m scripts.agent.imitation.render_lqr_vmc_demo` | mjlab_training/.venv/bin/python | Render one accepted LQR+VMC demonstration from recorded states, without simulation.；保留现有能力；未据名称或无引用推定废弃 |
| `render_ppo_smp_comparison.py` | `-m scripts.agent.imitation.render_ppo_smp_comparison` | mjlab_training/.venv/bin/python | Uncut20s four-panel teacher/BC/task-only/SMP replay from recorded physical states.；保留现有能力；未据名称或无引用推定废弃 |
| `render_preserved_bc.py` | `-m scripts.agent.imitation.render_preserved_bc` | mjlab_training/.venv/bin/python | Render preserved BC cases, prioritizing actual loss of balance over parking-only failures.；保留现有能力；未据名称或无引用推定废弃 |
| `replay_teacher_pd.py` | `-m scripts.agent.imitation.replay_teacher_pd` | mjlab_training/.venv/bin/python | Finite teacher-derived PD target replay through real student physics; no training.；保留现有能力；未据名称或无引用推定废弃 |
| `screen_frozen_bc_conditions.py` | `-m scripts.agent.imitation.screen_frozen_bc_conditions` | mjlab_training/.venv/bin/python | Fast frozen-BC screening from cached initial states; one candidate gets full40 replay.；保留现有能力；未据名称或无引用推定废弃 |
| `select_preserved_bc.py` | `-m scripts.agent.imitation.select_preserved_bc` | mjlab_training/.venv/bin/python | Evaluate already-saved preserved BC checkpoints; never starts training.；保留现有能力；未据名称或无引用推定废弃 |
| `train_ppo_smp_comparison.py` | `-m scripts.agent.imitation.train_ppo_smp_comparison` | mjlab_training/.venv/bin/python | Run one explicitly budgeted comparison group and preserve its true initial state.；保留现有能力；未据名称或无引用推定废弃 |
| `lqr_vmc_reference.c` | `scripts/agent/imitation/lqr_vmc_reference.c` | 非Python资产 | 辅助资产；随调用方迁移 |

## Isaac旧任务与资产工具

| 旧文件 | 新入口／状态 | 环境 | 用途与原因 |
|---|---|---|---|
| `check_action_delay.py` | `-m scripts.agent.isaac.check_action_delay` | Isaac Python 3.8 | CPU 检查真实源码中的动作队列片段；不创建仿真，不替代行为验收。；保留现有能力；未据名称或无引用推定废弃 |
| `check_chuanliantui_base_link_airborne_reward.py` | `-m scripts.agent.isaac.check_chuanliantui_base_link_airborne_reward` | Isaac Python 3.8 | CPU 检查起立任务 base_link 接地惩罚；不创建仿真或训练产物。；保留现有能力；未据名称或无引用推定废弃 |
| `check_chuanliantui_closed.py` | `-m scripts.agent.isaac.check_chuanliantui_closed` | Isaac Python 3.8 | chuanliantui 真实闭链 MJCF 的静态结构和约束检查。；保留现有能力；未据名称或无引用推定废弃 |
| `check_chuanliantui_gas_spring.py` | `-m scripts.agent.isaac.check_chuanliantui_gas_spring` | Isaac Python 3.8 | CPU 气弹簧数值/API契约检查；不创建 Isaac 仿真、不替代行为验收。；保留现有能力；未据名称或无引用推定废弃 |
| `check_chuanliantui_new1_train_urdf.py` | `-m scripts.agent.isaac.check_chuanliantui_new1_train_urdf` | Isaac Python 3.8 | 核对真实 chuanliantui_new_1 与其 6-DOF 训练派生 URDF。；保留现有能力；未据名称或无引用推定废弃 |
| `check_chuanliantui_observation.py` | `-m scripts.agent.isaac.check_chuanliantui_observation` | Isaac Python 3.8 | 检查 chuanliantui 25 维 actor 观测、历史和 MuJoCo 部署构造契约。；保留现有能力；未据名称或无引用推定废弃 |
| `check_chuanliantui_train_proxy.py` | `-m scripts.agent.isaac.check_chuanliantui_train_proxy` | Isaac Python 3.8 | 静态检查 chuanliantui 串联训练代理 MJCF。；保留现有能力；未据名称或无引用推定废弃 |
| `check_convex_velocity_rewards.py` | `-m scripts.agent.isaac.check_convex_velocity_rewards` | Isaac Python 3.8 | Check precision braking reward, transfer and real20env reset isolation.；保留现有能力；未据名称或无引用推定废弃 |
| `check_gas_spring_asset_geometry.py` | `-m scripts.agent.isaac.check_gas_spring_asset_geometry` | Isaac Python 3.8 | 气弹簧资产读取检查：MuJoCo tendon、任意轴/RPY、端点变更及左右改名。；保留现有能力；未据名称或无引用推定废弃 |
| `check_isaac_standup_rewards.py` | `-m scripts.agent.isaac.check_isaac_standup_rewards` | Isaac Python 3.8 | Source-formula parity and finite GPU reward/reset/entry checks; no training.；保留现有能力；未据名称或无引用推定废弃 |
| `check_policy_landing_gate.py` | `-m scripts.agent.isaac.check_policy_landing_gate` | Isaac Python 3.8 | CPU 检查首次落地门控：无策略阶段不运行网络，也不进入 PPO/encoder batch。；保留现有能力；未据名称或无引用推定废弃 |
| `check_standing_push_schedule.py` | `-m scripts.agent.isaac.check_standing_push_schedule` | Isaac Python 3.8 | CPU 推扰计划契约检查；无仿真，不代表抗扰行为验收。；保留现有能力；未据名称或无引用推定废弃 |
| `check_standup_precision_rewards.py` | `-m scripts.agent.isaac.check_standup_precision_rewards` | Isaac Python 3.8 | CPU 回归检查：高度奖励不再裁平、起立后膝关节保留行程余量。；保留现有能力；未据名称或无引用推定废弃 |
| `check_standup_pushes.py` | `-m scripts.agent.isaac.check_standup_pushes` | Isaac Python 3.8 | CPU 检查接地后的分级水平推力；不创建物理仿真，不替代行为验收。；保留现有能力；未据名称或无引用推定废弃 |
| `check_standup_termination.py` | `-m scripts.agent.isaac.check_standup_termination` | Isaac Python 3.8 | CPU 检查起立任务的接地门控、持续站稳、跌倒及腾空终止；不创建仿真。；保留现有能力；未据名称或无引用推定废弃 |
| `check_training_resume.py` | `-m scripts.agent.isaac.check_training_resume` | Isaac Python 3.8 | CPU 检查 checkpoint 续训状态；不创建仿真，不读取或改写历史 logs。；保留现有能力；未据名称或无引用推定废弃 |
| `check_zero_hold_training.py` | 已退役／删除 | Isaac Python 3.8 | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `compose_chuanliantui_factor_videos.py` | `-m scripts.agent.isaac.compose_chuanliantui_factor_videos` | Isaac Python 3.8 | Compose existing paired replay videos with their recorded support forces.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_isaac_ppo_rewards.py` | `-m scripts.agent.isaac.diagnose_isaac_ppo_rewards` | Isaac Python 3.8 | Replay a saved benchmark with exact per-term rewards; forbid learning.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_zero_hold_disturbances.py` | 已退役／删除 | Isaac Python 3.8 | 用户明确停止①～⑤；task及专属脚本删除，保留必要共用函数不保留旧CLI |
| `export_chuanliantui_onnx.py` | `-m scripts.agent.isaac.export_chuanliantui_onnx` | Isaac Python 3.8 | 导出完整 chuanliantui ActorCriticSequence 推理策略为 ONNX。；保留现有能力；未据名称或无引用推定废弃 |
| `generate_chuanliantui_closed_mjcf.py` | `-m scripts.agent.isaac.generate_chuanliantui_closed_mjcf` | Isaac Python 3.8 | 从 chuanliantui_new_1 原始 URDF 生成真实闭链静态 MJCF。；保留现有能力；未据名称或无引用推定废弃 |
| `generate_chuanliantui_new1_train_urdf.py` | `-m scripts.agent.isaac.generate_chuanliantui_new1_train_urdf` | Isaac Python 3.8 | 由 chuanliantui_new_1 真实源 URDF 生成 6-DOF 串联训练派生资产。；保留现有能力；未据名称或无引用推定废弃 |
| `generate_chuanliantui_train_proxy_mjcf.py` | `-m scripts.agent.isaac.generate_chuanliantui_train_proxy_mjcf` | Isaac Python 3.8 | 由 Isaac Gym 的 chuanliantui_train.urdf 生成串联训练代理 MJCF。；保留现有能力；未据名称或无引用推定废弃 |
| `plot_chuanliantui_ablation_evaluation.py` | `-m scripts.agent.isaac.plot_chuanliantui_ablation_evaluation` | Isaac Python 3.8 | Plot recorded ablation height trajectories; needs no simulation runtime.；保留现有能力；未据名称或无引用推定废弃 |
| `render_chuanliantui_ablation_replay.py` | `-m scripts.agent.isaac.render_chuanliantui_ablation_replay` | Isaac Python 3.8 | Render recorded MuJoCo qpos into four-policy comparison videos; no simulation.；保留现有能力；未据名称或无引用推定废弃 |
| `review_zero_hold_training.py` | 已归档；恢复后使用原路径 | Isaac Python 3.8 | Read a zero-hold training run and compare paired native CPU rollouts; no learning.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `simplify_chuanliantui_new1_base.py` | `-m scripts.agent.isaac.simplify_chuanliantui_new1_base` | Isaac Python 3.8 | 将 chuanliantui_new_1 的 CAD 底盘网格降到 MuJoCo STL 面数上限以下。；保留现有能力；未据名称或无引用推定废弃 |

## CPU回放／实采／标定

| 旧文件 | 新入口／状态 | 环境 | 用途与原因 |
|---|---|---|---|
| `analyze_chuanliantui_vofa_sysid.py` | `-m scripts.agent.real2sim.analyze_chuanliantui_vofa_sysid` | mjlab_training/.venv/bin/python | 生成 chuanliantui VOFA+ 原始 CSV 的准静态辨识准入报告。；保留现有能力；未据名称或无引用推定废弃 |
| `analyze_h7_latency.py` | `-m scripts.agent.real2sim.analyze_h7_latency` | mjlab_training/.venv/bin/python | Offline H7 timing distributions from hashed mirror-v2/H7OUT001 recordings.；保留现有能力；未据名称或无引用推定废弃 |
| `analyze_h7_rl_capture.py` | `-m scripts.agent.real2sim.analyze_h7_rl_capture` | mjlab_training/.venv/bin/python | Offline fault chronology from a hashed H7 recording; never opens a port.；保留现有能力；未据名称或无引用推定废弃 |
| `analyze_h7_zero_command.py` | `-m scripts.agent.real2sim.analyze_h7_zero_command` | mjlab_training/.venv/bin/python | Offline zero-command telemetry and policy sensitivity, without calibration.；保留现有能力；未据名称或无引用推定废弃 |
| `check_chuanliantui_h7_control.py` | `-m scripts.agent.real2sim.check_chuanliantui_h7_control` | mjlab_training/.venv/bin/python | CPU 对照实际 H7 C 的五杆解算与 RL 力矩链；不连接硬件或运行策略。；保留现有能力；未据名称或无引用推定废弃 |
| `check_fixed_h7_calibration.py` | `-m scripts.agent.real2sim.check_fixed_h7_calibration` | mjlab_training/.venv/bin/python | Verify saved measured-input predictions, splits and protected model fields.；保留现有能力；未据名称或无引用推定废弃 |
| `check_fixed_h7_rounding.py` | `-m scripts.agent.real2sim.check_fixed_h7_rounding` | mjlab_training/.venv/bin/python | Reproduce the historical compensation anomaly offline, separate from calibration.；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_b_zero_bilateral.py` | `-m scripts.agent.real2sim.check_h7_b_zero_bilateral` | mjlab_training/.venv/bin/python | Crosscheck a supported four-DM zero pose against the one-side CAD fit.；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_capture_calibration.py` | `-m scripts.agent.real2sim.check_h7_capture_calibration` | mjlab_training/.venv/bin/python | Finite checks of the new complete-input session fit; no hardware or training.；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_drive_response.py` | `-m scripts.agent.real2sim.check_h7_drive_response` | mjlab_training/.venv/bin/python | Finite offline checks for continuous drive/coast candidates, not acceptance.；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_height_command.py` | `-m scripts.agent.real2sim.check_h7_height_command` | mjlab_training/.venv/bin/python | Check real H7 RC/observation C for training-height reachability and mode regressions.；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_joint_response.py` | `-m scripts.agent.real2sim.check_h7_joint_response` | mjlab_training/.venv/bin/python | Offline four-hip response screening with incomplete 100Hz input snapshots.；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_latency_capture.py` | `-m scripts.agent.real2sim.check_h7_latency_capture` | mjlab_training/.venv/bin/python | Finite actual-C timing capture/CAN checks, decoder roundtrip and optional ARM link.；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_observation.py` | `-m scripts.agent.real2sim.check_h7_observation` | Isaac Python 3.8 | 对照 H7_clion C 观测、CAD 闭链与训练 build_obs；可复算 obs63 串口采集。；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_output_capture.py` | `-m scripts.agent.real2sim.check_h7_output_capture` | mjlab_training/.venv/bin/python | Finite host checks of actual capture/driver/USB code, optional two-machine ARM link.；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_policy_control.py` | `-m scripts.agent.real2sim.check_h7_policy_control` | Isaac Python 3.8 | 在电脑上对照 H7 纯 C 虚拟 PD 与 B 闭链力矩映射；不打开硬件。；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_policy_timing.py` | `-m scripts.agent.real2sim.check_h7_policy_timing` | mjlab_training/.venv/bin/python | Host check of H7 policy divider and recovery timers; no target connection.；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_policy_warmup.py` | `-m scripts.agent.real2sim.check_h7_policy_warmup` | Isaac Python 3.8 | CPU 检查 H7 闭链回放的十拍预热和无效观测恢复，不启动仿真。；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_rl_angle_branch.py` | `-m scripts.agent.real2sim.check_h7_rl_angle_branch` | mjlab_training/.venv/bin/python | Compile the actual H7 angle mapper, observation builder and PD for offline checks.；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_rl_reentry_recording.py` | `-m scripts.agent.real2sim.check_h7_rl_reentry_recording` | mjlab_training/.venv/bin/python | Check actual no-output RL re-entry recordings, never command hardware.；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_simulation_latency.py` | `-m scripts.agent.real2sim.check_h7_simulation_latency` | mjlab_training/.venv/bin/python | Check training PD delays/references and measured playback timing; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_zero_board.py` | `-m scripts.agent.real2sim.check_h7_zero_board` | mjlab_training/.venv/bin/python | Read-only board/CAD check of four zero-pose virtual leg angles.；保留现有能力；未据名称或无引用推定废弃 |
| `check_motor_response.py` | `-m scripts.agent.real2sim.check_motor_response` | mjlab_training/.venv/bin/python | Finite candidate drive-response checks; test tau is hypothetical, no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_policy_diag.py` | `-m scripts.agent.real2sim.check_policy_diag` | mjlab_training/.venv/bin/python | Offline 2701 framing/history/action diagnostic checks; no device or simulation.；保留现有能力；未据名称或无引用推定废弃 |
| `check_policy_run.py` | `-m scripts.agent.real2sim.check_policy_run` | mjlab_training/.venv/bin/python | 合成 2901 帧检查主机协议与人工接管边界；不打开串口。；保留现有能力；未据名称或无引用推定废弃 |
| `check_real2sim_coupling_huananhu.py` | `-m scripts.agent.real2sim.check_real2sim_coupling_huananhu` | mjlab_training/.venv/bin/python | Verify the bounded inertia-allocation diagnostic; no hardware connections.；保留现有能力；未据名称或无引用推定废弃 |
| `check_real2sim_dynamics_huananhu.py` | `-m scripts.agent.real2sim.check_real2sim_dynamics_huananhu` | mjlab_training/.venv/bin/python | Finite offline checks for real2sim dynamics; no serial connection or training.；保留现有能力；未据名称或无引用推定废弃 |
| `check_real2sim_huananhu.py` | `-m scripts.agent.real2sim.check_real2sim_huananhu` | mjlab_training/.venv/bin/python | Check actual H7 mirror packing, damaged streams and measured-motor kinematics.；保留现有能力；未据名称或无引用推定废弃 |
| `check_real2sim_policy_pairs.py` | `-m scripts.agent.real2sim.check_real2sim_policy_pairs` | mjlab_training/.venv/bin/python | Replay recorded H7 obs39 through an explicit ONNX and compare paired action6.；保留现有能力；未据名称或无引用推定废弃 |
| `check_real2sim_wheel_response.py` | `-m scripts.agent.real2sim.check_real2sim_wheel_response` | mjlab_training/.venv/bin/python | Verify saved wheel-response sensitivity experiments; no hardware acceptance.；保留现有能力；未据名称或无引用推定废弃 |
| `check_shared_wheel_response.py` | `-m scripts.agent.real2sim.check_shared_wheel_response` | mjlab_training/.venv/bin/python | Check promoted wheel response, train/native equivalence and recorded replay.；保留现有能力；未据名称或无引用推定废弃 |
| `check_wheel_odometry.py` | `-m scripts.agent.real2sim.check_wheel_odometry` | mjlab_training/.venv/bin/python | Check signed encoder observations and run 20 teacher environments, without learning.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_h7_measured_start.py` | `-m scripts.agent.real2sim.compare_h7_measured_start` | mjlab_training/.venv/bin/python | Offline measured-start-pose dynamics; frozen model, no hardware or training.；保留现有能力；未据名称或无引用推定废弃 |
| `fit_h7_b_zero_from_cad.py` | `-m scripts.agent.real2sim.fit_h7_b_zero_from_cad` | mjlab_training/.venv/bin/python | Fit a *candidate* B CAD-angle offset to a matching SolidWorks zero pose.；保留现有能力；未据名称或无引用推定废弃 |
| `render_h7_zero_command.py` | `-m scripts.agent.real2sim.render_h7_zero_command` | mjlab_training/.venv/bin/python | Render the verified frozen seed42 ground-start example at original speed.；保留现有能力；未据名称或无引用推定废弃 |
| `report_fixed_h7_calibration.py` | `-m scripts.agent.real2sim.report_fixed_h7_calibration` | mjlab_training/.venv/bin/python | Summarize all saved outputs; no physics, firmware or parameter fitting.；保留现有能力；未据名称或无引用推定废弃 |
| `reproduce_fixed_h7_calibration.sh` | `scripts/agent/real2sim/reproduce_fixed_h7_calibration.sh` | bash；指定 mjlab 解释器 | 固定H7校准重现命令；随调用方迁移 |

## 维护与归档复核

| 旧文件 | 新入口／状态 | 环境 | 用途与原因 |
|---|---|---|---|
| `check_wait_training_completion.py` | `-m scripts.agent.maintenance.check_wait_training_completion` | mjlab_training/.venv/bin/python | Finite real-process checks; no training and no external model invocation.；保留现有能力；未据名称或无引用推定废弃 |
| `wait_training_completion.py` | `-m scripts.agent.maintenance.wait_training_completion` | mjlab_training/.venv/bin/python | Wait for a Linux training process and return its exit event to this agent.；保留现有能力；未据名称或无引用推定废弃 |

