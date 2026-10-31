# 辅助脚本迁移表

旧路径不再提供包装入口；其他运行中任务使用的7个原文件暂留，见[runtime-protection.json](runtime-protection.json)。所有命令从仓库根目录执行，使用表中环境；mjlab 工具先设置 `export PYTHONPATH="$PWD/mjlab_training/src${PYTHONPATH:+:$PYTHONPATH}"`。

归档条目通过[原源码恢复工具](maintenance/restore_script_archive.py)在仓库外复核，见[归档说明](../archive/20261101-script-cleanup/README.md)。源码保全不等于旧实验已重新通过。

## 多模式任务

| 旧文件 | 新入口／状态 | 环境 | 用途与原因 |
|---|---|---|---|
| `assess_multimode_cpu_envelope.py` | `-m scripts.agent.multimode.assess_multimode_cpu_envelope` | mjlab_training/.venv/bin/python | CPU-entry envelope diagnostics: start distribution, friction sweep, engine trace.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_multimode_cpu_zero_hold.py` | `-m scripts.agent.multimode.assess_multimode_cpu_zero_hold` | mjlab_training/.venv/bin/python | Compare two actors in native CPU MuJoCo from identical seeded ground poses.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_multimode_envelope_friction.py` | `-m scripts.agent.multimode.assess_multimode_envelope_friction` | mjlab_training/.venv/bin/python | Envelope/start robustness on the training engine under training-time contact materials.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_multimode_knockdown_recovery.py` | `-m scripts.agent.multimode.assess_multimode_knockdown_recovery` | mjlab_training/.venv/bin/python | 中途倒地恢复验收：结束前连续 1 秒保持双轮平衡（v3）。；保留现有能力；未据名称或无引用推定废弃 |
| `assess_multimode_recovery_height.py` | `-m scripts.agent.multimode.assess_multimode_recovery_height` | mjlab_training/.venv/bin/python | Paired CPU sim2sim diagnostics before/after the recovery-height change.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_multimode_tracking.py` | `-m scripts.agent.multimode.assess_multimode_tracking` | mjlab_training/.venv/bin/python | Measure flat-ground linear-velocity and yaw-rate tracking of a multimode checkpoint.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_multimode_zero_hold.py` | `-m scripts.agent.multimode.assess_multimode_zero_hold` | mjlab_training/.venv/bin/python | Measure 20 s zero-command position holding after normal/ground starts; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_mjlab_multimode.py` | `-m scripts.agent.multimode.check_mjlab_multimode` | mjlab_training/.venv/bin/python | Check the single-line mjlab multimode package against the frozen model_3023 snapshot.；保留现有能力；未据名称或无引用推定废弃 |
| `check_multimode_cpu_sim2sim.py` | `-m scripts.agent.multimode.check_multimode_cpu_sim2sim` | mjlab_training/.venv/bin/python | Check the 39-D multimode CPU sim2sim entry against the training semantics it replays.；保留现有能力；未据名称或无引用推定废弃 |
| `check_multimode_final_balance.py` | `-m scripts.agent.multimode.check_multimode_final_balance` | mjlab_training/.venv/bin/python | 检查最终 1 秒平衡判据；可用已保存的传感器轨迹重算，不重新仿真。；保留现有能力；未据名称或无引用推定废弃 |
| `check_multimode_ground_ratio.py` | `-m scripts.agent.multimode.check_multimode_ground_ratio` | mjlab_training/.venv/bin/python | 检查 multimode 复位的地面（recover）初态比例是否等于配置值。；保留现有能力；未据名称或无引用推定废弃 |
| `check_multimode_recovery.py` | `-m scripts.agent.multimode.check_multimode_recovery` | mjlab_training/.venv/bin/python | 检查 multimode 中途倒地后的 recover 重新触发、恢复窗口与恢复期奖励门控。；保留现有能力；未据名称或无引用推定废弃 |
| `check_multimode_tracking.py` | `-m scripts.agent.motion.assess --selfcheck`（已合并） | mjlab_training/.venv/bin/python | Offline check of the flat tracking criteria used by assess_multimode_tracking.py.；已合并：统一运动判据／离线审计；--selfcheck |
| `compare_6848_leg_oscillation.py` | `-m scripts.agent.multimode.compare_6848_leg_oscillation` | mjlab_training/.venv/bin/python | Compare saved 6848 real telemetry and measured-start simulation; no dynamics.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_multimode_eval.py` | `-m scripts.agent.multimode.compare_multimode_eval` | mjlab_training/.venv/bin/python | 配对比较两次 eval（强制初态）输出：验收 v2 的退化否决。；保留现有能力；未据名称或无引用推定废弃 |
| `compare_multimode_knockdown.py` | `-m scripts.agent.multimode.compare_multimode_knockdown` | mjlab_training/.venv/bin/python | 配对比较两次中途击倒评估：默认用结束前连续 1 秒的最终平衡（v3）。；保留现有能力；未据名称或无引用推定废弃 |
| `inspect_multimode_leg_length.py` | `-m scripts.agent.multimode.inspect_multimode_leg_length` | mjlab_training/.venv/bin/python | Report static leg reach and height-command semantics; no training or rollout.；保留现有能力；未据名称或无引用推定废弃 |
| `measure_multimode_leg_motion.py` | `-m scripts.agent.multimode.measure_multimode_leg_motion` | mjlab_training/.venv/bin/python | Measure standing-still leg motion of a 39-D multimode checkpoint in two backends.；保留现有能力；未据名称或无引用推定废弃 |
| `record_multimode_knockdown_failures.py` | `-m scripts.agent.multimode.record_multimode_knockdown_failures` | mjlab_training/.venv/bin/python | 记录击倒评估；也可按修订的交付定义直接渲染已保存的本次运行轨迹。；保留现有能力；未据名称或无引用推定废弃 |
| `render_multimode_leg_length.py` | `-m scripts.agent.multimode.render_multimode_leg_length` | mjlab_training/.venv/bin/python | Plot measured leg lengths and render an already recorded mjlab trajectory.；保留现有能力；未据名称或无引用推定废弃 |
| `render_multimode_minimum_leg.py` | `-m scripts.agent.multimode.render_multimode_minimum_leg` | mjlab_training/.venv/bin/python | Render the closed static pose at lf1=-0.77/rf1=+0.77, without stepping physics.；保留现有能力；未据名称或无引用推定废弃 |
| `multimode_snapshot_deviations.json` | `scripts/agent/multimode/multimode_snapshot_deviations.json` | 非Python资产 | 辅助资产；随调用方迁移 |

## 独立稳站

| 旧文件 | 新入口／状态 | 环境 | 用途与原因 |
|---|---|---|---|
| `analyze_chuanliantui_stand.py` | `-m scripts.agent.stand.analyze_chuanliantui_stand` | mjlab_training/.venv/bin/python | Read standing-task TensorBoard/assessment artifacts and write an offline report.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_chuanliantui_stand.py` | `-m scripts.agent.stand.assess_chuanliantui_stand` | mjlab_training/.venv/bin/python | Evaluate the independent standing task; never train or operate hardware.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_ground_chuanliantui_stand.py` | `-m scripts.agent.stand.assess_ground_chuanliantui_stand` | mjlab_training/.venv/bin/python | Report original-bank ground get-up separately; never replace fixed acceptance.；保留现有能力；未据名称或无引用推定废弃 |
| `check_chuanliantui_stand.py` | `-m scripts.agent.stand.check_chuanliantui_stand` | mjlab_training/.venv/bin/python | Finite checks for the independent 0.27 m standing task; never calls learn().；保留现有能力；未据名称或无引用推定废弃 |
| `check_chuanliantui_stand_regressions.py` | `-m scripts.agent.stand.check_chuanliantui_stand_regressions` | mjlab_training/.venv/bin/python | Require every ever-passing fixed case, including successes from separate repeats.；保留现有能力；未据名称或无引用推定废弃 |
| `check_chuanliantui_stand_reset_migration.py` | `-m scripts.agent.stand.check_chuanliantui_stand_reset_migration` | mjlab_training/.venv/bin/python | Compare the same policy's fixed set before/after adding mixed training resets.；保留现有能力；未据名称或无引用推定废弃 |
| `check_chuanliantui_stand_velocity.py` | `-m scripts.agent.stand.check_chuanliantui_stand_velocity` | mjlab_training/.venv/bin/python | Historical r25/r26 velocity-profile checks; use its frozen source snapshot.；保留现有能力；未据名称或无引用推定废弃 |
| `migrate_chuanliantui_stand_velocity.py` | `-m scripts.agent.stand.migrate_chuanliantui_stand_velocity` | mjlab_training/.venv/bin/python | Historical 39-to-41 migration; target is the frozen r25/r26 velocity profile.；保留现有能力；未据名称或无引用推定废弃 |
| `probe_chuanliantui_stand_rewards.py` | `-m scripts.agent.stand.probe_chuanliantui_stand_rewards` | mjlab_training/.venv/bin/python | Read actual reward rates before first termination; no learning or acceptance.；保留现有能力；未据名称或无引用推定废弃 |
| `record_chuanliantui_stand.py` | `-m scripts.agent.stand.record_chuanliantui_stand` | mjlab_training/.venv/bin/python | Record a full quiet rollout for diagnosis; never issue an acceptance verdict.；保留现有能力；未据名称或无引用推定废弃 |
| `render_chuanliantui_stand.py` | `-m scripts.agent.stand.render_chuanliantui_stand` | mjlab_training/.venv/bin/python | Render actual stand-task reset states or saved evaluation states; no learning.；保留现有能力；未据名称或无引用推定废弃 |

## 稳站v2

| 旧文件 | 新入口／状态 | 环境 | 用途与原因 |
|---|---|---|---|
| `assess_chuanliantui_stand_v2.py` | `-m scripts.agent.stand_v2.assess_chuanliantui_stand_v2` | mjlab_training/.venv/bin/python | 20 s stand_v2 acceptance: 500 Hz balance, drift reported separately; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_stand_v2_speed_transients.py` | `-m scripts.agent.stand_v2.assess_stand_v2_speed_transients` | mjlab_training/.venv/bin/python | Frozen ±0.5/1/1.5m/s start/stop/reverse acceptance; separate zero-speed v3.1.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_stand_v2_tracking.py` | `-m scripts.agent.stand_v2.assess_stand_v2_tracking` | mjlab_training/.venv/bin/python | Frozen stand_v2 velocity-command probe after ground get-up; no training.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_stand_v2_yaw_control.py` | `-m scripts.agent.stand_v2.assess_stand_v2_yaw_control` | mjlab_training/.venv/bin/python | Frozen 500 Hz yaw-rate/heading-hold measurement; separate from balance v3.1.；保留现有能力；未据名称或无引用推定废弃 |
| `check_chuanliantui_stand_v2.py` | `-m scripts.agent.stand_v2.check_chuanliantui_stand_v2` | mjlab_training/.venv/bin/python | Isaac32 history, get-up boundaries, gradients and finite physics; no parameter updates.；保留现有能力；未据名称或无引用推定废弃 |
| `check_h7_stand_v2.py` | `-m scripts.agent.stand_v2.check_h7_stand_v2` | mjlab_training/.venv/bin/python | Check actual H7 stand-v2 observation/history/SDK code on a host; no hardware IO.；保留现有能力；未据名称或无引用推定废弃 |
| `check_stand_v2_active_weight.py` | `-m scripts.agent.stand_v2.check_stand_v2_active_weight` | mjlab_training/.venv/bin/python | Reward-bit equality, command timing and version isolation; frozen policies only.；保留现有能力；未据名称或无引用推定废弃 |
| `check_stand_v2_cpu_launch_stability.py` | `-m scripts.agent.stand_v2.check_stand_v2_cpu_launch_stability` | mjlab_training/.venv/bin/python | Screen all recorded starts for the reported early native CPU launch; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_stand_v2_cpu_sim2sim.py` | `-m scripts.agent.stand_v2.check_stand_v2_cpu_sim2sim` | mjlab_training/.venv/bin/python | Compare native stand_v2 preprocessing/PD/FIFO/mode timers with the real task.；保留现有能力；未据名称或无引用推定废弃 |
| `check_stand_v2_exploration_resume.py` | `-m scripts.agent.stand_v2.check_stand_v2_exploration_resume` | mjlab_training/.venv/bin/python | Verify the bounded std-floor continuation; no simulation or optimizer updates.；保留现有能力；未据名称或无引用推定废弃 |
| `check_stand_v2_late_balance.py` | `-m scripts.agent.stand_v2.check_stand_v2_late_balance` | mjlab_training/.venv/bin/python | Audit the detailed rerun, contact kinematics and per-condition late failures offline.；保留现有能力；未据名称或无引用推定废弃 |
| `check_stand_v2_nominal_training.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Check the authorized1999 continuation and source snapshots; no simulation or learning.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `check_stand_v2_position_hold.py` | `-m scripts.agent.stand_v2.check_stand_v2_position_hold` | mjlab_training/.venv/bin/python | Finite parking command, compatibility and frozen interface checks; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_stand_v2_post_stand_tracking.py` | `-m scripts.agent.stand_v2.check_stand_v2_post_stand_tracking` | mjlab_training/.venv/bin/python | Check command gates, reward isolation, resume/export and frozen rollouts; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_stand_v2_post_stand_training.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Verify the authorized tracking1000 continuation and saved recipe; no simulation.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `check_stand_v2_retention_seeds.py` | `-m scripts.agent.stand_v2.check_stand_v2_retention_seeds` | mjlab_training/.venv/bin/python | Verify matched randomization inputs and compare late-balance failures on new seeds.；保留现有能力；未据名称或无引用推定废弃 |
| `check_stand_v2_speed_commands.py` | `-m scripts.agent.stand_v2.check_stand_v2_speed_commands` | mjlab_training/.venv/bin/python | Finite command/curriculum/resume checks plus frozen rollout; never learns.；保留现有能力；未据名称或无引用推定废弃 |
| `check_stand_v2_speed_training.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Verify speed-v6 continuation budget, optimizer progress, source and curriculum.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `check_stand_v2_tight_term.py` | `-m scripts.agent.stand_v2.check_stand_v2_tight_term` | mjlab_training/.venv/bin/python | Same-state B/A reward equality, real RewardManager and checkpoint isolation; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_stand_v2_tight_training.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Verify a completed B continuation from its launch snapshot; never simulates or learns.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `check_stand_v2_tracking.py` | `-m scripts.agent.motion.assess --audit-tracking <测量目录>`（已合并） | mjlab_training/.venv/bin/python | Independently check delivered commands and summaries of saved stand_v2 tracking.；已合并：统一运动判据／离线审计；--audit-tracking <测量目录> |
| `check_stand_v2_yaw_control.py` | `-m scripts.agent.stand_v2.check_stand_v2_yaw_control` | mjlab_training/.venv/bin/python | Finite yaw/heading-hold, legacy compatibility and frozen inference checks; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_stand_v2_acceptance.py` | `-m scripts.agent.stand_v2.compare_stand_v2_acceptance` | mjlab_training/.venv/bin/python | Verify recorded 500 Hz stand_v2 verdicts and compare identical initial states.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_stand_v2_active_weight.py` | `-m scripts.agent.stand_v2.compare_stand_v2_active_weight` | mjlab_training/.venv/bin/python | After-training 3999/4999/new paired braking and separate zero-v3.1 report.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_stand_v2_nominal_inversion.py` | `-m scripts.agent.stand_v2.compare_stand_v2_nominal_inversion` | mjlab_training/.venv/bin/python | Recompute paired500Hz zero-speed inversion branches and v3.1 balance from saved data.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_stand_v2_phase_mix.py` | `-m scripts.agent.stand_v2.compare_stand_v2_phase_mix` | mjlab_training/.venv/bin/python | Verify paired phase-mix evaluations and aggregate per-seed outcomes, without simulation.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_stand_v2_wheel_normal.py` | `-m scripts.agent.stand_v2.compare_stand_v2_wheel_normal` | mjlab_training/.venv/bin/python | Compare the wheel-load NORMAL rule on identical recorded stand_v2 first episodes.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_chuanliantui_stand_v2.py` | `-m scripts.agent.stand_v2.diagnose_chuanliantui_stand_v2` | mjlab_training/.venv/bin/python | Measure first-episode stand_v2 motion at 100 Hz; no learning or behavior acceptance.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_stand_v2_cpu_launch.py` | `-m scripts.agent.stand_v2.diagnose_stand_v2_cpu_launch` | mjlab_training/.venv/bin/python | Trace the reported native CPU launch, with fixed initial state and no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_stand_v2_inversion.py` | `-m scripts.agent.stand_v2.diagnose_stand_v2_inversion` | mjlab_training/.venv/bin/python | Trace an existing stand_v2 diagnostic's get-up actions; never update a policy.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_stand_v2_late_balance.py` | `-m scripts.agent.stand_v2.diagnose_stand_v2_late_balance` | mjlab_training/.venv/bin/python | Replay the original ten late-balance cases; record actual contact-point slip, no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_stand_v2_launch_reference.py` | `-m scripts.agent.stand_v2.diagnose_stand_v2_launch_reference` | mjlab_training/.venv/bin/python | Run the reported saved state in the actual training engine without learning.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_stand_v2_mode_input.py` | `-m scripts.agent.stand_v2.diagnose_stand_v2_mode_input` | mjlab_training/.venv/bin/python | Ablate only the policy-visible NORMAL/RECOVER flag of a frozen stand_v2 model.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_stand_v2_pitch_progression.py` | `-m scripts.agent.stand_v2.diagnose_stand_v2_pitch_progression` | mjlab_training/.venv/bin/python | Frozen zero-command pitch measurements and paired checkpoint progression; no training.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_stand_v2_rewards_velocity.py` | `-m scripts.agent.stand_v2.diagnose_stand_v2_rewards_velocity` | mjlab_training/.venv/bin/python | Trace actual reward terms and estimated velocity; optionally replace velocity after 10s. No learning.；保留现有能力；未据名称或无引用推定废弃 |
| `evaluate_stand_v2_phase_mix.py` | `-m scripts.agent.stand_v2.evaluate_stand_v2_phase_mix` | mjlab_training/.venv/bin/python | Paired 40 s nominal/training-condition evaluation; 500 Hz balance, no updates.；保留现有能力；未据名称或无引用推定废弃 |
| `experiment_stand_v2_phase_mix.py` | `-m scripts.agent.stand_v2.experiment_stand_v2_phase_mix` | mjlab_training/.venv/bin/python | Physically stagger stand_v2 resets; isolated experiment, production recipe unchanged.；保留现有能力；未据名称或无引用推定废弃 |
| `record_stand_v2_sample.py` | `-m scripts.agent.stand_v2.record_stand_v2_sample` | mjlab_training/.venv/bin/python | Record one original stand_v2 diagnostic initial state with the same policy.；保留现有能力；未据名称或无引用推定废弃 |
| `render_stand_v2_acceptance.py` | `-m scripts.agent.stand_v2.render_stand_v2_acceptance` | mjlab_training/.venv/bin/python | Render a passed and failed case from one saved acceptance; never resimulate.；保留现有能力；未据名称或无引用推定废弃 |
| `render_stand_v2_late_balance.py` | `-m scripts.agent.stand_v2.render_stand_v2_late_balance` | mjlab_training/.venv/bin/python | Plot the ten rerun tails and render saved trajectories; no integration or learning.；保留现有能力；未据名称或无引用推定废弃 |
| `render_stand_v2_sample.py` | `-m scripts.agent.stand_v2.render_stand_v2_sample` | mjlab_training/.venv/bin/python | Render a recorded stand_v2 sample at its original simulation speed.；保留现有能力；未据名称或无引用推定废弃 |
| `render_stand_v2_speed_comparison.py` | `-m scripts.agent.stand_v2.render_stand_v2_speed_comparison` | mjlab_training/.venv/bin/python | Render matched old/new frozen transient poses, never re-simulates physics.；保留现有能力；未据名称或无引用推定废弃 |
| `render_stand_v2_tracking.py` | `-m scripts.agent.stand_v2.render_stand_v2_tracking` | mjlab_training/.venv/bin/python | Render full saved tracking-probe trajectories; no physics integration or learning.；保留现有能力；未据名称或无引用推定废弃 |
| `summarize_stand_v2_hold_trial.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Summarize a bounded yaw/parking trial and paired frozen probes; never trains.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `summarize_stand_v2_post_stand_tracking.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Join verified rise/balance and matched velocity probes; do not impose a new acceptance rule.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `summarize_stand_v2_speed_effect.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Verify paired zero/transient measurements and summarize speed-v6 training effects.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `summarize_stand_v2_split_inversion.py` | `-m scripts.agent.stand_v2.summarize_stand_v2_split_inversion` | mjlab_training/.venv/bin/python | 按初始左右腿前后分支统计稳站v2配对评估中的倒置（只读已有评估产物，不跑仿真）。；保留现有能力；未据名称或无引用推定废弃 |
| `summarize_stand_v2_tight_effect.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Summarize 3999/A/B paired precision, braking and retained zero-v3.1 cases.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `validate_stand_v2_handover.py` | `-m scripts.agent.stand_v2.validate_stand_v2_handover` | mjlab_training/.venv/bin/python | Compare a simulation-only leg-target handover with the same trained policy.；保留现有能力；未据名称或无引用推定废弃 |

## 起立／行走双策略

| 旧文件 | 新入口／状态 | 环境 | 用途与原因 |
|---|---|---|---|
| `analyze_walk_shadow_delay.py` | `-m scripts.agent.split.analyze_walk_shadow_delay` | mjlab_training/.venv/bin/python | Replay fixed grouped sensor delays through the frozen 2999 encoder only.；保留现有能力；未据名称或无引用推定废弃 |
| `analyze_walk_velocity_compensation.py` | `-m scripts.agent.split.analyze_walk_velocity_compensation` | mjlab_training/.venv/bin/python | Score a frozen sensor-only velocity formula on two separate saved runs, CPU only.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_chuanliantui_walk.py` | `-m scripts.agent.split.assess_chuanliantui_walk` | mjlab_training/.venv/bin/python | Diagnose a frozen walk policy over 64 nominal first episodes and render fixed samples.；保留现有能力；未据名称或无引用推定废弃 |
| `assess_getup_target_slew.py` | `-m scripts.agent.split.assess_getup_target_slew` | mjlab_training/.venv/bin/python | Frozen getup target slew comparison through the original delayed PD.；保留现有能力；未据名称或无引用推定废弃 |
| `audit_walk_compensation_startup.py` | `-m scripts.agent.split.audit_walk_compensation_startup` | mjlab_training/.venv/bin/python | Audit saved initialization errors and kinematic assumptions without physics steps.；保留现有能力；未据名称或无引用推定废弃 |
| `audit_walk_estimator_closedloop.py` | `-m scripts.agent.split.audit_walk_estimator_closedloop` | mjlab_training/.venv/bin/python | Independently audit all three saved diagnostic pairs; never runs a policy.；保留现有能力；未据名称或无引用推定废弃 |
| `audit_walk_estimator_final_result.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Audit saved final-test predictions, phase metrics and gates without inference.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `audit_walk_estimator_only_result.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Read-only saved-state audit; no network construction, inference or optimizer.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `audit_walk_estimator_timing.py` | `-m scripts.agent.split.audit_walk_estimator_timing` | mjlab_training/.venv/bin/python | CPU-only storage/history audit; no simulation, training or optimizer steps.；保留现有能力；未据名称或无引用推定废弃 |
| `audit_walk_estimator_training_data.py` | `-m scripts.agent.split.audit_walk_estimator_training_data` | mjlab_training/.venv/bin/python | Inventory existing complete2999 runs without fitting, scoring or simulation.；保留现有能力；未据名称或无引用推定废弃 |
| `audit_walk_goal_feedback.py` | `-m scripts.agent.split.audit_walk_goal_feedback` | mjlab_training/.venv/bin/python | Read saved feedback traces and MuJoCo quaternion math; never step physics.；保留现有能力；未据名称或无引用推定废弃 |
| `audit_walk_shadow_method.py` | `-m scripts.agent.split.audit_walk_shadow_method` | mjlab_training/.venv/bin/python | CPU source/hash and sentinel audit for the frozen shadow-delay experiment.；保留现有能力；未据名称或无引用推定废弃 |
| `audit_walk_shadow_result.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Independently verify sealed shadow-delay scores; never rerun policy or physics.；固定旧实验结果审计依赖冻结源码哈希；使用原源码归档复核，不改旧锁值 |
| `check_chuanliantui_two_policy.py` | `-m scripts.agent.split.check_chuanliantui_two_policy` | mjlab_training/.venv/bin/python | Formal finite/contract/routing check for separate 25/125 get-up and walk tasks; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_chuanliantui_two_policy_sim2sim.py` | `-m scripts.agent.split.check_chuanliantui_two_policy_sim2sim` | mjlab_training/.venv/bin/python | CPU dual-ONNX switching/outer-loop/observation checks, separate from single-task training.；保留现有能力；未据名称或无引用推定废弃 |
| `check_getup_handover.py` | `-m scripts.agent.split.check_getup_handover` | mjlab_training/.venv/bin/python | Check the actual getup PD handover at 2ms resolution without learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_getup_height_curriculum.py` | `-m scripts.agent.split.check_getup_height_curriculum` | mjlab_training/.venv/bin/python | Run the getup curriculum checks independently of concurrent walking changes.；保留现有能力；未据名称或无引用推定废弃 |
| `check_getup_height_training.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Verify an authorized fresh getup training budget and saved recipe; no learning.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `check_getup_leg_smoothing.py` | `-m scripts.agent.split.check_getup_leg_smoothing` | mjlab_training/.venv/bin/python | Check policy-rate leg smoothing through real getup PD, reset and metadata; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_getup_nominal_state.py` | `-m scripts.agent.split.check_getup_nominal_state` | mjlab_training/.venv/bin/python | Check full getup symmetry cost against the original 1999 reward snapshot; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_getup_target_slew.py` | `-m scripts.agent.split.check_getup_target_slew` | mjlab_training/.venv/bin/python | Independently check saved target-slew traces; no environment or learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_getup_target_slew_state.py` | `-m scripts.agent.split.check_getup_target_slew_state` | mjlab_training/.venv/bin/python | CPU checks for TargetSlew state only; no simulation or training.；保留现有能力；未据名称或无引用推定废弃 |
| `check_getup_torques.py` | `-m scripts.agent.split.check_getup_torques` | mjlab_training/.venv/bin/python | Check getup torque penalties before standing and unchanged weights/other rewards; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_walk_command_readiness.py` | `-m scripts.agent.split.check_walk_command_readiness` | mjlab_training/.venv/bin/python | Walk-native command gate: synthetic boundaries, archived regression and noisy inference; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_walk_direction_curriculum.py` | `-m scripts.agent.split.check_walk_direction_curriculum` | mjlab_training/.venv/bin/python | Walk v8 direction sampling, independent promotion/resume and frozen inference; never learns.；保留现有能力；未据名称或无引用推定废弃 |
| `check_walk_leg_symmetry.py` | `-m scripts.agent.split.check_walk_leg_symmetry` | mjlab_training/.venv/bin/python | Walk symmetry geometry, unchanged reward terms/contracts and archived v6 poses; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_walk_resume.py` | `-m scripts.agent.split.check_walk_resume` | mjlab_training/.venv/bin/python | Check the real walk checkpoint, curriculum boundaries and finite training rollout; never learn.；保留现有能力；未据名称或无引用推定废弃 |
| `check_walk_tracking_economy.py` | `-m scripts.agent.split.check_walk_tracking_economy` | mjlab_training/.venv/bin/python | Check walk v8 tracking boost, relative vx kernel, 5° tilt band, 75% moving episodes and std floor; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `check_walk_training.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Verify a fresh or resumed walk-v8 budget against launch artifacts; CPU only, no learning.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `check_walk_vertical_reward.py` | `-m scripts.agent.split.check_walk_vertical_reward` | mjlab_training/.venv/bin/python | CPU synthetic reward equivalence check; no simulator, checkpoints or updates.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_walk_checkpoints.py` | `-m scripts.agent.split.compare_walk_checkpoints` | mjlab_training/.venv/bin/python | Compare saved fixed-request walk traces on CPU; never simulate or learn.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_walk_estimator_trial.py` | `-m scripts.agent.split.compare_walk_estimator_trial` | mjlab_training/.venv/bin/python | Saved-trace behavior comparison with parent-locked windows; no simulation.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_walk_feedback.py` | `-m scripts.agent.split.compare_walk_feedback` | mjlab_training/.venv/bin/python | Compare saved same-checkpoint walk feedback interventions on CPU only.；保留现有能力；未据名称或无引用推定废弃 |
| `compare_walk_height_response.py` | `-m scripts.agent.split.compare_walk_height_response` | mjlab_training/.venv/bin/python | CPU-only comparison of saved walk height-command interventions.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_getup_release.py` | `-m scripts.agent.split.diagnose_getup_release` | mjlab_training/.venv/bin/python | Frozen getup release diagnosis: actual 500 Hz PD and contact, no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_getup_start.py` | `-m scripts.agent.split.diagnose_getup_start` | mjlab_training/.venv/bin/python | Record first-contact getup failures and first-active PD for a frozen checkpoint.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_walk_estimator_closedloop.py` | `-m scripts.agent.split.diagnose_walk_estimator_closedloop` | mjlab_training/.venv/bin/python | One bounded saved-trace diagnosis; frozen encoder inference, no fitting or physics.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_walk_forward_failure.py` | `-m scripts.agent.split.diagnose_walk_forward_failure` | mjlab_training/.venv/bin/python | Frozen walk-v7 diagnosis: controlled ramps, training-noise rewards and checkpoint comparison.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_walk_goal_feedback.py` | `-m scripts.agent.split.diagnose_walk_goal_feedback` | mjlab_training/.venv/bin/python | Frozen 30s stop/travel/brake comparison of encoder and sensor-wheel vx.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_walk_height.py` | `-m scripts.agent.split.diagnose_walk_height` | mjlab_training/.venv/bin/python | Compare zero/backward requests from matched states using a frozen walk policy.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_walk_v8_regression.py`（运行中原入口暂留） | `-m scripts.agent.split.diagnose_walk_v8_regression` | mjlab_training/.venv/bin/python | Compare frozen walk policies under identical fixed requests; never learn.；保留现有能力；未据名称或无引用推定废弃；运行中续训依赖链所需原文件暂留 |
| `diagnose_walk_velocity_compensation.py` | `-m scripts.agent.split.diagnose_walk_velocity_compensation` | mjlab_training/.venv/bin/python | Record frozen walk sensor packets for independent velocity-compensation trials.；保留现有能力；未据名称或无引用推定废弃 |
| `evaluate_walk_estimator_final.py` | `-m scripts.agent.split.evaluate_walk_estimator_final` | mjlab_training/.venv/bin/python | One fixed terminal encoder on two new complete runs; no training or control.；保留现有能力；未据名称或无引用推定废弃 |
| `probe_getup_torque_rewards.py` | `-m scripts.agent.split.probe_getup_torque_rewards` | mjlab_training/.venv/bin/python | Measure pre-stand torque reward size with the frozen EMA80 actor; never learn.；保留现有能力；未据名称或无引用推定废弃 |
| `record_walk_estimator_trial.py` | `-m scripts.agent.split.record_walk_estimator_trial` | mjlab_training/.venv/bin/python | Frozen parent/candidate estimator trial, with native500Hz sensors and own feedback.；保留现有能力；未据名称或无引用推定废弃 |
| `record_walk_shadow_sensors.py` | `-m scripts.agent.split.record_walk_shadow_sensors` | mjlab_training/.venv/bin/python | Record native 500Hz sensor primitives; shadow estimates never control the robot.；保留现有能力；未据名称或无引用推定废弃 |
| `render_getup_target_slew.py` | `-m scripts.agent.split.render_getup_target_slew` | mjlab_training/.venv/bin/python | Render saved baseline/candidate getup first episodes, without new simulation.；保留现有能力；未据名称或无引用推定废弃 |
| `report_walk_estimator_closedloop.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Read-only three-seed report of sealed estimator-in-control comparisons.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `report_walk_estimator_only.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Plot sealed encoder-only metric files and the optimizer journal, read-only.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `summarize_getup_release.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Audit saved getup release traces and plot the release mechanism; no simulation.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `summarize_walk_goal_feedback.py` | `-m scripts.agent.split.summarize_walk_goal_feedback` | mjlab_training/.venv/bin/python | CPU-only phased summary of saved 30s matched walk-feedback traces.；保留现有能力；未据名称或无引用推定废弃 |
| `summarize_walk_leg_swing.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | CPU-only audit of saved walk leg motion; no physics step, policy or learning.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `summarize_walk_v8_regression.py` | 已归档；恢复后使用原路径 | mjlab_training/.venv/bin/python | Read saved fixed-request traces; audit support-trigger timing and plot comparisons.；已结束实验的固定轮次检查或结果报告；在原源码树复核 |
| `train_walk_estimator_only.py` | `-m scripts.agent.split.train_walk_estimator_only` | mjlab_training/.venv/bin/python | Budgeted offline supervision of the original encoder; never runs PPO or physics.；保留现有能力；未据名称或无引用推定废弃 |
| `walk_estimator_offline.py` | `-m scripts.agent.split.walk_estimator_offline` | mjlab_training/.venv/bin/python | Locked complete-run data and per-segment scoring for encoder-only experiments.；保留现有能力；未据名称或无引用推定废弃 |
| `walk_sensor_velocity.py` | `-m scripts.agent.split.walk_sensor_velocity` | mjlab_training/.venv/bin/python | Fixed sensor-only rolling kinematics for the frozen walk velocity diagnosis.；保留现有能力；未据名称或无引用推定废弃 |

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
| `check_zero_hold_training.py` | `-m scripts.agent.isaac.check_zero_hold_training` | Isaac Python 3.8 | Check zero-command sampling and survey low balanced poses; no learning.；保留现有能力；未据名称或无引用推定废弃 |
| `compose_chuanliantui_factor_videos.py` | `-m scripts.agent.isaac.compose_chuanliantui_factor_videos` | Isaac Python 3.8 | Compose existing paired replay videos with their recorded support forces.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_isaac_ppo_rewards.py` | `-m scripts.agent.isaac.diagnose_isaac_ppo_rewards` | Isaac Python 3.8 | Replay a saved benchmark with exact per-term rewards; forbid learning.；保留现有能力；未据名称或无引用推定废弃 |
| `diagnose_zero_hold_disturbances.py` | `-m scripts.agent.isaac.diagnose_zero_hold_disturbances` | Isaac Python 3.8 | Measure actual zero-hold reward before/after training disturbances; no learning.；保留现有能力；未据名称或无引用推定废弃 |
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

