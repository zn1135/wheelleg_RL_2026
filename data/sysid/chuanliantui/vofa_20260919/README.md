# VOFA 原始数据准入结果

本目录由 `scripts/agent/analyze_chuanliantui_vofa_sysid.py` 生成；不替代或改写仓库根目录的原始 CSV。

## 髋/腿

- 原始帧：17544；按 200 Hz 重建时长 87.720 s。
- 全在线帧：17179；左右参考不同帧：710（斜坡从各侧上一帧实测角起算时允许出现）。
- `hip_trajectory_replay.csv`：连续全在线参考轨迹 14070 帧 / 70.350 s（源帧 3189–17258）；用于 MuJoCo 以同一控制器逐样本回放。
- 检出的稳定目标段：5；通过声明的 ±10.0 N·m 窗口限幅检查：5。
- `hip_quasistatic_windows.csv` 每行是一个稳定段末尾 1.5 s 的均值，只用于准静态对齐。
- 本帧无硬件时间戳；初次回放按固定 200 Hz 对齐，时延只可作为待优化的离散样本偏移。

## 轮

- 原始帧：15451；CSV 有 32 列，但轮测试只使用前 10 列 `I0…I9`。
- 识别到轮测试有效帧：14531；时间戳范围 91.166–114.832 s。
- 非零电流平台片段：11；通过稳态窗口检查：10；覆盖命令档位：10。
- 判定：accepted_for_steady_state_wheel_fit_with_frame_gaps。
- 只使用前 10 列轮测试载荷；额外列不参与轮侧辨识。+3 A 平台被 22 帧非轮测试数据切断，使用切断后的完整尾窗；电流平台末尾 0.3 s 的均值已写入 wheel_plateau_windows.csv。

## 本批可交付的辨识量

- `wheel_equivalent_fits.csv`：4 侧向/转向的稳态等效电流模型，形式为 `I = I_c + k_v·|ω|`。它是观测拟合，尚未按电流常数、总传动比及效率换算为轮端力矩，不能直接填 MuJoCo。
- `hip_trajectory_replay.csv`：完整动态参考、实测角、下发力矩与腿任务坐标；这是 real2sim 对齐输入，不是仅供静态统计的数据。
- `hip_quasistatic_tracking.csv`：稳定段摘要，用于辅助检查零位/几何/重力或气弹簧偏置。

## 后续边界

应在 MuJoCo 复刻固件的控制律后，以 `hip_trajectory_replay.csv` 的参考轨迹驱动模型，将模拟实测角与真机实测角逐点对齐，再拟合关节 `armature`、`damping`、`frictionloss`。下发力矩用于核对控制器输出；不可单独当作真实力矩标定。
