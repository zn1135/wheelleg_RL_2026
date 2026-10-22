# 悬空轮端候选选择与联合优化（2026-10-21）

**本轮选用joint_sharp作为后续悬空轮端回放候选。**现有留出会话轮速RMSE由上轮最佳3.0654降到2.7792rad/s，MAE1.6077降到1.4476rad/s。该选择以总体轮速RMSE为准，存在局部减速退化；不声称每一段都更好。选择记录为[selection.json](../../artifacts/h7-wheel-best-20261021/selection.json)，完整独立候选为[selected/](../../artifacts/h7-wheel-best-20261021/selected/README.md)。

只研究固定机身、双轮悬空的加速/滑行，不对照旧RL俯仰。正式XML/H7、物理标定、零件质量/重心/刚体惯量不改，不同步环境、不训练、不连接硬件、不提交/推送。仍是指令到响应的等效参数，不是实测轴端力矩/真实惯量摩擦。

## 本次怎样提高拟合度

沿用上轮驱动函数、同一轮轴armature和原记录输入，整个脉冲自由积分、不重设撤矩速度。原模型分先拟合滑行损耗、再拟合驱动两阶段；本次将阻尼、干摩擦与驱动G/V/p一起优化，让其协调拟合加速与减速。

先确定两个实验，均每轮200次实际MuJoCo目标函数调用、相同两起点、相同相位等权损失：joint保持原p=[0.5,4]；joint_sharp把p下界放到0.1，其余边界不变。上轮p到0.5下界，故探索更陡的平台形状；不是按验证结果逐次扩界。惯量仍固定，保持同一套驱动/滑行物理参数。不能将每段换惯量或贴实测状态当作改善。

原拟合集fit-complete和留出validation-complete均整会话不变，各4脉冲。所有新参数在载入本次留出数据前冻结；但是该会话先前已看过且本次用于选择模型，**本次分数是重复使用数据的探索/选择结果，不是新增盲验证**。源数据/固件/模型来源见[完整输入报告](20261021-fixed300-complete-input-calibration.md)。

| 候选 | MAE rad/s | RMSE rad/s | 0.5ms复算RMSE |
| --- | --- | --- | --- |
| previous best | 1.6077 | 3.0654 | 3.0680 |
| joint same bounds | 1.5703 | 2.9338 | 2.9374 |
| joint sharper response | 1.4476 | 2.7792 | 2.8098 |

默认候选从通过有限检查且0.5ms复算RMSE变化不超过2%的候选中选总体RMSE最低者。joint_sharp步长细化后误差约变化1.10%，排名保持；G/V各±1%的RMSE范围2.6629～2.9689rad/s，仅为本会话有限敏感性。p仍接近0.1下界，不能据此宣称最优或跨工况稳健。

## 左右、正反及相位的变化

MAE单位rad/s，旧=上轮coast_losses，新=本轮选用候选；正负为固件指令坐标。

| 轮/方向 | 加速旧 | 加速新 | 减速旧 | 减速新 |
| --- | --- | --- | --- | --- |
| left -1 | 2.2097 | 1.0609 | 4.7995 | 4.4027 |
| right +1 | 2.0664 | 0.7318 | 0.3961 | 0.3044 |
| right -1 | 1.9903 | 0.8111 | 0.4760 | 0.6081 |
| left +1 | 1.9303 | 1.4022 | 0.5495 | 0.6871 |

四方向加速均改善；左负向、右正向减速改善。右负向减速0.4760→0.6081、左正向0.5495→0.6871恶化，完整保留。左轮负向滑行仍贡献约92.1%的总平方误差，是下一步优先处理的残差。

![当前选择与原最佳](../../artifacts/h7-wheel-best-20261021/selected-comparison.png)

左右/方向/速度区间/相位全量指标、边沿及逐点数据见[study/bench.json](../../artifacts/h7-wheel-best-20261021/study/bench.json)和predictions.csv；评分沿用原相位内部掩码，边沿/初态完整曲线另保存，不用平滑或删点降低误差。

## 选择怎样生效

选用清单保存独立XML、完整精度参数和来源哈希；回放入口replay_selected.py读取selection.json，安装同一原生增益回调，再从实测初态连续积分现有留出会话。不重新拟合、不跑策略或RL、不接硬件。XML需要该回调，不能单独当作完整电机模型使用。

```bash
mjlab_training/.venv/bin/python artifacts/h7-wheel-best-20261021/replay_selected.py --out "artifacts/h7-wheel-replay-$(date +%Y%m%d-%H%M%S)"
```

重新执行本次优化和有限检查（使用新输出目录）：

```bash
OUT="artifacts/h7-wheel-joint-repeat-$(date +%Y%m%d-%H%M%S)"
mjlab_training/.venv/bin/python artifacts/h7-wheel-best-20261021/fit_joint.py --out "$OUT" --budget 200
mjlab_training/.venv/bin/python scripts/agent/check_h7_drive_response.py --out "$OUT"
```

参数完整精度/每次目标函数轨迹/边界见[study/parameters.json](../../artifacts/h7-wheel-best-20261021/study/parameters.json)。既有正式有限检查已重算23192个验证点差0、重复逐位一致，保护固件/正式模型/部件物理字段/非轮执行器不变，见[study/verification.json](../../artifacts/h7-wheel-best-20261021/study/verification.json)。selected-replay/result.json另验证默认选择入口与冻结评分一致。版本/来源/数据/产物哈希见[交付清单](../../artifacts/h7-wheel-best-20261021/delivery-manifest.json)。

## 还有什么办法

当前模型每轮正反共用同一损耗；下一项值得检验的是**让正反方向的等效损耗不同**，优先检查左负向滑行。它可能减小主要残差，但尚未实现/验证，不能直接判定为齿隙、真实摩擦或装配问题。应先补一份新独立正反滑行会话确认方向差异重复出现，再拟合和独立验证；继续复用同一留出会话追分会高估效果。

若还要提高最初约0.1秒的加速形状，可考虑驱动内部动态，而不是只依赖当前速度的静态函数；需不同幅值和短时长记录来约束，当前仅±3Nm/300ms不足以确认。现阶段先保留本次2.78rad/s候选作为轮端回放默认，不改正式资产或训练环境。
