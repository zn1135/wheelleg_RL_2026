# mjlab 参考工程控制对齐诊断

后续状态：GPU闭链偏差已修复并通过原5 mm门槛，见 [闭链修复记录](20261009-mjlab-closure-fix.md)。以下保留修复前的检查结果。

2026-10-09，分支26_wheelleg，基础HEAD 58947e8fabc4683b2025d42bda742b06c84dc566 加既有未提交修改与本次r5。作者为本次助手，独立复查待执行。[接口定义](../interfaces/chuanliantui-mjlab-huananhu-v14-flat-standup-r5.md)。

## 修改范围

按用户选中的三项对齐：腿PD60/2、四腿MJCF ref和2 ms×10物理步进，策略仍50 Hz。名义PD参数集中于config.py，CPU、GPU和训练随机化使用同一基值。腿目标限幅相对于ref，确定性及随机姿态缓存重新生成，实体几何、根初始位置和姿态保持原定义。旧r4 checkpoint经manifest拒绝，不自动转换模型。

参考文件SHA、完整版本、工作区状态和当前manifest见 [诊断JSON](20261009-mjlab-reference-control.json)。wheelleg_RL及其环境未修改。电机armature对照只在临时环境内进行，没有写入本仓资产。

## 实际执行

均在本训练仓根目录，用独立Python 3.12 mjlab解释器。没有PPO learn()、训练run、新策略或TensorBoard。

```bash
mjlab_training/.venv/bin/python scripts/agent/generate_chuanliantui_mjlab_pose.py
mjlab_training/.venv/bin/python scripts/agent/import_chuanliantui_mjlab_reset_bank.py
mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_mjlab.py --out /tmp/chuanliantui-mjlab-r5-cpu-check.json
mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_mjlab.py --gpu --out /tmp/chuanliantui-mjlab-r5-final-check.json
git diff --check
```

| 检查 | 退出码／结果 |
|---|---|
| 确定性姿态求解及随机姿态表导入 | 0；每模式1024行；最大闭合误差5.4383e-8 m、参考端点差1.0966e-13 m |
| CPU有限数学／物理／接口检查 | 0；Kp60、Kd2、ref相对限幅、50 Hz、r4版本拒绝、26项源奖励数值对照通过；动态最大销轴误差4.156 mm |
| 临时随机actor ONNX数值对照 | 最大误差1.0430813e-07；不是交付模型，不证明策略效果 |
| 最终GPU有限随机初态检查 | 1；4环境、64策略步，动态最大闭链误差6.181 mm超过原5 mm阈值；此前状态、观测、奖励有限，无数值重置。失败发生在销轴阈值断言，后续GPU局部reset检查未执行 |
| git diff --check | 0 |

CPU证据见 [检查JSON](20261009-mjlab-reference-control-cpu-check.json)，GPU证据见 [失败日志](20261009-mjlab-reference-control-gpu-check.txt)。GPU失败时该检查器未写出--out JSON，因此另外记录已知失败值，不伪造完整通过报告。

## 闭链偏差检查

首轮GPU检查使用6通道Kp倍率采样，最大误差6.464 mm。恢复原有四腿Kp采样次数后，最终误差6.181 mm，仍未通过5 mm阈值；未放宽检查标准。

临时对照使用seed7、4环境、64策略步，零请求动作：原电机armature最大误差6.444 mm；临时设为参考0.015795 kg·m²后仍有6.262 mm。两个最大值出现在第13拍（约0.26 s）后连杆闭合销轴，两组无数值重置。该对照表明仅同步惯量不能消除超限，不据此改变本仓默认动力学参数或宣称已定位根因。

配置对齐已完成，有限GPU闭链门槛尚未通过。正式训练、r5策略回放、持续站立、完整随机恢复、板端与真机验收均未执行；旧r4或参考工程的模型效果不能视作r5结果。下一步需要检查随机落地及启动PD瞬态下的约束响应。
