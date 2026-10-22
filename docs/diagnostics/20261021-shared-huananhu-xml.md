# 华南虎训练与仿真共用一份XML（2026-10-21）

后续状态：作者随后明确要求写入最优拟合响应，现已完成共享XML及训练/CPU执行器接入；当前状态见[接入报告](20261021-h7-wheel-response-integration.md)。下文保持“仅合并来源”当时的验证快照。

当前唯一可编辑模型源为`resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml`。训练`model.py:MODEL_XML`、CPU sim2sim的MODEL_ASSET及只读real2sim的MODEL均指向它。旧`multimode_model3023/chuanliantui.xml`改为相对符号链接`../mjlab/chuanliantui.xml`，不再维护独立副本；历史路径可用。

源XML的meshdir从meshes改为../meshes，指向同一个真实网格目录，并让通过新旧路径加载都有效。调整后的源文件字节等于原multimode XML，旧路径SHA256仍为1ea213314d8fd686841678d28246fbb588d53a5586e08c22ea977e95c0cb4ec5，因此既有录制的旧模型哈希检查不受本次合并影响。训练源的文本哈希因网格路径改写而改变，使用原始XML哈希的缓存可能需要重建；未改写姿态库。

CPU的POSE_BANK仍显式保留在multimode_model3023/huananhu_pose_bank.npz，没有因XML路径调整而切换到训练侧姿态库。模型权重、网格和历史模型目录的其他文件均保留，不合并策略或姿态库。

合并前后30项编译字段（质量、惯量、几何/网格、关节、轮端参数、执行器及约束）逐项完全一致。正式有限检查check_multimode_cpu_sim2sim.py通过，覆盖单一来源/兼容链接/姿态库路径、训练与CPU初态、观测/控制公式/限幅、弹簧、接触选项及恢复高度检查。仅表示这些有限检查，不是新策略行为验收。证据见[验证记录](../../artifacts/shared-huananhu-xml-20261021/verification.json)与[CPU结果](../../artifacts/shared-huananhu-xml-20261021/check-cpu.json)。

```bash
mjlab_training/.venv/bin/python scripts/agent/check_multimode_cpu_sim2sim.py --out artifacts/shared-huananhu-xml-20261021/recheck.json
```

本轮是用户最新“合成一份”的来源统一，物理参数保持不变；此前选定的joint_sharp仍独立保存在artifacts/h7-wheel-best-20261021/selected/，尚未写入该共享XML或训练执行器。不能将来源统一称为最优驱动响应已同步或重训已使用新模型。没有启动训练、同步到原工作区、操作硬件、提交或推送。

工作位于独立worktree；合并前文件字节与哈希保存在artifacts/shared-huananhu-xml-20261021/before/和before-manifest.json，便于复核。本次不删除任何模型权重、历史录制、网格或姿态库。
