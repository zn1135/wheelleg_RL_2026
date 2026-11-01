# SuperMimic 原始动作统一限幅

**后续决定（2026-11-01）**：用户要求先按V13动作分布与失败时序选范围，不能照搬参考数字。因此以下±3实现仅为实验候选，不是已验证、已采纳的训练范围；代码当前仍是该候选值，本轮不修改成另一个未经验证的数字，也不启动训练。新增10模型/固定seed43各96例的只读统计见[动作与失败时序报告](../../artifacts/supermimic-v13-action-failure-timing-20261101-r1/README.md)。

**随后授权的300次试训已完成**：SM615完整actor/估速器继承，世界/价值/四优化器冷启，世界4000次预热后策略300次，总W5200。初始96例存活/平衡均96，最终存活96、平衡0；方向/停车退化，±3仍未验证为合格训练范围。此结论不能单独归因于限幅，因为结构/坐标/初始化等适配同时存在。[试训结果与正常速度视频](../../artifacts/supermimic-v13-v5-sm615-a300-20261101-r1/README.md)。

2026-11-01，按用户要求，把新 SuperMimic 配方的六维原始动作统一限制到 `[-3,3]`。未启动训练。当前算法合同为 `chuanliantui-supermimic-control-endpoint-state-action-clamp-bptt-v13-v5`。

## 依据及作用位置

依据是本地参考仓库 `/home/zn/文档/supermimic_wheelbipe/supermimic_wheelbipe/constants.py:19` 的 `ACTION_CLIP=3.0`，以及 `train.py:256` 的想象限幅、`train.py:413` 起的“原始策略输出＋探索噪声→限幅”采集顺序。这是参考实现的工程配方，不声称原论文规定所有机器人必须取3，也未证明它能修复当前失稳。

| 路径 | 本次实现 |
|---|---|
| 新模型真实采集 | `__main__.collect`先取得未限幅输出，加原有探索噪声，检查有限值，再限幅；同一动作传仿真并保存回放 |
| 新模型想象/评估 | `networks.act`输出限幅动作；世界模型、奖励、下一拍观测中的动作一致 |
| 新模型ONNX | `export.ExportPolicy`图内限制动作输出；125维输入、6维动作、3维速度及速度缩放保持 |
| 历史模型原样推理/导出 | 按旧算法合同保留无限幅语义；导出元数据`action_clip=null` |
| 旧actor初始化新训练 | 只继承权重，启用新±3规则；初始化记录来源/目标范围，不声称旧行为相同 |
| 完整保存恢复 | 新合同及配方共同校验；拒绝把旧完整模型直接当成新配方续训；拒绝把旧推理actor保存成新配方 |

限幅在SuperMimic策略侧完成，不修改共享PPO/BC/SMP网络、XML、执行器、真实动作延迟、奖励或学习率。六动作顺序仍为左腿两关节/左轮/右腿两关节/右轮。按现有缩放，腿位置偏移范围成为±1.5rad（随后仍经过原目标位置限制），轮速目标成为±30rad/s；这不是保证实际轮速处于±30rad/s。

NaN/Inf不被转换成有限动作，采集在送入仿真前停止。有限输出在范围内梯度不变，超范围部分梯度为0；旧策略可能频繁饱和，限幅有效不等于行为改善。

## 检查记录

- [保存恢复检查](../../artifacts/supermimic-v13-action-clamp-restore-20261101-r2/report.json)：旧完整合同拒绝、错形状拒绝、新完整包往返、旧actor继承/推理权重保持；优化器0。
- [动作路径与ONNX检查](../../artifacts/supermimic-v13-action-clamp-check-20261101-r1/report.json)：强制饱和输出、范围内/外梯度、非有限保留、想象动作及观测一致；新ONNX动作最大误差0，估速最大误差4.77e-7，旧ONNX动作最大误差2.75e-6以内。
- 同一检查的GPU集成：12环境各1个控制步，12条真实转移、60个物理子步。用固定正负噪声区分“先限幅再加噪声”和正确顺序；执行/回放/奖励记录/下一拍观测一致。Inf输出在仿真调用前拒绝；四优化器均0次更新。
- 初次恢复检查r1误选了与当前同形状的v4包，导致“伪装新合同后必须因形状不同被拒绝”的检查假设不成立；记录保留。r2使用原A5200异形状包后通过，未因此放松生产校验。

复现命令（输出目录必须是新的）：

```bash
PYTHONPATH=mjlab_training/src:. mjlab_training/.venv/bin/python -m scripts.agent.supermimic.check_supermimic_no_history --output artifacts/supermimic-action-restore-NEW
PYTHONPATH=mjlab_training/src:. mjlab_training/.venv/bin/python -m scripts.agent.supermimic.check_supermimic_action_clamp --checkpoint artifacts/supermimic-action-restore-NEW/zero-update-roundtrip.pt --legacy artifacts/supermimic-v13-imagination-clamp-integration-20261101-r1/zero-update-roundtrip.pt --output artifacts/supermimic-action-check-NEW --integration
```

检查生成的饱和权重和ONNX仅为测试夹具，不能部署。未交付训练模型；真实96例行为、多种子及真机未验证。H7负责人核对：待确认，固件未修改。

下一步优先固定同一策略权重，对比加/不加限幅的96例表现及动作饱和比例；只有存活/双轮平衡保住且速度跟踪改善，才说明限制大动作对控制有帮助。此建议不授权自动启动评估或训练。
