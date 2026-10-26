# 旧技能保留：轮角39维本地新增5000次

2026-10-25，Codex执行、用户接收；独立成员复查及H7负责人核对待确认。最终5000次版本退化拒绝采用；新增4500次版本保住原40组平衡及平滑，停车0/40；仅记为阶段候选，不自动导出、部署或接PPO。

| 项目 | 实际记录 |
|---|---|
| 用途／接口 | 正常起立、静站、零速停车；轮角39／82／6，`ct_bc_actor39_pd6_wheel_angle_v1`；默认PPO／CPU／H7未适配新轮角语义 |
| 仓库／版本 | detached HEAD `2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`＋既有未提交改动；无提交／推送／PR；实际内容以input-manifest、snapshot、tracked-simulation-diff.patch为准 |
| 初始化 | 旧稳定累计15000模型SHA `c29ab3ecb16368114dc34375dec54fbabef4c581eeae955f561687368dcf6e83`，第一层14、15两列置零，其他参数保留；新Adam |
| 数据／训练 | 原138训练／31验证示范；新增46条实际制动／92000帧、31200起立保留帧、1656反馈差分对，仅训练姿态；指纹`e28195848a4f4bfc85de110d217299af93b01170c1c11c6061bb136d5b9fa121` |
| 时间／设备／预算 | 2026-10-25T20:53:55+08:00～2026-10-25T20:54:29+08:00，本地RTX4060 Laptop 8GB，lr0.0001／seed42／batch1024＋256＋256；新增5000，最终累计20000，预算用完 |
| 物理／控制 | 原XML SHA `cac8061de2850602bfd1421fc856bf217c4e5040a11c8f5d33cf1abe61d5f0cf`；2ms物理／PD、100Hz策略、2／4ms延迟、0.2秒真实零主动力；无参数或H7改动 |
| 最终模型 | [bc_5000.pt](../../logs/mjlab/chuanliantui_bc/20261025_preserved_wheel39_local5000-v1/bc_5000.pt)，SHA `8233aca4d292e396103c03d4623c7109b02c1de487ec4849b958593669414dd2`，起立40／平衡29／停车0，不采用 |
| 数值验收 | Adam及日志5000次、累计计数正确，原40完整初态一致，动作及轮角复算差0，保留数据无留出，模型／输入／原XML／先验未变；不能替代行为验收 |
| 中间模型 | 已保存10个模型全部零更新回放；原40组用于选择，不能再称新泛化测试。保留阶段候选[bc_4500.pt](../../logs/mjlab/chuanliantui_bc/20261025_preserved_wheel39_local5000-v1/bc_4500.pt)，累计19500次，SHA `c5acb3029f857888275a5f9e0c40a895b1dbb4a2ca3a5019cdea11dfe495f33e`；末5秒速度均值0.03123m/s、最差0.03358m/s。它是本轮已保存模型，没有额外训练。候选[完整20秒视频](../../artifacts/bc-preserved-wheel39-train-20261025-v1/video-selected/failed-comparison.mp4)中‘failed’指未过完整停车验收，起立／平衡通过；[候选反馈复核](../../artifacts/bc-preserved-wheel39-train-20261025-v1/selected-feedback.json)。旧稳定模型仍保留。 |
| 部署边界 | 仅BC actor／Adam／RNG／来源，没有PPO critic；无SMP／PPO更新，无ONNX／C网络／固件／烧录，H7负责人待确认 |

完整[报告、视频和核对](../../artifacts/bc-preserved-wheel39-train-20261025-v1/README.md)。未做移动、全姿态、训练随机化／噪声、物理推扰、10ms压力、CPU sim2sim或真机。保留项是软约束，有限输入上的反馈匹配不能承诺独立闭环不退化；本轮不以速度改善替代停车通过。
