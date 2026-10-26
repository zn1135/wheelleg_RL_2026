# 新轮角39维BC：本地纠偏新增5000次

2026-10-25，Codex执行、用户接收；独立复查／H7负责人核对待确认。**行为退化拒绝采用，无部署或替换旧模型。**

| 项目 | 实际记录 |
|---|---|
| 机器人／用途 | 共享闭链串联腿，正常起立、静站和平滑停车；仅BC actor，无移动／倒置恢复 |
| 接口 | [轮角39／82／6](../interfaces/20261025-wheel-angle-observations.md)，`wheel-angle-39-82-v1`；PPO／CPU／H7仍未适配新接口 |
| 仓库／版本 | wheelleg_RL_2026，detached HEAD `2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`＋原未提交改动；未提交／推送／PR |
| 运行源码与数据 | [冻结清单](../../artifacts/bc-wheel-angle39-correction-train-20261025-v1/input-manifest.json)、snapshot及tracked-simulation-diff.patch；HEAD不代表实际运行内容 |
| XML与物理 | 原共享XML SHA `cac8061de2850602bfd1421fc856bf217c4e5040a11c8f5d33cf1abe61d5f0cf`；500Hz物理／PD、100Hz策略、2／4ms延迟、0.2秒零主动力不变 |
| 输入模型与优化器 | 上一轮轮角39的bc_5000.pt，SHA `ef95b6dda7ff8137e95eb48f4feecef9e99d8ad2f7d86f166943ba332a70fa30`，累计5000；只加载actor，新Adam，未恢复优化器 |
| 数据 | 原138完整训练／31验证＋34800新实际教师纠偏帧；训练310800／验证62000帧，指纹`fc8cf939a7ff36fbcd96ba9351a91f404ec99d45919e19efbf59efe5efbf35e9`；无留出动作训练 |
| 命令／时间／设备 | [run-local.sh](../../artifacts/bc-wheel-angle39-correction-train-20261025-v1/run-local.sh)，2026-10-25T20:05:21+08:00～2026-10-25T20:05:37+08:00，本地RTX4060 Laptop 8GB；seed42、batch1024、lr.001；软件版本见runtime-local.json |
| 预算与网络 | 新增5000／累计10000，日志与Adam均5000；39→256→128→64→6 ELU、STD0.05；本轮预算用完 |
| 模型 | [bc_5000.pt](../../logs/mjlab/chuanliantui_bc/20261025_wheel_angle39_correction_local5000-v1/bc_5000.pt)，SHA `67e45a618e4cb04fde47701bda178e2c93e41eb83a75decbdc0a03a60ead387e`；BC actor／Adam／RNG／来源，无可用PPO critic |
| 数值检查 | 通过：原40完整初态一致、轮角及实际动作复算差0、启动实际零力；固定旧尺度验证MSE0.01105742→0.00498190；不等于行为通过 |
| 独立行为 | 失败：起立2/40→0/40，平衡／停车仍0/40，1.95～5.42秒终止；原env31、36起立丢失，退化拒绝采用 |
| 诊断边界 | 教师先站起再交接亦平衡0/40；混合控制非独立验收。隐藏状态及局部反馈贡献未分离，未定位唯一代码bug |
| 视频与结果 | [失败视频](../../artifacts/bc-wheel-angle39-correction-train-20261025-v1/evaluation/failed-comparison.mp4)、[完整报告](../../artifacts/bc-wheel-angle39-correction-train-20261025-v1/README.md)、同目录verification／assessment／comparison／video哈希 |
| SMP／PPO／部署 | 本轮SMP／PPO更新0、prior10000和旧模型保持；无ONNX／C网络／固件或烧录，H7未修改，负责人待确认 |

Isaac不适用；移动、全1024初态、训练随机化／噪声、推扰、10ms压力、独立CPU sim2sim、导出、板端与真机均未执行。当前模型只用于失败复现；回退需使用原接口与对应模型源码，不能仅因同为39维混用。
