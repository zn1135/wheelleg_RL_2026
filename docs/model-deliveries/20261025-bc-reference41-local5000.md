# BC共享参考41维：本地新增5000次模型

2026-10-25，Codex执行，用户本人接收；独立复查、H7负责人核对待确认。正常趴地起立、平滑和零速停车；**行为退化，不替换旧模型，不交付部署。**

| 项目 | 内容 |
|---|---|
| 接口／机器人 | 当前共享闭链串联腿，[41维actor／84维critic／6维动作](../interfaces/20261025-bc-shared-reference.md)，BC仅训练actor |
| 训练仓 | [wheelleg_RL_2026](https://github.com/zn1135/wheelleg_RL_2026.git)，detached HEAD `2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`＋既有未提交修改，未提交／推送／PR |
| 实际版本 | [冻结源码、输入](../../artifacts/bc-reference41-train-20261025-v1/input-manifest.json)、[已跟踪差异](../../artifacts/bc-reference41-train-20261025-v1/tracked-simulation-diff.patch)、[正式产物核对](../../artifacts/bc-reference41-train-20261025-v1/verification.json)；未跟踪运行源码在同目录snapshot，HEAD不能替代脏工作区 |
| 初始化 | 旧累计15000次39维模型SHA `c29ab3ecb16368114dc34375dec54fbabef4c581eeae955f561687368dcf6e83`，保留全部已学参数，首层新两列0；新Adam，不恢复旧优化器 |
| 训练 | 本地RTX4060 Laptop 8GB，seed42，batch1024，lr0.001，5000新增／20000累计，41→256→128→64→6 ELU，STD0.05不更新 |
| 实际命令、日志、时间 | [run-local.sh](../../artifacts/bc-reference41-train-20261025-v1/run-local.sh)、[training.log](../../artifacts/bc-reference41-train-20261025-v1/training.log)，2026-10-25T18:47:15+08:00～2026-10-25T18:47:35+08:00 |
| 数据 | 169条起立＋201279帧旁路纠偏＋82／120条合格接管；训练877279帧、验证66000帧；指纹`0ad55b06875635aee18a29ef1d0fbd6239c9a33360cb4505cb4b060e2e4a54a6`，原39／10姿态组隔离 |
| XML／控制 | XML SHA `cac8061de2850602bfd1421fc856bf217c4e5040a11c8f5d33cf1abe61d5f0cf`；2ms物理／PD、100Hz策略、2／4ms延迟、原正常趴地分布和0.2秒等待不变 |
| 环境版本与配置 | [runtime-local.json](../../artifacts/bc-reference41-train-20261025-v1/runtime-local.json)，模型run中的arguments.json、dataset.json和snapshot；独立回放env.yaml另存evaluation |
| 部署端 | 本仓内置H7保持39维，本轮未修改、导出／烧录或同步新网络，负责人核对待确认 |

[bc_5000.pt](../../logs/mjlab/chuanliantui_bc/20261025_reference41_local5000-v1/bc_5000.pt) SHA-256：`b9ba3f672ac59b59bc2762fc23fb58cc2f3be0ef09579d5ef009092009064549`。格式`ct_bc_actor41_pd6_reference_v1`，包含actor／BC Adam／RNG／训练尺度及来源，不包含可用PPO critic。无ONNX、C网络或固件产物。

| 阶段 | 状态与证据 |
|---|---|
| 参考时序／数据／初始化 | 通过；同拍参考递推、真实动作归属、来源快照、首0.2秒实际零力、部分复位和无教师执行；补零后的独立起立／平衡40/40 |
| 预算／拟合 | 通过；Adam与日志均5000，累计20000，固定验证缩放MSE0.0030526144；[正式检查](../../artifacts/bc-reference41-train-20261025-v1/assessment.json) |
| 独立mjlab行为 | **失败**；完整20秒21/40、起立19/40、最终平衡15/40、综合停车0/40，原起立丢失21、原平衡丢失25；[逐样本对照](../../artifacts/bc-reference41-train-20261025-v1/comparison.json) |
| 动作／观测数值 | 通过；实际学生动作、轮式里程、共享位置误差复算差0；过渡期位移差分与瞬时速度最大差0.18848085m/s，未称全部速度低误差 |
| 视频 | 保存提前失败及最终平衡未过两回合，各20秒60fps；提前终止明确定格，[选择与哈希](../../artifacts/bc-reference41-train-20261025-v1/evaluation/video.json) |
| Isaac／PPO／SMP | Isaac不适用；PPO／SMP未执行，无额外预算 |
| CPU sim2sim、移动、随机化、10ms、导出、板端、真机 | 未执行；本轮仅名义零速BC，41维尚未适配部署 |

旧39模型和历史数据保持不变，可按其冻结源码／数据组合回退。新增参考和示范组成同时变化，尚未定位训练后退化根因；不能把这轮结果当位移处理错误或观测有效性的单因素结论。只允许实验分析，当前不进入PPO。[完整报告、命令和视频](../../artifacts/bc-reference41-train-20261025-v1/README.md)。
