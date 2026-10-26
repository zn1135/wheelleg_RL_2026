# 新轮角39维BC本地5000次模型

2026-10-25，Codex执行、用户接收；独立复查及H7负责人核对待确认。**行为失败，不用于部署或替换旧模型。**

| 项目 | 来源与结果 |
|---|---|
| 机器人／接口 | 当前共享XML闭链串联腿；[轮角39／82／6](../interfaces/20261025-wheel-angle-observations.md)，BC仅训练actor |
| 仓库／版本 | [wheelleg_RL_2026](https://github.com/zn1135/wheelleg_RL_2026.git)，detached HEAD `2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`＋未提交修改，无分支切换、提交／推送／PR |
| 实际运行版本 | [冻结输入与源码](../../artifacts/bc-wheel-angle39-train-20261025-v1/input-manifest.json)、同目录snapshot及tracked-simulation-diff.patch；HEAD不足以代表运行代码 |
| 数据 | 全新169条；138训练／31验证，39训练姿态／9个合格验证姿态；276000／62000帧，指纹`d386dc41d5b5fb8ab08063e7e2110b3cdba2d54dfb3080b30b03f912b5ad69d1`；所有10个原留出姿态仍进入40组独立测试 |
| 初始化与参数 | 全新网络和Adam，无旧39／41权重；本地RTX4060 Laptop 8GB，seed42，batch1024，lr0.001，39→256→128→64→6 ELU，STD0.05 |
| 命令／时间／日志 | [run-local.sh](../../artifacts/bc-wheel-angle39-train-20261025-v1/run-local.sh)，2026-10-25T19:47:10+08:00～2026-10-25T19:47:26+08:00；training.log／TensorBoard事件、实际arguments.json和dataset.json位于新run |
| XML／物理 | SHA `cac8061de2850602bfd1421fc856bf217c4e5040a11c8f5d33cf1abe61d5f0cf`；2ms物理／PD、100Hz策略、2／4ms延迟及原0.2秒等待／正常趴地分布不变 |
| 模型 | [bc_5000.pt](../../logs/mjlab/chuanliantui_bc/20261025_wheel_angle39_local5000-v1/bc_5000.pt)，SHA `ef95b6dda7ff8137e95eb48f4feecef9e99d8ad2f7d86f166943ba332a70fa30`；格式`ct_bc_actor39_pd6_wheel_angle_v1`，新增／累计5000；包含BC Adam和RNG，无可用PPO critic |
| 预算／拟合 | 通过；Adam及日志均5000，初始化None，固定验证缩放MSE0.0110574244，[正式检查](../../artifacts/bc-wheel-angle39-train-20261025-v1/assessment.json) |
| 数值／独立行为 | 动作及两轮角输入复算差0，初态逐项一致、首0.2秒实际零力；起立2／平衡0／停车0／完整20秒0（均/40），**行为失败**，[对照](../../artifacts/bc-wheel-angle39-train-20261025-v1/comparison.json) |
| 视频／最终来源 | [20秒失败对照](../../artifacts/bc-wheel-angle39-train-20261025-v1/evaluation/failed-comparison.mp4)，提前终止明确定格；[核对与环境](../../artifacts/bc-wheel-angle39-train-20261025-v1/verification.json)、同目录runtime-local.json和evaluation/env.yaml |
| SMP／PPO／部署 | prior10000及旧模型、XML不变；本轮未训练PPO／SMP；新39未适配PPO／CPU／H7，无ONNX、C网络、固件或烧录，H7负责人待确认 |

Isaac不适用；移动、全姿态、随机化、10ms压力、CPU sim2sim、ONNX、板端数值／时序及真机均未执行。不能只因都是39维就与原模型混用；旧模型按其原接口／源码组合回退。原843姿态的教师不合格样本没有加入训练，仍保留在独立测试。新旧实验接口、数据及预算不同，不作单因素归因。[完整报告](../../artifacts/bc-wheel-angle39-train-20261025-v1/README.md)。
