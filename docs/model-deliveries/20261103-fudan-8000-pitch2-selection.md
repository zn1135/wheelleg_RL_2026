# 模型选用记录：复旦8000-v2＋停车+2°

后续实际复测已完成：[零延迟/4+4ms九停车对照](../../artifacts/fudan_ppo_walk/fudan-parking-delay-replay-20261103/README.md)，末段均9/9，但4+4ms高.26后退起步新增触地约591N/26°，高.30一例腿RMS增大5.7倍；不认定全面无退化，待用户完整视频验收。下文“未动态验证”保留其准备阶段历史含义。

后续sim2sim条件已按用户选择改为固定4ms动作＋4ms传感延迟，训练和模型不变。[独立延迟记录](../interfaces/20261103-fudan-sim2sim-delays.md)中的151项检查仅为无物理验证，新条件尚未动态验收；本页既有九停车成绩仍属零额外延迟。

| 项目 | 内容 |
|---|---|
| 日期、选用人、实现、独立复查 | 2026-11-03；用户明确“用这个”；Codex；H7负责人待确认 |
| 机器人与用途 | `fudan_ppo_walk`实体闭链机器人；当前GPU仿真停车方案，不是新训练模型 |
| 接口 | [选用推理配方](../interfaces/20261103-fudan-selected-pitch2.md)，25/125/6、reset/history v2 |
| 仓库、分支、HEAD | `zn1135/wheelleg_RL_2026`，`26_wheelleg`，`c3c8fdffe0739aed314fb71c69f2a2d3135b3fb9`；有未提交改动，无提交/推送/PR |
| 工作区差异与未跟踪源码 | [差异](../../artifacts/fudan_ppo_walk/fudan-selected-8000-pitch2-20261103/selected-runtime.diff)、[执行源与校验](../../artifacts/fudan_ppo_walk/fudan-selected-8000-pitch2-20261103/source-sha256.json)，不含其他人的文档修改 |
| 训练来源 | 原7000→8000，4096环境、seed1；[原训练命令/预算](../../artifacts/fudan_ppo_walk/fudan-parking-resume7000-to8000-20261102/RUNNING.md)与[完成审计](../../artifacts/fudan_ppo_walk/fudan-parking-resume7000-to8000-20261102/completion-audit.json) |
| 运行环境 | 新行为证据为MuJoCo Warp GPU；原模型环境/版本和物理快照沿原记录；本次未新训练/物理运行 |

| 产物 | 地址 | SHA256/说明 |
|---|---|---|
| 原完整PT | [model_final.pt](../../logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000/model_final.pt) | `39fa104d176f82a33db8317a75b4db28749362daf6bc4d2e06cb83d2c5ae32c9`，含Actor/Encoder；保持原位 |
| 图外选用配方 | [selected-policy.json](../../artifacts/fudan_ppo_walk/fudan-selected-8000-pitch2-20261103/selected-policy.json) | +2°停车重力参考、2s初始等待、0.5s斜坡、积分关闭 |
| 新ONNX/固件 | 未生成 | 原8000 ONNX只含原网络，不含+2°；不能仅换模型就认为板端具有此行为 |

| 阶段 | 状态 | 依据 |
|---|---|---|
| 原8000训练 | 完成 | 原审计与日志；本次无追加训练 |
| 当前输入与入口检查 | 通过 | [26项检查](../../artifacts/fudan_ppo_walk/fudan-selected-8000-pitch2-20261103/checks.json)，无物理/NN/更新 |
| GPU停车 | 九例数值达标、用户选用 | [三高度原地及前后停车](../../artifacts/fudan_ppo_walk/fudan-pitch-reference-small-20261103/README.md)，8000+2°9/9，低高度腿部运动增加 |
| 原45行走回归/正式987 | 未执行 | 新选用方案尚未完成该覆盖，不能称完整运动能力通过 |
| 起立能力 | 未验证 | 本任务从站姿开始的停车试验不等于起立验收 |
| 新ONNX/板端数值和时序/真机 | 未执行 | 本次未授权部署；H7负责人待确认 |

选用范围为本仓默认GPU评估和已录制轨迹回放。训练入口、原PT/ONNX及H7均未改行为；后续部署需成套实现图外预处理并单独验证。9000轮严重抖腿失败结论保持。回退方式是显式指定原检查点，恢复原输入评估。
