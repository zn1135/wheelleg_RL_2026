# 复查索引与遗留问题

2026-10-22 清理：移除已修复过程、过时参数、旧模型短时回放和重复验证数据。旧 Isaac 课程及奖励建议退出当前主线待办，不代表全部已修复；继续维护旧任务时需重新核查。历史审查原文可从 Git 历史追溯。

## 仍存在的问题

**旧版 ONNX 导出会覆盖同名文件（原 O1 / P2）。** 静态核对 `scripts/agent/export_chuanliantui_onnx.py` 的 `parse_args()` 与 `main()`：目前没有输出文件存在性检查或 `--force` 参数，直接调用导出和保存。建议默认拒绝覆盖，显式允许后才覆盖；仅影响 Isaac 25 维旧导出器，不据此判断当前 multimode 导出器。

## 当前记录入口

| 内容 | 查阅位置 |
|---|---|
| 当前模型、实验结论与尚未完成的验证 | [项目实验记录](docs/ai/project-log.md) |
| multimode 主线、运行方式与接口 | [mjlab 串联腿说明](docs/mjlab-chuanliantui.md) |
| 当前共享 XML 来源及适用范围 | [共享模型记录](docs/diagnostics/20261021-shared-huananhu-xml.md) |
| model_6848 导出、数值检查及后续交付状态 | [模型交付记录](docs/model-deliveries/20261021-model6848-h7-export.md) |
| model_6848 的 H7 接入与烧录证据 | [接入与烧录记录](docs/interfaces/20261021-h7-model6848-flash.md) |
| 断动力多圈重入的实测范围与剩余缺口 | [只读验证记录](docs/diagnostics/20261022-h7-angle-reentry-validation.md) |
| 旧 Isaac 任务的模型约束与验证方式 | [sim2sim 说明](docs/ai/sim2sim.md) |

## 使用边界

不同模型、XML、控制代码和验收口径的结果不能相互继承；历史“本轮未执行”也不自动成为当前项目待办。是否通过行为或实机验收，以对应版本的最新记录为准。

本次仅整理文档并静态核对旧导出器，未运行导出、仿真、训练或硬件验证。
