# 1999 ONNX导出交付索引

2026-10-30按用户要求实际重新导出旧高度门控1999。按exports/onnx/README的分类约定，模型、使用说明、实际交付记录、参考向量和校验清单统一放在[model_1999](../../exports/onnx/model_1999/)。

- [交付记录：来源、命令、环境和分阶段验证](../../exports/onnx/model_1999/20261030-model1999-onnx-export.md)
- [35维使用说明](../../exports/onnx/model_1999/onnx-height-model1999.md)
- [ONNX模型](../../exports/onnx/model_1999/model_1999_opset13.onnx)
- [实际产物校验清单](../../exports/onnx/model_1999/SHA256SUMS)

检查点哈希1c3487a7da8d1fe74c418466ec4068f621c162e36406c14786c972a2839e7027，ONNX哈希5d45c516c87eb554f8ffcdd4d28e31b197abff2cb8bdb595d6e2b1989599e5a4；64组PyTorch/ONNX最大差3.0994415e-6，小于1e-5。当前部署工作区1954个本地已跟踪文件对比无缺失、无字节差异；本轮仅导出及只读比较，没有提交/推送。
