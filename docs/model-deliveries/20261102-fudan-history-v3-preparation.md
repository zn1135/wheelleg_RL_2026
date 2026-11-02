# 复旦history v3准备记录（无新训练模型交付）

日期2026-11-02，Codex准备；接收人/H7负责人、独立复查人待确认。任务为`fudan_ppo_walk`，关联[历史v3接口](../interfaces/20261102-fudan-history-rollback-v3.md)。本记录仅确认输入语义和学习状态迁入准备，不是新模型发布或板端通过。

训练仓https://github.com/zn1135/wheelleg_RL_2026.git，`26_wheelleg`，HEAD `3794efd390a0eb68cb154839423a7d53e32ff61e`＋未提交源码。修改前后源码与SHA位于[准备目录](../../artifacts/fudan-history-rollback-v3-20261102/README.md)；H7未修改，部署端版本/签认待负责人核对。只保存本任务相关源码，原模型的实际训练来源仍是原logs/source，不能由当前脏工作区HEAD代替。

| 产物/状态 | 依据 |
|---|---|
| 计划源checkpoint | `logs/mjlab/fudan_ppo_walk/20261102_resume6000_to7000/model_final.pt`，SHA256 `ea5dac38af946359321e415aa661bc78cc929ddf8f29d46fec6f0116c610c9d2` |
| 源模型实际训练语义 | reset/history v2，7000轮、累计860160000转移、PPO/encoder各140000更新；不得改标为v3 |
| 新v3训练模型 | 没有；追加轮数待回复，未创建新训练run/logdir |
| 新ONNX/板端网络/固件 | 没有；未运行导出、编译或烧录 |
| 输入/输出 | 仍obs[1,25]＋obs_history[1,125]→actions[1,6]，五帧最旧在前；仅历史初始化和clip顺序改变 |
| 控制/资产 | 原2ms PD/物理、10ms策略、实体闭链XML及映射保持；源不变核对见检查报告 |
| 恢复范围 | CPU/CUDA真实模型/两个Adam/学习率/RNG/计数恢复核对通过；新物理/历史/回合重新初始化 |
| 静态输入检查 | 31项通过；替代后端只执行真实预处理/复位控制流，0真实物理、0模型前向、0优化器更新 |
| 新动态预检/行为/TensorBoard | 均未运行，无新日志；旧v2预检不能代批v3 |
| H7时序/真机 | 未验证，负责人待确认 |

原7000模型的GPU诊断和验收v2离线结果仅适用于原v2输入，不能当作v3行为结果。后续明确预算并完成v3对应物理预检后，方可用显式迁移入口继续4096环境训练；未授权自动追加。新模型完成后再填写实际训练命令、运行库版本、训练时长、模型SHA和ONNX数值证据。

回退组合：原7000 checkpoint＋原run的`source/ct_mjlab`，或当前评估器显式读取该checkpoint的v2标记；保留全部旧日志、视频、夹具和预检。H7同步及实机使用范围均未批准。
