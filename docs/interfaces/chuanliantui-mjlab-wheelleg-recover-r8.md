# 串联腿恢复四腿在前动作顺序 r8

2026-10-10，用户要求恢复策略输出顺序。26_wheelleg，基础HEAD58947e8fabc4683b2025d42bda742b06c84dc566加未提交改动；助手编写，独立复查待执行，H7适配待部署负责人完成。

当前接口 `chuanliantui-mjlab-wheelleg-recover-r8`。Actor35/Critic78/Action6；actor／critic观测布局和动态normal/recover仍沿用 [r7](chuanliantui-mjlab-wheelleg-recover-r7.md)，观测合同保持`chuanliantui-huananhu-flat-35-78-dynamic-modes-v2`。本次改变动作输出合同及对应控制合同，保留历史接口。

## 六维动作与观测

策略输出、关节位置／速度观测、上一拍请求动作观测均按 `[lf0,lf00,rf0,rf00,lfwheel,rfwheel]`。

| 输出槽 | 电机 | 控制 | 力矩限幅 |
|---|---|---|---|
| 0 | lf0 | 名义位置+动作×0.5，PD60/2 | ±40Nm |
| 1 | lf00 | 名义位置+动作×0.5，PD60/2 | ±40Nm |
| 2 | rf0 | 名义位置+动作×0.5，PD60/2 | ±40Nm |
| 3 | rf00 | 名义位置+动作×0.5，PD60/2 | ±40Nm |
| 4 | lfwheel | 动作×10rad/s，目标限±150rad/s，Kv0.1 | ±3.9Nm |
| 5 | rfwheel | 动作×10rad/s，目标限±150rad/s，Kv0.1 | ±3.9Nm |

实体XML、joint ID及内部物理执行器顺序保持原样。动作项保存请求动作的策略顺序，送入物理PD／延迟缓冲前按`[0,1,4,2,3,5]`重排到`[lf0,lf00,lfwheel,rf0,rf00,rfwheel]`；奖励内核也把当前／前一拍／前两拍动作同时映射到此物理顺序，与物理速度、加速度和力矩一一对应。腿轮索引不混用。该内部映射无需H7照抄；H7按输出电机名称和表中增益实现控制。

Actor22:28直接保存上述策略顺序的请求动作，未做控制缩放。Actor10:16/16:22、critic增益／力矩槽仍为四腿两轮，因此观测数值语义不变。恢复标志actor31、恢复目标33、恢复计时34维持r7定义。

27奖励公式／权重、高度门控、PPO、初态、随机化、PD、气弹簧和物理步长沿用r7；此前建议的奖励门控、起立奖励和课程尚未实施。

## 历史模型与导出

- r5仍使用历史direct环境和原模型。
- r6真实训练模型自动按旧交错动作顺序、固定NORMAL回放／CPU评估。
- r7保存合同自动按旧交错动作顺序、动态mode回放／CPU评估；没有正式r7训练模型，本次仅用历史manifest推理夹具检查加载路由。
- r6/r7完整manifest分别保存于reference_training/r6_interface.json及r7_interface.json；按原manifest及当前资产／初态缓存哈希核验，拒绝直接完整resume到r8。
- r8新actor／ONNX输出四腿两轮；历史模型导出保留原输出顺序及其JSON，未变换旧权重。ONNX float32[1,35]→[1,6]、opset13，PD／预处理在图外。

本次未训练、未启动TensorBoard、未提交Git。检查与范围见 [r8诊断](../diagnostics/20261010-mjlab-r8-action-order.md)。H7固件未修改，模型交付与真机验收待执行。
