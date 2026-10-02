# 模型使用记录：2026-10-01 chuanliantui 失能数据链诊断

| 项目 | 内容 |
| --- | --- |
| 日期／执行者／接收者 | 2026-10-01，Codex／用户当前工作区 |
| 机器人／任务 | chuanliantui_standup，大轮腿 STM32H723 |
| 用途 | 真实完整策略推理下的last_action、5帧历史和100Hz通信时序验收；全程失能 |
| 接口 | [diag2701](../interfaces/chuanliantui-h7-policy-diag2701.md) |
| PR | 无，未提交或推送 |
| 训练代码 | wheelleg_RL_2026，26_wheelleg，bb56dd40964dc569b4e2a406983b03fa19d943a1；脏工作区，不能仅以HEAD代表运行内容 |
| 部署代码 | H7_REPO_PATH的H7_clion，无Git；固件与源码SHA以证据manifest.json为准 |
| 运行差异及未跟踪源码 | 部署captures/20261001-policy-runtime-firmware/training-tracked.patch及最终host-source-snapshot-r2.tar.gz；固件source-snapshot.tar.gz；均不包含.env.local |

## 模型来源与运行环境

完整checkpoint：`logs/chuanliantui_standup/Sep22_16-26-11_standup_no_legangle_resume/model_6000.pt`，
1199938字节，SHA256 `8070977e19114cb9bbf358a090bd78cdbd1be568a76e02d417c369e828bc8f36`。
checkpoint内iter=6000，含model_state_dict及优化器状态；本次只读载入权重，不恢复训练。
实际历史训练命令、seed及当时配置快照本次未独立核实，不能从run名推断训练质量。

encoder=125→128→64→3；actor=28→128→64→32→6；critic=68→256→128→64→1。
部署诊断只调用act_inference(obs[1,25], history[1,125])，确定性返回action[1,6]、latent[1,3]。
使用WHEELLEGGED_PYTHON配置的Python3.8、CPU单线程，先import isaacgym再torch；无仿真创建。
原有ONNX不是本次执行输入，本次不新增导出或板端网络；推理运行在电脑上。

## 分阶段证据

| 阶段 | 状态／依据 |
| --- | --- |
| 完整模型加载、形状和encoder+actor数值 | 通过；model-cpu-precheck.json，输出有限、与显式组合相同 |
| 纯CPU推理耗时 | 1000次预热后1000次，中位0.0571ms、最大0.1133ms；不含通信或调度 |
| 主机离线接口检查 | 26项通过，含真实模型；host-policy-check.txt，退出0 |
| 固件构建和状态机检查 | Debug构建、5组C及27项Python通过，build.txt/host-tests.txt |
| 板端实际动作回填／历史／周期 | 通过；两段正常各1001帧与121帧预期漏发故障段，见下方实测结论及verification.json |
| 训练、Isaac回放、MuJoCo行为 | 本次未执行；仅使用既有模型验证新增硬件数据链，不修改训练/动力学 |
| ONNX检查／导出 | 本次不适用；没有运行ONNX |
| 实际电机非零动作／闭环稳定 | 未执行，诊断明确保持失能 |

物理零点、部分膝动态和轮速幅值仍有既有未验收项；不因数据链通过而解除这些限制。
旧固件回退组合及SHA见captures/20261001-remote-enable-firmware和此前观测映射记录。

## 实测结论（完成）

板端数据链阶段已通过：normal-02和reset-01各1001帧，采样全部10ms，原始策略动作回填及
125维FIFO逐位一致，最大应用接受延迟4ms。drop-01主动漏发sample120的动作，10ms后
正确进入DEADLINE FAULT，随后新session首帧零last_action和首帧五次填充正确。
三段退出码均0，全部结束于OFF、DM失能、六路目标0。详细证据见部署端
`docs/policy-runtime-validation.md`及`captures/20261001-policy-runtime-firmware/verification.json`。

实际在线推理耗时中位0.40–0.42ms、最大0.845ms（含输入转换）；Python3.8.20、
PyTorch2.1.0+cu118、NumPy1.19.5，CPU单线程。微基准不含实际收发节奏，两者统计口径不同。
最终主机r2修正START回应可能已准备sample0的合法时序；固件不变，新增边界检查通过，
实际源码为host-source-snapshot-r2.tar.gz，SHA清单在manifest.json。首次normal-01主机
拒绝结果保留，不计作成功。数据链允许使用范围为失能诊断，不能据此投入电机闭环。

## 下一阶段：动作执行前离线复算

2026-10-01 用同一模型的既有正常记录共 2002 帧复算名义虚拟 PD，未重新推理或烧录。
两轮在各自全部帧达到 ±3.9 N·m 限幅，腿虚拟力矩最大约 35.7 N·m；没有发送这些力矩。
新脚本、证据口径、与 USB 实体槽的差异及未验收项见[动作复算接口记录](../interfaces/chuanliantui-h7-actuation-preview.md)。
本结果仅支持先进行独立的小幅单轮台架试验，不构成完整策略运动通过。

随后已用固定单轮脉冲完成两轮输出通道和前进方向实测，记录见上述接口末节及部署端
`docs/wheel-output-validation.md`。该试验不运行策略，两轮越过主机停止阈值后提前切零失能，
后补只读记录确认静止。模型本身、训练代码及板上固件未改；不据此声称策略闭环通过。
