# 接口记录：共享闭链稳定约束与实测限位

| 项目 | 内容 |
|---|---|
| 记录日期、作者、复查人 | 2026-11-04；Codex；用户/H7负责人独立复查待确认 |
| 机器人、任务、机构／硬件版本 | chuanliantui_new_1实体闭链/滑杆气弹簧；当前mjlab supermimic、smp、复旦PPO及共享MuJoCo仿真 |
| 旧接口标识与记录链接 | 25/125/6复旦网络接口保持；[部署约定](../deployment-contract.md)；旧共享XML 14fdae8d… |
| 新接口标识 | 网络接口不变；物理资产记录 `shared-closed-chain-stability-20261104`，XML 505c9b9b… |
| 训练仓 URL、分支、SHA、工作区差异 | github.com/zn1135/wheelleg_RL_2026.git；26_wheelleg；c3c8fdffe0739aed314fb71c69f2a2d3135b3fb9＋脏工作区；[实际源快照](../../artifacts/shared_validation/closed-chain-training-integration-20261104-01/source-sha256.json) |
| 部署仓 URL、分支、SHA、工作区差异 | 同仓deployment/H7_RL及H7_RL_ros均未修改；1366个ROS固件来源保持；负责人待确认 |
| 权威实现文件／符号 | [共享XML](../../resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml) default/joint及equality；训练load_spec/create_robot_cfg；原生sim2sim build_model；复旦FudanEnv |

## 接口定义与差异

| 项目 | 旧定义 | 新定义 |
|---|---|---|
| 观测索引/单位/缩放/限幅 | 各任务原接口 | 保持；未新增观测 |
| 历史长度/初始化/更新 | 各任务原接口，复旦125历史 | 保持 |
| 前向/IMU/yaw含义 | 各任务原接口 | 保持 |
| 关节顺序/默认角/速度采样 | 六实体动作槽及原零位 | 保持 |
| 动作/PD/力矩范围 | 原设置，腿±40Nm、原轮执行器 | 保持 |
| 物理/PD/策略/延迟 | 当前主线2ms/2ms/10ms，训练2+2ms、GPUsim2sim4+4ms | 保持；refsafe启用，名义solref .4ms在2ms积分下夹至至少4ms |
| 初态/复位/策略接管 | 各任务原合法闭链复位 | 保持，20份初态/随机物理夹具逐值相同 |
| XML膝范围 | 已标定lf1[-.97,.14]、rf1[-.16,1.01]rad | 保留；20,214姿态复核0越界 |
| 六个闭链connect | solref .01/1；solimp .95/.99/.001/.5/2 | solref .0004/1；solimp .999/.9999/.001/.5/2 |
| 关节限位约束 | MuJoCo默认响应 | 继承同组solreflimit/solimplimit，范围本身不改 |
| 张量/模型格式/opset | 原checkpoint/ONNX | 保持，无新网络导出 |

## 兼容性与同步

旧checkpoint可以加载；20例同初态8000模型回归19→20通过且无旧成功退化，仅代表此次诊断，不代替完整987或真机验收。物理模型变了，不能把历史导出/报告的源hash改成新值；完整验收后再决定是否重训，当前不自动训练/导出。

实际策略姿态100Hz复算的六连接峰值为0.106436mm，略超0.1mm诊断目标；轻载逐2ms测试0.050962mm不能代表全部策略轨迹。更硬第二候选峰值0.120856mm、旧成功diagnostic_9新增起步触地超时，已拒绝并恢复本记录第一组；最终来源以final-selection.json和当前505c9b9b…XML为准。

| 仓库 | 待同步文件与改动 | 版本 | 负责人 | 状态 |
|---|---|---|---|---|
| 训练仓 | 共享XML连接/限位约束、新检查器与记录 | 未提交工作区，this-task.patch | 用户/RL侧 | 实现及GPU诊断完成，完整验收待做 |
| H7_RL/H7_RL_ros | 无控制/零点/力矩/固件变化，不把仿真约束作为板端保护 | 原版本保持 | H7负责人 | 待确认，无本次烧录交付 |

回退须使用本次before-source.xml及对应历史源组合；不能恢复/覆盖其他用户既有修改。原固定机身无接触EFC排序诊断不移植进动态接触训练后端。

## 验证依据

[模型交付/范围记录](../model-deliveries/20261104-shared-closed-chain-stability.md)、[来源和完整命令](../../artifacts/shared_validation/closed-chain-training-integration-20261104-01/README.md)。静态/几何检查、GPU接触对照、20例冻结策略回归及统一accept均退出0；0策略训练更新。完整987、其他训练线行为、板端时序和真机未测；不能用静态检查替代这些结果。
