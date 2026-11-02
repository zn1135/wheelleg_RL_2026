# 复旦PPO实体电机接口 v1

当前最终验证按[GPU验证约定](20261102-fudan-gpu-validation.md)执行；本记录中的Native实现仅属历史，原源码已归档。推理接口保持。

合同：`fudan-ppo-walk-physical25-history125-critic144-v1`，新任务 `fudan_ppo_walk`。当前为实现候选，无合格模型交付。逐槽、缩放、PD、默认角、命令、导出和参考差异以[任务规格](../mjlab-fudan-ppo-walk.md)为准。

Actor25＋history125→actions6；训练环境信息141＋估速3→Critic144。实体动作顺序lf0/lf00/lfwheel/rf0/rf00/rfwheel。禁止复用虚拟髋膝25维、旧split25维、SuperMimic、BC或SMP权重。ONNX输入obs[1,25]、obs_history[1,125]，输出actions[1,6]；图外维护历史/预处理/PD。策略100Hz、PD500Hz；本轮额外动作/传感延迟为0，只是实验条件。

H7代码、真实时序与物理标定未改；部署负责人核对待确认。后续须分别实测传感数据龄、策略输入/推理、指令排队/传输/生效延迟及抖动，并明确同步时钟和可测边界；控制周期与发送时间不能替代延迟/实际响应测量。本次不授权硬件实验，不声称真机通过。

## 来源、兼容与签认

| 项目 | 内容 |
|---|---|
| 作者／复查 | Codex整理；无另一方签认，H7负责人待确认 |
| 旧定义 | 新任务首次定义；已有split25/125/71及复旦虚拟25维均不兼容 |
| 训练来源 | https://github.com/zn1135/wheelleg_RL_2026.git，26_wheelleg，HEAD `3794efd390a0eb68cb154839423a7d53e32ff61e`＋本次未提交源码；模型以归档source为准 |
| 权威实现 | ct_mjlab/fudan_ppo_walk/env.py::frame/observations，PhysicalAction，policy.py::load_policy，export.py::Policy；旧Native见归档 |
| 机体与采样 | +X前向、+Y左、+Z上；gyro/重力按机体坐标，yaw命令为机体z角速度；电机速度2ms位置差分 |
| 初态／停止 | 0.26m几何初态＋2mm、单位姿态、六根速度±.5；无接管/停车外环；20秒训练超时，验收用冻结case时长 |
| 物理／同步 | 共享闭链XML和实体PD未改；H7无同步代码，待负责人另行适配，未部署 |
| 关联模型 | [首轮交付记录](../model-deliveries/20261101-fudan-ppo-walk-trial01.md) |

旧权重不可按张量同形加载；新任务必须重新训练。图外clip±100、PD/命令/历史等逐项定义见规格表。训练入口相位修复不改变推理张量，但本轮模型训练源码有明确偏差；修复版未重训。完整中途恢复未实现/未验证。

后续新源码已实施[reset/history v2](20261101-fudan-reset-v2.md)，以额外reset_contract隔离训练/夹具语义；本记录及trial01模型仍对应原实际实现。新动态预检/训练尚未执行。
