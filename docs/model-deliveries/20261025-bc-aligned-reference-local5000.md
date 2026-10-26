# 共享轮角／LQR参考39维BC：本地新增5000次

2026-10-25，Codex执行，用户接收；独立成员复查与H7负责人核对待确认。本轮起立／最终双轮平衡40/40、完整20秒40/40，停车0/40；仅留作阶段候选，保留旧模型，不接PPO或部署。

| 项目 | 实际记录 |
|---|---|
| 机器人／用途 | 当前串联腿共享XML；正常起立、零速静站与停车，不做移动 |
| 接口／模型 | `chuanliantui-huananhu-wheel-angle-39-82-handover-v2`；`ct_bc_actor39_pd6_wheel_angle_handover_v2`；39维actor、6维PD目标，环境critic82维但本模型不含训练好的PPO critic |
| 参考／同步 | 首次ground_start且recovered，同拍清两轮原点及LQR位置；标签`ct_bc_shared_wheel_position_reference_v2`，禁止混旧参考。H7、普通PPO及CPU入口待适配／负责人确认 |
| 仓库／版本 | origin `https://github.com/zn1135/wheelleg_RL_2026.git`，detached HEAD `2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`＋已有未提交修改；无Git操作，实际内容见snapshot／input-manifest／tracked-simulation-diff.patch |
| 初始化 | 旧累计15000模型SHA `c29ab3ecb16368114dc34375dec54fbabef4c581eeae955f561687368dcf6e83`；仅首层14、15列置零，其他参数保留，新Adam，不恢复旧优化器 |
| 数据 | 共196教师候选，136条合格：训练109／34姿态，验证27／9姿态；原39／10分组不变，5训练／1验证组无合格完整示范。实际接管10/156条、7训练姿态、18500帧；另有17031起立保留帧（39训练姿态）及1308反馈差分对（34训练姿态）。原40组测试均保留；未混旧动作或失败正例 |
| 训练／预算 | 2026-10-25T22:44:57+08:00～2026-10-25T22:46:56+08:00，本地RTX4060 Laptop 8GB；明确新增5000、累计20000，预算用完；lr0.0001、seed42、IEEE FP32、专家1024＋起立256＋反馈256，每500保存 |
| 命令／日志 | [run-local.sh](../../artifacts/bc-aligned-reference-20261025-v1/run-local.sh)，[training.log](../../artifacts/bc-aligned-reference-20261025-v1/training.log)；TensorBoard命令见报告，未启动服务 |
| checkpoint | [bc_5000.pt](../../logs/mjlab/chuanliantui_bc/20261025_aligned_reference_local5000-v1/bc_5000.pt)，SHA `2ac61f711e17e9e37347fefeb4a72149b17f02ee9219ebad18225bc919baaa89`；含actor／新Adam／RNG及来源，无ONNX、板端网络或固件产物 |
| 数据指纹 | `83ea7489c513f65a31fbf98a22fac42ad2bf99f704906ef6dbb5d6aedfc229fc` |
| XML／物理／时序 | SHA `cac8061de2850602bfd1421fc856bf217c4e5040a11c8f5d33cf1abe61d5f0cf`不变；2ms物理／PD、100Hz策略、2／4ms延迟、0.2秒零主动力，根初态分布及物理参数不变 |
| 运行版本 | Python3.12.3／torch2.14.0／MuJoCo3.11.0／mjlab1.6.0；[完整版本](../../artifacts/bc-aligned-reference-20261025-v1/runtime-local.json) |
| 数值验收 | 来源冻结、分组隔离、梯度、Adam／事件5000通过；固定验证缩放MSE 0.01549496；原40组完整初态一致，动作及轮角误差0，不能代替行为验收 |
| 行为验收 | 正常起立40/40、最终双轮平衡40/40、完整20秒40/40，末5秒腿速／偏航门槛均40/40；平均速度0.0108676m/s、最差0.0118964，平均漂移0.0516571m、最差0.0581932，停车0/40。上轮4500候选速度均值0.0312308m/s；新参考、数据、保留时段同时变化，不是单因素对照 |
| 视频／范围 | 原40组中的实际教师／学生完整20秒60fps对照，视频在报告；40组已用于历史模型选择，不称全新泛化。移动、10ms压力、噪声／随机化、推扰、全姿态、CPU sim2sim、ONNX／H7及真机未验证；Isaac不适用 |
| 旧模型／边界 | 旧稳定15000、旧v1候选4500、XML、SMP prior校验不变；SMP／PPO更新0，未改H7或默认策略，无追加BC预算 |

[完整报告、命令与视频](../../artifacts/bc-aligned-reference-20261025-v1/README.md)、[接口定义](../interfaces/20261025-wheel-lqr-position-reference.md)。有限保留约束不能保证下一轮闭环不退化；本轮只确认这40组未丢失起立与最终平衡，停车仍不通过。
