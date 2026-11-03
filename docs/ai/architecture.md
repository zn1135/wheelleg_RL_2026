# 架构与数据流

最终验收现统一为GPU、共用运动门槛及报告，见[统一GPU验收](../interfaces/20261102-unified-gpu-acceptance.md)。主线统一入口为`scripts/mjlab.sh accept --task <任务> --recording <GPU记录> --output <新目录>`；起立单列，旧CPU流程仅保留历史含义。

维护SuperMimic、BC、SMP及独立复旦PPO行走适配；模块入口见[模块职责](modules.md)，当前模型结论见[项目实验记录](project-log.md)。

## 训练流程

SuperMimic当前世界模型使用129维输入：47维航向无关运动特征＋75参数＋当前6动作＋1年龄，不读取前两拍动作。运动特征为高度、去航向旋转6D、机体系线/角速度和关节状态；参数中的外力/力矩也转为机体系。真实状态和输出积分仍用原物理坐标。策略仍为Actor25/历史125，真实回放保留动作历史。即使同为129维，旧绝对坐标模型也不能直接续训；见[坐标接口记录](../interfaces/20261101-supermimic-v13-heading-invariant.md)。

当前预测已进一步改为单个10ms终态：输出46维增量，监督与归一化只取真实控制步终点。想象观测/奖励直接由终态生成，不重建2ms传感队列；解析力矩使用当前动作。价值输入去掉162维传感队列后为162维。真实500Hz仿真、5子步记录和策略125维历史保持，不能把想象中的无队列处理当成真实延迟已取消；见[单步终态接口](../interfaces/20261101-supermimic-v13-control-endpoint.md)。

策略想象现在在每个预测终态生成后、计算观测/奖励/存活和下一拍前执行状态夹紧。世界模型拟合与普通预测默认不夹紧，真实标签不改；范围见[想象夹紧记录](../interfaces/20261101-supermimic-imagination-clamp.md)。

新SuperMimic配方原始动作统一±3，真实采集在探索噪声后限幅；想象、评估、ONNX以及回放/下一观测一致。旧合同原样推理/导出保留历史无限幅行为，新旧完整训练合同不混载；见[动作限幅记录](../interfaces/20261101-supermimic-action-clamp.md)。学习率调度未修改。

- SuperMimic：真实仿真采集 → 世界模型／估速器监督 → 预测运动中的策略及价值网络更新 → 真实仿真行为验收。算法不采用PPO。
- BC：传统控制器经学生动作／PD链路生成示范 → 观测与动作同拍配对 → BC actor监督学习 → 独立闭环评估；可再初始化⑦⑧中的PPO。
- SMP：合格运动示范 → 先验模型训练 → 冻结先验给运动评分 → SMP辅助PPO训练 → 独立行为验收。

共用真实XML、控制时序、观测和模型接口按各自合同核对；不同接口不能互套。注册环境为`smp`、`supermimic`和`fudan_ppo_walk`；五个旧独立训练task退出。

## 仿真与真机

`sim2sim/`内MuJoCo实现保持原样，包括双策略回放所需split模型验证与切换模块。训练侧目录仍保留这些共用底层实现，不保留独立起立／行走训练入口。MuJoCo回放、训练环境行为与H7真机结论分别记录。

真实控制仍在H7；Jetson用于显示、录制和离线分析。本次清理不修改`deployment/`或`resources/`，不改PD、频率、执行器、物理参数、观测顺序或H7代码。部署接口以[部署约定](../deployment-contract.md)为准；H7未核对的交付继续标待确认。

## 源码与模型来源

旧SuperMimic模型保存的任务／算法合同继续严格验证；撤销入口导致的源码变动通过明确的逐文件before／after SHA映射核对，原训练快照逐字节校验。未知修改继续拒绝，不跳过源码验证。记录见[清理范围](../maintenance/20261101-retain-678-review.md)。

复旦最终验证只用MuJoCo Warp GPU，原生CPU物理入口已撤销，共用模型加载在`policy.py`，见[约定](../interfaces/20261102-fudan-gpu-validation.md)。历史CPU产物保留，其他任务sim2sim保持。

复旦PPO独立线：真实10ms采样48步→PPO更新actor/critic/std→另一个Adam监督encoder估速；实体闭链机器人/PD沿用本机，命令、奖励、历史和域随机化独立配置，不经过旧起立/停车外环。Critic环境141维＋encoder3维；Actor不读真值线速度。[规格与差异](../mjlab-fudan-ppo-walk.md)。

## 复旦注册训练链路

复旦已按原wheel_legged_gym职责整理：`mjlab.sh train --task fudan_ppo_walk → scripts/train.py → utils/task_registry.py → envs/fudan_ppo_walk的环境/配置 → rsl_rl/runners → rsl_rl/algorithms`。注册器实际创建环境和runner；旧独立入口仅兼容转入这条链路。其他训练线沿原流程。默认值、RNG、v2/v3、模型接口与物理保持，验证只做无物理等价，见[迁移与日常位置](../maintenance/20261103-fudan-structure-alignment.md)。
