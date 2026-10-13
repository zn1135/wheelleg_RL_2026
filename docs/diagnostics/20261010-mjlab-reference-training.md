# wheelleg_RL训练完整移植

2026-10-10，26_wheelleg，HEAD 58947e8fabc4683b2025d42bda742b06c84dc566 加未提交改动。保留原工作区修改，未提交Git。

用户要求移植本机wheelleg_RL当前训练全部参数，保留机器人。完整接口见 [r6](../interfaces/chuanliantui-mjlab-wheelleg-recover-r6.md)。来源为参考脏工作区，不从HEAD猜参数，逐文件哈希见 [版本／执行记录](20261010-mjlab-reference-training.json)。

## 改动

复制参考主线huananhu配置、动作、观测、奖励、命令／终止、随机化、runner和相关公共模块至reference_training。使用其原生mjlab管理器环境，避免手工近似逻辑遗漏。model.py适配本仓机器人资产/ref/名义角；huananhu_poses保留参考运行时采样和碰撞界算法，仅把离线生成替换为本仓已核对的闭链表加载。V14观测公共模块仅解除未使用standup阶段模块依赖，保留常量值。

原SCUT flat v0与当前wheelleg_RL v12/recover不同：27奖励、倒数线速度跟踪、高度门控、膝限位、静止惩罚／平方权重、目标高度范围、全落地与根扰动、动作顺序、固定mode、延迟取整、轮目标限幅、材质语义、加速度/力矩取样、弹簧随机化、接触历史8、50轮保存、终止原实现和严格恢复指标全部对齐。
CLI保留本仓日志目录、命令格式、默认不开窗口；当前完整env/runner配置和脏源码快照随每次新run保存。r5旧模型自动选旧环境回放，r5奖励恢复原公式，不在旧回放里混入新门控。拒绝不匹配的r6续训或外部资产初始化，不修改H7和历史logs。

## 实际验证

[参考对照与检查JSON](20261010-mjlab-reference-training-check.json)：

- 分别导入参考源与移植模块，所有环境配置（唯一排除机器人Entity）相同；完整PPO配置相同，含继承默认值。
- 27原公式／权重逐项参考数值对照最大绝对误差0；高度门控在0.30m命令下覆盖全额／渐变／零的边界。
- GPU32环境、128策略步、1280物理步，seed42；每个积分后的qpos另在CPU做独立运动学，覆盖初态及自动reset前的终止状态，不向生产物理额外插入forward。
- 最大闭链偏差 **2.702 mm**，通过原5 mm门槛；数值异常0。普通reset 71次，零请求动作仍有PD和弹簧，不能当作起立成功。
- actor35／critic78、固定NORMAL、命令范围、动作3/5子步与传感3/5/8子步、部分reset不影响其余环境历史通过。
- runner仅构造/保存/加载，**没有learn或优化器更新**；checkpoint往返、r5 inference加载、r5续训拒绝、其他资产拒绝通过。
- 临时随机actor ONNX误差 1.19e-07，不交付为训练模型。
- 旧r5正式CPU数学／物理／导出检查通过，见 [历史保护检查](20261010-mjlab-r5-legacy-check.json)。
- r6 CPU同任务eval路径有限检查正常退出；零请求动作在0.84s触发失败，passed=false，不是策略行为验收。
- 20环境10s原生窗口预检正常退出，无学习；首次缺DISPLAY尝试失败，指定DISPLAY=:0重试成功。未声称人工观察验收通过。
- git diff --check通过。

上述均为参数/数值/有限物理/接口检查，不是训练后回放验收或真机结论。用户随后授权1000轮，正式训练另记录在 [r6训练启动](20261010-mjlab-r6-formal-training.md)。
