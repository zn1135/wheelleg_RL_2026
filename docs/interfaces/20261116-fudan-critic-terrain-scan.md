# 复旦行走：Critic真实77点地形扫描

| 项目 | 内容 |
|---|---|
| 日期、作者、复查 | 2026-11-16；AI按用户本次要求实现；H7负责人核对待确认 |
| 机器人／任务 | 当前串联腿模型，`fudan_ppo_walk`，训练侧专有地形输入 |
| 旧／新输入标识 | `fudan-critic-repeated-height-v1` → `fudan-critic-yaw-height-grid-v1` |
| 训练仓／分支／基点 | zn1135/wheelleg_RL_2026；`26_wheelleg`；`cf5accb82f54a8607f58207a649ba11c2288077d`加当前工作区；实际Python源码SHA见本次budget与训练快照 |
| 权威实现 | `envs/chuanliantui/fudan_ppo_walk/terrain.py::critic_record/configure/scan_state`、`FudanEnv._update_state/observations`、`FudanRunner.load/checkpoint_metadata`、`scripts/train.py::train_fudan` |
| 部署端 | 本次不交付模型、不改H7；Actor部署输入保持，板端行为与负责人核对待确认 |

## 输入含义

Critic是训练时估计当前状态能获得多少回报的价值网络。环境输入仍141维，接encoder的3维后仍144维；只替换从0开始的半开区间`46:123`。旧值为当前机体离地高度重复77次，新值为机体周围真实地面高度。

扫描范围以机体为中心，前后各0.5m、左右各0.3m，间隔0.1m，共11×7点；网格随yaw转动，射线始终沿世界竖直方向向下。`xy`网格排列，x变化最快，先扫y=-0.3一行。起点高于机身1m，最大测距3m，仅碰撞组0的真实地形参与。每点输入为`clamp(root_z-ground_z-height_center,-1,1)*height_scale`，本次center=0.5m、scale=5，随后仍执行原整体观测限幅。

Actor25维、历史125维、encoder3维、动作6维、历史填充/裁剪/延迟、关节顺序、坐标与动作缩放不改。奖励仍按机身正下方单点真实净高计算；不把周围77点均值改成奖励高度。任何扫描点缺测或地面非有限均明确结束相应回合，奖励清零、不按超时bootstrap，不增加速度课程得分；有限替代值仅供终止拍存储，不当作有效平地。

配置入口`FudanWalkCfg.observations.critic_height_scan`默认False；新训练显式使用`--terrain regions --critic-height-scan`。旧平地、旧分区、独立台阶任务保持各自入口和原语义，独立台阶已有自己的地形扫描，不套用本次版本。

## checkpoint与续训

新行走PT与恢复报告保存`critic_observation`完整定义，旧缺字段解释为旧重复高度。即使形状相同，输入含义改变也必须显式迁移；加载器拒绝隐式跨版本恢复，训练入口拒绝旧Critic预检。Actor的既有导出接口不因本次训练侧改动增加地形输入，但学习后的动作仍须重新验收，不能继承旧模型的成绩。

本次用户要求从5000新增1000轮，到6000停止，继承5000全部网络（含Critic）、双Adam、学习率、随机数状态与计数，物理回合重新初始化；没有重置Critic或优化器。这样与先前不带扫描的分区续训保持相同起点与迁移方式。源PT SHA256：`010106ac0b2ee356a630ab27ebe1c85ab2a6c843d21a1b4362ab81f1aafe0222`。

相对先前分区有效配方只增加扫描；10～20°金字塔、崎岖±1cm、50/25/25出生、vx最高3、yaw±10及高速vx=0、全幅随机化、实际摩擦0.4～1.2、各轴±3N、7秒/0.3m/s短推、20/1与1/5/10ms、双10～20ms延迟均保留。旧5000→8000任务已按用户新要求中止，历史PT/日志保留，不能称完成8000；原始异常退出与单独的`superseded.json`均留档。

## 验证与后续判断

正式检查入口`scripts/agent/fudan_ppo_walk/check_critic_scan.py::check`由20环境可视地形预检调用，不创建临时单测。实际GPU射线在平地、坡面、旋转90°、平移/转向和崎岖五处与CPU几何射线独立复算；CPU没有推进物理。最大高度误差2.3842e-7m；平地与旧77槽逐值一致，非平地输入变化，同随机数下Actor及其余Critic槽逐值一致。真实缺测终止、旧预检及隐式迁移拒绝检查通过。20环境2500批量GPU物理步，500次PD/2000保持、20次单拍推扰，施力叠加误差0，无容量溢出，零网络/学习更新。

4096容量、完整学习状态恢复与实时训练进度以[本次实验目录](../../artifacts/fudan_ppo_walk/20261116-critic-scan-resume5000-6000-01/)为准；通过启动检查不表示地形通行或真机行为通过。结束后优先比较相同上下坡/崎岖路线与原47平地，检查到达率、倾倒、速度误差及停车保留，判断扫描是否带来改善；不再同时改出生、yaw采样或奖励。
