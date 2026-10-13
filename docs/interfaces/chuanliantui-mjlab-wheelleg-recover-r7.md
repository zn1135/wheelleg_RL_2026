# 串联腿动态normal/recover模式恢复 r7

2026-10-10，用户要求把r6固定NORMAL槽改回动态normal/recover。分支26_wheelleg，基础HEAD58947e8fabc4683b2025d42bda742b06c84dc566，加未提交改动；本次助手编写，独立复查待执行。H7未适配。

新接口 `chuanliantui-mjlab-wheelleg-recover-r7`，观测合同 `chuanliantui-huananhu-flat-35-78-dynamic-modes-v2`。旧接口为 [r6](chuanliantui-mjlab-wheelleg-recover-r6.md)。Actor35/Critic78/Action6不变。

## 动态模式定义

七槽顺序 `[normal,stair,slope,recover,jump,height_target,state_time]`：

- normal：`[1,0,0,0,0,0,0]`。
- recover：`[0,0,0,1,0,height_command*5,recover_seconds]`。
- actor模式区间28:35；恢复标志index31、目标index33、计时index34。critic同一状态区间32:39。
- reset按ground_start进入recover；普通初态进入normal。当前recover_pose_v2全为地面初态，因此新回合进入recover，计时归零。
- 连续0.5s满足：高度误差≤0.03m、roll/pitch≤15°、base接触≤1N、双轮各>5N、有效数值且无终止失败，切换normal。
- normal下base接触>1N、roll/pitch>40°或根高<0.16m，连续0.1s进入recover。
- 站高判断及恢复目标槽读取**当前每环境高度命令0.27～0.34m**，不使用旧r5固定0.22m。
- 每策略transition更新一次；查询actor/critic/HUD不改变计时；重复计算同一transition不重复推进。部分reset只清选中环境。

## 实现与其他配置

`reference_training/control_modes.py`管理切换；command持有状态，failure每策略步推进，reset_pose同步复位，huananhu_observations.mode供actor/critic及HUD读取。
模式是用户要求的本地扩展，参考工程原固定NORMAL定义保留历史记录。27奖励／高度门控／权重、PPO、动作顺序、PD、延迟、物理、资产、随机初态、域随机化、终止和严格恢复指标继续使用r6配置。模式不会直接加减奖励或改变终止规则。

r7新checkpoint保存新接口和模式阈值。r6已训练模型自动用固定mode的旧接口回放／CPU评估，不能直接完整resume到r7；r5回放继续保留。r6历史manifest保存于reference_training/r6_interface.json并核对机器人哈希，避免绕过兼容校验。没有转换或重新训练旧权重。

本次未启动新训练／TensorBoard、未提交Git。检查结果见 [模式恢复诊断](../diagnostics/20261010-mjlab-r7-dynamic-modes.md)。H7需同步模式语义和复位；固件及真机验证待部署负责人完成。
