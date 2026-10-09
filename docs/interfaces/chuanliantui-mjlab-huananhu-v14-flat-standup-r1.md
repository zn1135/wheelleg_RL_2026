# 串联腿 mjlab 实体关节接口

| 项目 | 内容 |
|---|---|
| 日期、实施、复查 | 2026-10-09，Codex 实施；独立复查未执行 |
| 机器人、任务、机构 | chuanliantui_new_1 CAD 闭链，`chuanliantui`；硬件版本待核实 |
| 旧接口 | `chuanliantui-25d-100hz-r1`，见 [历史契约](../deployment-contract.md) |
| 新接口 | `chuanliantui-mjlab-huananhu-v14-flat-standup-r1` |
| 训练仓、分支、基础提交 | zn1135/wheelleg_RL_2026，`26_wheelleg`，`b4aea50f2b75a730eff0d6a7d993454e98c3be51` 加未提交迁移代码；差异源码标识见 [检查记录](../diagnostics/20261009-mjlab-migration.md) |
| 部署仓 | H7_RL 本轮未修改；目标版本、负责人待核实 |
| 权威实现 | `ct_mjlab.config.manifest`、`Recipe`、`ChuanliantuiEnv`、`CpuPlayback`、`policy.check_manifest` |

## 定义与差异

| 项目 | 旧接口 | 新接口 |
|---|---|---|
| 观测 | actor25、历史125、critic65（平地） | actor35、critic78；[字段表及缩放](../mjlab-chuanliantui.md#观测契约) |
| 历史及初始化 | encoder 五帧 FIFO | 无 encoder；四路状态和两路动作物理子步延迟；reset 清所选环境历史、动作差分和失败计数 |
| 坐标、命令 | +x 前进，速度／偏航速度／高度 | +x 前进，+y 左；vx/vy/wz 恒零，高度 .22 m，ctrl mode `[1,0,0,0,0,0,0]`；尚无真机 IMU 变换适配 |
| 关节 | 原虚拟通道物理顺序，更名后 `[rf0,rf1,rfwheel,lf0,lf1,lfwheel]` | `[lf0,lf00,rf0,rf00,lfwheel,rfwheel]` 六个实体电机；名义角约 `[.06,-.059270409,-.06,.059270409,0,0]`，准确值在 `poses.json` |
| 角度、速度 | 虚拟膝映射及差分速度 | 直接读取实体角和 native qvel；位置不额外回绕，加速度按最后物理子步速度差分 |
| 动作 | 串联 PD，闭链通过虚功映射 | 四腿位置目标 `nominal+.5*a` 裁剪 ±π；两轮速度目标 `10*a` 裁剪 ±20；Kp10/Kd1、轮Kv.1；固定限矩40/3.9 Nm，输出倍率在裁剪前施加 |
| 时序 | 2 ms 内环、100 Hz 策略 | 5 ms 内环、50 Hz 策略；状态延迟 `{1,2,3}` 子步，动作延迟 `{1,2}` 子步 |
| 初态与停止 | 后摆 .15 m，接地门控等旧起立逻辑 | 后摆 .15 m、完整闭链关节缓存、根速度零；前 .2 s 电机零力矩；普通失败连续20策略拍，数值异常立即停止，999拍超时 |
| 机构及弹簧 | 串联代理或旧闭链适配 | 真实 CAD 闭链四销轴 connect；每侧150 N spatial tendon，从资产端点随姿态施力，无虚拟关节控制 |
| 模型 | ActorCriticSequence checkpoint，不能只导 actor | RSL-RL 普通 PPO checkpoint，包含 actor/critic；导出 float32 `observation[1,35] → action[1,6]`，opset13，预处理在图外 |

## 兼容性与同步

旧 checkpoint/ONNX 不兼容，需要重新训练。加载必须校验接口、源配置、资产、姿态缓存、控制参数和维度；拒绝旧模型。旧资产只更名实体及引用，历史数字通道保持物理含义；历史回放应配套当时代码及记录解释名称。

| 仓库 | 待同步内容 | PR／版本 | 负责人 | 状态 |
|---|---|---|---|---|
| 训练仓 | 新训练、native/CPU 回放、ONNX 导出及实体接口 | 未提交，无 PR | 用户后续采纳 | 移植完成，未训练 |
| H7_RL | 35维观测、实体顺序与零点、50 Hz 策略、控制目标／延迟；实际同步须另行核对硬件 | 待核实 | 待核实 | 未适配 |

本轮无训练模型交付，故不创建已交付模型记录。不得直接替换旧 H7 网络；先训练及回放验收，再同步部署接口。历史权重和日志未修改。

## 验证依据

见 [2026-10-09 检查记录](../diagnostics/20261009-mjlab-migration.md)。源奖励数值对照、CPU/GPU 有限闭链物理、PPO runner 构造、随机 actor ONNX 图及数值检查通过；未调用 `learn()`。起立／静站行为、板端时序、真机行为均未执行；未训练策略不能用于这些验收。
