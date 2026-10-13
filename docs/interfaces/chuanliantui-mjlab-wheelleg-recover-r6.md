# 串联腿 wheelleg_RL 训练完整移植 r6

2026-10-10，按用户要求移植本机 `mjlab/wheelleg_RL` 当前 `Chuanliantui-Huananhu-Flat-v0` / `recover_pose_v2` 训练相关代码与参数，保留本仓机器人资产、ref、硬限位、闭链求解缓存。

| 字段 | 内容 |
|---|---|
| 训练仓／分支／基础提交 | wheelleg_RL_2026／26_wheelleg／58947e8fabc4683b2025d42bda742b06c84dc566，加未提交改动 |
| 参考版本 | mjlab_WL main，36d0d2fbf424276e082e7c3b9902b49879d894e4，加既有未提交改动 |
| 来源指纹 | `mjlab_training/src/ct_mjlab/reference_training/SOURCE_IMPORT.json`，逐文件 SHA256，不能仅以 HEAD 复现 |
| 新接口 | chuanliantui-mjlab-wheelleg-recover-r6 |
| 旧接口 | [r5](chuanliantui-mjlab-huananhu-v14-flat-standup-r5.md)；35/78/6 相同但动作顺序和输入语义不同 |
| 权威实现 | reference_training 的 huananhu_env_cfg/action/observations/mdp/rewards/randomization/runner；model.py 为本机器人适配 |
| 作者／独立复查 | 本次助手／待执行 |
| H7 | 内置 deployment/H7_RL 未修改、未适配，不可直接部署 |

## 观测与动作

Actor35 / Critic78 / Action6，50 Hz，物理2 ms×10。实体控制无虚拟关节。

**动作顺序**改为参考工程的 `[lf0, lf00, lfwheel, rf0, rf00, rfwheel]`。
**位置、速度和上一拍动作观测槽顺序**仍为 `[lf0, lf00, rf0, rf00, lfwheel, rfwheel]`，观测动作通过 `(0,1,3,4,2,5)` 重排；两者不可混用。

| Actor索引 | 内容 |
|---|---|
| 0:3 | `[vx,0,yaw]`，当前 recover 阶段均零 |
| 3:4 | 随机高度命令0.27～0.34 m，×5；每5～15 s重采样 |
| 4:7 | 机体系角速度×0.5，限幅±100 |
| 7:10 | 投影重力，限幅±100 |
| 10:16 | 相对本机器人的名义角，轮角置零后仍可加位置噪声，限幅±100 |
| 16:22 | 六关节速度×0.1，限幅±200 |
| 22:28 | 按观测关节顺序重排的上一拍请求动作，限幅±100 |
| 28:35 | **固定 `[1,0,0,0,0,0,0]`**，完全沿用参考主线；不是动态 recover 标志 |

Critic：干净core28、COM线速度3、根高度×5、mode7、Kp6、Kd6、qfrc_actuator6×0.05、传感延迟4、动作延迟2、两轮相对COM速度6、轮历史接触2、base质量倍率1、**MuJoCo sliding/torsional/rolling 材质6**。奖励线速度使用 root_link_lin_vel_b，关节加速度使用 MuJoCo qacc，均按参考实现。

腿位置 `nominal+0.5 action`，限幅各电机ref±3.14 rad；Kp60/Kd2；40 Nm。
轮速度 `10 action`，限幅 **±150 rad/s**，Kv0.1，3.9 Nm。策略动作不全局裁剪。
这是训练物理目标限幅；行为验收仍不得将速度命令默认设到旧安全包线之外。
名义角和ref由本仓 `poses.json` / MJCF读取，不覆盖为另一份资产的数值。

## 完整训练参数

- 原生 mjlab ManagerBasedRlEnv、同版本 RSL-RL PPO，管理器步进顺序与参考一致。
- 27项奖励原公式原权重：线速度倒数二次函数；速度正奖励及平方误差惩罚乘高度门控（误差≤3 cm全额、≥6 cm零、中间线性）；平方跟踪权重均−0.1；静止惩罚−2；被动膝限位软区0.15 rad、权重−0.5。其余逐项见 huananhu_rewards.py。RewardManager仅乘一次0.02 s。
- Actor/Critic256/128/64 ELU、不归一化；高斯std1；24步；5epoch/4batch；lr1e−4/adaptive；KL0.01；clip0.2；entropy0.005；value4；gamma0.99；lambda0.95；grad1；Adam。默认20000轮，**每50轮保存**；CLI可覆盖轮数，默认4096环境。
- 落地初态100%；其中50%根姿态不扰动、50%roll/pitch±5°、yaw±π、请求高度偏移0～1 cm；基准根高0.25 m，按碰撞界提升到至少2 mm间隙。左右腿各60%正摆／40%负摆；独立闭链表；腿速度±π/2、轮角±2π、轮速±50。保留本机器人表和对应坐标转换。
- 前0.2 s落地电机零力矩；气弹簧始终作用。关闭落地接触接管／站稳门控／分段推扰。速度命令全零，关闭heading/dash/spin当前阶段覆盖。
- 动作延迟5/10 ms向上取整为3/5子步（6/10 ms）；传感5/10/15 ms按源舍入为3/5/8子步（6/10/16 ms）；reset清各环境历史。
- 噪声角速±0.25、重力±0.05、位置±0.025、腿速度±0.5、轮速度±1；critic无噪声；观测查询不重采样。
- mass、inertia、COM、材质64桶、关节干/粘摩擦、增益0.75～1.25、腿强度0.8～1.1、轮强度0.9～1.1、独立5～10 s速度推扰及机体系持续外力按原代码。气弹簧偏置±50 N，阻尼25～75 N·s/m；参数reset间隔720策略步。弹性及总推力各限0～150 N。
- base接触>1 N或roll/pitch>40°连续20策略步失败；qpos/qvel非有限、速度超数值门槛立即失败；20 s timeout与失败区分。撤下r5自定义raw观测>120终止。
- 严格恢复指标：双轮各>5 N、base≤1 N、roll/pitch≤15°、命令高度误差≤3 cm、无数值／终止失败，连续0.5 s。该指标不改变mode输入。
- 源runner保留累计轮数和训练阶段记录；本仓另核对完整机器人/训练manifest。新模型需要重新训练；不自动转换旧r5或外部checkpoint。

## 保留的机器人差异与协作边界

本仓MJCF、mesh、惯量名义值、实体几何、硬限位、ref、名义角及已求解姿态表保留，connect solref仍0.005/1；本次不修改机器人资产。源模型adapter通过名称绑定执行器／关节；离线表只加载本仓哈希匹配缓存。源码已复制到本仓，不从外部 wheelleg_RL 运行时加载训练代码。CLI输出路径仍为本仓logs，窗口默认按本仓规定不开启；这些入口差异不改变训练参数。

旧r5 model_999由CLI自动选择旧direct/CPU环境回放，拒绝续训到r6。r6与r5不能只看张量形状判断兼容。
H7需核对新的动作顺序、观测动作重排、常量mode、高度命令、原生材质语义及轮目标限幅；板端编码器／极性／时序与行为验收待部署负责人完成。

验证与参数对照见 [完整移植诊断](../diagnostics/20261010-mjlab-reference-training.md)。用户随后授权4096环境／1000轮训练，已启动，见 [训练记录](../diagnostics/20261010-mjlab-r6-formal-training.md)；尚未完成或交付新学习权重。
