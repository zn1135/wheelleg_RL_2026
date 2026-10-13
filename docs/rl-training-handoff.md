# RL 训练侧协作交接

整理日期：2026-10-10。面向负责 H7 部署的协作者。

## 1. 分工与当前版本

用户负责 **机器人建模 → RL 训练 → 仿真验证 → 模型导出与交付**；H7 协作者负责 **板端观测 → 网络推理 → 电机控制 → 固件构建与真机验证**。双方共同维护接口和模型交付记录。

当前维护分支为 `26_wheelleg`。mjlab迁移已提交于 `0c8df707d874290c43429aed5da6797cc0959b28`，H7导入与本说明已提交于 `58947e8fabc4683b2025d42bda742b06c84dc566`。实体弹簧r2、随机初态r3和模式槽r4尚未提交，新增r5控制对齐也未提交，r6完整参考训练移植、r7动态模式、r8动作顺序恢复及r9关闭膝限位奖励惩罚亦未提交，新增r10恢复混合初态、r11恢复参考膝限位惩罚、r12关闭该项并完整续训、r13放宽速度奖励高度门控为5～10 cm，亦未提交，见 [当前r13接口](interfaces/chuanliantui-mjlab-wheelleg-recover-r13.md)。

两人各自克隆完整仓库，按负责目录分别提交并合入共同维护分支。观测、关节顺序、缩放、控制频率或模型格式变化时，双方共同核对接口记录。具体流程见 [协作指南](../CONTRIBUTING.md#两人维护分工)。

## 2. 训练侧代码地图

以下路径均相对仓库根目录；新串联腿任务名为 `chuanliantui`。

| 入口／文件 | 用户维护的内容 | 部署协作者需要了解的内容 |
|---|---|---|
| `scripts/mjlab.sh`、`mjlab_training/src/ct_mjlab/cli.py` | train / play / eval / export / check 入口 | 如何回放交付模型、生成参考输出 |
| `ct_mjlab/reference_training/` | 当前r13原生mjlab配置、PPO、27奖励、观测、控制、初态／域随机化；SOURCE_IMPORT.json来源指纹 | 当前部署预处理／动作／时序的权威参考 |
| `ct_mjlab/config.py`、`scut_v14.json` | 当前工厂路由及历史r5配置 | 核对模型对应版本 |
| `ct_mjlab/environment.py`、`recipe.py` | 旧r5 direct环境、观测和控制，仅历史回放 | 不作为当前训练／部署契约 |
| `ct_mjlab/asset.py`、`poses.json` | CAD 闭链资产接入、完整初态与名义姿态 | 实体关节定义；名义角不是硬件编码器零点 |
| `ct_mjlab/policy.py` | checkpoint、普通 actor、ONNX 导出及接口校验 | 网络输入输出、模型格式、哈希与误差记录 |
| `ct_mjlab/reference_training/evaluate.py`；历史 `cpu.py` | r6～r13 CPU同任务评估；r5独立MuJoCo回放 | 失败时停止／保留轨迹，不用自动reset隐藏失败 |
| `resources/robots/chuanliantui_new_1/`，新模型在 `mjlab/chuanliantui.xml` | URDF、网格、闭链与实体气弹簧物理模型 | 坐标、左右命名、轴向及安装几何；旧 `sim2sim/chuanliantui.xml` 属于历史模型 |
| `scripts/agent/`、`docs/diagnostics/` | 数值检查脚本和执行证据 | 区分已通过检查与待验证行为 |
| `wheel_legged_gym/`、旧 `sim2sim/` 脚本 | Isaac Gym 历史环境、模型回放与对照 | 仅用于对应旧模型，不能套用新接口 |

表中 `ct_mjlab/` 的完整目录为 `mjlab_training/src/ct_mjlab/`。新入口使用独立 Python 3.12 / mjlab 环境；旧入口使用 Isaac Gym / Python 3.8，两套依赖分开管理。历史训练产物在根目录 `logs/`，不得删除、移动或覆盖。

## 3. 当前训练方案

r9沿用r6完整移植的 `wheelleg_RL` 当前recover_pose_v2训练，并恢复动态normal/recover模式及四腿两轮动作顺序：原生mjlab管理器、27奖励原公式、其中膝限位奖励权重为0，其余26项权重及高度门控、PPO、观测、动作、随机化、终止和严格恢复指标。机器人资产／ref／硬限位／闭链缓存保留本仓实现。100%地面初态、其中50%机体扰动，随机目标0.27～0.34 m，速度命令零。详见 [当前说明](mjlab-chuanliantui.md)。

主要实现位于 `ct_mjlab/reference_training/`；旧 `environment.py`、`recipe.py`、`cpu.py`只用于r5历史回放和检查。物理2 ms、策略50 Hz，腿PD60/2、轮Kv0.1／目标±150 rad/s，动作和传感延迟按参考工程取整。气弹簧训练偏置±50 N／阻尼25～75，总推力0～150 N。没有虚拟关节。默认4096环境，每50轮保存；r6的4096环境／1000轮训练已完成，见 [r6运行记录](diagnostics/20261010-mjlab-r6-formal-training.md)；r9训练状态另见 [r9运行记录](diagnostics/20261010-mjlab-r9-formal-training.md)。

## 4. H7 必须对齐的策略接口

接口：`chuanliantui-mjlab-wheelleg-recover-r9`。ONNX float32 `[1,35]→[1,6]`、opset13；图内只有actor，预处理／PD在图外。

**动作顺序** `[lf0,lf00,rf0,rf00,lfwheel,rfwheel]`；**观测关节／上一拍动作槽顺序** `[lf0,lf00,rf0,rf00,lfwheel,rfwheel]`。模式槽恢复动态normal/recover；恢复时七槽为 `[0,0,0,1,0,height_command*5,t]`。命令高度×5；位置相对本机器人名义角。Critic材质槽改为MuJoCo sliding/torsional/rolling。

当前奖励版本与兼容边界见 [r9接口](interfaces/chuanliantui-mjlab-wheelleg-recover-r9.md)，动作／模式见 [r8接口](interfaces/chuanliantui-mjlab-wheelleg-recover-r8.md)，其他逐槽、增益、ref、时序、延迟和限幅沿用 [r6接口](interfaces/chuanliantui-mjlab-wheelleg-recover-r6.md)。35/78/6形状不变不表示旧checkpoint或固件兼容。旧r5/r6/r7/r8模型按各自旧动作顺序、模式及膝惩罚权重回放，拒绝完整续训到r9；外部模型也不绕过本机器人manifest。

H7固件／编码器／极性／IMU／500 Hz内环和行为验收由部署负责人适配，本轮没有修改deployment/H7_RL。

## 5. 双方交付清单

| RL 用户交给 H7 协作者 | H7 协作者反馈给 RL 用户 |
|---|---|
| 训练代码完整提交 SHA、实际运行配置、机器人资产版本 | 固件提交 SHA、硬件版本、实际构建配置 |
| checkpoint、ONNX、导出 JSON 与 SHA-256 | 网络转换结果、板端推理结果与参考输出误差 |
| 接口字段、名义姿态、关节顺序、控制参数 | 实物电机对应关系、极性／零点核对、IMU 坐标变换 |
| mjlab 回放、CPU MuJoCo 起立／静站结果和轨迹 | 实际采样／推理／执行频率、链路延迟和真机验证结果 |

每次交付填写 [模型交付模板](templates/model-handoff.md)，在 `docs/model-deliveries/` 保存实际记录。尚未完成的部署适配在 [r9接口记录](interfaces/chuanliantui-mjlab-wheelleg-recover-r9.md) 中明确标注；旧记录保留各自检查范围。

## 6. 当前完成程度与后续顺序

**已完成：** mjlab代码迁移、实体接口、CAD左右命名修正、姿态缓存及r2实体气弹簧修正；r2的CPU/GPU有限物理、奖励数值对照、PPO runner构造、随机未训练actor的ONNX检查见 [实体气弹簧检查](diagnostics/20261009-mjlab-physical-spring.md)。此前 [迁移检查记录](diagnostics/20261009-mjlab-migration.md) 属于r1。

**可视初训：** 已按用户授权运行20环境实时训练，完成29次PPO更新，关窗保存 `model_28.pt` 后正常退出，见 [运行记录](diagnostics/20261009-mjlab-train20-preview.md)。这是初训模型，尚未行为验收或交付。

**随机初态训练：** 随后按用户要求加入华南虎式初态，完成20环境可视预检并在新的r3 run开启20环境训练，见 [随机初态记录](diagnostics/20261009-mjlab-random-reset.md)。r2模型保留，不自动resume到r3。

**模式槽：** r4已接入正常／恢复模式和辅助目标／计时，窗口显示实际槽值；检查及20环境新训练记录见 [模式记录](diagnostics/20261009-mjlab-control-modes.md)。旧r3保存到 `model_445.pt`，不直接续训到r4。

**待完成：** r9训练及完成后的新策略起立／30秒静站行为验收、实际训练策略导出、H7接口适配和板端／真机验证。有限检查和可视初训不代表策略行为通过。

导入 H7 的网络仍为 `model_15000_h723`，输入 25 维当前观测和 125 维历史，策略 100 Hz，包含旧 encoder。它不兼容新接口；不能直接替换成新 ONNX。eIDE 缺失引用、固件源码指纹过期和历史频率文档差异见 [H7 导入说明](../deployment/H7_RL/README.md#导入时发现的配置问题)，由部署负责人在实际构建前核对。

后续按 **可视预检 → 训练 → 原生／CPU 行为验证 → 模型交付 → H7 适配与验证** 推进。命令集中维护在 [mjlab 说明](mjlab-chuanliantui.md#可执行命令)，训练执行状态见上文对应运行记录；尚未构建固件。

**r5控制对齐：** 腿PD、ref零点和2 ms×10物理时序已更新，旧r4模型必须保留原run源码回放，不能直接续训。H7电机编码器映射和500 Hz控制待部署负责人适配。检查限制见 [r5诊断](diagnostics/20261009-mjlab-reference-control.md)。

**r6完整训练移植（2026-10-10）：** 训练参数与参考当前工程逐项核对；有限GPU检查、checkpoint／ONNX通过，详情见 [r6诊断](diagnostics/20261010-mjlab-reference-training.md)。r5的4096环境／1000轮训练已完成，model_999回放平均站高0.168 m，未通过0.22 m高度验收；旧模型保留。r6的4096环境／1000轮训练已完成，见 [运行记录](diagnostics/20261010-mjlab-r6-formal-training.md)。

**r7恢复动态mode：** 用户要求恢复normal/recover；切换判据和目标跟随当前随机高度，历史接口见 [r7](interfaces/chuanliantui-mjlab-wheelleg-recover-r7.md)。r6旧模型保留固定mode回放，不完整resume到r7。本次只改mode并有限检查，尚未训练r7。

**r8恢复动作顺序：** 当前策略输出、观测关节及上一拍动作均为四腿两轮。六槽具名电机控制、动作历史奖励、GPU有限检查和ONNX通过，见 [r8诊断](diagnostics/20261010-mjlab-r8-action-order.md)。r6/r7历史推理保留原顺序，拒绝直接完整resume到r8；尚未启动新训练。此前建议的奖励／课程修改未实施。

**r9关闭膝限位惩罚：** 权重-0.5→0，原硬限位保持；其他26项奖励、动态模式、四腿两轮输出和PPO沿用r8。预检见 [诊断](diagnostics/20261010-mjlab-r9-no-knee-limit.md)，用户授权4096环境／1000轮新训练，状态见 [运行记录](diagnostics/20261010-mjlab-r9-formal-training.md)。
