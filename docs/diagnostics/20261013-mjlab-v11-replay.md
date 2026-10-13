# wheelleg v11 在本仓机器人环境的无训练回放

日期：2026-10-13。执行：Codex；交付后独立复跑：Claude（待执行）。

## 结论与适用范围

**v11能在本仓机器人上站高并保持平衡，但地面恢复没有覆盖A2500的全部原成功样本。** 0.30 m普通高度跟踪／平衡64/64、64/64，平均0.310539 m；地面分别58/64、58/64，平均0.296481 m。可用于本次已验证的仿真站立场景，并作为显式接口迁移后的续训候选起点；不能视为无取舍替代A2500或部署合格。

本报告只判断本机 GPU 上的零速度站立／地面恢复和高度响应，没有训练、模型更新、ONNX导出或部署。非零速度命令、H7部署与真机行为均**未评估**。

旧 A2500 两组1000轮任务的记录表明：两组最终模型已于今天14:26左右保存，训练预算已完成；本地过程检查无训练进程。本次未重启或追加训练、未终止远端任务，高位初态诊断不再运行。

## 接口门禁

模型复制前后SHA256均为 `62efa60cf3c0884d0abf4a2378d778371135a49c475ed3453ebec08136d22366`。完整checkpoint位于 `artifacts/v11-replay-20261013/models/model_999.pt`，仅载入确定性actor用于推理；不恢复或执行优化器。actor为35→256→128→64→6、ELU、无经验归一化／encoder；checkpoint iter999，累计4000更新。

wheelleg观测契约为 `chuanliantui-huananhu-flat-35-78-v1`，控制契约以 `sourcexml-spring90mm-v10` 结尾。本仓现有契约字符串为dynamic-modes-v2／legsfirst-v11，本次显式选择源动作顺序和固定NORMAL，不伪造新模型契约或调用已有模型续训加载器。

**源码纠正**：源动作顺序为`lf0,lf00,lfwheel,rf0,rf00,rfwheel`；但源`V14ObservationState.MOTOR_NAMES/ACTION_ORDER/basic/raw_sensors`已将位置、速度、上一步原始动作排成`lf0,lf00,rf0,rf00,lfwheel,rfwheel`。本仓观测采用相同顺序，不能把观测也改成左轮居中。只设置`cfg.actions["pd"].policy_joint_order=ACTUATED_JOINTS`，并关闭动态模式。

源码依据（均只读）：源`wheelleg_RL/src/wheelleg_rl/chuanliantui/{v14_observations,huananhu_observations,pd_action,huananhu_action,huananhu_env_cfg}.py`；本仓`mjlab_training/src/ct_mjlab/reference_training/`内对应文件与`control_modes.py`。原源码／资产指纹见`protected-before.json`，门禁数值证据见`interface-gate.json`。

35维逐通道对照（索引从0开始；名称按实现变量，固定模式的其余槽位只定义为0占位；双方对应相同物理关节、缩放和裁剪）：

| 索引 | wheelleg名称 | wheelleg缩放／裁剪 | 本仓回放名称 | 本仓回放缩放／裁剪 |
|---:|---|---|---|---|
| 0 | cmd_vx | 1; clamp ±100 | cmd_vx | 1; clamp ±100 |
| 1 | cmd_vy=0 | 1; clamp ±100 | cmd_vy=0 | 1; clamp ±100 |
| 2 | cmd_yaw_rate | 1; clamp ±100 | cmd_yaw_rate | 1; clamp ±100 |
| 3 | cmd_height | 5; clamp [0,1] | cmd_height | 5; clamp [0,1] |
| 4 | base_ang_vel_x | 0.5; clamp ±100 | base_ang_vel_x | 0.5; clamp ±100 |
| 5 | base_ang_vel_y | 0.5; clamp ±100 | base_ang_vel_y | 0.5; clamp ±100 |
| 6 | base_ang_vel_z | 0.5; clamp ±100 | base_ang_vel_z | 0.5; clamp ±100 |
| 7 | projected_gravity_x | 1; clamp ±100 | projected_gravity_x | 1; clamp ±100 |
| 8 | projected_gravity_y | 1; clamp ±100 | projected_gravity_y | 1; clamp ±100 |
| 9 | projected_gravity_z | 1; clamp ±100 | projected_gravity_z | 1; clamp ±100 |
| 10 | lf0_position_minus_nominal | 1; clamp ±100 | lf0_position_minus_nominal | 1; clamp ±100 |
| 11 | lf00_position_minus_nominal | 1; clamp ±100 | lf00_position_minus_nominal | 1; clamp ±100 |
| 12 | rf0_position_minus_nominal | 1; clamp ±100 | rf0_position_minus_nominal | 1; clamp ±100 |
| 13 | rf00_position_minus_nominal | 1; clamp ±100 | rf00_position_minus_nominal | 1; clamp ±100 |
| 14 | lfwheel_position=0 | 1; clamp ±100 | lfwheel_position=0 | 1; clamp ±100 |
| 15 | rfwheel_position=0 | 1; clamp ±100 | rfwheel_position=0 | 1; clamp ±100 |
| 16 | lf0_velocity | 0.1; clamp ±200 | lf0_velocity | 0.1; clamp ±200 |
| 17 | lf00_velocity | 0.1; clamp ±200 | lf00_velocity | 0.1; clamp ±200 |
| 18 | rf0_velocity | 0.1; clamp ±200 | rf0_velocity | 0.1; clamp ±200 |
| 19 | rf00_velocity | 0.1; clamp ±200 | rf00_velocity | 0.1; clamp ±200 |
| 20 | lfwheel_velocity | 0.1; clamp ±200 | lfwheel_velocity | 0.1; clamp ±200 |
| 21 | rfwheel_velocity | 0.1; clamp ±200 | rfwheel_velocity | 0.1; clamp ±200 |
| 22 | lf0_last_raw_action | 1; clamp ±100 | lf0_last_raw_action | 1; clamp ±100 |
| 23 | lf00_last_raw_action | 1; clamp ±100 | lf00_last_raw_action | 1; clamp ±100 |
| 24 | rf0_last_raw_action | 1; clamp ±100 | rf0_last_raw_action | 1; clamp ±100 |
| 25 | rf00_last_raw_action | 1; clamp ±100 | rf00_last_raw_action | 1; clamp ±100 |
| 26 | lfwheel_last_raw_action | 1; clamp ±100 | lfwheel_last_raw_action | 1; clamp ±100 |
| 27 | rfwheel_last_raw_action | 1; clamp ±100 | rfwheel_last_raw_action | 1; clamp ±100 |
| 28 | mode_normal | 1; fixed 1 | mode_normal | 1; fixed 1 |
| 29 | NORMAL固定占位槽1=0 | 1; fixed 0 | NORMAL固定占位槽1=0 | 1; fixed 0 |
| 30 | NORMAL固定占位槽2=0 | 1; fixed 0 | NORMAL固定占位槽2=0 | 1; fixed 0 |
| 31 | NORMAL固定占位槽3=0 | 1; fixed 0 | NORMAL固定占位槽3=0 | 1; fixed 0 |
| 32 | NORMAL固定占位槽4=0 | 1; fixed 0 | NORMAL固定占位槽4=0 | 1; fixed 0 |
| 33 | NORMAL固定占位槽5=0 | 1; fixed 0 | NORMAL固定占位槽5=0 | 1; fixed 0 |
| 34 | NORMAL固定占位槽6=0 | 1; fixed 0 | NORMAL固定占位槽6=0 | 1; fixed 0 |

表中通道按名义无噪声定义。训练级随机化在缩放前叠加原传感器均匀噪声：角速度±0.25、重力±0.05、六路关节位置±0.025（含原始位置固定为0的轮通道）、四腿速度±0.5、两轮速度±1；传感器四组延迟取5/10/15 ms并按2 ms物理步四舍五入为3/5/8步，腿／轮动作延迟取5/10 ms向上取整为3/5步。固定NORMAL及命令不加上述传感器噪声。

NORMAL在普通及地面初态均固定为`[1,0,0,0,0,0,0]`，高度只在通道3以h×5输入；不向模式槽33／34写恢复目标或时间。关节位置以站姿motor nominal作零点，轮位置强制0；角速度读取free-joint qvel，重力从qpos四元数逆旋转，电机速度直接读取qvel；上一步动作为原始actor输出，经观测±100裁剪，不先限为±1。

同一仿真状态及不同传感器输入（含裁剪边界）下，源／本仓观测逐项相等，最大绝对差值=0.0。四腿nominal角依次约−0.60472055、+0.60832417、+0.60472055、−0.60832417 rad，双方在1e−6容差内一致。

动作门禁在真实`apply_actions()`执行时读取目标局部变量和实际电机effort；以零输入为基线，逐维注入+1（8环境），检查其余五电机目标／力矩增量均为0：

| 动作维度 | 对应电机 | 目标变化 | 力矩变化 | 其余电机 |
|---:|---|---:|---:|---|
| 0 | lf0 | 0.500000 rad | 30.000000 N·m | 不变 |
| 1 | lf00 | 0.500000 rad | 30.000000 N·m | 不变 |
| 2 | lfwheel | 10.000000 rad/s | 1.000000 N·m | 不变 |
| 3 | rf0 | 0.500000 rad | 30.000004 N·m | 不变 |
| 4 | rf00 | 0.500000 rad | 30.000000 N·m | 不变 |
| 5 | rfwheel | 10.000000 rad/s | 1.000000 N·m | 不变 |

腿PD60/2、位置scale0.5、目标相对CAD reference限±3.14 rad／力矩限40 N·m；轮速度scale10、最大150 rad/s、速度增益0.1、力矩限3.9 N·m。物理步2 ms、策略步20 ms；地面初态保留前0.2 s电机力矩暂停。名义扫描无观测噪声、传感器／动作延迟、物理随机化或推扰。

## 本仓名义物理配对扫描

普通初态来自`artifacts/height-response-a2500-20261012/initial-normal.npz`；地面初态来自`artifacts/recovery-regression-paired-20261012/initial-ground.npz`。每个命令各64环境、20 s、seed42、零前向／偏航速度；每次逐项恢复qpos/qvel并核对origin及A2500参考的初态指纹。1000控制步，末250步（15.02～20.00 s）计量；成功要求末5秒至少95%时刻组合判据成立，全程无数值失败。

平衡采用验收v2：双轮真实对地接触，每轮≥10%重量且≥5 N，总承重80%～120%，非轮支撑≤5%，机身接触≤1 N，倾角≤15°，线速模≤0.15 m/s、角速模≤0.5 rad/s。高度跟踪在此基础上要求误差≤3 cm。膝激活比例来自原生MuJoCo joint-limit约束行，每膝单独统计末5秒；不是膝软奖励激活率。

| 初态 | 命令m | 末5s平均高度m | 高度跟踪/64 | 稳定平衡/64 | lf1限位激活 | rf1限位激活 | 数值失败 |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal | 0.20 | 0.206493 | 64 | 64 | 100.000% | 0.000% | 0 |
| normal | 0.22 | 0.222952 | 64 | 64 | 0.000% | 0.000% | 0 |
| normal | 0.25 | 0.253539 | 64 | 64 | 0.000% | 0.000% | 0 |
| normal | 0.27 | 0.274965 | 64 | 64 | 0.000% | 0.000% | 0 |
| normal | 0.30 | 0.310539 | 64 | 64 | 0.000% | 0.000% | 0 |
| normal | 0.34 | 0.352347 | 64 | 64 | 0.000% | 0.000% | 0 |
| ground | 0.20 | 0.203480 | 57 | 57 | 100.000% | 3.125% | 0 |
| ground | 0.22 | 0.218286 | 57 | 57 | 10.938% | 3.125% | 0 |
| ground | 0.25 | 0.245531 | 57 | 57 | 4.688% | 3.125% | 0 |
| ground | 0.27 | 0.264355 | 57 | 57 | 3.238% | 3.131% | 0 |
| ground | 0.30 | 0.296481 | 58 | 58 | 1.669% | 1.700% | 0 |
| ground | 0.34 | 0.336824 | 58 | 58 | 3.181% | 3.175% | 0 |

六命令普通最小二乘高度响应斜率（单位m/m，理想1），另报告全部高度均平衡的固定共同成功样本集合，避免各命令样本变化造成选择偏差：

| 初态 | 全64样本斜率 | 固定共同平衡数 | 固定共同平衡斜率 |
|---|---:|---:|---:|
| normal | 1.058486 | 64 | 1.0584856552304698 |
| ground | 0.962749 | 57 | 1.0584346103694355 |

## 与A2500的0.30 m对照

实际调用原`paired_acceptance`，协议与初态指纹门禁通过。两个模型没有续训关系，函数产生的`regressed_rejected`仅表示A原成功样本未全部保留的对照标签，**不称为续训退化**；新成功不能抵消丢失，逐样本集合照实列出。

| 初态 | A高度m | A跟踪/平衡 | v11高度m | v11跟踪/平衡 | A成功、v11失败样本 | v11新增平衡样本 |
|---|---:|---|---:|---|---|---|
| normal | 0.200775 | 0/64 | 0.310539 | 64/64 | [] | [] |
| ground | 0.199780 | 0/60 | 0.296481 | 58/58 | [4, 6, 7, 10, 14, 16] | [20, 42, 53, 57] |

平均高度包含全部64个数值有效样本；未平衡地面样本会拉低整批平均。0.30 m平衡成功集合的平均高度另为：

- normal：0.310539 m（64个平衡样本）。
- ground：0.310537 m（58个平衡样本）。

## wheelleg原环境复现

已执行。未发现源仓独立venv；已有本仓Python3.12/mjlab环境可直接导入wheelleg源码并加载其XML／姿态库，所以复用已安装运行时，没有安装依赖或向源仓写入文件。动作／观测使用wheelleg源实现，确定性actor同一模型；64环境、seed42、20s、随机普通／地面混合初态、零速固定高度，按存档`fixed_height.py`的t>5s首个存活回合汇总口径。它与本仓A配对初态和末5s验收不同，不能直接比较成功数；当前源源码相对于历史存档已有奖励／命令版本演变，固定命令覆盖、actor直接加载避免版本加载器拒绝，故不声称历史随机初态逐位复现。

| 命令m | 存档高度m | 本次源环境高度m | 存档±3cm时间占比 | 本次占比 | 本次摔倒/64 |
|---:|---:|---:|---:|---:|---:|
| 0.27 | 0.2725 | 0.272507 | 100% | 100.000% | 0 |
| 0.30 | 0.3085 | 0.308471 | 100% | 100.000% | 0 |
| 0.34 | 0.3502 | 0.350191 | 100% | 100.000% | 0 |

存档精确值来自`wheelleg_RL/logs/evaluation/v7init_kneelimit_v11_final_20261006_023925/fixed_height.log`，与`docs/ai/huananhu_experiment.md`的v11一节一致。源环境复现日志见`source-replay.log`、数组汇总见`source-replay.json`。

## 域随机化复核

至少一种初态0.30 m跟踪≥32/64，已触发256环境×3高度各20s。使用原训练startup/reset/interval事件、质量／COM／摩擦、motor增益／强度、气弹簧bias／阻尼、观测噪声／传感器及动作延迟和训练级推扰；固定NORMAL、0速度、固定高度，不改变本仓XML。以原failure或数值失败首次出现记摔倒，不重置／替换样本。表内5s前为t<5，5s起为t≥5；全20s及末5s高度／误差时间比例分母均为全部256样本，包含已摔倒样本，另与源存档“t>5s存活首回合”口径区别。

| 命令m | 地面初态数 | 5s前摔倒 | 5s起摔倒 | 总摔倒 | 全20s平均m | 末5s平均m | 全20s±3cm比例 | 末5s±3cm比例 | 数值失败 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.27 | 135 | 1 | 0 | 1 | 0.269853 | 0.271488 | 96.561% | 99.227% | 0 |
| 0.30 | 135 | 4 | 0 | 4 | 0.301007 | 0.303768 | 95.338% | 98.828% | 0 |
| 0.34 | 135 | 2 | 0 | 2 | 0.340573 | 0.346022 | 94.379% | 99.219% | 0 |

随机化v1在第一组首次物理失败后退出1：mjlab1.6要求`auto_reset=False`且已终止的world必须reset后才能再次step，源自沿用的既有robustness脚本连续步进写法。失败JSON／日志完整保留为`robustness-failed-v1.json/.log`，没有把该次启动判为完成。v2仅把原failure包装为监测项：每控制步仍调用一次相同原判据、保留其状态更新并记录首次失败，但对manager返回False以免触发强制reset；全程原物理状态连续步进、不复位、不替换失败样本，固定actor不受奖励值变化影响。未修改原failure、训练事件、XML或现有评估脚本。成功结果和日志为`robustness.json`／`robustness-v2.log`。

门禁／名义扫描／源环境复现实际执行版本保留为`eval-executed-v1.py`；随机化v2实际执行当前新增脚本，二者哈希与差异说明见`execution-provenance.json`。v2同时纠正描述性模式占位槽名称和后续源环境回放的ground-count记录时机；原门禁／源结果已保留raw-v1文件，数值观测、动作、高度、摔倒指标未改变。

## 动力学与模型交付边界

本仓机器人XML／资产未改。已知源／本仓差异保留：主动腿armature约0.015795／0，被动腿damping0.01／0.03（源主动腿为0），腿frictionloss0／0.015，轮damping/frictionloss0/0／0.002/0.001，六闭链connect solref第一项0.01／0.005。数值依据既有`20261010-mjlab-r12-reference-comparison.md`第三节；不得把这些差异的合并效果归因到单一参数。模型能在本仓稳定站高，也不证明非零速度、部署或真机已通过。

本报告合并填写模型交付／接口记录模板的本次适用项：接收人为用户，独立复查Claude待执行；机器人为本仓chuanliantui实体闭链；当前维护分支26_wheelleg，工作区已有大量改动，按用户限制未创建分支／提交／推送／PR，不用HEAD代替实际运行源码。完整模型仅只读复制，训练来源为源日志v11（v7基础3000+续训1000），本次无PPO／优化器恢复。部署仓信息、固件／ONNX、板端时序均不适用本次交付，H7仍旧接口待适配。运行源码／资产指纹为`protected-before.json`和`verification.json`，运行场景快照`nominal/scene.mjb`。

## 命令、日志与视频

所有WSL命令先写入.sh再执行；cwd在本批artifact目录，MUJOCO_LOG等自动产物不会写入仓库根或源仓。实际包装脚本：`gate.sh`（接口门禁）、`replay.sh`（名义扫描→源环境复现→随机化v1失败），随后`finish-v2.sh`只继续修正后的随机化和渲染，不重复已完成扫描或源环境回放。均set -euo pipefail，成功以输出JSON完整性及各阶段exit-code为准。

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'bash /mnt/d/IMCA/RL/Wheel-Legged-Gym/artifacts/v11-replay-20261013/gate.sh'
wsl -d Ubuntu-24.04 -- bash -lc 'bash /mnt/d/IMCA/RL/Wheel-Legged-Gym/artifacts/v11-replay-20261013/replay.sh'
wsl -d Ubuntu-24.04 -- bash -lc 'bash /mnt/d/IMCA/RL/Wheel-Legged-Gym/artifacts/v11-replay-20261013/finish-v2.sh'
wsl -d Ubuntu-24.04 -- bash -lc 'bash /mnt/d/IMCA/RL/Wheel-Legged-Gym/artifacts/v11-replay-20261013/verify.sh'
```

直接复跑时须使用新的名义输出目录，已有视频和评估目录禁止覆盖；`replay.sh`原样再次执行会因nominal已存在而停止。新脚本接口`--checkpoint/--output/--initial-normal/--initial-ground/--reference`完整命令见replay.sh。gate/source/robustness输出固定在本批artifact目录，复核前应保留原证据。

| 证据 | 路径（相对仓库） |
|---|---|
| 完整回放脚本 | scripts/agent/eval_chuanliantui_v11_replay.py |
| 门禁 | artifacts/v11-replay-20261013/interface-gate.json、gate.log |
| 12项扫描与逐条件比例／末5s高度／膝约束数组 | artifacts/v11-replay-20261013/nominal/evaluation.json、normal/ground-h*.json、nominal.log |
| 源环境复现 | artifacts/v11-replay-20261013/source-replay.json、source-replay.log |
| 随机化 | artifacts/v11-replay-20261013/robustness.json、robustness-v2.log |
| 普通样本0视频 | artifacts/v11-replay-20261013/nominal/comparison-normal.mp4 |
| 地面样本0视频 | artifacts/v11-replay-20261013/nominal/comparison-ground.mp4 |
| 渲染与末帧 | artifacts/v11-replay-20261013/render.log、nominal/render-record.json、nominal/comparison-final.png |
| 独立原始证据复核 | artifacts/v11-replay-20261013/verification.json |

视频直接渲染本机GPU回放记录的sample0 qpos，不重新积分；每份200帧／10fps／精确20秒，时间戳0～19.9s，画面叠加实际和目标高度0.30m。

独立复核：12项原始高度／膝约束数组、逐条件成功集合、斜率、A丢失样本与视频帧数／时长全部通过；262个既有源码／资产文件（含wheelleg只读src/assets）SHA256均保持。历史logs和deployment未写入。正式行为结论来自GPU回放，本次未声称单元测试或Isaac旧流程通过。

运行环境：{"device": "NVIDIA GeForce RTX 4060 Laptop GPU", "versions": {"torch": "2.14.0", "mjlab": "1.6.0", "mujoco": "3.11.0", "mujoco-warp": "3.11.0", "warp-lang": "1.17.0"}}。

名义回放执行时间：2026-10-13T15:35:32.547543+08:00 ～ 2026-10-13T15:45:17.811922+08:00。其他阶段执行时间见对应JSON。
