# 独立 mjlab 原地稳站任务

辅助脚本已按用途迁移；下文固定旧实验保留其历史配方，旧路径查[迁移表](../scripts/agent/MIGRATION.md)，归档复核不代表当前版本验收。


`chuanliantui_stand` 从零学习固定机体高度 0.27 m、零前进速度、零偏航速度下的静站、抗推与倒地恢复。受扰后允许停在新位置；起立与恢复期间允许腿运动，站稳后的连续20秒内，每个500Hz采样时刻四个腿电机关节各自绝对角速度≤0.5 rad/s，机体平面速度≤0.05 m/s。

## 配方与接口

新配方先用原噪声20%、相对名义物理/电机随机变化量25%训练，再按批次评估逐步恢复。正式随机化验收仍用原完整噪声与物理随机化，速度v2判据和种子不变。新任务网络使用IEEE FP32，奖励数学与权重保持。新授权覆盖下文历史配方中“观测、噪声和随机化全部保持”的描述。

2026-10-25按用户最新要求，新任务恢复39维actor、82维critic和6维动作，第15、16维保留原左右轮角零占位及原噪声处理，不追加前向/侧向平面速度；见[39维轮角占位接口](interfaces/20261025-stand-wheel-placeholders-39.md)。此前r25/r26的[41维稳站接口](interfaces/20261025-stand-planar-velocity-41.md)、模型与报告保留为历史，不能直接混用。原 `chuanliantui` 保持39/82/6，名义PD与参考角不改。物理／PD为2 ms（500 Hz），策略为10 ms（100 Hz）。共享XML `resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml` 不修改；创建任务时SHA256为 `cac8061de2850602bfd1421fc856bf217c4e5040a11c8f5d33cf1abe61d5f0cf`。

任务代码位于 `mjlab_training/src/ct_mjlab/envs/chuanliantui_stand/`。固定命令在复位、重采样和恢复时保持。每次全体或部分环境复位独立按50%概率选择近站姿、50%选择原任务地面初态；小批抽样不保证数量恰好一半。近站姿仍通过连杆闭合求解，根横滚／俯仰±3°、最低碰撞点留2 mm、初速度零。地面分支直接调用原 `multimode_mdp.reset_pose → huananhu_poses.sample_reset_state`，保留原闭链姿态库、根高度0.25 m、水平根姿态及腿电机±π/2、轮±50 rad/s的初速度采样，并沿用原0.2秒零力矩预备。原库允许最多2 mm穿地，新任务用该库的碰撞最低点界补足2 mm间隙，必要时仅上移根高度不超过约4 mm，关节角和所有速度不改；旧任务复位实现不变。复位姿态不替换PD或观测的名义参考角。保留原质量、摩擦、电机、弹簧、腿角偏置、噪声和时延随机化。

训练回合40秒，当前课程按每批完整评估结果在批次间推进，不再仅按累计轮数自动升级。新训练及旧课程本任务模型迁入从阶段0开始，续训默认恢复checkpoint实际阶段，可用`train --task chuanliantui_stand --stand-curriculum-stage 0|1|2`显式选择；阶段分别为：无外推；50%静站／50%弱水平速度扰动；50%静站／25%强水平扰动／25%翻倒扰动。水平两分量范围分别为±0.3／±0.6 m/s；翻倒为12～16 rad/s。每回合最多一次，在5～10秒且满足严格站稳资格后施扰；没有持续外力。保留连续倒地2秒终止和原恢复状态。课程版本与进度随checkpoint保存。

原奖励保留，去掉过窄的旧位置／速度指数保持项，初版增加低速倒数奖励及每五个物理子步四腿真实速度均方的静止奖励；恢复阶段不压制必要腿部运动。当前r21正常站立时恢复r07用过的软关节姿态奖励，目标由独立名义模型求解至0.27米、闭链合法且重心位于轮轴上方；它只是训练奖励，不替换PD参考角，不写关节状态，起立、恢复、倒地及空中时关闭。r19～r20曾试验使用实际随机化重心平衡奖励，但完整评估未保住原有静站能力，历史配方和轨迹全部保留；不能据此断言重心奖励是唯一原因。原合法近站复位姿态不变。静轮项在r20恢复为r07的有界奖励：`1/(1+五物理子步两轮平均角速度平方/0.05²)`、权重1；正常双轮支撑时启用，地面起立、受扰恢复和空中阶段豁免，仅改变训练奖励。r08～r19使用过静轮线性成本，历史配方保留。配方具体版本、尺度、调整依据与训练成绩记在[项目实验记录](ai/project-log.md)和对应实验目录。

地面起立及中途倒地都会锁存奖励豁免，静腿、静轮、站姿及低速保持项须等身体满足支撑／高度／姿态／低速条件连续0.5秒后才重新启用；资格不读取腿速，不要求腿先停止。该锁存只控制奖励，不改变控制器、动作或终止规则。

r10候选新增站稳后的位置漂移惩罚：身体连续稳定0.5秒后只锁定一次位置，距离按米计、权重−40、最多计0.1m（单项下限−4）。暂时速度增加或位置超限不会重锁；仅复位、实际施扰或倒地清除，受扰恢复后允许锁定新位置。恢复期间该项同样豁免。恢复朝向改用权重2的负机体重力Z分量，正立为+2、倒立为−2；原四个重力XY朝向项在恢复期间关闭、正常站立时原式原权重保留。此候选只改变新任务奖励，是否采纳以固定集及历史成功回归为准。

模型配方与旧multimode分开，首次不读取旧模型初始化环境变量，`--resume`只恢复本任务模型。日志目录为 `logs/mjlab/chuanliantui_stand_v1/`。旧任务的 `eval`、独立CPU及H7不能直接代表新任务验收。

## 检查、训练与回放

2026-10-25起，新任务动作缓冲改为每回合、每环境腿／轮独立随机延迟1或2个完整策略步，训练与play均使用；不使用固定秒数。配置只指定整数`action_delay_policy_steps=(1,2)`，原500Hz动作队列按`策略步数×decimation`得到物理子步数，因此当前100Hz策略下分别延后5或10个物理子步。部分复位清空该环境的旧动作并重新采样，不修改其他环境。PD公式、500/100Hz、观测噪声／观测时延、控制参考角、限幅和物理随机化保持。原`chuanliantui`继续使用原6/8/10ms训练随机延迟及play固定实测等待。新配方`stand-r23-policy-step-delay`保留r22奖励；旧模型只作为本任务续训起点，旧报告保留，不能把旧时延成绩当新时延复测成绩。新评估逐样本和轨迹保存实际腿／轮缓冲步数；验收速度与20秒等门槛不变。

所有命令在仓库根目录，用独立mjlab环境运行。有限检查不会训练。

```bash
mjlab_training/.venv/bin/python -m scripts.agent.stand.check_chuanliantui_stand \
  --output artifacts/<新目录>/finite-check.json
scripts/mjlab.sh check --task chuanliantui_stand --num-envs 4 --steps 64

# 新任务改变复位分布，先用20环境观察初态；无checkpoint时为零动作预检。
scripts/mjlab.sh play --task chuanliantui_stand --num-envs 20 \
  --viewer viser --viser-port 8093

# 首批从零学习；后续只用本任务checkpoint续训。
scripts/mjlab.sh train --task chuanliantui_stand --num-envs 2048 \
  --iterations 1000 --run-name stand-r01
mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_stand_v1 --port 8080

stand_checkpoint="logs/mjlab/chuanliantui_stand_v1/<实际run>/model_<编号>.pt"
scripts/mjlab.sh train --task chuanliantui_stand --num-envs 2048 \
  --iterations 1000 --run-name stand-r02 --resume "$stand_checkpoint"
mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_stand_v1 --port 8080

scripts/mjlab.sh play --task chuanliantui_stand --num-envs 1 \
  --checkpoint "$stand_checkpoint"
```

有限检查成功：XML与控制契约保持，39/82/6维度正确、500/100 Hz，数值有限、命令固定，模型和课程正确续载；原地面同种子关节角及全部速度逐值一致、只修正根高度间隙，部分复位混合采样与其他环境隔离正确。预检成功：20个实际混合初态连杆闭合、限位正确且无穿地。可用`render_chuanliantui_stand.py --precheck --output artifacts/<新预检目录>`保存和查看20环境图及状态；初态几何通过不等于现模型已经学会地面起立。训练成功启动：出现新的 `Training output`、配置快照、checkpoint和TensorBoard事件。显存不足可将2048降为1024，频率不变。

## 行为验收

使用专用入口，报告与轨迹写入新目录，不覆盖旧结果：

```bash
mjlab_training/.venv/bin/python -m scripts.agent.stand.assess_chuanliantui_stand \
  --checkpoint "$stand_checkpoint" --suite all --profile both --seeds 42,43 \
  --output artifacts/<新目录>/acceptance.json
```

名义参数与训练随机化分别统计；固定验收入口显式固定为原近站初态，训练的50%地面分支不会混入或替换这些样本。固定集通过后用44、45独立复核，失败样本保留到回归集合。无推扰最迟5秒稳定；单次推扰后最多5秒重新稳定，在新位置只锁定一次测量起点，此后完整观察20秒。测量中失败不能重新锁位置、重新计时或通过自动复位清除。

| 指标 | 同时满足的条件 |
|---|---|
| 连续观察 | 20秒，完整10000个2 ms采样间隔 |
| 机体平面速度 | 每个物理采样时刻`norm(vx,vy)`≤0.05 m/s |
| 四腿电机关节角速度 | `lf0/lf00/rf0/rf00`每个物理采样时刻各自绝对值≤0.5 rad/s |
| 最大倾角／高度误差 | 机体Z轴相对世界竖直方向的夹角≤25°／高度误差≤3 cm |
| 双轮承重时间占比 | 至少95%；每轮>5 N、双轮合计至少70%整机重量 |
| 机身接触占比 | 不超过1%；接触阈值1 N |
| 跌倒、重置、非有限值 | 均不得发生 |

当前验收版本`stand-speed-v2`由用户于2026-10-24明确修改：速度取500 Hz原始qvel，任一2 ms采样超速即失败，不用均方、子步平均、百分位数或动作指令替代。机体速度读取根自由关节世界XY平移速度的模，腿速按四个实体电机关节分别计算绝对峰值。旧2°角峰峰值和5cm位移仅报告、不再拒绝；旧报告和失败状态不改写，旧轨迹提前截断时必须完整重测。另记录角摆幅、摆动主频和髋轮连线角／腿长作为诊断。平移覆盖八方向及两档幅度、对角分量极值，翻倒覆盖八方向12／16 rad/s。实际送达、实际倒地和恢复分别计数；未倒地只能证明抗扰，不能计入起立恢复成功。

固定集与独立复核均通过，且存在实际跌倒后恢复成功的样本，才报告本任务mjlab验收通过；训练奖励上升、自检通过、能够导出均不替代该结论。

地面起立另用下方入口报告，记录起立资格、实际机身接地及其后的20秒保持，始终不替代原固定集。其轨迹保存0秒初姿与全部初速度、500 Hz电机关节和机体指标，以及100 Hz完整qpos；原速视频从0秒开始，失败时截到失败时刻，不把复位当起立。

```bash
mjlab_training/.venv/bin/python -m scripts.agent.stand.assess_ground_chuanliantui_stand \
  --checkpoint "$stand_checkpoint" --profile both --seeds 42,43 \
  --output artifacts/<新地面报告>/ground.json
```

## 分析与录制

固定集通过后，用同一checkpoint将`--seeds`改为`44,45`，输出到另一个新路径。两份报告分别代表其选择的样本矩阵，必须合并核对全部覆盖；失败样本纳入后续回归。

```bash
mjlab_training/.venv/bin/python -m scripts.agent.stand.analyze_chuanliantui_stand \
  --run logs/mjlab/chuanliantui_stand_v1/<实际run> \
  --assessment artifacts/<固定集>/acceptance.json artifacts/<复核集>/acceptance.json \
  --output artifacts/<新分析目录>

# scene.mjb和trajectory.npz来自报告内同一个batch；env-index对应样本在该batch的位置。
mjlab_training/.venv/bin/python -m scripts.agent.stand.render_chuanliantui_stand \
  --recording artifacts/<评估目录>/<batch>/trajectory.npz \
  --scene artifacts/<评估目录>/<batch>/scene.mjb --env-index 0 \
  --output artifacts/<新视频目录>
```

分析输出训练奖励／回合曲线、逐样本CSV、完整500 Hz代表轨迹图和汇总JSON。加入`--baseline <旧报告.json>`比较已有成功样本；旧成功缺失或失败均判退化，实际扰动向量、初态或关键验收源码不一致时不宣称已验证无退化。未形成20秒窗口的样本保留失败状态与缺失指标，不按零漂移／零腿摆统计。视频按已录制qpos离线渲染，不进行新的物理仿真，也不替代500 Hz验收；存在同批逐样本报告时，按该样本结束时间截断并标注PASSED或FAILED，避免把失败后为继续其他环境而发生的复位呈现成恢复。

需要分析提前失败后的振荡或慢漂移时，使用`mjlab_training/.venv/bin/python -m scripts.agent.stand.record_chuanliantui_stand --checkpoint "$stand_checkpoint" --profile nominal --seconds 30 --num-envs 4 --output artifacts/<新诊断目录>`。它完整保存500 Hz实际状态、100 Hz动作和恢复/终止标志，不给出验收通过结论；其中较晚或较短的片段不能替代原固定验收窗口。

## 训练结束后自动续接当前代理

`wait_training_completion.py`作为当前代理的一次等待工具调用，绑定训练进程；进程结束时返回结果，由当前会话直接继续分析。等待期间只检查进程，不调用模型、不启动Claude或其他CLI、不建立周期定时任务。调用需保持有效，不能把脚本放到后台后声称已经向空闲会话发送通知。

```bash
mjlab_training/.venv/bin/python -m scripts.agent.maintenance.wait_training_completion \
  --pid <训练Python进程PID> --output artifacts/<新的等待目录>
```

工具绑定PID、系统启动标识和进程启动时间，识别进程退出或PID复用；不会停止训练。目录内`status.json`的`waiting`表示等待，`finished`表示结束事件已返回，`interrupted`表示等待被打断。进程退出不代表训练成功，当前代理仍须核对真实退出状态、日志更新数及最终模型；工具不会伪造无法取得的训练退出码。

当前另外启用应用内的“稳站训练完成后继续迭代”自动续办，直接由本聊天Codex处理，不调用Claude。它每5分钟检查保存状态，在训练或评估完成、失败、中断时继续处理；已有进程运行时不重复启动，无变化时保持安静。它能续接空闲聊天，普通等待脚本仅能返回给仍有效的工具调用，两者作用不同。调度检查最多有一个周期的延迟；使用方式参考[OpenAI的评估改进循环说明](https://developers.openai.com/cookbook/examples/agents_sdk/agent_improvement_loop)。

有限检查：`mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_wait_training_completion --output artifacts/<新目录>/check.json`。覆盖真实短进程退出、PID复用、事件落盘、等待中断不影响受监控进程、已有结果不能覆盖；不运行训练或其他模型。


r11候选只新增恢复期的轮承重高度奖励，权重4：`upright * clamp((height-.16)/(.27-.16),0,1) * (.25+.75*load)`；`load`为两轮各自承重除以半机重后裁剪到0～1再平均。单轮承担全部机重不能满分，冲击力不能超过满分；恢复站稳后该项关闭。0.16m仅用于奖励进度，固定命令和验收目标仍为0.27m。r10固定0/256、历史成功0/2、地面0/128，已拒绝并回退本任务r08；正常站立奖励及50/50复位保持。后续按用户最新指令在本地训练。


r12仅将静腿与站姿奖励乘实际机身平速系数`1/(1+(v/0.005)^2)`，峰值权重3/2和真实五子步腿速核、所有地面/恢复豁免保持。r11实测机身约0.037m/s滚动时静腿奖励仍接近3/3，因此本批让停稳先于静腿得满分；这是训练奖励尺度，不改变20秒/5cm/各腿2°等验收。r11固定0/256、历史0/2、地面0/128，已拒绝；r12从本任务r08回退继续，PPO/控制/初态不改。


r13配方`stand-r13-visible-speed-recovery`：只在新任务恢复期豁免原`stand_still_lin_vel/no_fork/no_fork_square/leg_joint_vel/wheel_vel`，正常阶段原式原权重；用当前实际机身水平速度成本（权重−40，速度封顶0.1m/s）替换训练位置奖励，固定验收仍按原位置锚点测量。静腿奖励保留r12机身速度系数，正常站姿奖励恢复原目标约束，不再乘该系数。其余控制与原50/50混合复位不变；实际决策与来源见实验目录`recipe-r13/`，不把有限检查当行为通过。


r14配方`stand-r14-linear-leg-speed`只新增`stand_leg_speed_cost`，权重−20，统计全5物理子步四腿真实角速度的平均绝对值，补足原二次核对缓慢漂移区分不足；它不乘机身速度系数。仅正常身体支持阶段启用，地面起立、倒地、腾空、恢复、受扰未重新站稳时豁免，资格不依赖腿先静止。其余r13奖励和初态/控制/时序/接口/PPO保持；原500Hz极值验收不变。来源与决策见实验目录recipe-r14。


本次标准修改前的r01～r14原报告使用角摆幅/位移验收v1，上方历史配方段落中的2°/5cm仅指当时标准；当前停止条件以`stand-speed-v2`为准。两例原成功样本保留在新版本回归清单，显式记录用户授权的标准迁移，不把旧失败报告重新标成通过。


r15配方`stand-r15-speed-limits-curriculum`接入按表现推进的课程，新增腿速成本调整为`-20 * max(五物理子步四腿最大绝对速度 - 0.5, 0)`，限值内不施加这项成本，起立与恢复期仍豁免；原静腿平方核及其他r14奖励保留。阶段0无训练推扰，50/50原初态不变。每批仍跑全固定矩阵；静站128例全部通过且独立地面128例全部在5秒内起立、全部历史成功保留后进入弱推阶段；再通过全部弱推样本后进入强推和翻倒阶段。最终验收矩阵不随课程缩小。


r16配方`stand-r16-restore-proven-hold`恢复r08的正常静腿奖励（取消r12的机身0.005m/s乘子），撤销r13新增的−40机身速度成本，保留原低速倒数奖励（权重3、尺度0.01m/s）。r15的0.5rad/s真实峰值超速惩罚、所有地面/恢复豁免和恢复高度/朝向奖励、阶段0课程均保留。依据是速度v2下r08仍86/256，r15却0/256且近站姿奖励实际退化；这只改训练奖励，速度v2验收不变。

r17配方`stand-r17-recovery-height-progress`只把恢复直立项改为`2 * (signed_up - (1 - progress))`，其中`progress=clamp((height-.16)/(.27-.16),0,1)`。低矮机身撑地不能仅靠朝上获得完整直立奖励，抬高机体仍有连续引导；0.27m时保持原得分，不添加关节目标或静止资格。正常站立奖励、恢复轮承重高度项、PPO及全部接口和物理约束不变。用户再次授权远程续训，实际配方和结果以实验记录为准；回归并集已另存112例，旧88例清单及全部失败报告保留。

r18配方`stand-r18-long-horizon`只将本任务训练的折扣因子`gamma`由0.99改为0.999，让训练更重视数秒后的起立与保持收益；r17奖励、网络、其他PPO参数和所有控制/噪声/物理约束不变。按100Hz计算，5秒后的几何折扣由0.00657变为0.60638；这不是硬时间截止，价值估计仍会利用后续状态，不能据此认定唯一失败原因。折扣和价值递推见[rsl_rl PPO实现](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py)。r17首次地面成功样本另存地面回归清单，检查真实初姿与随机初速度；固定近站112例回归继续独立执行，不用地面成功替代固定翻倒验收。


r19配方`stand-r19-randomized-balance`将正常`stand_posture`项换为`exp(-(offset/0.015m)^2)`，权重2不变；`offset`为实际整机质量加权重心对两轮轴线中点的水平前向偏移。平移和转向不改变该项，不固定腿角，允许随质量/重心随机化调整腿姿；保持恢复豁免和原身体资格，不能以腿已静止作为启用条件。GPU原始惯性位置计算与独立CPU模型副本的运动学质心核对通过，仅验证奖励算术。恢复gamma=0.99，其他PPO默认、控制和传感参数不变。

r19训练进入弱推阶段1：r07名义静站64/64、弱推16/16已有完整成功，试验采用半数无推回合、半数符合原资格后单次弱推，原50/50近站/地面初态不变。此前r15要求全部随机化静站和地面通过后才晋级的训练门控本批不再沿用；该调整只改变训练经验，固定验收仍覆盖所有扰动并逐项满足原速度v2标准。新固定成功并集117例和独立地面成功1例继续作为回归，所有旧清单和失败报告保留。


r20配方`stand-r20-wheel-reward-restore`仅恢复r07静轮奖励核和权重，真实五子步统计与所有恢复豁免保留。r19固定0/256、117历史回归0、地面0/128及地面回归0/1，拒绝作为最终模型；实际20环境训练分布诊断中，正常支持阶段静轮成本约−2.45/秒。噪声单项消融仅用于诊断，不改变训练噪声或正式随机化验收，不计成功。下一批从保留的本任务r07续训1000轮，stage1、50/50复位、COM平衡奖励、PPO和全部控制约束保持r19；是否有效须完整固定集和历史回归判定。


r21配方`stand-r21-posture-prior-restore`只把r20的`stand_posture`换回原有软姿态核`exp(-四腿平均目标角误差平方/(0.05rad)²)`、权重2；静轮保持r20恢复的有界核，其余恢复奖励、完整恢复豁免、PPO/阶段1/混合初态/全部接口与时序保持。奖励目标与控制参考角分离，不能以腿已静止作为资格。实际r19名义seed42同32例在更新1后32/32、更新501后0/32；离线原姿态奖励从约0.864/秒降至0.00393/秒，最终近零，说明姿态明显漂移，但仍是诊断而非唯一因果证明。r20固定3/256、回归3/117、ground0/128及ground回归0/1，拒绝采用。


r22配方`stand-r22-body-speed-margin`保留r21的软姿态及静轮核，新增正常身体资格下的速度裕量成本：`-40*clamp(max(五物理子步world XY速度)-0.03,0,0.1)`。0.03 m/s只用于训练裕量，正式验收仍每500 Hz采样不超过0.05 m/s，20秒及其他判据不变；不使用位置锚点。地面起立、恢复、倒地、空中、恢复未稳及实际推扰未恢复资格时豁免；不以腿已静止作为资格。新峰值在现有物理子步奖励统计中采样，每策略周期清零；部分环境复位的身体资格立即清零，不能把旧峰值作为下一回合成本。不修改观测、动作、控制、时序或复位模块。固定成功回归清单另存新版本，旧清单和所有失败继续保留；当前结果只看[项目实验记录](ai/project-log.md)顶部及实验目录。
