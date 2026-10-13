# 串联腿 mjlab 接触传感器审计

2026-10-12。**前半部分为升级前的历史“仅 CPU 证据”；新增本机GPU配对验证及当前结论见文末“GPU 恢复与 CPU/GPU 配对补充”，当前结论以GPU结果为主要证据。**没有训练、TensorBoard、模型更新、部署修改或 Git 操作。Windows 迁移副本没有 `.git`，分支和暂存状态无法核对；没有创建 Git 仓库。现有 `experiments/`、`reference_training/`、两份 goals 源码及历史 `logs/` 未改写。

## 结论及证据边界

**已验证：最初“无 secondary 传感器漏检对地接触”的大幅不一致，主要来自比较了不同子步的 `.data` 缓存。** 同一份 Warp `sensordata` 内，未过滤传感器与 terrain 探针一致；原生 CPU MuJoCo 的未过滤传感器、terrain 探针和 `mj_contactForce` 手算也一致。因此没有证据把这些不一致归因于无 `reftype`、净力抵消、只匹配接触一侧或力方向错误。

此外有两项独立问题：

- **已验证：8 子步 history 未覆盖 10 子步控制周期。** 存在接触仅发生在前两子步、奖励窗口遗漏的情况。这是奖励采样窗口遗漏，传感器在该子步本身有输出。
- **已验证：少量同状态 Warp 与原生 CPU MuJoCo 接触结果不同。** 两个 Warp 传感器会同时漏掉 CPU 存在的接触。至少一个 base 样本定位到 Warp 没有相应接触条目；加 terrain 过滤无法解决。具体碰撞算法／容差／精度的原因尚未验证。

**未能验证：** 远端 GPU 是否同样出现这些差异，以及它们对已训练策略、课程推进或 E 组退化的实际贡献。本次零／随机动作数值检查不能代替已训练模型的 20 秒配对回放，更不能证明退化的因果机制。

## 环境和统计口径

WSL `Ubuntu-24.04`，解释器 `/home/administrator/.venvs/wheelleg-mjlab/bin/python`；Python 3.12，mjlab 1.6.0、MuJoCo 3.11.0、mujoco-warp 3.11.0、Torch 2.14.0、Warp 1.17.0。`torch.cuda.is_available()` 为 False。Warp 能列出显卡，不等于本次使用 GPU 进行有效验证；本次新采集的有效仿真和接触力证据均为 CPU。影响评估另标明只读的历史 GPU 回放 JSON，未访问远端重跑。

完整样本：地面、普通初态各 64 环境 × 150 控制步，seed42；偶数步均匀随机动作 `[-1,1]`，奇数步零动作，每步 10 × 2 ms。沿用 `keep_contact` 训练配置的随机化、终止及自动复位，复位记录在 `trace`；不是固定单一倒地姿态，也没有策略学习。只给审计探针增加只读 8 子步 history，不修改原传感器或物理参数。补充样本为各 16 环境 × 150 步；另有地面 64 环境 × 35 步的定点后端检查。

“接触次数”均为 **body–环境步**，以对地净力模长 >3 N 计数，不是接触点数量，也不包括只存在几何触碰而承重不足 3 N 的接触。`base_contact` 另按成功判定的 >1 N 阈值统计，轮传感器按 >3 N 统计。漏检率 = 漏检／对应地面阳性次数；分母为 0 时写不适用。

分别保留四种口径，不能混用：

1. `legacy_cached`：原 `miss_stats.py` 方法，`env.step()` 返回后读取 `.data.force`；可能混合缓存与新值。
2. `post_forward_raw`：同一份最终 `sensordata` 中直接按传感器地址读取；不经过 `.data` 缓存。
3. `reward_raw` / `reward_history`：奖励管理器计算之前的底层值，以及同期 8 子步 history；不提前访问 `.data` 改变缓存时机。
4. `cpu_manual_truth`：同一最终 qpos/qvel，新建原生 CPU `MjData` 后 `mj_forward`，按 `mj_contactForce` 合成真实 terrain 净力。完整统计复制各环境已随机化的模型字段和 ctrl、外力、warmstart；定点例子另保留直接使用 `env.sim.mj_model` 的结果。直接使用 nominal 模型不能要求力值与随机化的 Warp 世界相等。

原 `self_contact.py` 的 `fired_without_ground` 比较的是当前 `.data.force`，并非实际 `force_history` 惩罚；下面把当前力、历史窗口和奖励触发分开报告。

## 完整统计

### 相对 CPU 手算真值的底层传感器统计

两个原生 CPU 传感器相对手算，所有 body 漏检／误报均为0、三维向量差为0。下面 Warp 列的差异是**后端与原生 CPU 的差异**；两个 Warp 传感器的结果完全相同，不是未过滤 sensor 独有的问题。

| body | 地面 CPU 真接触 | 地面 Warp 漏／误 | 普通 CPU 真接触 | 普通 Warp 漏／误 |
|---|---:|---:|---:|---:|
| base_link | 4611 | 18 / 4 | 4653 | 10 / 5 |
| lf0 | 69 | 2 / 0 | 0 | 0 / 0 |
| lf1 | 238 | 1 / 1 | 0 | 0 / 0 |
| rf0 | 69 | 3 / 0 | 0 | 0 / 0 |
| rf1 | 192 | 3 / 0 | 0 | 0 / 0 |
| lf00 | 0 | 0 / 0 | 0 | 0 / 0 |
| lf01 | 0 | 0 / 0 | 0 | 0 / 0 |
| lf02 | 0 | 0 / 0 | 0 | 0 / 0 |
| lf03 | 0 | 0 / 0 | 0 | 0 / 0 |
| rf00 | 0 | 0 / 0 | 0 | 0 / 0 |
| rf01 | 0 | 0 / 0 | 0 | 0 / 0 |
| rf02 | 0 | 0 / 0 | 0 | 0 / 0 |
| rf03 | 0 | 0 / 0 | 0 | 0 / 0 |

地面合计 CPU 阳性5179 body步，Warp漏27／误5；普通阳性4653，漏10／误5。地面 base 的后端差异漏检率18/4611=0.39%，普通10/4653=0.21%。这些都是仅 CPU 证据。

### 原缓存比较方法的表观漏检／误报

| body | 地面探针阳性 | 地面缓存漏／误 | 普通探针阳性 | 普通缓存漏／误 |
|---|---:|---:|---:|---:|
| base_link | 4597 | 1117 / 1087 | 4648 | 1069 / 980 |
| lf0 | 67 | 12 / 8 | 0 | 0 / 0 |
| lf1 | 238 | 12 / 16 | 0 | 0 / 0 |
| rf0 | 66 | 5 / 10 | 0 | 0 / 0 |
| rf1 | 189 | 11 / 10 | 0 | 0 / 0 |
| lf00 | 0 | 0 / 0 | 0 | 0 / 0 |
| lf01 | 0 | 0 / 0 | 0 | 0 / 0 |
| lf02 | 0 | 0 / 0 | 0 | 0 / 0 |
| lf03 | 0 | 0 / 0 | 0 | 0 / 0 |
| rf00 | 0 | 0 / 0 | 0 | 0 / 0 |
| rf01 | 0 | 0 / 0 | 0 | 0 / 0 |
| rf02 | 0 | 0 / 0 | 0 | 0 / 0 |
| rf03 | 0 | 0 / 0 | 0 | 0 / 0 |

这些表观差异在 `reward_raw` 和 `post_forward_raw` 中全部归零，三维向量最大差也为0。地面／普通异时刻单body模长最大差为9210.601／4375.407 N。相同8子步窗口的非轮惩罚阳性环境步为7535／7344，漏／误均0。

### base_contact、wheel_contact 相对地面探针

| 初态 | sensor与阈值 | 探针阳性 | 缓存漏／误 | 表观漏检率 | 同刻 raw 漏／误 |
|---|---|---:|---:|---:|---:|
| 地面 | base >1 N | 4603 | 1116 / 1081 | 24.25% | 0 / 0 |
| 地面 | 两轮逐轮 >3 N，合计body步 | 4048 | 937 / 764 | 23.15% | 0 / 0 |
| 普通 | base >1 N | 4659 | 1067 / 970 | 22.90% | 0 / 0 |
| 普通 | 两轮逐轮 >3 N，合计body步 | 8220 | 1734 / 1528 | 21.09% | 0 / 0 |

单轮分别为：地面左2003阳性、474漏／369误，右2045、463／395；普通左4019、809／736，右4201、925／792。按原临时脚本的“地面 >3 N、base ≤1 N”混合口径，普通 base 漏1066/4648；上表统一使用成功判定1 N阈值，所以是1067/4659。原 `miss_stats.py` 已运行完毕、退出0，普通当前 undesired 统计1069漏／980误、轮合计1734漏，与本次复算一致。


## 根因：缓存、子步窗口和后端接触结果

### 缓存与 forward 的顺序

已安装 mjlab 1.6.0 的 `ManagerBasedRlEnv.step()` 顺序为：

```text
10 次：sim.step() → scene.update()（失效缓存并滚动 history）
→ termination / reward（访问训练传感器 .data，生成并缓存 cat 后的张量）
→ step events / reset
→ sim.forward()（从已积分的新 qpos/qvel 重算底层 sensordata）
→ command / sim.sense() / observations
→ 返回后，首次访问只读 ground_probe.data
```

`Sensor.data` 只在 `reset/update` 失效后重新计算；`forward()` 没有失效这些缓存。`ContactSensor._extract_sensor_data()` 的 `torch.cat` 得到独立张量，已缓存的 `force` 不会随底层 `sensordata` 更新。原 `undesired/base/wheel` 在奖励、终止或 critic 中已被读取；未参与本组奖励的地面探针则可能返回后才首次读取。所以“同一次 env.step 返回后的读数”不等于同一个物理状态。

典型地面样本，零基 step7、env14，完整 qpos/qvel/ctrl/外力见 [ground.json 的 examples](../../artifacts/contact-sensor-audit-20261012/full-v1/ground.json)。下表三维力为世界坐标、统一到传感器 primary→secondary 符号；**实际作用在 base_link 上的地面力取反**。

| 读取方式 | undesired base force，N | terrain probe base force，N |
|---|---|---|
| 返回后的 `.data` | `(0,0,0)` | `(23.565933,36.159618,-857.399109)` |
| (a) 同一 Warp `sensordata` | `(23.565933,36.159618,-857.399109)` | 同左 |
| (b) nominal `env.sim.mj_model` + 新 `MjData` + `mj_forward` | `(12.490703,62.435453,-127.345254)` | 同左 |
| 原生 CPU，复制 per-world 随机化字段 | `(23.566709,36.164340,-857.486090)` | 同左 |
| (c) 同一 CPU 接触逐个 `frame.T @ local_force` 合成，转换到上述符号 | `(23.566709,36.164340,-857.486090)` | — |

实际对 base 的手算向上支撑是 **857.486090 N**。不步进、不滚动 history，只失效并刷新 `.data` 缓存，原训练传感器立刻恢复为 (a) 的向量。单独再次 `sense()` 的 `sensordata` 最大变化为 0：这里 `sense()` 处理相机／射线上下文，接触力更新来自 `forward/step`，不是补调用 `sense()` 能解决的问题。

零基 step6、env30 是反向例子：缓存 base 力约 `(329.474,-216.484,-5122.785)`，最终 raw 训练传感器和探针均为 0，复制随机化字段后的 CPU 对地力也为 0。刷新缓存后归零；这是当前力缓存错位，不是自碰撞证据。

### history 与误报的区别

16 环境 × 150 步补充检查中：

| 初态 | 10 子步内有非轮对地接触的环境步 | 仅前两子步接触、8 子步窗口完全遗漏 | 占有接触环境步 |
|---|---:|---:|---:|
| 地面 | 1767 | 43 | 2.43% |
| 普通 | 1826 | 38 | 2.08% |

这些计数用 Warp 实际 10 个子步的 terrain 力，与奖励读取的 8 子步窗口比较，见 [substep-v2](../../artifacts/contact-sensor-audit-20261012/substep-v2/run.json)。普通64环境完整重跑还记录了114/7458（1.53%）个有接触环境步的窗口遗漏，不与上表16环境统计混用。同一个 8 子步窗口内，未过滤传感器与 terrain 探针的漏检／误报都为 0。因此不存在本样本支持的“history 乱滚动导致两类传感器不一致”；存在的是窗口长度不足。

如果 history 曾经有碰撞、当前子步已经脱离，惩罚仍触发是公式预期的窗口记忆，不能叫当前接触误报。地面／普通完整样本分别有 2664／2785 个环境步满足“history >3 N、奖励时刻当前 terrain 净力 ≤3 N”；同窗 terrain history 也触发。原生 CPU 完整样本没有 >3 N 非轮自碰撞净力，机器人 collision geom 的 `contype=1/conaffinity=0` 也与此一致。不能从异时刻的 `force` 差值推断自碰撞。

### 少量 Warp 与原生 CPU 差异

地面零基 step28、env15：两个 Warp raw base 力均为 0；nominal CPU 的两种传感器和手算均为 `(33.201433,-36.603939,-98.836905)` N；复制随机化字段后的 CPU 均为 `(4.548188,-4.548295,-152.425692)` N，模长约 152.561 N。

最终定点检查直接查看 Warp 的 contact buffer，并用 **CPU scope 下的 Warp 原生 `contact_force`** 合成活跃约束力。该状态没有 terrain–base_link 活跃约束条目，overflow 为 0；v3中仅采信的全部接触 geom/body 身份也无该接触，与v6的qpos/qvel逐项相同；原生 CPU 有该接触及非零解。由此将这一例定位到接触生成／进入约束之前的后端差异，而不是 sensor `reftype` 匹配或净力相互抵消。其他样本可能涉及接触点选择或约束求解差异，不能凭这一例概括全部。

**推断：** 浮点精度、凸网格接触生成或容差可能参与这些后端差异。**未能验证：** 哪一个具体算法分支造成它们、GPU 是否复现。没有把全部差异宣称为 Warp 传感器实现 bug。

实际编译参数：训练 force 传感器 `objtype=BODY(1)`、`reftype=UNKNOWN(0)`、`refid=-1`；terrain 探针 `reftype=GEOM(5)`、`refid=0`；两者 `intprm=[2,3,1]`。两个 CPU sensor 与手算一致，两个 Warp raw sensor 一致，均不支持“只匹配 geom1/geom2 一侧”的假说。MuJoCo 官方说明 `netforce` 在世界坐标合成匹配接触，并以指定对象顺序决定方向，见 [3.11.0 contact 文档](https://mujoco.readthedocs.io/en/3.11.0/XMLreference.html#sensor-contact)。本次结论以安装版本源码和实测值为准，安装源码及 SHA256 已保存到各 `run.json`。

## 影响清单

| 项目 | 已验证的调用关系与影响边界 |
|---|---|
| V14 `undesired_contact` | 原 V14／height／stand 延续版本读取原传感器 8 子步 history，权重 −2；本样本同窗与 terrain 一致。前两子步盲窗可影响短碰撞惩罚，后端真实接触缺失也会同时影响两类传感器。不能据缓存对比宣布奖励公式或符号错误。 |
| 六组及 E 扶正 | baseline 全阶段沿用原项；recovery 系列扶正阶段置零，E 仅在扶正恢复原项，后续阶段各组沿用相应配置。E 受到历史窗口潜在影响，但本次没有证据证明这导致丢失 0、55，也不能用它豁免退化。 |
| `base_contact`、`recovery_goals.py` | recovery `goals/failure` 在最后一个物理子步之后、最终 forward 之前读取，参与双轮平衡成功、mode 退出及课程推进。同期 raw 与 terrain 相同；不能拿返回后探针否定前一子步判据。无过滤在未来启用自碰撞／其他物体时会混入其他来源。 |
| `wheel_contact` / `wheel_support_force` | recovery 使用 `clamp(-Fz,0)`，轮平衡奖励 5 N 饱和，成功条件也用轮支撑；critic／部分 mode 使用 8 子步接触历史。本次 raw 对地读数一致、符号正确，历史窗口问题与后端差异仍适用。 |
| `wheel_support_goals.py` | 是纯 tensor 判据，不自行读取传感器。`huananhu_v14_wheel_support.make_env_cfg()` 已给 wheel/base 加 terrain 过滤；六组 `factors.goals()` 则组合 terrain probes 与原 base sensor。没有修改任何判据。 |
| 六组训练课程 | 课程调用 `recovery.goals()`，不是 `factors.goals()`；增加的 probes 没有把训练课程自动改成严格承重判据。F 组在阶段≥1的承重奖励使用 terrain probes，因而可能提前缓存它们。 |
| 旧 recovery 回放 | `eval_chuanliantui_recovery_height.py` 返回后读取原 wheel/base `.data`，可能仍是 reward/failure 缓存。成功／课程数值不能单独当作最终 forward 时刻的对地接触证明；GPU 是否改变样本归类未验证。 |
| 六组及验收 v2 回放 | `eval_chuanliantui_recovery_factors.py` 的轮接触、轮负载、非轮承重来自 terrain probes，具备对地来源过滤；`no_base_contact` 仍来自原 base 缓存。F 组 probes 还可能在 reward 被缓存，因此过滤本身不保证与其他状态量同一子步。瞬态的组合判定／边界成功率存在时序敏感风险，尚未复跑 GPU。 |

**已有结论中可以保留的独立证据：** 本次只读核对 [v2 原始回放数据](../../artifacts/contact-sensor-audit-20261012/impact-evidence.json)，源模型和六组的每个普通／地面样本在末5秒的承重组合比例均为 0 或 1；承重成功集合与平衡成功集合相同。E 地面样本0、55两轮分别为 0/0 N、非轮承重 131.4047/131.4019 N，独立违反轮端承重要求；样本55甚至 `no_base_contact` 全程通过，也仍失败。**E 丢失原成功样本0、55、退化否决继续成立**，不能因 base sensor 审计改写为合格。

已存报告的地面 60/58/58/58/56/58、普通均64/64是对应协议的历史统计，不是本次新复跑结果；terrain 承重数据不受“无 reftype 漏检”解释影响，但也不能承诺时序统一后精确数字永远不变。末5秒高度单项达标样本均0，低位高度数据独立于接触来源，因此“未达到0.30 m高度、不能部署验收”结论保留。视频轨迹、已完成500轮预算、迁移一致性及数值检查事实不因本次审计被撤销。

## 修复建议（未实施）

1. **先统一评估采样时刻。** 明确选择奖励子步状态或最终 forward 状态；同一次 snapshot 中读取 wheel、base、nonwheel、found、姿态、速度与高度。若选择最终 forward，传感器缓存也应同步失效／刷新，但不要通过 `scene.update(dt)` 额外滚动 history 来伪造新物理子步。需独立版本和同初态评估，不能改写冻结历史。
2. **将需覆盖整控制步的接触 history 设为 decimation=10，或显式累计该控制步全部子步。** 这是可直接针对盲窗的方案；不从当前样本推断它必然改善学习。仅改变 cache 刷新不能补回已丢弃的前两子步。
3. **新版本显式 `secondary=terrain`，`secondary_policy='error'`。** 有利于确保“对地”的业务含义及将来启用自碰撞时的来源隔离；当前资产下同刻值已相同，不应包装成已证实的无 secondary 修复。它不能补上 Warp 本身未生成的接触。
4. **可单独设计按非轮对地支撑比例的连续惩罚。** 例如先定义 `rho = sum(clamp(-Fz_nonwheel,0))/robot_weight`，在平衡阶段对 `max(rho-0.05,0)` 或其平方惩罚；是否封顶、各阶段启停及时间聚合需明确。扶正期允许必要机身接触，应作为另一项明确实验，避免把目标变化混同传感器修复。绝不使用固定腿角作为验收。
5. **剩余后端差异需先做 CPU/GPU 同状态 contact/constraint 对照。** 保留 qpos/qvel、随机化字段、外力、接触距离／点及约束力，检查 collision/constraint 条目；未定位前不要仅靠增大惩罚权重。重新评价时保留原成功集合退化否决，不启动新训练来掩盖缺口。

## 文件、命令与失败记录

新增 [审计脚本](../../scripts/agent/check_chuanliantui_contact_sensors.py)、本报告及 `artifacts/contact-sensor-audit-20261012/` 下 JSON。完整地面数据在 `full-v1/ground.json`；完整普通数据在 `normal-v4/normal.json`；history补充在 `substep-v2/`；最终后端定点证据在 `backend-v6/`。原脚本执行记录、v1审计源码快照、已有模型影响证据、失败／中断记录及最终校验也保留为 JSON。

实际有效命令与退出码在下表；各独立审计 `run.json` 含解释器、完整 argv、版本、北京时间起止、源码哈希和逐项数值。

| 实际执行的检查入口／参数（工作目录均为仓库根） | 输出 | 退出码 |
|---|---|---:|
| 指定venv Python `/mnt/c/Users/Administrator/AppData/Local/Temp/ccheck/miss_stats.py` | legacy-execution.json | 0 |
| 审计脚本 `--num-envs 2 --steps 2 --output artifacts/contact-sensor-audit-20261012/smoke-v1` | smoke-v1/ | 1 |
| 审计脚本 `--num-envs 2 --steps 2 --output artifacts/contact-sensor-audit-20261012/smoke-v2` | smoke-v2/ | 0 |
| 审计脚本 `--num-envs 64 --steps 150 --output artifacts/contact-sensor-audit-20261012/full-v1` | full-v1/ | 1 |
| 审计脚本 `--num-envs 16 --steps 150 --output artifacts/contact-sensor-audit-20261012/substep-v2` | substep-v2/ | 0 |
| 审计脚本 `--num-envs 64 --steps 35 --poses ground --output artifacts/contact-sensor-audit-20261012/backend-v3` | backend-v3/ | 0 |
| 审计脚本 `--num-envs 64 --steps 150 --poses normal --output artifacts/contact-sensor-audit-20261012/normal-v4` | normal-v4/ | 0 |
| 审计脚本 `--num-envs 64 --steps 35 --poses ground --output artifacts/contact-sensor-audit-20261012/backend-v5` | backend-v5/ | 1 |
| 审计脚本 `--num-envs 64 --steps 35 --poses ground --output artifacts/contact-sensor-audit-20261012/backend-v6` | backend-v6/ | 0 |

上述命令均通过 `wsl -d Ubuntu-24.04 -- bash -lc` 调用指定venv解释器，`PYTHONPATH=mjlab_training/src`；除原临时 `miss_stats.py` 外设置OMP/MKL线程1，substep-v2及其后各次还设置 `PYTHONDONTWRITEBYTECODE=1`。`full-v1` 的退出1为实际外层进程退出码，原 run.json 中 null 保留，不伪造正常结束。


失败与历史输出保留：`smoke-v1` 因 stdout Tee 缺 `isatty()` 退出1，`smoke-v2` 修正后退出0；`full-v1` 地面完成、普通中途退出1，未留下 Python 异常，不推定中断原因；普通已独立重跑。`backend-v3` 和 `normal-v4` 的 `warp_contact_buffer.manual_*` 旧诊断字段不作为力值证据：host转换重排了约束地址，非激活接触不能据此直接手算。该字段已改为 Warp 原生 `contact_force` 并只取 CONSTRAINT 接触。`backend-v5` helper 未进入 CPU scope，默认选择 CUDA 导致非法地址、退出1，没有有效 GPU 证据；最终 `backend-v6` 强制 CPU scope。raw sensordata、原生 CPU forward/contactForce 统计与这些旧转换字段独立。

Claude 复跑命令（在 WSL 内执行，输出目录必须是新目录）：

```bash
cd /mnt/d/IMCA/RL/Wheel-Legged-Gym
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=mjlab_training/src ~/.venvs/wheelleg-mjlab/bin/python \
  scripts/agent/check_chuanliantui_contact_sensors.py \
  --num-envs 64 --steps 150 --poses ground normal --actions mixed \
  --output artifacts/contact-sensor-audit-20261012/claude-rerun-01
```

最终脚本新增采样、原生 Warp 力验证和紧凑 JSON 输出，核心仿真动作／随机种子及统计公式保持一致；首次完整地面运行源码也保存为 `audit-source-full-v1.json`。这是一项传感器数值审计，不是单元测试、模型行为验证或独立第三方复查。没有提供或启动额外训练命令。


## 2026-10-12 GPU 恢复与 CPU/GPU 配对补充

**本节为升级后实测，本地 GPU 结果是当前结论的主要证据；前文“仅 CPU 证据”及原计数是历史采集口径。没有启动正式训练、TensorBoard 或实施传感器修复。**

Windows 驱动由 **551.61** 更新到 NVIDIA Studio **617.42 WHQL 笔记本版**，来自 [NVIDIA 官网](https://www.nvidia.com/en-us/drivers/details/280102/)。安装包和 setup.exe 的 NVIDIA 数字签名有效，安装包 SHA256 为 `D929D8B4587CC49717357B7EF3F60582304568C198F90F8263E1177A8C963B2F`。官网直连重定向后返回403，最终使用本机已有代理从 `us.download.nvidia.com` 完整下载。安装器 Display.Driver 收尾报 `0x80070005`（进程/注册表拒绝访问），错误日志保留，未将其退出码改写为成功。设备驱动已加载；用户在北京时间16:45:16重启 Windows，此后执行 `wsl --shutdown` 并重新进入 Ubuntu-24.04。没有重装或更改 venv 的 torch/mjlab/warp，没有安装 WSL Linux NVIDIA 驱动。

| 检查 | 升级前 | 重启并关闭 WSL 后 |
|---|---|---|
| NVIDIA Windows 内核驱动 | 551.61 | 617.42 |
| WSL nvidia-smi 驱动支持的 CUDA | 12.4 | CUDA UMD Version 13.4 |
| Torch 构建／编译 CUDA | 2.14.0+cu130／13.0 | 2.14.0+cu130／13.0 |
| torch.cuda.is_available() | False | True |
| keep_contact，16环境，seed42，零动作20步 | CPU观测／奖励均有限，退出0 | cuda:0观测／奖励均有限，退出0 |

新 nvidia-smi 界面采用 `KMD Version` / `CUDA UMD Version` 标签；WSL工具版本615.78.05、KMD617.42、CUDA UMD13.4。工具版本不是当前 Windows 驱动版本。完整输出见 `gpu-acceptance.json`、`gpu-driver-installation.json`。

### 当前结论（以本机GPU为主要证据）

- **GPU恢复三项验收通过。** Windows驱动551.61→617.42，WSL CUDA UMD13.4，Torch CUDA可用True，16环境cuda:0零动作20步观测和奖励均有限。安装器历史0x80070005收尾错误保留，不能将安装进程退出码改成0；重启后的GPU功能验收独立通过。
- **CPU/GPU不严格一致。** 同初态和动作的3秒轨迹逐渐分离；14个严格同状态案例也有force差异，一例CPU Warp base=0 N、GPU=162.321 N、原生CPU手算=163.021 N。同状态v2逐例清空所有工作区、恢复整批初态、模型及warmstart，排除了前一案例残留与模型派生参数不一致。
- **大幅“训练传感器漏检”仍主要是异时刻缓存比较。** 两端同刻raw和相同8子步history相对terrain probes漏/误均0；本批base及wheel同刻相对探针漏检率均0%。这不能证明Warp引擎没有漏生成真实接触。
- **8/10子步盲窗在GPU得到验证。** 地面127、普通109个环境步的接触只出现在前2子步而未被8子步奖励窗口覆盖；对应CPU为122、121。只加secondary=terrain不能补此窗口。
- **残余差异已定位到接触生成阶段，具体算法原因未验证。** 地面步29环境29，原生CPU生成3个base–terrain点、CPU/GPU Warp均只有1个；原生承重的另一接触点在Warp接触流形中缺失，导致原生1375.93 N而两端Warp均0。普通步23环境44，原生1点、两端Warp0点；地面步55环境30，CPU Warp0点、GPU1点。所有案例overflow为0。这不是无secondary专属语义问题，也不是只刷新sensor缓存可以修复的情况；不能凭本审计断言具体碰撞算法、精度容差或网格机制的原因。

以同状态原生CPU mj_contactForce为真值的13个非轮body合计漏/误：地面CPU21/7、GPU16/10，普通CPU16/12、GPU11/7。两端各用自身轨迹逐帧建立同状态原生参照，真接触分母不同，不能把差值当作GPU普遍更好或更差的证据。逐body数值在下表及JSON中。

**影响与建议保持原报告的边界。** GPU采集确认缓存时序、历史窗口及接触流形差异三类问题；没有新训练或已训练模型的配对回放，因此不能据此解释E组退化的因果贡献。原E样本0、55轮端承重失败和退化否决继续成立，“未满足0.30 m高度目标”的独立证据保留。推荐新版本统一采样时刻、覆盖完整10子步、显式地面过滤；流形缺点需先对照几何和碰撞算法再决定修复。连续非轮承重惩罚是奖励设计实验，不是已验证的传感器修复。**未实施任何修复。**

### 配对输入与统计口径

每种初态64环境×150控制步，seed42，偶数步均匀随机动作[-1,1]、奇数步零动作。CPU先保存精确 qpos/qvel/ctrl/外力/warmstart/time、所有 expanded model 字段、命令／动作／随机化／观测缓冲区及完整动作张量，GPU复用同一份JSON输入（张量按原始字节无损编码）。物理初态和控制缓冲区误差均0，随机化输入复用；模型派生常量由各后端重算，误差另存。没有只凭相同seed声称配对。两端均0次自动复位，3秒采集期未触发5秒之后的命令重采样或推扰。原训练内核只读，运行前后哈希一致。

审计只给两个 terrain probes 增加只读8子步history；原传感器和10子步控制周期保留。所有计数是body–环境步；非轮和轮以模长>3 N，base以>1 N。正式训练只允许GPU，每次本地≤256环境；本次所有GPU采集均64环境，验收16环境，未跑4096环境。

首次CPU/GPU同时构建64环境进程引起主存压力，WSL约7.7GB、Windows剩余不足1GB；主动停止首次GPU进程，保留失败目录，随后串行完成有效采集。小样本v1的非有限缓存占位值导致输入误差断言失败；保存改为原始字节后CPU/GPU小样本均通过。这些失败不作为有效数值证据。

### 同刻底层传感器相对 terrain 探针

| 初态 | 传感器 | CPU阳性 | CPU漏 | CPU误 | GPU阳性 | GPU漏 | GPU误 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 地面 | undesired | 5307 | 0 | 0 | 5258 | 0 | 0 |
| 地面 | base | 4764 | 0 | 0 | 4714 | 0 | 0 |
| 地面 | wheel | 3910 | 0 | 0 | 3979 | 0 | 0 |
| 普通 | undesired | 4729 | 0 | 0 | 4717 | 0 | 0 |
| 普通 | base | 4736 | 0 | 0 | 4723 | 0 | 0 |
| 普通 | wheel | 8249 | 0 | 0 | 8217 | 0 | 0 |

以上比较同一份 `post_forward_raw` sensordata；reward时刻和等长history另保留JSON，不混用缓存时刻。

### Warp 相对同状态原生 CPU mj_contactForce 手算

| 初态 | body | CPU真接触 | CPU Warp漏 | CPU Warp误 | GPU对应真接触 | GPU Warp漏 | GPU Warp误 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 地面 | base_link | 4771 | 13 | 4 | 4711 | 7 | 7 |
| 地面 | lf0 | 64 | 3 | 0 | 64 | 3 | 0 |
| 地面 | lf1 | 227 | 0 | 2 | 227 | 0 | 2 |
| 地面 | rf0 | 67 | 3 | 1 | 69 | 4 | 1 |
| 地面 | rf1 | 192 | 2 | 0 | 193 | 2 | 0 |
| 地面 | lf00 | 0 | 0 | 0 | 0 | 0 | 0 |
| 地面 | lf01 | 0 | 0 | 0 | 0 | 0 | 0 |
| 地面 | lf02 | 0 | 0 | 0 | 0 | 0 | 0 |
| 地面 | lf03 | 0 | 0 | 0 | 0 | 0 | 0 |
| 地面 | rf00 | 0 | 0 | 0 | 0 | 0 | 0 |
| 地面 | rf01 | 0 | 0 | 0 | 0 | 0 | 0 |
| 地面 | rf02 | 0 | 0 | 0 | 0 | 0 | 0 |
| 地面 | rf03 | 0 | 0 | 0 | 0 | 0 | 0 |
| 普通 | base_link | 4733 | 16 | 12 | 4721 | 11 | 7 |
| 普通 | lf0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 普通 | lf1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 普通 | rf0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 普通 | rf1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 普通 | lf00 | 0 | 0 | 0 | 0 | 0 | 0 |
| 普通 | lf01 | 0 | 0 | 0 | 0 | 0 | 0 |
| 普通 | lf02 | 0 | 0 | 0 | 0 | 0 | 0 |
| 普通 | lf03 | 0 | 0 | 0 | 0 | 0 | 0 |
| 普通 | rf00 | 0 | 0 | 0 | 0 | 0 | 0 |
| 普通 | rf01 | 0 | 0 | 0 | 0 | 0 | 0 |
| 普通 | rf02 | 0 | 0 | 0 | 0 | 0 | 0 |
| 普通 | rf03 | 0 | 0 | 0 | 0 | 0 | 0 |

每一端均用自己的当前状态和同一随机化模型参数重建原生CPU真值。两端演化轨迹发生数值差异，故两列的真接触分母可以不同；不能把计数差归因于传感器匹配语义。

### 异时刻缓存比较与历史窗口

| 初态 | 传感器 | CPU缓存漏 | CPU缓存误 | GPU缓存漏 | GPU缓存误 |
| --- | --- | --- | --- | --- | --- |
| 地面 | undesired | 1219 | 1104 | 1189 | 1109 |
| 地面 | base | 1179 | 1056 | 1153 | 1058 |
| 地面 | wheel | 914 | 831 | 961 | 796 |
| 普通 | undesired | 1186 | 1024 | 1142 | 1028 |
| 普通 | base | 1185 | 1017 | 1143 | 1023 |
| 普通 | wheel | 1860 | 1513 | 1816 | 1550 |

| 初态 | 后端 | 10子步阳性环境步 | 仅前2子步阳性遗漏环境步 | 遗漏body步 | 同8窗漏 | 同8窗误 |
| --- | --- | --- | --- | --- | --- | --- |
| 地面 | CPU | 7663 | 122 | 154 | 0 | 0 |
| 地面 | GPU | 7654 | 127 | 160 | 0 | 0 |
| 普通 | CPU | 7577 | 121 | 121 | 0 | 0 |
| 普通 | GPU | 7581 | 109 | 109 | 0 | 0 |

### 同输入轨迹是否数值一致

| 初态 | 初态最大误差 | 控制缓冲误差 | CPU复位 | GPU复位 | >3N分类不同body步 | qpos最大差 | 力分量最大差N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 地面 | 0 | 0 | 0 | 0 | 1643 | 2.31068 | 3711.67 |
| 普通 | 0 | 0 | 0 | 0 | 1344 | 0.519832 | 1570.94 |

相同初态和动作下的轨迹结果不保证位级一致，接触切换会放大浮点和求解差异。下表使用同一保存状态重算，排除轨迹已经分离的混淆。

### 任务2(a)：相同状态的 CPU Warp / GPU Warp sensordata

从CPU配对轨迹的缓存不一致和Warp–原生CPU不一致样本保存qpos/qvel、ctrl、外力、warmstart及每世界随机化字段，在两端分别forward重算。最终v2每个案例先reset清空全部solver/constraint工作区，再恢复整批相同初态，接着覆盖目标环境；恢复所有模型派生常量，排除set_const舍入差。每对输入SHA256相同、qpos/qvel等输入和模型字段误差均0；表中不把旧缓存读数与fresh forward混比。原生CPU两个sensor与手算的完整向量也保存在replay JSON。探索v1及selected-v1未逐例清空所有工作区，保留但不作为最终精确力值表。

| 初态 | 步 | 环境 | 样本类型 | CPU Warp base模长N | GPU Warp base模长N | 原生手算base模长N | 非轮全body最大分量差N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 地面 | 4 | 22 | cpu_positive_warp_other_zero | 0 | 0 | 0 | 0.00183105 |
| 地面 | 4 | 46 | cpu_positive_warp_other_zero | 0 | 0 | 0 | 0.000793457 |
| 地面 | 6 | 49 | cached_miss | 1520.12 | 1520.12 | 1322.16 | 0.00817871 |
| 地面 | 6 | 30 | cached_extra | 0 | 0 | 0 | 0 |
| 地面 | 7 | 14 | cached_miss | 858.451 | 858.455 | 858.572 | 0.003479 |
| 地面 | 7 | 10 | cached_extra | 0 | 0 | 0 | 0.0244713 |
| 地面 | 29 | 29 | cpu_positive_warp_base_zero | 0 | 0 | 1375.93 | 0 |
| 地面 | 55 | 30 | cpu_positive_warp_base_zero | 0 | 162.321 | 163.021 | 161.842 |
| 普通 | 6 | 54 | cached_miss | 190.527 | 190.519 | 190.505 | 0.00744629 |
| 普通 | 7 | 23 | cached_miss | 4882.05 | 4881.9 | 4882.03 | 0.146484 |
| 普通 | 7 | 6 | cached_extra | 0 | 0 | 0 | 0 |
| 普通 | 8 | 2 | cached_extra | 0 | 0 | 0 | 0 |
| 普通 | 23 | 44 | cpu_positive_warp_base_zero | 0 | 0 | 67.4599 | 0 |
| 普通 | 24 | 31 | cpu_positive_warp_base_zero | 0 | 0 | 41.9901 | 0 |

接触生成阶段的对照如下，包含所有Warp接触条目而非仅激活constraint，overflow均0：

| 初态 | 步 | 环境 | 原生base地面点数 | CPU Warp点数 | GPU Warp点数 | CPU点距离m | GPU点距离m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 地面 | 29 | 29 | 3 | 1 | 1 | -0.00186443329 | -0.00186433119 |
| 地面 | 55 | 30 | 1 | 0 | 1 | 无 | -3.20562151e-07 |
| 普通 | 23 | 44 | 1 | 0 | 0 | 无 | 无 |
| 普通 | 24 | 31 | 1 | 0 | 0 | 无 | 无 |

### 实际命令与退出码

| 输出目录 | device | 退出码 | 开始北京时间 | 结束北京时间 |
| --- | --- | --- | --- | --- |
| paired-cpu-v1 | cpu | 0 | 2026-10-12T17:02:00.849334+08:00 | 2026-10-12T17:24:53.095429+08:00 |
| paired-gpu-ground-v2 | cuda:0 | 0 | 2026-10-12T17:25:31.954523+08:00 | 2026-10-12T17:26:12.634621+08:00 |
| paired-gpu-normal-v2 | cuda:0 | 0 | 2026-10-12T17:26:55.467377+08:00 | 2026-10-12T17:27:40.899666+08:00 |
| paired-replay-cpu-v2 | cpu | 0 | 2026-10-12T17:40:33.660921+08:00 | 2026-10-12T17:40:57.131009+08:00 |
| paired-replay-gpu-v2 | cuda:0 | 0 | 2026-10-12T17:42:01.352388+08:00 | 2026-10-12T17:42:14.285658+08:00 |

完整解释器、参数及命令见各run.json。本次新有效脚本新增device、共享输入、轨迹记录和同状态replay；冻结训练源码未改。与原CPU统计使用不同的预生成动作序列，不能要求历史表的计数与本节相同。

复跑时必须CPU写入新的共享输入目录后，GPU读取同一目录；串行执行，并为每次输出使用新目录。`--num-envs 64 --steps 150 --poses ground normal --actions mixed --device cpu --paired-inputs <新共享目录> --output <新CPU目录>`，然后同参数 `--device cuda:0 --paired-inputs <该共享目录> --output <新GPU目录>`。定点重算增加 `--replay-from <新CPU目录>`，各device各用新输出目录。命令环境为 `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=mjlab_training/src ~/.venvs/wheelleg-mjlab/bin/python scripts/agent/check_chuanliantui_contact_sensors.py`。

### 文件与版本

本次只修改审计脚本 `scripts/agent/check_chuanliantui_contact_sensors.py` 和中文诊断报告 `docs/diagnostics/20261012-mjlab-contact-sensor-audit.md`，新增原始JSON在 `artifacts/contact-sensor-audit-20261012/`。输入在paired-inputs-v1，完整统计在paired-cpu-v1／paired-gpu-ground-v2／paired-gpu-normal-v2，最终同状态证据在paired-replay-cpu-v2／paired-replay-gpu-v2，汇总在paired-comparison.json。探索v1、selected-v1、两次小样本和中断记录保留；最终v2已逐案例清理全部工作区。仓库Windows迁移副本无.git，没有提交、分支、worktree或部署改动。

### 升级前后完整 WSL nvidia-smi 输出

升级前：

```text
Mon Oct 12 16:23:26 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.10              Driver Version: 551.61         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...    On  |   00000000:01:00.0  On |                  N/A |
| N/A   47C    P8              5W /  140W |    1216MiB /   8188MiB |     39%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A        27      G   /Xwayland                                   N/A      |
+-----------------------------------------------------------------------------------------+
```

重启并执行wsl --shutdown后：

```text
Mon Oct 12 16:53:37 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 615.78.05              KMD Version: 617.42        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...    On  |   00000000:01:00.0  On |                  N/A |
| N/A   48C    P8              4W /  115W |     995MiB /   8188MiB |      8%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

对应Torch检查：升级前False，升级后True。两次构建均2.14.0+cu130；完整GPU有限验收退出码0。
