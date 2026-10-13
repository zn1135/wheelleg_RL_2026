# 恢复退化单项对照：六组各追加500轮

**已完成：六组各追加500轮、累计2500轮，全部回放/视频/对照分析于北京时间13:37:52结束，协调器退出码0。最终地面平衡：baseline60/64，recovery_reference/skip_righting/keep_height/bearing_reward各58/64，keep_contact56/64；普通均64/64，0.30m高度跟踪均0/64。基线新增样本27、29，没有丢失原58个成功样本；keep_contact丢失0、55。六组同屏视频见[视频记录](20261012-mjlab-recovery-factors-videos.md)。未启动额外训练。以下保留启动与调度历史。**

**当前执行方式已按用户要求改为并行。12:39:38由`parallel_chuanliantui_recovery_factors.py`接管，训练配置和各组500轮预算不变；baseline原进程与产物保留，其余5组同时启动。当前状态以`parallel-status.json`/`status.json`为准，原串行启动记录保留。**

2026-10-12用户要求启动多组训练、逐项分析，并确认每组追加500轮。原计划5组另补一组相同500轮预算的recovery参照，共6组；均独立从已评估2000轮高度门控模型开始，4096环境、seed42，每组累计2500轮。没有串接上一组模型，没有启用辅助力、膝限位或默认角惩罚。

## 已完成的准备

- 原生mjlab Python3.12 / RTX4090：6组分别4096环境、16控制步，观测/奖励有限，数值异常0。
- 六组首次完整迁移actor、critic、Adam逐项相同，累计2000、学习率1e-5；自己的checkpoint保存/重载通过，无PPO更新。
- 实验组原配置除指定奖励wrapper外一致，recovery各组训练初始qpos逐项相同；skip_righting只将课程初始阶段设为1。
- 对同一真实仿真状态检查阶段0/1/2奖励输出：keep_height只改变指定高度项，keep_contact只改变扶正接触项，bearing_reward只改变轮支撑项，其余奖励逐项相同。
- GPU检查时空闲，24GB显存，数据盘约37GB可用，采用串行队列。

预检：`jobs/20261012-recovery-factors-preflight-v1`，退出0，无学习。最终冻结`releases/20261012-recovery-factors-v3`；v2修正协调器启动记录的原子写入，v3补齐仓库已有视频渲染脚本，并在协调器启动前检查脚本完整性。训练模块、检查脚本、训练入口与评估脚本同v1逐字节一致，因此原生配置检查仍适用。v1/v2及检查、失败日志保留。

## 当前运行

首个队列北京时间12:19:49启动，成功复现2000轮源模型普通64/64、地面58/64，但因冻结包缺少视频脚本在PPO开始前停止，没有进入学习。失败job`jobs/20261012-recovery-factors-500`完整保留。

补齐视频脚本后，最终队列北京时间12:23:34启动，协调器PID`29768`。同一源模型复核普通64/64、地面58/64通过，源回放视频生成成功。12:24:39进入baseline训练，训练PID`29909`；12:27:01日志已到累计迭代索引2101，实际PPO学习已核验，其余5组排队。

真实baseline run的迁移记录再次确认actor/critic/Adam完全相同，初始累计2000、目标2500、学习率1e-5。启动状态只证明第一组正在学习，其余组尚未开始；各组学习效果仍待完成后的自动回放。

协调器：

`/root/gpufree-data/wheelleg-remote/jobs/20261012-recovery-factors-500-v2`

六组训练输出共同根目录：

`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261012_122334_386753_recovery_factors`

子目录按顺序为`baseline`、`recovery_reference`、`skip_righting`、`keep_height`、`keep_contact`、`bearing_reward`。分组定义与比较边界见[接口说明](../interfaces/chuanliantui-mjlab-recovery-factors-v1.md)。

`status.json`保存当前组、进程、队列状态；每组`training.log`记录训练；追加101/500轮回放目录`early101`/`final500`保存`evaluation.json`、`analysis.json`、样本0视频与轨迹。协调器根`analysis.md`逐组更新。进程不依赖SSH会话，训练或评估失败会停止后续队列并保留错误与产物，不自行扩大训练预算。

## 切换为同时训练

用户随后明确要求多组同时训练。切换前RTX4090显存24GB，单个4096环境训练约2.6GB；并行入口要求空闲显存至少18.5GB，预留5个训练进程与一组回放。只改变调度方式，初态、奖励、算法、seed、模型迁移和轮数保持原值。

北京时间12:34:40只向原协调器PID29768发送SIGSTOP，基线训练PID29909继续运行。核对原进程完整命令与campaign身份，保存`parallel-handoff.json`后，12:39:38终止原调度器并启动并行协调器PID30009。不会重新启动baseline、重复启动已有run或覆盖历史模型。

冻结版本`releases/20261012-recovery-factors-v4-parallel`只增加并行协调器，原训练模块、入口、评估与视频脚本6份源码同v3逐字节一致；原六组原生检查仍适用。原launch和交接时状态保留，`parallel-launch.json`记录新的调度来源。

12:45:34核验：baseline最终`model_2499.pt`完整接口、累计2500轮与有限参数检查通过，正在回放；另5组均进入真实PPO学习，当前迭代索引2054/2055，训练PID分别30010～30014。它们真实run的actor/critic/Adam迁移逐项完全相同，初始学习率1e-5、目标2500轮，启动日志无Traceback。显存13700 MiB，余量10381 MiB，GPU利用率100%。

各训练进程并行运行；单一协调器收集训练完成情况并顺序执行回放、视频与分析，避免汇总文件被同时写入。完成后仍只追加500轮并停止。某组失败保留其错误，其他已启动组继续完成；不自动重复训练或增加轮数。

基线是接管的非直属子进程，其操作系统退出码无法由新协调器waitpid读取：完成依据为原进程结束、最终完整checkpoint通过接口/累计2500/有限参数检查，不伪造退出码。新启动的5组正常记录实际退出码。

已执行并行启动命令：

```bash
cd /root/gpufree-data/wheelleg-remote/releases/20261012-recovery-factors-v4-parallel
TZ=Asia/Shanghai /root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python \
  scripts/agent/parallel_chuanliantui_recovery_factors.py launch \
  --campaign /root/gpufree-data/wheelleg-remote/jobs/20261012-recovery-factors-500-v2
```

TensorBoard日志根目录保持本页下方命令。原顺序执行入口的历史启动命令保留，当前不可再次运行以免另开重复实验。

## 实际启动命令

```bash
cd /root/gpufree-data/wheelleg-remote/releases/20261012-recovery-factors-v3
TZ=Asia/Shanghai /root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python \
  scripts/agent/launch_chuanliantui_recovery_factors.py launch \
  --campaign /root/gpufree-data/wheelleg-remote/jobs/20261012-recovery-factors-500-v2 \
  --source /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_214628_750246_scut_v14_height_v14_height_resume_1000_to_2000/model_1999.pt \
  --initial-ground /root/gpufree-data/wheelleg-remote/jobs/20261012-recovery-regression-paired-v2/initial-ground.npz \
  --preflight /root/gpufree-data/wheelleg-remote/jobs/20261012-recovery-factors-preflight-v1
```

上面是已执行记录，重复原campaign路径会报错，不会覆盖既有训练。子进程显式设置冻结目录PYTHONPATH、`MUJOCO_GL=egl`、`TZ=Asia/Shanghai`、OMP/MKL线程1。

TensorBoard命令（未启动）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/tensorboard \
  --logdir=/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261012_122334_386753_recovery_factors \
  --port=8080
```

## 授权边界与结果解释

本批上限是6组各500次PPO更新，全部完成后停止；没有授权择优组额外续训。默认V14和历史训练源码/模型保持原内容。Windows迁移副本没有Git元数据，未创建仓库、分支、worktree或提交。未修改H7部署代码。

每组先与同轮数recovery_reference比较，原height继续训练的baseline作为旧目标对照。训练reward不能跨不同目标直接比较；优先检查双轮实际承重、原58个地面成功样本保留情况，再检查高度。当前是启动与有限检查证据，尚不能宣称某项修正已解决退化。

## 启动证据

本地包`artifacts/recovery-factors-startup-20261012.zip`，SHA256：

`3f6db0ca33d59a0b643e6014b0a6745d5447684250d66f464f04c7f623e688d1`

ZIP及21个载荷均已逐项校验，实际执行的6份源码与本地逐字节一致。包含最终launch、启动status、实际学习与迁移核验、6组预检、源2000轮回放与渲染记录、首个失败队列说明、baseline接口/实际环境/runner、启动日志；历史logs、模型、冻结版本完整保留。

并行接管证据包`artifacts/recovery-factors-parallel-20261012.zip`，SHA256：

`dba670cb210f49229ec9fa7ec38ab74ef8acd9b2edf73db20548c8c0e9c68f80`

ZIP及19个载荷逐项校验，7份实际执行源码与本地一致。包含调度交接、并行启动/状态、六组实际学习及迁移核验、baseline完成证据与各组启动日志。
