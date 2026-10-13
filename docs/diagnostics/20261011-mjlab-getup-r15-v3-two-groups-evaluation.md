# r15 v3 去除 pose_bias 后两组回放评估

## 状态与结论

2026-10-11 17:35:10（北京时间）查询：从头组完成约1800/2000轮，仍在运行；续训组已完成追加1000轮、累计5000轮，退出码0。本次固定评估从头组 `model_1400.pt`（已完成1401轮）及续训组最终 `model_4999.pt`（已完成5000轮），不能把1401轮回放作为从头2000轮最终结果。

三阶段均移除 pose_bias。两组模型仍处于扶正课程，没有进入站起或行走。关闭辅助力的评估中，高度接近0.27 m，但持续严格站立均为0；视频样本仍明显倾斜。去除默认角惩罚尚未解决持续扶正问题，不能据此认定该奖励是唯一原因。

## 统一评估条件

- 普通初态64个、地面初态64个，分别回放20秒，seed=42；两个模型使用完全相同的初态。
- 冻结扶正课程、目标高度0.27 m、零速度命令，关闭向上辅助力、域随机化、观测噪声和动作延迟；策略确定性推理。
- 不执行PPO更新；不因物理失败提前重置，使地面初态获得完整20秒恢复机会；单独记录物理失败，数值异常才截断。
- 严格恢复：高度误差≤3 cm，横滚、俯仰各≤15°，两轮接触历史峰值各>5 N，机体接触≤1 N，持续0.5秒。
- 持续站立：最后5秒至少95%的时间满足严格条件。这与课程升级的整批门槛不同。

## 结果

| 固定模型 | 初态 | 曾严格恢复 | 最后5秒持续站立 | 最后5秒平均高度误差 | 数值异常 |
|---|---|---:|---:|---:|---:|
| 从头1401轮 | 普通 | 0/64 | 0/64 | 0.483 mm | 0 |
| 从头1401轮 | 地面 | 0/64 | 0/64 | 0.502 mm | 0 |
| 续训5000轮 | 普通 | 0/64 | 0/64 | 2.745 mm | 0 |
| 续训5000轮 | 地面 | 1/64 | 0/64 | 4.236 mm | 0 |

视频仅展示每种初态的样本0，不能代替64个环境统计。该样本最后5秒平均绝对横滚/俯仰：从头普通18.79°/24.10°，从头地面20.52°/21.39°；续训普通13.22°/21.97°，续训地面10.49°/17.65°。俯仰仍超过严格门槛。

与之前v2累计4000轮、相同初态的评估比较：普通短暂恢复1/64→0/64，地面4/64→1/64；持续站立均为0。高度误差由3.306/5.747 mm降为2.745/4.236 mm。单批结果不足以证明整体统计退化或具体根因，但没有显示持续扶正改善。

续训末轮训练日志 reward=132.68、strict_recovered=42.36%、height_error=7.7 mm、ready_fraction=0.78%、辅助力=0、数值异常=0。训练指标包含随机化、采样动作和回合内曾恢复记录，不能作为上述确定性、无随机化评估的成功率。

## 本地证据

- [汇总JSON](../../artifacts/getup-r15-v3-evaluation-20261011/comparison.json)
- [12个交付文件SHA256](../../artifacts/getup-r15-v3-evaluation-20261011/file-hashes.json)
- [从头1401轮评估](../../artifacts/getup-r15-v3-scratch1401-evaluation-20261011/evaluation.json)、[普通视频](../../artifacts/getup-r15-v3-scratch1401-evaluation-20261011/replay-normal.mp4)、[地面视频](../../artifacts/getup-r15-v3-scratch1401-evaluation-20261011/replay-ground.mp4)
- [续训5000轮评估](../../artifacts/getup-r15-v3-resume5000-evaluation-20261011/evaluation.json)、[普通视频](../../artifacts/getup-r15-v3-resume5000-evaluation-20261011/replay-normal.mp4)、[地面视频](../../artifacts/getup-r15-v3-resume5000-evaluation-20261011/replay-ground.mp4)
- 各组目录另含 `replay-poster.png`、`render-record.json`、`sample0-pose-probe.json`。

四段视频各20秒，使用GPU回放记录的qpos，经MuJoCo `mj_forward` 渲染，不重新积分仿真；标题标注v3、固定轮数、样本0、assistance OFF。已查看两组海报，下载ZIP和12个文件SHA256全部通过。

## 可追溯信息

- 源码快照：`/root/gpufree-data/wheelleg-remote/releases/20261011T083634Z-getup-r15-pose-bias-removed`。评估前90个源码/资产文件与冻结清单一致。
- 接口：`chuanliantui-mjlab-getup-curriculum-r15-assisted-v3`，actor35/critic78/action6。
- 运行时：`/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python`，Python3.12、mjlab1.6、Torch2.14cu130、MuJoCo3.11、RTX4090。
- 从头模型：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_164219_836198_getup_r15_pose_bias_removed_scratch_2000/model_1400.pt`，SHA256 `63c2e0fee6486485b04667451e063656a583f911c8dfe88f0f72ccba59ab020f`。
- 续训模型：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_164219_970332_getup_r15_pose_bias_removed_resume_4000_to_5000/model_4999.pt`，SHA256 `759b36eaf0ec42ac340e814963836cccb06d352fe8124aff94e6f5db863d6ada`。
- 远端评估目录：`/root/gpufree-data/wheelleg-remote/evaluations/20261011T092719Z-r15-v3-scratch1401`、`/root/gpufree-data/wheelleg-remote/evaluations/20261011T092719Z-r15-v3-resume5000`，对应jobs同名目录内退出码均0。
- ZIP SHA256：`84ad39bedd0cf0c94f96628ee3882e82d3845eba9f4879bb18644d6f2a477517`。

本次只做评估、渲染和结果记录；未更改训练参数、未中止从头组、未启动新训练/TensorBoard、未提交Git。原始训练命令及TensorBoard命令见[启动记录](20261011-mjlab-getup-r15-v3-pose-bias-removed-training.md)。
