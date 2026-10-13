# r15 v3：移除默认角奖励并启动两组训练

## 最终要求和修改

用户指出默认角惩罚可能有问题，随后明确“不要这个奖励”。最终修改为三个阶段均删除 `pose_bias`，包括权重、奖励计算和管理器注册；不保留最初草案中的行走惩罚。左右腿对称奖励、控制/观测的默认角、初态、其他奖励、PPO、150 N气弹簧和辅助力规则保持原配置。

原v2公式将左腿平方和与右腿绝对值和相加后整体平方，左右交换后的代价不一致。固定CAD默认姿态的双轮几何触地站高约0.337 m（碰撞mesh顶点求得左0.337016 m、右0.337002 m），高于扶正0.27 m和站起0.30 m命令。固定姿态惩罚存在与目标的竞争，但这尚不能证明它是训练失败的唯一原因；新配方效果需训练后独立回放确认。

开发接口为 `chuanliantui-mjlab-getup-curriculum-r15-assisted-v3`。历史4000轮模型使用原v2冻结源码回放。当前v3只允许从已知v2模块 SHA256 `41d395799656b1473e10a72d6755b54f793913e1b2bece6278547a3d8d230746` 显式 `--resume-pose-bias-fix` 迁移：只允许环境配置删除标准 `pose_bias` 奖励，其他配置必须相同。未知旧源码、隐式跨版本、其他环境差异均拒绝。v1接触迁移仍需原v2冻结源码。

## 修改检查

远端原独立mjlab运行环境：20环境，三阶段各64步，观测/奖励有限、数值异常0。确认三个阶段及奖励管理器没有该项，奖励dt只乘一次、课程推进/恢复、接触方向、辅助力组合/退坡和关闭回放仍通过。

另外用实际4096环境加载累计4000轮源模型，不执行PPO更新，确认actor、critic、Adam完整状态逐项一致，课程和累计更新数完整保留，已归零的辅助力仍为0。

- [有限检查](../../artifacts/getup-r15-pose-bias-removed-validation-20261011/check.json)
- [实际模型迁移检查](../../artifacts/getup-r15-pose-bias-removed-validation-20261011/migration-check.json)
- [90个文件的冻结哈希](../../artifacts/getup-r15-pose-bias-removed-validation-20261011/source-state.json)
- 最终验证作业：`/root/gpufree-data/wheelleg-remote/jobs/20261011T083634Z-getup-r15-pose-bias-removed-check`，检查和迁移退出码均0。
- 最终冻结源码：`/root/gpufree-data/wheelleg-remote/releases/20261011T083634Z-getup-r15-pose-bias-removed`。
- 初始“前两阶段关闭、行走保留”的草案只做过有限检查，没有训练，已标为 superseded；不能使用其验证记录代替最终配置。

三个最终验证记录下载哈希核对一致；当前本地90个源码/资产文件与最终冻结哈希完全一致。验证本身不证明已经学会扶正。

## 两组启动

用户要求“重新训练2000轮，+续训1000轮”。北京时间2026-10-11 16:42:06启动，4096环境、seed42，在远端RTX4090同时运行。

| 项目 | 从头组 | 续训组 |
|---|---|---|
| 起点 | 随机权重，0更新 | v2的model_3999.pt，累计4000更新 |
| 本次更新 | 2000 | 1000 |
| 最终累计 | 2000 | 5000 |
| 辅助力 | 从0轮衰减，2000轮预算约剩1/3 | 全程0，继承累计进度 |
| 优化器 | 初始Adam状态为空，学习率1e-4 | 完整恢复Adam，学习率1.5e-5 |
| 初始课程 | 扶正，阶段步数0 | 扶正，阶段步数96000 |
| 首次更新模型 | model_0.pt，完成1更新 | model_4000.pt，完成4001更新 |
| 最终预计模型 | model_1999.pt | model_4999.pt |

实际两组首拍检查点确认默认角奖励已删除。从头组的优化器和课程从零开始；续训组actor/critic/Adam与源模型逐项完全一致。两组首次更新actor已变化，actor/critic参数有限。核对时日志分别28/2000、4029/5000，两组总显存5215 MiB、利用率99%。共享GPU时约2.1秒/轮，续训组预计约35分钟、从头组约70分钟；另一组完成后速度会变化。

源模型 SHA256：`3be163642e65481696e740a72d4f6b764cf214f2ba5ba5b24da6162f25016e50`。详细命令、实际输出路径、检查点哈希、起点和恢复证据见 [完整启动记录](20261011-mjlab-getup-r15-v3-two-groups-start.json)，记录SHA256：`f4a5ca429b5955ea375d962c7af38bf50ecbe0db89a9a11ffe5a5cb804478c08`。

## 产物目录

从头组：

- 模型/TensorBoard：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_164219_836198_getup_r15_pose_bias_removed_scratch_2000`。
- 作业日志：`/root/gpufree-data/wheelleg-remote/jobs/20261011T084138Z-getup-r15-v3-scratch-2000/output.log`，后台启动器14382。

续训组：

- 模型/TensorBoard：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_164219_970332_getup_r15_pose_bias_removed_resume_4000_to_5000`。
- 作业日志：`/root/gpufree-data/wheelleg-remote/jobs/20261011T084138Z-getup-r15-v3-resume-1000/output.log`，后台启动器14383。

两组均在退出时写入作业目录 `.exit-code`，与SSH会话分离。历史模型/logs保留，没有改部署代码、启动TensorBoard或提交Git。

## 已执行命令和TensorBoard

以下是复现记录，不要重复启动作业。两条训练命令均在最终冻结源码目录执行，环境 `PYTHONPATH=$PWD/mjlab_training/src`、`MUJOCO_GL=disable`，清除 `WHEELLEG_HUANANHU_INIT_CHECKPOINT`。

```bash
cd /root/gpufree-data/wheelleg-remote/releases/20261011T083634Z-getup-r15-pose-bias-removed
export PYTHONPATH="$PWD/mjlab_training/src"
export MUJOCO_GL=disable
unset WHEELLEG_HUANANHU_INIT_CHECKPOINT
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u -m ct_mjlab.cli train --training-profile getup-r15 --num-envs 4096 --iterations 2000 --seed 42 --device cuda:0 --run-name pose_bias_removed_scratch_2000
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u -m ct_mjlab.cli train --training-profile getup-r15 --num-envs 4096 --iterations 1000 --seed 42 --device cuda:0 --run-name pose_bias_removed_resume_4000_to_5000 --resume /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_151639_260563_getup_r15_contact_fix_resume_2000_to_4000/model_3999.pt --resume-pose-bias-fix
```

TensorBoard可在远端两个额外终端分别启动，本次未启动：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m tensorboard.main --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_164219_836198_getup_r15_pose_bias_removed_scratch_2000 --host 127.0.0.1 --port 8080
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m tensorboard.main --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_164219_970332_getup_r15_pose_bias_removed_resume_4000_to_5000 --host 127.0.0.1 --port 8081
```

本机端口转发：

```bash
ssh -p 30702 -N -L 8080:127.0.0.1:8080 -L 8081:127.0.0.1:8081 root@183.147.142.40
```
