# 华南虎原始V14配方恢复检查

2026-10-11，Windows工作区 `D:/IMCA/RL/Wheel-Legged-Gym`，按用户明确选择恢复最早 `wheeled-legged_RL-main_huananhu` V14，而非 recover-r14。Windows迁移工作区没有 `.git`，分支查询失败；本次未初始化Git或创建分支／worktree，也未提交。详细配置和后端适配见 [接口说明](../interfaces/chuanliantui-mjlab-scut-v14-flat-v1.md)。

## 修改范围

- 新增 `mjlab_training/src/ct_mjlab/experiments/huananhu_v14.py`：独立固定奖励配方、V14命令、无向上辅助力、完整模型续训和原生回放。
- `mjlab_training/src/ct_mjlab/cli.py`：新增 `huananhu-v14` 并设为train/check默认；V14检查点原生回放自动识别。原有r14/r15显式入口保留。
- 更新AGENTS、COMMANDS、mjlab训练README、总说明和本接口／检查记录。
- 未修改机器人资产、reference_training历史内核、r15奖励／辅助力内核、部署代码或根logs。

## 远端有限检查

运行时 `/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python`，CUDA环境。最终发布目录 `/root/gpufree-data/wheelleg-remote/releases/20261011-v14-original-parameters-final`，检查目录 `/root/gpufree-data/wheelleg-remote/jobs/20261011-v14-original-check-final`。新目录复制已有冻结发布并仅覆盖CLI、加入V14模块；旧发布和模型不变。

1. 正式包入口 `python -m ct_mjlab.cli check --num-envs 20 --steps 64 --device cuda:0`，未指定profile，确认默认走V14。退出码0，Actor35/Critic78、奖励有限，数值异常0。完整日志 [cli-check.log](../../artifacts/v14-original-check-20261011/cli-check.log)。
2. 同运行时 `probe.py`，退出码0：核对全部26项非零权重、关键命令/PPO参数、固定模式槽；20环境64步有限物理。将实际native状态转换为此前通过原始奖励AST对照的V14 `Recipe` 输入，零速度/.22 m命令下26项加权奖励最大绝对差 `4.3655745685100555e-11`。另检查非零命令下速度项未乘高度门控。原华南虎源码目录在当前Windows及远端不可用，**本次没有重新执行原始源码AST检查**。
3. 构造原生PPO runner，在检查目录临时子目录保存／恢复零更新完整模型，验证完成更新计数仍为0；未调用learn或optimizer.step。实际半辅助力r15 model_999检查点被新V14校验拒绝。
4. 下载证据ZIP并验证ZIP及全部文件SHA256，检查JSON记录的CLI/V14模块哈希与Windows当前文件一致。ZIP SHA256：`6d7cd922926a5da18829abbcfa09ad6664e0b54b85b916a601cdeac5c826d931`。

结果见 [检查JSON](20261011-mjlab-scut-v14-profile-check.json)，全部证据在 [本地检查目录](../../artifacts/v14-original-check-20261011/check.json)。这不是训练行为验收，未证明起立、行走或真机效果。

## 原运行和训练授权

本次没有启动V14训练或TensorBoard，没有停止或更改半辅助力1000→2000冻结续训组。收尾时确认原任务日志已到1999/2000（索引从0计数），退出码0，即累计2000轮正常完成。本次未对该完成模型做新的回放评估。

新训练和TensorBoard命令在 [V14说明](../interfaces/chuanliantui-mjlab-scut-v14-flat-v1.md#命令)，仅交付命令；历史模型仍用原profile续训，禁止从r15隐式切换到V14。
