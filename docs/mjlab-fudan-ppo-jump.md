# 复旦 PPO 跳跃任务

`fudan_ppo_jump` 从零学习行进中的蹬伸、腾空和收腿。训练场是平地。5 cm 和 10 cm 横条只在冻结模型后的 GPU 验收场出现。一次起跳的轮底高度、一次干净的越条和落地后的稳定行走分别计量。

## 可视预检

在仓库根目录，用新目录运行。输出 `precheck-20.mp4` 是同批 20 个 GPU 环境的正常速度回放，`precheck.json` 记录实际复位几何、奖励、训练缓存、反向计算与零参数更新。当前配方已生成[预检报告](../artifacts/fudan_ppo_jump/20261118-precheck-final/precheck.json)和[20 环境回放](../artifacts/fudan_ppo_jump/20261118-precheck-final/precheck-20.mp4)。

```bash
export PYTHONPATH=mjlab_training/src:.
export MUJOCO_GL=egl
PY=mjlab_training/.venv/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
CHECK=artifacts/fudan_ppo_jump/${STAMP}_precheck
"$PY" -m scripts.agent.fudan_ppo_jump.check --video --output "$CHECK"
```

## 训练与 TensorBoard

`JUMP_ITERS` 填本次预算轮数。以下默认 20 环境，与已验证的数据流一致，每轮采 48 拍。只使用已有预检时，可直接采用下方的 `PROOF`。每次启动选择新日志目录；同任务完整检查点可用 `--resume`，轮数预算为新增轮数。

```bash
PROOF=artifacts/fudan_ppo_jump/20261118-precheck-final/precheck.json
RUN=logs/mjlab/fudan_ppo_jump/$(date +%Y%m%d_%H%M%S)
scripts/mjlab.sh train --task fudan_ppo_jump --num-envs 20 \
  --iterations "${JUMP_ITERS:?请先设置本次预算轮数}" --precheck "$PROOF" --log-dir "$RUN"

# 在另一终端将 RUN 设成同一次运行的实际目录
mjlab_training/.venv/bin/python -m tensorboard.main --logdir "$RUN"
```

先看 `Jump/takeoffs_total` 和 `Jump/landings_total`。起跳要求先有双轮承重，再向上腾空。对已经落地的跳跃，`Jump/airtime_s` 统计累计平均腾空时长，`Jump/peak_wheel_clearance_m` 统计两轮较低轮底的平均峰值，`Jump/stable_landing_rate` 统计落地后双轮持续支撑约 1 秒且姿态稳定的比例。重启训练进程时专项计数从零开始；没有落地样本时这三项留空。机身高度奖励目标 0.65 m 是根坐标高度，轮底离地高度由两轮实际碰撞网格单独计算。

## 越障采集与验收

训练后在同一终端执行。第一条命令用 MuJoCo Warp GPU 采集两个高度各 20 例，保存实际命令、触发距离、接近速度、每毫秒的轮底和接触、每例 25 fps 姿态及各类结果的代表视频。第二条命令离线生成 `acceptance.json` 和 `README.md`。

```bash
EVAL=artifacts/fudan_ppo_jump/$(date +%Y%m%d_%H%M%S)_bars
MUJOCO_GL=egl PYTHONPATH=mjlab_training/src:. mjlab_training/.venv/bin/python \
  -m scripts.agent.fudan_ppo_jump.evaluate --checkpoint "$RUN/model_final.pt" --output "$EVAL"
scripts/mjlab.sh accept --task fudan_ppo_jump --recording "$EVAL" --output "${EVAL}_accept"
```

- 两档复用相同的 20 个固定初态。6838 行走模型以 2.0 m/s 指令正向接近。理想测距沿车头水平投影检测横条前沿，命中且距离小于 0.9 m 时切入 Jump，48 个策略步后切回行走，每例只触发一次。
- 横条厚 5 cm、横向宽 2 m。双轮须在横条宽度内腾空跨过前后沿，全车无碰条，远端双轮着地后继续行走 1 秒。绕行、单轮搭过去、提前落地、机身撑地、跌倒和缺测均计失败。
- 按高度查看 `details.by_height`。其中 `jump_passed` 统计完整越障，`technical_passed` 另要求落地后第 0.5～1 秒通过统一巡航门槛。逐例原因在 `samples` 中。缺测保留在每档 20 例的分母内。起立单列为未测。
- 接口验证使用随机初始 Jump 网络和冻结 6838，见[采集链检查](../artifacts/fudan_ppo_jump/20261118-routing-final/routing.json)。该记录用于验证切模、碰撞传感器、视频和验收入口。正式行为验收需要训练后的 Jump 模型。

## 配方来源与本机适配

固定复旦提交 `8204e853dfd2ed06d85a322e1a998c3d20a3be2c`，使用 Jump 目录的实际生效配置。16 项奖励逐项先乘权重和策略周期，再裁剪到 ±0.025。两轮当前和上一策略拍均无轮载才触发上游腾空奖励。保留 `encourage_jump` 的原始高度积分和清零次序，另建实际起跳计数。[训练配置](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/jump/wheel_legged_gym/envs/base/legged_robot_config.py) · [奖励实现](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/jump/wheel_legged_gym/envs/base/legged_robot.py)

本机使用实体闭链六电机，四腿沿用 6838 的 20/1 增益；物理、PD、策略周期分别为 1、5、10 ms。两腿均以机体系 XZ 平面的髋轴到轮轴距离定义腿长。上游 0.16 m 收腿与 0.31 m 蹬伸在当前 XML 均有合法闭链解，直接使用米值。训练采用本机机身高度指令 0.22～0.30 m 和实体复位，关闭外推与额外延迟。验收采用无观测噪声、无外推的名义条件。上游虚拟腿模型及其权重独立于本机策略。[输入、动作、时序与验证记录](interfaces/20261118-fudan-ppo-jump.md)

## 落地专项配方

训练入口增加 `--jump-landing`，用于本机闭链 Jump 的下降伸腿、双轮先承重和触地缓冲。新配方使用专门预检，允许从旧 Jump 检查点完整恢复学习状态。候选模型的 GPU 回放自动使用轮触地后 200 ms 交接。给旧模型的回放命令加 `--landing-handoff`，可跑同交接对照。原定时回放仍用于历史模型的默认复现。[本次配方、训练与 TensorBoard 命令](../artifacts/fudan_ppo_jump/20261119-landing2200-to3200-01/README.md)

止沉配方通过 `--jump-stop-sink` 启用，从落地配方完整恢复学习状态，单独预检后训练。只在落地缓冲中双轮实际同时承重后惩罚继续下沉、机身净空太低和双轮支撑丢失。[本次参数、训练与 TensorBoard 命令](../artifacts/fudan_ppo_jump/20261119-stop-sink3200-to4200-01/README.md)

承重前预伸配方通过 `--jump-preload` 从止沉配方续训，奖励下降末段的有效伸腿目标与首次轮承重的伸展力矩。80 ms 平地回放和两档横条用相同交接判据。[本轮预算、命令与测量口径](../artifacts/fudan_ppo_jump/20261119-preload4200-to5200-01/README.md)

## 双轮峰值专项

`--jump-peak` 从承重前预伸检查点完整续训。每次主动起跳中，按较低轮底峰值的平方增量给分，替代固定的机身腾空高度奖励。计分要求该策略拍内双轮及机身均无地面接触，并避开落地缓冲。沿用原落地奖励、40 N·m 腿电机限矩和控制周期。`Jump/highest_landed_peak_m` 记录本次进程中已着地跳跃的最高双轮轮底高度。

专项预检使用 `scripts.agent.fudan_ppo_jump.check --jump-peak --resume <预伸完整检查点> --output <新目录>`。训练使用上面的同一入口，加 `--jump-peak --resume <预伸完整检查点>` 并使用本专项预检报告。训练后用固定初态的 GPU 平地回放测双轮峰值和落地，赛场 27 cm 横梁单独测过梁净空。

`--jump-peak-additive` 同时保留承重前预伸配方的机身腾空高度项和双轮峰值增量项。与预伸配方逐项核对，唯一的训练参数差异是新增 `wheel_peak_gain=64`。20 环境专项预检和正式训练都使用 `--jump-peak-additive`。
