# 恢复华南虎原始 V14 Flat-v0 训练配方

2026-10-11，用户明确选择最早的 `wheeled-legged_RL-main_huananhu`，不是后来的 `mjlab/wheelleg_RL` recover-r14。

## 来源和入口

默认 CLI 配方改为 `huananhu-v14`，入口为 `mjlab_training/src/ct_mjlab/experiments/huananhu_v14.py`，模型接口 `chuanliantui-mjlab-scut-v14-flat-v1`。

使用本仓保存的 `scut_v14.json`，原任务 `Robotics-Wheelbipe-V14-Flat-v0`，来源提交 `868467c3427fc00d34ebb1176683652d9597ba1c`。该文件记录原配置字段、全部奖励权重及原 env/config/PPO 文件指纹。原始奖励曾逐项执行 AST 对照，见 [移植检查](../diagnostics/20261009-mjlab-migration.md)。Windows 没有原华南虎源码目录，本次使用已保存且经过审计的参数／奖励内核，没有重新对当前外部工作区取样。

## 实际配置

| 项目 | 原始 V14 配方 |
|---|---|
| 奖励 | `scut_v14.json` 中26项非零奖励，固定权重；无 getup_gym 三阶段奖励 |
| 线速度跟踪 | `exp(-clamp(error,-1,1)^2/0.5)`，权重 +1；移除 recover 配方的倒数二次公式 |
| 线速度误差平方 | `(0.25*error)^2`，权重 −1 |
| 偏航跟踪 | `exp(-clamp(error,-0.8,0.8)^2/0.25)`，权重 +1 |
| 偏航误差平方 | `(0.5*error)^2`，权重 −1 |
| 静止速度惩罚 | 命令绝对值小于0.1时，机体系水平速度绝对值之和，权重 −1 |
| 姿态 | 原 `flat_orientation_x/y_v` 各 −2，`flat_orientation_x/y_exp` 各 +1；无课程 orientation −30 项 |
| 高度 | 指数紧跟踪 +1、误差平方 −1；目标随机0.20～0.42 m |
| 速度奖励门控 | 不启用高度／姿态门控，包括原配置记录的未启用高度门控 |
| 失败 | 权重 −200；基座触地／倾角超过40°连续20控制步失败，数值异常立即终止；20 s超时 |
| 默认角、膝限位、双腿角度差 | 原 V14 对应奖励未启用；保留原轮位置 `no_fork` / `no_fork_square` 各 −1 |
| 前进／偏航命令 | ±2.7 m/s、±2π rad/s；50%航向控制，增益5；每5～15 s重采样 |
| 地面过渡命令 | 前1.5 s前进／偏航各±1，普通命令静止抽样约19% |
| 后期命令采样 | 恢复源实现：2000轮后30%冲刺，3000／4000轮后各15%自旋；不是整批扶正／站起／行走课程 |
| 模式槽 | 原始固定 `[1,0,0,0,0,0,0]`；恢复动态模式的 r14/r15 仍可显式选择 |
| 初始化 | 50%普通／50%地面，保留本机器人合法闭链姿态缓存；普通机体 roll/pitch±0.15 rad、yaw±π，地面零姿态 |
| 动作 | 四腿在前、两轮在后；腿位置倍率0.5，轮速倍率10、限速±100 rad/s |
| PPO | actor/critic 均256→128→64、ELU；无观测归一化，初始标准差1，24步 rollout |
| PPO 优化 | 学习率1e−4、自适应KL0.01，5轮更新、4 mini-batches；gamma0.99、lambda0.95；clip0.2、价值系数4、entropy0.005、梯度限幅1、clipped value loss |
| 默认规模 | 4096环境、20000轮、每500轮保存；CLI可覆盖环境数和轮数 |
| 向上辅助力 | 不施加；实体气弹簧继续工作 |

完整非零奖励及构造后的环境/PPO参数见 [检查JSON](../diagnostics/20261011-mjlab-scut-v14-profile-check.json)。RewardManager负责唯一一次控制周期0.02 s乘法，`termination=-200` 的单次加权贡献因此为 −4，而非未乘周期的 −200。

## 机器人和后端适配

机器人资产、安装几何、惯量、实体电机、名义零点、闭链约束、150 N实体气弹簧、限矩和PD60/2继续使用本仓设置。保留原生mjlab管理器、Actor35/Critic78，机器人映射仍为四腿两轮。普通根高度使用经过碰撞净空检查的0.38 m下界，地面0.25 m；不照搬华南虎机构的关节解。

物理步长保留本仓闭链验证过的2 ms×10，对应源5 ms×4的同一20 ms控制周期；传感／动作延迟按实际物理子步取整。MuJoCo摩擦／接触采用已有来源映射。以上是机器人与物理后端差异，因此参数恢复不等于复现原工程的训练结果。原高度命令0.20～0.42 m此次按用户选择恢复，其中部分目标超出本机器人已有静态可达范围，未增加适配裁剪。

## 历史兼容和运行状态

新增独立配方，没有改写 r14/r15 奖励或辅助力内核。现有模型仍按检查点标识使用原配方回放；续训必须显式选择其原 profile。原始 V14 新配方只允许自己的完整 actor/critic/optimizer 模型续训，拒绝 r14/r15 隐式迁入。

当前半辅助力1000→2000续训使用冻结发布目录，默认参数切换不会改变它。本次没有停止该组，也没有启动 V14 训练或 TensorBoard。

新 V14 模型的原生 `play --backend mjlab` 已接入自动识别。CPU `eval/play` 和现有通用 ONNX 导出暂未接入此新接口，会拒绝不匹配的模型；不要用 r14 CPU 配方替代新 V14 验证。配方恢复时尚无学习模型；随后用户授权的1000轮已完成，GPU评估脚本已支持此接口，回放未达到0.30 m起立目标，见 [model999评估](../diagnostics/20261011-mjlab-scut-v14-model999-evaluation.md)。

## 命令

在具备本仓独立 Python3.12/mjlab 环境的 Linux 仓库根目录执行，默认只交付命令：

```bash
scripts/mjlab.sh train --training-profile huananhu-v14 --num-envs 4096 --iterations 1000 --run-name scut_v14_flat
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui --port 8080

scripts/mjlab.sh play --checkpoint logs/mjlab/chuanliantui/<V14_run>/model_<n>.pt --backend mjlab --num-envs 20 --render
```

Windows本机没有训练运行时。本次远端新发布目录 `/root/gpufree-data/wheelleg-remote/releases/20261011-v14-original-parameters-final` 保存已检查的配方；它没有覆盖旧发布目录。检查位置和结果见 [检查记录](../diagnostics/20261011-mjlab-scut-v14-profile-switch.md)。
