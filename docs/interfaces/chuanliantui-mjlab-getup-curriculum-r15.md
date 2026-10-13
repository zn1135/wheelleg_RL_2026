# r15：整批环境的扶正、站起、行走课程

用户选择采用此前研究表中的三组奖励，并明确按训练课程切换整批环境。默认 `ct-mjlab train` / `scripts/mjlab.sh train` 已采用此配置；`--training-profile recover-r14` 显式选择历史r14。实现为 `ct_mjlab/experiments/getup_curriculum.py`，历史 `reference_training/` 内核不变。

2026-10-11加入向上辅助力并修正轮接触判定（v2）；随后移除默认角惩罚（v3）。当前开发接口为 `chuanliantui-mjlab-getup-curriculum-r15-assisted-v3`。此前模型与冻结源码保留，4000轮结果属于v2；v3已启动从头2000轮和4000→5000轮续训，尚未验证学习效果。

## 来源与奖励

来源固定为 [getup_gym 配置](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/envs/bipedal_wheeled/config.py) 的 `reward_group_2/3/4`，公式见 [奖励实现](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/common/reward_functions.py)。本文表格是三个实际使用组的完整权重，空项贡献为零。

| 项目 | 扶正 | 站起 | 行走 |
|---|---:|---:|---:|
| 高度跟踪 | 4 | 6 | 5 |
| 姿态 | -2.6 | -2 | -15 |
| 倒置判别 | -0.6 | -1 | 0 |
| 关节偏离默认角 | 移除 | 移除 | 移除 |
| 左右腿不对称 | -0.02 | -0.02 | 0 |
| 双轮未同时接地 | -2 | 0 | -0.2 |
| 轮端承重 | 4 | 0 | 0 |
| 前进速度跟踪 | 0 | 0 | 7 |
| 偏航速度跟踪 | 0 | 0 | 4 |
| 失败 | -200 | -100 | -500 |
| 电机加速度 | -2.5e-8 | -2.5e-6 | -2.5e-6 |
| 力矩平方 | 0 | -1e-6 | -1e-5 |
| 腿电机速度平方 | 0 | 0 | -0.004 |
| 动作一阶变化 | -0.06 | -0.02 | -0.1 |
| 动作二阶变化 | 0 | 0 | -0.02 |
| 轮间距越界 | 0 | 0 | -2 |
| 速度区间 | 0 | 0 | 1 |
| 主动腿电机软位置限位 | 0 | 0 | -5 |
| 电机软力矩限位 | 0 | 0 | -0.05 |
| 电机功率 | 0 | 0 | -1e-4 |

- 高度：`exp(-abs(h - h_target) / 0.12)`，无旧±0.15 m误差截断；10 cm误差时约0.435。
- 姿态：`0.02*(gx²+gy²) + 2*(gz+1)² - clamp(exp(-(gx²+gy²)/0.25), 0, 0.1)`。正立与倒置明确不同，保留来源公式的负常数项。
- 轮承重：`exp(-0.05*abs(sum(abs(Fwheel_z)) - total_mass*10))`；质量从每个环境的当前机器人刚体读取，包含质量随机化。
- 线速度：`exp(-((vx_cmd-vx)²+vy²)/0.12)`；行走阶段另要求根高度大于0.24 m。偏航：`exp(-(yaw_cmd-wz)²/0.12)`。
- v3按用户“不要这个奖励”的要求，三个阶段均移除 `pose_bias`：权重字典、奖励计算和管理器注册均删除，不再惩罚偏离固定CAD默认角。左右腿对称项是另一种约束，保留原权重及公式。历史v2的默认角惩罚权重是-0.06/-0.14/-0.12，左右非对称且默认角几何站高约0.337 m；它与0.27/0.30 m目标存在竞争。这是对原三组奖励的明确调整，并非全部照搬来源公式。
- 加速度按两次控制周期速度差计算，复位清空历史，避免复位瞬间产生虚假的加速度惩罚。
- 权重在奖励函数内按整批阶段选择，管理器外层权重为1，仅管理器乘一次20 ms控制周期；没有额外总奖励裁剪。

## 整批阶段推进

初始阶段始终是扶正，不按每个环境的normal/recover槽分别选择奖励。三阶段单向推进；换阶段后个别机器人再次倒地，动态normal/recover槽仍会变化，但不会让整批训练回退。

| 阶段 | 高度命令 | 速度命令 | 推进条件 |
|---|---|---|---|
| 扶正 | 0.27 m | 前进／偏航均0 | 至少训练100次PPO更新，且至少2/3环境连续2秒符合下述条件 |
| 站起 | 0.30 m | 前进／偏航均0 | 同上，计时从进入站起阶段重新开始 |
| 行走 | 0.30 m | r14采样：前进±2.7 m/s、偏航±2π rad/s、50%航向控制 | 最终阶段 |

条件：高度误差≤3 cm、机体总倾角≤15°（重力z判别）、双轮当前向上接触力各>5 N、机身接触力≤1 N、没有数值异常，且当前辅助力≤该环境整机重量的5%。全批比例连续不足会清零2秒计时。100更新对应每环境2400控制步，避免刚复位的短暂高位立即推进课程；未达标就继续当前阶段，不因固定轮数强行进入行走。

阶段更新在本拍奖励完成后运行，新权重和新高度从下一拍同时生效；阶段名、目标高度、切换历史及阶段内计数写入检查点。指标含 `getup_stage`（0/1/2）和 `getup_ready_fraction`。

## 针对机器人和平台的适配

- 原项目的0.30→0.45→0.60→0.75 m及整批高度统计切换不能直接套在此机构上。r15采用0.27→0.30→0.30 m及上述稳定条件。这是适配方案，不声称完整复现原训练结果。
- 动态normal/recover、Actor35/Critic78/Action6、四腿两轮顺序、50%普通／50%地面初态、闭链、150 N气弹簧、PD60/2和PPO沿用r14。
- 轮间距以求解后的CAD名义姿态 **0.426506 m** 为中心，容差±3 cm，替换原机器人0.32～0.38 m阈值；速度奖励高度阈值0.6 m按本机器人最终高度比例改为0.24 m，低位功率阈值0.7 m改为0.28 m。
- 软位置限位仅作用于四个实体主动腿电机，使用本模型限位内90%区间；原先关闭的被动膝 `knee_limit` 没有加入新奖励组。
- 前两阶段允许机身接触和倾斜探索至20秒回合结束，仅数值异常即时重置。行走阶段恢复原20步（0.4 s）机身接触／40°倾斜失败判定。-200/-100权重并不表示正常贴地探索必然被扣分；超时不是失败。数值异常继续沿用有限奖励保护。
- 按用户后续要求加入向上辅助力，具体公式和退坡见下一节。机器人质量、机构、接触与仿真引擎仍不同。
- r15只能从自身完整检查点续训；拒绝隐式加载r14或四组消融检查点。回放识别检查点中的阶段并冻结课程。旧r14及更早回放入口沿用历史实现。

## 命令

在独立Linux Python3.12/mjlab环境中、仓库根目录执行。Windows迁移目录没有本机mjlab运行环境；当前辅助版本的远端验证副本为 `/root/gpufree-data/wheelleg-remote/releases/20261011-getup-r15-assist-check`。以下是可执行命令，尚未启动训练或TensorBoard。

```bash
# 有限检查，无PPO更新。
scripts/mjlab.sh check --num-envs 20 --steps 64
mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_getup_curriculum.py --num-envs 20 --steps 64

# 新配置训练；只有用户授权才执行。
scripts/mjlab.sh train --num-envs 4096 --iterations 1000 --run-name getup_r15
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui --port 8080

# 相同r15 v3配置的续训，iterations是追加更新数。
scripts/mjlab.sh train --num-envs 4096 --iterations 1000 --resume logs/mjlab/chuanliantui/<r15-run>/model_999.pt
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui --port 8080

# r15 GPU回放；高度/奖励阶段来自模型，不推进课程。
DISPLAY=:0 scripts/mjlab.sh play --checkpoint logs/mjlab/chuanliantui/<r15-run>/model_999.pt --num-envs 20 --render

# 明确选历史r14训练配置。
scripts/mjlab.sh train --training-profile recover-r14 --num-envs 4096 --iterations 1000 --run-name historical_r14
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui --port 8080
```

r15原生CPU行为评估和ONNX交付尚未适配；历史导出器会拒绝新接口，不能借用旧模型契约跳过校验。当前工作仅涉及训练配置和GPU回放。

## 验证结果

远端RTX4090、既有Python3.12/mjlab1.6运行时：20环境，每个阶段64步，观测形状35/78、奖励有限，三个阶段数值异常均0。整批比例阈值、最短阶段时长、2秒连续计时、单向推进、同拍幂等、奖励与dt只乘一次、阶段进度保存／恢复、拒绝历史模型隐式续训均通过。已恢复检查点阶段与高度命令；检查期间没有PPO更新。

完整结果见 `artifacts/getup-r15-validation-20261010/check.json`。这些是配置、有限仿真和恢复流程检查，尚未验证学习效果或起立成功率。

## 2026-10-11向上辅助力

参照[原辅助力实现](https://github.com/2350575870/getup_gym/blob/238d2962a0fee8e3da4da914cadcea2be00ce402/getup_gym/envs/legged_robot.py#L1182)的高度差比例和3000更新线性退坡。实现独立保存在 `ct_mjlab/experiments/pull_up_assistance.py`，每个2 ms物理步在父类电机、气弹簧和扰动力写入后叠加到机身质心的世界+z方向，无额外辅助力矩。

```text
W = 每环境机器人总质量 × |gravity|
height_ratio = clamp((当前阶段目标高度 - 当前高度) / 当前阶段目标高度, 0, 1)
time_ratio = clamp(1 - 累计PPO进度 / 3000, 0, 1)
Fz = W × max_weight_ratio × height_ratio × time_ratio
```

- 2026-10-11起CLI新建训练的上限采用该环境 **0.5倍整机重量**，按随机化后的总质量更新。历史assisted-v1/v2/v3与orientation-v1模型续训保留原1倍上限及累计衰减进度；超过3000轮的辅助力仍为零。新配方使用独立half-assistance接口，见[修改说明](../diagnostics/20261011-mjlab-half-assistance.md)。原机器人400 N是其配置，未直接套用到本机。
- 仅扶正和站起阶段启用；高度达到目标时归零，1500次更新时预算减半，3000次更新时完全退出。力随子步状态变化，不是保持上一拍不变的推力。
- 行走阶段、默认GPU回放均为零；回放用于查看无辅助行为，效果仍需独立评估。
- 地面初态前0.2秒沿用电机预热／落地保持，辅助力也为零，避免电机尚未启用时提前托起机身。
- 与原局部扰动力及力矩组合，每步由父类重新写入原扰动，再加入辅助力，防止累计和覆盖。复位清空对应环境的辅助力。
- 课程达标增加 **辅助力≤0.05 W** 条件。它允许极小剩余辅助，因此课程推进条件不等价于完全无辅助的起立验收。
- 指标增加 `pull_up_force_n`（每环境当前辅助力）和 `pull_up_budget_ratio`（当前重量比例预算）。配置、辅助内核哈希、累计PPO更新数一起进入检查点，续训继续退坡。
- 原代码还有roll辅助力矩；本次用户要求加入辅助力，采用纯向上质心力。未复刻该姿态力矩。
- 加辅助版本与原无辅助r15有独立接口校验，拒绝隐式混用检查点。

验证：20环境三阶段各64步，有限观测／奖励、数值异常0；扰动保留、世界向上、力不累积、复位清除、1500减半、3000归零、行走和关闭状态归零均通过；续训进度用临时合成计数检查，未执行PPO更新。结果保存于 `artifacts/getup-r15-assistance-validation-20261011/check.json`。本轮没有启动训练。

## 2026-10-11轮承重方向修复与续训

后续独立姿态加权配置见 [r15 orientation收紧与分析](../diagnostics/20261011-mjlab-r15-orientation-weight-analysis.md)：用户选择仅增大权重，三阶段为-30/-30/-30，原公式与默认v3保持独立。从v3续训需要显式 `--training-profile getup-r15-orientation --resume-orientation-weight`，不允许隐式混用模型。本节以下v2接触修复仍是历史记录。

实际GPU回放测得双轮承重时该mjlab传感器世界Fz为负，v1使用 `Fz>0` 判定接触，造成课程达标比例始终0、no_fly持续惩罚、pose_bias被屏蔽。v2统一以 `clamp(-Fz,min=0)` 作为向上支撑力，供课程>5 N、双轮奖励>1 N和轮载荷奖励使用；反向力不视为承重。奖励权重、姿态／高度门槛、机器人和PPO参数沿用原配置。

历史v2默认严格拒绝v1。只有在原v2冻结源码内执行 `train --resume <v1-model> --resume-contact-force-fix` 才允许已知v1源码的接触迁移；当前v3拒绝该历史标志。以下v2命令与验证是历史记录，不能直接在当前v3入口执行。

```bash
scripts/mjlab.sh train --num-envs 4096 --iterations 1000 --resume logs/mjlab/chuanliantui/<assisted-v1-run>/model_999.pt --resume-contact-force-fix --run-name contact_fix_resume_1000
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui --port 8080
```

当前任务用户已授权从v1已完成1000轮的模型追加1000轮，累计2000轮。实际远端目录、修复检查与启动证据见 [续训记录](../diagnostics/20261011-mjlab-getup-r15-contact-fix-training.md)。旧v1模型按冻结的原训练源码回放；新v2模型使用当前入口。
