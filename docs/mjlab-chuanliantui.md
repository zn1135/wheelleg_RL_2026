# mjlab 串联腿训练（multimode 单线）

2026-10-21轮端模型更新：训练与CPU/real2sim共用mjlab/chuanliantui.xml（旧multimode路径为链接），已写入joint_sharp的等效armature/阻尼/摩擦及驱动G/V/p。训练动作执行与固定增益后端共同实现该响应，标称参数取自XML，原随机化继续叠加；未训练，不保证跨工况gap消除。当前检查和复现见[接入报告](diagnostics/20261021-h7-wheel-response-integration.md)。

mjlab 训练侧只保留一条线：华南虎 multimode 任务，基线模型 `model_3023.pt`。2026-10-15 起 V14、r5～r15、recover、V11 等旧配方已删除，代码与文档可从分支 `backup/gpt-structure-20261015`（`e451b3d`）或其父提交 `109da6d` 找回。

## 接口

| 项目 | 内容 |
|---|---|
| 任务名 | `chuanliantui`（原 `Chuanliantui-Huananhu-Multimode-v0`） |
| 观测契约 | `chuanliantui-huananhu-multimode-39-82-v1` |
| Actor／Critic／动作 | 39／82／6 |
| 动作顺序 | `[lf0, lf00, lfwheel, rf0, rf00, rfwheel]` |
| 时序 | 2 ms 物理、20 ms／50 Hz 策略 |
| 地形 | 无限平地（2026-10-17 起；此前为生成地形 flat 50%、slope 20%、step 15%、stair 15%） |
| 命令 | 前向速度 ±1.5 m/s、偏航角速度 ±1.5 rad/s、机身高度 0.2165～0.3664 m（对应近直立髋轮腿长约0.18～0.33m；2026-10-22待续训验收） |
| 初态 | 20% 普通／80% 地面（2026-10-15 起，原 50／50），姿态库 `resources/robots/chuanliantui_new_1/mjlab/huananhu_pose_bank.npz` |
| 基线 checkpoint | `artifacts/multimode-v1-replay-20261014/models/model_3023.pt`，SHA256 `624ded630635f0fb51a7378cdd04a7350eee98ffb305acbe1104c3cc7705a88a` |

观测槽、模式与标志位见 [multimode 接口](interfaces/chuanliantui-huananhu-multimode-replay-v1.md)。

2026-10-22：曾按作者确认的髋轴到轮轴约0.15m最低腿长，将共享XML膝关节屈膝限位扩展到
左−0.873677593512／右+0.873677593512rad；作者随后要求先恢复，当前已回到左−0.77／右+0.77rad的屈膝限位，
对应姿态缓存恢复原版本，当前最短腿长为0.171632642m。
腿长为机体XZ平面髋轮距离，排除轮轴横向偏置；竖直双轮着地时0.15m腿长对应机身高度约0.186447m，
此前据此将高度命令下限设为0.1865m；当前机械范围已不支持0.15m腿长，该历史训练保持停止。
随后作者明确只准备起立后零速保持续训，并最终指定**腿长0.18～0.33m**：当前高度命令按原机械范围换算为0.2165～0.3664m，低位膝限惩罚渐减及恢复门槛保持兼容，39/82维控制定义不变。
30%回合从启动起保持零速，新增站稳后的低速保持奖励及20秒/5厘米位移验收；[范围、有限检查与命令](interfaces/20261022-huananhu-zero-hold-training.md)。尚未启动本轮训练。
H7仍限制高度0.27～0.34m，不能直接投入低位实机控制。
范围、验证与训练预算见[低腿长接口记录](interfaces/20261022-huananhu-low-leg-length.md)。

## 代码结构

与 Isaac 侧 `wheel_legged_gym/` 同样分工，见 [mjlab_training/README.md](../mjlab_training/README.md)。环境源码逐字节取自 `model_3023` 训练时的冻结源码快照，只改了 `model.py` 的资源路径；训练 XML 与姿态库同样取自该快照，SHA256 分别为 `dfe01afb…` 与 `ca82470a…`。

2026-10-15 起有意偏离快照之处（其余仍逐字节一致）：地面初态比例提为 `multimode_mdp.MultimodeCommandCfg.ground_start_probability`（默认 0.8，快照原为写死的 0.5），复位事件经 `multimode_env_cfg.py` 以参数传入；该值随每次训练的 `params/env.yaml` 记录，续训不会隐式沿用旧比例。中途倒地恢复相关的偏离见下一节。

## 中途倒地后重新起立（2026-10-15 起，2026-10-16 修订）

快照里 `recover` 模式只由复位时的 `ground_start` 决定，且真倒地会在 0.4 秒后终止回合，所以「被弄倒后再自己起来」从未被训练或测量过。现在：

- **recover 会重新触发**：`recover = (ground_start | getting_up) & ~recovered`。`recovered` 在倒地期间被清空；`getting_up` 在倒地时置位、在机器人重新稳定 0.5 秒后清除，因此普通初态的中途恢复与地面初态起立覆盖同一个"站起来"阶段（若只看 `fallen`，通道会在机器人刚不触地时就熄灭）。倒地判定沿用终止规则里的量（机身接触或倾角 >40°）。
- **恢复窗口 2.0 秒**：`MultimodeCommandCfg.recovery_window_s`。窗口内连续倒地不判失败；计时在"机身不再触地且倾角回到 40° 以内"时归零（不只是离地），所以持续有进展的回合不会被掐断，但反复跨越该门槛的翻滚可以一直拖住超时。数值异常与越出地块仍立即终止。**取值依据（2026-10-16 更正）**：原值 1.0 s 的理由是姿态库地面初态起立只占 recover 通道 0.72～0.98 秒，但那是合法躺姿、不是 180° 倒扣。按冲量时刻起算实测（384 回合、seed 43），真被掀翻后的起身中位用时随强度在 1.12～2.30 秒之间、p90 2.32 秒、最长 3.98 秒，1.0 秒会把真实起身掐断在途中；2.0 秒也与验收判据自身给出的恢复时限一致。
- **恢复期接触惩罚门控**：`undesired_contact` 只在 `fallen` 期间置零，一站起来立刻恢复。既不给「爬起来必然触地」加反向梯度，又保留对 2026-10-12 审计发现的失败模式（正立但靠非轮部件撑地）的惩罚——那种姿态不属于 `fallen`。
- **训练不再主动掀翻**：按用户 2026-10-16 的最新要求，删除训练事件 `huananhu_knockdown`，不再每4～8秒施加18～28 rad/s的绕轴角速度冲量。倒地后重新进入 recover、2秒恢复窗口、恢复期奖励门控与80%地面初态保留。
- **普通训练扰动保留**：`push_root_velocity`（±0.6 m/s）与 `sample_local_wrench`（±20 N／±2 N·m）保持当前配置；自然发生跌倒时仍可恢复。
- **手动回放／检查仍可击倒**：`play --knockdown-every`、`eval --knockdown-at` 和机制检查中的手动冲量继续保留；`KNOCKDOWN_ANGULAR_SPEED_RANGE=(18.,28.)` 只作为手动回放与检查的强度参照，不是训练事件。
- **观测契约不变**：仍是 39／82／6。策略靠 `extra_flags()` 里的 `ground_start` 区分「复位起立」与「中途倒地起立」，因此不需要改 H7 接口。

新指标：`mode_knockdown`（每回合实际倒地次数）与 `recovery_timeout`（因窗口超时被终止的回合占比；接近 1 说明是窗口而非策略在结束起立尝试）。

**2026-10-16 的重要更正**：`model_4022` 那次续训落盘的 `params/env.yaml` 里**既没有 `recovery_window_s`、也没有 `huananhu_knockdown`**，两个推扰事件的 `params` 都是空字典（用的是函数默认的 ±0.25 m/s 与 ±10 N／±1 N·m）。所以此前"训练配方自 2026-10-15 起含恢复窗口与中途击倒"说的只是**源码**；中途倒地恢复这个能力到 2026-10-16 为止**没有被任何一次训练覆盖过**。2026-10-16 02:27:29 起按上述配方从 `model_4022` 续训（2048 环境 × 1000 轮）是第一次；训练前预检通过（32 环境、12 秒内 104 次击倒触发）。

训练配置不注册主动击倒事件。机制检查：`mjlab_training/.venv/bin/python scripts/agent/check_multimode_recovery.py` 通过手动冲量验证恢复模式重新触发、恢复窗口与奖励门控；加 `--training-config` 检查训练确实未注册主动击倒事件且恢复窗口仍保留。两者都不执行学习，不替代行为验收。

## 可执行命令

**当前最终平衡判据（2026-10-16 修订，v3.1）**：预定结束前连续 1 秒双轮承担80%～120%重量、每轮至少10%且≥5N、机身不撑地、倾角≤35°、横滚/俯仰角速度合量≤0.5 rad/s；允许前后移动、转向与少量腿部接触，不要求高度与恢复用时。上文末 5 秒≥95%的数字均为历史 v2。新测量读取 `final_balanced`，旧 `last5_*`／`recovered` 只作对照，详见[当前验收 v3](interfaces/chuanliantui-wheel-balance-acceptance-v2.md#当前标准最终双轮平衡-v32026-10-16)。

均在仓库根目录执行。环境安装：`scripts/setup_mjlab.sh <已装 mjlab 的 python>` 或 `uv sync --project mjlab_training --locked --python 3.12`。

```bash
# 有限检查（不训练）：加载模型、核对 39/82 维、步进若干次
scripts/mjlab.sh check --num-envs 2 --steps 16 \
  --checkpoint artifacts/multimode-v1-replay-20261014/models/model_3023.pt

# 训练（从零）；--resume 从 multimode checkpoint 完整续训
scripts/mjlab.sh train --num-envs 4096 --iterations 3000 --run-name <label>
scripts/mjlab.sh train --num-envs 2048 --iterations 300 --run-name <label> \
  --resume artifacts/multimode-v1-replay-20261014/models/model_3023.pt
# 训练监控
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui_huananhu_multimode_v1 --port 8080

# 可视回放（play 配置：无随机化、观测噪声与延迟）
scripts/mjlab.sh play --num-envs 1 --checkpoint <model_*.pt>
# 看中途倒地恢复的动态过程：每 5 秒把站着的机器人掀翻一次
# 手动冲量默认取 KNOCKDOWN_ANGULAR_SPEED_RANGE（18～28 rad/s），--knockdown-speed MIN MAX 可改
scripts/mjlab.sh play --num-envs 1 --checkpoint <model_*.pt> --knockdown-every 5
# 4 个环境一起看恢复；地形对比在平地上已不适用（生成地形配方保留在 multimode_terrains.py，可另行恢复）
scripts/mjlab.sh play --num-envs 4 --checkpoint <model_*.pt> --knockdown-every 6
# 无窗口 / 远程：viser 网页查看器（--viewer auto 只在没有任何 DISPLAY 变量时才选 viser；
# 8080 已被 TensorBoard 占用，viser 默认也是 8080，所以用 --viser-port 显式避开）
scripts/mjlab.sh play --num-envs 4 --checkpoint <model_*.pt> \
  --viewer viser --viser-port 8091 --knockdown-every 6

# 地面起立评估：64 个强制地面初态、20 秒
scripts/mjlab.sh eval --checkpoint <model_*.pt> --num-envs 64 --seeds 42 \
  --start-mode force --terrain flat --output logs/mjlab/evaluation/<name>/result.json
# 注：记录在案的参考基线 model3023-baseline-gpu64 用的是 --terrain all（平子集 31/64）；
# 与它配对必须同协议，scripts/agent/compare_multimode_eval.py 会在协议不一致时拒绝配对。

# 中途击倒后重新起立评估：同一套初态，第 5 秒把每个存活环境掀翻
scripts/mjlab.sh eval --checkpoint <model_*.pt> --num-envs 64 --seeds 42 \
  --start-mode force --terrain all --knockdown-at 5 \
  --output logs/mjlab/evaluation/<name>/result.json

# ONNX 导出（固定 batch=1，opset 13 供 Cube.AI，18 供电脑回放）
scripts/mjlab.sh export --checkpoint <model_*.pt> --output exports/<name>.onnx --opset 13

# 当前最终平衡验收（不训练；输出目录使用新名称）
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_knockdown_recovery.py \
  --checkpoint <model_*.pt> --envs-per-combo 32 --seeds 43 \
  --magnitudes 12,16,20,25,30,40 --knockdown-at 3,9 \
  --recovery-window 12 --recover-deadline 12 \
  --output artifacts/<新评估名>/result.json
```

```bash
# 平地速度/角度跟踪测量（训练前参考或训练后验收证据；口径见接口文档）
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_tracking.py \
  --checkpoint <model_*.pt> --output artifacts/<新名称>/result.json
# 判据离线检查（不建环境）
mjlab_training/.venv/bin/python scripts/agent/check_multimode_tracking.py
```

改动初态或复位分布后，训练前先做 20 环境可视预检：`scripts/mjlab.sh play --num-envs 20`。

## 平地速度／角度跟踪阶段（2026-10-17 起）

地形改为无限平地后，生成地形的地形触发（slope／step／stair）自然休眠，jump 请求对全部环境可触发，
故请求概率减半（`.08`→`.04`）以保持期望的空中环境步数（不是每个可跳环境至少跳一次的概率；`mode_airborne` 才是空中量，
`mode_jump` 累计的是请求次数）。出界半径由 2.5 m 提到 45 m
（`multimode_mdp.OUT_OF_PATCH_RADIUS_M`，高于一个 20 s 回合的满速直线位移 30 m 与约 25% 超速时的 37 m），它只用于结束失控回合；
mjlab 每个环境是独立物理世界（Warp 桥以 `nworld=num_envs` 建立），不同环境的机器人不会相遇，
因此 `env_spacing` 保持任务原值。`eval.py` 与恢复机制检查改用同一常量，不再各自硬写 2.5 m。命令包线收窄到 vx ±1.5 m/s、yaw ±1.5 rad/s，
special-mode 桶与 heading 外环一起限幅到同一包线；观测契约仍为 39／82／6。
测量口径、判据与边界见[平地速度与角度跟踪测量](interfaces/chuanliantui-huananhu-multimode-tracking-v1.md)。

**该阶段续训结果（2026-10-17）**：从 `model_4850` 按本配方续训 2048 环境 × 1000 轮（用户两次主动停止，分三段完成，run 目录 `2026-10-17_03-11-53`／`03-30-24`／`13-22-26_flat-track-vx1p5-yaw1p5`），最终为 `model_5849.pt`（SHA256 `d4be0bc9…`，含优化器与 `env_state`）。同一口径跟踪测量由 10／98 升到 **98／98**（mean |vx 误差| 0.118 m/s、|yaw 误差| 0.058 rad/s，容差 0.30），独立 CPU ONNX 回放在 vx 1.5／vx 1.0+yaw 0.5／零命令下均通过（入口摩擦口径 1.0；2026-10-17 晚对齐训练摩擦 0.5 后 vx=1.5 翻为原生引擎约 16.7 s 失稳、零速仍通过，见[接口记录](interfaces/chuanliantui-huananhu-multimode-replay-v1.md) 2026-10-17 晚节）；但同一套 64 强制地面初态的 `eval` 为 **63/64**（`env 27` 在 2.64 s 物理终止），父模型与基线均为 64/64。**未采纳为新基线**，基线仍是 `model_3023`；双轮平衡验收 v3.1 尚未对本模型完整重跑。

RTX 4060 Laptop（8 GiB）上 4096 环境曾在创建 Warp CUDA 图时显存不足，本机续训用 2048 环境。

独立 CPU MuJoCo sim2sim 与 ONNX 交付见 [ONNX 使用说明](onnx-huananhu-multimode-model5849.md)（当前交付 `model_5849`；历史 `model_3023` 说明保留）。

## 基线复核（2026-10-15）

迁移后用 `eval` 在 64 个强制地面初态、seed 42、20 秒下复跑 `model_3023`：初态签名与原基线一致（`458f15a3…`），严格起立 64/64，平地 31/31 全部存活，数值失败 0。存活 61/64，原记录为 62/64：差在 56 号楼梯样本，它在 19.16 s 越出 2.5 m 地块，不是跌倒，属于 GPU/Warp 逐轨迹不严格确定的范围。

`check` 和 `eval` 只做有限检查和地面起立统计，不是行为验收。行为验收按 [当前双轮平衡验收 v3](interfaces/chuanliantui-wheel-balance-acceptance-v2.md) 执行，文档保留历史 v2 定义。
