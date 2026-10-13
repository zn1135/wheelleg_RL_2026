# V11 原生 MuJoCo 可视 sim2sim

2026-10-13，Codex 执行，接收人为用户，独立复查未执行。用户指定 V11、MuJoCo 窗口与遥操作。本次增加独立 CPU 回放入口，不训练、不导出、不改机器人资产或 H7。

## 结果与限制

**用户随后反馈持续抖腿，已复现。** 固定普通初态末5秒四腿实际角度峰峰值约7～10°，当前源mjlab环境也有相近振荡。这里的高度/倾角短测不足以说明腿部动作平稳，详细三方对照见[摆腿诊断](20261013-v11-leg-jitter.md)。

接口对照通过，原生 MuJoCo 窗口已启动并持续回放。以下三项各运行 20 秒、1000 个策略步，均退出 0；统计末 5 秒。每项仅一个固定初态，无域随机化，不是 64 初态配对验收，也未测量验收 v2 的轮端承重条件。

| 场景 | 目标高度 m | 目标前速 m/s | 平均高度 m | 平均前速 m/s | 最大倾角 ° |
|---|---:|---:|---:|---:|---:|
| 普通站姿零速 | 0.30 | 0 | 0.309461 | -0.011081 | 2.2113 |
| 普通站姿行走 | 0.30 | 0.5 | 0.308223 | 0.388759 | 0.5436 |
| 固定地面后摆姿态 | 0.30 | 0 | 0.308987 | -0.017434 | 3.3056 |

行走存在约 0.111 m/s 欠速。以上说明这三个短时场景能够运行，不代表全初态、全命令或部署合格。地面复位采用 `poses.json` 的已解闭链固定后摆姿态（root 0.15 m），与此前配对评估的随机地面姿态不同。

`check_chuanliantui_v11_native.py` 创建一个实际 mjlab 环境，对照 32 组状态，覆盖正负单位动作、非对称通道、观测裁剪、力矩限幅、弹簧及地面电机暂停边界。最大绝对差：观测 `1.90735e-6`，电机力矩 `2.03009e-5 N·m`，弹簧力 `1.05005e-7 N`。它是接口检查，不能代替行为验收。

第一次站立运行因 actor 输入用了裸 Tensor 而非 actor 观测字典退出 1，保留空轨迹 `standing.json`；修正后 `standing-v2.json` 完成 1000 步。后续修改仅涉及 viewer 复位锁和报告完成标记，不影响已记录的无窗口动力学结果。

## 接口与模型交付

| 项目 | 本次记录 |
|---|---|
| 训练仓与代码版本 | `zn1135/wheelleg_RL_2026`，当前工作区 detached HEAD `bb6b337725218b7c816dfb0d775cee46996acfb0`；开始时干净，运行时有本次新增文件，未提交 |
| 模型 | wheelleg V11 完整 `model_999.pt`，iter 999、记录累计 4000 更新，只读载入确定性 actor |
| checkpoint SHA256 | `62efa60cf3c0884d0abf4a2378d778371135a49c475ed3453ebec08136d22366` |
| 模型来源与历史表现 | [V11 回放报告](20261013-mjlab-v11-replay.md)，训练时准确 Git 状态本次未重新核实 |
| XML SHA256 | `42a1b937b83ca034d4913306aeca5b219c3cd1c5a0bbd57b0220d7f5d5c826e0` |
| 权威入口 | `sim2sim/mj_sim2sim_v11.py` 的 `Playback`；对照当前 `reference_training/huananhu_observations.py` / `huananhu_action.py` |
| 观测/动作 | 35/6；固定 NORMAL `[1,0,0,0,0,0,0]`；无 encoder、无观测归一化 |
| 动作顺序 | `lf0, lf00, lfwheel, rf0, rf00, rfwheel` |
| 观测关节/历史动作顺序 | `lf0, lf00, rf0, rf00, lfwheel, rfwheel`，历史为上一步原始动作 |
| 观测索引与缩放 | 沿用上述 V11 报告逐通道表；高度实际计算为 `clip(h,0,1)*5`，角速度先裁剪±100再乘0.5，关节速度先裁剪±200再乘0.1 |
| 坐标/命令 | +x 前向；yaw 是偏航角速度；free-joint qvel 角速度、逆旋转重力 |
| 时序/控制 | 原生 CPU MuJoCo 2 ms，10 子步更新一次策略；腿 PD 60/2、scale 0.5，轮增益 0.1、scale 10、目标±150 rad/s；腿/轮力矩上限40/3.9 N·m |
| 物理与复位 | 本仓实体闭链 XML 与 nominal 角；弹簧0–150 N；无随机化和延迟；地面初态前0.2 s电机暂停；Home/End清空速度与上次动作 |
| 命令范围 | 遥操作前速/偏航各±1.5，高度0.20–0.34 m；只是软件输入边界，不是整段范围已验收 |
| H7 / ONNX / 真机 | 均不适用本次；旧 H7 接口未适配、未验证；没有新模型或接口迁移、没有优化器恢复 |
| 源码快照/实际命令/版本 | 本机 `artifacts/v11-native-sim2sim-20261013/` 的 `tests.md`、`provenance.json`、`source/`；不以 HEAD 代替新增源码 |

## 重开与操作

使用已有 Python 3.12 mjlab 环境，`MJLAB_PYTHON` 为其解释器绝对路径，`V11_CHECKPOINT` 为上述哈希对应模型。参数显式指定；拒绝加载其他权重。

```bash
"$MJLAB_PYTHON" sim2sim/mj_sim2sim_v11.py \
  --checkpoint "$V11_CHECKPOINT" --render --seconds 0
```

点击 MuJoCo 窗口后：↑/↓ 每次调前速 0.1 m/s；←/→ 每次调偏航角速度 0.1 rad/s；PgUp/PgDn 每次调高 1 cm；Enter 将前速/偏航归零；Home 回普通站姿；End 回地面姿态；F1 暂停/继续；关闭窗口退出。屏幕叠加当前命令与实际高度。方向键修改的是持续命令，松键不会自动归零。

无窗口短测使用 `--seconds 20 --report <新的报告.json>`，行走追加 `--cmd-vx 0.5`，地面追加 `--pose ground`。报告拒绝覆盖已有文件。程序遇到非有限状态或 MuJoCo warning 抛错退出，失败报告的 `completed` 为 false。

接口检查：

```bash
"$MJLAB_PYTHON" scripts/agent/check_chuanliantui_v11_native.py --output <新的接口报告.json>
```

未启动训练或 TensorBoard。没有提交、推送或创建分支。
