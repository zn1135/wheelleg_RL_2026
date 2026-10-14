# 华南虎 multimode model_3023 独立回放记录

## 原始混合地形 2048 环境续训 300 轮（通过固定初态回归）

为排除全平地采样分布，续训入口恢复冻结源码的terrain generator：flat 50%、slope 20%、step 15%、stair 15%。在当前工作区从`model_3023.pt`完整恢复actor、critic、Adam和课程进度，以2048环境、seed42、TF32追加300次更新；native3023→3322、有效累计7525→7825。候选`resume300-2048-mixed-run1-20261014/model_3322.pt` SHA256为`5adff9983190b7fde7139b7b4cc10a15fa176e5abdac0c39877efb3fe8d228d7`。

使用与基线相同的冻结评估器、seed42、64个固定强制地面初态和20秒窗口，严格恢复64/64，平地子集31/31全部存活，数值故障0；没有原成功初态失效。原仓以等同的冻结源码／资产、模型、PPO、2048环境和种子独立追加300次所得候选也通过同一集合。两候选权重哈希不同，符合Warp／TF32下逐轨迹不严格确定的边界；该差异未改变验收结果。相比全平地候选的平地31/31→26/31，本对照支持保留原混合地形，而不支持将12个patch均改为flat。

**用途限定为当前工作区的仿真恢复回放；不作为 V11 替换、训练续接、ONNX/H7 交付或真机模型。**

## 来源与模型

原仓为 `/home/zn/文档/mjlab/wheelleg_RL`，历史 checkpoint 为 `Chuanliantui-Huananhu-Multimode-v0` 的 `model_3023.pt`，记录累计训练 7525 次更新。当前工作区复制了当时验证使用的源码快照、资产、checkpoint 和评估器，副本位于 `artifacts/multimode-v1-replay-20261014/`；导入清单与逐文件 SHA256 见 `IMPORT_MANIFEST.json`。

| 产物 | SHA256 | 说明 |
|---|---|---|
| `models/model_3023.pt` | `624ded630635f0fb51a7378cdd04a7350eee98ffb305acbe1104c3cc7705a88a` | 完整 39/82 actor/critic checkpoint |
| `replay-gpu64/result.json` | `deedd65eac9bfbe22cabdffa670bb2f70cbd1c59f34470ed39e768c5a7a82bbc` | 当前机器 64 环境回放报告 |
| `replay-gpu64/ground-recovery-env0.mp4` | `4b220290d0423b42b3ce16ee16c0d10582b6a82d382007c4f6f67f7fb5af832d` | 当前回放 env0 的真实 qpos 离屏渲染，500 帧/25 fps/20 秒 |

接口为 `chuanliantui-huananhu-multimode-39-82-v1`，动作仍是 6 个实体电机、2 ms 物理和 20 ms 控制；和 V11 的 35/78 接口不兼容，详见[接口记录](../interfaces/chuanliantui-huananhu-multimode-replay-v1.md)。

## 当前回放

使用冻结评估器在 RTX 4060 Laptop GPU 上以 seed 42、64 环境、20 秒、`play` 配置运行：关闭域随机化、观测/动作延迟和推扰，所有环境地面初态。严格起立判据为高度误差不超过 3 cm、姿态各轴不超过 15°、无基座接触、双轮接触，连续保持至少 0.5 秒。

| 集合 | 严格起立 | 存活 20 秒 | 数值故障 | 说明 |
|---|---:|---:|---:|---|
| 全部 64 个地面初态 | 64/64 | 62/64 | 0 | 2 个楼梯地形样本越过 2.5 m 地形边界；非物理跌倒 |
| 平地子集 31 个 | 31/31 | 31/31 | 0 | 严格起立中位 0.92 秒，末态平均高度 0.306 m |

GPU 运行退出 0，且严格加载显示 actor/critic 为 39/82、动作维度为 6。视频的 env0 是平地地面初态，完成严格起立并运行至超时。原地形任务的 `--terrain flat` 只决定报告的平地子集，并不重建纯平地场景；因此不得把“64 环境”表述为 64 个平地样本。

未执行原生 CPU sim2sim 闭环、步态/速度验收、H7 板端或真机验证。回退方式是删除本次独立引用并继续使用原 V11；本次未改写任何默认模型或历史日志。

## 原仓 play 与 ONNX sim2sim 补充

原仓 `play_huananhu.py` 与 `sim2sim_huananhu_onnx.py` 已复制到本引用的 `play/`、`sim2sim/`，二者以及 ONNX `model_3023.onnx` 的来源与 SHA256 见 `SIM2SIM_PLAY_IMPORT_MANIFEST.json`。前者严格加载 checkpoint 的 PyTorch actor，在单一随机 NORMAL 初态执行 20 秒并写出 500 帧视频 `play-gpu1/play-env0.mp4`；该回放在 19.98 秒仍处于同一回合。

后者以 ONNX ReferenceEvaluator 作为唯一闭环控制器，并在同一观测旁路对比 PyTorch actor。seed 42、64 个强制地面初态、20 秒：严格起立 64/64，数值故障 0；其中平地子集 31/31 跑满 20 秒，整批的 2 个楼梯样本因越过 2.5 m 地形边界结束。ONNX 与 PyTorch 的最大动作差为 0.00865126。逐环境结果与视频为 `sim2sim-gpu64/result.json`、`sim2sim-gpu64/result_episodes.json` 和 `sim2sim-gpu64/sim2sim-env0.mp4`。

该 ONNX sim2sim 仍使用冻结 mjlab MuJoCo/Warp 的 GPU 物理环境；它不等同于 V11 的独立 CPU MuJoCo 后端，也不能补足速度、H7 或真机验收。

## 4096 环境续训尝试

用户指定从 `model_3023.pt` 恢复并以 4096 环境追加 300 次 PPO 更新。训练在首次环境 reset 创建 Warp CUDA 图时因 RTX 4060 Laptop GPU（8 GiB）报 `Warp CUDA error 2: out of memory` 而退出；尚未加载 checkpoint 进入 runner 或调用学习循环，PPO 更新数为 0，未产生候选 checkpoint。因此不存在“300 轮后是否退化”的结果。输入模型、冻结源码和失败配置保留在 `artifacts/multimode-v1-replay-20261014/resume300-4096-20261014/`，其中 `failure.md` 与 `provenance.json` 可复核；本次没有自动改用 2048 环境。

随后按用户要求把 12 个训练地形块全部改为原来的 flat 配置，同时保留原 terrain generator、传感器与 39/82 接口，再次以 4096 环境作无学习初始化。它仍在同一 Warp CUDA 图创建点 OOM，尚未加载模型或学习。这表明显存主要来自 4096 份机器人闭链物理、接触状态与 Warp 图，而非坡／台阶／楼梯地形；记录在 `resume300-4096-flat-check-20261014/failure.md`。

进一步分档验证全平地环境：2048、3072、3584 均能初始化并运行 100 步；3840、3968、4096 均 OOM。3584 档的 Python CUDA 进程峰值为 6580 MiB，而桌面常驻进程占约 855 MiB，只余约 753 MiB；线性外推4096环境／Warp进程约7520 MiB，加桌面约8375 MiB，已超过8188 MiB物理显存，图捕获的临时内存会进一步提高门槛。无遗留训练进程。详见 `artifacts/multimode-v1-replay-20261014/memory-analysis-20261014.md`。

## 全平地 2048 环境续训 300 轮（拒绝）

用户随后指定全平地、2048 环境从 `model_3023.pt` 完整恢复后追加 300 个 PPO 更新。训练使用冻结源码，保留 39/82 接口、奖励、随机化、控制时序和 terrain generator，仅将全部 12 个 patch 固定为 flat；actor、critic、Adam 和课程进度均逐项恢复。训练于 22:12～22:23 完成，原生迭代 3023→3322、有效累计进度 7525→7825，最终模型为 `resume300-2048-flat-run3-20261014/model_3322.pt`，SHA256 `37e22c3cfa55ecca8bd0f4fafeceb148f19891d913f92fce32065f179f8ce4dc`。

在冻结回放源中，以相同 seed 42、64 个强制地面初态、20 秒及同一严格判据做配对：原模型严格起立 64/64、存活 62/64；候选变为 56/64、50/64。真实 flat 子集原 31/31 严格起立且存活，候选降为 26/31，5 个原成功平地初态变为物理失败。按原成功样本任一失效的规则，候选退化、拒绝，不替换 `model_3023.pt`。完整迁移、训练和逐样本回放证据位于 `resume300-2048-flat-run3-20261014/`。

## 2026-10-15 固定 NORMAL 静站 ONNX 交付（通过限定仿真验收）

本节只验证原始 `model_3023.pt` 的普通平地静站：`vx=0`、`yaw=0`、高度 `0.30 m`、NORMAL 初态。未追加训练，未更改 H7 固件，也未下发真机电机命令。H7 部署负责人对 39 维接口、50 Hz 调度和板端数值对齐的核对状态为待确认。首次交付使用冻结源码／资产快照完成 GPU/Warp 验收。当前正式导出器直接读取 checkpoint；独立 CPU 入口读取版本化专用 XML（SHA256 `1ea213314d8fd686841678d28246fbb588d53a5586e08c22ea977e95c0cb4ec5`）与姿态库（SHA256 `ca82470ace721aac630f8376c06b56a6c182c2cda4931051fdb7e003d2a93db1`），不再依赖被忽略快照。

| 产物 | SHA256 | 说明 |
|---|---|---|
| `models/model_3023.pt` | `624ded630635f0fb51a7378cdd04a7350eee98ffb305acbe1104c3cc7705a88a` | 源 checkpoint；未改写 |
| `normal-static-20261015/onnx-inline/model_3023_opset18.onnx` | `e6103c6821ce0c3f434b363981c3caf61a2dea8ce88b96873e048937cf78d030` | 208,271 B，实际 opset 18，内嵌权重，电脑回放参考 |
| `normal-static-20261015/onnx-inline/model_3023_opset13.onnx` | `5a0650da636a96095b7cf80d90472275e1db9b7b18cd196f3592b664a3c535af` | 208,271 B，实际 opset 13，内嵌权重，Cube.AI 优先导入 |

两图均为 7 个节点（`Gemm`、`Elu`），固定接口 `obs: float32[1,39] -> actions: float32[1,6]`，动作顺序为 `[lf0, lf00, lfwheel, rf0, rf00, rfwheel]`。导出使用 PyTorch 2.14 的兼容 ONNX 路径；每个图经 `onnx.checker` 检查，并以 256 个固定随机输入和 `onnx.reference.ReferenceEvaluator` 与 PyTorch actor 比较，最大绝对误差均为 `1.9073486328125e-06`，小于 `1e-4`。参考输入输出保存在同目录的 `*_reference_vectors.npz`，完整图记录在 `*_export.json`。

专用验证强制平地、20 秒、2 ms 物理／20 ms 控制、`mode[28:35]=[1,0,0,0,0,0,0]`、`flags[35:39]=[0,1,0,0]`。高度误差阈值为 3 cm，滚转／俯仰各 15°，连续稳定窗 0.5 秒；稳定窗要求无机身接触及双轮承重。ONNX 闭环仅调用 `ReferenceEvaluator` 输出动作，PyTorch 只在同一观测上作动作比较，并按同样的固定 batch=1 调用，以匹配交付接口。

| 阶段 | 结果 | 证据 |
|---|---|---|
| PyTorch play，seed42，1 环境，20 秒 | 通过 | `normal-static-20261015/play-pytorch/result.json`；20 秒存活、严格稳定窗无失效、最大 ONNX 差 `9.54e-07`；视频 `play-pytorch/torch-play-env0.mp4`。10 秒画面人工检查为双轮支撑、机身无接触，画面读数高度 0.308 m。 |
| ONNX opset 18，seed42，64 固定 normal 平地初态，20 秒 | 通过 | `sim2sim-opset18-batch1/result.json`；64/64 存活、64/64 进入稳定窗、物理终止 0、数值故障 0、稳定窗失效 0、最大同观测动作差 `1.91e-06`。 |
| ONNX opset 13，seed42，64 固定 normal 平地初态，20 秒 | 通过 | `sim2sim-opset13-batch1/result.json`；64/64 存活、64/64 进入稳定窗、物理终止 0、数值故障 0、稳定窗失效 0、最大同观测动作差 `1.91e-06`。 |

稳定窗口内的腿部指标（动作顺序中的四个腿关节为 `lf0, lf00, rf0, rf00`）如下；这些是本模型现状描述，不构成“抖腿已修复”的结论。

| ONNX | 腿速度 RMS (rad/s) | 摆幅 (rad) | env0 主频 (Hz) | 动作 RMS |
|---|---|---|---|---|
| opset 18 | `[0.213, 0.209, 0.206, 0.256]` | `[0.297, 0.321, 0.274, 0.368]` | `1.827` | `[0.095, 0.194, 0.594, 0.171, 0.331, 0.481]` |
| opset 13 | `[0.214, 0.210, 0.206, 0.256]` | `[0.294, 0.318, 0.279, 0.373]` | `1.879` | `[0.095, 0.194, 0.594, 0.171, 0.331, 0.482]` |

此前把 64 个 PyTorch 环境作为单一 GPU batch 与固定 batch=1 ONNX 比较，出现最大差 `0.00589`；单项 batch=1 PyTorch 与 ONNX 的差为 `1.91e-06`。随机输入探针也验证了这一点：GPU batch 与单项 batch 的差可达约 `0.00994`，而单项 GPU 与 CPU 为 `5.72e-06`。因此本交付的逐步比较统一为 batch=1，代表明确的单机器人接口；批量仿真数值不能替代这一契约。

本节不包括起立、倒地恢复、坡地、台阶、楼梯、行走、转向、独立 CPU 物理后端、Cube.AI 实际导入耗时、H7 传感器打包或真机验证。实物侧仍须按本节 39 维格式实现观测缩放、上一拍动作、PD、限幅、关节映射与 50 Hz 调度，并在板端以参考向量先做数值比对。

## 2026-10-15 独立 CPU MuJoCo sim2sim 补充

已新增 `sim2sim/mj_sim2sim_huananhu_multimode.py`，修复旧 CPU 入口只能加载 35 维 V11 actor 的限制。新入口只支持本次交付的 39 维 normal 静站：版本化 XML 在内存追加平地，固定 `mode[28:35]=[1,0,0,0,0,0,0]` 和 `flags[35:39]=[0,1,0,0]`，并按历史 direct motor PD、气弹簧及 2ms／20ms时序控制。调用时显式传入 opset 13 ONNX；PyTorch actor仅在同一 batch=1 观测上比较。

20秒 CPU sim2sim报告为 `normal-static-20261015/cpu-sim2sim-onnx13/result.json`：1秒后951步，高度最大误差`0.02599m`、最大倾角`1.809°`、机身接触0、双轮接触951、最大ONNX/PyTorch动作差`8.344650268554688e-07`，判定通过。末态高度约0.324m，独立GPU/Warp回放约0.308m；虽然两者均在3cm条件内，这一物理后端差异仍保留为限制，不能当作真机结论。
