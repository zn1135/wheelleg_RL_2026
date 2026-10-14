# 华南虎 multimode 历史模型独立回放接口

本记录只定义历史模型 `model_3023.pt` 在当前工作区的隔离回放方式；它不替换默认 V11 训练接口，也不代表 H7 已适配。

| 项目 | 内容 |
|---|---|
| 记录日期 | 2026-10-14 |
| 训练来源 | `/home/zn/文档/mjlab/wheelleg_RL` 的冻结验证快照 |
| 任务 | `Chuanliantui-Huananhu-Multimode-v0` |
| 接口标识 | `chuanliantui-huananhu-multimode-39-82-v1` |
| checkpoint | `artifacts/multimode-v1-replay-20261014/models/model_3023.pt` |
| checkpoint SHA256 | `624ded630635f0fb51a7378cdd04a7350eee98ffb305acbe1104c3cc7705a88a` |
| 专用回放资产 | `resources/robots/chuanliantui_new_1/multimode_model3023/`（XML 与地面姿态库） |

## 与 V11 的关系

| 项目 | V11 平地策略 | multimode 历史策略 |
|---|---|---|
| Actor／Critic | 35／78 | 39／82 |
| 任务 | 固定 NORMAL 的平地站立／速度 | 平地、坡、台阶、楼梯与地面恢复 |
| 额外 Actor 槽 | 无 | `normal, stair, slope, recover, jump, jump_target_height×5, jump_state_time` 与四个状态标志 |
| 动作 | 6 个实体电机 | 同为 6 个实体电机 |
| 时序 | 2 ms 物理、20 ms 控制 | 2 ms 物理、20 ms 控制 |

39／82 与 35／78 形状不同，故 `model_3023.pt` 不能加载进 V11 actor，也不能从 V11 继续训练；本次没有修改默认任务、公共 XML、H7 控制或训练资产。

## 回放约束

独立 CPU 回放从版本化专用 XML 与姿态库构建，不导入当前默认 `ct_mjlab` 内核。原任务的模式标志由仿真状态直接给出；实机需另行重建这些标志。本模型的现有证据限于仿真地面恢复，不能据此推断行走、板端或真机可用性。

## 本机 play 与 ONNX sim2sim（2026-10-14）

`play/play_huananhu.py` 是原仓 `play_huananhu.py` 的逐字节副本；它严格加载 `.pt` 的 PyTorch actor，在一个 `play=True` 环境中直接执行 `env.step(policy(obs))`。本机 seed 42、1 环境、20 秒产生 500 帧离屏视频；该样本是任务随机的 NORMAL 初态，末帧仍在同一回合（19.98 秒），因此它只作可视回放，不代替地面恢复统计。

`sim2sim/sim2sim_huananhu_onnx.py` 是原仓 ONNX sim2sim 入口的逐字节副本。策略闭环只调用 `onnx.reference.ReferenceEvaluator`，不调用 PyTorch actor；`--compare-torch` 只在同一观测上计算动作差。它在冻结的 mjlab MuJoCo/Warp 环境（GPU `cuda:0`）中运行，并使用 MuJoCo 离屏渲染，所以这是 ONNX 控制链验证，**不是** V11 所用的独立 CPU MuJoCo 物理后端验证。

同一 checkpoint、seed 42、64 个强制地面初态、20 秒的 ONNX 结果为严格起立 64/64、数值失败 0、20 秒存活 62/64；平地子集 31/31 严格起立且全部存活。另两个楼梯样本越出 2.5 m 地形区域，未发生物理跌倒。ONNX 对同观测 PyTorch actor 的最大动作差是 0.00865126。可复核的逐环境报告在 `sim2sim-gpu64/result.json`，导入来源及哈希在 `SIM2SIM_PLAY_IMPORT_MANIFEST.json`。

关联模型记录：[model_3023 回放记录](../model-deliveries/20261014-huananhu-multimode-model3023-replay.md)。

## 固定 NORMAL 静站 ONNX 接口（2026-10-15）

`model_3023.pt` 的本次交付只覆盖平地、NORMAL、零速、目标高度 0.30 m 的静站。专用导出入口 `scripts/export_huananhu_multimode_onnx.py` 直接读取 checkpoint；CPU 验证入口从版本化专用 XML 与姿态库构建环境并在运行时强制平地与下列 actor 尾部槽；不修改 checkpoint、训练配方、默认 V11 或 H7。操作命令见 [ONNX 使用说明](../onnx-huananhu-multimode-model3023.md)。

| 输入／输出 | 契约 |
|---|---|
| 输入 | `obs: float32[1,39]` |
| 输出 | `actions: float32[1,6]` |
| 输出顺序 | `[lf0, lf00, lfwheel, rf0, rf00, rfwheel]` |
| 模式槽 `28:35` | `[1,0,0,0,0,0,0]`（NORMAL） |
| 标志槽 `35:39` | `[0,1,0,0]`（非地面起始、策略激活、非腾空、非台阶） |
| 时序 | 2 ms 物理、20 ms／50 Hz 策略 |

opset 18 是电脑回放参考版，opset 13 是 Cube.AI 优先导入版。历史生成物在本机 `artifacts/` 中作为被忽略证据保存，不随 Git 发布；使用版本化导出器重新生成时，以 `*_export.json` 记录的 SHA-256 和数值检查为准。图只含确定性 actor；观测缩放和打包、上一拍动作、PD、限幅、实体关节映射及 50 Hz 调度均在图外，由实物侧实现。旧 H7 是不同的 25/125 维、100 Hz 接口，不能直接替换。

## 独立 CPU MuJoCo sim2sim（2026-10-15）

`sim2sim/mj_sim2sim_huananhu_multimode.py` 是该 39 维模型的独立 CPU MuJoCo 入口。它编译版本化的 `resources/robots/chuanliantui_new_1/multimode_model3023/` XML 与姿态库，并只在内存增加摩擦 `(1, .005, .0001)` 的平地；用闭链求解器写入 normal 初态，复刻原始观测缩放、四腿前两轮后的观测顺序、物理电机 PD（腿 `60/2`，轮 `0.12/0.02`）、气弹簧和 2 ms／20 ms时序。传入固定 batch=1 ONNX 后，它在同一 batch=1 观测上与 PyTorch actor 对比；可用 `--pytorch-controller` 只作 PyTorch 对照。

```bash
/home/zn/文档/mjlab/wheelleg_RL/.venv/bin/python \
  sim2sim/mj_sim2sim_huananhu_multimode.py --onnx <model_3023_opset13.onnx> --render --seconds 0
```

20 秒 headless ONNX 结果在 `artifacts/multimode-v1-replay-20261014/normal-static-20261015/cpu-sim2sim-onnx13/result.json`：1 秒后951个策略步最大高度误差 2.60 cm、最大倾角 1.81°、机身接触0、双轮接触951、ONNX/PyTorch 最大动作差 `8.34e-7`，通过本次 normal 静站条件。末态高度约0.324m，高于冻结 GPU MuJoCo/Warp 回放约0.308m，仍在3cm门限内；该差异是独立物理后端差异，不能据此推断真机高度误差。
