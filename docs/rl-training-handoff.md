# RL 训练侧协作交接

整理日期：2026-10-10。面向负责 H7 部署的协作者。

## 1. 分工与当前版本

用户负责 **机器人建模 → RL 训练 → 仿真验证 → 模型导出与交付**；H7 协作者负责 **板端观测 → 网络推理 → 电机控制 → 固件构建与真机验证**。双方共同维护接口和模型交付记录。

当前维护分支为 `26_wheelleg`。mjlab 训练侧只保留华南虎 multimode 一条线，r2～r14 等旧版本说明已删除，可从分支 `backup/gpt-structure-20261015` 找回。

两人各自克隆完整仓库，按负责目录分别提交并合入共同维护分支。观测、关节顺序、缩放、控制频率或模型格式变化时，双方共同核对接口记录。具体流程见 [协作指南](../CONTRIBUTING.md#两人维护分工)。

## 2. 训练侧代码地图

2026-10-15 起 mjlab 训练侧只保留华南虎 multimode 一条线，旧 r5～r15、V14、recover、V11 代码已删除（可从分支 `backup/gpt-structure-20261015` 找回）。以下路径相对仓库根目录。

| 入口／文件 | 用户维护的内容 | 部署协作者需要了解的内容 |
|---|---|---|
| `scripts/mjlab.sh`、`mjlab_training/src/ct_mjlab/scripts/` | train / play / eval / export / check | 如何回放交付模型、导出 ONNX |
| `mjlab_training/src/ct_mjlab/envs/chuanliantui/` | multimode 配置、PPO、奖励、观测、控制、初态与随机化 | 观测打包、模式与标志位 |
| `resources/robots/chuanliantui_new_1/mjlab/` | 训练 XML 与地面姿态库 | 坐标、左右命名、闭链与气弹簧 |
| `sim2sim/mj_sim2sim_huananhu_multimode.py` | 独立 CPU MuJoCo sim2sim | 图外观测与 PD 的参考实现 |
| `wheel_legged_gym/`、旧 `sim2sim/` 脚本 | Isaac Gym 历史环境 | 仅用于对应旧模型，不能套用新接口 |

## 3. 当前训练方案

见 [mjlab 串联腿训练](mjlab-chuanliantui.md)：39/82/6，2 ms 物理、50 Hz 策略，flat／slope／step／stair 混合地形，50% 普通／50% 地面初态。基线模型 `model_3023.pt`。

## 4. H7 必须对齐的策略接口

接口 `chuanliantui-huananhu-multimode-39-82-v1`。ONNX float32 `[1,39]→[1,6]`，opset 13，图内只有 actor，观测打包、PD 和调度都在图外。动作顺序 `[lf0,lf00,lfwheel,rf0,rf00,rfwheel]`。槽位与交付契约见 [multimode 接口](interfaces/chuanliantui-huananhu-multimode-replay-v1.md) 与 [ONNX 使用说明](onnx-huananhu-multimode-model5849.md)（当前交付 `model_5849`）。H7 负责人的核对状态为待确认。

## 5. 双方交付清单

| RL 用户交给 H7 协作者 | H7 协作者反馈给 RL 用户 |
|---|---|
| 训练代码完整提交 SHA、实际运行配置、机器人资产版本 | 固件提交 SHA、硬件版本、实际构建配置 |
| checkpoint、ONNX、导出 JSON 与 SHA-256 | 网络转换结果、板端推理结果与参考输出误差 |
| 接口字段、名义姿态、关节顺序、控制参数 | 实物电机对应关系、极性／零点核对、IMU 坐标变换 |
| mjlab 回放、CPU MuJoCo 起立／静站结果和轨迹 | 实际采样／推理／执行频率、链路延迟和真机验证结果 |

每次交付填写 [模型交付模板](templates/model-handoff.md)，实际记录保存在 `docs/model-deliveries/`。

## 6. 当前完成程度与后续顺序

**已完成：** `model_3023` 的 ONNX 导出、独立 CPU sim2sim 限定静站、64 个地面初态起立复核（严格起立 64/64），以及 mjlab 代码改为单线结构。

**待完成：** H7 侧 39 维观测打包、50 Hz 调度、板端数值对齐和真机验证。导入 H7 的网络仍为旧 25/125 维、100 Hz 的 `model_15000_h723`，不兼容新接口。

后续按 **训练 → 回放与评估 → 模型交付 → H7 适配与验证** 推进。
