# model_5849 ONNX 导出与 H7 交付

## 任务与来源

2026-10-17 按用户指定导出当前最新 checkpoint `model_5849.pt`（平地跟踪续训终态）为固定 batch=1、opset 13 ONNX，整理为交 H7 协作者适配的交付包；保留原 39 维输入和 6 维动作，不修改模型或训练代码。接收人为用户，独立复查及 H7 部署负责人核对状态为待确认。

- 接口：`chuanliantui-huananhu-multimode-39-82-v1`，见[接口记录](../../../docs/interfaces/chuanliantui-huananhu-multimode-replay-v1.md)。
- 训练仓：`https://github.com/zn1135/wheelleg_RL_2026.git`，分支 `26_wheelleg`。导出时代码提交 `6e24617e3c7c8dafbfed36472c0ccc68502ae406`（含 39 维 CPU sim2sim 修正）；训练运行提交为 `5f96589`（平地化训练与包线收窄）。
- 训练来源：三段续训 `model_4850 → model_5849`（`2026-10-17_03-11-53`／`03-30-24`／`13-22-26` 三个 `flat-track-vx1p5-yaw1p5` run 目录，详见[项目记录](../../../docs/ai/project-log.md)）。运行期仓库状态为 `ed4458f` 加未提交改动，差异快照保存在运行目录 `logs/mjlab/chuanliantui_huananhu_multimode_v1/2026-10-17_13-22-26_flat-track-vx1p5-yaw1p5/git/`；checkpoint 已随 `5f96589` 纳入版本管理。
- 导出时工作区干净；交付准备增加本记录、ONNX 说明、接口交付补充及 `exports/` 文件包，无运行源码差异或新增运行源码，无需运行源码差异快照。
- 内置部署目录未修改；本次未运行 H7，实际板端固件版本待核实，无提交或 PR。
- 导出环境：仓库 `mjlab_training/.venv`，Python 3.12.3、PyTorch 2.14.0+cu130、ONNX 1.22.0、NumPy 2.5.3；CPU 导出。

## 模型与产物

原始产物位于本机 [导出目录](../../../artifacts/model5849-onnx-20261017)，被 Git 忽略。模型、参考向量与 CSV 已逐字节复制到 [H7 适配交付包](../../chuanliantui_multimode/model5849-h7-20261017/README.md)，随本交付记录一同纳入版本管理，供协作者拉取。交付包含无表头的 256×39 输入 CSV、256×6 期望输出 CSV 和 `SHA256SUMS`（覆盖交付目录其他六个文件）。

| 产物 | SHA-256 |
|---|---|
| [源 model_5849.pt](../../../artifacts/flat-track-20261017-trained/models/model_5849.pt) | `d4be0bc98c20348911b7e43a6f52f5e3719cb78d74f5764f485b4bce82bb71ca7` |
| [model_5849_opset13.onnx](model_5849_opset13.onnx) | `9143bc7ad8aabac05831dee9203f42f81c77d94e7616eccea9cc004c1f5616b4` |

ONNX 为 208039 字节，内嵌权重、固定 batch=1、opset 13；输入 `obs: float32[1,39]`，输出 `actions: float32[1,6]`，动作顺序 `[lf0, lf00, lfwheel, rf0, rf00, rfwheel]`。只导出 actor，不含观测构造、PD、限幅或电机映射（完整槽位说明见 [ONNX 使用说明](../../../docs/onnx-huananhu-multimode-model5849.md)）。相对 `model_3023` 交付，39 维接口与图结构不变，权重与参考向量更新；本 checkpoint 训练域为无限平地、命令包线 vx ±1.5 m/s、yaw ±1.5 rad/s。

## 导出与验证

2026-10-17 18:07（北京时间），在仓库根目录执行，退出码 0：

```bash
scripts/mjlab.sh export \
  --checkpoint artifacts/flat-track-20261017-trained/models/model_5849.pt \
  --output artifacts/model5849-onnx-20261017/model_5849_opset13.onnx --opset 13
```

ONNX 图检查、固定输入输出形状检查通过。256 组固定随机输入逐个比较 ONNX 参考执行器与 PyTorch，最大绝对误差 `3.814697265625e-06`，小于容差 `1e-4`。源 checkpoint 的观测契约与控制契约元数据由导出器强制校验。

交付的 opset 13 文件在独立 CPU MuJoCo 入口（`sim2sim/mj_sim2sim_huananhu_multimode.py`，含 2026-10-17 修正）上复测 20 秒：

| 阶段 | 状态 | 结果与证据 |
|---|---|---|
| ONNX 图检查与数值比较 | 通过 | 256 组，最大绝对误差 `3.814697265625e-06`；[导出日志](../../../artifacts/model5849-onnx-20261017/export.log)、[可移植交付副本](../../chuanliantui_multimode/model5849-h7-20261017/model_5849_opset13_export.json)、[板端核对参考向量](../../chuanliantui_multimode/model5849-h7-20261017/model_5849_opset13_reference_vectors.npz) |
| CPU MuJoCo 零速静站 20 s（入口摩擦口径 1.0） | 通过 | 1 秒后高度最大误差 0.0042 m、最大倾角 1.45°、双轮接触 951/951、ONNX／PyTorch 最大动作差 `1.0e-06`；[报告](../../../artifacts/model5849-onnx-20261017/zero-command-report-opset13.json) |
| CPU MuJoCo vx=1.5 m/s 20 s（入口摩擦口径 1.0） | 通过（2026-10-17 晚更正为引擎相关，见下） | mean |vx 误差| 0.178 m/s、最大倾角 1.50°、行进约 25.2 m；[报告](../../../artifacts/model5849-onnx-20261017/vx1p5-report-opset13.json) |
| GPU 训练后端／eval | 已有历史证据 | 跟踪 98/98、64 强制地面初态 63/64，见[项目记录](../../../docs/ai/project-log.md)；本次未重跑 |
| Cube.AI 导入、板端数值／时序、真机 | 未执行 | H7 部署负责人核对状态为待确认 |

**2026-10-17 晚更正**：独立审计发现入口未镜像训练场景的机器人 geom `priority=1` 补丁（机器人-地面滑动摩擦实际 1.0，为训练值 0.5 的两倍，见接口记录 2026-10-17 晚节）。入口对齐后用同一份交付文件复测：零速静站 20 s 仍通过（1 秒后高度最大误差 0.0110 m、最大倾角 2.083°、双轮接触 951/951）；vx=1.5 m/s 由通过改为在原生 CPU 引擎自约 16.7 s 起失稳（PyTorch 控制器 15.2 s；训练引擎 warp 同条件 20 s 通过）。该判据属引擎相关边界，不再作为「通过」项引用；ONNX 图、参考向量、CSV 与 256 组数值检查不受影响。证据：`artifacts/stand-leg-motion-20261017/trainfric-rerun/`（本机）。

交付的 ONNX 与本次已验证导出文件逐字节一致（直接复制，未重新导出）。本次未运行训练、Cube.AI 导入或真机验证；H7 的 39 维观测、50 Hz 调度及数值对齐仍由部署负责人待确认；未修改 `deployment/H7_RL`。
