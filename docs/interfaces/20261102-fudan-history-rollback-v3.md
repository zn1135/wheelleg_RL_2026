# 接口记录：复旦历史回退v3

后续变更：按[GPU最终验证约定](20261102-fudan-gpu-validation.md)撤销CPU物理后端；本页Native/两引擎核对记录仅代表当时实现与检查。GPU历史、扰动与模型合同保持。

| 项目 | 内容 |
|---|---|
| 记录日期、作者、复查人 | 2026-11-02，Codex；未由H7负责人复查，待确认 |
| 机器人、任务 | 本机实体闭链串联轮腿，`fudan_ppo_walk` |
| 旧接口 | [物理25/125张量v1](20261101-fudan-ppo-walk.md)、[reset/history v2](20261101-fudan-reset-v2.md) |
| 新接口 | 张量仍为`fudan-ppo-walk-physical25-history125-critic144-v1`；reset/history升为`fudan-physical-pd-centred-warmup-noisy5-clipped-history-v3` |
| 训练源 | https://github.com/zn1135/wheelleg_RL_2026.git，分支`26_wheelleg`，HEAD `3794efd390a0eb68cb154839423a7d53e32ff61e`＋未提交工作区；以本次[源码及检查记录](../../artifacts/fudan-history-rollback-v3-20261102/README.md)为准 |
| 部署侧 | H7代码/固件未修改、未核对，负责人待确认；没有独立部署参考包交付 |
| 权威实现 | `env.py::{reset_idx,observations,step}`、`measurement.py::{configure_samples,update_evaluation_commands}`、`native.py::Native.inputs`、`runner.py::FudanRunner.{load,save}`、`export.py::history_metadata` |

## 只改变历史语义

| 项目 | v2 | v3 |
|---|---|---|
| 初次历史/正常局部复位 | 4帧无噪声＋1帧带噪声 | 同一次采样得到的带噪声、clip后帧重复5次；current与最新历史帧相等，不额外重抽噪声 |
| 后续历史 | 带噪声、clip前帧 | 带噪声、clip后帧；每10ms仅FIFO推进一次 |
| current限幅 | ±100 | ±100，保持 |
| 局部复位 | 只重填选中环境 | 仍只重填选中环境，其余环境正常推进一次，直接`reset_idx(ids)`不改其历史 |
| 物理初态 | 当前PD默认中心（含固定启动偏置）的合法闭链解＋2mm网格间隙，根六速度±.5 | 保持，**不回退为旧固定0.26m** |
| 启动/缓存 | 一次10ms零动作热步，然后随机首轮局部相位；fresh reset、清动作 | 保持，不恢复旧跨回合缓存或current/history两次独立噪声bug |
| 字段/尺度/顺序 | 25当前帧＋125历史，oldest→newest；critic环境141＋估速3；实体动作6 | 全部保持，逐槽定义见物理接口v1 |
| 控制与物理 | 策略10ms、PD/物理2ms；原实体PD/限幅/极性/弹簧/质量/XML/随机化 | 全部保持 |
| 奖励/算法 | 当前16项奖励、48采样步、原PPO/encoder两Adam | 全部保持 |

新回退版本不能复用`fudan-trial01-fixed026-noisy5-clipped-history-v1`标记。v3仅回退历史填充和clip顺序，不代表恢复trial01的完整训练状态。

`reset_idx`先标记所选环境待填充，等该拍唯一一次观测采样后才重复5份。训练启动首次对策略可见的帧仍来自零动作热步之后；正式评估`configure_samples`仍从夹具t=0初始化，不在时间轴外增加热步。两条入口都按自身实际帧填历史。

## 兼容、恢复与导出

- 新训练默认v3；v2→v3不允许静默恢复。只有显式`--resume <v2模型> --migrate-history-v2`才允许这对指定版本迁移，仍严格核对物理张量合同、参考版本、PPO配置、完整更新和计数；legacy/missing/unknown合同拒绝。
- 本次指定源为7000轮`model_final.pt`，SHA256 `ea5dac38af946359321e415aa661bc78cc929ddf8f29d46fec6f0116c610c9d2`。保留模型、两个Adam全部状态/学习率、CPU/CUDA RNG、7000轮/860160000转移/两组各140000更新及课程时钟；物理场景、历史、局部回合重新初始化，不能称完整现场恢复。
- `restore_report.json`、checkpoint及训练报告记录源/目标reset合同和`history_migration`；新v3后续同合同恢复不重复迁移。原7000文件继续标v2，绝不将原文件改标成v3。
- GPU评估按checkpoint版本选择v2或v3环境；Native按夹具版本构造历史，模型与夹具仍须一致。旧v2重放继续使用原4净1噪/clip前语义；旧legacy原生输入逻辑保留，但GPU不把它当PD中心版本接入。
- ONNX图/张量形状不变；`interface.json`增加由**源模型实际版本**推导的历史预处理说明。导出旧7000仍标v2，不因当前源码默认v3就改标。本次只检查该元数据函数，未执行ONNX导出或图数值对比；没有新训练模型。实机使用实际传感读数，不额外加入仿真人工噪声。
- 原v2物理预检不能自动批准v3；新预检未运行，不能用本次静态/夹具检查或测试临时proof冒充真实预检。

## 同步与验证

| 侧 | 本次状态 |
|---|---|
| 训练/评估/Native/导出元数据 | 语义已同步；[31项输入检查](../../artifacts/fudan-history-rollback-v3-20261102/check-history-v3.json)通过，替代物理后端，张量/NumPy首帧与后续历史最大误差0 |
| 学习状态迁入 | [CPU/CUDA真实载荷检查](../../artifacts/fudan-history-rollback-v3-20261102/check-history-migration-final.json)通过：模型/两Adam/学习率/RNG/计数逐位或数值一致，未做前向/更新；临时迁入夹具模型已删除 |
| 旧v2保留 | 原正式`check_reset_v2`检查通过；保留旧模型/快照/视频/预检。旧源码保存在本次before目录及原训练source |
| 新物理/新策略行为 | 未运行，当前只准备；追加训练轮数尚未获得明确回复 |
| H7/板端时序/实机 | 未修改、未验证，负责人待确认 |

关联[模型准备记录](../model-deliveries/20261102-fudan-history-v3-preparation.md)。这次历史v3与[技术评分v2](20261102-fudan-walk-acceptance-v2.md)是两项独立变更，评分改善不能算作历史回退后的行为成绩。
