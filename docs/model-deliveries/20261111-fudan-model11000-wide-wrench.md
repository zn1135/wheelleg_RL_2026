# 复旦11000轮PT：扩大随机化与六轴外载

2026-11-11按用户要求整理最新11000轮完整PT及来源清单。交付人Codex；接收人/H7独立复查人待确认。本次仅交付检查点，不更换默认选用模型，不导出ONNX或修改H7；没有独立行为验收结论。

| 项目 | 实际值 |
|---|---|
| 机器人/任务 | 本仓闭链轮腿，`fudan_ppo_walk`，原线性气弹簧、闭链与标定限位保持 |
| PT | `artifacts/fudan_ppo_walk/models/20261111_model11000_wide_wrench.pt`，1441545字节 |
| PT SHA256 | `619c3e8f06bdebd8c37d506e1276d7101dc14488a6bc5ae717bdae29996948ed` |
| 合同 | `fudan-ppo-walk-physical25-history125-critic144-v1` |
| 历史 | `fudan-physical-pd-centred-warmup-clean4-noisy1-unclipped-history-v2` |
| 张量 | Actor当前25＋历史125→6动作；Critic144；完整PT含encoder/actor/critic、两Adam与RNG，不是独立ONNX图 |
| 预算与计数 | 从10000新增9＋991＝1000完整轮，到11000；4096环境×48步、seed1；两Adam各新增20000、累计220000更新 |
| 物理/时序 | `legacy-linear-spring-v1`，附加拟合关闭；物理/PD2ms、策略10ms，动作6～14ms、传感2ms |
| 随机化 | 电机与质量/惯量倍率0.85～1.15，附加质量−1.5～3kg、COM±30mm、接触/零位/噪声加宽 |
| 外载 | 每回合机身各轴±10N/±10N·m持续外载，叠加7秒/0.3m/s小推扰 |
| 倒地处理 | 原重力方向阈值−0.1保持，连续等待100→0拍，倒地即失败结束；NaN仍报错停机 |
| 结束时间 | 2026-11-11 00:20:43训练退出0，00:20:52审计退出0，Asia/Shanghai |

## 来源与可核对条件

训练仓`https://github.com/zn1135/wheelleg_RL_2026.git`，分支`26_wheelleg`。实际运行基准为`c6a6bba1252bca48677f17bd509ed0c22f8a607c`及当时工作区改动；之后逐文件证明142个训练Python源码及记录资产与已提交的`320df84904bc9b52caba088fcc3015301fd2e389`一致，可用该提交核对运行时代码，无需额外差异补丁。这是事后匹配，不将后提交时间当作实际训练时间。部署仓版本/负责人复查待确认，无关联PR或本次板端改动。

[来源与配置清单](data/20261111-model11000-manifest.json)保存完整运行时SHA、env/PPO相关条件、前驱检查点、恢复与计数摘要；SHA256 `32e35e37f276a92c52fd5ddc5c3e4f6d74651e750ddc36bd098f4f505513d687`。结构XML SHA256仍为`505c9b9bb2b7a20e15cf9d2cd4ca664332db402687e34185d8a57de173dedbe8`。

恢复输入为首次失败前最后完整学习状态10009：`logs/mjlab/fudan_ppo_walk/20261110_wide_wrench_resume10000_to11000_01/model_final.pt`，SHA256 `4711e668f0bc809c43172172e40c3c157c0278a18a386ec94ad555290ba673a7`。网络、两Adam、RNG逐值恢复通过；物理/回合/历史现场重新初始化。该前驱与原始日志仍仅在本机，不随本包交付；最终11000PT可独立用于配套代码加载/另行授权续训，不声称完整训练逐轨迹重现。

日志为`logs/mjlab/fudan_ppo_walk/20261110_wide_wrench_resume10009_to11000_02/`；报告在`artifacts/fudan_ppo_walk/20261110-wide-wrench-resume10009-to11000-02/`。原始转移累计1646604288，包含首次失败rollout未用于下一更新的12288条采样，不能将其都视为优化样本。[训练完成摘要](../diagnostics/20261111-fudan-wide-wrench-11000.md)。

实际恢复命令及TensorBoard命令（已完成，不要重复启动原日志）：

```bash
scripts/mjlab.sh train --task fudan_ppo_walk --num-envs 4096 --iterations 991 --resume logs/mjlab/fudan_ppo_walk/20261110_wide_wrench_resume10000_to11000_01/model_final.pt --preserve-history-v2 --precheck artifacts/fudan_ppo_walk/20261110-wide-wrench-resume10009-to11000-02/precheck/report.json --log-dir logs/mjlab/fudan_ppo_walk/20261110_wide_wrench_resume10009_to11000_02
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/fudan_ppo_walk/20261110_wide_wrench_resume10009_to11000_02 --port 6007
```

训练使用MuJoCo Warp GPU/RTX4060 Laptop，Isaac不适用。本次读取PT工具为本仓虚拟环境；当前交付环境PyTorch2.14.0、MuJoCo/MuJoCo Warp3.11.0、Warp1.17.0。训练依赖未另行完整冻结，不仅凭Git SHA宣称外部依赖也全部可复现。

## 验证与使用边界

| 阶段 | 结论 |
|---|---|
| 轮数/优化器/RNG恢复、16项奖励日志及源码核对 | 通过；9＋991轮完整预算已完成 |
| PT身份/参数 | `model_11000.pt`与`model_final.pt`载荷和SHA一致；复制校验、全部参数有限和std正值通过 |
| 软件路径/预检 | 六轴施力与局部复位、随机化、倒地失败/NaN停机等检查及同条件GPU预检通过；不是行为验收 |
| 11000模型独立GPU站立/行走/起立、987矩阵 | 未执行 |
| ONNX结构/数值、板端数值/最坏耗时、真机行为 | 未执行 |

首次10009物理NaN的处理是更早结束已倒地的失败回合，不是对任意姿态物理求解的根治。此前8000/10000或拟合物理的成绩不自动适用于本模型；回退可使用已提交的10000 PT及其交付记录。允许用于后续离线评估或另行授权续训，不声明已可部署；下一步优先显式加载11000做原成功工况回归和强外载GPU验证，H7负责人独立核对待确认。
