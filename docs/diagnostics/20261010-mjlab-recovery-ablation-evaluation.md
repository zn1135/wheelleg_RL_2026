# 四组最终策略统一回放评估

本记录对应用户“回放评估／继续这个任务”。Windows工作区为迁移副本，无`.git`及mjlab运行环境；实际仿真沿用远端Linux原训练源码快照和Python3.12/mjlab环境。四组最终1000轮策略已完成统一回放评估。

## 模型与源码

四组各1000轮已正常完成，退出码均0，最终`model_999.pt`。源码为`20261010T141132Z-58947e8-ablation`，原`26_wheelleg`分支、HEAD `58947e8fabc4683b2025d42bda742b06c84dc566`加未提交改动；86个运行源码／资产SHA256再次核对一致。

分组：A=r14基线，B=高度正奖励宽度3.16→7cm，C=recover模式屏蔽四项速度奖励及静止速度惩罚，D=recover物理失败延迟20→250步。模型组别、源码哈希、全部训练配置、观测／控制契约与网络形状在加载时核对。

## 统一评估方法

- 每组32个普通初态、32个地面初态，seed42；共256条20秒回放。
- 复用同一份普通／地面初态qpos、qvel、命令快照，显式回写到各组环境。校验输入哈希、地形原点及回写误差；避免GPU四元数计算末位差异妨碍配对。
- 固定前进速度0、偏航速度0、目标高度0.30m；实体关节顺序、动态normal/recover、闭链与气弹簧沿用原模型。
- 标称物理，无观测噪声、动作／观测延迟随机化和域随机化；确定性actor推理，不学习。
- 严格起立：高度误差≤3cm、roll/pitch≤15°、双轮接触力均>5N、机身接触力≤1N，连续0.5秒。
- 稳定末段：最后完整5秒中至少95%的时间符合严格起立条件。
- 统一物理失败记录：机身接触>1N或roll/pitch>40°连续20步，单独记录首次发生时间。物理失败之后继续观察恢复机会，四组均运行满20秒；数值异常样本复位后永久排除，避免后续复位被误计成原回合恢复。
- D组训练中延长失败容忍不会放宽上述成功／失败定义。20秒物理存活与严格起立分别报告。
- 最后5秒高度误差直接按实际轨迹求平均，包含有限但未站起的样本；不把训练中按回合期限归一化的height_error直接当成实际厘米误差。

本次仅验证零速度起立／静站，行走、速度跟踪及训练随机化鲁棒性未在本评估中验证。

## 执行与产物

入口：[评估脚本](../../scripts/agent/eval_chuanliantui_recovery_ablation.py)、[轨迹渲染脚本](../../scripts/agent/render_chuanliantui_ablation_replay.py)、[高度绘图脚本](../../scripts/agent/plot_chuanliantui_ablation_evaluation.py)。

远端新输出目录：`/root/gpufree-data/wheelleg-remote/evaluations/20261010_999_windows_job/results_v4`；本机产物目录：`artifacts/recovery-ablation-eval-999-20261010/`。

远端实际命令，在原源码PYTHONPATH及`MUJOCO_GL=disable`下：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python /tmp/eval_chuanliantui_recovery_ablation_windows_v4.py --campaign /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_141132_recovery_ablation --output /root/gpufree-data/wheelleg-remote/evaluations/20261010_999_windows_job/results_v4 --iteration 999
```

可视回放使用GPU轨迹保存的qpos，在同版本MuJoCo中执行`mj_forward`渲染，未重新积分物理。视频每组展示同一编号样本0；汇总指标覆盖全部32个样本。普通／地面分别生成四组并排视频。

先前加载预检发现两项运行期序列化差异：Scene构建会同步terrain数量／间距，`python -m`运行时实验函数名记录为`__main__`。修复仅规范这两类已核实字段，保留完整源码、其他配置与接口校验。前一次回放配对检查发现四元数末位差异约3.73e-9，故采用显式初态快照。未完成的尝试保留在远端results／results_v2／results_v3及对应控制台，最终结论使用results_v4。

为生成离屏视频，在远端补充EGL／OpenGL／OSMesa渲染库，EGL渲染已验证可初始化；未改模型物理、训练配置或驱动。

## 结果

四组各64条、合计256条回放全部运行满20秒，数值异常均0；严格起立、稳定末段均0。两类初态输入哈希在四组完全一致，显式回写后的qpos／qvel最大差异均为0。

| 组 | 严格起立普通／地面 | 最后5秒平均站高普通／地面 | 最后5秒平均高度误差普通／地面 | 统一20秒物理存活普通／地面 |
|---|---|---|---|---|
| A 基线 | 0/32、0/32 | 0.2132、0.2039 m | 8.68、9.65 cm | 32/32、20/32 |
| B 放宽高度正奖励 | 0/32、0/32 | 0.2226、0.2078 m | 7.74、9.22 cm | 32/32、23/32 |
| C recover奖励调度 | 0/32、0/32 | 0.1472、0.1490 m | 15.28、15.10 cm | 0/32、0/32 |
| D recover失败延迟 | 0/32、0/32 | 0.2143、0.2108 m | 8.57、8.92 cm | 32/32、31/32 |

接触诊断（最后5秒，每类32个样本的平均时间占比）：

- A、B、D普通初态中，机身接触占比0、腿部接触且机身离地占比100%。它们靠腿部撑地稳定在低姿态，没有达到0.30m目标。
- A地面初态机身接触19.70%、腿部支撑79.24%、倒置18.75%；B地面初态机身接触28.13%、腿部支撑71.88%、倒置0；D地面初态机身接触0、腿部支撑100%、倒置0。
- C普通／地面初态机身接触均100%，倒置占比分别3.13%／6.25%，平均站高不足0.15m。

本批样本中B相对A提高普通初态平均站高约0.95cm，地面初态约0.39cm，仍未严格起立；D提高物理存活至63/64，但保持低姿态腿部支撑；C发生明显机身落地。四个单因素配置在这批最终策略中均未解决起立任务。物理失败定义未包含腿部接地和最低高度，因此低姿态腿撑地仍可能获得20秒物理存活；后续需要单独分析这种行为的奖励收益和任务成功／失败设计。

### 证据与回放

- [完整逐组指标、模型SHA256及高度轨迹](../../artifacts/recovery-ablation-eval-999-20261010/evaluation.json)
- [源码校验与训练退出状态](../../artifacts/recovery-ablation-eval-999-20261010/training-completion.json)
- [实际评估控制台](../../artifacts/recovery-ablation-eval-999-20261010/evaluation-console-v4.log)
- [普通初态并排视频](../../artifacts/recovery-ablation-eval-999-20261010/comparison-normal.mp4)
- [地面初态并排视频](../../artifacts/recovery-ablation-eval-999-20261010/comparison-ground.mp4)
- [高度对照图](../../artifacts/recovery-ablation-eval-999-20261010/height-comparison.png)
- [产物SHA256](../../artifacts/recovery-ablation-eval-999-20261010/artifact-sha256.json)：16个JSON／初态与轨迹NPZ／PNG／MP4已下载并与远端逐项核对一致。

录制模型MuJoCo3.11.0；视频为各组同编号样本0的原GPU轨迹离线渲染，指标涵盖全部64个样本。两个视频均为1280×320、201帧、10fps、20.1秒（包含t=0）；视频元数据与最终帧画面已核对。结果并非部署端或真机验收。Windows侧未执行原生仿真；没有新训练、改动H7控制代码、提交Git或覆盖历史训练日志。
