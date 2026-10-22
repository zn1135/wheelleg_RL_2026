# 新轮端响应续训结果（2026-10-21）

**授权的2048环境×额外1000轮已正常完成。新模型适应新物理后的速度/转向跟踪明显改善，但不是完整行为或真机gap验收通过；保留为候选，不替换既有基线。**

## 训练完成证据

结束时间：2026-10-21T19:41:11.749906+08:00，退出0；总计1000次PPO更新、49,152,000次环境交互。日志编号5849～6848（两端包含），最终文件名model_6848是零起始循环的编号，不是只跑了999轮。checkpoint保存的common_step_counter比父模型增加24000（每轮24个策略步）。物理步2ms、策略周期20ms。

父模型：`artifacts/flat-track-20261017-trained/models/model_5849.pt`，SHA256 `d4be0bc98c20348911b7e43a6f52f5e3719cb78d74f5764f4854bce82bb71ca7`。

最终模型：`logs/mjlab/chuanliantui_huananhu_multimode_v1/2026-10-21_19-09-28_h7-wheel-joint-sharp-2ms/model_6848.pt`，SHA256 `4a96a4c216fed4cedc2153667defbc0ae7974b9a31eb857022a31b4658bb3001`。

68个checkpoint张量全部有限，日志未检出NaN/Inf/异常栈，训练期间启动源码/XML/姿态库/父模型哈希均保持。无自动重启/追加；没有启动TensorBoard、操作实机或恢复已暂停的关节校验。

训练奖励首次100条有值日志均值80.439，最后100条83.265；最后一条82.21，平均回合长度1000步（20秒）。这些是训练内部统计，不等于实机性能或验收成功率。

## 同一新物理下的父模型对照

两者均读取当前joint_sharp共享XML，2ms；play配置关闭域随机化、噪声/延迟与推扰。64个强制地面初态回放20秒，seed42，实际全部为平地；不能把--terrain all汇总选项说成测试了楼梯。相同初态指纹：`4a0b867588ff23aa9f7edef36dc27917ef4713e7aaebfb403c4c8016ef51d53b`。

| 起立检查（非完整v3.1验收） | 父model5849 | 新model6848 |
|---|---:|---:|
| 达到eval严格起立条件 | 64.0000 | 64.0000 |
| 完成20秒 | 64.0000 | 64.0000 |
| 数值失败 | 0.0000 | 0.0000 |
| 起立时间中位数s | 1.0600 | 1.0800 |
| 严格条件保持比例均值 | 0.9983 | 0.9981 |

父模型原成功起立样本丢失0个。两者初始即满足宽松stable条件，eval只做有限起立/保持诊断，不等于从各种倒扣姿态恢复，也未执行中途击倒的v3.1最终平衡配对。

平地跟踪：vx、yaw各取0/±0.5/±1.0/±1.5，目标高度0.27/0.34m，共98组固定命令；每组12秒、前2秒不计误差，seed42。阈值沿用脚本：线速度/角速度误差≤0.30、高度误差≤0.03m、倾角≤25°、无机身撑地并完成回放。

| 跟踪检查 | 父model5849 | 新model6848 |
|---|---:|---:|
| 通过组数 | 11/98 | 98/98 |
| 平均线速度误差m/s | 0.134957 | 0.076531 |
| 平均偏航角速度误差rad/s | 0.824749 | 0.033974 |
| 平均高度误差m | 0.004152 | 0.004518 |
| 平均侧向速度m/s | 0.049744 | 0.023899 |
| 每组最大倾角的平均值° | 2.281669 | 0.777606 |

线速度误差降低约43.3%，偏航角速度误差降低约95.9%；高度误差由4.152mm增加到4.518mm（+0.366mm，约8.8%），仍在原阈值内。父模型11个原通过命令无丢失，两者98组均完成、无数值失败。父模型此前在旧XML下的98/98不直接拿来做此配对；本次11/98说明旧策略在新响应下的表现。

## 原生CPU MuJoCo短回放

固定NORMAL初态、同一新XML、目标高度0.27m，零速和vx1.5m/s各12秒。直接使用PyTorch actor，没有导出ONNX。速度/转向/高度跟踪判据和逐拍双轮接触判据分别报告；退出码0只表示命令运行完成。

| 模型/命令 | 速度误差m/s | 偏航误差rad/s | 高度误差mm | 跟踪通过 | 严格接触/姿态/高度通过 |
|---|---:|---:|---:|---|---|
| parent-cpu-zero | 0.0676 | 0.9803 | 5.284 | False | False |
| candidate-cpu-zero | 0.0444 | 0.0245 | 2.042 | True | True |
| parent-cpu-vx1p5 | 0.2595 | 0.8620 | 10.792 | False | False |
| candidate-cpu-vx1p5 | 0.0663 | 0.0290 | 8.461 | True | False |

新模型零速两项均通过；1.5m/s跟踪通过，但11.82秒一个20ms采样点未检出双轮同时接触（1秒后550/551点满足），所以严格接触检查未通过，不据此放宽阈值或宣称整体验收通过。父模型同条件两个命令跟踪均失败，主要是偏航偏差；其接触检查也失败。

## 当前结论与范围

这次续训有效改善了策略对**新仿真轮端响应**的适应，特别是转向跟踪；没有证明实机gap已缩小。高度误差略增、CPU高速工况一次双轮接触未满足，单seed和少数CPU命令不代表全面稳健。尚未做完整中途击倒v3.1配对、多seed/全包线CPU、ONNX数值/回放、实机结果。没有自动挑中间checkpoint、追加训练或把最终模型命名为最优/新基线。

## 复现与证据

从本独立worktree根目录运行既有正式入口；给新的输出目录，避免覆盖原报告：

```bash
scripts/mjlab.sh eval --checkpoint <model_5849.pt或model_6848.pt> --num-envs 64 --seeds 42 --start-mode force --terrain all --output <新目录>/recover.json
mjlab_training/.venv/bin/python scripts/agent/assess_multimode_tracking.py --checkpoint <model_5849.pt或model_6848.pt> --seeds 42 --output <新目录>/tracking.json
mjlab_training/.venv/bin/python sim2sim/mj_sim2sim_huananhu_multimode.py --checkpoint <model_5849.pt或model_6848.pt> --pytorch-controller --seconds 12 --cmd-vx <0或1.5> --cmd-yaw 0 --cmd-height 0.27 --report <新目录>/cpu.json
```

训练进程退出记录为artifacts/h7-wheel-train-20261021/exit.json，预算/哈希/有限张量检查为training-summary.json，配对结果result.json，完整逐样本与逐拍数据在evaluation/，训练源码快照在source/。本次不修改训练/控制/物理参数，不提交/推送；H7部署侧核对仍待确认。

![跟踪误差对比](../../artifacts/h7-wheel-train-20261021/tracking-comparison.png)
