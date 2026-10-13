# 四组恢复单因素并行训练

用户授权“同时测试多组”及“开始”，按默认方案四组各4096环境、1000轮、seed=42，从相同随机网络起点训练。2026-10-10 22:16:05（北京时间）在原RTX4090实例统一开始学习。没有从历史1000轮模型续训。

## 分组

| 组 | 唯一改动 |
|---|---|
| A baseline | r14基线 |
| B wide_height | 高度正奖励核分母0.001→0.0049，宽度约3.16→7 cm；权重+1 |
| C mode_reward | recover模式屏蔽四项速度跟踪奖励及静止速度惩罚；normal保持基线 |
| D recover_timeout | recover模式接触／姿态失败延迟20→250步，normal仍20步；数值异常立即终止 |

动态模式、四腿两轮动作／观测顺序、50%普通／50%地面初态、r14速度与航向命令、机器人／气弹簧、PPO、膝惩罚0及速度奖励5～10 cm门控均相同。C使用终止管理器更新后的模式，与下一观测中的模式一致。D在更新模式前按当前模式选择延迟；普通模式出现持续失稳后仍通过原状态机进入恢复。

## 启动检查

- [配置差异](20261010-mjlab-recovery-ablation-config-check.json)只含约定奖励／终止函数替换。
- [公式检查](20261010-mjlab-recovery-ablation-semantic-check.json)：D除计数规则外与基线失败函数AST一致；20／250步边界及中断清零通过；B公式及数值屏蔽、C五项模式屏蔽通过。这不是行为效果验收。
- [远端GPU有限检查](20261010-mjlab-recovery-ablation-finite-check.json)：四组各32环境、32步；观测／奖励均有限，未观察到数值异常。
- [开始前核对](20261010-mjlab-recovery-ablation-start.json)：四组actor／critic参数及初始qpos、qvel、命令哈希一致；Adam状态均为空，学习率1e-4，地面初态2033/4096。初始化后总显存8962/24564 MiB，才释放共同START文件开始PPO。

## 源码与运行目录

- 分支 `26_wheelleg`，HEAD `58947e8fabc4683b2025d42bda742b06c84dc566`，包含工作区未提交代码；不代表仅该commit已验证。
- 源码快照：`/root/gpufree-data/wheelleg-remote/releases/20261010T141132Z-58947e8-ablation`，86个运行源码／资产文件SHA256与本机核对一致。该目录保存 `source-state.json`、`source-version.txt`、已跟踪文件 `source.diff`，完整新增文件包含在快照。
- 训练目录：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_141132_recovery_ablation`；四个子目录为 `baseline`、`wide_height`、`mode_reward`、`recover_timeout`。
- 每组保存 `model_initial.pt`、`ready.json`、`interface.json`、`environment.json`、`runner.json`；后续策略 `model_*.pt`、TensorBoard记录也在各组子目录。
- 控制台／PID／退出码：实验根目录 `<group>.console.log`／`<group>.pid`／`<group>.exit-code`。退出码只在组结束后生成。
- 实验入口：`mjlab_training/src/ct_mjlab/experiments/recovery_ablation.py`；启动器：`scripts/agent/launch_recovery_ablation.py`。独立接口含组名、完整配置和源码哈希，防止实验策略误当作r14普通策略加载。历史回放入口未改。

## 实际启动命令

已执行，不要重复启动同一路径。远端快照根目录：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python scripts/agent/launch_recovery_ablation.py launch --campaign /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_141132_recovery_ablation --num-envs 4096 --iterations 1000 --seed 42
```

启动器在核对前等待START。本次START在确认权重、初态和显存后生成，四个训练进程已经进入学习。

TensorBoard（远端另开终端，本次未启动）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m tensorboard.main --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_141132_recovery_ablation --host 127.0.0.1 --port 8080
```

本机端口转发：

```bash
ssh -p 30702 -i /home/zn/.ssh/id_ed25519 -N -L 8080:127.0.0.1:8080 root@183.147.142.40
```

访问 `http://127.0.0.1:8080`，四个run在同一面板比较。

## 效果验收

截至2026-10-10 22:18:43（北京时间），四个进程存活并持续学习。日志迭代索引A=34、B=33、C=33、D=32（索引从0开始）。四组 `model_0.pt` 的actor均已较初始模型变化、Adam状态非空，actor／critic参数均有限；最新数值异常指标均0。显存10466/24564 MiB、GPU利用率100%。见 [启动进度核对](20261010-mjlab-recovery-ablation-startup-progress.json)。

本轮只确认启动正常，尚无最终训练效果。后续同批初态／seed分别评价普通与地面样本，报告严格起立率、首次恢复时间、站高误差、20s物理存活率、速度／偏航误差和数值异常。严格起立仍要求高度±3 cm、姿态±15°、双轮接触、机身离地连续0.5s；D的失败容忍延长不改变成功标准。初期reward不可作为最终组间胜负结论。

本次未提交Git、改动H7部署、覆盖历史日志或更改实例关机设置。
