# r9关闭膝限位惩罚后正式训练

用户授权“关闭新增膝限位惩罚，再训练”。2026-10-10 10:17:12（北京时间）启动4096环境、1000次PPO更新，seed42／cuda:0，无窗口，从随机权重开始。分支26_wheelleg，基础版本58947e8fabc4683b2025d42bda742b06c84dc566加未提交改动。

新增knee_limit奖励惩罚权重-0.5→0；保留动态normal/recover、动作顺序[lf0, lf00, rf0, rf00, lfwheel, rfwheel]，其他26项奖励、PPO、物理与随机化沿用r8。有限预检通过，GPU最大闭链误差2.578 mm、数值异常0、奖励贡献为零，见 [预检记录](20261010-mjlab-r9-no-knee-limit.md)。

已执行（运行中，不要重复启动）：

```bash
scripts/mjlab.sh train --num-envs 4096 --iterations 1000 --seed 42 --device cuda:0 --run-name wheelleg_recover_r9_no_knee_4096_1000
```

实际输出：`logs/mjlab/chuanliantui/20261010_101723_976809_wheelleg_recover_r9_no_knee_4096_1000`。训练PID125190、监督PID125189；终端日志`/tmp/chuanliantui-mjlab-r9-no-knee-4096-1000-20261010_101712_365552.log`。监督进程结束时将更新`/tmp/chuanliantui-mjlab-r9-no-knee-4096-1000-20261010_101712_365552.json`中的退出状态，运行在独立会话。

2026-10-10T10:19:25.883441+08:00记录时，已观察iteration0～36，model_0.pt及TensorBoard事件已保存；日志knee_limit贡献全部为0，无Traceback、CUDA OOM或RuntimeError。已核对interface.json、environment.json、runner.json、checkpoint接口及47份源码／资产快照哈希，与当前工作区一致。首个checkpoint SHA256：`49fb5b61878e8f1afa0f1a6563d960b7647030725aac4ef6de43fbecef36ec1d`。

TensorBoard（仅交付，未启动）：

```bash
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui/20261010_101723_976809_wheelleg_recover_r9_no_knee_4096_1000 --port 8080
```

[启动JSON](20261010-mjlab-r9-formal-training.json)记录观察时状态；这次只确认训练已开始，不表示1000轮已完成或起立效果通过。完成后需要新策略回放／CPU行为评估及H7适配。未提交Git。

## 训练完成

2026-10-10 11:14:55（北京时间）正常结束，退出码0，完成1000次PPO更新／98,304,000条transition，训练计时57分03秒。model_999.pt及每50轮检查点已保存；日志无Traceback、CUDA OOM或RuntimeError。最终模型SHA256：`4696c9dc01caaad52e1827f599221f57d40a142ff9d32d7e6fc09972384e44d3`。

末轮平均reward-12.11，平均回合67.81步（约1.36秒），动作std0.48；strict_recovered及strict_stand_fraction均0，数值异常0，knee_limit贡献0。末轮height_error0.0090不是本回合平均绝对高度误差：源码按20秒固定回合期限归一化累计误差，提前失败的短回合会显著压低该值；不能称为站高误差9mm。

最终策略行为评估见 [r9评估](20261010-mjlab-r9-model999-evaluation.md)。
