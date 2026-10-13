# orientation −30：5000→6000轮续训

2026-10-11 18:45（北京时间）启动，4096环境，追加1000次PPO更新，目标累计6000轮。
使用冻结release `20261011T100539Z-getup-r15-orientation-weight`；92个源码/资产哈希与此前通过的检查一致。

- 来源：`20261011_164219_970332_getup_r15_pose_bias_removed_resume_4000_to_5000/model_4999.pt`。
- 来源SHA256：`759b36eaf0ec42ac340e814963836cccb06d352fe8124aff94e6f5db863d6ada`。
- 三阶段orientation权重−30，原公式保留；pose_bias关闭，knee_limit关闭。
- 完整恢复actor、critic、Adam和课程。恢复时三者张量/状态与来源逐项相同，学习率 `3.375e-5`。
- 累计5000轮开始，辅助力3000轮已退出，不重新计时；本次辅助力全程为零。
- 首次新更新生成 `model_5000.pt`，记录累计5001轮；actor已更新、actor/critic参数有限。
- 18:45:51核对时日志为迭代5027/6000，进程仍运行，无退出码。
- 随后最终入口检查结束时，续训日志已推进至迭代5181/6000，未产生退出码。

证据：[启动及恢复校验JSON](20261011-mjlab-r15-orientation-resume-5000-to-6000-start.json)。这些检查确认启动和状态恢复，没有验证新策略的最终行为效果。

## 远端实际命令

以下记录已经执行的训练命令，请勿重复启动同一组。

```bash
cd /root/gpufree-data/wheelleg-remote/releases/20261011T100539Z-getup-r15-orientation-weight
PYTHONPATH="$PWD/mjlab_training/src" MUJOCO_GL=disable \
  /root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u -m ct_mjlab.cli train \
  --training-profile getup-r15-orientation --num-envs 4096 --iterations 1000 \
  --seed 42 --device cuda:0 --run-name orientation_weight30_resume_5000_to_6000 \
  --resume /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_164219_970332_getup_r15_pose_bias_removed_resume_4000_to_5000/model_4999.pt \
  --resume-orientation-weight
```

TensorBoard命令（未启动）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/tensorboard \
  --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_184517_735702_getup_r15_orientation_orientation_weight30_resume_5000_to_6000 \
  --port 8080
```

后台任务记录及输出：
`/root/gpufree-data/wheelleg-remote/jobs/20261011T104504Z-getup-r15-orientation-resume-1000/`。
历史训练日志保留；新训练的50%辅助力配方见[减半说明](20261011-mjlab-half-assistance.md)。

后续：累计6000更新正常完成，最终模型已进行无辅助回放，见[6000轮训练结果](20261011-mjlab-orientation-resume-eval-6000.md)。
