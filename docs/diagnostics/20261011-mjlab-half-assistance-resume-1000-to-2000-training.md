# 50%辅助力配方：1000→2000轮续训

2026-10-11 20:17:29（北京时间）在原RTX4090远端启动，4096环境、追加1000次PPO更新，目标累计2000轮，seed42。

来源为50%辅助力从头训练组的 `model_999.pt`，记录累计1000更新。
来源SHA256：`3ffaaaeb8fd367f45119991c392f490c4d11f1bc390baec1f408b71718415b4f`。
使用原冻结release `20261011T104817Z-getup-r15-half-assistance-final`，93个源码/资产文件与源训练快照哈希完全一致。

## 配置及恢复检查

- 实际接口：`chuanliantui-mjlab-getup-curriculum-r15-orientation-half-assistance-v1`，同配方直接续训，无跨接口迁移。
- 三阶段orientation权重−30；symmetry仍为−0.02/−0.02/0，行走阶段未启用左右角度差惩罚。
- 辅助力上限0.5倍整机重量、3000更新退坡进度继续保留。恢复课程为行走阶段，辅助力全程为零，课程不会回退。
- actor、critic、Adam初始恢复状态与来源逐项相同；环境及接口相同，课程阶段、计数与历史相同。
- 初始优化器学习率 `2.2500000000000008e-05`；恢复后累计更新1000，首个新模型 `model_1000.pt` 记录累计1001更新，actor已经变化，actor/critic参数有限。
- 20:18:27检查时日志迭代1032/2000，后台进程运行，尚无退出码。

证据：[启动及恢复JSON](20261011-mjlab-half-assistance-resume-1000-to-2000-start.json)。此处确认启动及恢复，没有声称续训后行为改善。

## 实际训练命令

以下命令已经执行，请勿重复启动同一组。

```bash
cd /root/gpufree-data/wheelleg-remote/releases/20261011T104817Z-getup-r15-half-assistance-final
PYTHONPATH="$PWD/mjlab_training/src" MUJOCO_GL=disable \
  /root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u -m ct_mjlab.cli train \
  --training-profile getup-r15-orientation --num-envs 4096 --iterations 1000 \
  --seed 42 --device cuda:0 --run-name half_assistance_orientation_resume_1000_to_2000 \
  --resume /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_185145_808090_getup_r15_half_assistance_orientation_half_assistance_orientation_scratch_1000/model_999.pt
```

TensorBoard命令（未启动）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/tensorboard \
  --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_201742_781620_getup_r15_half_assistance_orientation_half_assistance_orientation_resume_1000_to_2000 \
  --port 8080
```

任务记录与后台输出：
`/root/gpufree-data/wheelleg-remote/jobs/20261011T121729Z-half-assistance-resume-1000-to-2000/`。
新日志独立保存；目标末模型为 `model_1999.pt`，原1000轮模型与历史日志保留。
