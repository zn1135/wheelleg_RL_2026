# 保留训练线命令

仅⑥SuperMimic、⑦BC、⑧SMP。所有命令从仓库根目录执行；先设置数据、模型与本次授权预算，输出目录必须新建。

## BC与SMP先验

```bash
scripts/mjlab.sh train-bc --dataset "$BC_DATASET" --iterations "$BC_UPDATES" --output "$BC_RUN"
mjlab_training/.venv/bin/python -m tensorboard.main --logdir "$BC_RUN" --port 8080

scripts/mjlab.sh train-prior --dataset "$SMP_DATASET" --iterations "$SMP_UPDATES" --output "$SMP_RUN"
mjlab_training/.venv/bin/python -m tensorboard.main --logdir "$SMP_RUN" --port 8081
```

观察成功判据：本次新输出目录出现checkpoint和训练日志；这不代表闭环运动通过。数据格式、模型初始化和续训规则查[BC说明](../mjlab-behavior-cloning.md)、[SMP说明](../mjlab-smp-prior.md)。

## BC初始化／SMP辅助PPO

```bash
scripts/mjlab.sh train --task smp --smp-prior "$SMP_PRIOR"   --bc-init "$BC_CHECKPOINT" --iterations "$PPO_ITERATIONS" --run-name "$PPO_RUN_LABEL"
# 将入口打印的Training output完整目录填入PPO_RUN，再开另一个终端。
mjlab_training/.venv/bin/python -m tensorboard.main --logdir "$PPO_RUN" --port 8080
```

初始化只复制经过校验的actor；完整续训使用`--resume`，不能同时传`--bc-init`。此处PPO属于保留的⑦⑧，不恢复①～⑤独立训练。

## SuperMimic

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m ct_mjlab.supermimic train   --seconds "$SM_SECONDS" --log-dir "$SM_RUN"
mjlab_training/.venv/bin/python -m tensorboard.main --logdir "$SM_RUN" --port 8080
```

检查点迁移、配方和初始化参数必须按[SuperMimic说明](../mjlab-chuanliantui-supermimic.md)核对；本节只说明入口，不指定当前推荐模型。

## 初态预检和回放

只改变初始位置或复位分布时，训练前交付20环境可视预检：

```bash
scripts/mjlab.sh play --task smp --num-envs 20 --smp-prior "$SMP_PRIOR" --checkpoint "$PPO_CHECKPOINT"
```

成功判据：观察20个环境的初态无穿地、弹飞或非预期分布。SuperMimic按其自身play入口和对应模型合同检查；不将其他模型互套。MuJoCo独立仿真仍使用原`sim2sim/`入口。
