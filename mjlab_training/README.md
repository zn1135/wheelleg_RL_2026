# mjlab训练

只维护SuperMimic、BC与SMP，命令见[命令导航](../COMMANDS.md)。独立环境使用本目录`.venv/bin/python`，不得使用系统Python。

- SuperMimic：`PYTHONPATH=mjlab_training/src .venv/bin/python -m ct_mjlab.supermimic`（从仓库根目录时解释器为`mjlab_training/.venv/bin/python`）。
- BC：`scripts/mjlab.sh train-bc`。
- SMP先验：`scripts/mjlab.sh train-prior`。
- BC／SMP PPO：`scripts/mjlab.sh train --task smp --smp-prior ...`。

①～⑤的五个旧task均已撤销。`envs/chuanliantui*`部分模块继续作为⑥⑦⑧及MuJoCo的共用控制、观测、物理、模型接口，不提供旧独立训练入口。注册只保留`smp`和`supermimic`。

MuJoCo仿真源码、资源和必要模型路径保持；跨目录引用需按[清理记录](../docs/maintenance/20261101-retain-678-review.md)验证。每次训练另行明确预算，配套TensorBoard；检查不授权新训练。
