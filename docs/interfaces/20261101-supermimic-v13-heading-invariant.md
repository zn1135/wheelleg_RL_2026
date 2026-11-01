# V13世界模型：航向无关输入与机体系速度

## 范围与来源

2026-11-01按用户要求对齐`/home/zn/文档/supermimic_wheelbipe/supermimic_wheelbipe/mdp.py::invariant_features`和`rotations.py::yaw_mat/mat_to_6d`。参考代码是社区轮腿移植，不将其声明为论文作者实现。

机器人/任务：串联轮腿SuperMimic。训练仓`Wheel-Legged-Gym`、分支`26_wheelleg`，基于`9f5eacd`后的共享工作区；本次仅在前次移除历史动作基础上修改`supermimic/motion.py`、`learner.py`和对应检查/文档，保留已有其他改动。

- 旧合同：`chuanliantui-supermimic-motion-only-alternating-bptt-v13-no-action-history-v1`。
- 新合同：`chuanliantui-supermimic-motion-only-alternating-bptt-v13-no-history-heading-invariant-v2`。
- 模型输入仍129维、输出仍5×46=230；形状相同不表示语义兼容。

## 新状态输入布局

| 索引 | 内容 | 坐标/单位 |
|---|---|---|
| 0:1 | 根高度 | 原高度单位m |
| 1:7 | 去掉航向后的旋转矩阵前两列，按行展开 | `HᵀR`，6D表示 |
| 7:10 | 根线速度 | `Rᵀv_world`，m/s |
| 10:13 | 根角速度 | MuJoCo free-joint qvel原本为机体系，rad/s |
| 13:29 | 非轮关节位置 | 原顺序和各关节单位 |
| 29:47 | 所有关节速度 | 原顺序和各关节单位，包含轮速 |
| 47:122 | 物理参数 | 仍75维；最后6维外力/力矩改为机体系 |
| 122:128 | 当前6动作 | 原物理顺序/raw尺度 |
| 128:129 | 回合年龄 | 秒 |

`H`由机身前轴在水平面的投影构造，使用参考代码相同的`1e-9`正则项；不输入绝对水平位置或绝对航向。机身前轴恰好竖直时航向本来就不唯一，检查仅保证该点数值/梯度有限，不声称其方向定义唯一。

根线速度要旋转，根角速度不能再转一次；参考仓的角速度来源是世界系，而当前MuJoCo广义角速度已是机体系。完整参考特征对照中先将本机角速度转换成参考代码的世界系输入，验证最终结果一致。

`capture_parameters`仍保存原世界系`xfrc_applied`；仅在神经网络输入中旋转力和力矩。它们必须随整体yaw一起转换，否则外力参数会继续泄漏绝对朝向。真实控制力、施力代码和物理环境未改。

## 保持与兼容性

- 真实47维运动状态、254维记录容器、物理路径、输出增量和积分仍保持原坐标，不把局部输入特征误当真实状态。
- 控制策略Actor25/历史125、估速器、价值网络输入结构、六动作、PD、延迟、100/500Hz、XML和H7不变；前两拍动作仍不进入世界模型MLP。
- 算法合同和损失配方记录新输入坐标。旧141维、旧129维绝对坐标检查点均不可直接完整续训；旧actor单独继承/推理保持。
- 未做旧首层/归一化自动转换，尚未训练新权重；后续训练需另获预算。没有模型部署交付，H7同步不适用，真机效果未验证。

## 验证

```bash
PYTHONPATH=mjlab_training/src CUDA_VISIBLE_DEVICES='' mjlab_training/.venv/bin/python -m scripts.agent.supermimic.check_supermimic_heading_inputs --output artifacts/<新的坐标检查目录>
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.supermimic.check_supermimic_no_history --integration --output artifacts/<新的采集检查目录>
```

[坐标报告](../../artifacts/supermimic-v13-heading-input-check-20261101-r1/report.json)：参考特征最大差3.58e−7，整体yaw/平移输入差1.43e−6，预测路径变换差5.96e−7；包括非零世界外力/力矩、四元数q/−q等价、有限梯度和同维旧合同拒绝。随机非零网络头参与检查，避免零初始化输出使对称性检查空过。优化器更新0。

[采集与恢复检查](../../artifacts/supermimic-v13-heading-input-integration-20261101-r1/report.json)另列CPU/GPU证据。程序检查不代表预测精度、平衡或运动行为提升；未做训练或真机验证。
