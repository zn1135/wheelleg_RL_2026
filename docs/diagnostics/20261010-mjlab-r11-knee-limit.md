# r11 膝限位惩罚恢复检查

2026-10-10，用户要求“加上限位惩罚”。分支 `26_wheelleg`，HEAD `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交改动；新接口r11。恢复参考工程两实体被动膝lf1/rf1的knee_limit，权重0→-0.5、余量0.15 rad。公式保持参考值；其他奖励／PPO、物理、初态、动态模式、动作和观测沿用r10。

## 已执行

```bash
mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_reference_training.py --num-envs 32 --steps 128 --seed 42 --out /tmp/chuanliantui-r11-knee-limit-check.json
```

- 退出码0；27项奖励权重与参考一致，随机输入公式最大误差0。
- 单侧距限位0.30/0.15/0.075/0/-0.15 rad，对应原始项0/0/0.25/1/2.25，覆盖软区入口和越界封顶。
- 启用／关闭本项的总奖励差与-0.5×控制步长×原始项一致；最大误差5.58793545e-08。
- 32 GPU环境、128控制步／1280物理子步、seed42；有限检查无PPO更新。
- 独立CPU运动学检查每个GPU积分后qpos（含自动复位前末帧）：最大闭链偏差2.5263 mm，小于5 mm；数值异常0。零请求动作物理失败／普通复位81次，不作为起立行为成功证据。
- 2048次真实复位，地面比例48.8281%，地面root高0.25m／单位四元数精度检查通过，普通root最低0.38m；模式、动作映射、部分复位均通过。
- 新checkpoint保存／加载通过；临时随机actor ONNX最大误差1.1920929e-07，不是训练模型交付。
- 实际r10远端model_999仍可加载，历史膝权重0、50%地面分布和四腿两轮顺序保持；直接完整resume为r11会拒绝。r5～r9历史加载／拒绝跨版本resume检查也通过。

完整数值和本次源码manifest见 [JSON](20261010-mjlab-r11-knee-limit-check.json)，配置语义见 [r11接口](../interfaces/chuanliantui-mjlab-wheelleg-recover-r11.md)。同时修正既有检查脚本高度扫点报告被GPU复位局部变量覆盖的问题，记录的是实际9个奖励扫点。

## 验证边界

本次仅恢复训练配置并执行数值／有限物理／接口检查，未启动新训练、TensorBoard或新策略行为回放。r10既有策略仍低姿态，见 [原评估](20261010-mjlab-r10-cloud-model999-evaluation.md)；此次奖励改动是否改善起立需训练后验证。未修改H7、机器人资产或历史logs。远端旧release配置未自动覆盖。
