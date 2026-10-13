# mjlab r13 宽高度门控检查

2026-10-10，分支 `26_wheelleg`，HEAD `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交改动。用户要求“改为宽高度门控”；接口见 [r13](../interfaces/chuanliantui-mjlab-wheelleg-recover-r13.md)。

## 变更与结果

四项线速度／偏航速度跟踪奖励门控由3～6 cm改为5～10 cm。误差≤5 cm全额、7.5 cm半额、≥10 cm零；高度跟踪奖励及严格起立±3 cm不变，knee_limit保持0。本次未启动训练。

- 64组独立输入：显式3～6 cm配置的27项奖励与参考当前源码逐项一致，最大误差0；默认宽门控只改变四项速度跟踪奖励。
- 正负高度误差边界／中点：四项奖励均与参考未门控值乘5～10 cm门控一致，包含非零平方项。
- 配置对照：仅已知机器人、动态模式、动作顺序、膝惩罚权重、初态及本次门控差异；PPO配置与参考相同。
- GPU：32环境×32控制步，seed42，320物理子步；每个积分qpos独立重算端点距离，最大闭链偏差2.246 mm，小于5 mm；数值失败0。零请求动作下普通终止2次，不是行为验收。
- 新checkpoint保存／加载、ONNX临时导出通过；ONNX最大绝对误差1.19e-07。
- 实际r11／r12模型可加载推理，回放配置保留3～6 cm门控；旧版本普通续训及r11膝惩罚单项迁移均被拒绝，拒绝后训练状态未改变。历史r5～r10回放／接口检查也通过。

完整结果见 [JSON](20261010-mjlab-r13-wide-height-gate-check.json)。检查脚本只初始化、有限步进、临时保存和加载，不执行PPO更新。不能据此声称宽门控改善了站高或策略恢复率。

## 复现

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_reference_training.py --num-envs 32 --steps 32 --seed 42 --out /tmp/chuanliantui-r13-wide-height-gate-check.json
```

历史日志／模型未覆盖，参考工程未修改。此结果针对本次r13配置；r12退化原因见 [分析](20261010-mjlab-r12-degradation.md)。
