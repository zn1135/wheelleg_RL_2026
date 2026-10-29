# model1999 ONNX实际导出交付

2026-10-30，用户要求先导出1999，再比较部署代码并拟定提交；执行人Codex，接收人用户，独立复查人待确认。按docs/deployment-contract.md及docs/templates/model-handoff.md记录，不将旧备份验证当作本次执行。

## 代码与训练来源

- 训练仓`https://github.com/zn1135/wheelleg_RL_2026.git`，26_wheelleg，HEAD `518360847bc804de5396edd75abc565174b1d96e`；有作者复制的部署改动及本次导出未提交内容。已跟踪差异见workspace.patch，必要未跟踪源码及历史内核见source-snapshot.zip，哈希见SHA256SUMS。
- 部署来源`https://github.com/zn1135/H7_RL.git`，main，提交`29ef90741358102c8992407f8ee40fd48b7693af`及既有格式差异。本轮没有修改部署控制代码。
- 检查点`artifacts/height-response-a2500-20261012/models/model_1999.pt`，SHA256 `1c3487a7da8d1fe74c418466ec4068f621c162e36406c14786c972a2839e7027`。完整PT含actor/critic/优化器，导出只消费确定性actor。
- 历史源码备份提交`e451b3d491b693de3d970f83a958a8dc0a2208ec`。任务chuanliantui_scut_v14_height，机器人华南虎闭链轮腿；实际env/PPO配置、接口标识和工具版本见delivery.json。原训练命令/seed/设备及完整资产运行快照本次未复核，保持待核实；不根据文件编号补写。未新增训练更新。

## 本次执行

使用`mjlab_training/.venv/Scripts/python.exe`，CPU导出环境通过被忽略的本地pth复用已有独立CPU导出依赖，不使用系统Python、不导入mjlab仿真；不宣称该环境满足完整训练依赖。Python/PyTorch/ONNX/ORT/NumPy实际版本和导出入口哈希在delivery.json。

```powershell
& mjlab_training/.venv/Scripts/python.exe scripts/agent/export_h7_height1999.py --checkpoint artifacts/height-response-a2500-20261012/models/model_1999.pt --output C:/Users/ynz/AppData/Local/Temp/height1999-export-38e5087514374e59abec059290d1ce39
```

退出0，校验后复制到exports/onnx/model_1999。ONNX SHA256 `5d45c516c87eb554f8ffcdd4d28e31b197abff2cb8bdb595d6e2b1989599e5a4`；float32/batch1/opset13，输入observation[1,35]、输出action[1,6]。原始PT和历史模型不覆盖。

## 分阶段验证

| 阶段 | 本次状态与证据 |
|---|---|
| actor结构与权重 | 根据实际ELU256/128/64、无归一化配置严格加载全部MLP键及形状 |
| ONNX图检查 | onnx.checker通过 |
| 数值对照 | seed42，64组正态随机输入含零向量；最大绝对差3.0994415e-6，小于1e-5；参考向量及原始日志随包保存 |
| 新训练、Isaac/MuJoCo物理回放 | 本次未执行，不构造仿真环境 |
| 新ST转换、板端SDK/时序、真机 | 本次未执行，不复用历史结果冒充新验证 |

本次未提交、推送、创建PR或烧录；H7负责人签认待确认。本记录是模型导出交付，不是物理行为验收。
