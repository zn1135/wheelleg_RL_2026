# r7动态模式恢复检查

2026-10-10，用户要求恢复normal/recover模式；26_wheelleg，HEAD58947e8fabc4683b2025d42bda742b06c84dc566，加未提交改动。未新建分支、未提交、未启动训练。接口见 [r7](../interfaces/chuanliantui-mjlab-wheelleg-recover-r7.md)。

检查命令：

```bash
scripts/mjlab.sh check --num-envs 4 --steps 32
mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_reference_training.py --out /tmp/chuanliantui-r7-dynamic-modes-20261010.json
```

均退出0；[结果JSON](20261010-mjlab-r7-dynamic-modes-check.json)记录完整manifest。

- 有限GPU32环境×128策略步×1280物理步：积分后qpos独立CPU运动学检查，最大闭链偏差2.659mm，通过5mm；数值异常0。零请求动作普通reset 75次，不作为策略起立成功。
- 原参考PPO和27奖励不变，奖励数值对照误差0。环境配置除机器人Entity和dynamic_modes标记外相同。
- 受控状态检查：地面reset进入recover；连续25步严格站稳切回normal；中断站稳计时清零；双轮恰好5N不算站稳；旧0.22m与当前0.30m命令不匹配时不能退出恢复；目标变动后按新目标重判。
- normal失稳须连续5步；中断不触发；新进入恢复计时从0开始，下一拍0.02s；低于0.16m可重入；无效／失败状态不算站稳。
- actor28:35与critic32:39共享模式；恢复目标等于当前actor高度命令槽；重复查询／重复同一transition不推进；部分reset不改变其他环境模式及动作／传感历史。
- 临时随机actor构造／checkpoint保存加载通过，无PPO学习；ONNX误差1.19e-07。r5/r6旧actor可加载，完整resume到r7被拒绝；其他资产被拒绝。
- r6最终model_999已按旧固定mode入口执行1秒无渲染回放兼容检查，退出0；该1秒不作为策略行为验收。
- git diff --check通过。

新动态mode只供后续r7训练／回放使用，已经完成的r6训练仍是固定mode；不能把有限检查等同于新策略训练效果或H7验收。
