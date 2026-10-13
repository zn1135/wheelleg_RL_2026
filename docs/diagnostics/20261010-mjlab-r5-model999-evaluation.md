# r5 model_999 回放评估

2026-10-10，26_wheelleg，HEAD 58947e8fabc4683b2025d42bda742b06c84dc566，加未提交 r5 改动。

模型：`logs/mjlab/chuanliantui/20261009_213808_247780_huananhu_r5_closurefix4096_1000/model_999.pt`。
SHA256：`3f7cacf828d1acb0b9037fbba93774218857ba547a9cb67dc5029ba9af094d16`。
4096 环境、1000 次 PPO 更新已完成。此模型属于 r5，不是后来移植的 r6。

执行 `scripts/mjlab.sh eval --checkpoint <上述模型> --seconds 30 --report /tmp/chuanliantui-r5-model999-eval-20261010.json`，退出码 0。
[原始 CPU 轨迹](20261010-mjlab-r5-model999-cpu-eval.json)：30 s 无失败，后半段平均高度 0.168224 m，最大高度误差 0.055328 m、最大俯仰 0.664°、最大平面速度 0.006534 m/s、最大漂移 0.027971 m。未通过 0.22 m 静站高度验收。
20 环境随机初态原生回放已打开；用户截图显示低位蹲姿。CPU 起立轨迹全程处于 recover。

实际渲染姿态图：`/tmp/chuanliantui-r5-model999-20261010/poses.png`；指标图：同目录 `metrics.png`。
本次没有 PPO 更新。修改训练奖励不会改变此模型已有权重。
