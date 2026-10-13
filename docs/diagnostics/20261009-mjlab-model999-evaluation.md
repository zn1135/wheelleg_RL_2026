# 串联腿 r4 model_999 首次效果评估

2026-10-09，用户要求查看训练效果，分支 `26_wheelleg`，基础提交 `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交r4修改。本次只回放／评估和记录，未修改训练、奖励、资产、控制或H7代码，未启动新训练。独立复查未执行。

## 模型与训练曲线

4096环境、1000次PPO更新已完成，iteration0～999，98,304,000条transition，用时1:20:30。模型 `logs/mjlab/chuanliantui/20261009_192207_944967_huananhu_r4_formal4096_1000/model_999.pt`，SHA256 `5c3c5cb91fb92c3a2620ef61e07717bd9e9b76dd32d78762bf6cf8672a0d50bb`；核对新接口r4和本次全部修改源文件SHA与run内来源快照一致，使用确定性actor输出。历史logs只读。

末轮平均奖励30.61、平均回合长983.86步、动作std0.16。训练日志的Mode/completed_recovered_fraction为1.92%，其含义是当前所记录已结束回合中曾严格站稳的批次统计，不是总体随机初态起立成功率；Mode/recover_fraction为69.12%。奖励上升并进入平台不等于达到站高目标。

训练／回放曲线图：`/tmp/chuanliantui-r4-model999-20261009/results.png`，独立MuJoCo的0／2／15秒实际机构画面：同目录 `poses.png`。

## 原生GPU可视回放

```bash
DISPLAY=:0 scripts/mjlab.sh play --num-envs 20 --random-reset --checkpoint logs/mjlab/chuanliantui/20261009_192207_944967_huananhu_r4_formal4096_1000/model_999.pt --render --seconds 120
```

打开20个实际随机初态环境的网格窗口，HUD显示当前模式、恢复目标／计时和七槽值。回放按原训练普通失败／timeout自动reset，因此不能凭反复回到初态的可视窗口宣称持续20秒成功；随机初态的首回合统计另见下节。Space暂停，[／]选环境；本次未调用learn()。终端日志 `/tmp/chuanliantui-r4-model999-play20-20261009.log`。

## 独立CPU MuJoCo 30秒验收

```bash
scripts/mjlab.sh eval --checkpoint logs/mjlab/chuanliantui/20261009_192207_944967_huananhu_r4_formal4096_1000/model_999.pt --seconds 30 --report /tmp/chuanliantui-r4-model999-eval-20261009.json
```

命令退出0，采用原0.15 m后摆初态，无观测噪声／域随机化，不自动reset掩盖失败。结果见 [CPU评估JSON](20261009-mjlab-model999-cpu-eval.json)。

| 指标 | 实测 | 验收条件 | 结果 |
|---|---|---|---|
| 持续时间／失败 | 30秒、无失败 | 30秒且无失败 | 满足 |
| 后半段最大高度误差 | 5.565 cm | ≤2 cm | 不满足 |
| 后半段最大pitch | 2.477° | ≤5° | 满足 |
| 后半段最大水平速度 | 0.0299 m/s | ≤0.05 m/s | 满足 |
| 全程最大漂移 | 3.237 cm | ≤20 cm | 满足 |

后半段平均根高度0.164785 m，目标0.22 m。整体passed=false，未完成站高验收。

## 128个随机初态的首回合评估

同一GPU资产和策略，seed20261009，128环境，各20秒，关闭观测噪声／域随机化，开启训练随机初态。为避免19.98秒训练timeout混入失败计数，评估将max_episode_length设为100000；普通失败／数值终止判据保持，所有指标只统计第一次回合，若失败便排除后续自动reset样本。reset前捕获终止帧。临时评估脚本副本 `/tmp/chuanliantui-r4-model999-20261009/batch-eval.py`，SHA256 `a090954c2aa5e7b8e736dc3d3e91b122405c4149455f14f884f81c79eb84587d`，结果见 [随机初态JSON](20261009-mjlab-model999-random-eval.json)。执行退出0。

| 初态 | 样本数 | 普通／数值失败 | 曾完成严格恢复 | 结束时连续0.5秒严格站稳 | 平均最终根高度 |
|---|---|---|---|---|---|
| 地面 | 63 | 0 | 1 | 0 | 0.164792 m |
| 普通 | 65 | 0 | 0 | 0 | 0.164982 m |
| 全部 | 128 | 0 | 1 | 0 | 0.164889 m |

全部128个样本持续到20秒，最终高度范围0.164316～0.165777 m。严格条件是高度0.22±0.03 m、roll/pitch≤15°、base接触≤1 N、两轮接触峰值>5 N连续0.5秒。1个地面样本曾短暂完成该条件，但最后同样回到低姿态；不能将“曾起立”解释为持续站稳成功。

## 结论与范围

当前模型学到低姿态平衡，尚未达到0.22 m目标站高；CPU与随机GPU评估均支持此结论。评估不含训练全套随机扰动、推扰、硬件误差或真机验证。可视回放和训练完成不意味着模型可以交付H7；本次未导出正式策略或适配部署。
