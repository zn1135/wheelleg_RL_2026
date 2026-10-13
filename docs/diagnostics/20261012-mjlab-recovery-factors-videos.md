# 六组2500轮同屏回放

2026-10-12用户要求查看本次单项对照视频。直接合成已经完成的原生GPU回放录像，不进行新的仿真、模型更新或训练；原始模型、轨迹、录像和冻结源码保持原内容。

## 视频

- [地面初态六组同屏](../../artifacts/recovery-factors-video-2500-20261012/six-groups-ground.mp4)
- [普通初态六组同屏](../../artifacts/recovery-factors-video-2500-20261012/six-groups-normal.mp4)
- [地面初态末帧](../../artifacts/recovery-factors-video-2500-20261012/ground-frame-200.png)

排列为两列三行，每段20.1秒、201帧、10FPS、1280×1176。均为累计2500轮模型，展示同一编号0样本；六组普通/地面初始qpos/qvel指纹分别完全一致。不能把样本0的结果等同整批64个环境。

| 左列 | 右列 |
|---|---|
| A 原height配置 | B 原recovery参照 |
| C 跳过扶正重启 | D 保留高度奖励 |
| E 保留扶正接触惩罚 | F 强化双轮承重奖励 |

原录像显示模拟时间、实际根节点高度和0.30m目标。合成画面的下方标注两轮对地向上支撑力和非轮部件对地支撑力，单位N，来自同一轨迹每0.1秒保存的原接触数据，与录像时间逐帧对应。帧0只标注共同初态，不编造尚未采样的力。

E组地面样本0最后5秒轮端支撑0/0 N、非轮支撑约131.4 N，根节点高度约0.159m；其他组该样本两轮主要承重，仍只有约0.20m高度。物理评价不限制腿关节姿态。

按后续[双轮平衡验收修正](20261012-mjlab-wheel-balance-acceptance-fix.md)，E组丢失原成功地面样本0、55，明确判为**退化，验收不通过**。56/64仅为逐样本统计，不代表组级合格；原成功样本失效不能由新增成功抵消。

## 生成与检查

使用独立mjlab Python3.12运行[合成脚本](../../scripts/agent/compose_chuanliantui_factor_videos.py)，读取每组`final500`已生成的MP4及`evaluation.json`。检查累计轮数2500、初态指纹一致、各输入201帧/10FPS/640×320、接触数据时间戳与视频帧一致；输出重新解码检查201帧/10FPS/1280×1176，合成退出码0。

源录像来自GPU轨迹qpos，经MuJoCo `mj_forward`渲染，无重新积分；本次只将原录像拼接并添加标签，保持每个原画面的尺寸与纵横比。

远端输出`jobs/20261012-recovery-factors-video-grid-v1`。本地证据包`artifacts/recovery-factors-video-2500-20261012.zip`，SHA256：

`1b0ca9e7063d33b25fd79613a6c6955a3d4e6358f3d9ba4dd222bd0bc135e0e6`

包含两段同屏视频、0/10/20秒截图、合成源码/日志/元数据、原输入录像哈希、源2000轮与六组101/500轮评估JSON、训练结束记录和单项对照汇总。模型未重新下载或转换。

下载后ZIP及32个载荷逐项校验，执行的合成源码与本地一致；地面20秒末帧已视觉检查，六组标签、力值和画面对应正确。

相关：[训练与结果记录](20261012-mjlab-recovery-factors-training.md)、[接口与分组](../interfaces/chuanliantui-mjlab-recovery-factors-v1.md)。
