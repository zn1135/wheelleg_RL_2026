# 稳站v2：160维历史单输入ONNX

后续NORMAL规则已追加[轮端承重判据](20261027-stand-v2-wheel-load-normal.md)，图形状不变，导出元数据记录checkpoint对应的`recovery_contract`。旧图与新图必须匹配各自的外部模式生成规则。

| 项目 | 内容 |
|---|---|
| 日期、作者、复查 | 2026-10-27，Codex；自检，未独立复查 |
| 任务、机构 | chuanliantui_stand_v2，既有华南虎闭链XML |
| 旧接口 | [32维当前帧＋160维历史双输入](20261027-stand-v2-isaac32-history.md) |
| 新导出标识 | stand-v2-history160-single-input-v1 |
| 训练仓及版本 | https://github.com/zn1135/wheelleg_RL_2026.git，26_wheelleg，952e4260aedf04e28fdbf366d5cb1c5562c75263＋本轮未提交改动 |
| 部署侧 | 同仓deployment/H7_RL与deployment/H7_RL_ros；本轮未修改，负责人核对待确认 |
| 权威实现 | mjlab_training/src/ct_mjlab/envs/chuanliantui_stand_v2/network.py的HistoryPolicy；同目录export.py |

## 接口差异

| 项目 | 旧 | 新 |
|---|---|---|
| ONNX输入 | observations[1,32]、observation_history[1,160] | 仅observation_history[1,160] |
| 当前帧来源 | 调用方单独传入 | 图内切片history[:,128:160] |
| 输出 | actions[1,6]、velocity[1,3] | 不变，velocity单位m/s |
| 类型、opset | float32、默认13 | 不变 |
| 参考向量NPZ | observations、history、actions、velocity | observation_history、actions、velocity；输入键与图一致 |

160维按5帧×32维展平，从最旧到最新。每10ms更新一次历史并推理，启动/复位时首帧填满五格。完整历史送入编码器得到3维缩放速度，最后32维与其拼接后送入原35维策略网络；对外速度输出除2恢复m/s。图本身不保存历史。

逐项观测索引、缩放、坐标、关节/动作顺序、默认角、PD、限幅、物理500Hz/策略100Hz、延迟、复位、接管和XML均沿用[原32维接口](20261027-stand-v2-isaac32-history.md)，本次只改变部署图的入参形式。训练侧仍有actor与history两个观测组，训练算法和checkpoint布局不变。HistoryPolicy也用于as_jit，因此其包装接口同步为单历史入参；本轮正式验证范围为ONNX。

## 兼容与同步

本版32维＋5帧编码器checkpoint可以直接重新导出，无需重训；旧39维checkpoint仍拒载。观测/profile标识保持，导出元数据另记export_contract区分图接口。旧双输入ONNX须配旧调用方式，新图须配单输入调用方式；不覆盖历史图或模型。

| 侧 | 待同步内容 | 状态 |
|---|---|---|
| 训练仓 | HistoryPolicy、导出数值检查/元数据/参考向量、任务说明 | 已修改并检查，未提交、未训练 |
| H7负责人 | 改为单历史输入；100Hz五帧FIFO与复位；新图转换及数值/时序核对 | 待确认；本轮未改固件 |
| CPU sim2sim | 接入本版32维观测、历史及编码器 | 仍未适配 |

## 验证依据

使用上一轮未训练接口checkpoint进行正式CLI导出，未生成训练模型交付：

```bash
scripts/mjlab.sh export \
  --checkpoint artifacts/stand-v2-isaac32-20261027/check-final/untrained-interface.pt \
  --output artifacts/stand-v2-history160-onnx-20261027/untrained-interface.onnx \
  --samples 256
```

本地退出码0；ONNX结构检查通过，图仅有[1,160]输入。256组各帧独立随机输入下，新包装与原checkpoint策略推理路径的动作/速度最大差均为0；ONNX相对PyTorch最大差分别1.35e-7以内、5.22e-8以内，阈值1e-4。图、checkpoint校验值、参考向量及检查结果见[导出报告](../../artifacts/stand-v2-history160-onnx-20261027/untrained-interface_export.json)，本轮源码快照见同目录snapshot。

本轮不运行训练、物理行为验收、板端工具转换或真机测试；此次数值等价不代表起立效果或板端兼容性通过。H7部署负责人核对状态为待确认。
