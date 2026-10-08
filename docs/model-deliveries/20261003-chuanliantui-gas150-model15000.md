# chuanliantui：气弹簧续训 model_15000 验收

## 任务与结论

2026-10-03，用户完成续训后，由 Codex 执行回放，独立子 Agent 审计训练日志、
新旧轨迹与推力恢复。用户目标为起立及零速站立，不包含行走。
机器人为 chuanliantui，任务 `chuanliantui_standup`，接口保持
`chuanliantui-25d-100hz-r1`（25 维观测、125 维历史、3 维 latent、6 维动作）。

**训练完成，但零速原地站立未通过。** 高度和速度估计明显改善，旧模型的持续前移
变为新模型的持续后退；Isaac 的 pitch 偏差增大。共同推力测试均可恢复至原运动状态，
没有恢复静止站立。回放验收后已按用户要求导出 ONNX，见文末记录；
尚未执行板端部署或真机验证，也没有再次启动训练。

## 模型与训练来源

| 项目 | 记录 |
|---|---|
| 训练仓 | `https://github.com/zn1135/wheelleg_RL_2026.git`，分支 `26_wheelleg` |
| HEAD | `3e5b1d88c0825784b4b25a5a5d7f1f0dfc6d02c5`，含既有未提交修改 |
| 训练 run | `logs/chuanliantui_standup/Oct03_16-23-38_standup_h022_gas150_push/` |
| 最终完整 checkpoint | 上述 run 下 `model_15000.pt`，内部 `iter=15000`，含 encoder |
| 模型 SHA-256 | `f7dfd5d2d123ea31e11c7b321229b4da69d144eba7861c147782f932009e4da2` |
| 输入旧 checkpoint | `logs/chuanliantui_standup/Oct03_14-27-59_standup_h022_knee005_push/model_12000.pt` |
| 输入 SHA-256 | `dcc3e8d6b9ddbb2d777ed9e0af3f9936fac1508314c962967ef9d8438b134ebe` |
| 实际续训命令 | `python wheel_legged_gym/scripts/train.py --task=chuanliantui_standup --headless --resume --load_run Oct03_14-27-59_standup_h022_knee005_push --checkpoint 12000 --max_iterations 3000 --run_name standup_h022_gas150_push` |
| 训练完成证据 | TB step 12000–14999 共 3000 轮；最终文件已保存、训练进程结束；原训练终端退出码未单独保存 |
| 训练时间 | TB 首末记录 16:23:40–17:34:35，约 71 分钟（Asia/Shanghai） |
| seed / 资产 | 保存的 PPO 配置 seed=1，上述 CLI 未覆盖；训练 URDF 为 `resources/robots/chuanliantui_new_1/urdf/chuanliantui_train.urdf`，默认闭链为 `sim2sim/chuanliantui.xml`，二者无未提交修改，版本由上述 HEAD 定位 |
| 课程与优化器 | 最终课程已解锁；PPO/encoder 优化器状态完整，两者 step 从输入的 144000 增至 180000 |
| 环境 | Python 3.8.20、Isaac Gym Preview 4、PyTorch 2.1.0+cu118、CUDA 11.8、MuJoCo 3.2.2；RTX 4060 Laptop GPU，MuJoCo 用 CPU |
| H7_RL / PR / ONNX | 不涉及部署仓修改或 PR；ONNX 已导出并验证，见文末；无新固件产物 |

run 保存了 standup/base 的类和配置源码，未保存继承的 `ChuanliantuiCfg`、
`Chuanliantui` 或气弹簧 helper，也未记录气弹簧力标量。因此 150 N/侧及继承的延迟设置
与本会话和当前父类一致，但不能仅凭 run 自产物独立证明完整历史有效配置。
本次回放显式指定 150 N/侧与零延迟，参数见下文。
快照校验值及训练配置字面量已存入训练审计 JSON；未知的完整解析配置不伪造为已冻结。

### 训练曲线

| 指标 | 前 100 轮均值 | 末 100 轮均值 |
|---|---:|---:|
| recovered_rate | 39.09% | 95.06% |
| 总奖励 | 10.345 | 41.242 |
| 回合长度（策略步） | 885 | 1960 |
| 高度奖励 | 0.3771 | 0.9216 |
| 前向速度跟踪奖励 | 0.0590 | 0.1746 |
| encoder loss | 0.01145 | 0.00494 |

权重、优化器状态及 TB 标量无 NaN/Inf。末段总奖励已接近平台，姿态惩罚有所加重。
TB 的奖励项不是实际高度、pitch 或速度；`recovered_rate` 只统计回合曾满足
高度/倾角/接触和持续时间要求，不检查零速度，也不是逐次抗扰成功率。

## 30 秒无推力：相同条件的新旧对比

固定零前向/偏航命令、高度命令 0.22 m、气弹簧 150 N/侧、摩擦 0.5、零动作延迟、
确定性动作，无观测噪声及其他域随机化。起立初态保持原来的 0.15 m 后摆配置。
MuJoCo 使用 H7 控制且显式取消轮速目标裁剪；电机力矩限幅保留。
“加强约束”沿用前轮临时诊断 XML，不替换默认 XML，不代表真机刚度标定。

下表统计 `5 ≤ t < 30 s`。前向速度为机体系 10 ms 位置差分，正值前进、负值后退。

| 环境 | 权重 | 高度 m | pitch ° | 前向速度 m/s | 估速 RMSE m/s | 约 30 s 平面净位移 m |
|---|---|---:|---:|---:|---:|---:|
| Isaac | 旧 12000 | 0.24258 | −1.334 | +0.23909 | 0.11196 | 6.724 |
| Isaac | 新 15000 | 0.22402 | −2.782 | −0.13875 | 0.00885 | 3.804 |
| MuJoCo 默认闭链 | 旧 12000 | 0.23517 | −1.890 | +0.15513 | 0.06818 | 4.105 |
| MuJoCo 默认闭链 | 新 15000 | 0.21595 | −2.141 | −0.22096 | 0.02342 | 5.953 |
| MuJoCo 加强约束 | 旧 12000 | 0.24061 | −1.880 | +0.25067 | 0.10848 | 6.601 |
| MuJoCo 加强约束 | 新 15000 | 0.22210 | −2.083 | −0.13189 | 0.00783 | 3.438 |

新 Isaac 平均真速度 −0.13875 m/s、encoder −0.14612 m/s，偏差 −0.00736 m/s。
估速已明显接近真实运动，策略仍持续后退；数据不支持把剩余漂移全部归因于 encoder。
加强约束后的 MuJoCo 与 Isaac 的速度、高度较接近，但两边都未停止。
默认 MuJoCo 的漂移速率绝对值较旧模型增大，不能概括为所有条件下漂移均改善。

三组新回放均 3000 拍、有限数值、退出码 0。Isaac 无复位或失败；MuJoCo 在 5 s 后
无基座地面接触或翻转，H7 几何状态有效。数值正常及保持直立不等于零速验收通过。

MuJoCo 物理膝余量：默认闭链左右均值 0.04958/0.04545 rad，最小值
0.04887/0.04476 rad，部分不足目标 0.05 rad，但未越硬限位；加强约束组均值
0.07900/0.07529 rad，最小值 0.07815/0.07461 rad。Isaac 轨迹未存膝角，未推断其余量。

## 30 秒共同随机推力

沿用旧模型同一 seed 42 计划：6 次水平力，5–15 N、每次 0.1 s，5 s 后开始。
两端每 2 ms 向基座质心施加相同世界系力；MuJoCo 使用加强约束的诊断 XML。

| 指标 | Isaac 新模型 | MuJoCo 新模型 |
|---|---:|---:|
| 相对无推力状态恢复 | 6/6 | 6/6 |
| 推力结束至恢复确认 | 0.806–1.774 s | 0.800–0.920 s |
| 5–30 s 平均前向速度 | −0.13857 m/s | −0.13141 m/s |
| 最接近零的前向速度样本 | −0.03584 m/s | −0.03120 m/s |
| 零速站立窗口 | 0 | 0 |

恢复定义与旧模型一致：相对同模拟器无推力轨迹，速度差 <0.02 m/s、pitch 差 <1°、
高度差 <0.005 m，连续保持 0.5 s；表中时间包含该确认段。
两边均恢复到后退运动状态。Isaac 无失败复位/倒置样本；MuJoCo 无基座触地/倒置样本。
两端按 2 ms 提交力累计的总冲量与计划一致，未用 100 Hz 采样力积分核对。

这是同一世界系扰动；不同航向使身体轴向分量不同，不能直接用恢复时间排名抗扰能力。
Isaac 没有保存姿态 yaw 或基座接触力，轨迹切线的航向只是近似；100 Hz 数据可能漏掉
更快的峰值。此次无渲染回放仅作数值对照，不构成可视/真机行为验收。

## 证据与复现

新评估目录：`/tmp/ct-gas150-final-eval-20261003.37QPEa/`。

- `evaluation-manifest.json`：六条实际命令、环境、输出时间、日志哈希及退出码。
  自检及五组回放均退出 0；网络形状自检不替代行为结论。
- `isaac-idle30.json`、`mj-default-idle.json`、`mj-rigid-idle.json`、
  `comparison-summary.json` 与 `analysis_compare.py`：新旧无推力对照。
- `isaac-push30.json`、`mj-rigid-push.json`、`stage3-summary.json` 与
  `analyze_push_recovery.py`：推力逐事件记录；原始 `.log/.exit` 同目录。
- `source-sha256.json`：已核对上一轮记录中的所有未提交 `.py/.xml/.urdf`
  运行源码均未变化，故可复用旧模型对照，不把代码变化误当模型效果。
- 当前代码快照 `source.tar.gz`，SHA-256
  `9d2e07a8f67c91532eefaed0889eb7f892b0de8bf31f469fdc8a5c657a2b3770`；
  差异 `source.patch`，SHA-256 `d0e63550d7659b6c1fcf7c98584881da5c5d892d3be09e27b94d133d37c533e9`。
  快照含参与运行的未跟踪代码，不包含凭据和历史模型；本结论文档生成于快照之后。
- 共同计划 SHA-256 `66842b1692f39fd5bd87f91f70bcc7db82cd8696f56f38b89860f44660c5171c`；
  诊断 XML SHA-256 `a8c622048e2a6726a54098f49d90808c0ba75c38067bad7f82a0b55b201bf716`。
- 训练审计：`/tmp/ct-gas-training-audit-20261003.vjjsepg2/training-summary.json`。
- 旧模型证据：`/tmp/ct-standing-stages-20261003.m98sgj/`，
  [此前分阶段核查](../diagnostics/20261003-chuanliantui-gas-standing-validation.md)。

以上为本机证据路径；共享交付时需另存到团队可访问位置。历史 `logs/` 未改写。
ONNX 检查已在后续导出阶段完成；板端数值/时序、真机行为未执行；行走不在本次用户目标中。

## 后续重点

当前可确认高度跟踪与估速改善，零速原地站立仍需改进，且训练源环境也存在后退。
下一步优先检查零速奖励与静站验收定义，并兼顾 pitch 和膝余量。
当前前向两项奖励权重均为 0.2，`tracking_sigma=0.25`；仅改变速度这一变量，
0.14 m/s 漂移时两项加权和从零速的 0.200 降为 0.18336（乘 dt 前），
其余高度、姿态和接触奖励还会影响取舍。这是检查优先级依据，不是已证明的唯一根因。
现有 `stand_still` 函数惩罚关节偏离默认角，不能直接把它当零速或定点约束启用。
本次未修改奖励、成功判据或默认机构参数，也未启动下一轮续训。

## 后续 ONNX 导出（2026-10-03）

用户要求导出上述最终 15000 权重，沿用 H723 格式：float32、固定 batch=1、opset 13，
包含 encoder+actor，输出确定性动作与 latent。图不包含观测构造、历史缓存、动作限幅或 PD。

- 产物：[`model_15000_h723.onnx`](../../exports/chuanliantui_standup/gas150-model15000-h723-zRAFxG/model_15000_h723.onnx)，158017 字节。
- SHA-256：`47fe958efee3b9e55d25e91a0031abd8a822ac8daafc9c6122e346ac56eba988`。
- 输入：`observations [1,25]`、`observation_history [1,125]`。
- 输出：`actions [1,6]`、`latent [1,3]`；全部 float32。
- 算子：7 个 Gemm、5 个 Elu、1 个 Concat；ONNX 1.14.1、PyTorch 2.1.0+cu118。
- 导出命令在仓库根、已激活 Python 3.8 环境执行，CPU 运行：

```bash
python -c 'import isaacgym; import runpy; runpy.run_module("scripts.agent.export_chuanliantui_onnx", run_name="__main__")' \
  --checkpoint logs/chuanliantui_standup/Oct03_16-23-38_standup_h022_gas150_push/model_15000.pt \
  --output exports/chuanliantui_standup/gas150-model15000-h723-zRAFxG/model_15000_h723.onnx \
  --fixed-batch --opset 13
```

导出脚本 SHA-256 为 `a959d7f674c65743d17e54546f2719c5a9c855489cb1ef623a2dfd061372f543`，
源码未因本次导出修改。产物位于新目录，未覆盖已有模型或改写历史 `logs/`。

ONNX checker（含 full_check）通过；导出器的一组固定随机输入比对通过；另从本轮
`mj-rigid-push.json` 选择 187 组实际观测/历史，覆盖起立、站立和六次推力前后，
以 ONNX ReferenceEvaluator 对照 PyTorch，均满足 `rtol=1e-5, atol=1e-6`。
最大绝对差为 actions `1.1921e-6`、latent `3.2783e-7`；容差是绝对项与相对项之和。
导出与补充校验退出码均为 0。校验明细、采样索引、源轨迹/模型哈希和日志位于
产物同目录的 `validation.json`、`validation.log`、`export.log` 及 `.exit` 文件。

本次确认格式转换数值一致；Cube.AI Analyze、生成 C 网络和目标板运行未执行。
ONNX 对应同一个存在后退漂移的 15000 策略，上述站立验收结论保持不变。
