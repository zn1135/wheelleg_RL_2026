# 模型交付：复旦PPO拟合物理11000→12000（2026-11-10）

**恰新增1000轮，训练正常停止；GPU完整验收7/987，相对11000模型11/987退化，不作为替换依据。** 本次仅训练侧PT与GPU证据，不包含ONNX、H7或真机交付。

## 来源与接口

- 任务`fudan_ppo_walk`，机器人chuanliantui_new_1，当前分支26_wheelleg。起始HEAD见[provenance/head.txt](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/provenance/head.txt)，工作区已有改动；实际来源由[配置/差异/资产/安装依赖指纹](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/provenance/)、训练日志source/及source_sha256.json共同记录，不能仅凭HEAD复现。
- 交付由AI代训练负责人执行，独立复查未执行；H7未改/未烧录，H7负责人核对待确认，无提交/推送/PR或分支切换。
- 当前帧25维/历史125维/六动作，history v2；原16项奖励、网络、PD60/2、物理/PD2ms和策略10ms保持。训练动作/传感2+2ms，默认GPU评估4+4ms，按原配置保留。
- 物理`mirrored-local-load-dahl-20261109-v1`；完整参数JSON SHA `e6ef82fa4380fa01341fa2166be56282c5290c82fec2b2a04e5b3107ce49218f`，左腿仍为右腿镜像，非独立标定；XML之外还需要JSON及共用负载/记忆代码。

## 训练及文件身份

| 项目 | 实际记录 |
|---|---|
| 续训源 | `logs/mjlab/fudan_ppo_walk/20261110_fitted_resume9000_to11000_01/model_final.pt` |
| 源SHA256 | `6595b54d69890da0fc28d11b69613339af297d3f1c89040902fb700c833e11e3` |
| 初始恢复 | 权重、两Adam和CPU/CUDA RNG逐值一致，恢复学习率/课程；环境、回合/历史及摩擦记忆重新初始化，不是恢复旧物理现场 |
| 预算 | 11000→12000，恰新增1000轮，到预算即停，无追加；进程退出0 |
| 并行/采样 | seed1，4096×48，四minibatch，每批49152；两Adam各5epoch×4批 |
| 转移/优化 | 新增196608000，累计1843200000转移；两Adam各新增20000、累计240000 |
| 实际时间 | 2026-11-10T08:38:05.408377+08:00～2026-11-10T10:04:31.333884+08:00，训练循环5148.92秒 |
| 实际命令/日志 | [训练命令](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/train-command.txt)；`logs/mjlab/fudan_ppo_walk/20261110_fitted_resume11000_to12000_01` |
| 最终冻结PT | [model_12000.pt](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/frozen/model_12000.pt)，SHA256 `39354a2344f9effd170064c0b9875c1ae7b197470a2682074640debcdcc84119` |
| 完整物理参数 | [fitted_leg_load.json](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/frozen/fitted_leg_load.json)；训练快照、检查点及GPU协议均同SHA |
| 运行工具/GPU | 仓库venv，RTX4060 Laptop 8GiB；Python/PyTorch/MuJoCo/Warp版本见[runtime.json](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/provenance/runtime.json)，安装库源码指纹已核对未变 |

[恢复完整审计](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/resume-audit.json)、[训练/参数/源码/奖励日志审计](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/training-audit.json)通过；142份训练Python匹配，16项及总量11001～12000日志完整。源/物理/资产145项身份与上轮相同，复用有效的名义参数、39项GPU实际施力/复位、v2可视预检和4096容量证据，未伪称重新执行；[证据复用范围](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/preflight/reused-evidence.json)。

## GPU验收与回退判断

- 20例完整有限，技术4/20；随后正式987/987完整有限，无数值异常、无提前终止、无缺测，技术7/987（名义7/47、随机0/940），各类别整体均未达标。
- [统一验收](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/full-acceptance/acceptance.json)复用唯一标准0.10m/s、0.10rad/s、停车0.05m/s、俯仰5°及原v2轮载等条件，未调参降低标准。
- 与11000按解码初态数组cohort精确匹配，原11成功保留4、丢失7、另新增3；7个退化均因轮载有效窗不足，按规则拒绝退化，新增成功不抵消。逐样本记录见[paired-comparison.json](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/paired-comparison.json)。11000源模型及其报告/日志原位保留，未自动替换默认选用模型。
- [47名义工况419秒正常视频](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/video-gpu/nominal_matrix.mp4)、[随机失败7秒视频](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/video-gpu/random_failure.mp4)，完整包含起步/刹停/变高与通过/失败；[视频帧数/章节/哈希验证](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/video-verification.json)。最终用户观看确认待完成，不能把技术结果写成最终用户通过。
- 全部速度误差/俯仰/轮载/机身接触与失败原因：[1239行CSV](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/motion-metrics.csv)、[完整报告](../../artifacts/fudan_ppo_walk/20261110-fitted-resume11000-to12000-01/README.md)。倒地起立/Isaac/ONNX/板端/真机均未执行，不作为这些阶段通过。

下一步优先固定11000权重和初态，保持500Hz PD/100Hz策略，对照0.2/2ms积分，检查轮载有效窗与末段角速度是否改善；不自动加训。左腿独立辨识、跨姿态、独立实录、初始记忆/重复性、隐状态及sim2real量化仍未完成。

TensorBoard（未启动服务）：

```bash
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/fudan_ppo_walk/20261110_fitted_resume11000_to12000_01 --port 6006
```
