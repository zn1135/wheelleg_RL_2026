# 模型交付：复旦PPO拟合物理12000→14000（2026-11-10）

**训练恢复和物理身份审计通过；完整GPU11/987未达标，相对11000仍丢失4个原成功，不是合格替换。** 仅训练侧PT/GPU证据交付，无ONNX/H7/真机交付。

## 来源、接口与训练

| 项目 | 实际记录 |
|---|---|
| 机器人/任务 | chuanliantui_new_1，fudan_ppo_walk |
| Git/职责 | 分支26_wheelleg，原工作区改动保留；未切分支/提交/推送/PR；H7未改/未烧录，负责人独立核对待确认 |
| 来源 | 起始HEAD见[head.txt](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/provenance/head.txt)，脏工作区实际[配置/差异/资产/依赖指纹](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/provenance/)及训练日志source/、source_sha256.json共同记录版本，不能只看HEAD |
| 交付/复查 | AI代训练负责人执行；独立人员复查未执行，本次为AI正式检查与GPU技术评分 |
| 接口/控制 | 25维当前帧、125维history v2、6动作；原网络/16奖励/PD60/2保持；物理/PD2ms，策略10ms，训练延迟2+2ms、GPU评估4+4ms |
| 续训源 | logs/mjlab/fudan_ppo_walk/20261110_fitted_resume11000_to12000_01/model_final.pt |
| 源SHA256 | 39354a2344f9effd170064c0b9875c1ae7b197470a2682074640debcdcc84119 |
| 恢复范围 | 权重/两Adam/RNG逐值相同，恢复学习率和课程；物理/回合/历史和回合摩擦记忆重新初始化，不是恢复旧物理现场 |
| 预算/完成 | 恰新增2000轮，12000→14000，按iteration_budget停止、退出0，无追加 |
| 实际时间 | 2026-11-10T12:42:22.238126+08:00～2026-11-10T17:20:31.205116+08:00，Asia/Shanghai，训练循环16649.65秒 |
| 并行/采样 | seed1，4096×48，每批49152，4批，两Adam各5epoch×4批 |
| 计数 | 新增393216000转移/两Adam各40000更新；累计2236416000转移/各280000更新 |
| 日志/命令 | logs/mjlab/fudan_ppo_walk/20261110_fitted_resume12000_to14000_01；[实际命令](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/train-command.txt)、[预算](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/budget.json) |
| 工具/资产 | 仓库venv、RTX4060 Laptop8GiB；[版本](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/provenance/runtime.json)及安装物理库/机器人资产指纹已核对未变 |

[恢复审计](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/resume-audit.json)、[训练/物理身份/源码/日志审计](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/training-audit.json)通过；142份训练Python快照及16项/总量12001～14000日志完整。145项来源和安装物理库/资产未改，复用有效39项GPU实际施力/名义值/镜像索引/单次施力/全体与局部记忆复位及同配置预检，没有伪称重跑：[范围与证据](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/preflight/reused-evidence.json)。

## 冻结交付与物理身份

- [最终14000完整PT](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/frozen/model_14000.pt)，SHA256 5320e927d2ef2387a2d9ef1678102abc79942cde1b1a616828de9b2f133dd6ce；[身份文件](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/frozen/identity.json)。模型含网络和两优化器，评估前冻结，评估不更新权重。
- profile mirrored-local-load-dahl-20261109-v1；[完整JSON](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/frozen/fitted_leg_load.json)，SHA256 e6ef82fa4380fa01341fa2166be56282c5290c82fec2b2a04e5b3107ce49218f；原150N/50Ns/m基础项和新qfrc负载/记忆同时保留，保守基线/矩阵/参考角/修正系数读取JSON。
- XML仅含机构/限位及原弹簧，完整物理还需要JSON和共用代码；训练快照、初始/最终检查点与GPU协议身份一致。左腿为右腿镜像，非独立标定，软件接入成功不代表sim2real消除。
- 没有导出ONNX、生成板端网络或烧录固件；既有模型、日志、默认选用策略均未覆盖。

## 分阶段验收

20例GPU诊断完整有限、技术5/20；之后完整987/987有限，无数值异常/提前终止/缺测，名义11/47、随机0/940，所有类别整体均未达到名义100%及随机至少95%要求。统一标准速度0.10m/s、偏航0.10rad/s、停车0.05m/s、俯仰5°及原轮载/接触/高度/末段平衡条件未改。GPU/视频上限依原预算600/2400/1200秒，没有临时调参求通过。

- 对11000的11/987：精确同初态，原11保留7、失去4、新增4；总数持平但按原成功任一丢失规则拒绝退化。
- 对12000的7/987：精确同初态，原7保留5、失去2、新增6；总数提高仍不能称无退化。
- 主问题：970例首段轮载失败、871例末段roll/pitch角速度>0.5rad/s；部分零速/停车漂移超限。
- [正式评分](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/full-acceptance/acceptance.json)、[双基线逐样本对照](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/paired-comparisons.json)、[1239行逐段指标](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/motion-metrics.csv)、[完整报告](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/README.md)。
- [47名义工况419秒正常视频](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/video-gpu/nominal_matrix.mp4)、[随机失败7秒](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/video-gpu/random_failure.mp4)、[视频章节/帧数/哈希检查](../../artifacts/fudan_ppo_walk/20261110-fitted-resume12000-to14000-01/video-verification.json)，完整起步/刹停/变高/通过与失败均保留，随机无成功样本。
- 最终用户视频确认待完成；倒地起立、Isaac、ONNX、板端数值/时序及真机未验，不作为这些阶段通过，H7独立核对待确认。

建议保留11000作原成功集合对照，暂不据总通过数增加采用14000。下一步固定权重/初态，保持500Hz PD/100Hz策略，对照0.2/2ms积分，看轮载有效窗和末段角速度是否改善；本次不自动加训或执行新诊断。左腿独立辨识、跨姿态、独立实录、初始记忆/重复性、隐状态及全行程sim2real量化仍未完成。

TensorBoard（未启动服务）：

```bash
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/fudan_ppo_walk/20261110_fitted_resume12000_to14000_01 --port 6006
```
