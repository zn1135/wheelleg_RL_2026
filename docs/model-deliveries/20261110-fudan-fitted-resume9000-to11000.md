# 模型交付：复旦PPO双腿拟合物理9000→11000（2026-11-10）

仅交付训练侧PT、GPU诊断与验收证据；**参数软件接入通过，完整运动标准未通过（11/987）**。不作为真机通过或H7可部署结论。

## 任务与接口

| 项目 | 实际记录 |
|---|---|
| 维护范围 | Wheel-Legged-Gym，chuanliantui_new_1，注册任务fudan_ppo_walk |
| 交付人/接收人 | 本次AI代用户执行，接收人为训练负责人；独立复查未执行 |
| 接口 | 当前帧25维、历史125维、动作6维；history v2，原16项奖励、网络及PD保持 |
| 时序 | 物理/PD2ms，策略10ms；训练动作/传感2+2ms，默认GPU评估4+4ms |
| 物理 | mirrored-local-load-dahl-20261109-v1；左腿是右腿镜像，非独立标定 |
| Git/部署 | 当前分支26_wheelleg，未切分支、未提交/推送、无PR；H7未改/未烧录，负责人核对待确认 |

## 代码与训练来源

起始HEAD `c3c8fdffe0739aed314fb71c69f2a2d3135b3fb9`，已有大量工作区改动，不能只以HEAD复现。[起始状态/实际配置/训练侧差异/资产及依赖指纹](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/provenance/)与训练日志`source/`、`source_sha256.json`共同记录实际来源；142份训练Python快照逐一匹配。只保存相关来源，未收入个人配置或凭据。

| 训练信息 | 实际记录 |
|---|---|
| 续训源 | `logs/mjlab/fudan_ppo_walk/20261104_resume8000_to9000_2ms/model_final.pt` |
| 源SHA256 | `4ff0c038f26b7aa9d9d9919c97e59cd4e8124f1ddee5eaec1d2944436f8a37a3` |
| 恢复范围 | 权重、两Adam、学习率、速度课程和CPU/CUDA RNG逐值一致；物理/回合/历史重新初始化，摩擦回合初力0 |
| 轮数 | 9000→11000，恰新增2000，到预算停止；退出0 |
| 并行/采样 | 4096环境×48步，4个minibatch，每批49152；PPO/encoder各5epoch×4批 |
| 转移/优化 | 新增393216000转移，累计1646592000；两Adam各新增40000，累计220000 |
| 时间 | 2026-11-10 01:41:57～04:30:49，Asia/Shanghai；训练循环10096.64秒 |
| seed/来源 | seed1创建环境，学习器RNG恢复9000源；[实际命令](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/train-command.txt) |
| 日志 | `logs/mjlab/fudan_ppo_walk/20261110_fitted_resume9000_to11000_01` |
| 运行环境 | Python3.12.3、PyTorch2.14.0、MuJoCo3.11.0、MuJoCo Warp3.11.0、Warp1.17.0；RTX4060 Laptop 8GiB |

安装库的祖先仓HEAD不代表其安装包来源；以[发行版版本及安装源码指纹](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/provenance/runtime.json)和同目录installed-physics-source-sha256.json为准。

## 模型与物理文件

| 产物 | 文件及SHA256 | 范围 |
|---|---|---|
| 最终完整PT | [冻结11000模型](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/frozen/model_11000.pt)，`6595b54d69890da0fc28d11b69613339af297d3f1c89040902fb700c833e11e3` | 全网络与两优化器状态，新增2000轮后冻结 |
| 参数JSON | [完整参数包](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/frozen/fitted_leg_load.json)，`e6ef82fa4380fa01341fa2166be56282c5290c82fec2b2a04e5b3107ce49218f` | 共用负载基线、参考角、修正项、摩擦及记忆参数 |
| XML/源码 | 日志source/ct_mjlab及provenance/chuanliantui.xml；逐项SHA见日志source_sha256.json和provenance/assets-sha256.json | XML只含机构/限位/原弹簧；新负载还需要JSON和共用代码，不能只交XML |
| ONNX/固件 | 未导出/未生成/未烧录 | 本次未要求，不构成板端交付 |

## 验证状态

| 阶段 | 结论与证据 |
|---|---|
| 参数实际接入 | [正式施力及39项GPU检查](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/preflight/)、[启动恢复/物理身份](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/startup-restore-identity.json)通过；每2ms施力一次，全体/局部记忆复位、其他环境状态、前后索引/镜像和原150N基础项均核对 |
| 预算/学习状态 | [续训完整审计](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/resume-audit.json)、[参数/源码/日志审计](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/training-audit.json)通过；16项及总量9001～11000日志完整 |
| 20例GPU诊断 | 完整/有限20/20、技术3/20；无数值异常后才进入完整矩阵 |
| 完整GPU运动 | 987/987完整有限、0数值异常/提前终止，11/987通过；名义11/47，随机0/940，各类别整体均不通过，无缺测；[统一报告](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/full-acceptance/acceptance.json) |
| 正常速度视频 | [47名义工况419秒](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/video-gpu/nominal_matrix.mp4)、[随机失败7秒](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/video-gpu/random_failure.mp4)，覆盖完整起步/刹停/变高/通过与失败；[帧数/章节/哈希校验](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/video-verification.json) |
| 最终用户视频确认 | 待确认；技术未通过不因观看而变为通过 |
| 倒地起立/Isaac/ONNX/板端/真机 | 未执行；本次任务只测GPU平地行走矩阵，不作为这些能力通过 |

主要失败为轮载有效窗不足和末段roll/pitch角速度；部分停车速度/运动跟踪也超限。完整逐段速度误差、俯仰、轮载、机身接触及原因见[1239行指标CSV](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/motion-metrics.csv)和[实验报告](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/README.md)。旧8000新物理0/20只是历史对照，不替代本次统一标准。先前从零2000轮试验独立保留，不混写为本次续训。

未完成项：保持500Hz PD/100Hz策略的0.2/2ms积分敏感性、左腿独立辨识、跨姿态负载、独立实录、初始记忆/重复性、隐状态及全行程sim2real量化。优先固定11000模型做积分对照，比较垂直速度RMS、轮载有效窗及末段角速度；本次不自动追加。H7独立核对、板端时序/数值及真机均待完成。

TensorBoard（未启动服务）：

```bash
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/fudan_ppo_walk/20261110_fitted_resume9000_to11000_01 --port 6006
```
