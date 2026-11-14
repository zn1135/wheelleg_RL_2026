# artifacts 目录说明

本目录保存实验产生的模型副本、原始数据、分析报告、回放视频、配置快照和检查结果。查找时先选实验类别，再进入对应日期和任务的子目录，优先阅读其中的 `README.md`、`RESULTS.md` 或 `REPORT.md`。

以下说明按2026年11月15日实际目录整理，覆盖顶层全部文件夹和文件，并单列当前实机回放任务中的文件用途。最新模型选择和实验结论统一查看[项目实验记录](../docs/ai/project-log.md)。

## 顶层每一项的用途

| 文件或文件夹 | 作用 | 什么时候查看 |
| --- | --- | --- |
| [fudan_ppo_walk/](fudan_ppo_walk/) | 复旦PPO行走线的训练协调记录、模型副本、GPU评估、控制参数检查和视频。`models/`保存按版本命名的PT副本；日期子目录保存各次实验。 | 查当前训练线、不同轮数的表现、PD与延迟改动。 |
| [real2sim/](real2sim/) | 真机采集、关节与轮端响应分析、标定核查、实录回放及仿真实机对照。每个子目录对应一次诊断任务。 | 查部署后为何与仿真不同，或找实机日志、曲线、回放。 |
| [pace-joint-capture/](pace-joint-capture/README.md) | PACE专用关节测试的准备、定位、腿长变化、USB采集及相关检查。 | 找之前单腿测试、右腿长度和角度的实测记录。 |
| [length-response-fit/](length-response-fit/README.md) | 基于右腿改变长度的实录，拟合弹簧、阻力、摩擦等响应参数；包含候选参数、扫描和验证结果。 | 回顾关节辨识方法、候选为什么没有通过、哪些轨迹拟合较好。 |
| [original-training-vs-fit-opj8_kz7/](original-training-vs-fit-opj8_kz7/README.md) | 原训练名义物理模型与腿长拟合候选，在相同冻结输入下的专项对照。名称末尾是该次输出目录的随机后缀。 | 比较“原物理参数”和“拟合参数”的响应差异。 |
| [shared_validation/](shared_validation/) | 共享物理和验收工具的检查证据，包括闭链约束、关节惯量翻倍、PD200保持节奏及统一GPU验收。 | 核对公共实现改动是否按预期生效，以及验证覆盖了什么。 |
| [model-versions/](model-versions/) | 按SHA256内容指纹归档的MuJoCo物理模型XML及网格资源。这里的model主要指机器人仿真资产。 | 旧实录回放要求加载当时的物理模型，或核对新旧XML来源。 |
| [mujoco-wheel-response/](mujoco-wheel-response/) | 轮端响应代码的编译缓存，目前有按源码指纹和MuJoCo版本命名的目录，内含 `drive_gain.so`。 | 排查轮端响应插件的编译、加载或缓存问题。 |
| [video-diagnosis/](video-diagnosis/) | 从实机视频截取的关键帧和诊断图片，目前按日期存放。 | 对照松手、停车、接触等关键时刻的可见动作。 |
| [jetson_rl/](jetson_rl/) | Jetson负责策略推理、H7负责执行的迁移验证和部署证据。 | 核对Jetson模型数值、通信和软件交付情况。 |
| [supermimic/](supermimic/) | SuperMimic训练线的世界模型、策略、预测误差、稳定性、历史输入及消融实验产物。 | 查SuperMimic相关实验；不要与复旦PPO模型混用。 |
| [imitation/](imitation/) | 模仿学习相关检查，现有行为克隆随机化冻结与运动数据归一化整理记录。行为克隆即让策略学习已有示范动作。 | 查BC或运动示范数据的配置与整理结果。 |
| [legacy_mjlab/](legacy_mjlab/) | 较早mjlab训练线的模型、导出和评估记录，包括multimode、轮式停车及旧walk等实验。 | 历史复现和回顾旧方案，不据目录里有模型就当成当前选用版。 |
| [isaac_legacy/](isaac_legacy/) | Isaac旧任务的源码快照与静态配置核对。 | 查旧Isaac任务的接口、配置和历史来源。 |
| [two_policy/](two_policy/) | 历史“独立起立策略＋行走策略”的实现与检查记录。 | 回顾双策略设计、切换方式和当时的验证边界。 |
| [maintenance/](maintenance/) | 提交草案、合并与同步核对、代码复查、目录整理和工作区迁移等维护产物。 | 查某次维护操作的范围、差异和检查证据。 |
| [README.md](README.md) | 本目录导航与文件用途说明。 | 不知道文件放在哪里、文件名代表什么时先看这里。 |

## 实验子目录里常见文件的作用

同名文件在不同实验中的字段可能不同，下表是阅读顺序与常见用途；具体定义以该实验说明和实际内容为准。

| 文件或名称 | 作用 |
| --- | --- |
| `README.md` | 实验入口，通常说明目的、输入、执行命令、结果及限制。 |
| `RESULTS.md`、`REPORT.md`、`COMPLETED.md` | 面向阅读的结果、分析或完成记录。判断实验结论先看这些。 |
| `summary.json`、`report.json`、`result.json` | 程序生成的数值、分项结果和检查统计，适合查精确指标。 |
| `plan.json`、`budget.json` | 运行前的条件、模型来源、实验矩阵或授权预算。 |
| `status.json` | 某次运行写下的状态；旧文件不会自动反映当前进程状态。 |
| `training-audit.json`、`runtime-audit.json`、`*-check.json` | 核对实际轮数、参数、源码或执行步骤。检查通过不自动等于行为通过。 |
| `*.pt` | PyTorch模型检查点，可能包含网络权重、优化器和训练计数；能否续训取决于实际内容与接口。 |
| `*.onnx` | 导出的推理网络，用于部署或离线推理；通常不包含训练优化器状态。 |
| `*.npz`、`*.npy` | NumPy数值数组，例如关节角、速度、力矩、初态或模型输入输出。不能仅凭文件名认定是实测或仿真数据。 |
| `*.csv` | 可按表格查看的逐帧信号或汇总数据，读取时核对列名与单位。 |
| `*.bin` | 可能是采集原始字节，也可能是固件文件；必须按所在实验和格式说明区分。 |
| `*.xml`、`*.mjb` | MuJoCo场景或机器人模型；XML是文本定义，MJB是编译后的模型。 |
| `*.png`、`*.jpg`、`*.pdf` | 曲线、截图、图表或报告渲染，用来直观看结果。 |
| `*.mp4` | 已录制或渲染的视频；阅读说明确认是实机拍摄、实测姿态重建还是仿真回放。 |
| `*.log` | 运行时控制台输出，包含进度、错误或退出信息。 |
| `*.pid` | 启动时记录的进程编号，不能单独证明进程现在还活着。 |
| `*-sha256.json`、`SHA256SUMS`、`*-manifest.json` | 来源和文件指纹清单，用于核对内容、模型及配置是否一致。 |
| `source/`、`before/`、`after/`、`*-source.py` | 实验当时的源码快照，便于追溯；日常改代码应定位仓库中的正式实现。 |
| `fixtures/` | 冻结的初态、模型和物理参数，用于让两次实验在相同条件下比较。 |
| `traces/`、`recording/`、`raw/` | 轨迹、录制或原始输入；原始数据是复核依据。 |
| `acceptance/` | 验收分项、通过和失败原因；注意是诊断子集还是完整验收。 |
| `qa*/` | 文档或图像生成过程中的排版核对文件，通常不是最终交付件。 |

## 当前这次实机回放任务的文件说明

目录：[real2sim/model3000-pd200-logs-20261115-01/](real2sim/model3000-pd200-logs-20261115-01/)。

### 回放相关

| 文件或文件夹 | 具体作用 |
| --- | --- |
| `README.md` | 这次5段实录的回放方式、命令和显示假设。 |
| `display-model.mjb` | 播放器显示机器人时使用的MuJoCo模型。 |
| `part-01.npz` | 第1段日志重建出的姿态、显示时间、有效标志和对应遥测。 |
| `part-02.npz` | 第2段的重建姿态及遥测。 |
| `part-03.npz` | 第3段的重建姿态及遥测。 |
| `part-04.npz` | 第4段的重建姿态及遥测。 |
| `part-05.npz` | 第5段的重建姿态及遥测。 |
| `report.json` | 5段来源文件、SHA256、帧数、时长、无效帧和几何闭合误差统计。 |
| `player.jpg` | 验证播放器显示效果时保存的截图。 |
| `viewer.log` | 播放器启动、连接和运行输出。 |
| `viewer.pid` | 启动播放器时保存的进程号。 |

`part-*.npz`来自实测数据的姿态重建：显示时机身XY固定、高度由几何推算，不是测得的完整空间轨迹。原始 `.bin` 日志保留在[当前导出包的rl_logs目录](../exports/fudan_ppo_walk/20261114_model3000_pd200_delay10to20_v3/rl_logs/)。

### gap分析相关

| 文件或文件夹 | 具体作用 |
| --- | --- |
| `gap-analysis-01/` | 第一轮分析的中间产物；参考哈希检查时退出，没有完整最终报告。 |
| `gap-analysis-02/` | 修正参考模型来源检查后生成的分析结果。 |
| `gap-analysis-03/` | 加入频率对照及后续补充的中间版本。 |
| [gap-analysis-final/](real2sim/model3000-pd200-logs-20261115-01/gap-analysis-final/) | 本次最终分析入口，优先阅读这一版。 |
| `gap-analysis-final/REPORT.md` | 完整文字结论、证据、限制和复算命令。 |
| `gap-analysis-final/summary.json` | 逐文件动作核对、PD公式、时间、停车窗口及GPU对照的详细数值。 |
| `gap-analysis-final/overview.png` | 5段日志的指令、机身倾斜和轮速总览。 |
| `gap-analysis-final/zero-command-tracking.png` | 选定零指令窗口中的目标角、实际角和轮速对照。 |
| `gap-analysis-final/frequency-gap.png` | 实录与GPU各信号的频谱，用来观察1Hz与5～6Hz等成分。 |
| `gap-analysis-final/records-*-policy.npz` | 每段离线重建的模型动作、记录动作和网络估速；共5份，与原始日志文件名对应。 |
| `gap-analysis-final/analysis-source.py` | 最终分析时使用的脚本快照。 |
| `gap-analysis-final/analysis-source-sha256.json` | 分析脚本及依赖文件的指纹。 |

### Word复盘相关

Word正式文档统一放在 [docs/word/](../docs/word/README.md)。本实验的生成脚本与排版检查仍在 [summary-word-20261115/](real2sim/model3000-pd200-logs-20261115-01/summary-word-20261115/)。

| 文件或文件夹 | 具体作用 |
| --- | --- |
| [20261115-轮腿机器人实机最佳版本问题总结.docx](../docs/word/20261115-轮腿机器人实机最佳版本问题总结.docx) | 当前最终Word，含排查历程、PD与频率、版本对照、剩余问题、基线和删除记录，已移至统一目录。 |
| `build_summary.py` | 最早摘要稿的生成脚本，同时提供后续脚本复用的字体和表格设置。直接运行会生成旧内容。 |
| `build_version_review.py` | 版本对照稿的生成脚本；直接运行会覆盖同名Word为该旧稿。 |
| `build_problem_journey.py` | 当前完整复盘的生成脚本，包含最后新增的基线与清理记录。 |
| `qa/`、`qa-final/` | 最早摘要稿的分页与排版检查结果。 |
| `qa-version-review/`、`qa-version-final/` | 版本对照稿的排版检查结果。 |
| `qa-journey/` | 按完整排查过程重写后的排版检查结果。 |
| `qa-baseline-cleanup/` | 当前加入基线清理记录后的5页渲染；用于核对最终排版。 |

各 `qa*` 目录内的 `page-N.png` 是第N页截图，PDF是对应Word的渲染副本。平时阅读和编辑最终 `.docx` 即可。

## 与仓库其他目录的区别

- [logs/](../logs/)：正式训练运行的检查点和训练日志。
- [exports/](../exports/)：供部署使用的导出包、ONNX与接口资料。
- [docs/](../docs/)：项目文档、接口约定和长期实验结论。
- [scripts/agent/](../scripts/agent/)：生成检查、分析和评估结果的正式脚本。
- [resources/](../resources/)：训练和仿真使用的机器人资产与物理参数。

`artifacts` 中同时有最终证据和中间结果。具体文件是否还被回放、报告或历史核查引用，需要查看其实验说明；不要把整个目录当成可一键清空的缓存。
