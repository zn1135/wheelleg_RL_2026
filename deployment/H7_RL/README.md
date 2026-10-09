# H7_RL 部署工程

本目录保存用户提供的 H7_RL 部署源码、工程配置与依赖库，与 Wheel-Legged-Gym 训练代码使用同一 Git 仓库、分支和提交。这里是普通目录，无子模块或嵌套 `.git`。

## 两人协作

用户本人负责训练、导出和仿真验证，另一位协作者负责本目录的H7固件与板端／真机验证。双方各自克隆仓库、在自己的功能分支开发，日常按负责目录分别提交，合入共同维护分支。完整分工见 [CONTRIBUTING.md](../../CONTRIBUTING.md#两人维护分工)。

RL侧交付模型和版本化接口；部署侧核对观测／动作、执行频率与板端结果。接口改变时共同更新 `docs/interfaces/` 与交付记录，尚未完成部署适配时保持明确状态。将本目录纳入同仓只统一源码版本管理。

## 来源与导入范围

- 来源仓库：zn1135/H7_RL；分支 `main`；HEAD `7aeba0bdc042e3494d55f0609d9e2393bfccbc46`。
- 日期：2026-10-09；来源工作区有1533个仅涉及 LF→CRLF 的修改文件，忽略行尾 CR 后与来源 HEAD 内容一致。
- 复制1643个已跟踪源码／工程／依赖文件，原始字节与权限保留，共211169979字节，约201.4 MiB。逐文件 SHA256 与范围记录在 [SOURCE_IMPORT.json](SOURCE_IMPORT.json)。
- 原目录保留。来源的 `.git`、未跟踪／忽略内容及286个已跟踪的 `MDK-ARM/CtrBoard-H7_ALL/` 编译产物未纳入本仓；排除清单在来源记录中。第三方静态库和许可保留，因为它们属于工程依赖。
- 仓库根 `.gitattributes` 为本目录禁用自动换行转换，使导入哈希在不同平台 checkout 后可复核。

后续修改不再要求与导入哈希一致；来源记录只标识此次导入快照。固件构建指纹仍需按构建脚本独立更新。

## 入口与本机路径

- eIDE：`.eide/eide.yml`；Keil：`MDK-ARM/CtrBoard-H7_ALL.uvprojx`；CubeMX：`CtrBoard-H7_ALL.ioc`。
- 控制代码：`imcalib/Algorithm/`、`imcalib/task/`、`imcalib/user-lib/`。
- 固件规范：[md/AGENTS.md](md/AGENTS.md)；构建环境：[md/eide-gcc-build.md](md/eide-gcc-build.md)。
- 在仓库根填写 `.env.local` 的 `H7_REPO_PATH` 为本目录绝对路径；已有本机配置不会自动改写。部分旧检查脚本要求算法根，应按其参数说明使用 `H7_REPO_PATH/imcalib`。
- `git -C deployment/H7_RL ...` 操作的是训练仓，来源仓远程／历史没有复制进来。

## 导入时发现的配置问题

1. **eIDE引用过期**：`.eide/eide.yml` 引用12个已删除的 jump/pin/stable/upstairs 网络源码、4份相应报告、3个缺失HAL源码，并引用已不存在的 `imcalib/Sysid` 源码和包含目录。现有网络是 `networkzn1`；Keil项目的文件引用检查未发现缺失。eIDE引用需在实际编译前清理和核对，本次保持导入配置原样。
2. **固件源码指纹过期**：执行 `tools/firmware_build_info.py --check` 退出1，报告 stale。脚本按原始字节计算指纹，换行转换及生成后改动都会使它失效。实际构建前按原流程重新生成并复查；本次不伪造新的构建身份。
3. **尚未适配新训练接口**：CubeAI网络仍是 `model_15000_h723`，输入 `[1,25]` 与 `[1,125]`、输出 `[1,6]` 与 `[1,3]`，policy100Hz；新mjlab是actor35、六实体电机、50Hz。这是旧部署版本，不能直接替换成新策略。
4. **旧文档频率有漂移**：实际 `machine_config.h` 大机器TIM6为1kHz、RL执行分频2即500Hz、策略分频10即100Hz；部分历史说明仍写大机器TIM6为500Hz。当前数值以导入代码为准，不能据旧说明改电机参数。

本次只做文件导入、字节一致性与配置引用检查，没有修改物理参数、控制逻辑或网络，没有重新编译、链接、下载或进行真机验收。源码放在一起便于版本管理，不代表训练与部署接口已经统一。
