# chuanliantui 策略导出 ONNX

本文对应 `26_wheelleg` 分支现有的 [导出脚本](scripts/agent/export_chuanliantui_onnx.py)，适用于 `chuanliantui` / `chuanliantui_standup` 的 **25 维观测版本** checkpoint。不是 imcawl / `mini_wheel_legged` 的通用导出说明。

## 1. 准备输入和环境

输入必须是训练保存的完整 `model_*.pt`，包含 `model_state_dict` 中的 encoder 和 actor。不能使用仅含 actor 的 `policy_1.pt`，旧的 27 维 chuanliantui checkpoint 也不兼容。

在仓库根目录执行，使用本机 Python 3.8 环境：

```bash
cd /home/zn/文档/Wheel-Legged-Gym
export PATH="/home/zn/miniforge3/envs/wheellegged_py38/bin:$PATH"
```

脚本使用环境中的 PyTorch、NumPy 和 ONNX，并通过 `onnx.reference.ReferenceEvaluator` 校验结果，无需安装 ONNX Runtime。若出现依赖缺失，应在上述环境中补齐兼容 Python 3.8 的依赖，不使用系统 Python。导出本身在 CPU 上运行，不创建 Isaac Gym 或 MuJoCo 仿真。

## 2. 导出命令

先把下面 checkpoint 路径替换为实际训练产物，再执行整段命令。输出放在 `exports/` 下，使用新目录避免覆盖已有文件；不要把输出指向历史训练 `logs/`。

```bash
checkpoint='logs/chuanliantui_standup/替换为实际run目录/model_6000.pt'
if [ -f "$checkpoint" ]; then
    mkdir -p exports/chuanliantui_standup
    export_dir=$(mktemp -d exports/chuanliantui_standup/onnx-export-XXXXXX)
    python -m scripts.agent.export_chuanliantui_onnx \
        --checkpoint "$checkpoint" \
        --output "$export_dir/model_6000_full.onnx"
else
    printf 'checkpoint 不存在，请先修改路径：%s\n' "$checkpoint"
fi
```

使用 `python -m` 从仓库根目录运行，确保能够导入 `sim2sim`。如果使用其他编号的 checkpoint，建议同步调整输出文件名。

成功时脚本打印 `ONNX 导出并校验通过` 及输入、输出形状。仅看到文件生成不代表校验成功：脚本会先写入文件，再进行数值对比，应以命令成功退出和通过提示为准。

## 3. ONNX 输入与输出

默认导出使用 opset 17，所有输入、输出都是 `float32`，batch 维度动态，单机器人推理使用 batch=1。H723 请使用第 6 节的固定 batch 导出命令。

| 名称 | 方向 | 形状 | 含义 |
|---|---|---|---|
| `observations` | 输入 | `[batch, 25]` | 当前已构造、缩放并裁剪的 actor 观测 |
| `observation_history` | 输入 | `[batch, 125]` | 5 帧观测展开，最旧在前、最新在末尾 |
| `actions` | 输出 | `[batch, 6]` | 确定性动作均值，尚未做动作裁剪和控制映射 |
| `latent` | 输出 | `[batch, 3]` | encoder 输出 |

导出图包含：

```text
latent = encoder(observation_history)
actions = actor(concat(observations, latent))
```

不包含 critic 推理、随机采样、观测构造、历史缓存更新、PD 控制或力矩限幅。虽然推理不使用 critic，现有加载器仍会校验并加载完整网络权重，不能只提供裁剪后的 actor/encoder 字典。

## 4. 调用端需要保持的接口

以 [mj_sim2sim_ct.py](sim2sim/mj_sim2sim_ct.py) 的 `build_obs` 和主循环为准，25 维观测按以下顺序拼接：

| 索引（从 0 开始） | 内容 |
|---|---|
| `0:3` | 机体系角速度 × 0.25 |
| `3:6` | 机体系投影重力 |
| `6:9` | 前向速度、偏航角速度、目标高度命令，分别 × `[2.0, 0.25, 5.0]` |
| `9:13` | 腿关节 `[lf0, lf1, rf0, rf1]` 相对默认角的位置 |
| `13:19` | 六关节速度 × 0.05 |
| `19:25` | 上一次动作，未做控制尺度缩放 |

整体观测裁剪到 `[-100, 100]`。六关节顺序为 `[lf0, lf1, lfwheel, rf0, rf1, rfwheel]`，默认角为 `[-0.06, 0.10, 0, 0.06, -0.10, 0]`；轮绝对位置不输入策略。

历史缓存由调用端维护：初始化时用首帧重复 5 次，每次推理前移除最旧帧、追加当前帧。因此历史最后 25 维包含本次 `observations`。ONNX 模型自身不保存历史状态。

chuanliantui 前向为机体 **+x**，当前命令通道 1 是偏航角速度。策略频率为 100 Hz，PD 内环为 500 Hz。输出动作需按现有控制代码裁剪，腿关节目标位置为 `action * 0.5 + default`，轮目标速度为 `action * 10.0`；动作不是可直接下发的电机力矩。

## 5. 校验范围与常见问题

导出脚本自动执行 ONNX 图检查，并用固定随机种子的单组 batch=1 输入比较 ONNX 与 PyTorch 的 `actions`、`latent`，容差为 `rtol=1e-5`、`atol=1e-6`。模型还记录输入输出形状、历史排列方式和 checkpoint 路径等 metadata。

| 现象 | 处理 |
|---|---|
| `No module named sim2sim` | 回到仓库根目录，使用上面的 `python -m` 命令 |
| 缺少 `model_state_dict` 或加载失败 | 确认输入为完整训练 checkpoint，而非 `policy_1.pt` |
| 提示 checkpoint 观测接口不匹配 | 使用本分支重新训练的 25 维 chuanliantui 权重；不要强行 reshape |
| 缺少 `onnx` 或 `onnx.reference` | 核对当前解释器和 ONNX 安装是否支持参考执行器 |
| 数值对比失败 | 该导出未通过校验；保留错误输出排查，不把已生成文件当作合格产物 |

上述检查只验证导出图及一组输入的数值一致性，不代表策略能站稳、跨引擎行为通过或真机部署通过。现有 `mj_sim2sim_ct.py` 接收的是完整 `.pt` checkpoint，并不直接加载这个 ONNX 文件。行为验证流程见 [COMMANDS.md](COMMANDS.md) 的 chuanliantui 小节；接入 ONNX 推理后仍需另外验证调用端的观测、历史与控制时序。

导出数值检查和板端行为验证属于不同阶段，具体执行记录见下文。

## 6. STM32H723 + STM32Cube.AI 部署

目标芯片已确认为 STM32H723，具体料号及本机 Cube.AI 版本尚待确认。H723 系列采用带 FPU 的 Cortex-M7，Flash 有 512 KB 和 1 MB 型号，SRAM 共 564 KB、分布在不同存储区，不能将总量当作单块可用 activation RAM。实际时钟和内存布局以工程配置为准。参考：[ST H723/733 产品说明](https://www.st.com/en/microcontrollers-microprocessors/stm32h723-733.html)。

现有 `model_6000_full.onnx` 的只读检查得到 38,825 个参数，float32 权重约 151.7 KiB，算子为 `Gemm`、`Elu`、`Concat`。这给出了模型量级；完整固件还需容纳运行库、控制代码和缓冲，最终以 Cube.AI Analyze 报告和链接结果为准。

### 导出兼容性

脚本新增 `--fixed-batch` 和 `--opset {13,15,17}`。H723 先使用固定 batch=1、float32、opset 13 的版本，保持两个输入 `[1,25]`、`[1,125]` 和两个输出 `[1,6]`、`[1,3]`。opset 13 是面向较旧导入器的保守起点，不代表所有 Cube.AI 版本均已验证。

下面的 checkpoint 来自已有 ONNX 的 metadata，表示同一份权重来源，不意味着它已经通过真机行为验收。若部署其他策略，请替换路径和输出文件名。

```bash
cd /home/zn/文档/Wheel-Legged-Gym
mkdir -p exports/chuanliantui_standup
export_dir=$(mktemp -d exports/chuanliantui_standup/h723-XXXXXX)
/home/zn/miniforge3/envs/wheellegged_py38/bin/python \
    -m scripts.agent.export_chuanliantui_onnx \
    --checkpoint logs/chuanliantui_standup/Sep22_16-26-11_standup_no_legangle_resume/model_6000.pt \
    --output "$export_dir/model_6000_h723.onnx" \
    --fixed-batch --opset 13
```

opset 需要匹配安装的工具版本，不能仅凭 `.onnx` 扩展名判断兼容。例如 ST Edge AI Core 2.0.0 文档列出的支持范围到 opset 15，4.0.0 文档列到 opset 20，且均只支持算子子集。若需降低 opset，应从 PyTorch 重新导出并重新做数值比较，不应只修改 ONNX 文件的版本标签。

参考：[ST Core 2.0.0 ONNX 支持](https://stedgeai-dc.st.com/assets/embedded-docs/2.0.0/supported_ops_onnx.html)、[ST Core 4.0.0 ONNX 支持](https://stedgeai-dc.st.com/assets/embedded-docs/supported_ops_onnx.html)。

### 导入与验证

先使用 float32 模型完成 Analyze、主机数值验证、代码生成和目标板验证，再考虑量化。以工具实际报告的 Flash、activation RAM 和目标板推理耗时判断能否部署，不能只依据 ONNX 文件大小判断。

1. 在 Cube.AI 工具中选择实际 H723 料号和工程，添加 `model_6000_h723.onnx`。
2. 执行 Analyze，核对输入为 25、125 个 float32，输出为 6、3 个 float32；记录权重、activation RAM 和算子支持结果。
3. 执行主机验证，比较生成的 C 网络与原始模型输出；除了随机输入，还应补充站立、起立和行走的真实观测及其配对历史。
4. 生成代码并集成工具提供的运行库、权重和头文件。缓冲大小、对齐、输入输出描述及初始化/推理 API 均以该版本生成的示例为准。
5. 在板端验证应用中检查输出一致性并测量推理时间，然后接入观测与控制任务。不要直接将推理调用放进 2 ms 电机控制中断。

当前 ST 文档说明 STM32Cube AI Studio 已替代 CubeMX 的 X-CUBE-AI 插件；已有旧版工程应按其实际版本操作，不混用两套界面和生成 API。参考：[ST 工具说明](https://wiki.st.com/stm32mcu/wiki/AI%3ASTM32Cube_AI_Studio_documentation)、[模型分析与验证流程](https://stm32ai.st.com/stm32-cube-ai/)。

### MCU 应用层职责

MCU 每 10 ms 构造观测、更新历史并调用网络，按生成代码的张量描述绑定两个输入和两个输出，不能假定它们被合并成一个数组。历史初始化及更新顺序遵循第 4 节；观测输入共 150 个 float32，即 600 字节，输出共 9 个 float32，即 36 字节。这些只是接口数据大小，不包含网络权重、activation、栈和应用缓冲。

推理产生的动作供 2 ms 周期的 PD 内环使用。应测量包含观测处理、推理及其他任务干扰的最坏耗时，确认满足 10 ms 策略周期并保持 2 ms 控制周期。观测预处理、历史缓存、动作限幅和电机映射仍由固件实现。

起立策略还需复现 `mj_sim2sim_ct.py` 的接管逻辑：首次轮接地前保持零策略动作但继续执行 PD 和更新历史，从接地后的下一策略步开始网络推理。导出图没有封装这段逻辑。

训练策略使用串联代理的关节空间；如果实体机器人采用闭链机构，需要核对实体关节反馈到训练关节的变换，以及训练力矩到实体电机的映射。不能仅因网络能在 H723 上运行，就直接将六维动作当作真实电机命令。相关差异见 [sim2sim 部署说明](docs/ai/sim2sim.md) 的 chuanliantui 小节。

下一步需要 Cube.AI / ST Edge AI Core 版本和目标固件工程位置，才能给出与生成代码一致的 C 调用实现及内存放置方案。

### 本次执行记录（2026-09-22）

- 在当前 `26_wheelleg` 工作区、指定 Python 3.8 环境执行 `--fixed-batch --opset 13` 导出，输入为上述 `model_6000.pt`。原工作区已有未提交修改，结果不代表干净提交的验证。
- 输出：[model_6000_h723.onnx](exports/chuanliantui_standup/h723-k49oNr/model_6000_h723.onnx)。输出位于新建目录，原有 ONNX 和训练权重保留。
- 导出退出码为 0，ONNX checker 及一组固定随机输入的 PyTorch/ONNX 数值比较通过；另核对图为 opset 13，输入输出形状为 `[1,25]`、`[1,125]`、`[1,6]`、`[1,3]`。
- 尚未执行 Cube.AI Analyze、生成 C 代码、板端运行或机器人行为验证。
