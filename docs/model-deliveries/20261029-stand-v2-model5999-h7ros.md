# 稳站v2 model5999：ONNX与H7_ros固件候选

2026-10-29，用户指定5999，随后明确范围为“适配H7_ros固件并编译，暂不烧录”。交付人Codex，接收人为用户；H7负责人签认待确认。ONNX、AIR估力/实机代理适配、正式软件检查、独立代码复查及最终大小机编译已完成；未烧录。用户随后明确确认可复用AIR支撑力估计。

## 任务与来源

| 项目 | 内容 |
|---|---|
| 机器人/任务 | 华南虎闭链，`chuanliantui_stand_v2`，B窄核跟踪候选 |
| 训练/部署仓 | 同仓`https://github.com/zn1135/wheelleg_RL_2026.git`，分支`26_wheelleg` |
| 完整HEAD | `952e4260aedf04e28fdbf366d5cb1c5562c75263`＋原有及本次未提交内容；无提交/推送/PR |
| checkpoint | `logs/mjlab/chuanliantui_stand_v2/20261029-155722_tight_fromA4999_2048_activew1_tight/model_5999.pt` |
| PT SHA256 | `a41b77caaf02389e2a1160a1781eda0d573ed33836e3cd6fe32ed02d090e825a` |
| 原实际训练 | seed42、2048环境；从A4999完整恢复两个优化器，5000～5999共1000次更新 |
| 父模型 SHA256 | `83314f8b551c45bbae9c1b678c650da87b945701a1064da536442ebccb5467ce` |
| 训练证据 | `artifacts/20261029-155722_tight_fromA4999_2048/`中的run-info/result、env/agent、source及workspace.patch；原始shell命令未单独保存，不重构成执行收据 |
| 训练完成 | 训练入口completed=true；无独立外部shell退出收据，不能把完成检查退出0当训练退出码 |
| 源码快照 | `artifacts/model5999-h7ros-20261029/export-source.tar.gz`、firmware-source.tar.gz、两份workspace差异及firmware-sources-sha256.json；训练时原快照仍在上行原训练证据目录 |
| XML | `resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml`，SHA256 `cac8061de2850602bfd1421fc856bf217c4e5040a11c8f5d33cf1abe61d5f0cf` |
| 配方 | `stand-v2-upright-first-tracking-active-weight1-tight-v6` |
| 课程 | 高度已解锁0.22m；速度仍第0级±0.5m/s，未解锁±1/±1.5 |

生效训练env/agent配置随模型包保存，校验值见`SHA256SUMS`。当前导出源快照已逐字节核对；旧日志与训练模型不改写。

## 模型与导出

模型包：`deployment/H7_RL_ros/models/model5999-20261029/`，含完整PT、ONNX、256组参考NPZ、导出JSON及训练配置；导出原件在`exports/chuanliantui_stand_v2/model5999-h7ros-20261029/`。

- ONNX SHA256：`a1b2b1508b15ca759a5303a5396360eb5d8ed05ee8ccbf4068bf304dd9953be1`。
- float32、batch1、opset13；`observation_history[1,160]`→`actions[1,6]`、`velocity[1,3]`。
- 五帧各32维，从旧到新；图内最后32维为当前帧。图不包含历史维护、预处理、命令限速、恢复状态机、PD或电机映射。
- 速度输出为机体系质心m/s，图内actor消费的估速latent仍为该值的2倍；外部不要再次乘2送回图。

实际导出命令（退出0）：

```bash
scripts/mjlab.sh export \
  --checkpoint logs/mjlab/chuanliantui_stand_v2/20261029-155722_tight_fromA4999_2048_activew1_tight/model_5999.pt \
  --output exports/chuanliantui_stand_v2/model5999-h7ros-20261029/model_5999_opset13.onnx \
  --samples 256
```

ONNX结构检查通过；256组[-1,1]随机历史，图与checkpoint策略路径动作/估速差均0，ONNX动作最大差`1.2040138244628906e-5`、速度`2.7567148208618164e-6`，绝对容差`1e-4`。

## 生成与软件核对

ST Edge AI Core `v2.2.0-20266` / CubeAI10.2.0，`--target stm32h7 --name networkzn1 --compression none --optimization time --c-api legacy`；生成退出0，成套7个C/H及报告替换。SDK运行库保持。319272字节生成权重中的14个ONNX参数张量均逐字节匹配。工具的RAM/ROM估算路径提示不等同于固件链接结果。

工具版本：Python3.12.3、PyTorch2.14.0+cu130、NumPy2.5.3、ONNX1.22.0、MuJoCo3.11.0。原始命令/日志与工具、源码及权重校验记录在`artifacts/model5999-h7ros-20261029/`。

| 阶段 | 结果/范围 |
|---|---|
| 生成C与ST宿主运行库 | 64组、576输出，最大绝对差`8.225440979003906e-6`＜`1e-4`；电脑执行，非板端 |
| 实际rl_policy封装/SDK | 同64组参考输入通过，包含160/6/3的尺寸、类型和独立缓冲绑定 |
| 实际观测/FIFO与训练actor对照 | 80拍合成反馈，最大差`5.960464477539063e-8`；覆盖速度重排、缩放后裁剪、首帧填满、退出重填、模式切换保留历史、限速/未就绪立即清零、无效数据拒推理 |
| 数值范围限制 | 百量级极端裁剪输入的图/C最大绝对差约`4.12e-4`，未满足统一绝对`1e-4`；只作为边界数值诊断，不计入交付数值通过的64/256组 |
| USB遥测 | 实际C打包7帧及v3解析通过；32维帧放57～88，89～95保留0；v1/v2解析、分片和CRC拒绝通过 |
| 最终大小机编译 | `final-build/check.json`，各108单元编译与完整链接通过；仅既有BMI088 packed告警 |
| 恢复/承重 | 实际C计时/门槛合成检查通过；复用AIR及实机代理已编译，未对真实估力/完整执行任务进行物理验证 |
| 异步生产发布 | 实际task_policy＋SDK/RTOS替身，观测/动作配对、共享数据在推理中变更、reset_count拒旧动作通过 |
| 烧录、板端数值/最坏耗时、真机 | 未执行；用户本次明确不烧录 |

正式检查脚本为`scripts/agent/check_h7_stand_v2.py`。首次运行的训练对象属性构造错误已修复；两次极端数值容差失败已保留原日志，分析见`check-core-v3/numeric-diagnosis.json`。随后将观测边界/有限性检查与固定参考集数值验收分开，未放宽参考集原绝对容差。最终正式结果为`check-final-producer/result.json`，收录17项相关源码指纹；旧v1/v2混合录制/回放/CSV/PNG回归通过，其`real_c_extension_bytes_record_replay_csv=false`如实表示使用合成旧帧；v3另有实际C打包检查。混合v2/v3绘图分支仅静态复查。最终独立复查为`review-final.md`，未发现实质代码缺陷，不代替H7负责人签认。


## 固件候选与接口适配

[本次接口记录](../interfaces/20261029-h7ros-stand-v2-model5999.md)说明32维字段、FIFO、命令、AIR估力与实机代理。实际投入统一200ms零力矩保持；恢复/首次起立按100Hz计时，未站起前保留超过旧2秒的恢复机会；站起后增加失支撑/倒置/倒地锁存。速度只接受已解锁±0.5m/s、固定1m/s²斜率，yaw=0，高度目标0.22m。

AIR按1kHz运行独立状态，采用现有电机反馈估力，不读取LQR目标F或共享flight。仅为估力准备实体弹簧/重力，不叠加到RL输出。机高为现有几何＋训练参考偏移；缺少机身触地传感器，首次起立采用姿态/几何/估算双轮承重代理。W取XML名义131.4008700N，**不是实机称重或物理标定**。这些代理与新增1kHz计算负荷尚未台架确认，不宣称严格训练状态等价。

大机固件：`artifacts/model5999-h7ros-20261029/final-build/h7-machine-0/firmware.hex`，SHA256 `63ce2f7123644c5799de3df14ef5167f5eb9863025a25b2dd05b8fc013d4fd0a`。同时保存ELF/BIN；小机只编译且RL配置仍未启用，不作为本模型的部署对象。两机BIN中均找到唯一319272字节、逐字节一致的生成网络权重块，证据见`firmware-verification.json`。

1381项受保护输入只改变19项：7个生成网络C/H、32维观测/历史/SDK接口、执行层恢复/估力接入、遥测、策略发布及观测缩放/频率注释；物理标定、原PD增益/限矩、AIR/LQR算法和硬件配置保持。原C机器表逐字节核对，除gyro/joint速度观测缩放0.5/0.1→0.25/0.05外保持。源快照/差异/文件与固件校验值均保留，不覆盖历史产物。

实际最终构建命令（退出0）：

```bash
ARM_CC=/usr/bin/arm-none-eabi-gcc mjlab_training/.venv/bin/python \
  scripts/agent/check_real2sim_huananhu.py --h7-root deployment/H7_RL_ros \
  --build-only --out artifacts/model5999-h7ros-20261029/final-build/check.json
```

后续CubeMX重新生成前须将IOC中历史Windows模型来源改选本包ONNX；当前CLI转换和编译不读取该旧来源。本轮未连接下载器、Jetson、串口或电机。

## 行为边界与回退

本次不重跑训练或物理回放。历史仿真见[窄核效果记录](../../artifacts/stand-v2-tight-effect-20261029/README.md)：相对A4999速度跟踪有所改善，但两组零速原成功各丢20例，拒绝作为全面升级替换A。此次为用户指定候选导入，不撤销原行为退化结论。

该ONNX与旧39维model13842固件不兼容，不能仅换权重。回退必须成套恢复旧网络及旧观测/策略/恢复代码；旧模型包仍在`models/model13842-20261023/`，本次修改前H7源文件备份及1381项受保护文件哈希在`artifacts/model5999-h7ros-20261029/before/`和`protected-before.json`。UART工程`deployment/H7_RL`未修改。

未执行Isaac、独立CPU物理sim2sim、板端真实100Hz最坏周期、CAN反馈/出力及真机行为验收；当前CPU入口不支持本速度配方。H7部署负责人核对待确认。
