# Jetson RL（C++17 / TensorRT FP32）

Jetson负责25维观测、125维历史、17679网络和日志；H7_ros负责物理反馈处理、500Hz PD、遥控和电机保护。只适用于本次大机接口 `JRL2 / 1767902`，不连接 CAN，不依赖 ROS，不自动使能或启动服务。

## 当前交付与限制

代码只适配 `../H7_RL_ros`；`../H7_RL` 是数值参考、没有修改。模型是实验候选，软件迁移通过不等于机器人稳定。正式部署前至少留出 **5 GiB**，低于 **1 GiB** 或日志写入失败会退出控制；不会删除已有文件。准备阶段仅剩约550MB，测试先在内存临时目录进行；后续复查已有约185GB可用，已部署到 `~/jetson_rl/20261113-17679`。真实TensorRT的虚拟串口联调通过，没有烧录或连接真实电机控制链路。

产物及验证入口见仓库 `docs/model-deliveries/20261113-jetson-17679.md`、`docs/interfaces/20261113-jetson-h7ros-rl.md`。新代码/模型在同一版本包中交付；旧 H7_ros 的 UART 探测协议不能接收本程序动作。TensorRT engine 绑定本次 Jetson / TensorRT / CUDA；换系统后从 ONNX 重新生成。

## 接线与首次运行

UART10 已从八通道 VOFA 改为专用 RL 串口；板载 USB 原采集仍可用。H7使用1152000、8N1、无流控。

| H7_ros 信号 | Jetson 3.3V UART 信号 |
| --- | --- |
| USART10 PE3 / TX | RX |
| USART10 PE2 / RX | TX |
| GND | GND |

不连接两板电源脚。Jetson板上实际插座/引脚需按当前载板确认，不能把此表当作40针排针编号；候选设备为 `/dev/ttyTHS1`。先退出遥控投入、确认电机动力断开，再做连线与候选固件检查。烧录和带动力测试均需当次授权。

## 编译与模型转换

已部署版本可先执行 `cd ~/jetson_rl/20261113-17679`；程序和engine已就绪，可直接使用录制/回放入口。下列编译命令用于重建，在包含 `deployment/` 和 `scripts/` 的仓库/交付包根目录执行。先检查 `df -h .`；新建独立目录，不覆盖 `~/real2sim`。已安装 TensorRT、CUDA Toolkit 和 CMake，不安装新框架。

```bash
cmake -S deployment/jetson_rl -B build/jetson_rl -DCMAKE_BUILD_TYPE=Release
cmake --build build/jetson_rl -j2
./build/jetson_rl/jetson_core_check
mkdir -p engines logs
./build/jetson_rl/jetson_rl convert --onnx models/policy.onnx --engine engines/policy17679.engine
```

`models/policy.onnx` 来自本次交付包。转换器核验冻结ONNX SHA，拒绝其他模型与已有engine文件。生成 `engine.meta`，载入时检查engine与模型标识。FP32禁用TF32，不启用FP16/INT8。首次加载在控制循环前预热；网络计算使用 [TensorRT C++ API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/c-api-docs.html)。

## 录制、控制和回放

所有日志目录必须是新目录，父目录须已存在；程序独占串口。`record` 不发送任何字节，包括握手和STOP。

```bash
# 首先只读录制；新固件的状态流可用于确认协议，不产生动作。
./build/jetson_rl/jetson_rl record --port /dev/ttyTHS1 --log-dir logs/read-only-01 --seconds 30

# 仅在允许进行控制验证时使用。启动时遥控必须退出RL投入，握手后仍由遥控决定投入。
./build/jetson_rl/jetson_rl control --port /dev/ttyTHS1 --engine engines/policy17679.engine --log-dir logs/control-01 --seconds 1800

# 用记录中的完整网络输入重新推理，输出动作误差；不打开串口。
./build/jetson_rl/jetson_rl replay --engine engines/policy17679.engine --input logs/control-01/records.bin --output logs/replay-01.csv
```

Ctrl+C或进程正常退出发送STOP；异常退出/断线由H7的100ms链路/动作保护撤力。需要重新退出遥控投入并握手；不会自动切回板载网络。零action对应名义姿态，**不是零力矩**。

离线曲线在开发电脑的仓库环境生成，不要求Jetson安装Python绘图库：

```bash
mjlab_training/.venv/bin/python deployment/jetson_rl/tools/report.py --run artifacts/<取回的日志目录> --out artifacts/<新的分析目录>
```

输出 `state.csv`、`policy.csv`、`summary.json` 和 `diagnostics.png`。记录包含原始串口RX/TX、150个网络输入、六动作、100Hz状态快照、确认/首次PD序号、故障及耗时。`torque_request` 是H7门控后力矩请求，不是实测电机力矩。串口快照不能替代原USB的500Hz采集；无效标志或未更新的诊断不得用于判断控制有效。

`meta.txt`保存可执行文件、配置、engine及engine元数据SHA；`engine.meta`保存ONNX/checkpoint SHA。`summary.txt`保存后台写入/丢失/失败计数。队列最多4096条，溢出不阻塞控制，丢失计数使该会话不满足完整录制验收。日志不自动轮删。

## 控制与协议语义

- H7保留原ROS传感器坐标、连续角分支、直立承重、遥控及LQR保护。运动命令仍等待ROS的500ms解锁；解锁后按主工程限幅vx±1.5、yaw±3、机高0.22～0.30。
- Jetson按主工程使用−2°图外重力修正、gyro×0.25、dq×0.05、命令×[2,0.25,5]；当前位置减名义角，动作/当前观测裁剪±100，历史保存裁剪前帧，最旧在前。
- 首帧填满五帧历史，发送一次零动作预热。H7确认它实际进入五个有效PD子步后才运行网络；不会重复推理同一状态或追赶积压帧。上一动作来自H7确认，缺确认直接停止，避免历史与实际控制分离。
- 四腿60/2、轮增益0.045；其他动作尺度/限矩/物理标定保持ROS值，不额外乘训练中模拟响应的0.45。
- `common/jetson_protocol.h`是字节格式唯一来源，固定小端、CRC32。HELLO绑定会话和合同，ACTION绑定状态序号、控制段及H7时钟。动作必须有限、范围有效且源样本年龄<10ms；控制段改变、旧包和重复包拒绝。
- `accepted`表示动作已接收，`applied`表示它首次参与PD计算，不代表电机实际兑现。首PD耗时由H7同一单调时钟计算，起点是反馈缓存时刻，不是各传感器物理采样时刻。应用中不得相减Jetson与H7的绝对时间戳。
- USB帧/输出/时序记录结构保留。USB旧policy观测槽标无效，不伪装成Jetson当前观测；完整观测及历史改看Jetson日志。小机没有新增RL支持，原非UART10 VOFA保持。

## 软件验证入口

正式检查在 `scripts/agent/jetson_rl/`：`prepare_model.py` 对照实际主工程生成网络；`check_observation.py` 对照实际主工程连续观测；`check_core.cpp` 检查协议/会话/历史；`check_firmware_bridge.py` 执行实际ROS串口代码（HAL/时钟替身）；`check_runtime.py` 使用伪串口检查实际CLI/日志（仅推理用替身）。这不是实机验收或现有单元测试框架。

固件使用仓库原正式入口：

```bash
ARM_CC=/usr/bin/arm-none-eabi-gcc mjlab_training/.venv/bin/python -m scripts.agent.real2sim.check_real2sim_huananhu --h7-root deployment/H7_RL_ros --build-only --out artifacts/jetson_rl/<新构建目录>/check.json
```

断动力联机需30分钟无正常流序号缺口、无日志丢失，反馈缓存到首次PD的最大耗时<10ms，并验证超时/退出。完整GPU行为及真机运动另按项目统一标准验收，缺测不能记为通过。
