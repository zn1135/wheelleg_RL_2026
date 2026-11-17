# Jetson RL 部署代码

2026-11-17从Jetson的`/home/imca/rl-ynzz_33/`取回，保留现场源码、脚本、模型和参考向量。逐文件来源与SHA-256见[source-manifest.json](source-manifest.json)。运行日志、编译缓存、临时构建目录、旧备份文件和重复ZIP未纳入；Jetson原文件保留。

## 目录与当前入口

| 目录 | 用途 |
| --- | --- |
| `policy6838-stage/` | 当前6838模型、ONNX、TensorRT engine、C++源码、协议头和启动脚本 |
| `shadow-policy4000PD200/` | 保留现场4000部署版本；当前6838脚本仍使用这里的工作目录和日志分析工具 |
| `shadow-policy3000PD200/` | 保留现场3000部署版本及历史说明 |
| `rl-usb-recorder/` | 旧Python录制器，只保存RL投入期间的type 5帧，不能用于完整辨识采集 |
| `raw-usb-recorder/` | 现场`/usr/local/bin/h7-usb-record`只读原始录制脚本 |
| `systemd/` | 当前实控服务、6838覆盖配置及独立只读录制服务的文件快照 |

实际启动链为：

```text
jetson-rl-external-4000.service
  → 6838-model.conf覆盖ExecStart
  → policy6838-stage/run-policy6838.sh
  → policy6838-stage/build/jetson_rl shadow
  → policy6838PD200.engine
```

服务名包含4000，但当前加载6838。现场服务设置`JRL_EXTERNAL_ACTION=1`，程序会向H7发送READY和动作回包，属于外部实控；不能仅凭命令名`shadow`将它当只读程序。H7负责观测、输出许可、PD和电机通信，Jetson负责策略推理及日志。

每个版本自带的`deployment/H7_RL_ros/imcalib/common/`仅是编译所需协议头快照，不是完整H7固件。目录名、历史说明和模型元数据按现场原样保留；模型身份以ONNX、engine及`.meta`的SHA-256核对。

## 构建与放置

各版本保留独立的`deployment/jetson_rl/CMakeLists.txt`及依赖布局。在安装CUDA和TensorRT的Jetson上，从对应版本目录构建，例如：

```bash
cd /home/imca/rl-ynzz_33/policy6838-stage
cmake -S deployment/jetson_rl -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/jetson_core_check
```

需要只构建协议/观测检查程序时，配置增加`-DJETSON_TENSORRT=OFF`。本次只同步和核验文件，没有重新构建、安装或启停设备服务。

启动脚本和服务保留现场的`/home/imca/rl-ynzz_33`绝对路径。部署时应将本目录内容放到该路径，再在设备上构建；在仓库目录中直接运行启动脚本仍会访问现场路径。TensorRT engine是现有Jetson的构建产物，换设备或运行环境后需从配套ONNX重新生成。

## 气弹簧采集使用的录制入口

独立只读录制链为：

```text
h7-usb-record.service
  → /usr/local/bin/h7-usb-record
  → /home/imca/jetson_rl/20261113-17679/build/jetson_rl/jetson_rl record
```

该入口保存串口原始RX字节，能保留type 13等辨识帧，不发送握手、动作或STOP。它依赖设备上既有的17679版录制程序；对应旧版源码入口见[../jetson_rl](../jetson_rl/README.md)。原始数据保存到`/home/imca/jetson_rl/logs/joint-sysid/`。

`h7-usb-record.service`与实控服务配置为互斥，USB串口同时只能由一个程序占用。当前设备仍运行6838实控服务，纯录制服务未启动；气弹簧采集需要另外安排切换。本次没有修改设备运行状态。
