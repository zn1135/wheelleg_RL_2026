# Jetson 实采与实测回放

以下命令均从仓库根目录执行。模型采纳状态以[项目实验记录](../ai/project-log.md)为准，示例模型不代表当前推荐。

## Jetson操作：连接、录制与回放

当前Jetson只读接收H7反馈，不发送策略动作或电机指令。下面分别标明电脑端和Jetson端；Jetson项目为 `~/real2sim`，使用其 `.venv/bin/python`。2026-10-22已核对Jetson实际脚本的 `--help`；当前不支持 `--firmware-manifest` 或 `--warmup`，不要在下列Jetson命令中加入这两个参数。

<a id="jetson只读usb采集连接"></a>

### 电脑端：连接Jetson

在电脑的本项目根目录运行。本机用户名、SSH地址、Jetson项目目录和H7 USB设备保存在Git忽略的 `.env.local`，字段见 `.env.example`。`imca`是当前Jetson用户名；SSH连接使用配置中的 `JETSON_SSH_TARGET`：

```bash
source ./.env.local
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$JETSON_SSH_TARGET" 'id -un; hostname; ls -l /dev/serial/by-id'
# 进入Jetson终端；后续“Jetson端”命令也可直接在Jetson本机终端运行。
ssh "$JETSON_SSH_TARGET"
```

成功判据：用户名正确，且列出STM32 Virtual ComPort设备；H7数据口不是PowerDebugger的CDC口。SSH密钥或密码不写入项目。

### Jetson端：准备USB设备与实时画面

```bash
cd ~/real2sim
ls -l /dev/serial/by-id/
# 取H7的稳定设备路径；应只输出一个实际存在的STM32 USB口。
h7_usb_port="$(ls /dev/serial/by-id/*STM32_Virtual_ComPort*-if00)"
printf '%s\n' "$h7_usb_port"
fuser "$h7_usb_port"
```

`fuser`没有输出表示当前没有接收进程占用；有进程时先在其终端按Ctrl+C结束。每个新Jetson终端重新设置 `h7_usb_port`，只开一个USB接收进程。

```bash
# Jetson本机桌面：显示实机姿态，Ctrl+C或关窗结束。
.venv/bin/python sim2sim/real2sim_huananhu.py --port "$h7_usb_port"
```

成功判据：窗口姿态随实机变化，来源不持续显示STALE（数据过期）。屏幕位置固定，不表示实测位移。普通SSH终端采用下面的无窗口录制；开画面的命令在Jetson本机桌面执行。

### Jetson端：录制120秒实机RL数据

先结束上面的实时显示进程，再在同一Jetson终端运行：

```bash
rl_record_dir="artifacts/model6848-real-rl-$(date +%Y%m%d-%H%M%S)"
printf '本次录制目录：%s\n' "$rl_record_dir"
.venv/bin/python -u sim2sim/real2sim_huananhu.py \
  --port "$h7_usb_port" \
  --record "$rl_record_dir" \
  --headless --seconds 120
```

成功判据：会话目录生成 `stream.jsonl` 和 `meta.json`，正常结束后 `meta.json` 中 `complete` 为 `true`。录制启动不会自动投入RL；是否投入以遥控档位和记录中的实际投入状态为准。USB断开后应另开新会话，保留中断段，不手工改成完整记录。

当前 `deployment/H7_RL_ros` 遥控档位如下；轮矩测试独占期间的切档另受台架状态机约束。

| 左拨杆／右拨杆 | H7请求的模式 |
|---|---|
| 左下／右下 | 禁用 |
| 左上／右下 | 选择RL，不投入出力 |
| 左上／右中 | 投入RL，实际出力还受反馈和许可条件约束 |
| 左上／右上 | 轮矩脉冲测试档，不是RL投入档 |

### Jetson端：回放录制

在Jetson本机桌面、同一终端中回放刚录的会话；回放不占用USB：

```bash
cd ~/real2sim
.venv/bin/python sim2sim/real2sim_huananhu.py --replay "$rl_record_dir"
```

新终端中先列出会话，再把所选目录赋给变量：

```bash
cd ~/real2sim
ls -dt artifacts/model6848-real-rl-*/
rl_record_dir="artifacts/<要回放的会话目录>"
.venv/bin/python sim2sim/real2sim_huananhu.py --replay "$rl_record_dir"
# 普通SSH下只检查回放，不开窗口。
.venv/bin/python sim2sim/real2sim_huananhu.py --replay "$rl_record_dir" --headless
```

列表按目录修改时间排序，不保证第一项就是最近开始录制的会话。成功判据：能够读完整会话，画面按录制时间变化；这是实测姿态重放，不是MuJoCo自由积分预测。

### 电脑端：取回录制并生成CSV和曲线

回到电脑的本项目根目录；把 `rl_session_name` 换成Jetson上实际的会话名：

```bash
source ./.env.local
rl_session_name="<实际会话名，例如model6848-real-rl-YYYYMMDD-HHMMSS>"
rl_local_dir="artifacts/jetson-import-$(date +%Y%m%d-%H%M%S)/$rl_session_name"
mkdir -p "$rl_local_dir"
rsync -a "$JETSON_SSH_TARGET:$JETSON_PROJECT_DIR/artifacts/$rl_session_name/" "$rl_local_dir/"
mjlab_training/.venv/bin/python sim2sim/real2sim_huananhu.py --analyze "$rl_local_dir"
```

成功判据：本地会话的 `analysis/` 下生成 `trajectory.csv`、`report.json` 和 `signals.png`。分析会核对录制所用模型及机器配置哈希；缺少对应版本时补齐原版本，不跳过核对。完整控制输出的采集与检查另见[完整输出接口](../../docs/interfaces/20261021-h7-output-capture-v1.md)。

