# S2R1 上位机策略复算与 MuJoCo 离线复刻

`sim2sim/s2r_offline.py` 读取 H7 `tools/s2r_capture.py` 解码后的单个
`boot_*/session_*` 目录，不连接串口，也不向电机发送命令。它提供两个独立结果：

- `--checkpoint`：以记录的 25 维观测、125 维历史在上位机重算 actor + encoder；
  与 S2R1 已转换为训练空间的 `POLICY.action_published` 比较动作误差。
- `--mapping`：按记录的六路 `tau_motor_request` 时间序列驱动真实闭链 MuJoCo 模型，
  输出六路电机角的实测／模型对照。使用 100 Hz 短窗 CONTROL；普通 5 Hz
  遥测档缺口太大，脚本会拒绝。模型以给定基座高度、水平姿态和实测关节初态启动。

用本仓库 `.env.local` 指定的 Isaac Gym Python 3.8 环境运行。`import isaacgym`
须先于 `torch`；若动态加载器找不到 `libpython3.8.so.1.0`，先按本机 conda
环境设置库搜索路径。运行示例前先 `source .env.local`；本机环境还需设置
`LD_LIBRARY_PATH="$(dirname "$WHEELLEGGED_PYTHON")/../lib:${LD_LIBRARY_PATH:-}"`。
以下命令只处理已采集文件：

```bash
"$WHEELLEGGED_PYTHON" sim2sim/s2r_offline.py \
  --session <S2R1采集目录>/boot_<id>/session_<id> \
  --checkpoint <完整的25维model_*.pt> \
  --mapping <已核对的六电机映射.json> \
  --base-height <该段开始时的机身高度_m> \
  --out <不存在的新结果目录>
```

可只传 `--checkpoint` 或只传 `--mapping`。`--checkpoint` 必须是包含 encoder 的
`model_*.pt`，不能用 `policy_1.pt`。输出是 `policy/policy_comparison.csv`、
`policy/policy_report.json`、`mujoco/motor_comparison.csv`、
`mujoco/mujoco_report.json` 和汇总 `summary.json`。每次用新结果目录，不覆盖历史数据。

映射模板如下。六个 `joint`、`sign`、`offset_rad` 均需现场逐项核对；
`approved` 由台架人员确认后才改为 `true`。不能根据本仓库关节名猜实机极性和零点。

```json
{
  "approved": false,
  "motor_to_model": {
    "front_left": {"joint": null, "sign": null, "offset_rad": null},
    "rear_left": {"joint": null, "sign": null, "offset_rad": null},
    "front_right": {"joint": null, "sign": null, "offset_rad": null},
    "rear_right": {"joint": null, "sign": null, "offset_rad": null},
    "wheel_left": {"joint": null, "sign": null, "offset_rad": null},
    "wheel_right": {"joint": null, "sign": null, "offset_rad": null}
  }
}
```

**结果边界：**当前 S2R1 不记录机身世界位置、真实线速度和接触真值；
`tau_motor_request` 是请求值，实际发出与电机内部生效时刻也未完整记录。
因此角度误差只能作为电机／模型响应诊断，不能据此宣称整机运动精确复刻或真机验证通过。
起始高度、姿态、气弹簧力、地面和载荷不准都会改变结果；脚本会在报告中保存
初始高度和气弹簧力假设。策略复算只有在 checkpoint 与板端模型权重对应时才有数值对齐意义。

## 上位机实时推理

`sim2sim/host_policy_usb.py` 使用完整 checkpoint 接收 H7 `HPI1` 实时观测并返回
训练空间动作；单片机仍负责现有 PD、闭链映射、电机限幅和失联锁止。运行协议、
物理许可和验收边界见 H7 仓库 `md/host-policy-usb.md`。默认命令只探测接口，
**不会 ARM**：

此脚本另需在同一 Python 环境安装 `pyserial`（`"$WHEELLEGGED_PYTHON" -m pip install pyserial`）。

```bash
"$WHEELLEGGED_PYTHON" sim2sim/host_policy_usb.py \
  --port <H7板载USB CDC串口> \
  --checkpoint <完整的25维model_*.pt> \
  --out <不存在的新目录>
```

固定机身台架完成物理许可、急停、符号和时序检查后，才可由现场人员显式追加
`--enable-output --duration-s <秒>`。该功能在当前工作区只经过源码编译与主机协议检查，
没有烧录或真机验证；不要把它当成已可安全运行的实机策略。
