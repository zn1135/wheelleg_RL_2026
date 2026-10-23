# USB H7：目标腿长换算训练机体高度

## 版本与状态

2026-10-23，作者选定“几何换算”并要求实现，执行者Codex，H7部署负责人核对**待确认**。仅修改USB工程`deployment/H7_RL_ros`的大机RL命令入口；软件检查和大小机完整构建通过；作者本次重新确认断开电机动力、保留H7/RX供电后，**已于16:23:53完成大机写入、读回校验并复位，未验证真机效果**。本次替代当天15:53:22写入的直接截取腿长到0.27～0.34m版本；网络仍为model13842。

训练与部署位于同仓`26_wheelleg`，基准HEAD为`27bbb4d0cd3587fae767fff24a969041af36115d`，工作区有先前训练、网络导入和本次未提交改动。交付模型来源、仓库来源与工具版本沿用[13842模型交付](../model-deliveries/20261023-model13842-h7-import.md)。本次精确源码快照、差异、受保护文件哈希、构建与固件哈希见[证据目录](../../artifacts/h7-leg-to-body-height-20261023/summary.json)。

## 命令含义与公式

`rc.len`继续表示目标髋轴至轮轴的腿长，单位米；actor观测第3项（从0计数）改为训练所需的目标`base_link`离地高度乘5。目标按平地直立姿态换算：

```text
h_cmd = clamp(L_cmd + 0.06000 - 0.02352, 0.2165, 0.3664)
obs[3] = h_cmd * 5
```

0.06000m为训练XML的名义轮接地半径，0.02352m为其中髋轴相对`base_link`的竖直偏移，来源为`resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml`；训练高度范围来自`huananhu_mdp.py::HEIGHT_COMMAND_RANGE`。这是**训练参考下的直立目标换算**，不改变实机轮半径、杆长、零点、极性或任何物理标定，也不是实测离地高度估计。倾斜或腾空时不能据此声称实测机体高度准确；目标不随瞬时实测腿角漂动。

| 输入／输出 | 旧板上版本 | 本次USB大机源码 |
|---|---|---|
| RC腿长目标 | 原机械与K表域交集 | RL模式0.18002～0.32992m，其他模式原范围 |
| actor高度命令 | `clamp(rc.len,0.27,0.34)` | `RL_Command_Height(rc.len)`，0.2165～0.3664m |
| 腿长0.30m示例 | 高度命令0.30m | 高度命令0.33648m，观测1.6824 |
| 反馈`height` | 两腿`L*cos(theta)`均值 | 保持原定义：髋轮竖直投影，不是base离地高度 |
| 恢复低位判据 | 投影高度≤0.20m | ≤`min(0.20,L_cmd_clamped-0.01)`，允许低位命令退出恢复 |

倒地角度判据仍为pitch/roll绝对值超过40°；启动保持200ms、稳定500ms、连续倒地严格超过2000ms锁定失败及退出清理不变。恢复低位阈值调整只解决新增低位目标被原0.20m门槛一直判倒地的问题，不代表已验证低位起立行为。

权威入口为`machine_config.h`中的训练参考范围/偏移、`rc_command.c::Rc_Command_Update`中的RL腿长限幅、`rl_observation.c::RL_Command_Height`与`RL_Recovery_Update`；执行层传入本拍`control_frame.rc.len`。对应头文件同步声明。小机偏移为0、原0.27～0.34m裁剪保持；LQR机械/K表范围保持。

## 兼容性与同步

接口仍为39维actor／82维critic／6维动作，网络和model13842权重不变；模型SHA256为`b8e536241827b2e2a6311ea368b5ccc56088402b6be1f1b7bd9a956946853c0d`。历史长度、动作顺序、坐标映射、缩放、PD、延迟、初态与复位保持；策略100Hz、PD500Hz保持。无需为本次命令换算重新训练或导出模型，但输入高度范围变宽后的实机行为待验证。

| 部分 | 同步状态 |
|---|---|
| USB大机C/H及现行文档 | 已修改、未提交；构建通过，16:23:53已烧录并校验复位 |
| mjlab与CPU训练高度范围 | 已有0.2165～0.3664m，本次不改 |
| UART工程`deployment/H7_RL` | 不在本次修改范围 |
| H7部署负责人复查、板端时序与真机行为 | 待确认／未执行 |

后续烧录应使用本次完整固件；如需回退，恢复模型交付记录中的已烧录固件，不能只回退一侧RC限幅或观测公式。原model13842零速保持退化结论保持，构建通过不代表行为改善。

## 验证依据

- [高度与模式检查](../../artifacts/h7-leg-to-body-height-20261023/height-check.json)：编译实际RC、观测、配置及K表C，设备状态用替身；两机各180组模式/在线/保留档/拨轮回归，五个高度点、死区、积分速率、限幅、离线复位、切LQR、恢复计时与失败锁定通过。改变实测关节/重力输入不会改变高度目标。
- 同一检查使用当前XML验证4个直立腿长点，名义公式与网格接地高度最大差约0.04mm；这只核对训练模型参考，不是实机几何标定。
- [策略计时检查](../../artifacts/h7-leg-to-body-height-20261023/policy-timing.json)：实际分频/恢复C的有限宿主检查通过，不证明板端调度。
- [完整构建](../../artifacts/h7-leg-to-body-height-20261023/build-report.json)：大小机各108个编译单元及链接通过；两份BIN均含同一207128字节model13842权重。1290项受保护来源中仅上述5个C/H改变，其余保持。

复现入口与检查范围见[构建与测试](../ai/build-test.md)。本次未训练、重新导出模型、使能电机或采集板端控制数据。烧录前核对1290项来源与大机ELF/BIN/HEX哈希，保留旧板固件作回退；OpenOCD SWD200kHz退出0，耗时79.934秒，Programming Finished、Verified OK、Resetting Target齐备。大机HEX SHA256为`c1b1dcdc67550d8e7163321f8166947fb85a9692bef680ca1428454dcbce6305`。[烧录结果](../../artifacts/h7-leg-to-body-height-20261023/flash/result.json)、[原始日志](../../artifacts/h7-leg-to-body-height-20261023/flash/openocd.log)、[烧录前核对](../../artifacts/h7-leg-to-body-height-20261023/flash/preflight.json)。
