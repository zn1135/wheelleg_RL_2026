# 平地训练指令：yaw扩大到±10，高速转向取消vx

2026-11-15，用户确认仅高速原地转：`|yaw|>2rad/s`时强制`vx=0`，低速yaw保留行驶转弯；随后明确yaw上限取10，故新平地训练默认范围为−10～10rad/s。没有启动新训练、没有修改已有模型或部署默认。

## 实现与实际行为

- `envs/chuanliantui/fudan_ppo_walk/fudan_ppo_walk_config.py::FudanWalkCfg.commands.yaw_range`：原−2～2改为−10～10rad/s，仍均匀采样。
- `envs/chuanliantui/fudan_ppo_walk/commands.py::HIGH_YAW_ZERO_VX_ABOVE`：阈值2rad/s；`command_config`在yaw范围超出旧±2时记录`fudan-high-yaw-parking-commands-v2`、实际yaw范围及`zero_vx_above_yaw=2`。
- `envs/chuanliantui/fudan_ppo_walk/fudan_ppo_walk.py::FudanEnv.resample`：抽取vx/yaw/高度后，将绝对yaw严格大于2的环境vx指令置0，再执行原静止/停车计划。恰好±2仍允许vx；停车计划仍覆盖为原线速度行驶或零速。

普通行走80%、原地保持10%、行驶后停车10%保持。普通环境每5秒重抽指令；普通采样中80%的yaw落在绝对值大于2的区间，因此一次采样快照中理论约64%环境高速原地转、16%低速行驶转弯、其余20%执行原静止/停车计划，不是64%完整回合始终高速转。

旧范围不超过±2的配方仍返回原v1命令元数据，不增加采样随机数；单级平台任务自行指定yaw±.5且覆盖重采样方法，本次没有修改其配置或实现。固定GPU验收使用`resample=False`时保留显式测试命令，不将旧测试矩阵改写为高速原地转。

Actor25/历史125/六动作、指令索引/缩放、奖励、PD/力矩限幅、物理步长和初态均保持。已有PT/ONNX/日志及H7不变；新元数据沿既有`command_sampling`写入后续PT，训练入口要求预检命令配方匹配，旧v1预检不能直接放行新v2训练。此次无模型交付，H7独立行为核对不适用。

## 检查与限制

正式入口，从仓库根目录运行，输出使用新文件：

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  -m scripts.agent.fudan_ppo_walk.check_high_yaw_commands \
  --output artifacts/fudan_ppo_walk/<新实验>/command-check.json
```

CPU/CUDA各10000环境执行实际重采样：高速vx全0、低速保留行驶转弯、yaw覆盖正负10附近且不越界、静止/停车原指令保持；另核±2及其两侧边界、局部重采样、旧±2配方命令/RNG逐值保持、台阶±.5元数据不变。非物理夹具另通过真实Actor观测字段、停车时钟和固定验收命令路径检查。

实际证据：`artifacts/fudan_ppo_walk/20261115-high-yaw10-no-vx-01/command-check-final.json`，真实物理积分/网络推理/优化更新均0。仅证明命令采样正确，不证明现有3000或4000模型能够跟踪10rad/s，更不代表真机通过。下一步优先在GPU对已有3000做±10纯转向量测，检查稳定窗口yaw误差和双轮支撑，再决定新训练预算；本次未执行该行为量测或启动训练。

## 2026-11-16：用户指定4000继续新增1000

用户明确选择刚归档4000→5000，不回到3000；保留源全幅随机化、真实摩擦.4～1.2、机身各轴±3N/矩0和7秒/.3m/s短推扰，其余PD/时序/双延迟/高度与17奖励逐项继承。源SHA `1513279bcc519814af3f1aea266967a937f845244002c7c03e83037fa16878b6`；4096×48/seed1/v3，新增196608000完整转移/双Adam各20000，目标5000/双Adam各100000即停。此前源4000平地40/47退化事实不变，不因选择续训将其标为实机基线。

新命令采样23项检查及本次20环境GPU7020积分/1404PD/5616保持通过，40次短推扰/外力叠加误差0，高速yaw的vx指令全0；非已学会10rad/s或行为验收。初始化与训练结束审计分别核实，未授权预算之外追加或替换模型/H7。继承的184320条弃用采样计数继续单列。[实际运行与命令](../../artifacts/fudan_ppo_walk/20261116-high-yaw10-resume4000-5000-01/README.md)。

## 2026-11-16最终5000

新增1000完整轮按预算完成并停止，双Adam各100000、恢复/预算/源码快照及当前源码核验通过；最终SHA `010106ac0b2ee356a630ab27ebe1c85ab2a6c843d21a1b4362ab81f1aafe0222`。GPU标称47例45通过，剩.30m高速前进与−2原地转的yaw误差约.10985/.10259超.1；静止3/3、停车6/6，但.26m零指令30秒仍漂.538m，不称定点保持。高速±3/5/10共14/18，±10按原判据2/6；原机体系末段XY角速度混入倾斜时竖直偏航投影，另采1ms四元数/角速度分解核实，但未改变正式门槛或重标通过。原3000基线保留，无新部署/追加训练。[最终对照与评分口径诊断](../../artifacts/fudan_ppo_walk/20261116-high-yaw5000-effect-01/RESULTS.md)。
