# 串联腿：教师动作配对与BC actor初始化

本页保留历史39维BC／PPO初始化定义。2026-10-25用户后续要求在BC增加通用轮式里程，当前新BC为40／83，详见[新观测接口](20261025-bc-wheel-odometry.md)；旧39维模型／数据可回放，不能直接当新40维初始化或部署。

2026-10-25，Codex执行，用户本人接收，独立复查未执行。正常趴地起立和零速站稳，沿用[固定单位尺度SMP PPO v3](20261025-smp-unit-reward-v3.md)及[学生PD控制](20261025-lqr-student-closed-loop.md)。训练仓为Wheel-Legged-Gym，当前detached HEAD `2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`，有既有未提交修改；[源码逐文件哈希](../../artifacts/bc-lqr-20261025/source-manifest.json)区分首轮实际训练源码和后来新增的纠偏入口，HEAD不能单独代表运行版本。没有提交／推送／PR。

## 定义与兼容

| 项目 | 本轮定义与变化 |
|---|---|
| 配对数据 | `chuanliantui_actor39_pd6_prestep_v1`，每10ms先保存39维观测，再计算教师归一化PD目标，随后原PD/FIFO执行；两者时间等于步进后记录时间减10ms |
| 观测、坐标、关节、动作 | 现有39／82／6不变，名义无观测噪声／延迟采集；腿动作×0.5为相对默认角目标，轮动作×10为速度目标；不是六路直接力矩 |
| 历史与复位 | 不增加学生历史编码器；完整初态写入后重建观测缓存，首拍上一动作清零；教师额外状态仅计算标签 |
| 时序、初态和停止 | 原2ms物理／PD、100Hz策略、2／4ms腿轮FIFO、正常趴地高度0.25m／零初速度、前0.2秒主动零出力、零速／零转向保持；保留原终止条件，不进入倒置恢复 |
| 物理资产 | 共享XML `cac8061de2850602bfd1421fc856bf217c4e5040a11c8f5d33cf1abe61d5f0cf`，映射、增益、力矩限幅和弹簧补偿不变 |
| BC模型 | `ct_bc_actor39_pd6_v1`，RSL MLPModel：39→256→128→64→6、ELU、无输入归一化；checkpoint含actor、监督训练Adam、采样RNG、来源／数据指纹；不含可用PPO critic |
| PPO初始化 | 可选`--bc-init`仅接受新的SMP PPO运行，与`--resume`互斥；核对XML／观测合同／动作实现哈希，复制actor及0.05标准差，critic与PPO优化器不变；来源随PPO checkpoint保存／恢复 |
| 纠偏数据 | `ct_bc_corrections_v1`仅训练初态，保存学生真实状态上的教师查询标签；只加入动作训练，不加入SMP成功轨迹；验证帧不变 |

权威实现：`ct_mjlab/bc.py:ActionDataset/build_actor`、`scripts/train_bc.py:train`、`smp/ppo.py:SmpOnPolicyRunner.initialize_from_bc`。普通PPO不传参数时保持原流程，v3奖励合同未改。BC文件不是PPO续训文件，不能当`--resume model_*.pt`加载；已有PPO／SMP先验文件不转换、不覆盖。纠偏初始化从actor加载，开启新的Adam状态及新显式预算，累计actor更新次数单独记录。

## 验证与交付边界

配对、5000次预算、实际RSL actor数值及PPO初始化往返检查通过；独立行为失败0/40，完整FP32复测仍0/40，不能部署。证据和源码版本见[模型记录](../model-deliveries/20261025-bc5000.md)及[正式报告](../../artifacts/bc-lqr-20261025/README.md)。

训练侧新增入口和检查已保存当前工作区，未合入。H7侧无待同步控制代码，本轮不交付固件／ONNX；H7部署负责人对未来策略及真实时序核对仍待确认。真机／板端／完整随机化与全1024姿态未执行，首轮失败模型只允许失败复现及后续监督初始化。原PPO／SMP模型与各自历史源码组合保留作回退。

2026-10-25补充：制动BC训练新增可选`rise-brake-turn-stop-v2`采样及来源元数据，不改本页任何部署输入／动作／时序定义。默认旧采样保持；回放按检查点定义核对数据指纹。累计15000模型起立／平衡40/40、偏航与平滑通过，停车仍0/40；[本轮模型与分阶段边界](../model-deliveries/20261025-bc-braking5000.md)。PPO初始化本轮未重测，未交付部署模型，负责人未来核对仍待确认。
