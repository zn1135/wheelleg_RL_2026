# 统一GPU验收 v1

用户2026-11-02确认：各task统一GPU、流程、报告和通用运动门槛；起立能力单列。标准标识`wheelleg-unified-gpu-acceptance-v1`，覆盖SuperMimic、BC、SMP及复旦行走；Isaac历史任务在自身GPU环境验证，不与mjlab混用Python或权重。

## 唯一运动门槛

| 指标 | 所有行走策略的门槛 |
|---|---|
| 前后速度 | 稳定阶段机体前向速度平均绝对误差≤0.10m/s |
| 转向速度 | 稳定阶段机体偏航角速度平均绝对误差≤0.10rad/s |
| 停车 | 停车稳定阶段世界水平速度峰值≤0.05m/s |
| 俯仰 | 稳定巡航及停车阶段俯仰绝对值峰值≤5° |

包含等号，正负误差不抵消，峰值不用均值替代。唯一计算实现为`scripts/agent/motion/assess.py::{metrics_verdict,phase_verdict}`；新统一入口复用该实现。旧复旦0.20/0.20/0.10/10°只保留历史评分v1/v2，新报告不能采用旧宽门槛。

稳定窗口、指令范围、初态和随机条件必须在采集前按任务协议声明并记录；此次不把已撤销的旧task重新注册，也不把不同模型的接口/范围互套。复旦保留原987矩阵、2秒响应及5秒测量、原停车窗口、v2轮载/短触地/高度门槛；SuperMimic保留原预先声明的cruise/parking窗口和末1秒双轮平衡。起立不硬套巡航门槛；起立率和最终运动通过率分别记录。

## 统一流程

1. 使用各task自己的模型加载、观测和控制接口，冻结权重，只在GPU物理后端采集。mjlab为MuJoCo Warp CUDA；Isaac为原PhysX GPU，原坐标/控制/契约保持。当前play、SuperMimic测量、BC评估和SMP评估拒绝CPU设备；Isaac评估要求GPU仿真和GPU策略设备，并启用GPU pipeline。
2. 保存初态、命令、逐步轨迹、模型与源文件指纹及采样条件。原生CPU仿真只作历史复现/诊断，不作为最终GPU验收证据。CPU上的离线数学、几何、ONNX数值校验和视频渲染不属于CPU物理验收。
3. 使用同一命令生成新目录中的`acceptance.json`和`README.md`。报告字段统一为标准、任务、后端、覆盖/缺测能力、逐样本结果、起立单列、技术门槛和待用户视频验收。缺测、非有限、提前结束和未送达命令不算通过，失败留原分母。
4. 同标准/任务/初态协议的候选可追加`--baseline`。按解码后的初态数组、ID和协议核对，原通过样本任一丢失即退化；新增成功不能抵消。不同标准旧报告不能直接配对，先各自按新标准复算。
5. 用户查看正常速度、完整起步/刹停和失败片段后确认最终结果。GPU技术达标不表示跨引擎或真机通过；没有完整协议或能力覆盖时，不能称完整运动能力通过。

## 统一入口

仓库根目录执行；只读已有记录，不推进物理或网络：

```bash
scripts/mjlab.sh accept --task fudan_ppo_walk --recording <GPU录制目录> --output <新验收目录>
scripts/mjlab.sh accept --task supermimic --recording <summary与trajectory目录> --output <新验收目录>
scripts/mjlab.sh accept --task bc --recording <result与trace目录> --output <新验收目录>
scripts/mjlab.sh accept --task smp --recording <GPU规范轨迹目录或起立结果JSON> --output <新验收目录>
scripts/mjlab.sh accept --task isaac --recording <Isaac_GPU结果JSON> --output <新验收目录>
```

- 规范轨迹为`protocol.json`预声明样本和`traces/<id>.npz`，采用现有500Hz物理字段；必须证明GPU后端。完整复旦987样本只归属复旦；其他任务的正式矩阵使用自己的协议标识，不改名套用。
- SuperMimic已有完整运动记录走原保存阶段复算，再合并原最终双轮平衡；缺GPU设备记录的旧产物只列数值结果，不凭训练源校验就标GPU验证通过。
- BC新评估额外保存实际命令，末5秒按共用停车/姿态标准复算；原起立/静站队列没有前进、后退、转向覆盖，报告明确缺测，不能成为完整行走通过。旧记录缺实际命令也不补造。
- SMP普通`eval`是起立诊断，其15°等起立条件不替代行走5°门槛；统一报告将起立单列，缺运动轨迹时记“未测”。SMP行走验收须提供预声明的GPU规范轨迹。
- `isaac`只读旧环境新生成的GPU结果JSON，不在mjlab内导入Isaac。现有起立/PD结果没有完整运动测量，统一报告保留起立摘要并标记运动缺测；不把旧任务或旧CPU结果改标为运动通过。新增守卫/元数据仅作AST检查，未在旧Python3.8动态运行。
- `--baseline <同标准acceptance.json>`目前要求规范轨迹中有可核对的完整初态夹具；缺同初态证据拒绝配对。

普通`play`只用于观察，SuperMimic简易play的reward/存活、BC原`qualified`、SMP原`strict_recovered`均保留诊断含义，不替代`acceptance.json`。

## 版本与验证

只改GPU入口限制、采集元数据/BC命令记录、共用评分接入和文档；训练函数、奖励、物理、观测、XML及H7不变。旧成绩、源快照、模型和视频原位保留，新标准写新目录。检查和既有轨迹复算见[实现记录](../../artifacts/shared_validation/unified-gpu-acceptance-20261102/README.md)；当前没有新GPU仿真、策略前向、训练或真机验证。
