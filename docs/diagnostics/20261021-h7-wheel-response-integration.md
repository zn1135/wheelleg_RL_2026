# 最优轮端响应接入共享XML及训练执行器（2026-10-21）

**用户明确要求“最优拟合响应写入”，现已将joint_sharp接入共享XML、mjlab训练动作执行和原生CPU MuJoCo加载链。**旧multimode_model3023 XML路径仍链接同一源，后续不维护两份模型。此次覆盖此前“不修改正式XML/不同步训练”的边界；仅提升已选轮端候选，不改变H7控制或烧录，不启动训练/推送/提交。

唯一可编辑源为[共享XML](../../resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml)。改动双轮armature、damping、frictionloss和执行器G/V/p；零件质量、重心、刚体惯量、几何、非轮关节及电机极性/零点/电流换算保持。参数仍只称指令到响应的等效参数，不宣称真实轴端惯量或摩擦。

用户后续明确选用统一2ms：当前训练配置`huananhu_env_cfg.py`和CPU sim2sim的`DT`原本均为0.002s，保持现值；每10个物理步更新策略，即20ms（50Hz）。后续以2ms悬空轮速RMSE **2.880550rad/s**作为该运行步长的参考，1ms结果仅保留历史对照，不重拟合参数、不启动训练。复核命令：`mjlab_training/.venv/bin/python scripts/agent/check_multimode_cpu_sim2sim.py --out artifacts/h7-wheel-response-integration-20261021/timestep-2ms-cpu-check.json`；检查只覆盖有限接口/数值一致性，不代表整机行为验收。

| 轮 | armature kg·m² | damping Nm·s/rad | frictionloss名义Nm | G | V rad/s | p |
| --- | --- | --- | --- | --- | --- | --- |
| 左 | 0.00805970346267 | 0.000838690903863 | 0.250144137856 | 1.84733233865 | 62.5414261673 | 0.100001698656 |
| 右 | 0.00755670659891 | 0.00450302353157 | 0.142461960425 | 1.8556065261 | 63.7108069477 | 0.101716467885 |

## 两端怎样执行同一响应

XML存储名义输入ctrlrange±3.9Nm、原轮速PD对应的名义请求以及每轮G/V/p，正常策略目标/动作尺度、0.1轮速P、500Hz控制/50Hz策略、腿控制、恢复门控及名义限幅不改。轮端损耗和附加惯量由物理引擎计算。

```text
along = max(0, sign(u) * wheel_velocity)
shape = max(0, 1 - along / V)^p
equivalent_drive = G * u * shape
```

CPU：XML为user增益，sim2sim/wheel_response.py加载已验证的h7_drive_response.c原生回调，MuJoCo每个物理步按当前仿真速度计算G*shape。训练：model.load_spec()只在内存将双轮改为fixed增益G；HuananhuPdAction在每个物理步向后端送u*shape，后端再乘G。因此既不漏速度因子，也不重复乘G。参数只来自共享XML，无第二份手写拟合数值。传给H7的名义控制定义没有改变。

model.load_model/load_view_model属于训练侧初始化/姿态辅助，使用训练后端表示；若要独立进行原生控制回放，应走已接入的CPU sim2sim入口。单独调用MuJoCo.from_xml_path并直接步进仍需注册user增益回调。

原生回调首次在artifacts/mujoco-wheel-response/<源码哈希>-<MuJoCo版本>/缓存共享库，之后直接加载。没有可用cc/gcc时设置CC为已有宿主编译器；不安装Python依赖、不编译H7。原Python回调在再次编译XML时无法包装临时模型，首次检查失败已留档；复用已验证原生C回调后，多次加载与正式检查通过。

## 验证结果与范围

- 共享XML直接读取拟合值，不在步进循环注入参数/未来状态；4次留出悬空脉冲自由积分，1ms轮速RMSE **2.779181rad/s**、MAE **1.447647rad/s**，与独立候选逐点差 **0.0**。
- 原生2ms步长（训练步长）重新积分并按2ms保持输入，RMSE **2.880550rad/s**。该值包含时间网格/输入边沿变化，不当作实测控制延迟，也不是GPU float32全轨迹精度验证。
- 126个训练/CPU力矩计算点最大差 **8.88e-16**，涵盖零输入、正反、同向/反向和速度截止，验证G仅乘一次。该计算等价不证明未采工况的物理正确性。
- 正式CPU接口有限检查通过，包括共享来源、闭链初态、观测/控制/限幅、弹簧、接触选项和恢复高度。
- mjlab实际GPU 2环境×16步有限检查通过，actor/critic为[2,39]/[2,82]、动作6维，无非有限值。此检查创建策略做有限推理，没有调用学习、没有训练日志目录，不是站立/行走/恢复行为验收。
- H7来源、机器标定和网络权重哈希保持；部件质量/惯量/重心/几何/接触、非轮自由度/执行器未变，wheel字段精确等于选定候选。旧冻结候选也在新加载链下复算通过。

证据为[接入检查](../../artifacts/h7-wheel-response-integration-20261021/check/report.json)、[CPU检查](../../artifacts/h7-wheel-response-integration-20261021/cpu-contract-native.json)、[GPU有限检查](../../artifacts/h7-wheel-response-integration-20261021/mjlab-check-summary.json)、[历史候选检查](../../artifacts/h7-wheel-response-integration-20261021/frozen-candidate-check/verification.json)。未对照旧RL俯仰，未启动训练、TensorBoard或硬件。

## 历史数据与参数来源

旧原始XML按原始字节归档至artifacts/model-versions/<sha256>/source/model.xml，网格链接仍指向同一只读资产。real2sim加载旧录制时，当前XML哈希不匹配会查找该冻结版本，仍严格验证原哈希；不改写meta或录制、不跳过哈希检查。历史校准检查也对照录制版本，而非把已授权更新的活动模型误当作旧基线。

当前同一录制会话先前已用于选择模型，本次2.779只是接入结果与既有候选一致的复现，**不是新的独立盲验证**。独立候选、首次失败、旧源码和旧指标均保留，来源/版本/哈希见[交付清单](../../artifacts/h7-wheel-response-integration-20261021/delivery-manifest.json)。

## 训练影响与尚未验证项

今后启动该工作树中的mjlab训练会读取新标称模型和驱动响应，不再只是CPU仿真变化。本次保持既有训练随机化规则：默认轮干摩擦还会在新XML基准上加[0.05,0.25]、阻尼加[0,0.01]，质量/重心/电机强度等原随机化继续执行；所以随机化后的每个环境不会等于标称悬空曲线。本次没有擅自重定随机化范围或宣称它已校准。

只有±3名义Nm/300ms悬空驱动和零输入滑行用于拟合；小幅值、主动反向制动、温度/母线电压/接地负载和整机gap未验证，不能保证重训后整体sim2real改善。G/V/p和armature为耦合等效参数，现有数据不足以分离真实惯量/电气响应。p接近搜索下界，1ms到2ms误差略增，尚不能称数值或跨工况完全稳健。

## 复现检查（不训练）

从本worktree根目录执行，使用新的输出目录：

```bash
# 仅当没有cc/gcc且尚未生成缓存时，先将CC设为已有宿主编译器路径
CC=/home/zn/miniforge3/envs/fudan_env/bin/x86_64-conda-linux-gnu-gcc \
  mjlab_training/.venv/bin/python scripts/agent/check_shared_wheel_response.py --out artifacts/h7-wheel-integrated-recheck
mjlab_training/.venv/bin/python scripts/agent/check_multimode_cpu_sim2sim.py --out artifacts/h7-wheel-cpu-recheck.json
scripts/mjlab.sh check --task chuanliantui --num-envs 2 --steps 16
```

未提供或执行训练命令；训练预算和后续新模型验收不在本次写入任务内。H7部署负责人对本次仿真变化的核对状态为待确认，板端固件与策略未改。
