# 双腿镜像负载正式采用记录

2026-11-09用户观察新仿真表现接近实机，明确要求正式替换训练和MuJoCo。当前26_wheelleg、HEAD c3c8fdffe0739aed314fb71c69f2a2d3135b3fb9，有未提交改动；采用是开发选择，不是独立实机误差通过。H7无修改/烧录，负责人独立复查待确认。

## 版本与权威入口

- 物理profile：`mirrored-local-load-dahl-20261109-v1`；参数`resources/robots/chuanliantui_new_1/mjlab/fitted_leg_load.json`，SHA256 `e6ef82fa4380fa01341fa2166be56282c5290c82fec2b2a04e5b3107ce49218f`。
- 共用公式/状态：`ct_mjlab.envs.chuanliantui.fitted_leg_load`。PD动作默认开启；复旦`sim.fitted_leg_load=True`；原生Playback默认开启，StandV2的每个积分子步更新记忆。
- JSON、代码和名义气弹簧共同构成模型，仅交XML不完整。结构XML保持505c9b9b…；闭链/实测限位、轮模型、质量/惯量、动作/观测/控制和网络形状保持。
- 新PPO及SuperMimic检查点记录物理profile/SHA；复旦/SuperMimic新训练快照保留JSON。旧权重及其原始source/hash不重写；旧行为成绩不能移作新物理成绩。

## 运行语义

保留名义0～150N弹性基础项和50Ns/m阻尼，在其上叠加局部保守关节负载、0.5Nm转动摩擦、前/后0.143606/0.972227Nms/rad阻尼、2000Ns/m滑杆阻尼和110/55.5N容量、300000N/m刚度的预滑移记忆。左腿镜像右腿关节角与力矩，滑杆方向不翻转；左腿非独立标定。回合记忆初力0，下一物理拍锁存复位姿态；局部复位不清其他环境状态。

共用Hu启动随机化在新电机摩擦/阻尼基线上叠加原扰动，避免覆盖成旧基线；其他已有随机化保持。显式旧`--response-parameters`诊断包会先关闭正式默认负载，防止重复施力。回退：复旦评估`--leg-load-profile legacy`，训练`sim.fitted_leg_load=False`，其他动作/原生构造同名字段False。

## 验证与限制

归档施力、新NumPy/Torch CPU/CUDA、原生索引/复位、20环境GPU局部复位及启动随机化复查通过。默认新物理8000模型20例完整，技术0/20；legacy回退20/20、无原19个成功丢失（GPU有数值差，不称逐位重现）。旧双腿试验及新默认均未达到运动指标，未启动训练；无新完整987、独立左腿、接地真机或跨姿态标定通过。

未完成项、拟合流程、数值问题及优化顺序保留在[Word记录](../diagnostics/20261109-双腿负载拟合流程与训练仿真接入.docx)；[验证和执行来源](../../artifacts/length-response-fit/20261109-formal-integration-01/README.md)。推荐先保持PD/策略节拍做0.2/2ms积分对照，再决定新数据与训练。
