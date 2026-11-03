# 模块职责

最终验收现统一为GPU、共用运动门槛及报告，见[统一GPU验收](../interfaces/20261102-unified-gpu-acceptance.md)。主线统一入口为`scripts/mjlab.sh accept --task <任务> --recording <GPU记录> --output <新目录>`；起立单列，旧CPU流程仅保留历史含义。

维护SuperMimic、BC、SMP及独立复旦PPO行走适配，实验结论见[项目实验记录](project-log.md)，已批准经验见[训练经验](training-lessons.md)。

| 模块 | 职责 |
|---|---|
| `ct_mjlab/supermimic/` | 世界模型、估速器、策略和价值网络训练；自身train/play/export入口 |
| `ct_mjlab/bc.py`、`bc_preservation.py`、`scripts/train_bc.py` | 行为克隆、纠偏、技能保留与模型初始化 |
| `ct_mjlab/smp/`、`scripts/train_prior.py` | 示范特征、运动先验、SMP辅助PPO及BC初始化PPO |
| `ct_mjlab/teacher/`、`scripts/teacher.py`、`scripts/collect.py` | LQR／VMC对照与示范数据采集，服务BC和SMP |
| `ct_mjlab/envs/` | 注册`smp`、`supermimic`、`fudan_ppo_walk`；旧目录保留必要控制／物理／模型接口依赖，不提供五个旧task |
| `scripts/agent/supermimic/`、`imitation/`、`motion/` | 保留线检查、测量、评分与渲染 |
| `scripts/agent/shared/` | 原样提取的渲染、物理采集和静态几何助手，无旧训练CLI |
| `scripts/agent/real2sim/`、`sim2sim/` | 实采／标定及独立MuJoCo仿真；仿真实现与必要接口保持 |
| `wheel_legged_gym/` | Isaac历史环境，独立Python3.8；不在本次mjlab清理范围 |
| `resources/`、`deployment/` | 机器人资产与部署控制；本次不改物理参数或H7 |

## 共用依赖

SuperMimic使用原walk环境结构，其split实现仍复用stand_v2及stand的控制和接口；BC／SMP仍使用multimode的基础观测、动作和runner。MuJoCo双策略仿真直接导入split的配置／模型验证／切换模块，所以这些原路径保持。停止独立训练线不等于删除这些共用物理实现。

旧注册`chuanliantui`、`chuanliantui_stand`、`chuanliantui_stand_v2`、`chuanliantui_getup`、`chuanliantui_walk`均已删除。模型内部既有合同标识保留，以继续严格校验⑥⑦⑧及MuJoCo依赖的旧权重；它们不是可选训练task。

当前维护分支为`26_wheelleg`；H7分工、Git操作与Isaac保护规则仍按根AGENTS.md及[CONTRIBUTING](../../CONTRIBUTING.md)。

`ct_mjlab/fudan_ppo_walk/`新增固定复旦plane配方的实体接口适配：独立PPO/估速优化器、25/125/141+3输入、物理控制复用、独立命令/奖励/GPU最终验证；`scripts/agent/fudan_ppo_walk/`负责正式检查、有限预算预检、两引擎测量/评分/视频。规格见[复旦PPO](../mjlab-fudan-ppo-walk.md)。

复旦当前唯一环境/配置在`envs/fudan_ppo_walk/`，算法/网络/存储/runner在`rsl_rl/`；旧`fudan_ppo_walk/`保留合同、模型/评估工具与导入兼容，不再维护另一套训练器。新入口及职责详见[结构整理记录](../maintenance/20261103-fudan-structure-alignment.md)。
