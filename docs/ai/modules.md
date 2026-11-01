# 模块职责

仅维护SuperMimic、BC与SMP，实验结论见[项目实验记录](project-log.md)，已批准经验见[训练经验](training-lessons.md)。

| 模块 | 职责 |
|---|---|
| `ct_mjlab/supermimic/` | 世界模型、估速器、策略和价值网络训练；自身train/play/export入口 |
| `ct_mjlab/bc.py`、`bc_preservation.py`、`scripts/train_bc.py` | 行为克隆、纠偏、技能保留与模型初始化 |
| `ct_mjlab/smp/`、`scripts/train_prior.py` | 示范特征、运动先验、SMP辅助PPO及BC初始化PPO |
| `ct_mjlab/teacher/`、`scripts/teacher.py`、`scripts/collect.py` | LQR／VMC对照与示范数据采集，服务BC和SMP |
| `ct_mjlab/envs/` | 只注册`smp`、`supermimic`；旧目录保留必要控制／物理／模型接口依赖，不提供五个旧task |
| `scripts/agent/supermimic/`、`imitation/`、`motion/` | 保留线检查、测量、评分与渲染 |
| `scripts/agent/shared/` | 原样提取的渲染、物理采集和静态几何助手，无旧训练CLI |
| `scripts/agent/real2sim/`、`sim2sim/` | 实采／标定及独立MuJoCo仿真；仿真实现与必要接口保持 |
| `wheel_legged_gym/` | Isaac历史环境，独立Python3.8；不在本次mjlab清理范围 |
| `resources/`、`deployment/` | 机器人资产与部署控制；本次不改物理参数或H7 |

## 共用依赖

SuperMimic使用原walk环境结构，其split实现仍复用stand_v2及stand的控制和接口；BC／SMP仍使用multimode的基础观测、动作和runner。MuJoCo双策略仿真直接导入split的配置／模型验证／切换模块，所以这些原路径保持。停止独立训练线不等于删除这些共用物理实现。

旧注册`chuanliantui`、`chuanliantui_stand`、`chuanliantui_stand_v2`、`chuanliantui_getup`、`chuanliantui_walk`均已删除。模型内部既有合同标识保留，以继续严格校验⑥⑦⑧及MuJoCo依赖的旧权重；它们不是可选训练task。

当前维护分支为`26_wheelleg`；H7分工、Git操作与Isaac保护规则仍按根AGENTS.md及[CONTRIBUTING](../../CONTRIBUTING.md)。
