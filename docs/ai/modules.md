# 模块职责

最终验收现统一为GPU、共用运动门槛及报告，见[统一GPU验收](../interfaces/20261102-unified-gpu-acceptance.md)。主线统一入口为`scripts/mjlab.sh accept --task <任务> --recording <GPU记录> --output <新目录>`；起立单列，旧CPU流程仅保留历史含义。

维护SuperMimic、BC、SMP及独立复旦PPO行走适配，实验结论见[项目实验记录](project-log.md)，已批准经验见[训练经验](training-lessons.md)。

| 模块 | 职责 |
|---|---|
| `ct_mjlab/supermimic/` | 世界模型、估速器、策略和价值网络训练；自身train/play/export入口 |
| `ct_mjlab/bc.py`、`bc_preservation.py`、`scripts/train_bc.py` | 行为克隆、纠偏、技能保留与模型初始化 |
| `ct_mjlab/smp/`、`scripts/train_prior.py` | 示范特征、运动先验、SMP辅助PPO及BC初始化PPO |
| `ct_mjlab/teacher/`、`scripts/teacher.py`、`scripts/collect.py` | LQR／VMC对照与示范数据采集，服务BC和SMP |
| `ct_mjlab/envs/` | 注册`smp`、`supermimic`、`fudan_ppo_walk`、`fudan_ppo_step`、`fudan_ppo_jump`；旧目录保留必要控制／物理／模型接口依赖，不提供五个旧task |
| `ct_mjlab/envs/chuanliantui/fudan_ppo_jump/`、`scripts/agent/fudan_ppo_jump/` | 独立跳跃配方、实际轮底与接触、双策略越条GPU采集和评分，见[跳跃任务](../mjlab-fudan-ppo-jump.md) |
| `scripts/agent/supermimic/`、`imitation/`、`motion/` | 保留线检查、测量、评分与渲染 |
| `scripts/agent/shared/` | 原样提取的渲染、物理采集和静态几何助手，无旧训练CLI |
| `scripts/agent/real2sim/`、`sim2sim/` | 实采／标定及独立MuJoCo仿真；仿真实现与必要接口保持 |
| `wheel_legged_gym/` | Isaac历史环境，独立Python3.8；不在本次mjlab清理范围 |
| `resources/`、`deployment/` | 机器人资产与部署控制；本次不改物理参数或H7 |
| `deployment/rl-ynzz_33/` | Jetson现场6838/4000/3000部署源码、模型、启动与录制脚本及服务配置；当前入口和来源SHA见该目录README，不包含运行日志/编译缓存 |

## 2027 机器人与旧任务归组

旧任务注册集中在 `envs/chuanliantui/tasks.py`，所有相关环境实现都在 `envs/chuanliantui/` 内，旧导入由 `envs/_legacy_imports.py` 转向实际模块。新 `envs/2027_wheelleg/` 注册独立 `2027_wheelleg_walk`，资源位于 `resources/robots/2027_wheelleg/`。质量疑点、物理接口、检查与命令见 [2027训练说明](../mjlab-2027-wheelleg.md)。

## 共用依赖

`sim2sim/arena_policies.py` 在 RMUC 场地复用台阶模型切换，维护 Jump 独立五帧历史和落地交接；`sim2sim/rmuc2027_field.py` 从固定上游场地生成移除中央装置、补平安装孔的默认地图。[赛场入口与操作](../interfaces/20261115-rmuc2027-playground.md#rmuc比赛场地)。

`sim2sim/step_playground.py`为`rmuc2027.sh --terrain steps`拼装多高度原生MuJoCo场景，复用训练盒体并提供数字键出生点；`ManualStepPolicies`预载兼容的行走/台阶Actor，只切推理模型和高度指令，保留同一物理/PD/延迟/历史。不注册新任务、不改变训练/机器人参数。`check_native_steps.py`核几何/出生点/有限控制，`check_native_policy_switch.py`核切模连续性；GPU同状态接口仍复用`check_arena_native_interface --terrain steps`。

`scripts/agent/fudan_ppo_step/profile_speed.py`及`check_speed_optimization.py/check_speed_tensors.py/review_speed_optimization.py`只用于隔离性能/同输入正确性检查，禁止Adam更新。性能候选仅存`artifacts`快照/补丁，未接入生产环境或正在执行的远程冻结目录；完整训练速度、行为和模型迁移仍须按原正式入口验证。[范围与结果](../../artifacts/fudan_ppo_step/20261117-speed-optimization-01/README.md)。

`envs/chuanliantui/fudan_ppo_step/reference_motion.py`只计算训练期弱运动状态奖励；本任务`StepRunner`的平地模仿回调使用固定3000教师，不修改共享PPO或行走实现。接触许可、目标支撑与最终解除接触停车属于独立台阶配方，Actor输入/物理接口不扩展。[v7范围与检查](../../artifacts/fudan_ppo_step/20261116-contact-exploration-v7-01/README.md)。

固定200mm/转向v8仍在同一台阶任务内：`config.py/env.py`负责200mm固定等级与0.35m平地转向采样，`StepRunner`限制旧教师仅作用≤0.30m。`scripts/agent/fudan_ppo_step/prepare_fixed200_turn.py`显式迁移3200学习状态并提供真实GPU命令覆盖检查；一般`--resume`仍拒绝配方不一致，不修改共享行走训练线。

`envs/chuanliantui/fudan_ppo_step/`维护无视觉单级平台配置、盒体地形、逐物理步接触、课程和独立迁移/恢复约束；复用平地实体动作与PPO。`scripts/agent/fudan_ppo_step/`提供只读场地量测、零学习检查/吞吐以及GPU矩阵与离线评分。Actor接口保持25/125/6，任务配方另存，不互换两任务的`--resume`。见[任务与命令](../mjlab-fudan-ppo-step.md)。

SuperMimic使用原walk环境结构，其split实现仍复用stand_v2及stand的控制和接口；BC／SMP仍使用multimode的基础观测、动作和runner。MuJoCo双策略仿真直接导入split的配置／模型验证／切换模块，所以这些原路径保持。停止独立训练线不等于删除这些共用物理实现。

旧注册`chuanliantui`、`chuanliantui_stand`、`chuanliantui_stand_v2`、`chuanliantui_getup`、`chuanliantui_walk`均已删除。模型内部既有合同标识保留，以继续严格校验⑥⑦⑧及MuJoCo依赖的旧权重；它们不是可选训练task。

当前维护分支为`26_wheelleg`；H7分工、Git操作与Isaac保护规则仍按根AGENTS.md及[CONTRIBUTING](../../CONTRIBUTING.md)。

`ct_mjlab/fudan_ppo_walk/`新增固定复旦plane配方的实体接口适配：独立PPO/估速优化器、25/125/141+3输入、物理控制复用、独立命令/奖励/GPU最终验证；`scripts/agent/fudan_ppo_walk/`负责正式检查、有限预算预检、两引擎测量/评分/视频。规格见[复旦PPO](../mjlab-fudan-ppo-walk.md)。

复旦当前唯一环境/配置在`envs/chuanliantui/fudan_ppo_walk/`，算法/网络/存储/runner在`rsl_rl/`；旧`fudan_ppo_walk/`保留合同、模型/评估工具与导入兼容，不再维护另一套训练器。新入口及职责详见[结构整理记录](../maintenance/20261103-fudan-structure-alignment.md)。
