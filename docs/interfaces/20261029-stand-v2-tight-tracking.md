# 稳站v2：B配方的独立窄核速度奖励

| 项目 | 内容 |
|---|---|
| 日期／作者／复查 | 2026-10-29；Codex实现与自查；无独立复查 |
| 机器人／任务 | chuanliantui_stand_v2；硬件与共享XML保持 |
| 旧接口 | [A激活期权重v5](20261029-stand-v2-active-tracking-weight.md)、[速度课程](20261029-stand-v2-speed-slew.md) |
| 新recipe_version | `stand-v2-upright-first-tracking-active-weight1-tight-v6` |
| profile／command | 仍为`chuanliantui-stand-v2-isaac32-speed-slew-v6`／`stand-v2-post-stand-half-vx15-slew-v2` |
| 训练仓／分支／基点 | Wheel-Legged-Gym；26_wheelleg；952e4260aedf04e28fdbf366d5cb1c5562c75263＋已有脏工作区；未提交推送；远程URL未重新核对 |
| 部署仓／SHA／负责人 | 未读取或改动H7；版本、负责人及原有接口适配核对**待确认** |
| 权威实现 | stand_v2的mdp.py、rewards.py、env_cfg.py、runner.py、export.py；experiment_stand_v2_phase_mix.py |

**仅实现与检查，无训练**。用户已完成A训练核对，才开始改共享源码；按最新要求，未来续训父模型改为A的新4999（不是旧速度课程4999）。

## 奖励差异与配置

新增`StandV2CommandCfg.tracking_tight_term=False`。须同时启用`tracking_weight_while_active`，否则配置构建和实验CLI报错。默认`tracking_tight_kernel_denominator=0.02`、`tracking_tight_weight=0.5`，后者也写入`WEIGHTS['tracking_lin_vel_tight']`。

跟踪激活时，独立项原值为`exp(-(u-vx)^2/0.02)`；u为实际下发的斜坡命令`state.command[:,0]`，vx与主项一致，取`root_link_lin_vel_b[:,0]`。未激活时为0，覆盖50%整回合零速角色、起立未就绪和恢复；沿用准备期与非有限状态的统一奖励屏蔽。RewardManager只乘一次权重与dt，e=0时加权奖励率0.5、每10ms为0.005。

宽核分母0.25、主项×5及其他19项算法保持。新项不并入宽核，避免宽核在原值5处限幅后丢掉新增奖励；新项自身范围0～1，不做该限幅。奖励仍使用实际物理步执行的命令/active，未调整命令、失稳或奖励更新顺序。

关闭时默认配置字典、RewardState返回字典和RewardManager均只有原19项，名称、顺序及`Episode_Reward/*`日志键保持。命令配置build只在打开B时把窄核项加到环境奖励配置；命令管理器先于奖励管理器创建，故原生B环境在RewardManager初始化前就注册第20项。

## checkpoint、迁移及日常入口

`task_contracts`新增tight参数；仅B额外记录`tracking_tight_config={kernel_denominator: 0.02, weight: 0.5}`。save写入，load在加载权重之前严格校验recipe与参数。A及旧版本关闭B时保持原契约字段；跨版本原生加载拒绝，参数不匹配也拒绝。

- 3999、旧速度4999、A v5父模型：先按父契约恢复Actor/Critic、两个优化器与课程，再显式启用B并重建RewardManager；发生在40秒预运行之前，没有训练回合累计奖励需要迁移。3999沿用已有的速度课程迁移；旧速度4999和A保留其课程状态。
- B父模型：构建环境前读取B配置并打开开关，加载前已注册第20项，再原生加载。漏传任一必要CLI开关拒绝继续。
- 实验入口加`--tracking-tight-term`，run名追加`_tight`，与A后缀合为`_activew1_tight`。轮数由用户填写，2048环境，无可视化。
- play显式加`--post-stand-tracking --speed-curriculum --tracking-weight-while-active --tracking-tight-term`；导出自动识别B。瞬态和独立零速评估按checkpoint元数据选择B，保持原零速初态/RNG构建顺序。

## 验收统计

保持原MAE≤0.30、倾角/接触/急停判据与全部课程晋级门槛。新增保持段MAE≤0.10计数/比例，并按正负各速度报告稳态误差均值`vx-u`及平均绝对误差。比例分母是该组全部保持段，未就绪/失稳/样本不足不从分母剔除；误差均值只取完整门控且具有足够稳态样本的段，同时报告有效段数。速度精度比例不替代姿态/接触等行为通过率。

三模型3999/A/B、a=1与0.5分别配对，初态/物理参数与重量一致性检查保留；零速组单独复算原v3.1，允许慢移。两斜率同seed不作为两个独立评估种子。尚无B训练后的行为结论。

## H7与部署接口

**H7不受本次奖励改动影响**：Actor32／历史160／Critic89／动作6，观测命令槽6:9及缩放、五帧历史顺序/更新、500Hz物理/PD与100Hz策略、实体电机顺序/PD/限幅、初态和物理均保持。ONNX仍为float32/opset13，`observation_history[1,160] → actions[1,6], velocity[1,3]`，训练奖励不进入图。

H7无需同步本次奖励代码或新增协议字段；原v6历史、模式及真实时序适配仍**待确认**。本轮不交付新训练模型；接口检查不能替代部署负责人核对、CPU sim2sim或真机验收。

## 证据

[完整文件清单、检查命令与退出码、A4999续训及训练后配对命令](../../artifacts/stand-v2-tight-term-20261029/README.md)。interface-only检查点只用于冻结兼容性验证，不是新训练模型。
