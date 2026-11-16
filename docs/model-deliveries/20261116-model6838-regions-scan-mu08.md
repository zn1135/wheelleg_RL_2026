# 模型归档：6838轮分区地形与Critic高度扫描

2026-11-16，按用户“提交并推送PT”准备归档当前MuJoCo体验使用的6838轮完整检查点。交付人Codex，接收人为用户，独立复查与H7负责人核对待确认；机器人chuanliantui_new_1实体闭链轮腿，任务fudan_ppo_walk。用途为仿真体验与实验留档，**行为保留标准未通过，不替换原3000实机基线**。本次不导出ONNX或修改固件。

## 模型与接口

| 项目 | 实际值 |
| --- | --- |
| 完整PT | [model_final.pt](../../artifacts/fudan_ppo_walk/models/20261116_model6838_regions_scan_slope5to20_mu08/model_final.pt) |
| 大小 | 1,444,809字节 |
| SHA256 | `8a42776d8acb0bfdbbc922375d19b594964432f4fb065b23fef3a1384c3d96ad` |
| 生效训练配置 | [effective-training-config.json](../../artifacts/fudan_ppo_walk/models/20261116_model6838_regions_scan_slope5to20_mu08/effective-training-config.json) |
| 来源与校验清单 | [manifest.json](data/20261116-model6838-regions-scan-mu08-manifest.json)，含完整配置、源码和物理资源指纹、检查点元数据及归档检查 |
| 原始PT | `logs/mjlab/fudan_ppo_walk/20261116_scan_slope5to20_mu08_resume5838_to6838_01/model_final.pt`，原位保留，副本逐字节相同 |
| 网络合同 | `fudan-ppo-walk-physical25-history125-critic144-v1` |
| 复位与历史 | `fudan-physical-pd-centred-warmup-noisy5-clipped-history-v3` |
| 输入输出 | Actor当前25＋历史125→6动作，含3维encoder估速；Critic环境141＋估速3；77点地形扫描只供训练Critic |
| 累计学习状态 | 6838完整轮、1,344,589,824采样转移、双Adam各136760步，partial_update=false，保留CPU/CUDA RNG |
| 控制与物理 | 腿P20/D1、轮P.1/D0；积分1ms、PD5ms、策略10ms，机械轮、原闭链/气弹簧和限矩 |
| 交互接口 | [分区MuJoCo说明](../interfaces/20261115-rmuc2027-playground.md#分区地形交互)，观测/动作/周期不因本次归档改变 |

训练配方为32×32m平地50%、5/10/15/20°金字塔坡25%、±1cm崎岖25%，实际轮地滑动摩擦固定0.8；参数/噪声全幅、动作和传感各10～20ms、机身各轴±3N持续外力/外矩0、7秒/.3m/s短推扰。命令vx最高3m/s、yaw±10rad/s，|yaw|>2时vx=0。原17项奖励、高度幅值1/方差.001保持，碰撞项仍为占位零值，坡地允许真实机身接触。精确参数以清单和训练配置为准。

## 代码与训练来源

仓库`https://github.com/zn1135/wheelleg_RL_2026.git`，分支`26_wheelleg`。归档准备基于`419ca8235d7dc06f2a8da2923dda4a77dfd5364a`，工作区另有训练和台阶任务改动，不纳入本次归档；发布提交以本记录所属Git提交为准。真实训练源码及逐文件SHA保存在原run的`source/`和`source_sha256.json`，其指纹纳入清单，未打包全部未提交的训练实现，不能仅用本次Git版本冒充完整训练版本。

本阶段从5838完整检查点继承网络、两Adam、RNG和计数，原SHA `96bcc67c3eb9453ff527e3bb33dac57c940e8c05a911760f8b63c9fb2ae77444`；新增1000轮到6838、两Adam各新增20000步，按预算停止。seed1、4096×48，实际命令、覆盖配置和恢复来源在清单的budget及training_config字段。本机训练日志和原始行为轨迹保留，未把它们全部上传Git。

共享XML及轮机械参数与训练指纹核对一致，完整资源SHA见清单。归档工具使用仓库mjlab虚拟环境；原Python/依赖版本未在本次独立锁定，不以归档环境版本代替训练版本证明。部署侧未参与本次PT归档，ONNX/板端代码、时序和真机核对不适用本次交付，H7负责人核对仍待确认。

## 分阶段验证

| 阶段 | 结果及范围 |
| --- | --- |
| 训练完成 | 原预算审计通过，6838/双Adam136760完整，未追加训练 |
| PT复制与加载 | 原文件和副本SHA一致、严格加载成功、网络与两优化器有限、固定输入输出形状通过；新增学习0 |
| 归档路径回放 | 默认分区入口读取归档副本完成1秒/100策略步，状态有限、无错误，实际接触摩擦0.8；不作行为验收 |
| 已有GPU对照 | 同新地形/μ.8，源5838→6838平地30/47→33/47但丢原成功1；±10rad/s由4/6→1/6；地形完整通行1/64→1/64，61/64停车速度峰超标，**不采纳为基线** |
| 已有原生接口 | 100组同态GPU控制/观测对照、六区域有限性和键盘操作检查通过；平地5秒末秒实际自转10.0184rad/s仅单个短例 |
| 完整987／起立／真实硬件 | 未执行，不从子集或原生体验迁移结论 |
| ONNX／板端／固件 | 未导出、未接入、未烧录；H7负责人待确认 |
| Isaac | 不适用，独立mjlab任务 |

已有结果和原始证据见[项目实验记录](../ai/project-log.md)。本次归档检查在`artifacts/fudan_ppo_walk/20261116-model6838-archive-check-01/`，本机原完整GPU效果在`artifacts/fudan_ppo_walk/20261116-slope5to20-mu08-6838-effect-01/`；二者用途分别记录，不能以原生数值检查代替GPU行为通过。

## 使用与回退

```bash
# 分区模式默认读取仓库归档PT，不依赖本机历史logs目录。
scripts/rmuc2027.sh --terrain regions
# 平地直接体验小陀螺，目标vx=0/yaw=10rad/s；X停车。
scripts/rmuc2027.sh --terrain regions --spawn -4 4 --spin
```

体验保持名义物理/无噪声，关闭训练外力与短推扰，保留双延迟；配置和训练条件不同，均写入运行报告。仅区域体验默认路径改为同一6838权重的归档副本，原RMUC3000和历史模型保持。下一步优先做零速奖励的单项对照并要求原成功保留、坡地停车改善；本次不自动启动训练。
