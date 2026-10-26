# 轮角39／82接入SMP PPO v4

本轮接通当前BC到PPO的初始化、保存／恢复及运行入口，**没有训练PPO，也没有改变奖励权重**。目的是让PPO从已会起立和平衡的策略继续改善停车，教师停车通过率不再作为硬性前置条件。

| 项目 | 内容 |
|---|---|
| 日期、作者、复查 | 2026-10-25任务，Codex执行；独立成员复查及H7负责人核对待确认 |
| 机器人／任务 | 共享XML闭链串联腿，`chuanliantui`，正常起立和零速站稳 |
| 训练仓／版本 | `https://github.com/zn1135/wheelleg_RL_2026.git`；当前detached HEAD `2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`及既有未提交改动；不创建／切换分支，无Git提交或推送 |
| 旧观测／实验定义 | `chuanliantui-huananhu-multimode-39-82-v1`／`chuanliantui-smp-ppo-normal-ground-stand-unit-v3`，见[单位尺度SMP](20261025-smp-unit-reward-v3.md) |
| 新观测／实验定义 | `chuanliantui-huananhu-wheel-angle-39-82-handover-v2`／`chuanliantui-smp-ppo-normal-ground-stand-wheel-handover-v4` |
| 权威实现 | `ct_mjlab/smp/ppo.py`的`policy_observation_contract`、`configure_smp`及`SmpOnPolicyRunner`；`scripts/train.py`、`play.py`、`eval.py`、`check.py`按模型元数据选择观测 |
| 部署仓 | 同仓`deployment/H7_RL`及USB工程，本轮均未修改、未生成或交付固件；来源及版本沿用[部署接口记录](../deployment-contract.md)，不声称同步 |
| 实际运行源码 | [本轮报告、快照及源码哈希](../../artifacts/ppo-wheel39-interface-20261025-v1/README.md)；脏工作区不能只用HEAD复现 |

## 观测、动作与时序

| 项目 | 旧v3 | 新v4 |
|---|---|---|
| actor／critic／动作 | 39／82／6，float32 | 相同 |
| 轮位置槽 | actor和critic索引14、15固定为0 | 左右轮连续关节角减各自原点，rad、缩放1、裁剪±100、不回绕；沿用[首次站稳清零v2](20261025-wheel-angle-handover-reset.md) |
| 轮角参考／历史 | 不使用轮角位移 | 回合复位初始化；首次`ground_start && recovered`重锁一次，清轮位置两列延迟历史，其他历史和动作FIFO保留；每个环境独立 |
| 其他观测／历史 | 原multimode字段、顺序、噪声、延迟及上一动作 | 保持；没有新增LQR内部状态、参考估计器或教师依赖 |
| 前向／yaw | 机体+x前向，偏航角速度命令 | 保持；SMP任务全程零速、零转向 |
| 动作 | `[lf0,lf00,lfwheel,rf0,rf00,rfwheel]`；腿×0.5为相对默认角，轮×10为速度目标 | 保持PD、限幅、补偿、默认角及连续轮角符号 |
| 时序 | 2ms物理／PD、10ms策略，训练腿轮各自独立抽取1或2个物理步 | 保持；20环境检查使用真实训练随机化，原40组名义检查覆盖2／4ms四种组合 |
| 初态／接管 | 正常趴地姿态库、高度0.25m、零初速度，前0.2秒主动零出力，物理弹簧继续生效 | 保持；倒置终止自动复位，不训练倒置恢复 |
| 物理／奖励 | 共享XML及原任务奖励＋单位尺度SMP权重1 | 保持；XML SHA `cac8061de2850602bfd1421fc856bf217c4e5040a11c8f5d33cf1abe61d5f0cf` |

SMP的30维运动特征和冻结先验保持，轮角输入属于动作策略，不是给SMP先验新增输入。`reward_manager`仍统一乘一次dt。

## 模型初始化和恢复

`train --smp-prior ... --bc-init ...`先读取BC元数据，再创建对应观测环境。新v4只接受当前首次站稳清零v2的BC；旧零轮角39维BC进入旧v3，旧轮角v1及40／41维BC拒绝接入。XML、动作实现哈希、轮角定义和网络结构均检查，不能凭维度相同混载。

只复制BC actor及其0.05动作标准差。critic和PPO优化器从新状态开始，不导入BC的Adam；BC累计20000次是历史监督训练次数，不是PPO轮数。当前初始化来源见[BC交付记录](../model-deliveries/20261025-bc-aligned-reference-local5000.md)。

新PPO检查点在原控制合同、actor／critic／优化器及课程进度外保存新观测／实验定义、轮角描述、XML哈希、BC来源及数据指纹、固定SMP评分统计和评分RNG。恢复时逐项核对。`--resume`恢复完整PPO状态，不能用BC文件；`--bc-init`与`--resume`互斥。

PPO检查点不保存物理世界快照。续训／回放开始新回合，轮角原点、首次站稳锁存、观测历史、动作FIFO和SMP窗口由环境复位生成；不能把旧回合原点套在新初态上。

`play`、`eval`、`check`也需同时指定`--smp-prior`和PPO检查点，自动选择旧v3或新v4。无`--smp-prior`时普通PPO保持原流程。独立CPU／ONNX／H7尚未适配新轮角合同，旧导出器明确拒绝，不能把v4文件当普通39维策略部署。

## 使用

本轮没有PPO预算。以下命令需以后明确本轮预算后运行；不会把BC累计次数或本次检查夹具当训练预算。

```bash
scripts/mjlab.sh train \
  --smp-prior logs/mjlab/chuanliantui_smp_prior/20261025_student170_remote10000-v2/prior_10000.pt \
  --bc-init logs/mjlab/chuanliantui_bc/20261025_aligned_reference_local5000-v1/bc_5000.pt \
  --num-envs 2048 --iterations "${CT_PPO_STEPS:?先设置本次授权的PPO轮数}" \
  --run-name wheel39-first

# 另一终端；本轮未启动TensorBoard
mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_huananhu_multimode_smp_wheel_handover_v4 --port 8081
```

本轮未改变根初始分布，无需新增初态预检门槛；已有20环境有限检查及原40组名义回放仅证明其各自范围。后续训练保存至上方v4日志目录；完整续训以同先验及`--resume <model_*.pt>`替换`--bc-init`。

## 验证与交付边界

正式检查脚本为`check_wheel39_ppo.py`（真实20环境采样／returns与恢复）、`check_wheel39_ppo_entries.py`（真实入口，learn前拦截）、`eval_wheel39_ppo_init.py`（原40组真实PPO actor回放与奖励明细）。逐项结果、日志、命令、视频及失败检查记录见[本轮报告](../../artifacts/ppo-wheel39-interface-20261025-v1/README.md)。

交付的是当前工作区的仿真侧接口代码和零更新证据，未训练或导出新策略，不证明PPO改善停车或超过LQR；完整训练随机化行为、移动、10ms压力、板端数值／最坏控制时序及真机未验。H7负责人未来需核对轮角原点、共享站稳事件和历史清理，状态待确认。本轮不触及固件、物理参数和旧训练产物；回退可用本轮修改前源码快照加原BC／先验组合，禁止静默换合同。
