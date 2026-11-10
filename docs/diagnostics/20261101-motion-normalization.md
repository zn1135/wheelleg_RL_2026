# 统一运动评估实现

本轮将[四项统一运动标准](../interfaces/chuanliantui-motion-acceptance-v1.md)接入运动评估与其直接调用方；起立不纳入四项评分。工作分支保持`26_wheelleg`，保留原有未提交改动。

## 实际修改

共同实现位于`scripts/agent/motion/assess.py`：速度／转向平均绝对误差≤0.10，停车稳定阶段水平速度峰值≤0.05 m/s，稳定阶段俯仰绝对值峰值≤5°。同时处理缺项、空窗口、提前终止、数值异常与非零指令未送达。

multimode主评估、CPU包线、摩擦包线、stand_v2 tracking、直线启停、转向／停车以及SuperMimic共七个采集／评估脚本接入共用判据。稳站旧刹车对比从轨迹重算运动结论，原报告通过数保留为明确的历史字段。任务各自的高度、承重、接触和航向等诊断保留原计算，不重新定义起立或双轮承重标准。

删除两个已被替代的独立入口，不提供旧路径包装：

| 原分类路径 | 合并后的入口 |
|---|---|
| `scripts/agent/multimode/check_multimode_tracking.py` | `python -m scripts.agent.motion.assess --selfcheck` |
| `scripts/agent/stand_v2/check_stand_v2_tracking.py` | `python -m scripts.agent.motion.assess --audit-tracking <测量目录>` |

源码保全见[本轮前快照](../../scripts/archive/20261101-motion-normalization/README.md)，共457份源码／配置文件逐字节校验。原始平铺路径的连续迁移关系见[迁移表](../../scripts/agent/MIGRATION.md)。

## 使用

从仓库根目录，使用`mjlab_training/.venv/bin/python`：

```bash
mjlab_training/.venv/bin/python -m scripts.agent.motion.assess --selfcheck
mjlab_training/.venv/bin/python -m scripts.agent.motion.assess \
  --measurement <已有测量目录> --output <新结果.json>
mjlab_training/.venv/bin/python -m scripts.agent.motion.assess \
  --audit-tracking <原stand_v2测量目录> --initial-reference <原初态.npz> --output <新审计.json>
```

`--measurement`复算保存轨迹，输出退出码0表示复算执行成功，是否达到指标查看逐样本`passed`；不以复算成功代表模型合格。`--audit-tracking`保留旧入口对模型、源码、轨迹、命令送达、初态／质量、统计及500Hz承重数据的全部核对；任一核对失败仍拒绝。`--selfcheck`只检查算术与接口，不启动策略或物理。

## 验证与证据

证据目录为[本轮验证产物](../../artifacts/imitation/motion-normalization-20261101-r1/)。正式入口：

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_script_merges
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_script_layout
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_script_entries \
  --isaac-python /home/zn/miniforge3/envs/wheellegged_py38/bin/python
```

- 边界与异常检查覆盖等号、超界、缺项、NaN、负指标、正负误差抵消、横向漂移、空窗口、提前终止、命令送达及CPU调用方。
- 对保存的SuperMimic轨迹160个稳定阶段，与归档的原函数独立比较四项数值，最大差约`9.93e-8`，在`rtol=1e-6, atol=1e-7`范围内。旧0.30门槛能通过而新0.10拒绝的情况另设检查，不要求不同门槛的结论保持相同。
- 保存的stand_v2 tracking共64例复算为13例通过四项指标，原数据审计保持通过；保存的SuperMimic V13 final共96例为0例通过。均为原记录窗口内的四项指标，不是完整任务通过。
- 原接触统计、SuperMimic预测／评分、计数舍入和历史来源检查继续通过；直线、转向及停车归约自检通过。
- 独立复查发现的停车外环选窗、恰在段末终止、转向指令未送达三项问题均补充先失败后通过的检查并修复；另补充瞬态数值异常不能被稳态选窗掩盖的检查。

## 计数与限制

本次改动将两个独立检查合为一个模块，净减少1个文件；并行任务新增了另一个脚本，所以计数时工作区辅助Python文件仍为320个。该并行任务的新脚本与其对预测采集器的修改均未由本任务改动。本次负责文件的总行数增加309行，主要来自格式适配、数据完整性与回归检查，具体工作区及本次范围分列见[counts.json](../../artifacts/imitation/motion-normalization-20261101-r1/counts.json)。不将本轮声称为整体代码行减少，也不把文件移动计入减少。

七个运行中保护入口的字节保持不变，训练／模型接口／物理参数、sim2sim生产源码与H7未修改；未启动新仿真、训练或设备采集，未提交／推送。当前只验证保存轨迹与有限程序检查，新增采集信号尚未做新物理运行验证。

旧直线记录只有四列命令数据，没有偏航速度，不能补造为零；统一判定明确报告缺项，未来采集追加偏航实测列。部分旧速度记录没有轨迹或源码校验值；一份旧转向记录缺少匹配的`ct_mjlab/envs/__init__.py`源码，认证复算拒绝，未跳过校验或标作历史归档完成。

稳定窗口沿用原记录声明；转向／启停段只按指令到位等待0.5秒，不按实际运动是否已经跟上挑选窗口。固定站立初态、统一指令序列、刹停时限、瞬态倾斜及完整覆盖范围尚未统一。因此不同协议的通过比例不能直接用于模型排序。
