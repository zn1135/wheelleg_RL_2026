# 仅维护⑥⑦⑧：清理完成记录

2026-11-01，当前分支`26_wheelleg`，分析基线`9f5eacd58ebc813fcf189b1e5cbb7d2ac778cf00`。用户已明确要求撤销mjlab①～⑤，并补充批准删除对应模型、训练日志和实验产物；7条经验已经逐条审批。未提交或推送。

## 保留与删除的边界

保留⑥SuperMimic、⑦行为克隆／BC初始化PPO、⑧SMP先验／SMP辅助PPO。当前环境注册名只有`supermimic`和`smp`。普通独立PPO训练在创建环境前即被拒绝；保留线PPO需通过SMP／BC配置启动，包含既有`--smp-weight 0`对照。

五个旧task注册、对应命令选项及独立入口均删除：`chuanliantui`、`chuanliantui_stand`、`chuanliantui_stand_v2`、`chuanliantui_getup`、`chuanliantui_walk`。旧模型内部已有的任务／接口合同字符串不改写，目录名也不作为活跃task注册。

⑥仍需要原walk环境；split实现复用stand_v2及stand基础；⑦⑧需要multimode的控制、观测、模型和PPO组件。MuJoCo双策略仿真还直接导入split的配置、验证及切换模块。因此保留这些必要共用实现的原路径，删除无调用的两个getup专用工厂包装，不改变其余物理／控制函数。

## 实际删除与减少

| 项目 | 结果 |
|---|---|
| 旧独立task | 删除5个 |
| 专属辅助文件 | 删除121个Python脚本和1个JSON辅助文件 |
| 共用部分 | 3个模块原样提取渲染／500Hz采集／静态几何函数，旧CLI删除；新增1个正式清理检查 |
| 本任务辅助Python净减少 | 117个；约减少29,033行，未把文件移动当作减少 |
| 工作区辅助Python计数 | 开始328个，结束212个；期间另一任务新增1个SuperMimic检查，未计入本任务新增 |
| 专属文档 | 删除22份，另清理项目实验记录207个旧线条目／章节；保留BC、SMP、SuperMimic及硬件／MuJoCo记录 |
| 对应模型、日志及产物 | 删除约54.0 GiB；逐文件记录SHA-256、字节数及路径 |

逐文件清单与统计：

- [实际删除清单](../../artifacts/maintenance/retain-678-cleanup-20261101/deletion-plan.json)
- [删除文件校验台账](../../artifacts/maintenance/retain-678-cleanup-20261101/deleted-files.jsonl)
- [删除结果](../../artifacts/maintenance/retain-678-cleanup-20261101/deletion-result.json)
- [代码与文件计数](../../artifacts/maintenance/retain-678-cleanup-20261101/counts.json)
- [原脚本路径的最终处置](../../scripts/agent/MIGRATION.md)

删除前的源码／文字记录快照为`artifacts/maintenance/retain-678-cleanup-20261101/before-source.tar.gz`，其校验清单为同目录`before-sha256.json`；仅作审计保全，不作为日常训练或旧线入口。没有备份54 GiB被批准删除的模型／产物。旧源码档案仍在，但缺失资产的旧实验不能宣称完整可复现。

## 明确保留的资产例外

24处文件或子目录有保留线／MuJoCo／部署的明确引用，继续原位保留；其引用来源列入删除清单的`dependency_evidence`。包括已部署／标定使用的3023、5849、6848、8846、9845、13842、5999等必要权重、来源快照、ONNX及校验向量，以及保留线原PPO对照所需的物理来源元数据。

这些资产不再拥有独立训练task；未保留它们同run中不需要的中间模型、TensorBoard日志和训练产物。`resources/`、`deployment/`、Isaac训练环境未清理；不操作仓库外的模型数据。

## 已批准的经验

最终文档：[已批准训练经验](../ai/training-lessons.md)。

- 保留E01、E04、E07、E09、E10、E11、E12。
- 舍弃E02、E03、E05、E06、E08，不纳入保留经验。
- [逐条审批凭据](20261101-retain-678-inventory.json)保留用户的明确选择。

经验审批不自动改变算法、奖励、物理参数或既有验收标准。

## 验证结果

[完整验证记录](../../artifacts/maintenance/retain-678-cleanup-20261101/verification.json)：

- 五个旧task的注册查询及CLI选择均拒绝；普通PPO入口拒绝后不创建环境。
- 保留环境及BC／SMP／SuperMimic导入通过；7个正式CLI的参数帮助与13个代表辅助入口检查通过，Isaac导出入口使用其Python3.8环境。
- 51份共用环境源码逐字节不变；另1份split配置模块仅删除两个无调用的getup工厂包装，其余AST完全相同。3个共用助手与原实现的函数／类AST一致。
- `sim2sim/`保护清单的55个文件逐字节不变；multimode、stand_v2和双策略三个MuJoCo入口的导入／`--help`通过，其跨目录依赖仍成立。
- 真实历史SuperMimic A700 actor在CPU按原算法／接口合同加载，权重与检查点逐项相同；课程使用接收器验证加载接口，未创建物理环境。
- 原训练源码快照仍逐字节验证；仅接受[明确的before／after迁移校验值](20261101-source-migration.json)，额外篡改会被拒绝。旧PPO对照只允许已知SuperMimic新增模块，未跳过源码校验。
- 保存的SuperMimic160个稳定阶段数值与原函数一致至既有容差；预测／评分与计数舍入检查通过。旧stand专属复算与检查随任务退出删除。
- 迁移表覆盖、全体活跃辅助源码的语法和内部模块导入检查通过；原有未提交续训工具保持本轮开始时字节，其他对话新增的SuperMimic文件保留。

正式检查命令见[构建与验证](../ai/build-test.md)。本轮训练更新0、物理步数0；未重新跑训练、行为仿真、板端或真机。导入、权重加载与参数检查不能替代完整行为验收。
