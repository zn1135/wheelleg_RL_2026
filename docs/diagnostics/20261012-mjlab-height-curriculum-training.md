# A2500高度课程两组500轮续训与评估

**2026-10-13收尾：两组训练、四个检查点评估与视频全部完成，协调器于北京时间2026-10-12 23:58:53退出0。两组均退化，验收不通过。** `curriculum` 3000轮普通64/地面58，丢失A原成功地面样本27、29；`curriculum_knee` 3000轮普通和地面均0/64。0.30 m高度跟踪全部0/64，最终课程L均0.24。本次收尾只复核已保存产物并补齐报告，没有追加训练。

2026-10-12授权范围：curriculum／curriculum_knee，从同一A2500独立追加500次更新，各4096环境、seed42，远端并行，预算累计3000轮，到此停止；不扩组、不追加。接口与规则见 [高度课程接口](../interfaces/chuanliantui-mjlab-height-curriculum-v1.md)。

源模型：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261012_122334_386753_recovery_factors/baseline/model_2499.pt`；SHA256 `706c8387cba4815b491be3f5914a5951881c2966edabbd5fbb90d0da9c9d3dbd`，本地及远端均核对一致。历史源码、资产与logs保留，未执行Git操作。

## 已完成的训练与门禁

已验证：本地RTX4060两组各16环境64步有限、数值失败0；同命令真实状态下仅宽高度项输出变化，速度命令依赖项单独隔离，膝项来自原公式。课程门槛、连续2秒、下降清零、checkpoint课程状态重载通过。

远端RTX4090，Python3.12／mjlab1.6.0／torch2.14.0+cu130／MuJoCo3.11.0；发布目录 `releases/20261012-height-curriculum-v1`、campaign `jobs/20261012-height-curriculum-500-v1` 均核验不存在后新建。远端16/4096检查及完整迁移通过，actor/critic/Adam逐项相等，初始2500更新、L=0.21、Adam学习率1e-5；自身checkpoint重载通过。A同初态复现普通64/地面60，验收v2成功集合保留，数值失败0。

本地原始输出 `artifacts/height-curriculum-training-20261012/`。早期本地检查和发布包版本保留；最终实际发布采用release-v4.zip，SHA256 `bd0485b76e560687e6b572cfbdaf8ff598a21f309af33be75d3d07e6a5340090`，发布内容127个文件逐项哈希并设为只读。新增脚本独立，不修改现有诊断与评估脚本。冻结前修正新评估脚本重复字段语法错误；首次完整复现被配对门禁拦截，原因是错误选用缺少v2标识的旧报告，换为已有 `balance-acceptance-v2-20261012/baseline/evaluation.json` 后完整重跑退出0。失败日志/产物保留，未绕过门禁或改判据。

北京时间2026-10-12 23:18:10协调器启动，启动命令退出0，PID472。新训练run根：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261012_231810_449023_height_curriculum/`，子目录curriculum与curriculum_knee独立并行。两组各500更新已完成，累计3000；协调器记录完成时间分别23:37:25.510826、23:37:32.392208，训练退出码均0，最终L均0.24。四次完整评估、四次视频渲染随后串行完成，协调器最终完成时间23:58:53.007449，worker.exit-code=0、failures为空。训练预算结束，没有重启或追加。

## 训练前门禁与课程记录

| 检查 | curriculum | curriculum_knee |
|---|---|---|
| 本地16环境×64步 | 全有限，数值失败0 | 全有限，数值失败0 |
| 远端16环境×64步 | 全有限，数值失败0 | 全有限，数值失败0 |
| 远端4096环境×64步 | 全有限，数值失败0 | 全有限，数值失败0 |
| 同状态、同命令奖励隔离 | 仅宽高度公式变化 | 上述＋原膝项−0.5 |
| actor／critic／Adam逐项相等 | true／true／true | true／true／true |
| 初始累计更新／等级／Adam LR | 2500／0.21／1e-5 | 2500／0.21／1e-5 |
| 自身checkpoint重载 | 通过 | 通过 |

| case | 累计更新 | 控制步 | L变化 | 达标比例 | 连续保持秒数 |
|---|---:|---:|---|---:|---:|
| curriculum | 2550 | 61201 | 0.21→0.22 | 0.934326 | 3.14 |
| curriculum | 2600 | 62401 | 0.22→0.23 | 0.914307 | 7.04 |
| curriculum | 2650 | 63601 | 0.23→0.24 | 0.855713 | 10.94 |
| curriculum_knee | 2550 | 61201 | 0.21→0.22 | 0.938232 | 3.18 |
| curriculum_knee | 2600 | 62401 | 0.22→0.23 | 0.914551 | 7.10 |
| curriculum_knee | 2650 | 63601 | 0.23→0.24 | 0.872070 | 10.98 |

两组之后没有升级。末控制步72000的达标比例分别0.417236／0.485107，连续计时0；最后一级已训练350次更新，50更新驻留门槛早已满足，尚未满足的是2/3比例连续2秒。不能将课程L=0.24等同已跟踪0.24 m，实际命令区间为0.22～0.24；等级变化只影响后续重采样。完整HEIGHT_LEVEL保留在两组training.log、training-completion.json和checkpoint历史中，收尾时逐条对照一致。2601轮checkpoint的L为0.23，3000轮为0.24。

## 实际执行命令

工作目录为冻结发布目录。所有训练子进程显式使用 `PYTHONPATH=<release>/mjlab_training/src`、`MUJOCO_GL=egl`、`TZ=Asia/Shanghai`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`；未启动TensorBoard。

```bash
# 实际由协调器并行执行，两组各500更新；完整参数另保存在launch/status JSON
runtime_python=/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python
release=/root/gpufree-data/wheelleg-remote/releases/20261012-height-curriculum-v1
source=/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261012_122334_386753_recovery_factors/baseline/model_2499.pt
run_root=/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261012_231810_449023_height_curriculum
PYTHONPATH=$release/mjlab_training/src MUJOCO_GL=egl TZ=Asia/Shanghai OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $runtime_python $release/scripts/agent/run_chuanliantui_height_curriculum.py --case curriculum --resume $source --run-dir $run_root/curriculum --iterations 500 --num-envs 4096 --seed 42
PYTHONPATH=$release/mjlab_training/src MUJOCO_GL=egl TZ=Asia/Shanghai OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $runtime_python $release/scripts/agent/run_chuanliantui_height_curriculum.py --case curriculum_knee --resume $source --run-dir $run_root/curriculum_knee --iterations 500 --num-envs 4096 --seed 42
# 仅提供，不启动：
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/tensorboard --logdir=/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261012_231810_449023_height_curriculum --port=8080
```

以上为历史实际训练命令，预算已经完成，不再执行。“通过检查”不等于训练效果或高度验收通过。

## 回放协议与证据来源

四个候选检查点各做6种高度×2种初态，共48项扫描。每项64环境、20秒、确定性actor、零速度、名义物理、关闭噪声和动作延迟，末5秒250帧统计；评估关闭课程升级。远端RTX4090执行，Python3.12／mjlab1.6.0／MuJoCo及mujoco-warp3.11.0／Torch2.14.0／Warp1.17.0。本次收尾在本地用指定mjlab venv读取已有GPU回放数据，没有重新运行物理或学习。

普通和地面qpos/qvel指纹分别为`daf5166ee2ff3243f897bc45ff054ecd932d182bb8edd688b2b8017fabfda209`、`ac1c67ade95c9153881bf87510d045b899a51884f0695b0b04ca839cf1c26027`，每条命令独立恢复，与A2500相同。A对照采用已有[高度扫描](../../artifacts/height-response-a2500-20261012/scan-v2/summary.json)，退化比较采用[验收v2基线](../../artifacts/balance-acceptance-v2-20261012/baseline/evaluation.json)。正式学习前，完整迁移的初始模型在同一评估入口复现A普通64／地面60，见[source-replay-v3](../../artifacts/height-curriculum-training-20261012/source-replay-v3/evaluation.json)。

高度为根节点对地高度；均值包括数值正常但平衡失败的全部64份样本。稳定平衡与高度跟踪使用原验收v2组合判据，末5秒至少95%帧通过。读取物理转移末尾时，仅失效接触缓存后同步读取，不额外步进或滚动history。全部48项数值失败为0。数值有限不代表静止稳定。

斜率为六条高度命令的OLS拟合，单位m/m，理想值1。“共同成功”指**同一模型在六条命令下均稳定的固定样本集合**，不是不同模型共有集合；没有共同成功样本时斜率不适用，不能记为0。

## 0.30 m汇总与逐样本退化否决

| 模型／累计更新 | 初态 | 稳定平衡 | 高度跟踪 | 末5秒平均高度 m | 全体斜率 | 共同成功数／斜率 | 丢失A成功样本 | checkpoint L m |
|---|---|---:|---:|---:|---:|---|---|---:|
| A2500 | 普通 | 64/64 | 0/64 | 0.200775 | 0.006307 | 64／0.006307 | 对照 | 不适用 |
| A2500 | 地面 | 60/64 | 0/64 | 0.199780 | 0.012841 | 59／0.006298 | 对照 | 不适用 |
| curriculum 2601 | 普通 | 64/64 | 0/64 | 0.200953 | 0.006510 | 63／0.006511 | 无 | 0.23 |
| curriculum 2601 | 地面 | 59/64 | 0/64 | 0.199140 | 0.006064 | 58／0.006517 | 29 | 0.23 |
| curriculum 3000 | 普通 | 64/64 | 0/64 | 0.201004 | 0.015019 | 64／0.015019 | 无 | 0.24 |
| curriculum 3000 | 地面 | 58/64 | 0/64 | 0.199154 | 0.013802 | 58／0.015693 | 27、29 | 0.24 |
| curriculum_knee 2601 | 普通 | 31/64 | 0/64 | 0.201297 | 0.006026 | 0／不适用 | 33份，集合见下文 | 0.23 |
| curriculum_knee 2601 | 地面 | 23/64 | 0/64 | 0.199457 | 0.004407 | 0／不适用 | 37份，集合见下文 | 0.23 |
| curriculum_knee 3000 | 普通 | 0/64 | 0/64 | 0.199656 | −0.001582 | 0／不适用 | A全部64份 | 0.24 |
| curriculum_knee 3000 | 地面 | 0/64 | 0/64 | 0.196409 | −0.004607 | 0／不适用 | A全部60份 | 0.24 |

**四个候选checkpoint均为`regressed_rejected`：退化，验收不通过。** 全部新增成功集合为空；即使有新增成功也不能抵消原成功丢失。A本身只是低位平衡对照，也未达到0.30 m目标，不是部署合格策略。

`curriculum_knee` 2601轮丢失集合（零基编号）：

- 普通33份：3、5、8、10、11、12、15、18、19、22、24、25、28、29、32、34、35、39、40、43、44、45、46、47、48、49、50、51、52、55、59、60、62。
- 地面37份：0、3、4、5、8、9、11、13、15、16、19、21、22、23、24、25、26、27、28、29、30、32、33、34、35、39、41、45、46、47、49、50、52、55、58、60、62。

3000轮普通丢失0～63全部；地面丢失A原60份，即0～63除20、42、53、57。机器可读的完整集合和原始分项比例保留在每份evaluation.json及[收尾复核](../../artifacts/height-curriculum-training-20261012/closeout-20261013-v2/verification.json)。

## 失败分项与膝限位证据

下表为0.30 m下**单项**末5秒成立比例不足95%的样本数，原因可重叠。各单项分别通过不能代替组合条件同时通过，因此单项失败数不能相加或直接当组合失败数。

| 模型／更新 | 初态 | 双轮接地／每轮承重 | 总轮负载 | 非轮支撑 | 机身接触 | 线速度 | 角速度 | 平衡组合失败 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| curriculum 2601 | 普通 | 0／0 | 0 | 0 | 0 | 0 | 0 | 0 |
| curriculum 2601 | 地面 | 5／5 | 5 | 5 | 2 | 0 | 0 | 5 |
| curriculum 3000 | 普通 | 0／0 | 0 | 0 | 0 | 0 | 0 | 0 |
| curriculum 3000 | 地面 | 6／6 | 6 | 6 | 2 | 0 | 0 | 6 |
| curriculum_knee 2601 | 普通 | 0／0 | 0 | 0 | 0 | 0 | 31 | 33 |
| curriculum_knee 2601 | 地面 | 5／5 | 5 | 5 | 2 | 0 | 29 | 41 |
| curriculum_knee 3000 | 普通 | 0／0 | 64 | 0 | 0 | 64 | 64 | 64 |
| curriculum_knee 3000 | 地面 | 6／6 | 64 | 6 | 2 | 58 | 58 | 64 |

全部8行近水平单项失败数为0，高度单项失败数为64，数值失败0。膝组3000轮**不能因近水平或轮端接地就称稳定平衡**：普通64份均违反总负载、线速度和角速度条件；地面58份有速度失败，全部64份违反总轮负载。当前保存的分项比例没有总负载的上下界分解，不能据此断言全部属于欠载或超载。

原生`mjCNSTR_LIMIT_JOINT`行计数及`JᵀF`广义反力矩由每项末5秒的250×64×2数组复算；只读取已有约束，不施加辅助力。以下力矩为**带符号均值**，按lf1／rf1顺序；不是平均绝对值或峰值。

| 模型／更新 | 初态 | 膝限位激活比例 lf1／rf1 | 广义反力矩均值 N·m lf1／rf1 |
|---|---|---|---|
| curriculum 2601 | 普通 | 100%／100% | +9.538／−17.173 |
| curriculum 2601 | 地面 | 100%／100% | +4.911／−11.845 |
| curriculum 3000 | 普通 | 100%／100% | +17.266／−21.050 |
| curriculum 3000 | 地面 | 100%／100% | +12.165／−14.062 |
| curriculum_knee 2601 | 普通 | 100%／100% | +17.774／−16.717 |
| curriculum_knee 2601 | 地面 | 100%／100% | +12.417／−11.434 |
| curriculum_knee 3000 | 普通 | 99.9375%／100% | +17.566／−16.394 |
| curriculum_knee 3000 | 地面 | 99.9250%／100% | +10.148／−9.547 |

**已验证：** 两组500轮并未离开约0.20 m低位，膝限位仍几乎全帧激活。`curriculum`普通斜率由A的0.00631升至0.01502，但六命令均值跨度仍仅约2.14 mm；0.30 m时平均高度比A普通增加约0.23 mm，地面下降约0.63 mm，不能称高度目标改善。膝组损失原有静止平衡能力，3000轮两种初态均无稳定样本。

**推断：** 宽高度奖励和这段高度课程在本批预算内不足以使策略脱离低位吸引域；膝惩罚组的平衡损失与持续限位、速度条件恶化同时出现，可作为后续诊断线索。

**未能验证：** 单seed、64初态不能支持膝惩罚必然造成退化的普遍因果结论，也没有验证梯度、探索分布、气弹簧或接触后端各自的因果贡献。没有本批完整动作／速度／总负载逐帧原始数组，不补造其峰值或失稳机制。训练课程采用随机化、混合初态和采样高度，固定0.30 m确定性静止回放更严格；训练L推进不能替代回放验收。不得因预算完成或课程升级称模型更好，也不得由本报告自动追加训练。

## 全部48项高度扫描

均值为末5秒全部64样本；表中“跟踪”已经包含平衡组合。低命令0.20／0.22 m成功不代表0.30 m成功。每行原始高度、膝约束、单样本分项比例见对应`normal-h*.json`／`ground-h*.json`。

| case | 累计更新 | 初态 | 命令 m | 实际高度 m | 稳定平衡 | 高度跟踪 |
|---|---:|---|---:|---:|---:|---:|
| curriculum | 2601 | 普通 | 0.20 | 0.200258 | 64/64 | 64/64 |
| curriculum | 2601 | 普通 | 0.22 | 0.200432 | 64/64 | 64/64 |
| curriculum | 2601 | 普通 | 0.25 | 0.200640 | 64/64 | 0/64 |
| curriculum | 2601 | 普通 | 0.27 | 0.200764 | 64/64 | 0/64 |
| curriculum | 2601 | 普通 | 0.30 | 0.200953 | 64/64 | 0/64 |
| curriculum | 2601 | 普通 | 0.34 | 0.201179 | 63/64 | 0/64 |
| curriculum | 2601 | 地面 | 0.20 | 0.198497 | 59/64 | 59/64 |
| curriculum | 2601 | 地面 | 0.22 | 0.198662 | 59/64 | 59/64 |
| curriculum | 2601 | 地面 | 0.25 | 0.198851 | 59/64 | 0/64 |
| curriculum | 2601 | 地面 | 0.27 | 0.198964 | 59/64 | 0/64 |
| curriculum | 2601 | 地面 | 0.30 | 0.199140 | 59/64 | 0/64 |
| curriculum | 2601 | 地面 | 0.34 | 0.199358 | 58/64 | 0/64 |
| curriculum | 3000 | 普通 | 0.20 | 0.199583 | 64/64 | 64/64 |
| curriculum | 3000 | 普通 | 0.22 | 0.199844 | 64/64 | 64/64 |
| curriculum | 3000 | 普通 | 0.25 | 0.200423 | 64/64 | 0/64 |
| curriculum | 3000 | 普通 | 0.27 | 0.200616 | 64/64 | 0/64 |
| curriculum | 3000 | 普通 | 0.30 | 0.201004 | 64/64 | 0/64 |
| curriculum | 3000 | 普通 | 0.34 | 0.201719 | 64/64 | 0/64 |
| curriculum | 3000 | 地面 | 0.20 | 0.197888 | 59/64 | 59/64 |
| curriculum | 3000 | 地面 | 0.22 | 0.198031 | 58/64 | 58/64 |
| curriculum | 3000 | 地面 | 0.25 | 0.198578 | 58/64 | 0/64 |
| curriculum | 3000 | 地面 | 0.27 | 0.198774 | 58/64 | 0/64 |
| curriculum | 3000 | 地面 | 0.30 | 0.199154 | 58/64 | 0/64 |
| curriculum | 3000 | 地面 | 0.34 | 0.199806 | 58/64 | 0/64 |
| curriculum_knee | 2601 | 普通 | 0.20 | 0.200681 | 26/64 | 26/64 |
| curriculum_knee | 2601 | 普通 | 0.22 | 0.200453 | 28/64 | 28/64 |
| curriculum_knee | 2601 | 普通 | 0.25 | 0.201222 | 64/64 | 0/64 |
| curriculum_knee | 2601 | 普通 | 0.27 | 0.201256 | 64/64 | 0/64 |
| curriculum_knee | 2601 | 普通 | 0.30 | 0.201297 | 31/64 | 0/64 |
| curriculum_knee | 2601 | 普通 | 0.34 | 0.201350 | 0/64 | 0/64 |
| curriculum_knee | 2601 | 地面 | 0.20 | 0.199025 | 33/64 | 33/64 |
| curriculum_knee | 2601 | 地面 | 0.22 | 0.198790 | 31/64 | 31/64 |
| curriculum_knee | 2601 | 地面 | 0.25 | 0.199382 | 59/64 | 0/64 |
| curriculum_knee | 2601 | 地面 | 0.27 | 0.199419 | 59/64 | 0/64 |
| curriculum_knee | 2601 | 地面 | 0.30 | 0.199457 | 23/64 | 0/64 |
| curriculum_knee | 2601 | 地面 | 0.34 | 0.199480 | 0/64 | 0/64 |
| curriculum_knee | 3000 | 普通 | 0.20 | 0.200101 | 31/64 | 31/64 |
| curriculum_knee | 3000 | 普通 | 0.22 | 0.199554 | 2/64 | 2/64 |
| curriculum_knee | 3000 | 普通 | 0.25 | 0.201760 | 0/64 | 0/64 |
| curriculum_knee | 3000 | 普通 | 0.27 | 0.201328 | 0/64 | 0/64 |
| curriculum_knee | 3000 | 普通 | 0.30 | 0.199656 | 0/64 | 0/64 |
| curriculum_knee | 3000 | 普通 | 0.34 | 0.199911 | 0/64 | 0/64 |
| curriculum_knee | 3000 | 地面 | 0.20 | 0.197429 | 23/64 | 23/64 |
| curriculum_knee | 3000 | 地面 | 0.22 | 0.196775 | 0/64 | 0/64 |
| curriculum_knee | 3000 | 地面 | 0.25 | 0.199187 | 0/64 | 0/64 |
| curriculum_knee | 3000 | 地面 | 0.27 | 0.199276 | 0/64 | 0/64 |
| curriculum_knee | 3000 | 地面 | 0.30 | 0.196409 | 0/64 | 0/64 |
| curriculum_knee | 3000 | 地面 | 0.34 | 0.196891 | 0/64 | 0/64 |

## 检查点指纹、命令与完成时间

本地模型在`artifacts/height-curriculum-training-20261012/models/<case>/`，接口、累计计数、有限actor和源完整迁移均已核对。两份`model_resume_initial.pt`的actor／critic／Adam与A逐项相同，L=0.21、2500更新，见[下载模型核验](../../artifacts/height-curriculum-training-20261012/downloaded-model-verification.json)。

| case | 文件 | 累计更新 | SHA256 |
|---|---|---:|---|
| curriculum | model_2600.pt | 2601 | e434efef60b203803422bf8af1ac00f145618cffa2631802d808f410edb613c1 |
| curriculum | model_2999.pt | 3000 | 75850fa432995d0cb26b47f4ca6450f0fdea1014e2df4054b5576966f799d5d3 |
| curriculum_knee | model_2600.pt | 2601 | 294d10a9320228aeaa93b752f81ad314ec4f19e1e2fa6d144095140c7a5b0a3b |
| curriculum_knee | model_2999.pt | 3000 | 498b66ca47acd87d9eda98e14dd723791484cf4af9cd5e88ec10718a9f5b4e5b |

下表日期均为2026-10-12，北京时间。early101对应新增101轮／累计2601，final500对应新增500轮／累计3000。

| 操作 | 开始 | 完成 | 退出码 |
|---|---|---|---:|
| curriculum训练 | 23:18:10.489047 | 23:37:25.510826 | 0 |
| curriculum_knee训练 | 23:18:10.489516 | 23:37:32.392208 | 0 |
| curriculum early101评估 | 23:37:32.412325 | 23:42:53.727235 | 0 |
| curriculum early101渲染 | 23:42:53.729889 | 23:42:56.888705 | 0 |
| curriculum final500评估 | 23:42:56.890239 | 23:48:08.660550 | 0 |
| curriculum final500渲染 | 23:48:08.663518 | 23:48:11.790801 | 0 |
| curriculum_knee early101评估 | 23:48:11.793437 | 23:53:28.040502 | 0 |
| curriculum_knee early101渲染 | 23:53:28.044056 | 23:53:31.198065 | 0 |
| curriculum_knee final500评估 | 23:53:31.203086 | 23:58:49.738962 | 0 |
| curriculum_knee final500渲染 | 23:58:49.742847 | 23:58:53.004902 | 0 |

完整实际argv、日志、起止时间和退出码见[campaign status](../../artifacts/height-curriculum-training-20261012/campaign-launch/status.json)，启动与门禁执行记录见[executed-commands.json](../../artifacts/height-curriculum-training-20261012/executed-commands.json)。每次评估调用冻结`eval_chuanliantui_height_curriculum.py`，传入上述checkpoint、对应jobs下`<case>/<point>`输出目录、两份固定初态及v2 baseline参考；随后`render_chuanliantui_ablation_replay.py --input <该输出目录> --groups <case> --width 640`只渲染qpos，不重新积分物理。

收尾实际执行：

```bash
cd /mnt/d/IMCA/RL/Wheel-Legged-Gym
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=mjlab_training/src ~/.venvs/wheelleg-mjlab/bin/python \
  scripts/agent/verify_chuanliantui_height_curriculum_results.py \
  --output artifacts/height-curriculum-training-20261012/closeout-20261013-v2
```

退出0：重算48项高度均值／分布、膝限位计数及力矩均值、六命令斜率、保存的组合判据成功ID；逐项重跑`paired_acceptance`并核对四份checkpoint与课程状态；解码检查8份视频；冻结127文件哈希一致。该脚本不创建仿真、不更新优化器，也不是另一次行为回放。首次收尾v1因checkpoint里的tuple与JSON里的list直接比较而退出1；修正序列序列化比较后v2通过，失败JSON保留，没有放宽数值或验收门槛。

本次只读远端状态尝试未成功：旧SSH复用连接无响应，停止该读取后独立BatchMode连接的外层WSL命令退出1，输出认证失败，未取得远端命令退出码。完成状态依据已下载的status／worker退出码、完整模型和原始评估产物；不据此声称本次又访问远端或检查了当前机器进程。所有必需交付文件已在本地，不需要再次启动协调器。

## 视频与复跑入口

每个检查点有普通／地面样本0，目标0.30 m，显示实际高度与目标高度。原渲染保存初态和20秒末帧，共201帧／20.1秒；交付版只取前200帧，10 fps、精确20秒，保留原视频、原qpos和最终20秒截图。没有重新积分或修改轨迹；裁剪与解码记录见[video-verification.json](../../artifacts/height-curriculum-training-20261012/video-verification.json)。早期stream-copy裁剪未精确去掉末帧，失败尝试保留，不作为交付版。

| case／更新 | 普通样本0 | 地面样本0 |
|---|---|---|
| curriculum 2601 | [20秒视频](../../artifacts/height-curriculum-training-20261012/campaign-results/curriculum/early101/sample0-normal-exact20s.mp4) | [20秒视频](../../artifacts/height-curriculum-training-20261012/campaign-results/curriculum/early101/sample0-ground-exact20s.mp4) |
| curriculum 3000 | [20秒视频](../../artifacts/height-curriculum-training-20261012/campaign-results/curriculum/final500/sample0-normal-exact20s.mp4) | [20秒视频](../../artifacts/height-curriculum-training-20261012/campaign-results/curriculum/final500/sample0-ground-exact20s.mp4) |
| curriculum_knee 2601 | [20秒视频](../../artifacts/height-curriculum-training-20261012/campaign-results/curriculum_knee/early101/sample0-normal-exact20s.mp4) | [20秒视频](../../artifacts/height-curriculum-training-20261012/campaign-results/curriculum_knee/early101/sample0-ground-exact20s.mp4) |
| curriculum_knee 3000 | [20秒视频](../../artifacts/height-curriculum-training-20261012/campaign-results/curriculum_knee/final500/sample0-normal-exact20s.mp4) | [20秒视频](../../artifacts/height-curriculum-training-20261012/campaign-results/curriculum_knee/final500/sample0-ground-exact20s.mp4) |

Claude可以先运行收尾核验脚本，`--output`改为新的目录；独立GPU行为复跑在WSL中串行执行以下四次，不学习，每次64环境：

```bash
cd /mnt/d/IMCA/RL/Wheel-Legged-Gym
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=mjlab_training/src MUJOCO_GL=egl TZ=Asia/Shanghai
audit_output=artifacts/height-curriculum-training-20261012/claude-eval-rerun-01
test ! -e "$audit_output" || exit 1
for case in curriculum curriculum_knee; do
  for checkpoint in 2600 2999; do
    ~/.venvs/wheelleg-mjlab/bin/python scripts/agent/eval_chuanliantui_height_curriculum.py \
      --checkpoint "artifacts/height-curriculum-training-20261012/models/$case/model_$checkpoint.pt" \
      --output "$audit_output/$case/$checkpoint" \
      --initial-normal artifacts/height-response-a2500-20261012/initial-normal.npz \
      --initial-ground artifacts/recovery-regression-paired-20261012/initial-ground.npz \
      --reference artifacts/balance-acceptance-v2-20261012/baseline/evaluation.json \
      --device cuda:0 || exit $?
  done
done
```

同版本GPU仍可能有浮点／后端差异；核对初态、版本、完整成功集合及分项，不能只比较整批均值。本次收尾没有代Claude执行独立复跑。

## 文件范围

前期新增独立配方`mjlab_training/src/ct_mjlab/experiments/height_curriculum.py`，门禁／训练／调度／评估入口分别为`scripts/agent/check_chuanliantui_height_curriculum.py`、`run_chuanliantui_height_curriculum.py`、`launch_chuanliantui_height_curriculum.py`、`eval_chuanliantui_height_curriculum.py`；接口说明和本报告、AGENTS任务记录已经建立。

本次收尾新增`scripts/agent/verify_chuanliantui_height_curriculum_results.py`及`artifacts/height-curriculum-training-20261012/closeout-20261013-v1/`、`closeout-20261013-v2/`核验JSON；补齐本报告，并在AGENTS.md前部追加完成记录。已有原始报告、模型、视频、失败记录和冻结发布包保留。冻结训练内核、已有评估实现、资产、默认V14、部署代码和历史logs未修改；没有Git提交、分支或worktree操作。
