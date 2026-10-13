# V14 低姿态诊断与1000→2000轮续训

**完成状态：累计2000轮已正常结束，退出码0。回放高度约0.20 m，严格站起仍0/64，见 [2000轮效果报告](20261011-mjlab-v14-height-model1999-evaluation.md)。以下保留启动时记录。**

## 诊断依据

源模型为原V14 `model_999.pt`，SHA256 `0201396ba38378abba8ba986aa8e32405d7ebdb874eecd119911ee354939fda7`。既有独立回放：64普通/64地面初态、20秒、目标0.30 m、零速度、确定性actor、无辅助力，严格恢复均0/64，末5秒平均高度约0.190/0.186 m；详见 [原评估](20261011-mjlab-scut-v14-model999-evaluation.md)。

本次追加诊断使用同一模型、每个目标16个环境、六组配对普通初态、20秒、名义物理。无学习，数值失败0。末5秒结果：

| 高度命令 m | 实际平均高度 m |
|---|---|
| 0.20 | 0.18847 |
| 0.22 | 0.18872 |
| 0.27 | 0.18946 |
| 0.30 | 0.18999 |
| 0.34 | 0.19078 |
| 0.42 | 0.19242 |

高度命令升高22 cm，实际只升高约3.95 mm。同一观测仅替换高度槽时actor输出确实变化，说明高度输入存在，但没有学到足够的实际抬升响应。

0.30 m目标下的加权奖励贡献（每秒值，非训练日志的回合累计值）：

| 项目 | 贡献 |
|---|---|
| 前进速度指数奖励 | +0.99436 |
| 偏航速度指数奖励 | +0.99964 |
| 两项姿态指数奖励合计 | +1.99469 |
| 高度指数奖励 | +0.00000555 |
| 高度平方惩罚 | −1.21022 |
| 电机力矩惩罚 | −0.20638 |
| 终止项 | 0 |
| 非期望接触 | 0 |

这些结果支持“低姿态形成奖励局部最优”：保持低姿态和小速度仍有正奖励，物理失败条件不等于未达到高度目标，因此没有自动触发−200失败惩罚。高度指数奖励在约11 cm误差下几乎消失；0.20～0.42 m目标跨度同时包含较低目标与超出本机器人已核验站高范围的上端。配方相同也不能消除实体闭链、气弹簧与参考机器人可达姿态的差异。

诊断不是因果消融，也未证明所有训练初态都同样表现；本次零速度回放不作为高速行走测试。首次探针传入普通Tensor而非actor要求的TensorDict，未成功；修正后的 `probe-v2.py` 完成上述检查，保留远端首次失败日志，本文引用成功结果。

## 本次只改变两项训练配置

1. 高度采样收窄为0.27～0.34 m。
2. 四项速度跟踪奖励乘宽高度门控：误差≤5 cm全额，5～10 cm线性下降，≥10 cm为0，包括相应平方负项。

其余V14参数保留。完整接口和迁移约束见 [说明](../interfaces/chuanliantui-mjlab-scut-v14-height-v1.md)。独立配方保留原V14内核及原模型回放；本次不加向上辅助力、膝限位或默认角偏离惩罚。

## 检查与运行证据

- 最终发布：`/root/gpufree-data/wheelleg-remote/releases/20261011-v14-height-fix-final`。
- native CLI物理检查：20环境、64步，actor35/critic78维，奖励和观测有限，数值失败0。
- 4096环境迁移检查：仅允许高度范围与奖励函数适配变化；actor、critic、Adam全部逐项相同；恢复1000次更新与1e-5学习率。新配方回读通过，隐式跨配方加载被拒绝。
- 门控边界验证：误差0/5/7.5/10/15 cm及NaN，门控1/1/0.5/0/0/0；同一步不会重复乘门控，其他奖励不变。
- 最终检查wrapper退出码0。未执行额外PPO更新作为预检。
- 开始：2026-10-11 21:46:15（Asia/Shanghai），训练PID22915；job `20261011T134615Z-v14-height-resume-1000-to-2000`。
- 运行输出：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_214628_750246_scut_v14_height_v14_height_resume_1000_to_2000`。
- 21:47:15启动核验：`model_resume_initial.pt`计数1000、`model_1000.pt`计数1001，actor已变化且有限，日志1034/2000，GPU81%、2620 MiB，进程存活，未退出。
- 21:49:33收尾复查：累计1146/2000，PID22915仍运行，最新数值失败指标0；SSH断开不影响后台训练。
- 目标最终文件 `model_1999.pt`，本记录是启动记录，不表示2000轮已完成或站高已改善。

证据下载包 `artifacts/v14-height-resume-evidence-20261011.zip` SHA256：
`b494b1a5e6a98f78a26f19b5787050d1e7ed978f41abf6f106224d549888f0a4`。

下载后完整ZIP指纹与远端一致，展开的启动JSON中的四份本地源码指纹均匹配。证据目录含 `startup/startup.json`、完整命令、当时日志、`checks/migration-check.json`、检查脚本、`diagnosis/diagnosis.json` 和成功诊断脚本。新内核SHA256 `672647d77620c75ecfb03611fd6fe77267fe3850bba98649f66ac8c403d06bcc`，原V14内核指纹未变。

## 已执行命令

以下训练已在独立后台wrapper启动，无需重复运行：

```bash
cd /root/gpufree-data/wheelleg-remote/releases/20261011-v14-height-fix-final
PYTHONPATH=$PWD/mjlab_training/src \
  /root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u -m ct_mjlab.cli train \
  --training-profile huananhu-v14-height --num-envs 4096 --iterations 1000 \
  --seed 42 --device cuda:0 --resume-v14-height-fix \
  --resume /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_205306_764251_scut_v14_flat_scut_v14_flat_scratch_1000/model_999.pt \
  --run-name v14_height_resume_1000_to_2000

# 另一远端终端查看；本次没有启动TensorBoard。
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m tensorboard.main \
  --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_214628_750246_scut_v14_height_v14_height_resume_1000_to_2000 \
  --port 8080
```

后台进程不依赖SSH连接，job的 `output.log` 和 `.exit-code` 用于后续进度核对。原logs、原模型和冻结发布未删除、移动或覆盖。本机仍无Linux mjlab运行时，实际检查与学习都在远端Python3.12 / RTX4090完成。

完成后使用统一回放脚本比较原1000轮与本次2000轮。奖励门控改变了总reward的尺度，效果以实际高度、严格恢复、站立维持、触地和数值失败衡量，不能仅凭reward涨跌判断。
