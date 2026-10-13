# r15 v2 续训累计 2000 轮：无辅助力回放

## 对象与执行记录

- 本次响应“查看续训组、带 video”，只做评估和视频生成，没有启动新训练或修改训练代码。
- 训练组：`20261011_142207_855754_getup_r15_contact_fix_resume_1000`，从 v1 的 1000 轮恢复后追加 1000 轮。
- 训练正常结束，退出码 0；最终 `model_1999.pt`，记录的累计更新数 2000。
- 模型 SHA256：`9493b4f9d57d937500e9fabbed7c9d8a0631d76b7d0a0079e9e9e55eb1b403f1`。
- 配方：`chuanliantui-mjlab-getup-curriculum-r15-assisted-v2`，检查点仍处于 `righting` 扶正阶段，目标高度 0.27 m。
- 使用冻结源码 `releases/20261011T061841Z-getup-r15-contact-fix` 和远端 mjlab 运行环境；检查点接口及课程配置验证通过。
- 远端作业目录：`/root/gpufree-data/wheelleg-remote/jobs/20261011T070552Z-r15-resume2000-eval`。
- 远端完整记录：`/root/gpufree-data/wheelleg-remote/evaluations/20261011T070552Z-r15-resume-model1999`。
- 评估与渲染退出码均为 0；同时运行的从头训练组没有被停止或修改。

## 评估条件

普通初态与地面初态各 64 个环境，每组 20 秒，seed 42。确定性 actor、名义物理、零前进/偏航命令、关闭观测噪声和动作延迟、关闭辅助力、冻结课程、不更新策略。记录物理失败但不中断地面恢复机会；数值失败永久排除。

严格恢复：高度误差 ≤3 cm，横滚和俯仰分别 ≤15°，双轮接触力历史峰值模长分别 >5 N，机体接触力 ≤1 N，连续保持 0.5 秒。末 5 秒持续达标要求至少 95% 时间满足条件。此口径与训练课程晋级条件不同：后者要求整批比例、总倾角、当前轮支撑力及 2 秒持续时间，不能混用。

## 批量结果

| 指标 | 普通初态 | 地面初态 |
|---|---:|---:|
| 严格恢复 | 0/64 | 0/64 |
| 末 5 秒持续达标 | 0/64 | 0/64 |
| 末 5 秒平均高度 | 0.270308 m | 0.267965 m |
| 末 5 秒平均绝对高度误差 | 0.759 mm | 2.035 mm |
| 末 5 秒机体接触时间比例 | 0% | 1.4875% |
| 数值异常 | 0 | 0 |

高度接近目标，但仍未学会严格扶正。视频样本均停在倾斜姿态：样本 0 的末 5 秒平均绝对横滚/俯仰分别为普通初态 18.31°/19.96°、地面初态 18.19°/21.31°。这些姿态值仅来自视频样本，不代表所有 64 个环境的角度统计。姿态超出 15° 门槛足以解释该样本未达标，不能据此排除其他样本还存在轮接触等问题。

## 视频与证据

两段视频各展示样本 0，20 秒、10 fps。视频由 GPU 评估记录的 qpos 经 `mj_forward` 渲染，未用 CPU 重新积分；字幕标明辅助力关闭、初态类型、时间和目标高度。检查了 0/2/10/20 秒拼图；普通和地面样本在约 2 秒后均进入倾斜平衡，后续没有扶正。

- [普通初态视频](../../artifacts/getup-r15-v2-resume-model1999-evaluation-20261011/replay-normal.mp4)
- [地面初态视频](../../artifacts/getup-r15-v2-resume-model1999-evaluation-20261011/replay-ground.mp4)
- [拼图](../../artifacts/getup-r15-v2-resume-model1999-evaluation-20261011/replay-poster.png)
- [完整批量报告](../../artifacts/getup-r15-v2-resume-model1999-evaluation-20261011/evaluation.json)
- [样本姿态探针](../../artifacts/getup-r15-v2-resume-model1999-evaluation-20261011/sample0-pose-probe.json)
- [渲染记录](../../artifacts/getup-r15-v2-resume-model1999-evaluation-20261011/render-record.json)

报告、视频、拼图与渲染记录通过 SCP 下载到新的本地目录，五个文件的 SHA256 与远端逐一核对一致。完整 GPU 轨迹、初态和场景模型保留在远端评估目录；历史 logs 未修改。

## 复现入口

远端设置 `PYTHONPATH=/root/gpufree-data/wheelleg-remote/releases/20261011T061841Z-getup-r15-contact-fix/mjlab_training/src`，运行：

```bash
MUJOCO_GL=disable /root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u \
  /root/gpufree-data/wheelleg-remote/jobs/20261011T070552Z-r15-resume2000-eval/tools/eval_chuanliantui_getup_curriculum.py \
  --checkpoint /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_142207_855754_getup_r15_contact_fix_resume_1000/model_1999.pt \
  --output <新的评估目录> --num-envs 64 --seconds 20 --seed 42
```

评估脚本 SHA256：`21463d18828bac83e124498d0a212e064be57da59585eefa2a03ecdaa0441798`；共享回放脚本 SHA256：`d3edcb5dd2475797d98e9fec8db2370441655f51a41c8319a85b523270c80a92`。渲染使用同一运行环境，`MUJOCO_GL=egl`，参数和代码保存在该评估作业的 `tools/render_getup.py`。
