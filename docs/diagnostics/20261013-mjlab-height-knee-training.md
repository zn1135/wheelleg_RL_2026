# A2500 两组膝限位续训1000轮与评估

状态（2026-10-13）：两组各追加1000更新、累计3500预算已完成，model_3499均验证累计3500／iter3499。knee_only训练退出0；原协调器读取稀疏日志时索引2599失败、退出1，未能收集knee_gate训练退出码。14:31:31启动仅评估调度，未重启或追加学习；**评估进行中，尚无完整验收结论。**

用户授权仅两组：`knee_gate`、`knee_only`，各从A2500独立追加1000次PPO更新，4096环境、seed42、远端并行，累计3500为预算上限，不追加、不扩组。完整定义见[接口说明](../interfaces/chuanliantui-mjlab-height-knee-v1.md)。本次没有Git操作，未改默认V14、已有实验／评估／参考内核、机器人资产、部署代码或参考仓库。

新增代码文件：`mjlab_training/src/ct_mjlab/experiments/height_knee.py`；`scripts/agent/run_chuanliantui_height_knee.py`、`check_chuanliantui_height_knee.py`、`eval_chuanliantui_height_knee.py`、`eval_chuanliantui_height_knee_robustness.py`、`launch_chuanliantui_height_knee.py`、`verify_chuanliantui_height_knee_results.py`。新增文档为本报告和接口说明；按用户最新要求，当日记录只写 `docs/ai/project-log.md` 顶部，AGENTS.md已撤掉本次新增段落。新增产物及WSL传输／准备包装脚本统一保存在 `artifacts/height-knee-training-20261013/`。194个既有源码／资产文件在本地开发前后逐项哈希不变；参考仓库只有读取，没有写入。

## 运行位置及源模型

新SSH：`ssh -p 30248 root@183.147.142.40`，RTX4090。新机最初无 `/root/gpufree-data/wheelleg-remote`、运行环境或模型；从本地只读复制A2500至说明指定的远端源路径，核验SHA256，没有覆盖历史run。固定Python3.12／mjlab1.6.0／Torch2.14.0／MuJoCo3.11.0／mujoco-warp3.11.0／Warp1.17.0／RSL-RL5.4.2。创建venv首尝试因缺ensurepip失败，保留失败日志；用系统pip的 `--python` 仅给独立venv装pip与固定依赖，不改系统Python或训练参数。

源：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261012_122334_386753_recovery_factors/baseline/model_2499.pt`，本地 `artifacts/height-response-a2500-20261012/models/model_2499.pt`，SHA256 `706c8387cba4815b491be3f5914a5951881c2966edabbd5fbb90d0da9c9d3dbd`。A2500普通64/64、地面60/64，但0.30m高度跟踪0/64，约0.20m低位，不是高度或部署合格模型。

发布：`/root/gpufree-data/wheelleg-remote/releases/20261013-height-knee-v1`。
Campaign：`/root/gpufree-data/wheelleg-remote/jobs/20261013-height-knee-1000-v1`。
两路径均先确认不存在再新建。176文件逐项SHA256，实际门禁采用bundle `release-v2.zip`，SHA256 `d1a0ba08c08a078e561e3217ef676711cba4ceb183543aefb002cc132c2777e6`。v1保留为准备记录，v2在门禁／冻结前仅将新增评估脚本样本0录像改为t=0～19.9s的200帧、10fps、准确20s；没有修改扫描步数、判据或训练模块。新机核心依赖及19项CUDA组件逐项匹配本地环境，Torch实测2.14.0+cu130、RTX4090可用；完整安装版本记录为runtime-verification.json。初始依赖下载缓慢，曾停止只在下载阶段的pip进程，使用SHA256已验证的相同官方wheel重试；没有中止、重启任何训练。本地新增证据根 `artifacts/height-knee-training-20261013/`；历史logs/jobs/releases不删除、不移动、不覆盖。

## 训练前结果

| 检查 | knee_gate | knee_only |
|---|---|---|
| 本地16环境×64步 | 观测／逐项奖励全有限，数值失败0 | 同左 |
| 同真实状态奖励隔离 | 仅四项门控速度奖励及新膝项可变化，其余逐项相同 | 仅新膝项可变化，其余逐项相同 |
| 误差2／4.5／7cm门控 | 1／0.5／0 | 1／1／0.6（保持A） |
| 膝公式 | 原核lf1/rf1精确一致，权重−0.5 | 同左 |
| 配置隔离 | 除奖励函数／新膝项，配置与A逐项一致 | 同左 |
| 远端16／4096环境×64步 | 全有限、数值失败0 | 同左 |
| actor／critic／Adam迁移逐项相等 | true／true／true | true／true／true |
| 初始更新／Adam LR／自身重载 | 2500／1e-5／通过 | 同左 |
| 初始模型A配对成功集合复现 | 普通64／地面60，保留A全部原成功样本，数值失败0 | 同左 |

本地检查2026-10-13 13:15:55～13:16:12，退出0、旧实验／参考内核指纹保持。证据 `local-preflight-v1/preflight.json`。任何远端门禁失败都不进入学习。

## 训练执行与退出状态

北京时间13:45:26.524656协调器启动，PID633，启动退出0；两组并行。run根为 `/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261013_134526_524565_height_knee`，子目录knee_gate／knee_only。两组最终保存时间分别14:26:26.332572／14:26:08.040935；knee_only协调器记录完成14:26:11.673130、退出0。knee_gate实际进程退出时间和退出码未被父协调器收集，保持未能获取；最终模型3500更新、完整终轮日志及进程已不存在证明预算完成，不替代退出码。TensorBoard仅交付命令，不启动。子进程显式冻结PYTHONPATH、MUJOCO_GL=egl、TZ=Asia/Shanghai、OMP_NUM_THREADS=1、MKL_NUM_THREADS=1。每100更新保存，最终model_3499.pt累计3500。累计3000后保留原15% spin_low特殊命令桶，不能归因于本次改法；dash也保持开启。本次无高度课程，不改标准差、熵系数、随机化、初态或原高度奖励。

实际训练命令（由协调器并行各执行一次，已有run禁止重复执行）：

```bash
runtime_python=/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python
release=/root/gpufree-data/wheelleg-remote/releases/20261013-height-knee-v1
source=/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261012_122334_386753_recovery_factors/baseline/model_2499.pt
run_root=/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261013_134526_524565_height_knee
PYTHONPATH=$release/mjlab_training/src MUJOCO_GL=egl TZ=Asia/Shanghai OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $runtime_python $release/scripts/agent/run_chuanliantui_height_knee.py --case knee_gate --resume $source --run-dir $run_root/knee_gate --iterations 1000 --num-envs 4096 --seed 42
PYTHONPATH=$release/mjlab_training/src MUJOCO_GL=egl TZ=Asia/Shanghai OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $runtime_python $release/scripts/agent/run_chuanliantui_height_knee.py --case knee_only --resume $source --run-dir $run_root/knee_only --iterations 1000 --num-envs 4096 --seed 42
# 仅交付，没有启动：
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/tensorboard --logdir=/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261013_134526_524565_height_knee --port=8080
```

完成后评估每组3000／3500／最低站立100倍数检查点（去重），六高度×两初态×64环境、20s，验收v2逐样本退化否决、原生膝约束行、固定共同成功集合斜率和0.30m样本0视频。只有最终模型0.30m高度跟踪至少一种初态≥32/64才触发256环境随机化复核。**非零速度行走能力未评估。**

原协调器于14:26:19.876118退出1，原因是按保存轮数读取TensorBoard值时假定每轮都有回合日志，`KeyError:2599`；这属于新增协调器的后处理错误。knee_only退出0已记录，knee_gate随后完成最终模型保存，但其实际进程退出码与退出时刻没有被父进程收集，不能恢复。原`status.json`、`coordinator.log`和`worker.exit-code`原样保留。14:31:31.697760启动只做评估的包装脚本，先核对两训练进程已不存在、20个保存点的计数和终轮日志，再调用同一冻结评估、渲染脚本；没有重新调用学习或改变参数。

## 新增和修改文件

新增训练模块 `mjlab_training/src/ct_mjlab/experiments/height_knee.py`；新增入口／门禁／评估／随机化／协调器／原始证据复核分别为 `scripts/agent/run_chuanliantui_height_knee.py`、`check_chuanliantui_height_knee.py`、`eval_chuanliantui_height_knee.py`、`eval_chuanliantui_height_knee_robustness.py`、`launch_chuanliantui_height_knee.py`、`verify_chuanliantui_height_knee_results.py`。新增本报告和[接口说明](../interfaces/chuanliantui-mjlab-height-knee-v1.md)；当日记录仅修改[project-log.md](../ai/project-log.md)顶部，未向AGENTS.md添加记录。运输／安装／冻结／后处理脚本与证据均新增于 `artifacts/height-knee-training-20261013/`，其中`assessment-only-remote.py`只负责无学习评估恢复。

旧实验模块、reference_training、既有评估／诊断脚本与资产共194份受保护文件保持开始时SHA256；没有Git操作，没有修改参考仓或deployment/H7_RL。新发布176份源码／输入在训练和评估结束后再次逐项核对。

## 结论状态

**已验证**：本地16和远端16／4096环境有限、奖励和配置隔离、门控样本、原膝公式、两组actor／critic／Adam完整迁移精确相等、2500更新及1e-5恢复、自身checkpoint重载、各组初始模型A配对64／60复现、本地与新远端源模型SHA256、176文件冻结发布及新run没有覆盖已有目录。启动通过不等于效果通过。

**推断**：膝软惩罚可能改变低位策略吸引域，是否学习高度跟踪必须以本批训练后配对回放判断；wheelleg v11的成功只提供改法参考。

**未能验证**：训练效果、退化与域随机化结果尚待运行。两个仓XML膝damping0.01／0.03且本仓frictionloss0.015，不能直接复制参考仓结果作为预期。单seed和64初态仅代表本批趋势，不支持普遍因果或真机部署结论。不得以本报告触发额外预算。
