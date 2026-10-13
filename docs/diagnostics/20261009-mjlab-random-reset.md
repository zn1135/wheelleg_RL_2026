# 华南虎式随机初态与20环境可视训练

2026-10-09用户在20环境训练后要求初始位姿随机化，参考华南虎。分支 `26_wheelleg`，基础提交 `58947e8` 加未提交r3代码；独立复查未执行。

## 实现与来源

参考SCUT提交 `868467c3427fc00d34ebb1176683652d9597ba1c` 的V14 flat v0：reset_base的roll/pitch±0.15、yaw±3.14 rad；cfg_utils的地面正／负模式概率0.3／0.2，地面根高0.25 m；普通腿长0.13～0.32 m、地面腿长0.14～0.36 m；四主动腿初速±π/2 rad/s、轮初速±50 rad/s、轮角±2π。

机器人适配使用用户指定本机 `mjlab/wheelleg_RL` 已求解的CAD姿态表；每模式1024条，按qpos0转换当前电机坐标，并核对机体变换、轴、限位、主腿／弹簧闭合及参考端点。腿长按归一化伸展映射到约171.637～336.744 mm。表包含解算、装配分支、限位与碰撞过滤，非原始连续分布的精确复制。

运行时独立GPU采样，左右腿独立，地面条件正／负概率60%／40%。普通根高至少0.35 m并按最大倾斜碰撞界提升；地面根水平且0.25 m。根速度和被动速度零，主动速度按源采样，不额外投影约束速度。原任务静止命令、0.22 m高度、全电机启动0.2 s零力矩保持当前覆盖。确定性回放仍用原后摆初态。

接口升级r3，新增初态表／采样器SHA；r2短训模型不自动resume。新增源码和姿态表均纳入新run的来源快照。

## 验证

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/import_chuanliantui_mjlab_reset_bank.py
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_mjlab.py --gpu --out /tmp/chuanliantui-random-reset-check-20261009.json
```

两条命令退出0。完整物理／奖励／ONNX结果见 [检查JSON](20261009-mjlab-random-reset-check.json)，采样结果见 [采样JSON](20261009-mjlab-reset-sampling.json)。

- 三模式各1024条姿态移植后，闭合最大误差6.547026e−8 m，与参考销轴端点最大差2.235386e−8 m。
- 1024次随机采样，地面占比0.526367，根高度0.25～0.420679 m；同seed重复一致，主动／轮初速在规定范围。
- 128个随机组合经独立原生MuJoCo核对，含非零root姿态和随机轮角，六处闭合最大误差5.391025e−8 m。
- GPU4环境64策略拍，状态／奖励有限，无数值异常重置掩盖；六处闭合最大误差4.559271 mm。部分reset历史清理通过。
- CPU确定性60拍、源奖励对照、PPO runner构造和随机actor ONNX检查通过；这些仍不代表随机初态策略行为验收。

## 可视预检与实际训练

依次执行，预检成功退出后才启动训练：

```bash
DISPLAY=:0 scripts/mjlab.sh play --num-envs 20 --random-reset --render --seconds 10
DISPLAY=:0 scripts/mjlab.sh train --num-envs 20 --render --iterations 20000 --run-name huananhu_random_pose20
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui/20261009_190253_790595_huananhu_random_pose20 --port 8080
```

预检为实际20环境网格、零请求动作，无学习，运行10秒后正常退出。随后训练从随机权重开始，开启域随机化，已观察完整PPO更新并生成首个checkpoint和TensorBoard事件；整理本记录时至少完成iteration0～45，共46次更新，仍在运行。TensorBoard命令仅交付，未启动服务。

新输出：`logs/mjlab/chuanliantui/20261009_190253_790595_huananhu_random_pose20/`。interface／runner配置、实际命令、基础SHA和修改源码／新资产快照保存在该run内。终端记录在 `/tmp/chuanliantui-mjlab-random-train20-20261009-01.log`。旧r2 `model_28.pt`和历史日志未覆盖。

空格暂停采样、V开关刷新、[／]选环境，关闭窗口会在当前PPO更新完成后保存退出。后续最终轮数及停止状态以实际日志与checkpoint为准；本记录不是完成20000轮、训练收敛或正式起立／静站验收的证据。未做H7适配、固件构建或真机验证。

## 后续停止记录

为响应用户启用模式槽的要求，r3训练收到首次SIGINT后完成当前PPO更新并保存退出，退出码0。最终iteration0～445，共446次更新，保存 `model_445.pt`，SHA256 `d589e5441fd619d4d024042f865b19bcf5771b3bc2e86864ef8602d4695181af`，checkpoint接口仍为r3。该模型不会直接续训到r4。后续见 [模式槽记录](20261009-mjlab-control-modes.md)。
