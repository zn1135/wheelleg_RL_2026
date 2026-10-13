# r15 v2 从累计 2000 轮续训到 4000 轮

用户授权“继续续训”，并选择追加 2000 轮，累计 4000 轮。于北京时间 2026-10-11 15:16:26 启动远端后台作业，4096 环境、seed 42、当前冻结 r15 辅助 v2 配方。不修改奖励、课程门槛、初态、控制、气弹簧、辅助力退坡规则或 PPO 参数。

## 辅助力关闭时点

辅助力为整机重量 × 高度缺口比例 × `max(0, 1 - 累计更新进度 / 3000)`；进度包括当前 rollout 的步进，续训继承累计计数。累计 2000 轮时最大预算约为整机重量的 1/3，实际力可能更小或为零。累计 3000 轮开始完全为零，因此本次包含约 1000 轮无辅助力训练。如果提前进入行走阶段，也立即关闭。回放始终关闭辅助力；地面电机保持期间不施加辅助力。

## 恢复与启动证据

- 冻结源码和资产全部 90 个文件 SHA256 核对通过，使用原独立 mjlab Python 运行环境。
- 起点：`20261011_142207_855754_getup_r15_contact_fix_resume_1000/model_1999.pt`，累计 2000 更新，模型 SHA256 `9493b4f9d57d937500e9fabbed7c9d8a0631d76b7d0a0079e9e9e55eb1b403f1`。
- 新目录 `model_resume_initial.pt` 的 actor、critic、Adam 状态与源模型逐项完全一致，课程阶段与计数完全一致；恢复的学习率为 `1.5000000000000005e-05`。
- 课程恢复为扶正，阶段步数 48000、达标计时 0、历史为空；累计辅助力退坡从 2000 轮继续。
- 首次更新检查点 `model_2000.pt` 记录累计 2001 更新，actor 已变化，actor/critic 参数全部有限。
- 启动后约 1 分半日志平均 reward 123.63、平均回合 1000 步；这只是训练启动状态，不是回放达标结论。
- 两组共享 GPU，核对时 5217 MiB、99% 利用率，约 2.2 秒/轮，剩余时间估算约 1 小时 8 分钟。预计时间会随另一组完成和机器负载变化。
- [完整启动记录](20261011-mjlab-getup-r15-resume-2000-to-4000-start.json)，下载 SHA256 与远端一致：`a10247fb05ba711e517fcfef8c10b35aa0a70d5bc2f9409cdb313be8935c0ebb4`。

## 目录

- 模型、TensorBoard 和源码快照：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_151639_260563_getup_r15_contact_fix_resume_2000_to_4000`。
- 作业启动、控制台和退出码：`/root/gpufree-data/wheelleg-remote/jobs/20261011T071610Z-getup-r15-resume-2000-to-4000`，控制台 `output.log`，退出时生成 `.exit-code`。
- 后台启动器 PID 12296，SSH 退出不停止训练。最终预计检查点 `model_3999.pt`。
- 原续训产物和从头训练组均保留；历史 logs 未删除、移动或覆盖。

## 已执行命令

以下仅为复现记录，不要重复启动本次作业。

```bash
cd /root/gpufree-data/wheelleg-remote/releases/20261011T061841Z-getup-r15-contact-fix
export PYTHONPATH="$PWD/mjlab_training/src"
export MUJOCO_GL=disable
unset WHEELLEG_HUANANHU_INIT_CHECKPOINT
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u -m ct_mjlab.cli train --training-profile getup-r15 --num-envs 4096 --iterations 2000 --seed 42 --device cuda:0 --run-name contact_fix_resume_2000_to_4000 --resume /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_142207_855754_getup_r15_contact_fix_resume_1000/model_1999.pt
```

TensorBoard（远端另开终端，本次未启动）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m tensorboard.main --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_151639_260563_getup_r15_contact_fix_resume_2000_to_4000 --host 127.0.0.1 --port 8080
```

本机转发后访问 `http://127.0.0.1:8080`：

```bash
ssh -p 30702 -N -L 8080:127.0.0.1:8080 root@183.147.142.40
```

本次没有修改训练或部署内核，也没有执行 Git 提交；Windows 副本无 `.git`。需在结束后独立关闭辅助力回放确认效果，不能用启动检查或训练 reward 代替行为验证。
