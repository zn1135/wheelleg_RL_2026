# r15轮承重方向修复后续训

用户授权“修，再训1000轮”。修正轮支撑方向，从辅助版v1已完成1000轮的 `model_999.pt` 完整续训，4096环境、seed42，追加1000轮，目标累计2000轮。

## 修复范围

- 当前接口为 `chuanliantui-mjlab-getup-curriculum-r15-assisted-v2`。
- mjlab轮传感器的承重竖直力为负，统一转换为 `clamp(-Fz,min=0)`；用于双轮承重奖励门控、姿态偏置门控、轮载荷奖励及整批课程达标判断。正方向的反向作用力不算向上承重。
- 奖励权重、15°姿态／3 cm高度门槛、阶段比例与连续时间、初态、物理、实体关节顺序、气弹簧和PPO参数沿用原配置。
- 增加显式 `--resume-contact-force-fix`，只允许已知原辅助v1源码及相同完整配置迁移。默认不接受跨版本恢复。
- 迁移保存源模型SHA256、接口变化及累计更新数；恢复后的初始检查点为 `model_resume_initial.pt`，便于核对PPO开始前的权重和优化器。

## 检查与实际恢复

[完整启动记录](20261011-mjlab-getup-r15-contact-fix-start.json)包含源码哈希、有限检查、实际命令和恢复核对。

- 20环境，三阶段各64步，35/78维观测及奖励有限、数值异常0。
- 直接检验双轮向上承重、单轮离地、反向作用力、接触力阈值，以及姿态／辅助力门槛；实际奖励内核的no_fly、pose_bias与轮载荷公式通过。
- 阶段推进／恢复、辅助力退坡、扰动力组合和回放检查通过；合法旧源码需显式迁移，未知旧源码拒绝。
- 实际4096环境恢复后，actor、critic、Adam完整状态与原模型逐项完全一致，课程状态完全一致。原学习率 `2.25e-5` 已恢复。
- 续训初始累计更新数1000，首次保存 `model_1000.pt` 累计1001，actor已经变化、actor／critic参数有限。
- 核对时控制台索引1068/2000，GPU2610 MiB、利用率74%。最新回合统计索引1041：数值异常0，no_fly由旧版恒定-2变为约-0.118，pose_bias由0变为约-0.142，确认真实训练奖励已使用修复后的接触判定。
- 当前仍是扶正阶段；仅证明修复和续训正常，不代表已经通过严格站立验收。

首次扩展检查在首拍前读取尚未初始化的终止缓冲失败；将真实奖励探针移到完成有限步进后，重跑全部检查通过。没有修改训练内核来绕过该问题。

## 产物

- 源模型：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_134150_495971_getup_r15_assisted/model_999.pt`，SHA256 `a31ca84e5b91c110cda3beb13f95f0749938bccd0d2bfe9e934cdac8dff03b59`。
- 冻结训练源码：`/root/gpufree-data/wheelleg-remote/releases/20261011T061841Z-getup-r15-contact-fix`。
- 新训练产物：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_142207_855754_getup_r15_contact_fix_resume_1000`。
- 作业目录：`/root/gpufree-data/wheelleg-remote/jobs/20261011T061841Z-getup-r15-contact-fix`，保存验证、控制台、启动／恢复记录和退出码。
- 训练PID11122，后台启动器11121。新目录保护原训练产物，SSH退出不停止后台训练；最终预计保存 `model_1999.pt`。

## 已执行训练命令

以下为复现记录，不要重复启动本次任务。

```bash
cd /root/gpufree-data/wheelleg-remote/releases/20261011T061841Z-getup-r15-contact-fix
export PYTHONPATH="$PWD/mjlab_training/src"
export MUJOCO_GL=disable
unset WHEELLEG_HUANANHU_INIT_CHECKPOINT
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -u -m ct_mjlab.cli train --training-profile getup-r15 --num-envs 4096 --iterations 1000 --seed 42 --device cuda:0 --run-name contact_fix_resume_1000 --resume /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_134150_495971_getup_r15_assisted/model_999.pt --resume-contact-force-fix
```

TensorBoard（远端另开终端，本次未启动）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m tensorboard.main --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261011_142207_855754_getup_r15_contact_fix_resume_1000 --host 127.0.0.1 --port 8080
```

本机转发后访问 `http://127.0.0.1:8080`：

```bash
ssh -p 30702 -N -L 8080:127.0.0.1:8080 root@183.147.142.40
```

本轮未执行Git提交；Windows迁移副本没有 `.git`。未修改部署侧代码。
