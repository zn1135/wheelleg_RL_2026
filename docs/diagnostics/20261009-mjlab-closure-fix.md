# mjlab 闭链销轴偏差修复

日期2026-10-09，分支26_wheelleg，基础HEAD 58947e8fabc4683b2025d42bda742b06c84dc566 加既有未提交修改及本次修复。作者为本次助手，独立复查待执行。此前 [r5控制对齐检查](20261009-mjlab-reference-control.md) 的GPU 6.181 mm失败在本次修复后通过；保留失败记录。

## 定位依据

初态缓存最大闭合误差约5.44e-8 m，几何初值已闭合。用相同seed42、4个随机环境、128策略步，并观察每个2 ms子步：

| 对照 | 最大闭链偏差 | 数值重置 |
|---|---:|---:|
| 原10 ms connect响应，60/2 PD | 6.825 mm | 0 |
| 原约束、禁用电机力矩（保留弹簧） | 2.689 mm | 0 |
| 5 ms connect响应，60/2 PD | 2.205 mm | 0 |
| 4 ms connect响应，60/2 PD | 1.498 mm | 0 |

原峰值发生在0.290 s、右后连杆第二销轴，电机已经进入高力矩控制；0.5 s后仍测得6.642 mm。以上结果支持主要问题是现有connect软约束在PD载荷下的偏差，不能归因于缓存未闭合或仅归因于电机转动惯量。记录见 [对照JSON](20261009-mjlab-closure-ab.json)及 [当时的诊断脚本快照](20261009-mjlab-closure-ab-script.txt)。它们属于定位实验，正式回归使用下述仓内有限检查脚本。

## 修复

将本仓独立mjlab MJCF的四组腿销轴及两组气弹簧端点connect统一从 `solref="0.01 1"` 改为 `solref="0.005 1"`。阻尼比1及原solimp不变。CPU和GPU均编译同一资产，并在检查中逐项比对实际eq_solref／eq_solimp。

正值solref的时间常数控制约束响应；5 ms保持高于2 ms步长的两倍。该取值依据 [MuJoCo官方求解器参数说明](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters) 的数值稳定建议，并以本机实测校验。没有关闭refsafe、调整原5 mm门槛、取消接触或削弱60/2 PD来通过检查。

姿态JSON及随机姿态表只因资产指纹更新而重新生成，根初态与关节物理姿态保持原定义。电机armature及气弹簧力律不变，策略周期仍20 ms。新资产指纹为 `42a1b937b83ca034d4913306aeca5b219c3cd1c5a0bbd57b0220d7f5d5c826e0`。

r5尚无新训练模型，本次作为r5控制迁移的后续修复；manifest的资产／姿态表SHA更新，使用旧资产的同名r5记录仍会被拒绝。wheelleg_RL参考仓、旧sim2sim、H7和历史logs未改。

## 正式有限回归

在本仓根目录，用独立Python 3.12 mjlab环境执行：

```bash
mjlab_training/.venv/bin/python scripts/agent/generate_chuanliantui_mjlab_pose.py
mjlab_training/.venv/bin/python scripts/agent/import_chuanliantui_mjlab_reset_bank.py
mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_mjlab.py --gpu --num-envs 32 --steps 128 --seed 42 --out /tmp/chuanliantui-closure-fix-seed42.json
mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_mjlab.py --gpu --num-envs 32 --steps 128 --seed 7 --out /tmp/chuanliantui-closure-fix-seed7.json
git diff --check
```

| seed | GPU环境／策略步 | 初始地面／普通 | 全物理子步最大销轴偏差 | 普通回合重置次数 | 退出码／结果 |
|---|---|---|---|---|---|
| 42 | 32／128 | 19／13 | 2.440 mm | 75 | 0／通过 |
| 7 | 32／128 | 14／18 | 2.467 mm | 77 | 0／通过 |

每环境物理检查时长2.56 s，采样1280个物理子步；包含PD启动、随机落地和回合复位。检查使用实际site距离，并在自动reset前保留子步峰值，避免仅观察策略边界或reset后掩盖瞬态。两组无数值重置，有限状态、观测、奖励、0～150 N弹簧力、CPU/GPU约束参数、源奖励对照及模式接口检查均通过。

CPU原生回放同样逐2 ms子步记录，最大销轴误差 1.498 mm。随机actor ONNX误差约1.04e-7，仅用于接口数值检查，不是交付策略。seed7检查还核对32环境中局部reset不会清除其他31个环境的延迟历史与模式；seed42记录执行于该覆盖扩展前，原局部reset检查覆盖环境0、2、3。

证据：[seed42](20261009-mjlab-closure-fix-seed42.json)、[seed7](20261009-mjlab-closure-fix-seed7.json)、[版本／SHA清单](20261009-mjlab-closure-fix.json)。生成脚本／姿态表导入及diff检查退出码均0。

零请求动作仍会通过位置PD拉向名义姿态，因此上述普通回合失败和重置是有限物理检查中的现象，不能算起立成功，也不能将数值检查通过称为策略行为验收。本次没有启动训练或TensorBoard；持续站立、完整随机恢复、板端与真机验证仍需对应模型。
