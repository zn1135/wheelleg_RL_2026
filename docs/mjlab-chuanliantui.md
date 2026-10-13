# 串联腿 mjlab 训练

**当前默认新训练为 [华南虎原始V14 Flat-v0配方](interfaces/chuanliantui-mjlab-scut-v14-flat-v1.md)**，配置名 `huananhu-v14`，恢复 `wheeled-legged_RL-main_huananhu` 的奖励与训练参数。本页以下配方和数值描述的是保留的r14及历史实现；运行r14需添加 `--training-profile recover-r14`。r15整批课程需显式选择 `getup-r15` / `getup-r15-orientation`，见 [r15说明](interfaces/chuanliantui-mjlab-getup-curriculum-r15.md)。旧检查点回放仍按原版本。

当前入口 `scripts/mjlab.sh` 使用独立Python3.12/mjlab环境，默认任务 `chuanliantui` 在 wheelleg_RL recover 配方上保留动态normal/recover、四腿两轮动作顺序；r14加入华南虎普通前进／偏航命令和50%航向控制，使用5～10 cm速度奖励高度门控，knee_limit保持0，当前拒绝隐式跨版本续训，初态为 **50%普通＋50%地面，地面机体无额外姿态扰动**（沿用r10）。旧Isaac/Python3.8入口保留，依赖不得混装。

本机复用已有mjlab依赖，通过本仓 `.venv` 引用依赖，不修改外部工程。全新机器可 `uv sync --project mjlab_training --locked --python 3.12`。

## 当前配置与来源

参考本机 `mjlab/wheelleg_RL` 的当前工作区，HEAD `36d0d2fbf424276e082e7c3b9902b49879d894e4` 加未提交修改；逐文件指纹在 `reference_training/SOURCE_IMPORT.json`。训练模块复制到本仓，运行时不依赖外部工程源码／资产。

当前命令配置与兼容性见 [r14接口](interfaces/chuanliantui-mjlab-wheelleg-recover-r14.md)，奖励门控见 [r13接口](interfaces/chuanliantui-mjlab-wheelleg-recover-r13.md)，初态见 [r10接口](interfaces/chuanliantui-mjlab-wheelleg-recover-r10.md)，其余参数见 [历史r6接口](interfaces/chuanliantui-mjlab-wheelleg-recover-r6.md)。当前配置和训练历史未完整复现参考实验，实际差异见 [对比分析](diagnostics/20261010-mjlab-r9-reference-analysis.md)。

- 每次复位以0.5概率选择地面初态；地面root高0.25 m、零姿态、无额外高度偏移。普通初态root最低0.38 m、roll/pitch±0.15 rad、yaw±π；腿仍从合法闭链缓存抽样。
- 高度命令随机0.27～0.34 m，前进速度−2.7～2.7 m/s、偏航速度−2π～2π rad/s；50%航向控制、增益5，每5～15 s重采样；20 s回合。地面初态前1.5 s前进／偏航各−1～1；普通静止抽样合计约19%；后期冲刺／自旋关闭。
- 高度误差≤5 cm时四项线速度／偏航跟踪奖励全额，5～10 cm线性降低，≥10 cm时零；严格起立验收仍为误差≤3 cm。knee_limit权重0，其余权重保持参考值；高度奖励公式保持原值，历史回放按模型记录的3～6 cm门控。
- Actor35/Critic78/Action6；模式槽动态normal/recover（地面初态recover，连续站稳0.5s转normal、持续失稳0.1s回recover）；动作顺序 **lf0,lf00,rf0,rf00,lfwheel,rfwheel**，观测关节／历史动作顺序 **lf0,lf00,rf0,rf00,lfwheel,rfwheel**。
- 物理2 ms×10，PD60/2，轮Kv0.1、目标±150 rad/s；5/10 ms动作延迟及5/10/15 ms传感延迟按参考取整。
- 默认4096环境／20000轮，50轮保存；r6的4096环境／1000轮训练已完成，见 [训练记录](diagnostics/20261010-mjlab-r6-formal-training.md)。r9训练状态见 [运行记录](diagnostics/20261010-mjlab-r9-formal-training.md)。原1000轮r5模型保留，回放高度未通过，见 [r5评估](diagnostics/20261010-mjlab-r5-model999-evaluation.md)。

## 机器人

+x前进，+y左；lf为实际左侧，rf为实际右侧。原CAD网格文件名保留，不能按body名称猜网格。实体资产 `resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml` 含19关节、6connect、0tendon、8执行器。保留本仓ref、硬限位、名义角、主腿惯量与安装几何；4腿电机直接位置PD、2轮速度控制，无虚拟关节。

气弹簧实体转动／伸缩建模，预压10 mm，零速度零偏置压缩90 mm时150 N；先限弹性推力、再加偏置和阻尼、总推力限0～150 N。它不占策略动作、始终作用；训练随机偏置±50 N、阻尼25～75。预压／力律是用户模型设定，尚无实物标定。connect solref0.005/1保留r5闭链修复。

## 可执行命令

以下在仓库根执行，训练默认只交付命令。

```bash
# 根初始分布改变后的20环境可视预检；零请求动作，无学习。
DISPLAY=:0 scripts/mjlab.sh play --num-envs 20 --random-reset --render --seconds 10

# 用户授权后才训练；1000轮沿用上次运行预算，可按需修改。
scripts/mjlab.sh train --training-profile recover-r14 --num-envs 4096 --iterations 1000 --run-name wheelleg_recover_r14_velocity_heading
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui --port 8080

# 原生回放；按历史模型选择动作顺序、mode、膝惩罚、初态分布及高度门控。
DISPLAY=:0 scripts/mjlab.sh play --checkpoint logs/mjlab/chuanliantui/<run>/model_<n>.pt --num-envs 20 --render

# CPU同任务评估；按checkpoint选择对应动作顺序和mode。
scripts/mjlab.sh eval --checkpoint logs/mjlab/chuanliantui/<run>/model_<n>.pt --seconds 20 --report /tmp/chuanliantui-eval.json

# 旧r5 checkpoint自动选旧接口回放，不用于当前接口续训。
DISPLAY=:0 scripts/mjlab.sh play --checkpoint logs/mjlab/chuanliantui/20261009_213808_247780_huananhu_r5_closurefix4096_1000/model_999.pt --num-envs 20 --random-reset --render

# 完整参数/奖励参考对照+动态模式+GPU有限步+checkpoint/ONNX检查；不训练。
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_reference_training.py --num-envs 32 --steps 32 --seed 42 --out /tmp/chuanliantui-r13-wide-height-gate-check.json

# 基本有限物理／张量检查；不是行为验收。
scripts/mjlab.sh check --num-envs 32 --steps 128

scripts/mjlab.sh export --checkpoint logs/mjlab/chuanliantui/<run>/model_<n>.pt --output exports/chuanliantui_mjlab/<new-name>/policy.onnx
```

本轮检查范围和结果见 [r13诊断](diagnostics/20261010-mjlab-r13-wide-height-gate.md)。r10远端1000轮训练已完成，未达到目标站高，见 [评估记录](diagnostics/20261010-mjlab-r10-cloud-model999-evaluation.md)；r11已完成4096环境／1000轮远端训练，见 [运行记录](diagnostics/20261010-mjlab-r11-cloud-training.md)。新策略行为验收、模型交付和H7真机验收仍需后续执行。不得将有限检查或随机未训练actor导出等同于策略可用。

旧r11→r12的 `--resume-disable-knee-limit` 入口仅属于历史r12源码；当前r13拒绝该入口及旧模型普通续训，避免同时改变门控却按膝惩罚单项迁移记录。同版本r13保存／加载可用；用户明确授权的r11→r13可用 `--resume-wide-height-gate` 同时记录门控和膝惩罚变化，恢复全部PPO状态。

## 历史

[r1](interfaces/chuanliantui-mjlab-huananhu-v14-flat-standup-r1.md)原V14移植、[r2](interfaces/chuanliantui-mjlab-huananhu-v14-flat-standup-r2.md)实体气弹簧、[r3](interfaces/chuanliantui-mjlab-huananhu-v14-flat-standup-r3.md)随机初态、[r4](interfaces/chuanliantui-mjlab-huananhu-v14-flat-standup-r4.md)动态模式、[r5](interfaces/chuanliantui-mjlab-huananhu-v14-flat-standup-r5.md)PD/ref/步长与闭链修复。历史记录按各自时点解释；日志与模型全部保留。

2026-10-10按用户要求恢复动态mode，r7有限检查通过，尚未启动新训练。已完成的r6 4096环境／1000轮训练保持原固定mode合同，详情见 [r7模式诊断](diagnostics/20261010-mjlab-r7-dynamic-modes.md)。

2026-10-10继续按用户要求恢复四腿在前、两轮在后的动作顺序，当时接口r8。动态模式与奖励／PPO配置沿用r7；r8未启动训练，历史模型保持原合同，见 [r8诊断](diagnostics/20261010-mjlab-r8-action-order.md)。

2026-10-10按用户要求关闭新增knee_limit奖励惩罚，当前配置r9。权重-0.5→0，其余奖励／物理／PPO沿用r8；用户授权4096环境／1000轮新训练，见 [运行记录](diagnostics/20261010-mjlab-r9-formal-training.md)。

2026-10-10按用户要求将初态改为参考v7的50%普通／50%地面、地面root零姿态，当前配置r10；其余训练设置沿用r9。本次未启动训练。

2026-10-10按用户要求恢复参考膝限位惩罚，当前配置r11；权重0→-0.5，余量0.15 rad，初态和其他训练设置沿用r10。本次未启动训练。

随后用户授权启动r11训练；已在远端4090以4096环境／1000轮、seed42、随机权重开始，详情见 [启动记录](diagnostics/20261010-mjlab-r11-cloud-training.md)。

2026-10-10用户要求关闭knee_limit，从r11完成1000轮的model_999追加2000轮；当前配置r12。只改变本项权重，使用显式迁移入口恢复actor、critic、Adam、学习率和累计进度，最终累计3000轮。

r12续训已在远端4090启动，actor／critic／Adam和计数恢复已核对，4096环境、从1000追加2000至累计3000轮，详见 [运行记录](diagnostics/20261010-mjlab-r12-cloud-training.md)。

2026-10-10用户要求改为宽高度门控，当前配置r13：仅将四项速度奖励门控3～6 cm→5～10 cm；严格起立验收仍3 cm，其他设置沿用r12。历史r12续训已完成且回放退化，见 [分析](diagnostics/20261010-mjlab-r12-degradation.md)。r13有限检查见 [诊断](diagnostics/20261010-mjlab-r13-wide-height-gate.md)，本次未启动训练。

随后用户授权从r11完成1000轮的model_999追加1000轮、累计2000轮；4096环境，宽高度门控、knee_limit=0。新显式迁移和远端运行见 [续训记录](diagnostics/20261010-mjlab-r13-cloud-training.md)。
