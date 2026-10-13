# 串联腿 mjlab 训练

当前默认为 `huananhu-v14`，按用户选择恢复最早华南虎V14 Flat-v0配方，见 [恢复说明](../docs/interfaces/chuanliantui-mjlab-scut-v14-flat-v1.md)。入口 `src/ct_mjlab/experiments/huananhu_v14.py`；以下为显式选择 `recover-r14` 的历史说明。r15仍可显式选择，已有模型按原版本续训／回放。

独立 Python3.12 环境；显式r14加入前进速度±2.7 m/s、偏航速度±2π rad/s、50%航向控制，使用5～10 cm速度奖励高度门控，knee_limit权重保持0，当前拒绝隐式跨版本续训，初态沿用r10的50%普通／50%地面、地面机体零姿态。动态normal/recover和四腿两轮动作顺序继续使用，保留本仓实体闭链机器人／气弹簧资产。原生mjlab管理器、Actor35/Critic78、普通PPO。

配置／参数／命令见 [训练说明](../docs/mjlab-chuanliantui.md)，接口见 [r14](../docs/interfaces/chuanliantui-mjlab-wheelleg-recover-r14.md)。根目录使用 `scripts/mjlab.sh`，默认只交付训练命令。r11已完成1000轮；用户授权r12从其model_999追加2000轮，已在远端4090完成续训；r13已按用户授权从r11完成追加1000轮、累计2000轮，见 [运行记录](../docs/diagnostics/20261010-mjlab-r13-cloud-training.md)及 [效果](../docs/diagnostics/20261010-mjlab-r13-cloud-model1999-evaluation.md)。r14尚未训练。

`src/ct_mjlab/reference_training/` 为当前训练实现，SOURCE_IMPORT.json记录参考脏工作区逐文件指纹。`environment.py`/`recipe.py`/`cpu.py`保留r5回放。禁止覆写／删除根logs。

r6～r12历史模型按原动作顺序、mode、膝惩罚、初态分布及高度门控回放；当前检查见 [r13诊断](../docs/diagnostics/20261010-mjlab-r13-wide-height-gate.md)，r10远端1000轮训练及效果见 [评估记录](../docs/diagnostics/20261010-mjlab-r10-cloud-model999-evaluation.md)。
