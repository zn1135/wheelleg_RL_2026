# 串联腿 mjlab 迁移检查

## 版本与范围

- 日期：2026-10-09，Asia/Shanghai；在当前仓库根目录执行。
- 分支：`26_wheelleg`；基础提交 `b4aea50f2b75a730eff0d6a7d993454e98c3be51`，工作区有未提交改动及新源码。本轮未暂存、提交、切换分支或创建 worktree。
- 被测47个改动／新增源码与配置文件的 SHA256 见 [源码清单](20261009-mjlab-source-sha256.json)；清单集合标识 `19944dd9b37ed5966dde56bfd6e2ba092bc7065652049b0f1fb13866a8d5db9b`。未修改的 mesh 由基础提交提供，文档不参与该标识。此标识不表示所有文件均有行为覆盖。
- 源训练任务：华南虎 `Robotics-Wheelbipe-V14-Flat-v0`，提交 `868467c3427fc00d34ebb1176683652d9597ba1c`；源 env/config/PPO/自定义事件源码 SHA 在 `scut_v14.json`，检查执行时逐项核对。
- 新接口及后端差异见 [接口记录](../interfaces/chuanliantui-mjlab-huananhu-v14-flat-standup-r1.md)、[迁移说明](../mjlab-chuanliantui.md)。初姿、零速度／.22 m 高度命令、机器人控制及150 N弹簧属于明确覆盖。

复用本机已安装 mjlab 的虚拟环境依赖，通过本仓独立 `.venv` 路径引用加载；没有修改外部机器人仓库、环境、任务或资产。实际版本：Python3.12.3、mjlab1.6.0、Torch2.14.0、MuJoCo3.11.0、MuJoCo Warp3.11.0、Warp1.17.0、RSL-RL5.4.2、TensorDict0.14.2、ONNX1.22.0、ONNX Runtime1.26.0；ORT只安装到本仓环境。GPU为 RTX4060 Laptop 8 GiB。`uv.lock` 已同步这些核心版本，但未执行全新机器的完整依赖安装验证。

## 最终代码已执行检查

| 命令 | 退出码 | 结果与范围 |
|---|---|---|
| `mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_mjlab.py --gpu --out docs/diagnostics/20261009-mjlab-check.json` | 0 | 华南虎原始奖励 AST 独立对照、观测布局、子步延迟、部分 reset、输出倍率先于固定限矩、CPU/GPU物理、runner构造、旧接口拒绝及ONNX比较 |
| `scripts/mjlab.sh check --num-envs 4 --steps 64` | 0 | 包入口、native mjlab 四环境64策略周期有限物理／观测／奖励检查，无学习 |
| `scripts/mjlab.sh --help` | 0 | train/play/eval/export/check入口可用 |
| 完整激活 Python3.8 Isaac Gym conda 环境后，`python scripts/agent/check_chuanliantui_observation.py` | 0 | 旧25维actor、125维历史、65维平地critic与旧MuJoCo观测构造契约；采用最小mock，不启动Isaac物理 |
| `uv lock --project mjlab_training` | 0 | 142依赖解析，Warp锁到本机已检查的1.17.0；未运行训练或安装全套依赖 |
| `git diff --check` | 0 | 无空白错误 |

完整机器可读结果见 [检查JSON](20261009-mjlab-check.json)。26项全部非零奖励对照的最大误差 `9.313225746e-10`。CPU60策略周期（1.2 s）中，最大销轴误差 `0.002199177338 m`、最大电机关节速度 `9.767216682 rad/s`；GPU四环境64策略周期（1.28 s），最大销轴误差 `0.002631696640 m`，均小于检查门槛5 mm。直接检查原始 qpos/qvel 与数值异常停止标志，未出现通过 reset 或 NaN 清理掩盖的异常。普通接地失败允许按源规则 reset，此检查不证明起立。

机体系持续外力转换使用独立 native MuJoCo 旋转矩阵对照，并检查部分环境 reset 后力缓存清零。部分 reset 不改其他环境的延迟历史。PPO runner仅构造，不调用 `learn()`、rollout采集或优化器更新；没有启动TensorBoard服务。随机未训练actor的ONNX经过图检查和32组数值对照，最大误差 `1.043081284e-7`，临时导出已随临时目录退出清理，不作为模型交付。

## 修复及前序检查

迁移期间修正了native API名称／状态dtype／sensor历史维度、物理子步延迟和加速度时序、轮位置加噪顺序、RSL-RL模型配置、接口序列化检查，以及输出力矩随机化、质量对应惯量、机体系持续外力、地面零恢复合成。文档逐项列出物理后端对应方式，不宣称PhysX与MuJoCo行为完全相同。

旧观测检查先因新增气弹簧导入缺mock、以及沿用过期的±11 rad起立初角预期失败。补齐mock，并按现有±1.566 rad初态修正检查后通过；未为通过检查更改运行初态。

10月8日阶段在完整激活的旧Python3.8环境中已通过：`check_chuanliantui_new1_train_urdf.py`、`check_chuanliantui_train_proxy.py`、`check_chuanliantui_closed.py`、`check_chuanliantui_gas_spring.py`、`check_gas_spring_asset_geometry.py`、`check_h7_observation.py`。气弹簧任意轴／RPY数学检查最大双精度力矩差 `8.88e-15 Nm`，既有27姿态×3推力对照最大差约 `9.82e-7 Nm`；详情及该阶段范围见 [气弹簧记录](20261008-gas-spring-asset-geometry.md)。这些不是策略行为验收。

旧 `check_chuanliantui_h7_control.py` 未完成：本机部署路径缺少该脚本要求的 `Algorithm/leg_solver.c`，退出非零。没有修改部署仓补造该文件；新实体接口也尚未适配H7。

## 未执行与交付结论

本轮完成代码移植和有限检查；**没有启动训练、TensorBoard、旧／新策略行为验收或真机测试**。初始根位置变化后的20环境可视预检命令已交付，尚未执行。未生成新训练权重，不能声称30秒起立／静站验收通过；CPU/GPU有限物理与随机actor导出只支持迁移接口和计算路径结论。H7模型、时序和实体零点适配仍待后续任务。

根目录历史 `logs/`、旧exports及外部mjlab机器人工作区均未修改。新运行入口和配套训练／TensorBoard、native/CPU回放、验收、导出命令见 [迁移说明](../mjlab-chuanliantui.md#可执行命令本次未启动训练)。
