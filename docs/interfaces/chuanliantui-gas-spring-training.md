# chuanliantui：训练气弹簧物理条件

| 项目 | 内容 |
|---|---|
| 日期、实施、复查 | 2026-10-03；Codex 实施；独立子 Agent 核查弹簧数值与施力合成 |
| 范围 | chuanliantui / chuanliantui_standup 的串联训练与串联 MuJoCo 代理 |
| 训练仓 | zn1135/wheelleg_RL_2026，26_wheelleg，3e5b1d88c0825784b4b25a5a5d7f1f0dfc6d02c5 加既有及本轮未提交改动 |
| 物理条件标识 | chuanliantui-serial-gas150-r1；不改变 25 维策略接口 |
| 权威实现 | GasSpringGeometry.knee_torques、Chuanliantui._apply_external_forces、代理 MJCF 生成器及 mj_sim2sim_ct.py |
| 部署仓 | 本轮不修改 H7 源码或固件；不是板端接口发布 |

## 差异

旧训练没有气弹簧；新训练与串联代理默认每侧 150 N 恒定伸张力，端点来自 CAD
闭链 `chuanliantui.xml` 的四个 gas_spring site。Torch 根据训练 URDF 膝轴、关节原点及
当前物理膝角计算力矩，每个 2 ms 物理子步更新。上下连杆施加等大反向膝轴扭矩；
这是串联理想铰链的广义力等效，不模拟弹簧杆自身质量或闭链约束载荷。

被动气弹簧与电机限矩、力矩随机倍率、动作延迟独立，起立接管前即生效；
与水平推扰合并为每子步一次刚体力/扭矩提交。电机日志和能耗奖励保持主动电机含义，
弹簧另记 `gas_spring_knee_torque_nm`。150 N 是现有仿真假设，实测推力尚待核实。

观测 25、历史 125、latent 3、动作 6；关节顺序、零位、反馈、PD、命令与周期不变。
初始位置、重置和策略接管条件不变。闭链模型原有弹簧仍保留，未修改 H7 控制实现。
后支路惯量及虚拟膝反馈差异不在本轮改动范围。

## 兼容性与回放

旧 `model_*.pt` 可加载，但物理条件变化后行为不保证保持，需要续训 encoder 和 actor。
推荐基点为 `Oct03_14-27-59_standup_h022_knee005_push/model_12000.pt`，
SHA-256 `dcc3e8d6b9ddbb2d777ed9e0af3f9936fac1508314c962967ef9d8438b134ebe`。
本轮没有产生新权重，也未启动训练或 TensorBoard。

训练开关：`ChuanliantuiCfg.gas_spring.force_n`（0～150 N）；
Isaac 与 MuJoCo 回放开关：`--gas_spring_force`。显式 0 恢复旧无弹簧串联条件，
旧起立配置快照继承当前父类，不能单凭快照文件名认定气弹簧已关闭。
代理新增两路独立 tendon actuator，六路策略电机仍按名字查询，模型输入输出不变。

## 验证记录

本轮证据目录：`/tmp/ct-add-gas-spring-20261003.a9qDNf/`。
工作区中间快照和差异为 `before-source.tar.gz`、`before.patch`；最终源码、差异和哈希
为 `final-source.tar.gz`、`final.patch`、`source-sha256.json`，包含复现所需的未跟踪代码。

- `check_chuanliantui_gas_spring.py`：27 姿态 × 0/75/150 N，Torch float32/64 对照
  MuJoCo 闭链/串联 tendon 及长度差分；最大误差 3.60×10⁻⁷ N·m。
  验证旋转后膝轴、上下反向扭矩、每子步单次合成、零推力清理和电机限幅分离。
- 代理生成器、代理检查、既有 `check_standup_pushes.py`、网络 `--selfcheck`：均退出 0。
- 使用上述 12000 轮旧权重，Isaac 20 s、5 次推扰、6 ms 延迟，150 N/侧与0 N均无失败重置。
  MuJoCo 串联代理15 s，150 N/侧与0 N均为有限状态；5 s后无机身接地或双轮同时离地。
- 0 N 两组回放的状态、动作与先前无弹簧报告逐样本完全一致。

第 5～15 秒均值（Isaac 含上述推扰/延迟，MuJoCo 无推扰/延迟，不能视为完全相同场景）：

| 回放 | 每侧弹簧力 | 高度 | pitch | 前向速度 |
|---|---:|---:|---:|---:|
| Isaac | 0 N | 0.2243 m | −0.509° | −0.00003 m/s |
| Isaac | 150 N | 0.2426 m | −1.272° | +0.2413 m/s |
| MuJoCo 串联 | 0 N | 0.2227 m | −0.293° | −0.00329 m/s |
| MuJoCo 串联 | 150 N | 0.2405 m | −1.867° | +0.2300 m/s |

以上验证弹簧实现及关闭后的回归，没有证明旧策略在新物理条件下站稳；旧权重仍持续前移，
需要在有弹簧的环境续训。完整逐步记录及与旧报告的差值见证据目录 `summary.json`。
正式续训、新权重验收、板端时序和真机行为未执行。

后续同姿态反馈、闭链软约束偏差、30 秒无推力和共同推力的结果，见
[分阶段核查记录](../diagnostics/20261003-chuanliantui-gas-standing-validation.md)。
该记录仍使用旧 12000 轮权重，不代表随后气弹簧续训的新模型效果。
