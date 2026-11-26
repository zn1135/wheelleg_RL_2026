# 2027 轮腿机器人训练移植

资源位于 `resources/robots/2027_wheelleg/`，环境位于 `mjlab_training/src/ct_mjlab/envs/2027_wheelleg/`。本次先接入复旦 PPO 行走任务 `2027_wheelleg_walk`，算法和网络复用公共 `rsl_rl/`，机器人、复位几何、物理控制与检查点身份独立维护。

旧机器人的五个注册任务集中在 `envs/chuanliantui/tasks.py`，相关行走、台阶、跳跃和保留的共用环境实际代码位于 `envs/chuanliantui/` 内。旧导入由 `envs/_legacy_imports.py` 映射到新位置，原任务名称继续使用。

## 模型与质量核查

| 项目 | 当前值或结论 |
|---|---|
| 原 CAD 总质量 | 14.606061706 kg |
| 当前 URDF 总质量 | 13.262079113 kg，车身使用用户确认的新CAD数据，左右腿质量/镜像惯量以右侧统一 |
| 当前车身质量 | 9.407 kg，来自用户确认的2027新CAD截图 |
| 当前左、右侧连杆组件 | 1.926020798 / 1.926020798 kg，按用户要求以右侧质量和镜像惯量统一，重心使用各自 CAD 安装坐标 |
| 新增气弹簧筒、杆质量 | 每侧 0.0305218 + 0.0923989 kg，沿用参考模型，待称重 |
| 训练模型总质量 | 13.507920513 kg |
| CAD 惯量矩阵 | 24 个连杆均满足正质量、正定和主惯量三角不等式 |
| 车身惯量来源 | 按新CAD截图中重心处数据换算，Ixx/Iyy/Izz为0.116086939/0.138426327/0.211918467 kg·m²；等效回转半径111～150 mm，已核单位与正张量符号 |
| 实机总质量 | 待整机称重核对 |
| 气弹簧 | 用户确认每根标称 400 N，当前恒推力近似；力—长度曲线、阻尼与实际伸缩行程待测 |

质量报告位于 `artifacts/2027_wheelleg/20261126-cad-base-mass-verified/mass_audit.json`。上述物理可行性检查只能排除明显无效矩阵，本次车身参数已与用户确认的CAD截图同步；截图显示精度之外的数值和实物总质量仍待原始精度数据/称重核对。

## 2027 模型接口

- 四个髋轴按用户确认设为连续转动。后支链六个被动关节随闭链自由转动。
- 两侧膝范围为左 `[0.033, 0.96]`、右 `[-0.96, -0.033]` rad。近零端按 001 止挡网格避开相交，另一端沿用 001 CAD 范围。
- MuJoCo 使用自由机身、四条腿部闭环和两条气弹簧闭环。训练资产包含六个直接关节力矩电机和两个内部气弹簧执行器。
- 动作顺序为 `lf0, lf00, lfwheel, rf0, rf00, rfwheel`。腿、轮限矩分别 40 / 3.9 N·m，轮子减速比记录为 15.7，动作按轮轴端力矩处理。
- 当前轮端使用单位力矩增益和 001 的 CAD 刚体惯量。减速后的转子附加惯量、传动摩擦与损耗待标定。
- 当前策略输入 25 维、五帧历史 125 维、环境 Critic 141 维，网络 Critic 再拼 3 维估速。初始机身高度由 001 几何计算，约 0.262 m。
- 当前先用 0.5 m/s 初始速度课程、0.24～0.28 m 高度命令。参数集中在 `envs/2027_wheelleg/config.py`。
- XML 内保留机体系 IMU 的角速度、加速度、四元数传感器。实际硬件安装方向待部署核对。

新任务使用独立检查点身份，原 chuanliantui 权重的直接恢复会被拒绝。H7 接口与负责人核对状态为待确认。

## 检查与训练命令

在仓库根目录执行。20 环境可视预检展示真实 GPU 状态，检查完成会关闭服务。可增加 `--steps 2000` 延长观察时间，零动作检查不执行学习更新。

```bash
PYTHONPATH=mjlab_training/src:. MUJOCO_GL=egl mjlab_training/.venv/bin/python   -m scripts.agent.wheelleg_2027.check   --viewer viser --steps 2000 --port 8097   --output artifacts/2027_wheelleg/visible-precheck-01
```

浏览器打开 `http://127.0.0.1:8097`，核对 20 个环境能正确复位、闭链相连、气弹簧随腿运动。报告中 `passed=true` 表示有限物理与接口检查通过，训练后的行走能力仍须独立验收。

训练前先将 `<轮数预算>` 替换为本次确定的正整数，并使用新的日志子目录。以下命令同时给出对应 TensorBoard 入口。

```bash
scripts/mjlab.sh train --task 2027_wheelleg_walk   --device cuda:0 --num-envs 512 --iterations <轮数预算>   --precheck artifacts/2027_wheelleg/visible-precheck-01/report.json   --log-dir logs/mjlab/2027_wheelleg_walk/20261126_first

mjlab_training/.venv/bin/python -m tensorboard.main   --logdir logs/mjlab/2027_wheelleg_walk/20261126_first --port 6006
```

训练后可以用 GPU 浏览器回放。将检查点路径替换为实际生成的文件。

```bash
scripts/mjlab.sh play --task 2027_wheelleg_walk --device cuda:0   --num-envs 1 --viewer viser --viser-port 8097   --checkpoint logs/mjlab/2027_wheelleg_walk/20261126_first/model_final.pt
```

## 本次验证范围

已执行质量检查、53 个迁移文件的函数体 AST 等价检查、旧导入身份检查、旧 8000 轮行走权重加载、原行走正式静态检查，以及新任务 20 环境 GPU 零学习检查。新任务检查覆盖注册入口、闭链复位、有限动力学、400 N 执行通道、网络形状、初始检查点网络参数往返与未训练检查点恢复拒绝。

正式 train CLI 已完成环境创建、运行器和新机器人资源快照检查，PPO 学习调用在检查中被拦截，更新数为0。play CLI 也已创建 GPU 浏览器回放服务并进入回放循环，使用随机检查网络作12秒受限检查。

本次未启动 PPO 学习。旧任务的全部行为回归、新机器人训练后的运动验收和真机验证待执行。


## 最新CAD车身同步记录

用户确认质量属性截图对应2027_wheelleg的最新完整车身。此次车身质量9.407 kg、重心[15.656,8.462,10.609] mm、重心处惯量采用kg·mm²乘10⁻⁶换成kg·m²。SolidWorks正张量交叉项取相反数写入URDF惯量矩阵；主惯量与截图Px/Py/Pz吻合。MJCF用主惯量及主轴四元数表达同一矩阵。

起始髋角重新按此车身重心计算，左前/左后/右前/右后约[-0.487548558,0.427590504,0.487548557,-0.427590505] rad，根高约0.261954 m。新20环境1000物理步的GPU预检及初始网络参数保存/恢复通过，0优化器更新。验证报告与截图位于artifacts/2027_wheelleg/20261126-cad-base-mass-verified/。

此前旧资产摘要对应的预检和检查点已属于旧物理版本，新训练使用当前资产重新预检产生的报告。
