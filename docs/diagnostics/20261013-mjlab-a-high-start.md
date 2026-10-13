# A2500高位初态诊断：还原完成，初始化门禁未通过

2026-10-13，北京时间。**按用户停止条件，在0.27 m的64环境初始化门禁失败后停止。没有运行20秒策略回放，不能判断“保持／降回／摔倒／混合”，没有训练方案建议。** 三组完整qpos已按原求解器约定还原并保存；没有自行求解或编造姿态。

## 输入与还原证据

模型为`artifacts/height-response-a2500-20261012/models/model_2499.pt`，SHA256为`706c8387cba4815b491be3f5914a5951881c2966edabbd5fbb90d0da9c9d3dbd`，与指定值一致；原`recovery_factors.validate_checkpoint(...,'baseline')`校验累计2500更新。没有加载优化器或调用actor回放。

姿态来源为`artifacts/wheel-support-xml-height-check-20261012/jobs/20261012-xml-wheel-support-height-v5/xml-height-audit.json`，SHA256为`369352f9d3a7b3ab5e1c07676cb3f657385c31581b9b42a331d669ca2fb9009e`。完整qpos保存于[static-qpos.json](../../artifacts/high-start-a2500-20261013/static-qpos.json)。

使用原`scripts/agent/check_chuanliantui_xml_support_height.py`的`Audit.set_pose()`：`x=[joint_positions[n] for n in LEFT]+[radians(root_roll_deg)]`，高度取`target_height_m`。由原求解器填入nominal、根节点位置和滚转四元数、左右对称关节；轮角为0。没有重新运行优化求解。求解器SHA256为`a16f3a99157d6a4c380d48a66bcd82ffbe5667eba59f8c8f55d04f3820a855d2`，poses.json SHA256为`5d8af638a1e240e22e9e22ee4b781419972ebd99610def7a9a471938f5d83d90`。

完整qpos为25维：根节点7维＋18个标量关节；模型joint计数包括自由关节共19个。与源JSON比较的是其中16个求解关节，两个轮关节来自原求解器nominal约定。

| 目标高度 m | 完整qpos维数 | 16关节最大误差 rad或m | CPU最大销轴误差 m | 左／右轮底高度 m |
|---|---:|---:|---:|---|
| 0.27 | 25 | 0 | 2.101457829×10⁻¹³ | +8.283651542×10⁻¹⁴／+8.376632721×10⁻¹⁴ |
| 0.30 | 25 | 0 | 2.041824352×10⁻¹³ | −2.046973702×10⁻¹⁵／−1.873501354×10⁻¹⁵ |
| 0.34 | 25 | 0 | 1.988602950×10⁻¹³ | −2.775557562×10⁻¹⁷／−1.387778781×10⁻¹⁷ |

三组轮底高度与源文件逐位一致。表中CPU数据是原求解器几何核对，不是mjlab GPU接触承重验收。

## mjlab初始化与失败门禁

WSL Ubuntu-24.04，`/home/administrator/.venvs/wheelleg-mjlab/bin/python`，RTX4060 Laptop GPU、cuda:0，mjlab1.6.0／MuJoCo及mujoco-warp3.11.0／Torch2.14.0／Warp1.17.0。每次64环境，没有超过256环境。

配置使用`factors.make_env_cfg('baseline',64,42,play=True)`，名义物理、无噪声／延迟、零速度、关闭自动复位与命令重采样。逐个读取`robot/<joint_name>`的qpos地址，包括`robot/floating_base`的7维；根节点xyz加各环境origins。核实terrain为plane、z=0，各origins的z也为0。当前CPU和场景地址恰好相同，但代码逐名映射，未直接复制下标。

写入零qvel，清空外力和warmstart。四腿动作历史全部槽填`(q−nominal)/0.5`，两轮为0；同步原始上一动作和物理控制动作。0.27 m的policy顺序为lf0／lf00／rf0／rf00／lfwheel／rfwheel，上一动作是`[0.596749783,−0.310829163,−0.596749783,0.310829163,0,0]`。随后通过原PD动作应用一次当前姿态目标与原气弹簧力，将entity effort写入sim ctrl；这一步不推进物理时间。再forward、sense并重置传感／观测历史，不使用全零腿动作历史。

所有接触量均失效缓存后同步读取原terrain probes。闭链误差直接取六对`robot/<site>`世界坐标距离。逐名写入的qpos与按模型float32类型表示并加origins后的目标误差为0；这不代表GPU派生位置具有CPU float64的几何精度。

| 姿态 | 双轮接地通过 | 总轮承重80%～120%通过 | 非轮承重≤5%通过 | 闭链误差<10⁻⁶ m通过 | 动作历史 | 门禁 |
|---|---:|---:|---:|---:|---|---|
| 0.27 m | 25/64 | 0/64 | 64/64 | 56/64 | 当前姿态动作，非零腿历史 | 失败，停止 |
| 0.30 m | 未运行 | 未运行 | 未运行 | 未运行 | 尚未写入mjlab | 前一门禁失败后停止 |
| 0.34 m | 未运行 | 未运行 | 未运行 | 未运行 | 尚未写入mjlab | 前一门禁失败后停止 |

0.27 m详细值：

- 整机重量131.400878906 N；双轮合计支撑／重量的范围为0～0.306763232，即0%～30.6763232%，不满足80%～120%。
- 左轮向上支撑0～20.433795929 N，右轮0～19.875335693 N；两轮范围分别取极值，不能将两个最大值直接视为同一环境总支撑。
- 非轮向上支撑全部为0 N；qvel全部为0。
- 最大销轴误差范围1.332800394×10⁻⁷～1.067905032×10⁻⁶ m；超门槛环境为零基1、2、5、6、57、58、61、62。
- 当前六个电机ctrl全部为0；两个气弹簧ctrl均为53.617469788 N。

完整64环境接触、承重、误差、动作、ctrl及名称映射见[initialization.json](../../artifacts/high-start-a2500-20261013/gates-v1/initialization.json)。

## 已验证、推断和未能验证

**已验证：** 还原出的CPU姿态与原求解器记录一致。GPU逐名写入检查、terrain和origins检查通过，但首次forward的接触／承重／闭链门禁未通过。因此初始化门禁失败不能当作A2500策略摔倒，也不能据此给出保持或降回比例。

**已验证的初始化与静力求解差别：** 源0.27 m静力平衡记录要求电机力矩按lf0／lf00／lfwheel／rf0／rf00／rfwheel为`[−6.557309157,+6.868746236,+0.101780074,+5.434777456,−5.611003685,+0.165741450]` N·m，求解支撑为69.557206492／61.843663533 N。用户要求的姿态动作`(q−nominal)/0.5`只让PD目标等于当前关节位置，在零qvel下实际初始电机ctrl为0；它没有施加上述静力负载力矩。源“存在静力平衡解”和当前“首次forward的实际解”是不同检查。

**推断：** 没有施加静力求解的电机力矩可能影响当前承重；CPU float64→Warp float32以及带不同xy偏移的派生几何计算可能参与销轴误差和近零轮底接触的差异。当前数据不能分解各自贡献，也不能将全部失败直接归因为名字或下标映射错误。

**未能验证：** 上述因素是否是此次门禁失败的具体根因；0.30／0.34 m的GPU门禁；A2500的高位保持、降回或摔倒行为。没有增加下沉、预热、外力、loaded-action偏置或其他改动来通过门禁，没有放宽阈值。

## 原任务结果与视频状态

3种姿态×2种命令×2种扰动的12项20秒回放，以及普通初态对照，全部未运行。高度时间序列、返回0.20±0.02 m时间分布、末5秒平衡／高度跟踪、物理失败时刻、膝限位比例和actor动作对照均无有效结果，不补造数值。每种姿态的“保持／降回／摔倒／混合”分类及比例均未能验证。没有生成视频。

## 文件与执行记录

新增文件仅在授权位置：

- [diagnose_chuanliantui_high_start.py](../../scripts/agent/diagnose_chuanliantui_high_start.py)：目前完成还原与初始化门禁入口，尚未启用回放阶段；`--output`必须是新的输出子目录。
- 本报告。
- `artifacts/high-start-a2500-20261013/static-qpos.json`、`launch-gates-v1.sh`、`gates-v1.log`、`gates-v1.exit-code`以及`gates-v1/initialization.json`、`gates-v1/run.json`。

实际执行命令：

```powershell
wsl -d Ubuntu-24.04 -- bash -lc 'bash /mnt/d/IMCA/RL/Wheel-Legged-Gym/artifacts/high-start-a2500-20261013/launch-gates-v1.sh'
```

`.sh`内部实际调用：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=mjlab_training/src MUJOCO_GL=egl TZ=Asia/Shanghai \
  /home/administrator/.venvs/wheelleg-mjlab/bin/python \
  scripts/agent/diagnose_chuanliantui_high_start.py --gate-only \
  --output artifacts/high-start-a2500-20261013/gates-v1
```

Python、`.sh`退出码文件及外层命令均为1。北京时间2026-10-13 01:07:37.388205～01:07:42.895948，日志有明确门禁失败与异常栈，run.json为exit_code=1；不是被引号或变量展开掩盖的“退出0”。所有`$变量`、`$?`只保存在`.sh`内部，未通过长字符串直接传给wsl.exe。

既有experiments／reference_training／判据和agent脚本的运行前后哈希一致，见[run.json](../../artifacts/high-start-a2500-20261013/gates-v1/run.json)。未修改资产、模型、AGENTS.md、历史logs或部署代码，未启动训练／TensorBoard，未做Git操作。

Claude复跑时应复制`.sh`为输出目录下的新文件，并把其中log及output名称都改为新名称，再执行该文件；原产物不能覆盖。复跑入口目前仅支持初始化门禁；先核对三组还原和0.27 m门禁，不应以此声称已经完成12项策略诊断。
