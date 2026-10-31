# Isaac历史估速稳站任务 chuanliantui_stand_v2

辅助脚本已按用途迁移；下文固定旧实验保留其历史配方，旧路径查[迁移表](../scripts/agent/MIGRATION.md)，归档复核不代表当前版本验收。


2026-10-30后续改用[起立/行走两个独立任务](interfaces/20261030-chuanliantui-two-policy.md)进行新训练：`chuanliantui_getup`和`chuanliantui_walk`，25/125/71/6且无模式输入，只有独立CPU sim2sim做两模型切换。原stand_v2任务/旧32/160模型/历史命令保留，本页下文仍说明旧任务；新接口需重训，未启动训练或修改H7。

2026-10-29新增[停车位置外环v8](interfaces/20261029-stand-v2-position-hold.md)：`--position-hold`依赖yaw；原始vx=0且yaw_ready后，实测前向低于0.05m/s连续0.5秒才锁点，按当前航向前向误差输出死区0.02m、kp=1/s、上限0.1m/s的反向速度，再走原vx斜率。静止角色锁定后复用跟踪奖励，采样和晋级仍只看原运动角色；奖励权重及维度不变。只实现/冻结检查，未训练，H7待确认；接口内附从5999同时续训yaw和停车的命令及TensorBoard命令。

2026-10-29新增[转向与松杆航向保持v7](interfaces/20261029-stand-v2-yaw-heading-hold.md)：显式`--yaw-control`依赖站稳后速度课程；角速度默认±0.5rad/s、斜率1rad/s²，松杆锁定当前朝向并生成纠偏角速度。运动回合采样停止/直行/原地转/弯行，静止角色也可保持朝向。32/160/89/6维度与原奖励权重保持，命令契约升版；5999需先续训新配方，开启开关不等于已有转向能力。训练/TensorBoard命令及检查范围见链接。

2026-10-29新增首次起立的`nominal_state`缩放：原左右腿摆角差平方及权重-3保持，先做原限幅，再乘`0.1+0.9*clamp(standing_time/0.5,0,1)`；`has_stood=True`后系数锁为1，回合复位后重新从0.1开始。只复用现有站稳计时，不改成功门槛、观测、动作或物理。[正式有限检查](../artifacts/stand-v2-nominal-first-stand-20261029/check/result.json)通过、优化器更新0；原1999续训1000轮的完成与行为结论见[项目实验记录顶部](ai/project-log.md)。

2026-10-29新增[B独立窄核奖励v6](interfaces/20261029-stand-v2-tight-tracking.md)：显式`--tracking-tight-term`须同时开激活期权重；只在tracking_active时增加`0.5*exp(-(u-vx)²/0.02)`，宽核与其他项保持。关闭时仍原19项及日志键。仅实现与检查，无训练；按用户最新要求交付从A新4999续训B的命令。[检查、续训/TensorBoard与3999/A/B两斜率及零速v3.1配对命令](../artifacts/stand-v2-tight-term-20261029/README.md)。

2026-10-29新增[激活期间主速度奖励v5](interfaces/20261029-stand-v2-active-tracking-weight.md)：显式`--tracking-weight-while-active`（须同时开启速度课程），只把×5的条件从u非零改为tracking_active，急停归零仍保持有效权重1.0；整回合零速/恢复仍0.2，核分母0.25和其他项不变。profile/command保留v6，recipe严格隔离；仅实现与检查，无训练。[文件清单、检查退出码、用户自填轮数的训练/TensorBoard和三模型两斜率验收命令](../artifacts/stand-v2-active-weight-20261029/README.md)。

2026-10-29新增独立[±1.5m/s课程与限斜率配方](interfaces/20261029-stand-v2-speed-slew.md)：显式`--post-stand-tracking --speed-curriculum`，三级0.5/1.0/1.5m/s、回合内急走/急停/换向/保持、50%整回合零速，斜率每回合0.5～2m/s²；门控、奖励、地面初态和物理保持。只做检查，未训练；晋级不按轮数，1000轮不保证达到顶档。[检查、验收和3999续训1000轮及TensorBoard命令](../artifacts/stand-v2-speed-20261029/README.md)。

2026-10-28新增显式[站稳后50%速度配方](interfaces/20261028-stand-v2-post-stand-tracking.md)：全部地面起立，已站起且NORMAL额外保持0.5秒后，50%回合接受前后0.1～0.3m/s，失稳即清零；非零命令主跟踪有效权重1.0。维度/时序保持，新profile与原零速配方分开。训练用实验入口`--post-stand-tracking --iterations 1000 --min-std 0.05`；新模型回放同样加`--post-stand-tracking`，旧模型不加。[本次命令与检查](../artifacts/stand-v2-tracking-20261028/README.md)。

独立CPU MuJoCo入口`sim2sim/mj_sim2sim_chuanliantui_stand_v2.py`原2ms积分在初态17数值弹飞，回放已停；当前改为0.5ms积分（2000Hz）、500Hz PD/弹簧、100Hz策略/历史，使用原160维ONNX和保存的名义初态。128初态前5秒异常筛查通过，但CPU起立仍与训练引擎不一致，sim2sim行为未通过。训练时序和共享XML不变，当前仅已解锁0.22m、零速/零噪声/零时延，不是完整H7仿真。[接口与证据](interfaces/20261028-stand-v2-native-cpu.md)。

2026-10-27当前版采用32维单帧＋5帧速度编码器，并按用户要求改成“先直立、后精确高度”；奖励和成功课程已偏离Isaac原配方，见[本轮变更](interfaces/20261027-stand-v2-upright-first.md)。原39维v2接口保留在[历史记录](interfaces/20261026-stand-v2-isaac-recipe.md)，观测布局见[32维接口](interfaces/20261027-stand-v2-isaac32-history.md)。本任务不启用SMP，不自动加载旧任务模型。

## 当前参数

| 项目 | 设置 |
|---|---|
| 机器人、动作 | 原共享闭链XML，六实体电机；腿PD60/2、动作角度尺度0.5，轮速度增益0.1、尺度10；气弹簧名义曲线和轮端响应不改 |
| 频率与初态 | 500Hz物理／100Hz策略，100%原地面姿态库；复位前0.2秒零主动电机力矩 |
| Actor | 32维当前观测＋编码器输出3维＝35维网络输入；256/128/64 ELU，6动作 |
| 编码器 | 5×32＝160维，128/64 ELU→3维；监督目标为机体系质心速度×2，匹配Isaac速度缩放 |
| Critic | 89维＝原82维＋5个Isaac阶段/计时通道＋恢复观测计时＋总回合进度；只用于训练 |
| 历史 | 每策略步推进一次，最旧到最新；首帧/复位帧填满5格，局部复位不影响其他环境；100Hz下首尾相隔40ms |
| 初始探索 | Gaussian init_std=0.5；其余PPO沿用mjlab，lr1e-4、value系数4、entropy0.005、5epochs/4minibatches、24步采样 |
| 编码器训练 | 独立Adam lr1e-3、速度均方误差、梯度裁剪0.1；PPO使用detach后的估速，不更新编码器；编码器优化器随checkpoint保存 |
| 高度课程 | 初始目标0.30m、宽高度奖励；成功要求倾角≤15°、高度≥0.20m、机身接触≤0.1N及双轮达到承重门槛，连续50策略拍。4096个已完成回合窗口成功≥30%后目标切到0.22m；每次复位仍须先满足当前连续站稳0.5秒，才启用精确高度奖励，失稳立即退回宽奖励 |
| 奖励 | 19项；高度乘倾角门控：≤15°为1、15～40°按角度线性降到0、≥40°为0，保留机身接触折扣；高度公式exp(-(h-h_cmd)²/d)，宽d=0.01m²、精确d=0.0025m²。姿态项1+gravity_z、权重-1，不乘高度、不乘10、不做单项饱和；其余项仍加权限幅±1，统一乘dt，负总奖励不裁零 |
| 起立速度约束 | 首次站稳前lin_vel_z有效权重-0.1、ang_vel_xy为-0.02；首次站稳后恢复-1/-0.2。其余原有has_stood门控保留；不再有单独getup_inverted项 |
| 奖励屏蔽 | 按本次动作进入物理步前的policy_active锁存；0.2秒准备期全部奖励为0，PPO仍保留这些采样，不宣称等于Isaac样本屏蔽 |
| 起立前终止 | 未has_stood时倒置/双轮离地不终止；离地仍有原-0.3权重惩罚，倒置由连续姿态项扣分；非有限qpos/qvel仍立即终止 |
| 起立后终止 | has_stood后倒置立即终止；双轮连续无向上支撑0.2秒；机身接触>10N或重力Z>-0.1持续超过1秒 |
| 总超时 | 从复位起40秒，包含从未接地回合；超时回合计入课程，训练按time_out进行bootstrap |
| 恢复观测 | 独立于Isaac成功状态：倾角≤40°、机身高度>0.20m，且每轮支撑≥重量10%及5N、双轮合计≥重量80%，连续50策略步后NORMAL；任一不满足立即RECOVER；不以机身接触力直接门控 |
| airborne | 本任务继承状态强制0；新32维Actor不含额外4个状态标志，Critic中的对应槽为0 |
| 噪声 | 缩放前角速度±0.2、重力±0.05、四腿位置±0.02、六电机速度±1.5；历史保存实际带噪观测 |
| 观测/动作时延 | 观测四组独立每回合6/10/16ms；动作启动时0～10ms取整到2ms，腿轮共用；普通play关闭噪声及两类延迟 |
| 电机随机化 | 启动时Kp/Kd倍率0.95～1.05、力矩倍率0.8～1.2、名义角偏置±0.03rad；复位不重采样 |
| 物理随机化 | 机身附加质量-1～2kg、COM每轴±2cm、源等效刚体组质量/惯量倍率0.9～1.1；弹簧新增杆件不随机 |
| 接触 | 有效MuJoCo滑动摩擦0.20～0.30或0.45～0.55；源恢复系数经现有阻尼振子近似转换，不宣称PhysX动力学等价 |
| 外扰 | 接地后间隔3～5秒水平力脉冲0.1秒；未稳1～5N、当前站稳5～15N |
| 日志与兼容 | logs/mjlab/chuanliantui_stand_v2；新profile拒载旧v2/stand/multimode/Isaac模型 |

五种模式的通道保留普通／楼梯／斜坡／恢复／跳跃及跳跃目标、时间。当前任务仍是平地起立，实际只触发普通/恢复；加入模式槽不代表已经训练楼梯或跳跃。关节位置/速度和动作仍使用实体电机，不能把旧Isaac串联髋/膝权重直接套入。

当前任务训练的前向/偏航命令固定为0；有命令通道不代表已学会非零速度跟踪。仅做冻结模型能力探测时，可用`scripts/agent/stand_v2/assess_stand_v2_tracking.py --checkpoint <模型> --seed 42 --output <新目录>`（前置`PYTHONPATH=mjlab_training/src`并使用本仓mjlab解释器）。它保留地面起立8秒，再向已起立且NORMAL者注入0/±0.1/±0.3/±0.5m/s命令10秒，测10～18秒、末4秒归零；初态、训练代码和检查点不改。入口与独立检查见[构建与测试](ai/build-test.md)，实测结论只写项目实验记录。

## 检查与回放

20环境可视训练（示例预算100轮；需用户逐次授权）：

```bash
scripts/mjlab.sh train --task chuanliantui_stand_v2 --num-envs 20 \
  --iterations 100 --run-name visual20_100 --viewer viser --viser-port 8093
mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir logs/mjlab/chuanliantui_stand_v2 --port 8080
```

浏览器打开http://127.0.0.1:8093。训练入口仅显示正在采样的环境，按接近实时的速度推进；PPO继续独占环境步进，保留训练噪声与随机化。100轮为每环境2400策略步、24秒仿真，不足一个40秒完整回合；只适合观察动作和检查实际学习链路。训练结束后自动关闭画面服务。TensorBoard命令单独运行，训练不会自动启动该服务。

```bash
scripts/mjlab.sh check --task chuanliantui_stand_v2 --num-envs 4 --steps 32
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  scripts/agent/stand_v2/check_chuanliantui_stand_v2.py \
  --output artifacts/<新检查目录> --probe-envs 1024
```

正式检查覆盖：保留的15项Isaac奖励公式对照、4项新奖励的15°/27.5°/40°与倒置边界、站稳门控/失稳回退、主动控制屏蔽、起立前后终止与总超时、32/160/89/6、逐项顺序/缩放、局部历史复位、估速监督与PPO梯度隔离、优化器step被拦截的完整update路径、课程与模型续载、旧配方拒载、encoder＋actor ONNX数值对照。probe是零动作和std0.5随机动作的首回合诊断，不是策略训练或行为验收。检查checkpoint名含untrained，禁止当作训练模型。

末模型的首回合运动诊断（100Hz采样，不替代500Hz/20秒正式稳站验收）：

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  scripts/agent/stand_v2/diagnose_chuanliantui_stand_v2.py --checkpoint <本版model_*.pt> \
  --num-envs 64 --seconds 40 --output artifacts/<新诊断目录>
```

该工具保留任务play初态、关闭自动复位与学习，记录支撑力、机身接触、倾角、漂移、实体腿速和估速误差。末1秒平衡与另加低速度/低腿速/目标高度的组合条件分开报告；`final_quiet_1s=0`不能独自判为腿在抖动，也可能仅因高度未达标。只诊断有限种子，不宣称完整验收或训练随机化条件下成绩。

样本提前结束时冻结其首回合结果，仅复位已结束环境供其余样本继续步进；复位后的轨迹全部排除，失败样本不会重新计成功，实际复位步骤记在`diagnostic_resets_after_first_result_frozen`。

2026-10-27按用户“允许缓慢移动，但机体必须起立”新增独立验收口径`stand-v2-rise-balance-drift-separated-v1`：起立标记仍沿用当前任务的直立、双轮承重、机身离地连续0.5秒条件；最终还须在20秒预定结束前连续1秒满足双轮平衡v3.1（每轮≥重量10%且≥5N、双轮合计80%～120%、机身接触≤1N、max横滚/俯仰≤35°、横滚/俯仰角速度合量≤0.5rad/s）。按实际500Hz物理子步逐拍检查，数值异常或提前终止失败。漂移速度、位移和高度单独报告，不以旧0.05m/s静止门槛否决起立。曾倒置与最终失败分别记录，成功恢复后的早期倒置不新增一票否决。旧`final_quiet_1s`保留原义；不覆盖multimode或旧稳站任务的验收记录，不改变训练奖励、课程或部署接口。

```bash
PYTHONPATH=mjlab_training/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mjlab_training/.venv/bin/python -m scripts.agent.stand_v2.assess_chuanliantui_stand_v2 \
  --checkpoint <本版model_*.pt> --num-envs 64 --seed 42 --output artifacts/<新验收目录>
```

此入口只做固定种子的名义play地面起立验收，记录500Hz物理轨迹、100Hz完整姿态轨迹及参数未变核对。模型配对必须匹配版本、初态、质量、物理与时间协议，任一旧成功样本丢失仍判退化。不能把这项名义验收写成训练随机化或真机已通过。

接触力诊断仅统计，不参与恢复模式、观测、奖励或终止。新启动的训练和回放增加以下`Episode_Metrics/`指标，均按100Hz策略采样计算真实回合平均：

| 指标 | 含义 |
|---|---|
| base_contact_force_N | 机身接触力向量模长的平均值，单位N |
| base_contact_fraction | 机身接触力>1N的采样占比，范围0～1 |
| normal_base_contact_fraction | 同一拍接触力>1N且当前模式为NORMAL的占比，范围0～1 |

1N只是诊断阈值，不替代Isaac成功判据中的0.1N。独立报告追加逐环境接触时长/占比，以及末5秒出现过、持续出现（占比≥50%）的环境数。已有保存轨迹可以直接复算，不重新运行物理：

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  scripts/agent/stand_v2/diagnose_chuanliantui_stand_v2.py \
  --from-report artifacts/<已有诊断目录>/result.json --output artifacts/<新统计目录>
```

旧日志不能自动补出新指标。随后按用户要求在NORMAL规则追加轮端承重条件，见[轮端承重接口](interfaces/20261027-stand-v2-wheel-load-normal.md)。维度与单输入ONNX形状保持，但新增`recovery_contract`区分规则；旧model499直接套新规则有明显退化，普通回放/续训拒载旧规则模型。NORMAL未实施15°或随目标高度的退出方案；当前15°仅用于奖励成功与精确高度阶段。

轮端日志另有`left_wheel_support_N`、`right_wheel_support_N`、`wheel_support_weight_ratio`和`normal_wheel_support_fraction`。支撑力取世界Z向上的分量，不取接触力模长；重量按每个环境当前随机化质量和重力计算。机身撑地的诊断统计继续保留。

旧模型跨规则的诊断必须显式加`--allow-recovery-change`；该选项仅开放在独立诊断工具，不代表旧模型可直接部署或续训。新旧首回合诊断可以配对比较：

```bash
mjlab_training/.venv/bin/python -m scripts.agent.stand_v2.compare_stand_v2_wheel_normal \
  --before artifacts/<旧诊断>/result.json --after artifacts/<新诊断>/result.json \
  --output artifacts/<新比较报告>.json
```

比较默认检查模型/XML哈希、初始qpos/qvel和质量一致；比较不同模型时须显式添加`--allow-different-checkpoint`，不能将结果归因于单一改动。旧平衡样本有丢失时退出2并记录退化，不以新增样本抵消。

20环境可视初态预检命令（零动作，非训练）：

```bash
scripts/mjlab.sh play --task chuanliantui_stand_v2 --num-envs 20 --viewer viser --viser-port 8093
```

已有本版checkpoint后，用`play --task chuanliantui_stand_v2 --checkpoint <model_*.pt>`回放；用以下入口导出完整编码器＋Actor：

```bash
scripts/mjlab.sh export --checkpoint <本版model_*.pt> --output <新目录>/policy.onnx
```

ONNX只有一个输入：observation_history[1,160]，图内取最后32维作为当前观测；输出actions[1,6]和velocity[1,3]（机体系质心速度m/s）。外部仍按100Hz维护从旧到新的5帧历史，复位用首帧填满，并负责预处理和PD。历史上只改单输入的变更无需重训；本轮奖励/课程已升版，须交付新训练模型，不能仅重导旧模型。旧双输入ONNX的调用方须同步改为单输入。导出自动检查图输入形状，并对比原策略推理路径及ONNX输出，详见[单输入导出接口](interfaces/20261027-stand-v2-history160-onnx.md)。旧默认eval与固定0.27m验收不能直接作为本任务行为验收。

训练按用户逐次明确轮数预算执行，最新实验状态见[项目记录](ai/project-log.md)；训练入口要求显式正整数`--iterations`。本轮未改H7固件。H7的模型、32维构造、历史、模式判据与真实时序均待负责人适配/确认；仿真root高度不等于H7髋轮高度估计，不宣称两端已对齐。
