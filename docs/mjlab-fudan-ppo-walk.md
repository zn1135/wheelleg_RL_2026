# 复旦 PPO 行走配方适配规格（fudan-ppo-walk-physical-v1）

2026-11-10强外载续训在10009发生物理NaN后，保留外载/随机化与原2ms积分/PD、10ms策略，仅将倒地连续等待100拍改为0，同角度阈值触发即失败结束，NaN仍报错停机。4096环境20.01秒对照及5项语义检查通过，已接续原预算剩991轮到11000；不代表任意倒地姿态的求解已修复。[终止语义](interfaces/20261110-fudan-fallen-episode-termination.md)、[恢复运行](../artifacts/fudan_ppo_walk/20261110-wide-wrench-resume10009-to11000-02/README.md)。

2026-11-10后续新增每回合机身各轴±10N/±10N·m持续外载，在机身质心施加，机身坐标保持、逐物理步旋转，与短推扰相加；局部复位仅重抽对应环境。训练/GPU新进程共用，当前9000→10000训练不变，旧无外载成绩不移作抗扰通过。[实现和验证](interfaces/20261110-fudan-episode-wrench.md)。

2026-11-10按用户要求回退：训练`sim.fitted_leg_load=False`，GPU默认`--leg-load-profile legacy`，恢复原线性气弹簧与原关节物理。拟合JSON和实现保留，只有显式`--leg-load-profile fitted`才回放旧候选；闭链和标定限位保持。[回退范围与验证](interfaces/20261110-leg-load-rollback.md)。既有拟合物理权重和原成绩不改写。

当前名义气弹簧弹性曲线为0～150N/90mm、CAD零位压缩10mm，并按用户确认保留50N·s/m阻尼；最终推力夹至0～150N，复旦训练及GPU评估的弹簧偏置均为0。两端共用`PhysicalAction`实际施力路径。实现与历史范围区别见[sim2sim约定](ai/sim2sim.md)，检查命令见[构建与验证](ai/build-test.md)。

2026-11-10用户选择只准备下一次扩大随机化，后续要求其余随机化也打开/扩大：训练动作6～14ms、电机倍率±15%、质量/惯量倍率±15%、附加质量−1.5～3kg、COM±30mm、接触/零位/噪声加宽、小推扰每7秒/0.3m/s；传感仍2ms。当前9000→10000轮保留原设置；新GPU评估延迟仍10/4ms，但共用推扰开启和随机组加大噪声，旧成绩不是同条件基线。[最新范围与检查](interfaces/20261110-fudan-full-randomization-next.md)。

2026-11-03用户选用**8000-v2＋停车+2°图外参考**，积分关闭。默认GPU评估省略`--checkpoint`即用该方案；显式指定检查点仍走原输入复现。配置见`FudanWalkCfg.inference`，[接口和未验证范围](interfaces/20261103-fudan-selected-pitch2.md)。九停车数值9/9，低高度平滑性有取舍，尚未完成45行走回归/987；不自动改训练或原ONNX。

训练统一进入`mjlab.sh train --task fudan_ppo_walk → scripts/train.py → task_registry → 任务环境/配置 → runner → PPO`。环境与集中配置在`envs/fudan_ppo_walk/`，训练算法组件在`rsl_rl/`；旧命令只作兼容转入。下文历史命令及记录按原日期保留，实际日常位置见[迁移与验证说明](maintenance/20261103-fudan-structure-alignment.md)，最新模型状态以[实验记录顶部](ai/project-log.md)为准。

2026-11-02新增[停车命令采样v1](interfaces/20261102-fudan-parking-commands.md)：`config.py/FudanWalkCfg.commands`默认10%原地保持、10%行驶3～5秒后停车至复位，停车回合覆盖0.22/0.26/0.30m；其余80%沿用原命令采样。两个比例设0可恢复原采样。只做代码准备与无物理检查，未训练新权重；16奖励、网络、PD与历史定义不变，固定评估命令不变。

当前最终评分改用[统一GPU验收](interfaces/20261102-unified-gpu-acceptance.md)：`scripts/mjlab.sh accept --task fudan_ppo_walk --recording <GPU目录> --output <新验收目录>`。运动门槛0.10m/s、0.10rad/s、停车峰值0.05m/s、俯仰5°，旧v2宽门槛保留历史，轮载/短触地/高度/矩阵保持。

2026-11-02按用户要求，复旦最终sim2sim验证统一使用GPU，CPU物理入口已删除，见[GPU最终验证约定](interfaces/20261102-fudan-gpu-validation.md)。当前`evaluate`默认GPU，也兼容显式`evaluate gpu`；原CPU成绩与下文首轮双引擎描述仅属历史，不是当前验收要求。训练／模型／物理和v2评分门槛保持。

2026-11-02新增[动作延迟、传感延迟、主动推扰三个独立开关](interfaces/20261102-fudan-disturbance-switches.md)：在`config.py`的`FudanWalkCfg.domain_rand`中直接配置，默认全关；预检／训练／GPU评估读取当前代码配置，预检须同条件；传感延迟开启必须填范围，实际延迟仍须实机测量。下文第一阶段“无额外延迟／无推扰”描述保留历史含义，不代表已开启新功能。仅实现与无物理检查，未新增训练。

当前开发输入为用户要求的[history回退v3](interfaces/20261102-fudan-history-rollback-v3.md)：首帧同一带噪/限幅后观测重复5次、后续历史保存限幅后帧；PD中心合法复位/热步/相位和奖励/控制/物理保持。只完成无物理/无模型前向/无更新准备，追加轮数待回复；现有7000模型仍为v2。下文有日期的v1/v2实验记录保持原义，不能将旧成绩算作v3行为。

**最新复核：CPU样本18确有弹飞，根高最高.7922m；有限数值/无警告不能代表动态可靠，暂不推进新训练。优先核对同初态2ms/.5ms CPU积分，尚未执行。** [已保存轨迹证据](../artifacts/fudan_ppo_walk/fudan-reset-v2-20261101/launch-diagnosis-r2/report.json)。

状态：2026-11-01，首轮1678轮有限试训结束；用户已指出机身撑地，视频验收不通过，且本轮存在下述训练入口偏差，不是完整参考配方复现。任务名 `fudan_ppo_walk`，独立命名空间；仅扩展当前三条维护线，不恢复五个旧 task。

## 范围与冻结来源

固定源码：[yly-true/fudan_rl_wheel_leg，8204e853](https://github.com/yly-true/fudan_rl_wheel_leg/tree/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane)。原文件及 SHA256 放 `artifacts/fudan_ppo_walk/fudan-ppo-walk-20261101/reference/`。本机同名源码不是该版本，不作权威。注册链：envs/__init__.py → WheelLeggedCfg/WheelLeggedCfgPPO → LeggedRobot → task_registry → OnPolicyRunner → ActorCriticSequence/PPO；配置须合并基类和子类。读取源码而不运行原 Isaac 脚本。

平地：前进、后退、原地转、边走边转、停车、变高；不含障碍、跳跃、倒地恢复、H7或真机验收。从零初始化，独立 PPO；不得暗接旧模型、SMP、BC、SuperMimic或旧奖励/外环。仅当前分支工作，无提交/推送，无H7/共享XML/质量/弹簧/极性改动，保留历史产物。

## 参考与本机适配差异

| 项目 | 固定参考实际调用 | 本机适配与限制 |
|---|---|---|
| 机器人 | infantry_V4_increase，六虚拟关节 | 共享 chuanliantui.xml 闭链、四实体腿电机和双轮；不能互载模型 |
| 当前帧/历史 | 25，5×25，最旧→最新 | 相同维度，实体槽语义独立版本 |
| 训练信息/Critic | 环境141，runner拼估速3→144 | 保留字段和网络，不用71维替代 |
| 网络 | encoder125→128→64→3；actor28→128→64→32→6；critic144→256→128→64→1；ELU | 保留；部署actor无真值线速度 |
| 优化器 | PPO Adam只管actor/critic/std；encoder另Adam；actor用latent.detach；critic存rollout当时latent | 保留；PPO五epoch×四minibatch，然后encoder五epoch×四minibatch；估速监督目标为机体系速度×2，MSE，encoder梯度限0.1 |
| PPO | 48步/环境，8192环境，393216转移/轮，minibatch98304；lr1e-3，自适应KL .005，γ.99/λ.95，clip .2，entropy .01，value系数1，梯度限1 | 环境数待显存实测；只缩并行规模，48步/四minibatch/五epoch和算法保留；需报告实际批量与更新数 |
| 初始探索 | 六维独立可训练std=.5，没有探索下限 | 保留，无旧walk std下限 |
| 时序 | sim .005×decimation2=.010s | PD/物理 .002×5=.010s；CPU若需细分积分须留证据，不改PD/策略周期 |
| 速度命令 | 每5s独立均匀vx[-2,2]、yaw[-2,2]、高度[.10,.20] | vx/yaw保留，机身高度改[.22,.30]；无互斥/限斜率/航向或停车位置外环 |
| 课程 | plane在_parse_cfg关闭terrain curriculum；每2000策略步复位时，当前复位组vx奖励均值>.7且yaw>.56才将全体vx范围每侧扩.1，封顶2.5 | 按实际plane分支保留，不用地形分支的.5增量，不用注释“80%”替代代码 |
| 复位 | 默认虚拟角[.2,.4,0,-.2,-.4,0]，root z=.1；六根速度均匀±.5 | 求本机合法闭链姿态并按接触几何放置，根六速度±.5；不承诺静止起步，不直接复制.1高度 |
| 实体PD | 虚拟腿20/1，轮速度D=.2；腿动作×.5、轮×10；限幅取原URDF | 沿用本机实体腿60/2、轮.1/0；腿目标中心取本机微蹲默认角，目标限XML参考角±3.14，轮目标±150rad/s；输入力矩腿±40、轮±3.9Nm，轮XML G/V/p响应及气弹簧保留 |
| 本机默认角 | 无对应实体常数 | [lf0,lf00,rf0,rf00]=[-.6047205544565015,.6083241726212976,.6047205544560037,-.6083241726229305]rad；PD与观测减角同源；XML ref不是默认角 |
| 动作 | 正态采样，进入环境clip±100；上次动作与奖励用裁剪后值 | 保留±100；不得继承SuperMimic±3 |
| 传感速度 | dof速度由每物理步角差/dt；根线速度由每策略步位置差，转到当前机体系 | 对实体电机和本机根参考点实现同样采样定义；PD使用本机当前物理速度；此差异单列 |
| 虚拟腿对称 | theta0左右差平方；不是电机角直接相减 | 用髋轴→轮轴向量转机体XZ面，theta=atan2(x,-z)；不得把lf00冒充虚拟膝 |
| 关节限位奖励 | 原URDF虚拟髋膝范围内缩97% | 用户已确认：用本机真实前髋lf0/rf0与被动膝lf1/rf1现有限位，内缩97%；保持原−1及逐项裁剪，不改XML、不加入旧膝余量奖励 |
| 摩擦 | 启动时64桶，机器人[.6,1.4]；地面.5 | 保留采样；MuJoCo接触组合/恢复系数非PhysX同一模型，必须列明近似并验证，不声称逐位复现 |
| 其他随机化 | 恢复系数[.6,1]、base加质量[-1,2]kg、COM每轴±.02m；各body质量和惯量同乘[.9,1.1]；Kp/Kd/力矩倍率[.95,1.05]、默认角偏置±.03 | 作用于本机对应实体；不引入旧spring/推力/0.8～1.2强度随机化；闭链复位必须满足几何约束 |
| 延迟/推扰 | 动作delay=False；无单独传感队列；push=False | 第一阶段额外动作延迟=0、额外传感延迟=0、无主动推扰；不读取旧6/10/16ms队列 |
| 导出 | 两输入obs[25]/obs_history[125]，一个actions[6]，opset17 | 独立接口与元数据，模型哈希和物理哈希，旧模型拒载；不含PD/噪声/历史维护；H7待确认 |

## 观测与信息来源

Actor帧槽：0:3机体系角速度×.25；3:6重力投影；6:9命令[vx×2,yaw×.25,机身高度×5]；9:13实体腿相对默认角[lf0,lf00,rf0,rf00]；13:19实体速度[lf0,lf00,lfwheel,rf0,rf00,rfwheel]×.05；19:25上次裁剪后六动作（同动作顺序）。噪声为独立均匀：角速度±.05、重力±.05、角度±.02、速度±.075；命令和动作无噪声。整帧clip±100。

环境训练信息141维：0:3真值机体线速度×2；3:28无噪声当前帧；28:34上拍动作；34:40上上拍动作；40:46六电机加速度×.0025；46:123共77个平地高度观测clip(z−.5−地面高度,−1,1)×5；123:129六PD力矩×.05；129base质量减批均值（按参考质量随机化前记录语义）；130:133 COM偏移；133:139六默认角偏移；139摩擦采样值；140恢复系数采样值。网络再加当前encoder三维估速，合计144。高度77槽平地可相同，但不能删去导致接口变形。critic在actor加噪声之前构造。

## 奖励、终止、验收分别定义

2026-11-03停车奖励试验已按用户要求撤销，删除新增的`parking_velocity`及其尺度配置，当前恢复下列原16项奖励。9000轮的最终PT已删除，训练日志、指标、视频和源码快照保留为失败试验记录。当前选用的8000-v2＋停车+2°图外参考保持，积分关闭。

奖励16项：tracking_lin_vel=1，tracking_lin_vel_enhance=1，tracking_ang_vel=1，tracking_ang_vel_enhance=1，base_height=1，nominal_state=−1，lin_vel_z=−1，ang_vel_xy=−.2，orientation=−100，dof_vel=−5e−5，dof_acc=−2.5e−7，torques=−.0001，action_rate=−.01，action_smooth=−.01，collision=−1，dof_pos_limits=−1。每项先乘系数×.01再夹到[−.01,.01]，总和允许负值。基础跟踪exp(−误差²/.25)，enhance=exp(−误差²/2.5)−1；高度exp(−误差²/.001)。nominal_state为左右真实腿角差平方；速度罚四腿，动作二阶差分仅四腿，加速度/力矩/一阶动作差分为六路。参考collision列表为空，实际为0，不擅自加机身接触惩罚。

参考训练终止：projected_gravity.z>−.1连续超过1秒（参考contact终止列表为空）；超20秒超时并bootstrap。验收则对动作全程摔倒、提前终止、数值异常保留失败；即使最后站起也不能覆盖。奖励上涨、存活、程序检查均不代表行为合格。

## 当前技术验收：fudan-walk-stage1-v2

2026-11-02用户确认独立[验收v2](interfaces/20261102-fudan-walk-acceptance-v2.md)：20ms轮载均值、95%有效窗、连续100ms否决；原始双轮同时<1N达20ms仍否决；首次2s内允许受限短触地，之后20ms机身力均值>1N否决。原跟踪/姿态/停车门槛保持，旧瞬时轮载/接触只作v1诊断。只改评分，不改奖励/物理/模型；命令为`score_v2 --recording <既有目录> --output <新JSON>`。原协议与成绩保留如下，不把reset/history版本与评分版本混为一谈。

## 最终用户验收与技术协议：fudan-walk-stage1-v1

**用户最新规则优先：最终验收通过必须由用户看完真实视频并明确确认无问题。当前和自动报告只给技术门槛达标/未达标/缺测；数值达标、助手自查和用户沉默都不是最终通过。确认前固定标为“待用户视频验收”。本变更不改奖励/物理/训练配方/预算。** 视频须正常速度、完整含起步与刹停，覆盖前进/后退/原地转/行驶转弯/变高/停车，标注指令/实际速度/高度/时间/模型/样本，地面参考用于辨别漂移；当前仅GPU提供，有失败则包含失败样本。覆盖不足或预算不足如实标注。

最终仅使用mjlab / MuJoCo Warp GPU；独立原生CPU物理评估已撤销。训练种子1；阶段诊断种子30001，不计正式通过；正式留出种子50001（固定名义物理）及51001～51020（20组完整随机物理），训练不访问它们。每个case固定1个名义样本+20个随机样本，GPU逐个执行、保留原分母。不因预算缩减矩阵或放宽门槛。

固定矩阵：三档高度.22/.26/.30，各含零速、vx=±.25/±2/±2.5直行、yaw=±2原地转、(vx,yaw)=(±1,±1)四种行驶转弯，共39case；各高度前进和后退±1刹停共6case；变高站立/前进1m/s分别.22→.30→.26→.22，共2case。共47case×21=987个GPU样本。每个常值段先响应2秒，再测5秒；变高每段均7秒、28秒完整；停车先±1行驶7秒，从t=7发零vx/yaw起3秒内完成连续1秒水平速度≤.1，最早完成点作为停稳时刻，随后再观察5秒低速，并记录相对该点最大水平漂移。停车最长15秒，缺完整窗口判失败。

参考点：base_link原点，平地z=0，速度测量用机体系base_link瞬时vx，yaw用机体系角速度z；根位置差分另存用于参考奖励/估速监督，不与验收瞬时速度混写。停车速度为世界水平平移速度模，漂移为世界XY欧氏距离；高度为base_link世界z−地面高度，非腿长/质心高度。roll/pitch从base_link姿态计算。原始轨迹至少100Hz；接触/姿态峰值按2ms控制采样记录，不用时间平均掩盖超限；各窗口左闭右开，严格以整数tick选择。

速度：MAE≤.20m/s且≥95%测量时间绝对误差≤.30m/s；转向同样.20/.30rad/s；高度MAE≤.03m且≥95%时间误差≤.05m。响应期峰值单列。稳态/停车测量窗口max|pitch|≤10°。停车观测5秒每拍水平速度≤.1m/s，不设置漂移通过门槛，不称定点保持。

每case最后连续1秒同时满足v3.1：每轮向上支撑≥max(.1mg,5N)，两轮合计[.8mg,1.2mg]，机身接触力≤1N，max(|roll|,|pitch|)≤35°，机体roll/pitch角速度合量≤.5rad/s；yaw不算失衡，少量腿接触不独立否决。全程数值有限且完整，无摔倒/提前终止。动作过程摔倒固定判据：任一控制采样机身接触力>10N或max(|roll|,|pitch|)>60°即判失败；重新站起不能覆盖。稳态俯仰10°与上述条件联合，不能只报告末段平衡。

每样本所有适用条件联合通过；每类别固定名义100%，随机≥95%。前进、后退、原地转、行驶转弯、停车、变高分别汇总，GPU逐类别汇总。缺测/失败均留原分母；复位第二回合不得替代第一回合。生成逐样本轨迹、指标JSON/汇总、正常速度完整视频，成功和失败存在时各保留代表，样本ID/模型/种子/时间/哈希一一对应。没有成功样本就写无成功视频，不伪造。取得合格固定集合后，后续候选不得丢失任何原通过样本。后续.10/.20速度及5°俯仰只登记目标，不改现有项目5°总标准，也不自动收紧本轮。

## 首轮历史一小时预算（已结束，不授权新运行）

源码阅读、文件编辑、无物理推进/无优化器更新的逻辑与静态几何检查不计；所有新物理预检、仿真、训练、阶段和最终评估均计连续墙钟≤3600秒。首个物理运行前公布Asia/Shanghai绝对起止时间、命令和配额并保存审计。暂定预检300秒、训练1800秒、阶段检查120秒、最终两引擎和渲染1380秒，按小规模耗时向评估倾斜，不扩总预算。20环境可视预检通过才启动训练；任何数值/程序错误停止并保留现场。初期行为差不提前停止固定训练配额。阶段存盘不授权加训；截止停止更新并交付未完成项。TensorBoard仅提供实际logs子目录命令，不自启。

## 后续实机延迟测量（用户新增硬约束）

本轮零额外延迟仅为实验条件，既不表示真机延迟为零，也不表示旧默认队列已经实测。后续必须在真实运行链分别测：传感采样/滤波/通信导致的数据龄；策略取输入与推理耗时；指令排队、传输、进入PD/驱动生效的延迟和抖动。各项须明确起止点、时间戳、时钟同步与可测边界；控制周期不是延迟，发送时间不是电机实际响应时间。以测量分布及独立留出实测数据确定仿真固定延迟/随机化范围，再重训或验证适用性。当前仅登记，不授权H7改码、连接/操作真机或硬件试验，不扩大3600秒预算。H7负责人核对：待确认。

## 实现入口与明确复位适配

实现位于 `mjlab_training/src/ct_mjlab/fudan_ppo_walk/`。专用CLI为 `scripts/mjlab.sh fudan-ppo-walk`；普通SMP的 `train` 路径保持不变。参考network/PPO/storage/utils保留版权与算法，只改包内导入；运行器另行实现期限、计数、两优化器存盘与独立导出。

本机初姿固定为几何求解的0.26m闭链姿态＋2mm接地间隙、单位姿态四元数；六根速度独立±.5。默认角随机偏置±.03保留在PD中心与相对观测，初始闭链坐标不直接加四个独立电机偏置，以免破坏约束；这是物理闭链复位适配，与参考“复位到随机默认角”不同。首帧使用当前复位传感状态清零上次动作，并重复填满历史；不复制原参考可能遗留的终止帧动作/缓存。额外传感延迟仍为0。

本轮名义验收无噪声、无物理参数随机化，但仍按冻结种子采样±.5根速度；随机验收包含原范围物理/电机随机化及观测噪声。每样本独立SeedSequence([seed,case_id的SHA256前32bit])采样初态与参数，GPU保存初态、实际物理参数与噪声条件，供同后端复现和历史比较。正式验收禁用训练20秒超时，观察到冻结case结束；摔倒/数值失败始终记录，禁止失败后复位替代。

正式验收命令见本轮预算文件；下面是固定入口（期限参数为绝对Unix秒，须由当前授权预算计算，不能复制旧期限重新训练）：

```bash
# 不推进物理/不更新策略的检查
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.check --output <新静态目录>
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.check_scoring
# 必须先给出的20环境可视预检
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.precheck --viewer viser --port 8768 --deadline <预检截止秒> --output <新预检目录>
# 实测并行规模，零优化器更新
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.probe --num-envs 512 --deadline <预检截止秒> --output <新probe.json>
# 仅预检与规模门槛通过后
scripts/mjlab.sh fudan-ppo-walk train --num-envs 512 --log-dir logs/mjlab/fudan_ppo_walk/20261101_trial01 --precheck <预检report.json> --deadline <训练截止秒>
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/fudan_ppo_walk/20261101_trial01 --port 6006
# 冻结模型后两个引擎分别测量；阶段检查加--diagnostic，不算正式验收
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.evaluate gpu --checkpoint <model.pt> --output <新GPU目录> --deadline <截止秒>
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.score --traces <引擎目录/traces> --output <新score.json>
scripts/mjlab.sh fudan-ppo-walk export --checkpoint <model.pt> --output <新导出目录>
```

阶段模型每600秒另存；本轮确认显存余量后，在训练不中断时对第523轮模型进行了独立20例双引擎诊断，实测GPU49.17秒、CPU7.52秒，合计小于120秒阶段配额；正式留出样本不参与训练/课程。训练运行器仅按训练期限停止，不按初期行为表现早停。

## 本轮明确实现偏差与修复（不得隐藏）

补读固定提交scripts/train.py后发现，参考实际调用learn(init_at_random_ep_len=True)，本轮runner遗漏了随机错开首轮episode长度；这不是用户授权的配方差异，而是实现遗漏。本轮1678轮模型保留原样，只能作为带该偏差的适配试训候选。没有通过再次训练掩盖问题。后续源码已增加0～1999的随机episode计数，并同步真实环境计数；初次当前Actor帧与历史末帧也改为共用一次复位观测，避免此前启动阶段两次独立噪声采样。正式冻结评估本来已统一首帧。

修复后的入口仅通过不推进物理/不更新优化器的回归检查；未用修复版重训，也未授权追加。原训练源、原始模型、源码哈希、导出和两引擎轨迹均保留。本轮模型的原训练实现以logs/mjlab/fudan_ppo_walk/20261101_trial01/source/ct_mjlab为准，不能把当前修复版源码冒充其训练来源。

## 本轮结果入口

[完整报告/视频/逐样本指标](../artifacts/fudan_ppo_walk/fudan-ppo-walk-20261101/README.md)、[模型交付记录](model-deliveries/20261101-fudan-ppo-walk-trial01.md)。最终GPU987完整、CPU984完整/3数值提前结束，两个引擎联合技术均0/987，用户视频验收不通过。CPU0.5ms只对原3个数值失败样本完成诊断，3例完整/0数值异常，未替换原分母、未做该积分方案的全987例验证。

`check_initialization`已在无物理/无优化器更新夹具中调用真实FudanEnv.step/reset，验证500拍重采样、严格>2000超时、局部复位、全局total_steps课程边界和默认CLI。实际保存往返只验证模型/优化器/RNG载荷；没有完整恢复入口或完整环境状态，不宣称中途续训支持。旧模型仍对应原run/source快照，修复源码未追加训练。

## 用户验收反馈与当前调查范围

用户观看后反馈“机身撑地啊”，本轮视频验收**不通过**，不再是待确认状态。速度跟踪或程序运行成功不能代表合格。当前仅用原轨迹和固定复旦8204e853源码/资产逐项对照；暂不推荐或实施新增接触惩罚、旧站立经验/奖励或其他补丁。预算已结束，不新增仿真/训练。

## 只读来源审计结论（用户观看后）

[完整固定复旦来源与实现差异审计](diagnostics/20261101-fudan-fixed-source-audit.md)已完成。公开ONNX在8月18日入库，早于8月22日当前plane配方变更；训练来源未能绑定，不能据公开回放声称当前配方避免撑地。原URDF限幅为四腿50Nm、两轮5Nm；接触惩罚/终止列表原本为空，姿态项饱和也与参考一致，不能当作本次遗漏。

D1启动零动作热步、D2复位角与PD中心/默认偏置关系、D3正常reset的4净＋1噪历史及动作/缓存语义、D4历史clip顺序均只记录最小对齐建议，**尚未实施**。此前修复只解决首轮相位及首次current/history二次采噪声，不等于完整reset语义已恢复。清动作、刷新缓存、重复带噪帧等不能因已写入实现/文档就认作用户已授权；需单独决定。此次新物理/模型前向/优化器更新均为0，未修改奖励/终止/物理，也未追加训练。

## 当前源码修正：reset/history v2（未新试训）

用户后续已授权按参考关系实施初始化/历史修正。现有PD中心含每环境启动±.03偏置已可静态解出合法闭链并缓存，复位零动作P项为0；root初态约.330～.348m，目标仍.22～.30。默认训练顺序为reset→一次零动作10ms热步→随机首轮相位；正常reset为4净1噪，raw history先于current clip。保留本机fresh reset/清动作，不复制原缓存副作用。配置/预检/CPU/GPU输入路径已同步，新增reset_contract明确隔离旧trial01。

[接口与检查范围](interfaces/20261101-fudan-reset-v2.md)、[正式结果及未运行训练命令](../artifacts/fudan_ppo_walk/fudan-reset-v2-20261101/README.md)。用户后续另授权23:32—23:42最多10分钟预检：20环境初态合法、20样本CPU/GPU同状态无噪声观测误差0、两端各2秒数值有限，默认启动热步＋48采样实际通过。两端零动作均触地，CPU部分翻倒；无平衡策略参与，不能等同修正版策略失败。更新0、新训练未运行，旧训练预算不恢复，旧模型仍不通过。

## 2026-11-02用户新授权：从零3000轮

用户最新要求直接训练3000轮。新预算以完整轮数为准，无新墙钟上限；专用入口支持`--iterations 3000`（与`--deadline`互斥）。seed1/512×48，目标73728000转移、PPO/encoder各60000次Adam；旧CPU弹飞仍未解决，不将本轮GPU训练当作CPU通过，也不隐式改积分或奖励/PD/接触参数。真实命令、TensorBoard、新logdir和当前状态见[本轮记录](../artifacts/fudan_ppo_walk/fudan-ppo-walk-20261102-iter3000/RUNNING.md)。本轮不自动开展额外仿真验收。

3000轮现已完成并核对：73728000转移、PPO/encoder各60000次更新，训练循环53分19秒。未做新行为评估或导出、CPU弹飞未解决；外层子训练退出码未捕获，不声称退出0。[最终模型与核对](../artifacts/fudan_ppo_walk/fudan-ppo-walk-20261102-iter3000/README.md)。

## 2026-11-02逐项奖励日志

新训练记录`Rewards/<16项名>`，值是每轮48策略步×全部环境的加权/裁剪后平均贡献；`Train/reward`与控制台reward同步此口径，各项和应与总量一致。横轴step为完成训练轮数。旧3000轮Train/reward是每轮末步批均值且未存16项，历史不补写、不与新均值冒充同口径。TensorBoard用SCALARS→Horizontal Axis=STEP；Relative列只是耗时，不是迭代轴。[实现、事件检查与截图](../artifacts/fudan_ppo_walk/fudan-reward-logging-20261102/README.md)。本次未启动新训练或仿真。

## 2026-11-02学习状态续训与4096环境

用户新授权从3000轮继续新增3000轮并指定4096环境；CLI `--resume <checkpoint> --iterations 3000`的iterations为新增量，step累积到6000。严格恢复模型、两Adam/自适应学习率、累计计数、速度课程及CPU/CUDA RNG；物理、历史、命令、局部回合重新初始化，完整环境状态仍未保存/恢复。旧512源的全局拍数按原源样本数还原，不因4096变化错算。新增589824000转移、累计663552000，两Adam各新增60000/累计120000；未改奖励/PD/物理。16项日志在新run的3001～6000轮生效。实际命令、TensorBoard和来源/检查见[续训记录](../artifacts/fudan_ppo_walk/fudan-resume-3000-to6000-20261102/RUNNING.md)。不持续监控、不自动追加评估。

4096续训已于2026-11-02 05:39完成6000轮，新增3000轮/589824000转移、PPO及encoder各60000更新，累计663552000转移/两Adam各120000；训练/审计退出0，16项日志3001～6000完整。训练分数改善但尚无新回放，CPU弹飞未解决。[完成模型与结果](../artifacts/fudan_ppo_walk/fudan-resume-3000-to6000-20261102/README.md)。

2026-11-02用户再次授权6000→7000新增1000轮，继续4096环境、同配方与16项日志；新增196608000转移/两Adam各20000，累计860160000/各140000。只继承学习状态、仿真现场重置，不持续监控或自动评估。[本轮授权和新logdir](../artifacts/fudan_ppo_walk/fudan-resume-6000-to7000-20261102/RUNNING.md)。

4096续训6000→7000已于2026-11-02 13:43:45完成，训练/审计退出0；新增196608000转移、两Adam各20000，累计860160000/各140000，16项日志6001～7000完整。7000行为尚未回放，6000原13个诊断成功样本仍须保留。[最终模型与结果](../artifacts/fudan_ppo_walk/fudan-resume-6000-to7000-20261102/README.md)。
