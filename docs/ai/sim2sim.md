# sim2sim 与部署契约

2026-11-14从半幅1000扩大动作/传感延迟至各10～20ms继续到总3000，P20/D1与PD200/1ms积分/100Hz策略保持。仅启动器进程覆盖，源1000学习状态完整恢复，物理/历史重置；H7/Jetson未改，不在硬件人为叠加训练延迟。[接口记录](../interfaces/20261114-fudan-pd200-delay-continuation.md)。

2026-11-13新增仅本次进程的PD200训练候选：积分1ms、PD每5拍、策略每10拍，腿20/1、轮0.1/0、动作与传感分别10～20ms，关闭外力/矩/推扰，从零3000。共享默认/旧模型仍原时序；新checkpoint保存并恢复实际时序，预检/轨迹不再硬编码5子步或2ms时间轴。H7未改且待核对，未行为通过。[接口](../interfaces/20261113-fudan-pd200-scratch3000.md)。


2026-11-13轮矩响应范围实现已准备但未采用：`sim.wheel_motor_time_constant_range_ms`默认None，固定时间常数默认0；显式范围时每机器人双轮共用、每次局部复位仅重抽对应环境，整回合固定，限矩后按2ms一阶更新。机械轮仍拒绝额外输出增益、纯轮延迟及额外限矩，PD/策略周期不变。预检、PT、恢复报告与GPU评估记录`wheel_response`，旧模型缺字段恢复无附加响应。25～35ms实录轮速误差反而增大，用户已选择暂不启动原500轮，不把该范围设为默认或部署参数。[诊断反证](../../artifacts/real2sim/model3000-usb-pd500-20261113-01/wheel-torque-short-replay-01/README.md)。

2026-11-13实录gap参数诊断与五连杆传力核对仅新增独立检查：去弹簧短窗误差增大，局部候选虽短窗/闭环改善但未同时通过长同目标对照，不写默认或当前训练。XML仍含真实主动电机及闭链约束，不应把电机PD再次乘树形到电机的雅可比；10样本CPU几何虚功检查不替代GPU/实机受力验收。共享XML、轮机械参数、观测/PD/时序/H7保持。[结果及适用边界](../../artifacts/real2sim/model3000-usb-pd500-20261113-01/gap-study-01/README.md)。

2026-11-13复旦共享domain_rand关闭持续外力/持续外力矩和主动推扰，因此新训练及新GPU默认评估都无外加扰动；参数随机化、动作/传感延迟开关独立保持。历史带外扰采集/成绩按原条件保留，不能根据当前配置重标旧数据。无观测/动作/控制时序或H7改动。[1500轮试验](../../artifacts/fudan_ppo_walk/20261113-mechanical-noexternal-scratch1500-01/README.md)。

2026-11-12用户要求用机械参数替代轮曲线与0.45：复旦训练/MuJoCo Warp GPU默认wheel_model=mechanical，轮单位力矩驱动，轮关节armature/阻尼/摩擦读wheel_mechanical.json。旧曲线及gain0.45需显式legacy_curve，历史结果/模型保持。共享XML、其他任务、旧CPU入口和H7未改；新PT/夹具记录实际机械参数。站立/停车对照接近但历史较大脉冲轮速误差增加，不称全工况标定。见[机械轮接口](../interfaces/20261112-fudan-mechanical-wheels.md)。

2026-11-12复旦轮P恢复.1/D0，新增独立轮端响应层，默认gain1/无额外延迟/滞后/输出上限；四腿/XML保持。仅诊断使用gain.45等候选，未新训练；14179仍按原.045训练物理解释。新夹具记录响应参数，旧数据不改写，CPU/H7未接入。[定义与验证](../interfaces/20261112-fudan-wheel-motor-response.md)。

2026-11-11用户将轮端45%候选纳入复旦训练：wheel_stiffness .1→.045、D0，当前两倍腿惯量保持；新启动普通GPU评估使用同配置。历史参数探针显式固定旧轮P.1，避免再乘候选倍率。腿P60/D2仅本次13679→14179续训进程，旧模型按封存物理解释；H7未改。[范围](../interfaces/20261111-fudan-wheel045-training.md)。

2026-11-11用户采用四主动腿关节两倍armature：共享XML`lf0/lf00/rf0/rf00`均为0.03159kg·m²，新启动MuJoCo训练/回放生效。其他物理/PD/时序保持，未加入PD速度噪声或6ms滞后。20环境GPU加载/有限积分检查通过；11700新物理停车子集仍未全面通过。旧模型/录像按原物理保留，外矩配对最终录制已先完成，H7未改。[默认惯量约定](../interfaces/20261111-default-leg-armature-x2.md)。

2026-11-11用户再将四腿标称PD设为30/2，起点明确为改PD前11179，新增1000轮到12179停止；roll/pitch17项配方与其余物理／随机化保持。共享复旦配置影响新训练与GPU默认，旧P60/D2模型和成绩按其原条件保留，未改H7或交付新权重。[本次准备](../../artifacts/fudan_ppo_walk/20261111-p30d2-resume11179-01/README.md)。

2026-11-11用户要求四腿PD恢复P60/D2，已更新共享复旦配置，原倍率随机化保持，新roll/pitch分项奖励保留。用户指定改PD前11179检查点，续训2000轮到13179；无H7改动或新模型交付，旧P30/D10训练与诊断按原记录解释，不改标签。[本次准备与运行](../../artifacts/fudan_ppo_walk/20261111-p60d2-retrain2000-01/README.md)。

2026-11-11撤回固定2°停车奖励实现，复旦新训练改用roll/pitch独立惩罚，仅前倾0～+5°弱罚1/20。真实重力输入及评分不变，GPU按PT的`posture_reward`恢复奖励，缺字段旧PT仍为原16项合并惩罚；既有8000图外补偿路径保留。无新训练/ONNX/H7交付。[语义与兼容](../interfaces/20261111-fudan-roll-pitch-soft-band.md)。

2026-11-11按用户要求，复旦训练与GPU评估的四腿电机标称P/D改为30/10，保留原倍率随机化；轮增益、限矩、控制周期、闭链/气弹簧/限位及其他任务不变。旧P60/D2模型、轨迹及指纹保留，新PD续训使用独立日志；H7未同步、负责人核对待确认。实际预算与控制参数见[新PD续训记录](../../artifacts/fudan_ppo_walk/20261111-p30d10-resume11179-to14000-01/README.md)。

2026-11-10扩大随机化/持续外矩下，步行训练将倒地等待100→0拍，同阈值倒地即失败重置，NaN仍停机；已恢复剩991轮，未改2ms积分/PD或10ms策略。正式GPU回放不自动重置失败样本，统一评分标准保持，不能把更早训练重置当通过。[调整、验证和限制](../interfaces/20261110-fudan-fallen-episode-termination.md)。

2026-11-10再新增回合持续六轴外载：机身各轴力±10N、矩±10N·m，逐物理步转世界坐标施加于质心，与原短推扰相加；局部复位只重抽选中环境。GPU夹具按case/seed保存body外载，当前训练不变。[接口/验证](../interfaces/20261110-fudan-episode-wrench.md)。

2026-11-10用户进一步要求其余随机化也打开/扩大，下一次训练的物理/零位/噪声范围加宽并开启7秒/0.3m/s小推扰。已有实现共享推扰开关与随机组噪声，新GPU评估也随之改变，动作/传感仍10/4ms；冻结工况/物理抽样代码保持不表示全部评估条件保持。当前1000轮不变。[最新范围与验证](../interfaces/20261110-fudan-full-randomization-next.md)。

2026-11-10后续仅为下一次训练准备更宽随机化：动作6～14ms、Kp/Kd/力矩倍率0.85～1.15，训练传感2ms；当前9000→10000进程保持原10ms/±5%，GPU评估仍10/4ms与原冻结矩阵，不因新训练范围扩大而改旧验收分布。[范围和检查](../interfaces/20261110-fudan-wider-randomization-next.md)。

2026-11-10按用户要求回退双腿负载拟合：共同PD动作、复旦训练/GPU评估及原生MuJoCo默认关闭`fitted_leg_load`，恢复原线性气弹簧及原关节摩擦/阻尼基线；闭链、实测限位、轮模型不变。下方11-09默认采用描述为历史记录。参数JSON、实录、模型和Word保留，H7未改。[回退范围与验证](../interfaces/20261110-leg-load-rollback.md)。

2026-11-09后续用户明确正式采用双腿镜像局部负载：PD训练/复旦GPU/原生Playback默认开启同源负载和独立记忆；物理JSON与代码共同构成模型，XML仅机构/限位/原弹簧不变。复旦`--leg-load-profile legacy`回退，显式旧诊断包先关闭默认，防重复。启动随机化在新基线上叠加；新检查点保存profile与参数SHA。0训练/硬件操作，默认8000模型20例仍0/20，不称独立real校准通过。下方同日“未采纳默认”是本次决定前的历史状态。[正式采用与边界](../interfaces/20261109-adopted-mirrored-leg-load.md)、[Word流程与待办](../diagnostics/20261109-双腿负载拟合流程与训练仿真接入.docx)。

2026-11-09新增显式局部负载试验：复旦GPU`evaluate --response-parameters <冻结包>`按`sides`选择右腿或双腿镜像，保留原8000-v2/停车2°、PD/策略时序和轮模型。双腿同初态20例技术19→0、有轮载中断，未采纳为训练/默认物理；左腿是右腿镜像，记忆初力0、本轮2ms不同于原架空.2ms。省略选项、共享XML及H7保持。[施力/镜像检查与限制](../interfaces/20261109-fudan-length-fit-probe.md)。

2026-11-04用户要求将已验证闭链约束/实测限位同步训练与仿真：共享XML六connect设solref=.0004/1、solimp=.999/.9999/.001/.5/2，关节限位继承同组响应；保留lf1[-.97,.14]、rf1[-.16,1.01]rad。当前2ms物理/PD与10ms策略、PD/气弹簧/质量/接触保持，refsafe令有效时间常数至少4ms，不能等同架空0.2ms积分精度。训练注册线/复旦GPU/原生仿真实际读到新值，20,214姿态0越界；20环境接触测试最大间隙2.778→.051mm，8000+2°同20例GPU回归19→20、无旧通过退化。策略轨迹100Hz采样最大间隙4.056→.10644mm，略高旧.1mm诊断门槛，未写完整闭链精度通过。更硬候选间隙.12086mm且新增起步触地失败，已拒绝并恢复第一组。不移植固定机身EFC排序内核，不并入未通过的弹簧/摩擦拟合；H7/网络权重保持，未训练，完整987待验。[设置、版本与限制](../interfaces/20261104-shared-closed-chain-stability.md)。

2026-11-04现有实录联合拟合继续：质量/惯量/零点/极性/力矩增益固定，仅诊断副本估计弹簧参考力/斜率/阻尼与带初始储能的阻力。初载由第一姿态/请求及本候选弹簧推导，未来位置只评分，禁止自由调每轨迹初载。大幅与原8秒联合选优，四条1kHz单候选GPU自由回放均未逐轴达标；综合误差或同目标较近不等于四角通过，不导入训练。模型推力/阻力为等效参数，600N仅诊断允许范围，非实测最大力。共享训练弹簧和XML/固件保持，标量旧四字段response-parameters入口不支持新候选，使用报告所列单候选批量GPU入口。[联合拟合和负结果](../../artifacts/real2sim/h7-joint-response-fit-20261104-02/README.md)。

2026-11-04四关节响应复核：用户确认气弹簧150N为厂家标称、装配约预压10mm；旧0～150N/90mm共享公式仍保留，但不能据额定力认定其刚度/最大力正确。新“参考力+压缩增力”仅在离线诊断副本运行，300N为候选执行器允许值、非实测上限/固件保护。固定候选在多轨迹逐轴验收均失败，不采纳、不改真实量程/力矩增益，也不写训练；实际产品曲线待核实。详见[响应重拟合报告](../../artifacts/real2sim/h7-joint-response-refit-20261104-01/README.md)。

2026-11-04保护修复后，专用腿长实际越工作域仅限制轴向力方向、允许返回，目标域/40Nm不变。输出类型4flags bit6标识此版；`replay_leg_test_gpu`及批量反馈回放按位选择新返回控制或旧越界锁止，旧数据不能套新控制。旧8秒双分支GPU16,000环境步兼容复查位置max差4.381e−6rad/m、两次弹簧公式误差0N，未重拟合，不撤销新轨迹响应失败。此前冻结清单保留旧运行时哈希，后续新实录应建新冻结清单，不能改旧清单冒充来源相同。[固件变化与验证范围](../../artifacts/real2sim/h7-leg-soft-boundary-20261104-01/README.md)。

2026-11-04后续新速度/反向实录已否定原响应候选的跨轨迹适用性：冻结同一参数、35.92秒预保护段GPU同力矩综合角误差24.83°、同目标8.92°，闭链max.00941mm；真机前关节明显活动而模型近乎不动，不能只按同目标腿长较近认定对齐。机身重力方向max变化.148°；平移未测。末次伸腿触发H7既有0.33m保护后用户退出/断动力，完整预设协议中断。未重拟合/未写训练，共享XML/弹簧/参数保持；原0.74/.90°仅指旧单向数据。当前H7_ros为临时完整USB采集格式、遥控LQR仍暂关，恢复待办，USB已释放。[新数据及GPU失败证据](../../artifacts/real2sim/h7-response-validation-20261104-01/README.md)。

2026-11-04已有单向收腿实录得到响应候选：固定机身GPU同力矩/同目标综合角误差.74/.90°，最大瞬态4.78°，额外15秒保持末端max1.005°。候选包含紧闭链、等效摩擦/阻尼，以及仅lf00预滑动状态和反射惯量修正；未写共享XML/训练/固件，也未证明物理参数唯一真实。`--response-parameters`仅诊断副本应用，JSON/配套状态运行时不可省略，不能把固定机身候选当自由机身地面验收。[参数、逐项检查与完整命令](../../artifacts/real2sim/h7-response-align-20261104-01/README.md)。下一步独立速度/反向输入验证，不直接覆盖历史导出指纹。

2026-11-04实测膝限位GPU对照已按用户要求使用训练共享线性弹簧+50N·s/m阻尼。初态越界消除，但同目标四角综合误差14.57→14.48°，直接力矩38.41→154.93°且连接变形达9.7mm，未复现真机。源XML/固件在本轮保持；正式命令须显式`--spring-model linear-damped`，新模型用`--calibrated-model`受控选择；旧`linear`无阻尼，默认constant为历史诊断。[结果与限制](../../artifacts/real2sim/h7-calibrated-limits-gpu-20261104-01/README.md)。

2026-11-04串联腿共享XML两膝按USB实测活动覆盖更新：`lf1=[−0.97,0.14]`、`rf1=[−0.16,1.01]`rad，8端点各侧重复和全部实录/参考静态检查通过；人工重复误差max1.96mm，未做GPU动力回放，不能当行为验收。零点/极性/H7保持，旧快照/指纹不改。[实测范围与边界](../../artifacts/real2sim/h7-joint-limits-20261104-01/calibration-final/README.md)。后续GPU用默认源范围，不加会覆盖新范围的旧`--extend-knee-limits`诊断参数。

2026-11-04按用户确认保留阻尼：当前mjlab训练（含复旦PPO）与本机正式sim2sim的名义气弹簧弹性项已为线性0～150N，`x=max(0,0.010-q_slide)`、`F_elastic=clip(150*x/0.090,0,150)`，总力再叠加原有50N·s/m阻尼并夹至0～150N。复旦任务弹簧力偏置为0；其他任务的既有随机化保留。来源为`ct_mjlab/envs/chuanliantui/gas_spring.py`、实际`HuananhuPdAction`及`sim2sim/mj_sim2sim_huananhu_multimode.py`，无需再次把其改成线性。下文恒力150N叙述属于旧Isaac/串联代理和独立台架对照的历史范围，不能套作当前mjlab训练的模型。实录诊断脚本的无阻尼线性对照也不能称与当前训练动力学完全一致。

2026-11-10用户指定复旦训练/GPU仿真动作延迟均为10ms（五个2ms物理拍），传感保持训练2ms、GPU仿真4ms；控制周期和H7保持。历史实测2+2ms与4+4ms成绩不改写。[当前设置与验证](../interfaces/20261110-fudan-action-delay10ms.md)。

最终验收现统一为GPU、共用运动门槛及报告，见[统一GPU验收](../interfaces/20261102-unified-gpu-acceptance.md)。主线统一入口为`scripts/mjlab.sh accept --task <任务> --recording <GPU记录> --output <新目录>`；起立单列，旧CPU流程仅保留历史含义。

复旦`fudan_ppo_walk`当前最终验证仅使用mjlab / MuJoCo Warp GPU，原生CPU入口与活动动态诊断已删除，见[GPU验证约定](../interfaces/20261102-fudan-gpu-validation.md)。其他任务的CPU sim2sim保持；下文复旦CPU描述仅为历史实现，旧结果不改写。

复旦任务新增[三个独立延迟／推扰开关](../interfaces/20261102-fudan-disturbance-switches.md)。GPU夹具保存实际2ms延迟拍数和推扰种子／尺度，Native按夹具执行同一条件；当前`config.py`与CPU夹具不匹配时拒绝。默认全关及旧模型记录语义保持；新开关的动态效果未验收，H7未改。

2026-11-02复旦[history v3](../interfaces/20261102-fudan-history-rollback-v3.md)只改变历史填充和clip顺序；GPU评估按模型v2/v3标记选输入，Native按对应夹具标记构造，导出保留源模型实际标识。当前只完成无物理/无模型前向检查，CPU弹飞未修复，未新仿真/训练/部署。该版本不适用于下文其他任务的25维模型。

2026-10-30 getup当前恢复原四腿EMA80平滑（对应旧训练线已清理）：100Hz混合新动作80%与上一拍已平滑动作20%，raw观测/ONNX保持，等待期/复位清缓存、轮子不滤波；100ms实测角接管已撤回，图外执行及JSON恢复原v2滤波配置，不再要求锁存角与接管计时。当前奖励v4（对应旧训练线已清理）仅在训练中启用起立前力矩罚，部署不计算奖励。独立CPU仍未实现EMA并明确拒载；H7未改，滤波适配/负责人核对待确认，未交付新权重。

2026-10-30起立profile v2恢复训练高度课程0.30→0.22m；mjlab单模型play使用checkpoint保存阶段。独立CPU双策略仍固定起立0.22m，明确拒绝未解锁的v2模型，已解锁模型沿原路由；旧v1模型需对应历史源码。其余观测、ONNX、PD/物理及行走不变，H7待确认。课程记录（对应旧训练线已清理）。

双策略行走v2新增`--height 0.22～0.30`，指定本次回放固定行走高度；起立始终0.22m。训练仅复位时随机高度，不随机切换高度段；CPU不自动随机，输入vx/yaw原始请求互斥。当前CPU命令版本v2，检查行走profile/命令版本和高度范围，getup v1接口保持。[参数与交付](../interfaces/20261030-chuanliantui-two-policy.md)。

2026-10-30新增[单任务训练、仅CPU双策略切换](../interfaces/20261030-chuanliantui-two-policy.md)：`chuanliantui_getup`和`chuanliantui_walk`均25维帧/125维历史、无模式输入。独立入口`sim2sim/mj_sim2sim_chuanliantui_two_policy.py`读取各自PT对应的两个ONNX，按共同物理条件交接，保持FIFO/上次动作/PD，清航向及停车基准。CPU沿用0.5ms积分/2ms PD/100Hz策略，训练2ms物理不改。当前只有未训练夹具的观测对齐、有限CPU/切换检查，没有双策略行为通过或新训练模型；旧32/160 CPU入口不接新模型，H7待适配。

2026-10-29稳站v2新增停车位置外环v8（对应旧训练线已清理）：训练命令项使用仿真位置真值；独立冻结评估支持按实际模型轮轴符号/半径的500Hz轮速积分，比较里程计和真值误差。32/160/89/6及ONNX形状保持，停车基准、投影、死区/P限幅及清除由调用方维护。CPU sim2sim和H7尚未适配；H7能否复用LQR速度积分量待确认。

稳站v2 CPU后续发现初态17在原2ms积分下数值弹飞。当前仅该入口细分为0.5ms积分（2000Hz），PD/弹簧500Hz、策略/历史100Hz保持；训练及共享XML不改。128初态前5秒异常筛查通过，但CPU与训练引擎起立仍有差异，行为未通过，回放已停止。异常证据（对应旧训练线已清理）。下文500/100Hz CPU描述为原v1记录，当前时序以[本版接口](../interfaces/20261028-stand-v2-native-cpu.md)为准。

`chuanliantui_stand_v2`自2026-10-27改为32维当前观测＋160维历史编码器；旧39维独立CPU入口不能直接读取本版模型。2026-10-28新增`sim2sim/mj_sim2sim_chuanliantui_stand_v2.py`：原生CPU MuJoCo＋160维ONNX，已解锁0.22m阶段、名义零噪声/零时延，500/100Hz与0.2秒保持、承重NORMAL规则按训练版对齐；从已有名义initial.npz读取初态，浏览器单实例可切换。只有限接口检查和单例回放，未宣称完整行为或H7通过，见[CPU接入记录](../interfaces/20261028-stand-v2-native-cpu.md)与[32维接口](../interfaces/20261027-stand-v2-isaac32-history.md)。

2026-10-25：仅`chuanliantui_stand`动作缓冲改为每环境每回合腿／轮独立随机1或2个策略步，按整数decimation接入原物理队列，无固定秒数。物理／PD仍500Hz、策略仍100Hz，部分复位清空对应队列。multimode、独立CPU和H7保持原时延，未同步此稳站任务改动；模型及行为状态见[项目实验记录](project-log.md)，实现见稳站任务（对应旧训练线已清理）。

2026-10-23：mjlab训练／play／eval及独立CPU入口均试用10 ms／100 Hz策略，物理和PD保持2 ms／500 Hz。CPU趴地零力矩仍保持0.2秒（20个策略步），腾空确认仍20 ms，原2 ms动作等待不变，报告中的有效策略周期同步为10 ms；共享H7实测JSON保留原来源。后续USB H7源码随model13842导入同步10 ms／100 Hz，UART工程仍20 ms／50 Hz；烧录及板端待验收状态见[交付记录](../model-deliveries/20261023-model13842-h7-import.md)，不能把名义周期一致当作真实时序已测；旧CPU结果继续只描述当时的50 Hz。见[变频接口与验证](../interfaces/20261023-mjlab-policy100hz.md)。

2026-10-22：华南虎训练动作等待按作者后续要求改为每回合腿/轮独立随机6、8、10 ms；mjlab play及独立CPU回放保持H7实测缓存到首次PD约2.003 ms量化后的固定2 ms。网络调用已包含在此路径中，控制周期仍2 ms，微秒抖动不模拟；传感器数据龄随机化保持原配置。训练随机范围不代表真机实测，行为仍需评估，见[接入与验证](../interfaces/20261022-h7-simulation-timing.md)。

mjlab 串联腿的 multimode 与独立稳站任务共用39维actor／82维critic／六个实体电机，分别见 multimode训练（对应旧训练线已清理）与稳站任务（对应旧训练线已清理）。现有独立CPU入口只验收multimode配方，不能将稳站任务的mjlab成绩直接当作CPU或真机成绩。本页旧串联／H7策略通道保留历史物理顺序，只适用于 Isaac 旧任务。

华南虎 multimode 使用 39 维 actor／82 维 critic（`model_3023` 是历史基线）。其独立 CPU MuJoCo 入口为
`sim2sim/mj_sim2sim_huananhu_multimode.py`：按契约元数据接受任何 39 维 checkpoint（可用 `--expect-sha256` 钉哈希），
`--cmd-vx/--cmd-yaw/--cmd-height` 给定常值命令，报告新增 `tracking` 块；`--teleop`（需 `--render`，与 `--report` 互斥）把三个命令通道交给键盘：↑/↓ 前进速度 ±0.1 m/s、←/→ 偏航角速度 ±0.2 rad/s、PgUp/PgDn 或 `,`/`.` 高度 ±0.02 m、空格/回车 停车、Home 或 Backspace 复位（无 Home 的键盘可用 Backspace／Insert／Delete／End）、F1 暂停，限幅取训练包线（vx ±1.5 m/s、yaw ±1.5 rad/s、高度 0.2165～0.3664 m；对应近直立腿长约0.18～0.33m，见[零速保持训练准备](../interfaces/20261022-huananhu-zero-hold-training.md)）。机体系速度经
`mj_objectVelocity(mjOBJ_XBODY, flg_local=1)` 读取，该 API 返回**角速度在前、线速度在后**（`result[3:]` 才是线速度）。
观测腿槽与腿 PD 目标以训练动作项的站立标称角 `q_nominal`（`qpos0` 加微蹲目标解闭链）为中心、限幅以 XML `qpos0` 为中心，
轮速回路增益 0.1／0，接触参数镜像训练场景的机器人 geom `priority=1` 补丁（机器人-地面滑动摩擦取训练值 0.5，非等优先级 max 的 1.0；2026-10-17 两次修正，见 [multimode 接口记录](../interfaces/chuanliantui-huananhu-multimode-replay-v1.md)）。
导出与使用规则见 [multimode ONNX 使用说明](../onnx-huananhu-multimode-model5849.md)（当前交付模型 `model_5849`；`model_3023` 为历史基线）。它不兼容下文旧 25/125 维 H7 流程。

训练与CPU/real2sim已统一读取`resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml`；旧multimode_model3023/chuanliantui.xml仅作兼容链接。CPU姿态库仍为原multimode_model3023版本，不随模型源路径迁移。来源合并后，作者已明确要求将joint_sharp写入共享XML；当前双轮等效惯量/阻尼/摩擦及G/V/p已接入训练执行与CPU原生回调，参数仅存XML，见[轮端响应接入](../diagnostics/20261021-h7-wheel-response-integration.md)。

2026-10-18起，独立CPU入口的地面起立“恢复完成”高度条件使用与H7同含义的髋轮投影均值：
`h = (L_left*cos(theta_left) + L_right*cos(theta_right))/2`，要求 `h > 0.20 m` 连续满足0.5秒。
每侧取髋关节轴心到轮轴心在机体XZ腿平面内的向量，腿角以机体向下为零；分别投影后取均值，
不加轮半径，不使用世界坐标机身高度，也不在机身倾斜时改为世界竖直投影。
报告及窗口新增 `hip_wheel_height_m`；原 `height_m` 仍为真实机身离地高度，目标高度命令、39维观测、
跟踪／静站验收和mjlab训练不变。本次仅替换恢复的高度条件，原机身无接触与倾角小于40度条件仍保留，
没有移植H7的完整NORMAL／RECOVER／FAILED状态机或中途恢复重触发，不能称为完整H7仿真。

2026-10-22低腿长训练扩展后，CPU恢复门槛对低位命令改为
`min(0.20,cmd_height−轮半径+髋轴相对机身高度−0.01)`，原0.27～0.34m命令仍用0.20m投影门槛。
H7尚未同步低位命令和恢复门槛；几何、兼容性与验证范围见[低腿长接口记录](../interfaces/20261022-huananhu-low-leg-length.md)。

旧25/125维整机S2R1采集的策略复算与闭链MuJoCo离线复刻见
[历史S2R1／HPI1说明](s2r-offline.md)。它不适用于当前H7 RL39，当前固件没有匹配旧协议。只读镜像使用
`sim2sim/real2sim_huananhu.py`，新增196字节镜像帧与电机驱动闭链重构；软件检查通过、真实联机待确认，
见[部署接口](../deployment-contract.md#host-debug-plan)。它不运行物理步进，也不代替sim2sim或真机控制验收。

关节 USB 台架采集后的固定机身闭链对照使用 `sim2sim/fit_joint_usb.py`，数据协议及安全门见 H7 仓库 `md/joint-usb-sysid.md`。它只拟合实测力矩输入下的模型响应，不修改训练策略、部署物理极性或现有 XML。

采集到独立的拟合 run 和留出 run、并由台架作者确认四实体电机的关节对应、符号与零点后，在训练仓用已配置的 `WHEELLEGGED_PYTHON` 执行：

```bash
"$WHEELLEGGED_PYTHON" sim2sim/fit_joint_usb.py \
  --run <拟合采集目录> --holdout <留出采集目录> \
  --mapping <已确认映射.json> --out <新结果目录>
```

脚本拒绝未批准映射、相同的拟合/留出 CSV、故障、坏帧或丢帧超过 1% 的 run；输出参数、四实体关节及左右腿几何的实测/仿真对照。`delay_s` 从主控 CAN 排队时刻估计，不能解释为电机内部生效延迟；几何角度约定须先做静态核对，参数 Jacobian 秩不足时不能断言所有参数都已辨识。

改动观测构造、控制时序、URDF/XML、或准备真机部署时读这篇。imcawl 的权威来源是 `sim2sim/mj_sim2sim.py` 的文件头注释；chuanliantui 见本文后面的对应小节及 `sim2sim/mj_sim2sim_ct.py`、训练实现。跨仓库接口版本与模型交付要求见 [部署接口约定](../deployment-contract.md)。

## imcawl 部署契约

imcawl 的部署端（MuJoCo 脚本、真机驱动）必须逐条对齐训练实现；本表不适用于 chuanliantui：

| 项 | 值 |
|---|---|
| 策略 | `ActorCriticSequence`，`action = actor(cat(obs_27, encoder(history_135)))` |
| 权重 | 用 `model_*.pt`；`policy_1.pt` 缺 encoder 不可用 |
| DOF 顺序 | `[lf0, lf1, l_wheel, rf0, rf1, r_wheel]` |
| 观测 27 维 | `base_ang_vel*0.25(3)`, `projected_gravity(3)`, `cmd*[2.0,0.25,5.0](3)`, `(dof_pos-default)*1.0(6)`, `dof_vel*0.05(6)`, `last_action(6)`，裁剪 ±100 |
| 历史 135 | 27×5 FIFO，最旧在前、最新在末；上电用首帧重复 5 次填充 |
| `dof_vel` | 位置差分 `wrap_to_pi(Δdof_pos)/sim_dt`，每个 sim 子步更新一次（**不是**读速度传感器） |
| 动作→力矩 | 腿位置控制 Kp=60/Kd=2；轮速度控制 Kp=0/Kd=0.5 |
| 时序 | `sim_dt=0.005`，`decimation=2` → 策略 100 Hz，PD 内环 200 Hz |
| 力矩上限 | `[30, 30, 5, 30, 30, 5]` N·m |
| 链路延迟 | 训练随机化 0~10 ms。真机通信+执行延迟必须 ≤10 ms，超出即出分布 |
| 前进方向 | 机体 **+y**。`cmd_vx` 是「前向速度命令」（通道 0），对应机体系 vy |
| yaw 通道 | **是外环反馈，不是常数**：`cmd[1] = 1.5*(目标航向 - 当前航向)` 剪 ±5、每步刷新 |

最后一条最容易漏：训练时 `heading_command=True`。部署端喂恒定 yaw 值会导致偏航漂移无人纠正，巡航数秒后摔车。真机必须实现同样的航向保持外环。

## MuJoCo 侧三个已修的坑

`sim2sim/mj_sim2sim.py` 里已全部修复，**勿回退**：

1. **`mj_objectVelocity` 必须用 `mjOBJ_XBODY`**（机体系）。`mjOBJ_BODY` 返回惯性主轴系，base 惯量特征值降序导致轴置换，陀螺仪三分量整个换位。症状：站立勉强、一加速就翻。
2. **`mj_step` 后派生量（`cvel` 等）滞后一个子步**，读状态前必须 `mj_forward` 刷新。
3. **PD 内环顺序必须是「算力矩 → 步进 → 差分更新 dof_vel」**，对齐 `legged_robot.py`。顺序错了 obs 里的 `dof_vel` 滞后 5 ms。

真机同构风险：IMU 角速度的坐标系定义与采样时序必须与策略推理同拍。5 ms 级延迟对静态站立无感、对加速瞬态致命。验证方法是双源交叉（位置差分 vs 速度读数）。

## 摩擦

MuJoCo 摩擦 = 两 geom 逐元素 **max**；PhysX 取**平均**。训练等效摩擦区间约 [0.4, 1.0]，等效均值约 0.75。MJCF 里 0.5 踩在下沿，`mj_sim2sim.py --friction` 可覆盖。

XML 已用 `cone=elliptic` + `impratio=10`（轮式标准配置）。

## Isaac 对照评估

`eval_isaac.py` 必须设 `train_cfg.runner.resume = True`（对齐 `play.py`），否则 `make_alg_runner` **静默**跑随机初始化网络。症状：换 checkpoint 输出逐字节不变。自检习惯：换权重必须换行为。

## 当前性能上限：encoder 速度估计正偏置

截至 2026-07-21，基准权重 `logs/mini_wheel_legged/Jul21_14-09-24_/model_4000.pt` 在两引擎站立/加速/巡航零摔倒，sim2sim 通过。

余下三个现象同一个根源——latent 前 3 维（`base_lin_vel*2` 的估计，见 `rsl_rl/algorithms/ppo.py`）**系统性估高 0.15~0.25**：

- 稳态欠速：真实速度恒低于命令约 0.2 m/s
- MuJoCo 低速后拉：偏置 +0.1 时，调 v̂ 到 0 对应真实 -0.1
- 高速 ≥1.8 m/s 刹车瞬态 encoder 跟丢 → 发散翻车

结构性原因是 `obs_history_length = 5`，encoder 只有 50 ms 窗口。

**已接受现状，安全包线 ±1.5 m/s**（遥操作已限幅）。若将来要更高速：`obs_history_length` 5→20 全重训，或加大 encoder 损失权重续训作廉价改良。

诊断手段：`mj_sim2sim.py` 与 `eval_isaac.py` 日志里的 `v̂` 列（策略内部速度估计）对比真实 `v_fwd`。

## URDF / XML 同步

`resources/robots/imcawl/urdf/imcawl.urdf`（Isaac 侧）与 `sim2sim/imcawl.xml`（MuJoCo 侧）是**两份手工维护的模型**。改一边必须改另一边，然后跑 `python sim2sim/check_model.py`。连杆长度变了还要同步 `mini_wheel_legged_config.py` 的 `asset.l1 / l2`。

## chuanliantui 串联训练代理回放

`sim2sim/mj_sim2sim_ct.py` 默认加载 `chuanliantui_train_proxy.xml`。它由
`resources/robots/chuanliantui_new_1/urdf/chuanliantui_train.urdf` 生成，保留训练端
6-DOF 串联拓扑、固定后支链、地面、根 free joint 及六个同名力矩电机；模型必须保持
`neq=0`。改训练 URDF 后运行
`scripts/agent/isaac/generate_chuanliantui_train_proxy_mjcf.py`，再用
`scripts/agent/isaac/check_chuanliantui_train_proxy.py` 检查该契约。

chuanliantui actor 观测为 25 维：机体系角速度(3)、重力投影(3)、命令(3)、四个腿关节
`[lf0, lf1, rf0, rf1]` 的位置(4)、六个关节速度(6)、上次动作(6)。连续轮的绝对位置
不进入 actor 或 encoder；历史为 `25×5=125`。这改变了网络的 encoder、actor 和 critic
输入形状，所有旧 27 维 chuanliantui checkpoint 均不能加载，必须重新训练。
提交观测变更前可运行 `python -m scripts.agent.isaac.check_chuanliantui_observation`，核对训练端布局、
噪声、critic 维度、历史 FIFO 和 MuJoCo 构造是否一致。

训练与 MuJoCo 回放均用 500 Hz 的物理/PD 内环（`sim_dt=0.002`），每 5 个内环步推理一次，
因此策略、观测和历史 FIFO 保持 100 Hz（`0.002×5=0.01 s`）。策略输出在中间 4 个内环步零阶保持；
延迟后的 PD 目标按下文动作 FIFO 时序生效，可在策略周期中途切换。
气弹簧力在每个 2 ms 内环步重写。

`chuanliantui` 和继承它的 `chuanliantui_standup` 开启动作目标延迟随机化，
范围为 0–10 ms。每个环境在初始化时连续均匀采样、四舍五入到物理子步，
得到 0/2/4/6/8/10 ms；一个环境的延迟随后固定，并非逐拍抖动或等概率离散采样。
FIFO 每个物理子步入队一次，容量包含当前动作和最大延迟对应的历史动作；
回合重置会清空该环境的动作及 FIFO，防止上一回合的目标继续生效。
延迟发生在动作目标进入 PD 之前，PD 使用当前状态；它不单独模拟观测延迟或
电机力矩响应。已有 checkpoint 需续训或重训才能学习该随机化。
`play.py` 和默认 `eval_isaac.py` 会关闭动作延迟，属于无延迟对照；
`eval_isaac.py --domain_rand` 保留配置中的域随机化，但同时包含其他随机化项。
单独固定动作延迟使用 `--action_delay_ms <毫秒>`，实际值按物理子步量化并打印；不传 `--domain_rand` 时其他随机化仍关闭。
对比起立模型可用 `--standup_post_unlock` 在加载后统一解锁阶段，避免 checkpoint 的课程状态造成成功判据不同；`--metrics_report <新 JSON 路径>` 记录逐控制步轨迹和 5 秒之后的指标，重置步不参与连续指标，失败次数单列。
MuJoCo 零速诊断可在有限时长、无渲染回放上追加 `--diagnostic_report <新 JSON 路径>`，记录同一推理时刻的状态、原始 encoder 估计、动作、关节和接触数据。
`--diagnostic_true_vx --diagnostic_true_vx_after 5` 仅从第 5 秒起把送入 actor 的前向速度 latent 替换为仿真真值，按训练相同的 10 ms 根位置差分、当前姿态转机体系计算；其余输入不变，报告仍保留原估计。这是消融诊断，不是部署输入或通过验收的策略。
排查时同时核对实际关节角与限位：MuJoCo 软约束允许少量越界，不能仅凭两边 `range` 相同就认定限位执行一致；关节限位刚度、轮地接触刚度和控制 PD 的 Kd 是不同参数。
串联训练代理的 `lf1/rf1` 单独设置 `solreflimit="0.004 1"`，生成器和代理检查同步该值。
此调整只改变两膝限位约束的响应，保留 `lf1=[-0.12,0.77]`、`rf1=[-0.77,0.12]` rad
的物理范围，不改变全局接触参数、关节耗散或控制 PD；不能把该设置等同于实体闭链或真机验证。
队列上界、实际子步延后时长和部分回合重置可用
`python -m scripts.agent.isaac.check_action_delay` 做 CPU 检查；这不替代正式行为验收。

### chuanliantui 气弹簧支撑

训练环境与串联 MuJoCo 代理现默认加入每侧 150 N 恒定伸张气弹簧。
两端安装点与 CAD 闭链一致，训练按物理膝角求方向和力臂，每 2 ms 重算；
对上下连杆施加等大反向膝轴扭矩，与串联理想铰链上的端点力具有相同广义力。
它与水平推力合并后每子步只提交一次，不经过电机限矩、倍率或动作延迟，
`self.torques` 和电机能耗奖励仍仅含主动电机力矩。起立接管前也生效。
MuJoCo 代理通过独立的两个 tendon motor 表示弹簧，主动电机仍为六路。

`eval_isaac.py`、`mj_sim2sim_ct.py` 的 `--gas_spring_force 0` 可恢复无弹簧对照；
训练配置为 `ChuanliantuiCfg.gas_spring.force_n`。旧配置快照会继承当前父类，
仅加载旧快照不会自动关闭弹簧。旧 checkpoint 形状兼容但行为需重新评估/续训。
数值检查：`python -m scripts.agent.isaac.check_chuanliantui_gas_spring`；
正式起立、站立回放仍按 [命令速查](../../COMMANDS.md) 执行。
本轮没有把约 0.03 rad 的闭链虚拟/物理膝差写成补偿，也没有改变关节反馈。
150 N 是现有仿真假设，真机弹簧推力待测量核实。

气弹簧数学模块现从 URDF 的 `lf001/lf002`、`rf001/rf002` 定位 link 零位坐标
推导上、下安装点，并从实际 joint 读取轴向和安装 RPY，不再使用固定坐标表或
根据名字推断轴向。定位件只定义 CAD 零位安装位置；仿真中两个端点分别随上、下
连杆运动。物理力矩为 `tau = F * d(length)/dq`，同时可读取长度与带符号力臂。
等效刚体力矩的世界方向包含根姿态、髋转角和关节安装旋转；共轴情形缓存固定轴。
零轴、缺失定位件、端点可重合等无效几何显式报错。
闭链与串联代理 XML 的四个气弹簧 site 已按源 URDF 生成器同步。
本次优化仍使用每侧 150 N 恒力，未引入未标定的力—长度曲线、阻尼或电机前馈。
数学模块可通过 `spring_joints` 显式指定髋/膝 joint 和定位 link，支持左右命名迁移；
当前资产名称及旧策略接口尚未在本次气弹簧改动中迁移。

该模式用于对齐 Isaac 串联训练资产与 MuJoCo 回放资产，**不是**真实闭链或
真机 sim2sim。默认会拒绝任何含 equality/connect 约束的模型，防止两条链路混用。

仅在闭链差异诊断时，可显式传入 `--closed_chain`。该选项加载本机
`chuanliantui.xml`（`neq=4`），实体前、后输入轴分别是
`lf0/lf00` 与 `rf0/rf00`。模型用两个 spatial tendon 模拟左右气弹簧；端点由本机
CAD 定位件换算到前支路，默认每侧恒定伸张推力 150 N，可用 `--gas_spring_force 0`
关闭做对照。复旦 XML 仅作为单独的模型参考，不参与本机策略回放。该恒力模型不含
真实气弹簧的力—长度曲线、阻尼或行程，也不能作为真机验证通过的证据。

闭链回放默认使用 `H7ClosedChainAdapter`，以用户指定的 H7 仓库源码复现五连杆解算、
反馈速度、环绕角差 PD、轮目标 ±20 rad/s、虚功映射和十拍预热。
CAD 机构、物理限位与气弹簧保留；状态估算不读取被动膝状态。后轴到 H7 坐标的默认值
来自 CAD 输入杆方向，只是名义几何注册，需共同姿态数据才能验证实机编码器对应。
H7 软件气弹簧补偿当前关闭，模型物理气弹簧仍保留。详见
[H7 闭链回放接口](../interfaces/chuanliantui-h7-closed-replay.md)。

加 `--closed_chain_controller cad` 保留原 `ClosedChainAdapter` 对照：它从本机 MJCF 的前/后
支路尺寸、零位装配分支和实体电机 `lf0/lf00`、`rf0/rf00` 的角度解两次平面圆交点，
得到训练策略所需的虚拟膝角；闭链几何方程的解析 Jacobian 把电机速度变为虚拟膝速度，
并按虚功关系把虚拟膝力矩映射到两个实体电机。轮力矩仍直通。这沿用复旦部署的
“电机状态→虚拟膝状态、虚拟力矩→电机力矩”路径，但使用 chuanliantui 自己的
CAD 几何，不复制复旦的连杆长度或偏置。几何不可达或 Jacobian 奇异会显式报错。
该 CAD 对照的虚拟膝速度来自电机速度和几何 Jacobian，其余 DOF 仍按 2 ms 子步差分。
H7 路径则全部从主动轴反馈速度构造，MuJoCo 用 `qvel` 表示反馈，不模拟 CAN 量化与异步到达。

用 `python sim2sim/mj_sim2sim_ct.py --gas_spring_view` 打开气弹簧专用视图（自动开启
闭链和渲染）：半透明机构中，左侧青色、右侧橙色，圆点为安装端点，粗线为气弹簧轴线。
显示加粗仅影响渲染，不改变气弹簧力或碰撞几何；每侧默认仍为 150 N。

如需固定机身、手动拖动腿部观察气弹簧，运行
`python sim2sim/view_gas_spring_ct.py`。该独立演示在内存中移除基座自由关节，
将机身固定于 0.8 m，不加载策略，六个实体电机输出始终为零，只保留每侧 150 N
气弹簧和重力。双击腿部后 Ctrl+右键拖动施力；空格暂停/继续，Home 复位，F5
开关气弹簧。`--headless --duration 10` 可运行无窗口检查；原始 XML 不被修改。
机身上方实时显示左右气弹簧对 `lf1/rf1` 的直接关节力矩（N·m）、轴向力（N）和
有效力臂（mm）。力矩按 `actuator_force * actuator_moment` 计算，符号遵循各自
关节轴，不包含重力、鼠标外力和闭链约束反力，也不是主动电机轴的等效补偿力矩。

复旦原腿的固定机身对照：`python sim2sim/view_gas_spring_fudan.py`，使用 `--xml`
指定本机参考仓库中的原 XML，不依赖脚本内某位成员的默认路径，
每侧使用 150 N 便于同力对照；`--gas_spring_force 300` 恢复原脚本的力值。
该视图无策略、无实体电机驱动，保留原模型的限位、阻尼和闭链锚点；在内存中把
`connect site1/site2` 转为 MuJoCo 3.2.2 支持的两刚体局部锚点，不修改源 XML。
复旦模型的 `l20/r20` 不限位；当前闭链 XML 的 `lf00/rf00` 也已改为
`limited="false"`，允许后输入轴连续旋转，避免原 ±3.14 rad 限位阻挡髋部转过一圈。
这是 MuJoCo 模型对原始 CAD URDF 限位的显式修正；闭链生成器已同步该规则和气弹簧定义，
重新生成后运行 `python -m scripts.agent.isaac.check_chuanliantui_closed`，检查左右膝限位、
端点连接、每侧 0/150 N 施力、±360° 闭合姿态以及原有的 5 秒无控制仿真。
用户确认实机膝关节行程以 `chuanliantui_train.urdf` 为准：
`lf1=[-0.12,0.77]`、`rf1=[-0.77,0.12]` rad。原始 `chuanliantui.urdf`
的这两处限位曾与实机/训练资产相反，现已修正。闭链 XML 中的膝限位已同步；
生成器也直接读取修正后的源 URDF。其余被动关节仍保留原始 CAD 范围。
原始四对 site 存在约 10 mm 的横向错位，加载后的演示保留此偏差，不能视为闭合精度验证通过。

## chuanliantui 站高精度与膝余量续训

2026-10-03 的站立续训配置采用解锁后目标 **0.22 m**、成功高度门槛 **0.20 m**，
连续满足高度、`pg_z<=-0.90` 和机身离地条件 0.5 s 后记为已起立。解锁前仍使用
0.30 m 目标、0.28 m 成功门槛。初始位置与后摆关节姿态不变。
当前模型在近零 pitch、两膝距硬端点至少 0.05 rad 的几何条件下，最低站高约为
0.218 m；因此 0.20 m 站高与这两项约束不能同时满足，当前选用 0.22 m 目标。
该几何估算不等于策略已经能达到目标。

训练仍按 `dt=0.01 s` 缩放奖励，并将每项裁剪到 `[-dt,dt]`。站立任务的调整为：

| 项目 | 当前计算与作用 |
|---|---|
| 高度项 | 权重从 2 改为 1，原始值为 `exp(-(h-h_cmd)^2/sigma) * gate`；`gate` 随机身接触力由 1 降至 0.2，因此目标附近不再被单项裁剪成相同得分。 |
| 高度容差 | 解锁前 `sigma=0.01 m²`，解锁后 `sigma=0.0025 m²`；解锁后误差 5 cm 时原始指数项为 `exp(-1)`。解锁前保留宽容差，但高度权重同样使用 1。 |
| 膝端点余量 | 从 URDF 原始 DOF 属性保存两膝硬限位，在两端各设 `m=0.05 rad` 奖励区；按 `d=max(L+m-q,q-U+m,0)/m`，取两膝 `d²` 的均值，权重 `-0.3`。 |
| 起立门控 | 新膝余量项仅在 `has_stood=True` 后激活；本回合曾连续站稳 0.5 s 后保持激活，到 reset 清除。旧 `dof_pos_limits` 项继续保留。 |

两膝恰在硬端点时，新余量项每策略步为 `-0.003`；端点外的小超限继续增罚。
它是奖励安全区，不修改物理 `range` 或裁剪 PD 位置目标，也不能保证策略一定留足余量。
单项裁剪仍存在：两膝同时越过硬端点约 0.0413 rad 后，该新项会达到裁剪下限。
姿态奖励仍乘高度进度门控；例如目标 0.22 m、实际高度 0.208 m 时门控约为 0.829，
因此应同时检查实际高度、pitch 和两膝到硬端点的距离，不能只看恢复率。

CPU 回归检查调用真实的奖励准备和计算路径，覆盖实际裁剪后的高度区分、接触门控、
硬限位保存、膝余量的对称性与起立门控：

```bash
python -m scripts.agent.isaac.check_standup_precision_rewards
```

该检查不创建仿真或训练产物，不能代替正式训练、Isaac 回放和 MuJoCo 行为验收。
历史续训命令见 [Isaac历史命令](../commands/isaac.md)：
从 `Oct03_03-28-58_standup_action_delay_0_10ms/model_9000.pt` 额外训练 3000 轮，
当前叠加下述推扰，使用新的 `standup_h022_knee005_push` run 名及日志目录；保留旧模型做同条件对照。
恢复该已解锁 checkpoint 后，课程命令按当前配置切换为 0.22 m，成功高度为 0.20 m。

此次没有改变观测、动作、PD 或模型输入输出形状，旧 25 维 checkpoint 仍可加载，
但旧模型没有因配置更新而自动学会新高度与余量要求；当前推荐回放显式使用
`--cmd_vx 0 --cmd_height 0.22`，历史 0.20 m 对照应另行标注。
本次未自动启动训练，尚不能据此宣称新目标训练成功或真机问题已修复；H7 代码未修改。

### 接地后分档随机推力

`chuanliantui_standup` 开启专用水平推扰，其他机器人继续使用各自配置与基类实现。
首次任一轮接地后，独立的每环境计时开始；初始自由下落阶段不推。每次脉冲开始时：

- 当前满足 `has_stood` 且连续站稳时间达到 0.5 s：合力模长均匀采样 5–15 N。
- 尚未站稳或当前失稳：合力模长均匀采样 1–5 N。
- 环境坐标系水平面方向均匀采样，竖直力与额外外力矩为零；力施加于 `base_link` 质心。
- 恒力持续 0.10 s，脉冲内不随姿态或稳站状态重采样。每次结束后留出 3–5 s 随机空档。
- 调度按 10 ms 策略步推进，力在每个 2 ms 物理子步重提交；不直接设置根速度。
  单次冲量模长分别为 0.1–0.5 / 0.5–1.5 N·s。回合重置仅清除对应环境状态。

参数均在起立配置的 `domain_rand.standup_push_*` 中。这些是首轮训练强度，未由真机
手推测量标定。站高、膝余量、动作延迟及观测/动作接口保持当前设置；本轮没有把
H7 的轮速目标 ±20 rad/s 裁剪加入训练，推扰不能替代这项控制一致性工作。

`python -m scripts.agent.isaac.check_standup_pushes` 检查分档、时序、实际提交张量和重置。
正式行为检查使用 `eval_isaac.py --pushes`，可叠加 `--action_delay_ms 6` 单独比较；
`--domain_rand` 则保留全部训练随机化。默认回放关闭推扰。`--metrics_report` 保存
实际推力和每回合触发次数，非零施力时长按 100 Hz 采样估算并排除重置步。

`standup_push_count` 记录每环境本回合的实际触发次数，回合重置清空，
通过评估报告逐步检查覆盖和推后状态。训练 `recovered_rate` 仍只表示曾起立，
不把它或重置批次平均值当作逐次抗扰成功率。
