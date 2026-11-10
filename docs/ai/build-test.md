# 构建与验证

非线性气弹簧诊断：`replay_length_response_gpu`的候选JSON可设`spring_curve_compression_m/force_n`和`remove_baseline_right_load`，或显式`curve_with_joint_correction`只换弹性；仅私有模型放宽执行器范围，不改正式参数。`report_length_response_fit`另复算曲线实际施力、去除的关节负载及首帧初载。此次软件通过但响应未改善，不作采纳。[有限搜索、图和限制](../../artifacts/length-response-fit/20261110-nonlinear-axial-result-01/README.md)。

复旦拟合负载GPU运行检查：`scripts.agent.fudan_ppo_walk.check_fitted_load_runtime --output <新目录>`关闭随机化核对名义值，再检查真实训练配置；逐2ms积分拍核对完整JSON保守负载、`qfrc_applied`记忆力、单次施力、原气弹簧、全体/局部复位和其他环境状态。共39项，零策略更新，不代表运动验收。

拟合物理轮数预算审计：`scripts.agent.fudan_ppo_walk.audit_fitted_run --run <新训练日志目录> --experiment <含budget.json的本次目录> --output <新JSON>`支持从零和续训，核对初始状态/来源、完整轮数/转移/两Adam计数、训练快照与检查点物理身份、源码未变和16项TensorBoard奖励日志；续训另运行`audit_resume_run --experiment <本次目录> --output <新JSON>`逐值核对源模型/两优化器/RNG及恢复计数。不推进物理或执行网络前向。[从零2000轮记录](../../artifacts/fudan_ppo_walk/20261110-fitted-scratch-2000-01/README.md)、[9000→11000续训记录](../../artifacts/fudan_ppo_walk/20261110-fitted-resume9000-to11000-01/README.md)。

GPU视频校验：`scripts.agent.fudan_ppo_walk.check_gpu_videos --index <video-gpu/index.json> --recording <GPU目录> --checkpoint <冻结PT> --acceptance <acceptance.json> --output <新JSON>`核对完整名义矩阵覆盖、逐章轨迹/模型/视频SHA、解码帧数与50fps正常速度；有随机失败则须保留对应视频。只读已有轨迹/解码视频，不新增物理或策略更新，不替代用户观看验收。

双腿正式默认检查：`scripts.agent.real2sim.check_formal_leg_load --output <新JSON>`对照旧实际施力、NumPy/Torch CPU/CUDA、原生作用与复位及20环境训练配置GPU短步；新随机化须围绕拟合基线。局部复位检查及默认/legacy两组20例GPU已保存，默认0/20、legacy20/20，仅采用不代表行为通过。[本轮检查/来源](../../artifacts/length-response-fit/20261109-formal-integration-01/README.md)。

8000局部负载接入：`scripts.agent.real2sim.check_length_fit_policy --recording <含baseline与candidate-both的根目录>`离线核同初态、关节/滑杆索引、左侧镜像、实际摩擦/阻尼数组、未改限矩/惯量和逐策略拍状态。运行自身`response-verification.json`另核所有物理步的预滑移状态和施力；本次通过但20例统一技术19→0，不能把接入正确当行为通过。[命令和来源](../../artifacts/length-response-fit/20261109-fudan8000-right-load-01/README.md)。

只读录制工程选择：`real2sim_huananhu --port <设备> --h7-project H7_ros --record <新目录>`读取ROS工程配置；`H7`对应主工程，省略时仍默认H7。`check_real2sim_huananhu.check_sessions`增加两工程CLI选路、元数据路径/SHA与冲突/历史回放参数检查；`check_viewer_timing`核最终报告工程/配置身份，使用真实C打包的合成流和PTY验证，无硬件访问。[本次定向检查](../../artifacts/length-response-fit/20261109-recorder-project-check/focused-check-04.json)通过。时序检查改为完整预算时钟≥0.15秒，并允许首个串口read的一个20ms周期，不改变运行时采样；整套检查另在原`check_v2_sessions`的v2版本断言失败（当前实际打包为v3），不能称整套通过。

右腿长度响应复现：`prepare_length_response_fit`封存原始配对并编译归档控制器重建；`replay_length_response_gpu --inputs <封存目录> --parameters <候选JSON> --seconds <时长> --save-traces --output <新目录>`运行同请求矩/同目标两分支GPU自由积分；`report_length_response_fit --inputs <封存目录> --run <结果>`复核归档C、MIT编码、施力、闭链和实际模型坐标，显式分开软件通过与响应对齐。摩擦记忆候选另保存每0.2ms的状态/施力审计，不能用逆力拟合或到位终点替代两分支动态检查。[当前证据与边界](../../artifacts/length-response-fit/README.md)。`audit_length_held_commands --session <原始会话> --paired <配对目录> --output <新目录>`核全部1kHz分发是否保持上次500Hz请求；`check_seal_coordinate_gpu --inputs <封存目录> --model <带right_seal_span的诊断XML> --output <新目录>`以相同轴向静载核滑杆/安装点两种摩擦表示的漂移和虚功，属于GPU合成检查，不是真机标定验收。

只改腿长预检：`precheck_right_leg_pose --profile length-only --observation <姿态JSON> --prepared-length-mm 255 --prepared-hold --output <新目录>`使用实际C、锁定源姿态腿角、三负载GPU；`report_right_leg_pose --run <目录>`新增目标角恒定及实际全窗/移动段偏移复算。`check_joint_capture_wire`识别bit4新profile；`report_right_pose_tracking`/只读页面/监看按profile选择固定入场角或旧30°，旧数据仍按其原定义，不混算。仅模型预检不是真机验收。[本次证据](../../artifacts/pace-joint-capture/20261109-right-length-release-01/README.md)。

单右腿慢速定位：`prepare_single_leg_pose_load --output <新目录>`从冻结模型导出目标姿态负载表（0积分）；`precheck_right_leg_pose --observation <只读状态JSON> --prepared-length-mm 255 --prepared-hold --output <新目录>`将实际C放入三假设负载Warp GPU，必须从准备档保持矩接管；不加`--prepared-hold`是零矩启动压力场景，不能混称实际慢速准备。`report_right_leg_pose --run <目录>`从保存状态逐拍重算C请求/目标、守卫及独立几何，0新增积分；`check_joint_capture_wire`检查当前bit3身份，旧small-high导出拒绝该profile。宿主C检查需配置有效CC，均在训练仓执行，不在H7目录新增Python工具。[候选、失败边界和未烧录状态](../../artifacts/pace-joint-capture/20261109-right-pose-release-03/README.md)。

固定机身腿长/腿角控制：`scripts.agent.real2sim.control_leg_pose_gpu --length-mm 190 --angle-deg 30 --parameters <冻结局部参数JSON> --reference <原GPU模型目录> --output <新目录>`按H7几何逆解、关节PID+目标姿态静态补偿，Warp GPU自由积分；`check_leg_pose_control --run <目录>`核源/参数身份、PID积分/负载、实际轮轴坐标几何及末段误差、膝限位；`view_leg_pose_control --run <目录> --port 8779`只读三维回放。0策略训练/硬件访问；同模型控制到位不能替代真机拟合验证，不是行走平衡验收。[190mm/30°结果](../../artifacts/pace-joint-capture/20261109-leg-pose-control-190mm-30deg-02/README.md)。

新small-high数据导出/验证：`prepare_pace_capture --session <新会话> --output <新配对>`只收profile bit2的stage2运行样本，stage3只作完成证据，核新鲜/有效/零轮矩/PD/序号/CRC/队列及CAN提交；`check_small_high_capture --output <新JSON>`跑合成完整120秒与旧profile/缺配对拒绝。`replay_pace_capture_gpu --segment small-high --parameters <冻结JSON>`独立四段2ms PD/0.2ms GPU回放；`report_pace_capture_replay`核原始输入/初态/模拟施力，`report_small_high_validation --run <结果> --paired <配对> --frozen-fit <原拟合目录> --output <新目录>`核参数身份、姿态差异、原始角RMSE及频带振幅/相位。不能把数值检查通过当动态对齐，也不能将右腿姿态改变混作纯频率检验。[本次数据与边界](../../artifacts/pace-joint-capture/20261109-small-high-validation-01/README.md)。

PACE气弹簧重拟合：`refit_pace_gas_spring --session <会话> --paired <输入> --static-summary <静置summary.json> --output <新目录>`逐帧姿态/重力/惯性分离，只拟合气弹簧曲线与非负阻力，静置摩擦用区间；`replay_pace_capture_gpu --parameters <parameters.json>`支持gas_spring_linear_with_joint_resistance，诊断副本显式放宽弹簧允许值并回放已录重力方向。`report_pace_capture_replay`核力公式；`report_gas_gravity_refit --fit <目录> --baseline <名义回放> --local-fit <局部补偿回放> --output <新检查目录>`核重力向量/原生允许值/无额外关节补偿力与同输入指标。不能把旧重力已计入的结果写成“补了遗漏重力”，也不能把联合估计当独立厂家曲线标定。[本次结论](../../artifacts/pace-joint-capture/20261109-gas-gravity-refit-02/README.md)。

PACE低频拟合：`fit_pace_precheck_load --reference <名义回放> --paired <配对输入> --output <新目录>`只用5.5～14.5秒拟合局部负载和阻力，15.5～24.5秒留出。`replay_pace_capture_gpu`增加`--parameters <parameters.json>`，只改诊断副本的对称位置负载、原生非负摩擦/阻尼。`report_pace_capture_replay`另核实际附加载荷；`report_pace_lowfreq_fit --baseline <名义> --fitted <候选回放> --fit <拟合目录> --output <新目录>`核同输入、原生阻力/参数公式/留出掩码并输出未减偏差的留出角误差。逆负载拟合不能替代自由GPU回放；局部低频通过不能代替高频/新姿态。[本轮候选及结果](../../artifacts/pace-joint-capture/20261108-lowfreq-load-fit-01/README.md)。

PACE实录预检回放：`replay_pace_capture_gpu --session <已封存会话> --paired <first-sweep-audit目录> --output <新目录>`逐条核原始前缀与配对输入，四个独立30秒预检段用原目标/实测初态、同PD作Warp GPU回放；`report_pace_capture_replay --run <结果> --paired <配对目录> --output <新检查目录>`独立核目标/实测/初态/模拟PD及闭链，0新增物理步。当前入口只支持四个完整预检段，不支持未完成的正式/验证组，不是参数拟合。[本次实录结果](../../artifacts/pace-joint-capture/20261108-real-precheck-gpu-02/README.md)。

PACE采集准备：`precheck_pace_capture --output artifacts/pace-joint-capture/<新目录>`作三起点/三负载/四轴三阶段GPU筛查，`--continuous`为265mm三负载360秒不中断序列；`report_pace_precheck --run <结果> --output <新核验>`核目标/PD与边界报告，0新积分。实际C入口`check_joint_chirp`、`check_joint_capture`、`check_joint_capture_wire`分别核目标全过程、PD/接管/75几何姿态/实际任务包装、目标状态打包及缺帧拒绝；均用`--output <新JSON>`。同时跑旧`check_h7_leg_test_mode`、`check_h7_leg_capture`、`check_h7_validation_capture`和指定H7_RL_ros的`check_real2sim_huananhu --build-only`。有宿主C编译器才可声称C检查通过；无CC环境不能只用NumPy代替。实际串口吞吐/时序及动力须另验。[本轮状态](../../artifacts/pace-joint-capture/README.md)。

旧关节动态数据审计：`scripts.agent.real2sim.audit_joint_recording_data --output <新目录>`核原会话/CSV身份，分别检查1kHz控制与最多500Hz位置记录、所用窗事件序号/门控/反馈龄、拟合归档力矩和实测状态逐点一致，并统计力矩输入秩、速度导数及请求/反馈差。完整文件有缺口不自动等同选段缺口，反之也不把选段通过写成全会话完整；独立初速量化观察/尚未执行的零初速对照见[审计报告](../../artifacts/real2sim/h7-original-dynamic-data-audit-20261105-04/README.md)。0新物理步，不能据输入覆盖秩1断言已有轨迹必然拟合不了。

放开气弹簧规格的动态拟合：`scripts.agent.real2sim.fit_free_spring_joint_response --output <新目录>`只按旧original/wide同请求矩四角位置误差选完整候选，保留旧96池最佳原值，12×2配置允许负斜率和广范围推力/阻尼，slow/fast固定回归；数值600N夹限非厂家上限。核验继续用`report_static_constrained_joint_refit --experiment <完整自由参数结果> --output <新目录>`，不要求静态包络存在。更自由的`fit_empirical_joint_load --output <新目录>`用q/v/位移方向记忆的64状态基/腿、8配置逆负载种子及GPU自由回放；`audit_empirical_joint_load --experiment <该结果> --output <新JSON>`逐0.2ms复核经验施力、原请求矩和源身份。经验载荷允许供能，不据逆负载拟合或低反馈角误差宣称响应通过；禁止用于训练或真机而不经另行采纳/验证。[本轮有限搜索及负结果](../../artifacts/real2sim/h7-free-spring-joint-response-20261105-01/README.md)。

静态约束旧关节动态拟合：`scripts.agent.real2sim.refit_joint_from_static_sweep --static-fit <分方向静态report.json> --output <新目录> --population 16 --generations 2`冻结新静态两方向中线/半差，复用旧original/wide选优和slow/fast回归；只迁移已采用的闭链/限位响应，旧数据/报告不改。`probe_static_load_allocation --previous <该完整结果目录> --output <新目录>`对静态允许的压力偏置与滑杆/关节阻力分摊作18种有限假设。`report_static_constrained_joint_refit --experiment <完整结果> --output <新目录>`复核每个0.2ms提交力/阻力容量和输入矩、1ms逐轴指标及销点闭合，单候选/批量组合差单列；不得用静态0.51Nm误差冒充动态角度通过，不把参数更新称策略训练。[首阶段负结果与范围](../../artifacts/real2sim/h7-static-constrained-joint-refit-20261105-02/README.md)。

逐档静置报告：`scripts.agent.real2sim.analyze_static_point_collection --collections <采点目录...> --output <新目录>`核点NPZ哈希、模型/配置身份，按实际侧点筛选并按原始帧去重，输出同姿态重力/弹簧力臂报告。`fit_static_sweep_directions --fit <fit_static_spring_unbounded的report.json> --output <新目录>`排除未知初始方向，对单线、分方向两线、左右分方向四线做同轮交错留出；不能当独立重复或动态验证。`audit_static_sweep_capture --run <带usb/points/static-loads的本轮根目录> --output <新JSON>`复查所有有效静置窗口及闭链/势能导数/虚功，记录原始SHA；不声明全运动流无缺口或实测轴矩。[新轮58侧点结果](../../artifacts/real2sim/h7-static-calibration-restart-20261105-01/README.md)。

2026-11-05后续采样经用户改为“保留10mm拨轮，按真实腿长拟合”：`collect_static_leg_grid --actual-length --session <新步进USB会话> --output <新目录>`按每个稳定目标档保存左右实际长度，不再要求落在整数10mm网格；默认固定网格模式保留用于历史复现。`view_leg_capture_status --session <同会话> --viewer-port 8778 --point-status <采点目录/status.json>`显示当前目标两侧均已记录才允许提示换档。新增非网格长度接受检查后共11项候选过滤通过，现场5秒500帧两侧保存与页面提示核对成功。拨轮10mm目标变化不等同实长10mm；移动/停留不足5秒不作为合格点。

10mm静态采点：`collect_static_leg_grid --selfcheck-frames <冻结static-frames.npz> --output <新检查.json>`核候选5秒/10mm±2mm网格、连续序号、模式/故障/反馈龄/静置过滤；`--session <运行中USB记录> --output <新目录>`只观察后续数据，左右各自180～320mm按10mm记录，不接管USB、不发控制。拨轮单次10mm固件另用`check_h7_leg_test_mode`核实际C准备/步进/按住不重发/100ms回中/50mm/s平滑和保护，并运行`check_h7_validation_capture`、`check_h7_leg_capture`及两机build-only。类型4 bit7的新步进实录不得套旧连续速率反馈回放。[候选与范围](../interfaces/20261105-h7-leg-step10mm.md)。

静态弹簧无上限假设：`mjlab_training/.venv/bin/python -m scripts.agent.real2sim.fit_static_spring_unbounded --reports <冻结静置report.json> <另一份report.json> --output <新目录>`仅复用已验证同姿态重力/弹簧力臂，比较150N封顶、逐腿逐姿态不封顶、共享恒力及共享正斜率线性曲线；解析解与lstsq及投影正交残差核对，0物理积分。拟合/评分使用相同数据，不能当独立验证或直接改训练；只解释所测压缩区间。[放开150N后的结果](../../artifacts/real2sim/h7-static-spring-unbounded-20261105-01/README.md)。

当前静置受力：`mjlab_training/.venv/bin/python -m scripts.agent.real2sim.analyze_current_static_load --session <运行中USB记录> --output <新目录>`冻结最后32MiB完整行，筛≥1秒连续/新鲜/回中/姿态与力矩稳定片段，以现有模型虚功求重力/气弹簧折算力矩，再计算电机贡献的等效轴向力。0积分，不以其代替GPU动力学验收；外部接触状态须由操作者说明，单姿态不能唯一辨识摩擦/弹簧或校准轴矩。[新静置样本与独立导数核对](../../artifacts/real2sim/h7-current-static-load-20261105-01/README.md)。

摆腿/腿长诊断：`mjlab_training/.venv/bin/python -m scripts.agent.real2sim.analyze_tilt_length --session <USB记录目录> --output <新目录>`冻结正在录制流的完整行前缀并保存字节数/哈希，只读配对的新鲜关节角与板端腿长/腿角，比较前后角差、二维交点重算及固定角差的共同转动。0物理积分；记录有缺口时只报告已观察跨度，不宣称完整运动验收。真实中心距仍需独立测量，非零请求/策略6不能作为纯失能手摆。[本次结果](../../artifacts/real2sim/h7-tilt-length-20261105-01/README.md)。

多姿态静置组合负载：`scripts.agent.real2sim.analyze_leg_stationary_load --output <新目录>`只读3次既有架空记录，从100Hz镜像筛策略6/出力有效/拨轮回中/新鲜且≥.9s、位置跨度≤.003rad、报告速度≤.03rad/s的静置，沿闭链虚功核重力与弹簧力臂并给出假设静态支撑容量下的推力区间，0积分。近限位点不做条件曲线估计，零请求长腿初态单列对照；不能将条件最小残差解当实测气弹簧/摩擦或用静置辨识阻尼惯量。实际18段、180项区间/LP核对与假设边界见[静置负载分析](../../artifacts/real2sim/h7-stationary-load-20261105-01/README.md)。

共享闭链接入：`scripts.agent.real2sim.check_training_loop_integration --output <已准备before.json/before-source.xml/before-model.xml的新目录>`检查当前训练注册线/原生加载器、20,214标定姿态及前后各20环境3秒真实GPU接触；`--policy-only`另做冻结8000+2°、20例同初态GPU回归，之后用`mjlab.sh accept`评分。`audit_training_policy_geometry --experiment <运行目录> --output <新目录>`对保存策略姿态逐10ms复算闭链与限位，0积分，不漏报与2ms接触测试不同的精度范围。本次接触测试通过/20例策略无退化，但策略采样闭链峰值.10644mm略高.1mm；更硬候选.12086mm且一例策略退化，已拒绝。正式987尚未跑，不把20例诊断称完整验收；命令/已选版本/冻结来源见[接入记录](../../artifacts/shared_validation/closed-chain-training-integration-20261104-01/README.md)。

多轨迹响应联合拟合：`scripts.agent.real2sim.refine_leg_response_joint`为原生静/动阻力13维/侧，`refine_leg_response_elastic`为带预滑动15维/侧，均显式给`--references`、配对`--rc-inputs`、`--seed-parameters`及有限代数/种群。仅固定机身无接触可分侧选优，组合后必须重新单候选GPU复核。`select_leg_response_candidate --report-pairs <JSON> --output <新目录>`核相同实录/模型/输入长度，按两轨迹全窗双分支均权评分，0积分；`report_leg_response_joint --experiment <根目录> --runs <单候选GPU目录...> --output <新目录>`复算逐轴指标、原始输入/RC哈希、步前控制/弹簧、初载/闭链/耗散和受保护来源。最终保存用`fit_leg_response_gpu --sample-stride 1 --friction-law elastic`，1ms积分只在开始注入实测状态。位形衰减摩擦/单调分段负载仅为诊断候选，Dahl目标容量不等于实际阻力。当前正式核验一致、四轨迹响应门槛全未通过，不写共享训练。[完整参数和命令](../../artifacts/real2sim/h7-joint-response-fit-20261104-02/README.md)。

四关节响应重拟合核验：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.real2sim.report_leg_response_refit --experiment artifacts/real2sim/h7-joint-response-refit-20261104-01 --output <新目录>`复算固定候选在宽幅/慢/快/原轨迹的逐轴误差，检查实录初态/插值、共享源与1366固件哈希、诊断推力公式和实际执行器夹限；0积分。`fit_leg_response_gpu`参数JSON可显式给`spring_reference_force_n`、`spring_stiffness_n_m`、`spring_force_limit_n`双侧数组，表示5mm压缩参考力+压缩增力，仅改变诊断副本并放宽其弹簧执行器允许范围。默认共享公式不变；厂家150N不等于150N最大力，实际曲线仍待核实。本次核验一致、响应验收失败，不写回训练。[结果与复现](../../artifacts/real2sim/h7-joint-response-refit-20261104-01/README.md)。

新版保护实录复查：`scripts.agent.real2sim.report_h7_boundary_return --session <完成会话> --decoded <正式解码> --start-board-s <投入后起点> --end-board-s <主动退出前终点> --output <新目录>`检查同id状态/请求配对、选段序号连续、bit6/许可/故障、反馈龄、轮零矩和40Nm，并按既有几何反算边界外轴向请求及初态返回/正常跨界返回。两端10ms仅用于完整配对和序号核对；未观察到正常跨界就明确未验证，不以初态恢复替代。只用请求推算力，不是独立轴力/轴矩；0物理积分。已录265.15秒无锁止，正常跨界因未发生仍缺测。[证据与复查命令](../../artifacts/real2sim/h7-leg-soft-boundary-20261104-01/README.md)。

腿长边界返回保护：`check_h7_leg_test_mode`覆盖实录330.1345mm跨界保留收回、两端抑制反方向力（含D项）、连续目标限幅/40Nm及原失能/故障/准备/初态流程；`check_h7_leg_capture`使用生产头文件并核类型4flags bit6。回放/批量反馈按bit6选择新边界返回或旧锁止，不给旧实录套新算法。`scripts.agent.real2sim.check_h7_boundary_replay --before <原GPU回放> --after <新代码重放> --output <新JSON>`核源/初态/实录输入/模型相同，直接请求逐位一致，自反馈输出按GPU数值容差比较，逐拍弹簧公式复算，不把输出当输入要求逐位相等。本次旧8秒新GPU16,000环境步、0拟合/NN/更新；不替代新保护的带动力验证。[构建、烧录、断动力检查及边界](../../artifacts/real2sim/h7-leg-soft-boundary-20261104-01/README.md)。

独立往返响应验证：`scripts.agent.real2sim.validate_leg_response_gpu --session <新USB会话> --decoded <正式解码目录> --frozen-candidate artifacts/real2sim/h7-response-validation-20261104-01 --window-start-board-s <静止起点板时> --duration-s <连续毫秒整数时长> --output <新目录>`。核候选/源哈希、会话晚于冻结且不同于拟合实录、两方向各≥1.5倍速度差、反馈龄<5ms、序号/输入覆盖，再做GPU同力矩/同目标双分支连续回放，反向不重置状态。固定门槛为各关节RMSE<3°、峰值<5°、末端<1°、闭链<.1mm；不足1°活动的轴另列弱激励，不认定全行程响应标定。`check_leg_response_validation --output <新JSON>`用原实录核8秒兼容/23秒前缀/截断拒绝与新速度协议；0物理积分。固件当前v3专项`check_h7_validation_capture --output <新JSON>`检查实际C打包与USB忙时/失败提交，不替代板端或动力验收。通用检查现有v2混合会话断言不兼容v3，不写成整套通过。[当前采集状态与完整命令](../../artifacts/real2sim/h7-response-validation-20261104-01/README.md)。

响应拟合候选：`scripts.agent.real2sim.fit_leg_response_gpu --reference <已校验stiff回放> --parameters <有限参数JSON列表> --fit-until-s 8 --feedback-validation --rc-inputs <已校验RC缓存> --output <新目录>`按候选并行GPU自由回放；仅初态注入，后续实测只评分。`--friction-law elastic`用于含正stiffness_nm_rad的预滑动候选，刚度0维仍原生摩擦；初载仅由静止初态和静态虚功投影确定，禁止自由机身或运动初态。armature_kg_m2须每批相同并重建全部模型常量。`replay_leg_test_gpu`加`--constraint-check stiff --response-parameters <候选JSON>`做1kHz双分支独立复查，支持阻力/预滑动/等效惯量，源XML/固件不改，reuse必须匹配完整参数。`extend_leg_response_inputs`冻结连续23秒同轨迹输入；`report_leg_response_alignment`复核输入/初态/允许模型差异、峰值/末端/额外保持、推力与预滑动状态/离散耗散，导出仅固定机身候选，0新积分。详见[完整命令及候选边界](../../artifacts/real2sim/h7-response-align-20261104-01/README.md)。

实测限位GPU回放：`scripts.agent.real2sim.replay_leg_test_gpu --session <原实录> --decoded <正式解码> --spring-model linear-damped --output <新目录>`按原快照做旧限位；加`--calibrated-model resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml`显式选择新限位，仅允许lf1/rf1 range改变，保留原数据哈希校验，原XML从model-versions冻结归档加载。`linear-damped`复用当前训练线性弹簧和50阻尼，旧`linear`仅弹性，默认constant为历史条件，命令须显式选定。GPU逐拍保存两膝实际约束反力；`scripts.agent.real2sim.compare_calibrated_leg_replays --before <旧目录> --after <新目录> --output <新目录>`核对仅限位改变、相同初态/输入/网格，逐拍复核共享弹簧力，输出四电机角误差和限位反力。比较0积分。[本次结果/复现与数值变形限制](../../artifacts/real2sim/h7-calibrated-limits-gpu-20261104-01/README.md)。

USB限位汇总与应用后静态检查：`scripts.agent.real2sim.calibrate_h7_joint_ranges --session <会话> --endpoint-reports <lf收/伸/收/伸、rf收/伸/收/伸八个目录> --reference-raw <已验证同映射8路原始字节> --output <新目录>`读取原会话XML快照，核对当前模型只允许膝范围变化，将全部有效姿态/参考向外取整0.01rad，交付候选、不写源文件。`scripts.agent.real2sim.check_h7_joint_limit_calibration --calibration <汇总目录> --firmware-source-manifest <已烧录来源清单> --output <新JSON>`要求源XML匹配候选，核对全部姿态、8端点重新解算、历史参考和固件来源。0物理积分；端点报告在仅两膝范围变化后可复算，其他几何或配置变化拒绝。[最终实测与完整命令](../../artifacts/real2sim/h7-joint-limits-20261104-01/calibration-final/README.md)。

八路USB端点报告：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.real2sim.report_h7_joint_endpoint --session <USB标定会话> --mark-index <从1开始> --model-side lf或rf --output <新目录>`。`--model-side`仅按用户已确认的机器人自身侧别填写。检查标记前1.1秒≥90个完整主机解码样本、索引连续、位置跨度≤0.003rad及源XML/配置哈希，输出原始子集、位置/反馈矩统计、闭链角与髋轮投影距离；不将观察极值写入XML，不执行积分。侧别和机械到位需要现场确认，主机连续性不证明CAN序号/源年龄。[首端点和侧别说明](../../artifacts/real2sim/h7-joint-limits-20261104-01/short-endpoint-01/README.md)。

八通道USB真机姿态/限位观察：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.real2sim.view_h7_joint_limits --port /dev/serial/by-id/<H7> --output <新目录> --viewer-port 8778`，先让VOFA断开串口。页面固定机身，仅映射四电机及静态求解闭链；勾选“记录活动范围”，选择端点并保持≥1秒后记录，每个收/伸端重复到达两次。导出的观察极值不是已标定机械限位，须人工核对实物与画面及端点覆盖；原8路无在线/接收年龄/序号/CRC，必须先验证反馈随手动摆腿更新。`--check-only --output <新检查目录>`验证8路分片/损坏重同步/非有限拒绝与原映射闭链，0硬件访问、0积分。[检查与当前运行证据](../../artifacts/real2sim/h7-joint-limits-20261104-01/live-initial-check.json)。

当前训练/正式sim2sim气弹簧一致性：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.check_spring_curve --output <新目录>`检查0～150N线性弹性项、90mm行程、10mm零位压缩量与保留的50N·s/m阻尼；覆盖NumPy、Torch CPU/CUDA、真实动作方法和原生模型执行器写入，但输入为合成状态，0积分/网络前向/优化器更新，不能作行为验收。[本次结果](../../artifacts/real2sim/side-training-linear-spring-20261104-01/README.md)。

腿部静态负载：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.real2sim.check_leg_static_load --session <实录目录> --decoded <正式解码目录> --output <新目录>`自动取拨轮动作前8秒/后10秒的静止段，核对序号/门控/新鲜度/位置跨度，沿闭链可动方向投影重力和150N弹簧负载，并用势能、轴向长度导数及约束虚功复核。只做静态计算，0积分/网络前向/更新；等效弹簧力不等于已辨识物理参数。[实际结果与公式](../../artifacts/real2sim/h7-leg-capture-20261104/static-load-check-02/README.md)。

腿部闭链诊断在上述GPU入口附加`--constraint-check solver`或`--constraint-check stiff`：前者只收紧求解条件，后者在相同求解条件下另减小六个闭链连接的柔度，仅作用诊断副本；输出逐拍求解迭代数及连接点/弹簧端残差。`scripts.agent.real2sim.compare_leg_loop_replays --baseline <基线报告目录> --candidate <求解诊断目录> <柔度诊断目录> --output <新目录>`核对同初态/输入及XML允许差异，并生成比较报告，0新物理步。[本次执行结果与步数](../../artifacts/real2sim/h7-leg-capture-20261104/loop-constraint-comparison-01/README.md)。

腿部实录GPU限位诊断：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.real2sim.replay_leg_test_gpu --session <实录目录> --decoded <正式解码目录> --output <新目录>`默认保留源限位；`--extend-knee-limits`仅在固定机身诊断副本把左膝上限0.12→0.15、右膝下限−0.12→−0.15rad，`--disable-knee-limits`则临时关闭两膝限位，两者互斥。均不改源XML/固件；新范围不是实测物理标定，`--reuse-replay`须匹配诊断选项及保存模型。[实际对照与步数](../../artifacts/real2sim/h7-leg-capture-20261104/knee-limit-comparison-01/README.md)。

H7遥控腿长只读记录：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.real2sim.check_h7_leg_capture --output <新JSON>`检查合成流解码和缺配对拒绝；报告明确标注实际C助手是否得到主机运行检查。[采集准备与局限](../interfaces/20261103-leg-trajectory-capture-plan.md)。

H7_ros架空腿长专用模式：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.real2sim.check_h7_leg_test_mode --output <新JSON>`，需要主机C编译器；实际遥控解码、求解和最终分发函数配替身验证准备流程、限矩、失联及出力门控，不代表整机调度或动力通过。

复旦延迟检查：`scripts.agent.fudan_ppo_walk.check_sim2sim_delays --output <新JSON>`检查独立sim2sim 4+4ms与训练2+2ms配置、实际PD/观测、复位和GPU入口；`check_disturbances`核队列CPU/CUDA边界。仅合成状态，不能当行为验收。[记录](../interfaces/20261103-fudan-measured-training-delays.md)。

复旦原并联模型零速诊断：`scripts.agent.fudan_ppo_walk.probe_reference_zero check --output <新目录>`仅核对原脚本观测/历史/动作和资产指纹；`run`须另获600秒授权，使用原默认ONNX和控制函数，Warp GPU物理，不与本仓串联策略混载。原场景在30cm平台上，窗口内用`report_reference_zero --recording <目录>`对已保存姿态核对支撑面、生成高度修正的派生记录，再用`scripts/mjlab.sh accept --task fudan_ppo_walk --recording <目录>/platform-relative --output <新评分目录>`统一评分。原世界坐标、原评分和视频保留；实际控制周期、证据及复现命令见[本次记录](../../artifacts/fudan_ppo_walk/fudan-original-parallel-zero-20261103/run/README.md)。

复旦显式v2保留续训：`check_preserve_v2 --checkpoint <原7000模型> --output <新JSON>`验证24项实际CPU/CUDA学习状态/RNG恢复、预检合同与采样拒绝、默认v3保留及真实循环7000→8000边界（采样/更新替代，0物理/NN/Adam）；`precheck --history-version v2 --viewer viser ...`才生成本次匹配的真实20环境预检。训练入口显式`--preserve-history-v2`，与`--migrate-history-v2`互斥。完成后`audit_resume_run`核对1000轮、v2→v2、源快照、16项日志与两Adam计数。[实际命令及授权](../../artifacts/fudan_ppo_walk/fudan-parking-resume7000-to8000-20261102/RUNNING.md)。

复旦停车诊断：`fudan_ppo_walk.diagnose_parking --check-only`只检查替换倍率和作用行；动态入口须另获最多300秒预算，记录同初态原模型／停车前向真值／三轴真值三组，冻结权重。`report_parking --experiment <记录> --output <新报告目录>`仅离线CSV／绘图。`check_parking_commands --output <新JSON>`检查实际复位／步进中的10%保持＋10%停车采样、时间边界、关闭回退及保存恢复，物理／前向／更新0；[接口和命令](../interfaces/20261102-fudan-parking-commands.md)。

最终验收现统一为GPU、共用运动门槛及报告，见[统一GPU验收](../interfaces/20261102-unified-gpu-acceptance.md)。主线统一入口为`scripts/mjlab.sh accept --task <任务> --recording <GPU记录> --output <新目录>`；起立单列，旧CPU流程仅保留历史含义。

复旦当前最终验证仅GPU：[约定](../interfaces/20261102-fudan-gpu-validation.md)。`check_gpu_validation --migration artifacts/fudan_ppo_walk/fudan-gpu-only-validation-20261102 --output <新JSON>`检查默认GPU、拒绝CPU参数、几何预检及导出依赖，0物理/NN/更新。历史CPU只读报告不再提供动态入口。

复旦延迟／推扰开关：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.check_disturbances --output <新JSON>`，覆盖实际PD前动作队列、传感槽位／Critic、局部复位、单拍推力、CPU/CUDA一致性、GPU夹具条件、checkpoint及预检条件核对；全部真实物理／网络前向／优化器更新为0。动态预检与行为另行验收，见[配置和时序](../interfaces/20261102-fudan-disturbance-switches.md)。

SuperMimic V13 r2已按实际更新数完成：从A700完整恢复后恰新增1500次actor Adam到A2200，无用户时长上限，不因摔倒或平衡丢失早停；精确封存A1200/A1700，训练结束后才串行评估。正式训练/完成审计、三次固定seed43/96例评估及pipeline均退出0；累计W9056/E/A/C各2200，新增W6000/E/A/C各1500、新399360/累计642048转移，学习循环4509.199秒/训练进程4522.847秒。A1200/A1700均存活0/96、平衡0/96，阶段128/384与112/384；A2200存活96/96、平衡仍0/96、384/384完整，数值异常均0。A2200前进vx −.493～−.492/MAE .992～.993m/s，后退vx −.684～−.683/MAE .183～.184m/s；停车水平mean .553～.589/peak .795～.850m/s、高度.303～.305m。末1秒每轮/总承重持续条件均96例失败，机身离地/姿态/角速度各0例失败；不能写成整段无接触。仅存活和部分跟踪恢复，平衡未恢复，不推断更多更新必然有效或无效。[r2实际命令、退出与结果](../../artifacts/supermimic/supermimic-v13-continuation-20261101-r2/README.md)。101生产源/算法/物理不改；固定入口及Recorder兼容副本恢复另存属于目录迁移维护，不作算法证据。

按任务选择正式入口；检查输出必须注明模型、配方和覆盖范围。旧实验的详细命令保留在[历史专项检查登记](script-check-reference.md)，脚本旧路径统一查[迁移表](../../scripts/agent/MIGRATION.md)。

## 环境与入口

- mjlab 使用 `scripts/mjlab.sh` 或 `mjlab_training/.venv/bin/python`。直接运行辅助模块前设置 `PYTHONPATH=mjlab_training/src`。
- Isaac 使用 `.env.local` 的 `WHEELLEGGED_PYTHON`；先导入 isaacgym 再导入 torch，不安装新依赖或混用 mjlab 环境，见[旧任务规则](isaac-legacy.md)。
- 仓库没有单元测试框架或 CI。下表是正式程序检查／行为测量入口，不统一称作单测。

| 范围 | 程序检查模块（前缀均为 `scripts.agent.`） | 行为验收与命令 |
|---|---|---|
| V13 v5非零延迟/随机化 | `supermimic.check_v13_v5_training_conditions --conditions delay_only或training`；`supermimic.report_v13_v5_training_conditions` | 冻结双模型，真实PD FIFO和推力调度快照，0～10ms分桶、近零/虚响应及首回合失效记录；严格完整窗口复算、实际训练参数覆盖。0次更新，未覆盖活跃推力时必须注明 |
| V13 v5直接速度约束验证 | `supermimic.check_v13_v5_direct_response`；`supermimic.score_v13_v5_repair_horizons` | 独立MuJoCo原生Jacobian核对真实LINK vx、终态积分/公开step等价、双精度数值梯度；固定数据/尺度比较10/80/240ms自由预测。检查0次更新，通过不代表控制恢复 |
| V13 v5真实轮响应与修正候选 | `supermimic.diagnose_v13_v5_response`；`supermimic.repair_v13_v5_response --check-only`；`supermimic.report_v13_v5_response_repair` | 同状态原生GPU正负轮动作10ms分支与预热/最终模型比较；真实配对监督、直接LINK速度响应、原回放约束、独立seed留出及冻结/梯度检查。实际修正须另获明确更新及时间预算，候选不自动部署；旧300次执行源码按SHA另存，不能用新版损失冒充旧结果 |
| V13 v5限额试训审计 | `supermimic.audit_v13_v5_trial --run <本轮目录> --output <新JSON>`；`supermimic.report_v13_v5_trial --experiment <本轮实验目录>` | 核对SM615初始化、4000世界预热隔离、实际策略300次上限、7项网络/优化器与课程/CPU RNG恢复、源码及96例初始/最终测量；CUDA RNG是否验证单列，不把恢复检查当行为通过 |
| V13动作与失败时序 | `supermimic.audit_v13_action_failure_timing --output <新目录>` | 只读10模型各96例原始轨迹，哈希/首回合/100Hz去重核对，完整和统一时窗、左右轮/腿超限及失败前时序；不执行仿真，不据相关性确定阈值 |
| SuperMimic动作±3 | `supermimic.check_supermimic_action_clamp --integration`；参数见[接口记录](../interfaces/20261101-supermimic-action-clamp.md) | 验证噪声后限幅、执行/回放/奖励/下一观测动作相等、想象梯度、非有限停止、ONNX及新旧保存恢复语义；零优化器更新，不代表行为改善 |
| SuperMimic想象状态夹紧 | `supermimic.check_supermimic_imagination_clamp`；`supermimic.check_supermimic_no_history --integration` | 检查本机XML限位、机体系速度边界、范围内恒等/超界梯度、非有限保留、仅想象调用及真实标签不变；零优化器更新，不代表行为改善 |
| SuperMimic稳定性配方核查 | `supermimic.audit_supermimic_stability_controls` | 只读比较参考代码的状态夹紧、动作限幅和余弦调度；统计封存轨迹动作超界/关节限位和实际优化器学习率，不训练、不改配方 |
| SuperMimic单步终态 | `supermimic.check_supermimic_control_endpoint`；`supermimic.check_supermimic_no_history --integration` | 验证129→46、真实10ms终点监督/归一化、单端点路径、直接终态观测奖励、162维价值输入与有限反传；真实仿真5子步记录保持；零优化器更新 |
| SuperMimic航向无关输入 | `supermimic.check_supermimic_heading_inputs` | 对照参考47维特征、检查整体yaw/平移不变性、外力坐标、预测路径等变、四元数符号和梯度；同为129维的旧坐标合同也拒绝续训；零优化器更新 |
| SuperMimic无动作历史世界模型 | `supermimic.check_supermimic_no_history`（可加`--integration`） | 验证129维输入、历史扰动/梯度独立、当前动作梯度、归一化、旧包拒绝/旧actor继承和新包往返；零优化器更新，不代表行为通过 |
| SuperMimic | `supermimic.check_chuanliantui_supermimic`、`supermimic.check_supermimic_alternating` | [配方与验证](../mjlab-chuanliantui-supermimic.md)；旧配方检查不能代替当前算法检查 |
| SuperMimic预测误差 | `supermimic.record_supermimic_physical_prediction`、`supermimic.score_supermimic_prediction_details` | 冻结策略采集24环境/seed55诊断，再离线测10/80/240ms运动、双腿几何、观测与奖励误差；参数零更新，不代替96例行为验收。历史对照输入由 `supermimic.score_supermimic_prediction` 生成 |
| SuperMimic预测进步对照 | `supermimic.compare_supermimic_prediction_progress` | A700/A1808/A2200在相同当前及历史窗口比较；整体误差统一用A2200归一化尺度，避免尺度变化造成虚假改善；零训练、零新仿真 |
| SuperMimic塌落诊断 | `supermimic.diagnose_v13_collapse`、`supermimic.report_v13_collapse` | A4200/A5200估速器与控制网络交叉，`--parts-only`固定A5200估速、分别替换腿/轮动作；2秒、seed43/96例冻结诊断，同初态/物理逐项核对，不代替30秒行为验收 |
| SuperMimic轮动作方向诊断 | `supermimic.diagnose_v13_wheel_response`、`supermimic.report_v13_wheel_response` | A5200在0.2/0.4/0.6秒恢复原生GPU状态与PD历史，测左右轮两幅度正负扰动的10ms响应；以重复/原生基线误差筛选可判定样本，并核对自动微分方向；无训练，不代表完整240ms策略目标梯度 |
| SuperMimic左右轮输入核对 | `supermimic.check_v13_wheel_inputs` | 对已封存3×96状态检查141维输入的动作/历史/轮速槽、归一化、物理轴向及响应；CPU零训练，不把局部映射通过解释为全部训练分布或梯度正确 |
| SuperMimic错误响应机制 | `supermimic.diagnose_v13_wheel_response --history-only`、`supermimic.analyze_v13_response_mechanism` | 零延迟下固定物理状态/当前动作，仅改历史FIFO及模型上一拍动作；结合A4200/A5200同状态的速度/旋转贡献分解，保留噪声不明确样本，不等同完整策略损失归因 |
| SuperMimic历史数据分析 | `supermimic.analyze_v13_history_data` | 只读原训练初始延迟分布、原源码与诊断录制；核对动作历史标签和相关性，明确缺少完整在线回放时不能追溯具体训练批次 |
| SuperMimic世界模型修正审计 | `supermimic.check_v13_world_repair --experiment <修正目录>` | 审计300次世界模型更新、冻结网络/归一化、训练数据哈希及独立验证门槛；审计通过不等于候选采纳或机器人行为恢复 |
| LQR／BC／SMP | `imitation.check_lqr_vmc`、`imitation.check_bc`、`imitation.check_smp` | [teacher](../mjlab-lqr-vmc-teacher.md)、[BC](../mjlab-behavior-cloning.md)、[SMP](../mjlab-smp-prior.md) |
| 实采／CPU镜像 | `real2sim.check_real2sim_huananhu`，其他按 [real2sim目录](../../scripts/agent/real2sim/) | [实采命令](../commands/real2sim.md)、[sim2sim约定](sim2sim.md) |
| Isaac | [isaac目录](../../scripts/agent/isaac/)及原 `sim2sim/` 入口 | [训练→回放→sim2sim](../commands/isaac.md)；保留旧环境和接口限制 |

模型、参数和必需输入查表中任务文档；不要只凭模块名运行。`check` 有限数值检查、`--selfcheck` 形状检查和地面起立统计都不能单独代表行为通过。串联腿正式验收遵循[双轮平衡标准](../interfaces/chuanliantui-wheel-balance-acceptance-v2.md)，原成功样本任一失效即判退化。

2026-11-01确认[统一运动标准 v1](../interfaces/chuanliantui-motion-acceptance-v1.md)：前后速度平均绝对误差≤0.10 m/s、转向平均绝对误差≤0.10 rad/s、停车稳定阶段水平速度峰值≤0.05 m/s、稳定巡航及停车阶段俯仰绝对值峰值≤5°。起立暂不纳入本次统一。当前保留SuperMimic的共用运动判定；旧平地跟踪（对应旧训练线已清理）的0.30门槛不能代表新标准通过；刹停时限等完整协议仍待统一。

## 保留训练线的正式检查

```bash
export PYTHONPATH="$PWD/mjlab_training/src${PYTHONPATH:+:$PYTHONPATH}"
mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_retained_training
mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_script_layout
mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_script_merges
mjlab_training/.venv/bin/python -m scripts.agent.maintenance.check_script_entries
```

第一项检查五个旧task拒绝、普通PPO训练拒绝、⑥⑦⑧导入、共用环境源码／MuJoCo保护哈希、共用函数AST和实际删除清单。后续检查覆盖迁移表、内部导入、保存的SuperMimic预测／阶段指标、计数舍入与入口参数；均不训练、不推进物理，不证明行为通过。

统一运动复算：`python -m scripts.agent.motion.assess --measurement <SuperMimic测量目录> --output <新结果.json>`；算术检查用`--selfcheck`。原stand_v2专属审计入口随任务清理删除。完整范围见[保留线清理记录](../maintenance/20261101-retain-678-review.md)。

复旦PPO新增正式入口：`scripts.agent.fudan_ppo_walk.check`（静态几何/接口/梯度路由，0物理步/0优化器更新）、`check_scoring`（合成轨迹评分）、`precheck --viewer viser`（20环境可视物理预检）、`probe`（并行规模）、`evaluate [gpu]`、`score`、`render`。物理入口均计本轮3600秒，命令及门槛见[规格](../mjlab-fudan-ppo-walk.md)，程序检查通过不能代表行为通过。

复旦v2正式零物理入口：`scripts.agent.fudan_ppo_walk.check_reset_geometry`（含float32写回）与`check_reset_v2`（真实控制流/替代物理、计划热步1次但实际0）；`check_initialization`当时指向后者，当前默认检查已升history v3。动态20环境预检尚未运行，见[交付](../../artifacts/fudan_ppo_walk/fudan-reset-v2-20261101/README.md)。

复旦验收v2（独立于reset/history版本）：`fudan_ppo_walk.check_scoring_v2 --output <新JSON>`合成70项边界与拒绝检查；`score_v2 --recording <既有目录> --output <新JSON>`；`compare_scoring_v2 --before <6000录制> --after <7000录制> --height030 <可选另列录制> --protected-source-manifest <本次源SHA清单> --output <新目录>`。只读既有记录，无物理/策略前向/优化器更新，不代表新行为或实机验收；[独立协议](../interfaces/20261102-fudan-walk-acceptance-v2.md)。

复旦history v3正式零物理入口：`fudan_ppo_walk.check_history_v3 --output <新JSON>`（当前27项真实输入/复位/CLI控制流，替代后端，无NN前向）；`check_history_migration --checkpoint <7000模型> --baseline-source <修改前fudan_ppo_walk目录> --output <新JSON>`（CPU/CUDA模型/两Adam/学习率/RNG/计数，受限显式v2→v3迁入及拒绝检查，0更新/0前向）。当前`check_initialization`指向v3；历史`check_reset_v2`仍可验证旧语义。无物理夹具不批准真实预检或训练，详见[接口](../interfaces/20261102-fudan-history-rollback-v3.md)。

复旦原生sim2sim只读对照：`fudan_ppo_walk.compare_native_replay --experiment <配对录制目录> --gpu-score <GPU-v2分数> --output <新JSON>`核对同15例/初态并列指标；`compare_native_video --experiment <配对目录> --output <新目录>`只处理已编码GPU/CPU视频帧，0物理/0NN/0网格渲染。`check_scoring_v2`增至71项，补不完整数值终止的标志保留检查，不改门槛。[本次实际两组15例及预算](../../artifacts/fudan_ppo_walk/fudan-sim2sim-7000-h030-20261102/README.md)。

复旦后退起步只读分析：`fudan_ppo_walk.diagnose_reverse_startup --experiment <既有原生配对目录> --output <新目录>`核对旧轨迹/源码哈希、倾斜与触地先后、姿态罚饱和及CPU/GPU forward/力记录相位；不导入策略或推进物理，不以时间先后代替唯一因果证明。[此次结果](../../artifacts/fudan_ppo_walk/fudan-sim2sim-7000-h030-20261102/reverse-startup-analysis-r3/README.md)。

复旦历史刷新时序诊断的动态入口`probe_refresh_timing`、`check_refresh_probe`、`audit_refresh_repeat`已随CPU后端撤销；原源码保存在[清理前快照](../../artifacts/fudan_ppo_walk/fudan-gpu-only-validation-20261102/before/)，只读`report_refresh_probe`与[原结果](../../artifacts/fudan_ppo_walk/fudan-refresh-timing-7000-20261102/README.md)保留。

复旦估速只读核查：`fudan_ppo_walk.diagnose_estimator --experiment <既有refresh探针目录> --output <新目录>`检查7000归档的标签/倍率/真实storage随机批配对、初始速度可观测性、25维打包及125维v3重建；复用已保存预测，不执行模型前向或新物理。[证据](../../artifacts/fudan_ppo_walk/fudan-refresh-timing-7000-20261102/estimator-analysis-r2/README.md)。

复旦估速输入归因：`fudan_ppo_walk.attribute_estimator_inputs --check-only`做12项分组/替换/线性与交互归因合成检查，0物理/NN；`--budget <新窗口JSON> --output <新结果目录>`从原7000归档构造冻结encoder，仅对既有历史做同CPU复算和7组128子集离线替换，禁止复用过期窗口；`--report-only <已有实验根目录>`只读预测绘图，0新增NN。归因解释网络输出差，不能代替新物理闭环因果实验。[此次结果](../../artifacts/fudan_ppo_walk/fudan-estimator-input-cause-7000-20261102/README.md)。

复旦CPU/GPU分叉追溯：`fudan_ppo_walk.diagnose_engine_divergence --experiment <已有refresh探针目录> --output <新目录>`只读取5条状态/动作/实际力矩记录及7000封存弹簧公式，加载mjb仅取关节/选项元数据；核对初态、同GPU动作、早期转移与GPU重复性，0物理步/0NN/0更新。不以相位未对齐的接触力判断先触地，不代表完成同状态单步物理干预。[本次结果](../../artifacts/fudan_ppo_walk/fudan-engine-divergence-7000-20261102/README.md)。

复旦历史同状态单2ms动态入口`probe_single_step`已撤销，原源码见[清理前快照](../../artifacts/fudan_ppo_walk/fudan-gpu-only-validation-20261102/before/)；`report_single_step`仍可只读已有记录，0新增物理/NN。[历史结果](../../artifacts/fudan_ppo_walk/fudan-single-step-7000-20261102/report-final/README.md)保持，不作为当前CPU入口。

统一验收正式程序检查：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.motion.check_acceptance --baseline artifacts/shared_validation/unified-gpu-acceptance-20261102 --output <新JSON>`，核对四项边界、CPU拒绝、丢样本/缺字段、任务适配及同初态退化否决，0物理/NN/更新。Isaac仅检查新增GPU守卫AST，未运行旧环境动态。旧评分v2检查仍单独保留。
复旦8000停车积分：`fudan_ppo_walk.check_parking_integral --output <新目录>/checks.json`做门控/20ms均值/限幅/局部复位/非有限/真实v2历史推进、135归档初态替代恢复、严格差值留证、步后反馈时序及失败报告的无物理检查；`run_parking_integral --output <同目录>`仅在当次授权内执行135条GPU三分支、统一评分和三份并排视频。默认关闭，不加功能CLI，不训练；单一600秒及360秒回放硬截止，具体反馈相位、固定窗口和边界见[独立协议](../interfaces/20261103-fudan-parking-integral.md)。

复旦结构整理正式入口：`PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.check_structure_suite --output <新检查目录>`。外层禁止真实物理环境构造、积分及Adam更新；覆盖注册链路、配置消费者、迁移前后逐值比较、冻结8000模型、PPO梯度/顺序、两优化器恢复、v2/v3/预算/日志、积分r2及GPU入口。源迁移用明确哈希链核对，不能关闭来源校验；[说明与证据](../maintenance/20261103-fudan-structure-alignment.md)。
复旦当前选用8000-v2＋停车+2°：`scripts.agent.fudan_ppo_walk.check_selected_policy --output <新JSON>`核对47工况时序、与已执行探针逐值一致、默认/显式模型路由与哈希/积分条件拒绝，0物理/NN/更新。默认GPU评估省略`--checkpoint`才应用选用配方；[接口记录](../interfaces/20261103-fudan-selected-pitch2.md)。
