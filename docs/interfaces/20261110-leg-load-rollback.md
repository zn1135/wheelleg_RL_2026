# 双腿拟合负载默认回退

2026-11-10按用户“将训练和仿真的拟合回退”执行。当前分支`26_wheelleg`，开始时工作区干净，基准HEAD为`12d650623681d9740fbb3f684ac2b3318970480b`。

## 当前默认

- 共用`ChuanliantuiPdActionCfg.fitted_leg_load=False`：SuperMimic、BC、SMP及其共用物理不再默认构造拟合负载。
- 复旦`FudanWalkCfg.sim.fitted_leg_load=False`；GPU`evaluate`省略选项时为`--leg-load-profile legacy`，运行记录为`legacy-linear-spring-v1`。
- 原生MuJoCo`Playback`和`StandV2Playback`同名构造参数默认False，双策略继承此设置。
- 不再叠加局部关节负载、拟合转动摩擦/阻尼、2000Ns/m滑杆阻尼和Dahl摩擦记忆；原启动随机化回到原基线上施加扰动。
- 恢复既有气弹簧：`x=max(0,0.010-q_slide)`，弹性力`clip(150*x/0.090,0,150)`，再减原50Ns/m速度阻尼并夹至0～150N；各任务原有偏置/随机化保持。

共享XML、已标定限位、闭链约束、轮响应、质量/惯量、PD、时序、观测和网络保持。仅显式`fitted_leg_load=True`或GPU`--leg-load-profile fitted`会重放历史候选；独立研究脚本的显式`--response-parameters`不作为默认入口。

参数JSON、拟合实现、原始实录、模型/日志和[Word流程](../diagnostics/20261109-双腿负载拟合流程与训练仿真接入.docx)全部保留。Word及[11-09采用记录](20261109-adopted-mirrored-leg-load.md)是历史记录，不表示当前默认。已有拟合物理训练的模型不会随开关变化而重训或自动变得适用于原物理。

## 验证与限制

当前检查脚本`check_leg_load_rollback`：名义/训练配置各20环境、各155次MuJoCo Warp GPU积分调用；核实际摩擦/阻尼回到XML基线、额外关节施力全零、逐步原气弹簧公式一致、状态有限。共同训练配置及原生构造默认关闭也已核对，共17项通过。原气弹簧独立公式/实际控制路径26项检查通过，未积分。

原8000-v2＋停车2°，使用省略物理选项的默认GPU入口，20例均完整，统一评分技术20/20。此为20例诊断，不是完整987例、起立或用户视频验收通过。当前部署14000模型尚未在回退物理下复测。

XML SHA256仍为`505c9b9bb2b7a20e15cf9d2cd4ca664332db402687e34185d8a57de173dedbe8`；拟合JSON仍为`e6ef82fa4380fa01341fa2166be56282c5290c82fec2b2a04e5b3107ce49218f`。

本次0训练更新、0硬件命令，无新PT/ONNX/固件交付；H7独立复查待确认。[执行命令及证据](../../artifacts/length-response-fit/20261110-leg-load-rollback-01/README.md)。后续优先对当前使用的14000模型按回退物理复测，再决定使用哪个模型。
