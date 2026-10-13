# 串联腿 mjlab 实体气弹簧接口 r2

历史r2记录：随后按用户要求加入随机初态，当前定义见 [r3接口](chuanliantui-mjlab-huananhu-v14-flat-standup-r3.md)。本文保留实体弹簧修正时的检查范围。

日期：2026-10-09。当前分支 `26_wheelleg`，基础提交 `58947e8fabc4683b2025d42bda742b06c84dc566`；本次修正尚未提交。无新训练模型交付，H7未适配。

## 修正与来源

用户在可视化后指出气弹簧有误，指定本机 `mjlab/wheelleg_RL` 为参考。r1使用恒定150 N tendon，缺少真实筒体／活塞杆、质量、压缩曲线和定位辅助杆清理；闭链初态残差很小不能证明这些建模假设正确。

r2移植该工程的实体弹簧结构、0～150 N压缩曲线、阻尼和清理后的两个主腿网格。来源版本及文件哈希记录在 `resources/robots/chuanliantui_new_1/mjlab/spring-source.json`。参考工程存在其他未提交改动，本次只读取指定弹簧实现与资产，没有修改该工程。

新资产独立于旧 `sim2sim/chuanliantui.xml`，含19关节、6组connect、0个tendon、8个执行器。每侧新增转动／滑动关节及末端connect，保留四组主腿闭合、CAD端点与25 mm横向偏移。筒体／活塞杆参考质量0.0305218／0.0923989 kg；主腿原惯量保持，无法据现有资料排除定位辅助件在CAD合计惯量中的贡献。

## 力律与初始化

```text
compression = max(0.01 - q_slide, 0)
elastic = clip((150 / 0.09) * compression, 0, 150)
force = clip(elastic - 50 * qdot_slide, 0, 150)
```

正位移表示伸长，内部执行器仅推不拉；XML ctrlrange和forcerange均为0～150 N。每物理子步独立更新，不经过电机动作延迟或启动门控，不计入六电机功率。预压、参考质量及曲线是模型设定，非实测标定；90 mm是力曲线参考压缩量，不是机械硬行程。

伸缩零位轴向长度0.219204300981 m；零位零速度推力16.6667 N。新增弹簧关节在reset和nominal姿态中均解析求解，不能直接全部清零。

## 策略兼容性与部署

接口ID更新为 `chuanliantui-mjlab-huananhu-v14-flat-standup-r2`，actor35／critic78／action6、六实体电机顺序、50Hz策略、5ms物理子步、Kp10／Kd1及V14 flat v0 PPO保持现有定义。未复制参考工程的另一套电机零位、控制增益或训练任务。

r1恒力模型及所有旧Isaac／H7模型不兼容；加载校验接口ID、资产、姿态、弹簧力律和网格哈希。部署端仍为旧25／125维100Hz模型，需新策略训练与行为验收后再适配。双方共同核对实际弹簧参数、关节映射和时序，并按 [交付模板](../templates/model-handoff.md) 记录模型。

## 验证

见 [实体气弹簧检查记录](../diagnostics/20261009-mjlab-physical-spring.md)。本次无策略训练、真机验证或独立代码复查。
