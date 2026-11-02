# 模型交付记录：2026-11-01／闭链轮腿／复旦PPO首轮候选

**仅仿真候选，用户视频验收不通过（机身撑地）；不是完整参考配方复现。** 本轮漏掉参考入口的首轮随机回合相位，且启动Actor帧/历史噪声曾分开采样；修复后的代码未重训，不能冒充本模型来源。

## 任务与接口

| 项目 | 内容 |
|---|---|
| 日期、交付人、接收人、独立复查人 | 2026-11-01；Codex整理；用户接收；无独立代码复查签认，H7负责人待确认 |
| 机器人／任务 | 本机chuanliantui_new_1实体闭链／fudan_ppo_walk |
| 用途 | 平地前后行驶、原地转、组合转弯、停车、变高；不含真机、越障或恢复 |
| 接口 | [fudan-ppo-walk-physical25-history125-critic144-v1](../interfaces/20261101-fudan-ppo-walk.md) |
| PR | 无，未提交/推送/创建PR |

## 代码与训练来源

| 项目 | 训练仓 | 部署侧 |
|---|---|---|
| URL／分支／HEAD | https://github.com/zn1135/wheelleg_RL_2026.git；26_wheelleg；`3794efd390a0eb68cb154839423a7d53e32ff61e` | H7未改、未交付 |
| 工作区 | 有本任务未提交/新文件；HEAD不能代表实际训练源码 | 本次无修改 |
| 运行源码 | [完整归档](../../artifacts/fudan-ppo-walk-20261101/training-source.tar.gz)，SHA `0eadd487a32f292920901b0e3ac29f5b39bb71ddde91c2d6887f7030270ac971` | 不适用 |
| 运行差异 | [训练差异](../../artifacts/fudan-ppo-walk-20261101/training-source.patch)，SHA `58b26da2765b5e535b748840f3bc58c89e6d196faf82e3d72bf0b19daa1afbb5` | 不适用 |
| 未跟踪源码 | 已包含在完整运行源码归档与原run/source/ct_mjlab中 | 不适用 |

| 训练信息 | 内容 |
|---|---|
| 实际命令／seed | 专用CLI train，512环境，seed1，从零训练；[实际命令](../../artifacts/fudan-ppo-walk-20261101/COMMANDS.md) |
| 更新量 | 1678轮，41238528转移，PPO/encoder各33560次Adam；训练循环1773.3547秒 |
| 配置 | [原始来源哈希](../../logs/mjlab/fudan_ppo_walk/20261101_trial01/source_sha256.json)及原run/source；PPO1e-3、48步/4minibatch/5epoch，实际batch24576/minibatch6144；[规格与差异](../mjlab-fudan-ppo-walk.md) |
| 模型资产 | 共享XML SHA `cac8061de2850602bfd1421fc856bf217c4e5040a11c8f5d33cf1abe61d5f0cf`；[XML与所有引用mesh哈希](../../artifacts/fudan-ppo-walk-20261101/asset-sha256.json) |
| 运行环境 | 本仓venv；RTX4060 Laptop 8GB；Torch2.14.0、MuJoCo/mujoco-warp3.11.0、mjlab1.6.0、Warp1.17.0；[实际版本](../../logs/mjlab/fudan_ppo_walk/20261101_trial01/runtime_versions.json) |
| 初始化／恢复 | 无旧模型初始化；checkpoint含两优化器和RNG，但无完整环境/回合恢复入口，不能宣称可中途完整恢复 |
| 日志 | [logs/mjlab/fudan_ppo_walk/20261101_trial01](../../logs/mjlab/fudan_ppo_walk/20261101_trial01)；TensorBoard仅提供命令，未自动启动 |

## 模型与导出

| 产物 | 地址 | SHA256 | 定义 |
|---|---|---|---|
| 完整checkpoint | [model_final.pt](../../logs/mjlab/fudan_ppo_walk/20261101_trial01/model_final.pt) | `b44091b76ca693a1f5a3d030e4149d38b77b770917aa735640f54160a3b6bcce` | actor/encoder/critic、两优化器、计数与RNG |
| ONNX | [policy.onnx](../../artifacts/fudan-ppo-walk-20261101/export/policy.onnx) | `63dea1670328d5d8a4919a6d2adbcf05d82e8a3285d5f11f23c5233ca73f2f14` | float32，obs[1,25]＋obs_history[1,125]→actions[1,6]，opset17 |
| 板端网络／固件 | 无 | 不适用 | H7未适配 |

导出使用ONNX1.22.0，图检查通过；单个随机输入与Torch对比最大绝对误差7.15255737e-07（门槛1e-5）。这不代表板端误差/时序或行为通过。图外仍需帧预处理、历史、动作±100裁剪和实体PD。

## 分阶段验证

| 阶段 | 状态 | 范围 |
|---|---|---|
| 训练运行 | 完成 | 固定期限正常结束，原模型及source不可改写；存在已披露入口遗漏 |
| Isaac | 不适用 | 本轮只有mjlab/MuJoCo |
| 静态接口／20环境预检 | 通过有限检查 | 见[证据目录](../../artifacts/fudan-ppo-walk-20261101/)；不是行为通过 |
| mjlab正式 | 技术未达标 | 987/987完整，0/987联合技术通过，数值异常0 |
| 原生CPU正式 | 技术未达标 | 984/987完整，3数值提前终止，0/987联合技术通过；失败留原分母 |
| 用户视频验收 | 不通过 | [两引擎完整矩阵视频与数值](../../artifacts/fudan-ppo-walk-20261101/README.md) |
| ONNX | 有限通过 | 图/形状与单夹具数值，非板端验证 |
| 板端时序／真机 | 未执行 | 未授权操作硬件，负责人待确认 |

全量正式CPU使用2ms积分；后续0.5ms积分只对3个数值失败原样本做诊断，不替换正式结果，不代表全矩阵验证。闭链、三档高度及动作/传感零额外延迟等条件见规格；后续所有延迟必须实机分段测量。

没有合格源模型，不能声称保留既有成功基线。原模型/源码可供复现此轮；修复版入口在同一工作区，但仅零物理/零优化器检查通过。未授权追加训练。

后续用户反馈“机身撑地啊”，本轮最终视频验收不通过。[只读来源审计](../diagnostics/20261101-fudan-fixed-source-audit.md)已完成；D1～D4尚未实施，原模型、视频、轨迹及运行源码归档不变。
