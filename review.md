# 气弹簧与闭链可视化代码复查

## 修复回归（2026-09-18）

用户要求全部修复后，R1、R2、R3、E1 已修复；下方原始审查结果与哈希保留为历史证据，不代表修复后版本。

- R1：生成器从 CAD marker 推导四个气弹簧 site，生成两组 tendon/motor，并保留 `lf00/rf00 limited=false`，六个实体电机顺序不变。
- R2：检查器更新为 8 个执行器；新增 site 所属 body、tendon 两端、执行器传动/限幅、0/150 N 施力、实体电机地址以及 ±360° 闭合检查，保留原有动力学断言。
- R3：XML、CLI 提示/帮助和文档统一为“伸张推力”，不改变力的符号。
- E1：主循环退出时只要 viewer 存在就关闭；已创建窗口后的视图初始化异常也先关闭再降级运行。

回归结果（指定 wheellegged_py38 环境）：

1. `scripts/agent/check_chuanliantui_closed.py` 通过；5 秒无控制最大销轴误差为 0.000463017501 m，小于原有 0.001 m 阈值。
2. 执行生成器真实 main，拦截 XML 写入，在内存编译其结果；结构、site、关节限位、执行器、tendon 连接和 equality 参数与当前 XML 一致（浮点容差 2e-12），生成结果也通过完整闭链检查。没有覆盖当前 XML 的其他内容。
3. 提取实际源码的 finally/初始化异常处理分支，使用模拟 viewer 检查正常退出、KeyboardInterrupt、RuntimeError 与窗口打开/已停止/不存在的组合，以及初始化失败：所有适用分支均调用 close。此为清理分支诊断，不是完整 main 或真实 GUI 端到端测试。
4. 两个独立 viewer 分别执行 `--headless --duration 1` 通过：机身固定、实体电机零力矩、每侧气弹簧 150 N。复旦源模型约 10 mm 的既存锚点偏差保持警告，没有擅自改动。
5. `git diff --check` 通过。未启动训练、未改动历史权重、未提交；未执行新的站立/行走或实机验收，也未关闭用户已有窗口。

## 版本与范围

- 审查开始：2026-09-18 19:12 +08:00。
- review_commit / 比较基线：`7edd47d98234f3e17dc0022a8c71f4587d13075d`（`26_wheelleg` HEAD）。
- 候选版本：该 HEAD 上的未提交工作区，含两个未跟踪的 viewer 脚本；不是纯提交审查。
- 本次任务没有 manifest、base_commit 或 tests.md。已读取现存 T-20260913-01、T-20260915-02、T-20260915-03 的 manifest，均为其他训练/验证任务，不复用其任务身份。
- 按仓库约定直接在当前工作区审查，不创建 worktree、私有任务目录或资源锁；报告放在根目录。
- 本任务范围：`sim2sim/chuanliantui.xml` 气弹簧与后输入轴限位，`sim2sim/mj_sim2sim_ct.py` 的气弹簧接入/专用视图，新增 `sim2sim/view_gas_spring_ct.py`、`sim2sim/view_gas_spring_fudan.py`，`docs/ai/sim2sim.md`。
- `COMMANDS.md` 和主脚本移除限时、默认遥操作、无限循环等是任务开始前已有的工作区修改；检查其与新代码的交互，但不归因本次实现。
- 相关调用：`sim2sim/chuanliantui_closed_adapter.py`、训练代理 XML、原始 CAD URDF、复旦源 XML。原始复旦目录只读。
- 审查人员：主 Agent 汇总/复核；独立只读 Reviewer `review_dynamics`、`review_viewers`。独立复查已完成，不代表用户采纳。

候选文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| sim2sim/chuanliantui.xml | b02097b5b841db69b7425818abc2d1461178b5dc188b3c03880dc28ef000449f |
| sim2sim/mj_sim2sim_ct.py | f57978eb5c8ca0b8ed49bc8f73cc2d3417688a4294a7e187630e30ecec1823a9 |
| sim2sim/view_gas_spring_ct.py | 1bf71eec4ff6f5dfe12a6eb9cf09d9a04d33fb6c43f1a59aedcf42d8418f278e |
| sim2sim/view_gas_spring_fudan.py | 5169fcb68c2b21baf2cd79f5fb2f630f9c727f842945a37cc3ae93ed9e52de2f |
| docs/ai/sim2sim.md | 9ddfc00e9002c7246fcfc4a4003f0a128ac729f1e08d130f04e9ee16c479efd6 |
| COMMANDS.md（已有改动） | 45bff20aa16b941151074dc4059f09f2b44cf337540d13ce972fa37f4c058a7d |

## 审查结果

结论：需要修改。本次实现存在 2 项 P2 集成遗漏、1 项 P3 文案错误；另记录 1 项任务开始前已有的 P2 viewer 退出问题。没有确认的 P0/P1 问题。未修改候选实现，未提交或启动训练。

### R1 / P2：生成器会丢掉气弹簧并恢复后输入轴限位

- 位置：`scripts/agent/generate_chuanliantui_closed_mjcf.py:219-224,249-255`；关联 `sim2sim/chuanliantui.xml:95-96,122,155-176`、`sim2sim/mj_sim2sim_ct.py:272-278`。
- 触发：执行已有闭链模型生成器，默认输出就是当前 `sim2sim/chuanliantui.xml`。
- 实际影响：生成结果仍只有 6 个实体电机，没有新增的 4 个 spring site、2 个 tendon 和 2 个 spring motor；`lf00/rf00` 从 CAD URDF 恢复为 ±3.14 rad。随后 `--closed_chain` 报缺少气弹簧执行器，360° 旋转修正也丢失。
- 证据：主 Agent 用 `unittest.mock.patch.object(ET.ElementTree, 'write', capture)` 截获生成树、禁止写文件，执行生成器的真实 `main()`。输出 actuator 数量 6、gas spring site 列表为空、两后输入关节 `range='-3.14 3.14'`。候选 XML 前后 SHA-256 一致。独立 Reviewer 核对生成调用链后确认。
- 最小修复：把新增 site 坐标转换、tendon/motor 和 `lf00/rf00 limited=false` 规则纳入生成器；增加再生成后检查。文档要求“重新生成时保留”不能让现有生成命令实际保留这些内容。

### R2 / P2：闭链检查脚本仍要求 nu=6，导致验证入口立即失败

- 位置：`scripts/agent/check_chuanliantui_closed.py:25-28`；关联新增执行器 `sim2sim/chuanliantui.xml:174-175`。
- 触发：使用指定 Python 执行仓库现有闭链检查命令。
- 实际影响：模型当前正确包含 8 个执行器，但检查器期待 6 个；在结构断言处退出，后续闭合误差和 5 秒动力学检查完全没有执行，也未覆盖弹簧施力与后输入轴连续旋转。
- 复现命令：`/home/zn/miniforge3/envs/wheellegged_py38/bin/python scripts/agent/check_chuanliantui_closed.py`。
- 实际输出：`(21, 20, 8, 16, 15, 4) != (21, 20, 6, 16, 15, 4)`；退出码 1。
- 最小修复：更新模型结构预期，检查新增 actuator/tendon/site 的名称、连接及每侧力值，并覆盖去限位的整圈姿态。保留原有闭链误差检查，不能只删除失败断言。

### R3 / P3：实现为伸张推力，文档与提示仍写“拉力”

- 位置：`sim2sim/chuanliantui.xml:156,173`；`sim2sim/mj_sim2sim_ct.py:281,655`；`docs/ai/sim2sim.md:81`。
- 触发：阅读模型说明、CLI 帮助或闭链启动输出，依据其描述判断气弹簧方向或设计补偿。
- 实际影响：`gear=1`、正 actuator force 对应沿 tendon 长度增大方向做功，推开两端；“拉力”表示相反方向，可能误导后续补偿符号。
- 证据：独立检查 `actuator_moment = dl/dq` 与长度有限差分一致；150 N 在默认姿态产生左/右约 −7.150267/+7.150267 N·m。此前对话已确认术语有误，但候选文件尚未更正。
- 最小修复：统一改为“恒定伸张力”或“推力”；不要为迎合错误文案反转当前 `gear` 或 `ctrl`。

### E1 / P2（已有工作区修改）：仍打开的 viewer 在退出时不会被关闭

- 位置：`sim2sim/mj_sim2sim_ct.py:618-620`。
- 来源：任务开始前已有的主循环/遥操作修改，不归因新增气弹簧代码。由于新 `--gas_spring_view` 复用同一循环，列为当前候选交互风险。
- 触发：窗口仍开着时主线程收到 Ctrl+C，或主循环中发生异常。
- 实际影响：`finally` 仅在 `not viewer.is_running()` 时调用 `close()`，恰好跳过仍在运行的 viewer。调用 `run/main` 返回或抛错时无法保证释放窗口及渲染资源。
- 证据：真实 MuJoCo 初始化，假 policy/viewer，令 `mj_step` 抛 `KeyboardInterrupt`，运行真实 `main()`；结果 main 返回但 viewer 的 `close` 标记为 False。不涉及真实 GUI 进程残留的断言。
- 最小修复：若 viewer 非 None 就调用 close，或用上下文管理器覆盖完整生命周期。若启动后设置气弹簧视图失败，也应先关闭已创建的 handle 再处理错误。

## 已检查事项与证据

| 项目 | 结论与证据 |
|---|---|
| 功能正确性：端点 | 4 个气弹簧 site 的零位基座坐标与 CAD 标记误差最大 1.068×10⁻⁹ m，左右镜像一致。 |
| 功能正确性：施力/标签 | 弹簧 ID 与六个实体电机不重叠；每内环步进前写入力值。两 viewer 的显示力矩与独立长度差分吻合，误差 <3×10⁻¹¹ m/rad；只显示直接局部关节力矩，不冒充电机轴补偿。 |
| 闭链与连续转动 | 前后输入轴联动 −360°～+360° 的 65 个姿态，销轴残差 ≤3.05×10⁻¹⁴ m，力矩映射差约 10⁻¹⁴；没有发现因去掉后输入轴限位而新增的几何错误。 |
| 架构一致性 | 串联训练代理默认保持 neq=0；真实闭链路径显式启用；训练代码和历史权重未改。未发现三处既有观测/步进修复被本任务回退。生成器/检查器未同步见 R1/R2。 |
| 新增 viewer 的控制流程 | 两脚本各执行 1 秒 headless；并用真实 model/data、模拟 viewer 运行各 6 帧，验证 Space 暂停/恢复、暂停时 F5 关弹簧、Home 重置 qpos/time、实体电机 actuator_force 全程零及上下文退出。 |
| 线程与资源 | 两独立 viewer 的共享 model/data/user_scn 写入在 lock 内，sync 在 lock 外，with 负责关闭；主策略 viewer 退出问题见 E1。实际键盘线程竞态未压力测试。 |
| 鼠标外力生命周期 | 独立 Reviewer 查阅 MuJoCo 3.2.2 的 Simulate::Sync，passive 模式先清空 xfrc_applied 再施加当前鼠标扰动力，现有 sync-before-step 顺序与之匹配。 |
| 复旦兼容转换 | 4 组 connect 转换后双方 body 和两端局部锚点与源 site 逐项一致；源模型约 10 mm 横向错位仍存在，headless 与文档明确披露，不能视为闭合质量通过。 |
| 参数边界 | 两 viewer 对负数、NaN、无穷气弹簧力/时长的范围判断可拒绝非法输入；主 CLI 每侧力限为 0～150 N。 |
| 注释与文档 | R3 方向术语应修正；恒力模型不含力—长度曲线、阻尼、行程等边界已明确。未发现需要单独报告的无效/重复注释问题。 |
| 范围外改动 | COMMANDS.md 和主脚本遥操作/无限循环改动为既有工作区修改，未覆盖。复旦 source、原始 CAD、训练环境和 logs 均未修改。 |
| 提交/PR 文案 | 本任务未提供提交信息或 PR 正文，无可审文案；没有提交、amend、推送或 PR 操作。 |

MuJoCo 鼠标外力语义参考：<https://raw.githubusercontent.com/google-deepmind/mujoco/3.2.2/simulate/simulate.cc>，独立 Reviewer 核对 passive Sync 分支约 2121–2125 行。

## 检查范围与未验证部分

- 使用 `/home/zn/miniforge3/envs/wheellegged_py38/bin/python`，MuJoCo 3.2.2；不使用系统 Python。
- 主 Agent 首次为生命周期诊断先 import isaacgym 时遇到 `libpython3.8.so.1.0` 搜索路径缺失；仅对重试命令设置 `LD_LIBRARY_PATH=/home/zn/miniforge3/envs/wheellegged_py38/lib:${LD_LIBRARY_PATH:-}` 后成功。按要求在 torch 前 import isaacgym，没有修改环境安装。
- `git diff --check` 通过。R2 的正式闭链检查失败；其后续动力学断言未执行，不能称该入口验证通过。
- 上述有限差分、模拟 viewer 和 headless 诊断不是仓库单元测试框架，也不替代“训练 → Isaac 回放 → MuJoCo 行为验证”。
- 本轮未启动训练、TensorBoard、Isaac 回放、真实 GUI，未关闭已有用户窗口；未执行有气弹簧策略的完整站立/行走验收。
- 之前会话的可视运行观察：150 N 旧站立策略存在漂移，闭链起立未成功；这些不是本轮独立行为验证结果，不据此把气弹簧恒力实现认定为数值错误，也不能宣称策略已适配气弹簧。
- 没有完整的未提交基线快照能机械剥离主脚本既有工作区修改，归属依据本会话开始时的记录；HEAD diff 和上述最终文件哈希共同界定本次审查候选。
- Fudan viewer 依赖本机外部目录，支持 `--xml` 指定路径；未在其他机器验证资源路径。真实 GUI 文字遮挡、窗口缩放和鼠标拖拽手感未重做人工验收。
- 未执行物理硬件标定、气弹簧行程/力曲线标定或实机测试；未验证其他 MuJoCo 版本。

## 后续处理

先同步生成器和检查器，再修正文案；E1 可作为已有工作区退出问题单独修复。修复后需更新候选哈希并重跑受影响检查；本报告不等于已采纳或已完成行为验收。

---

# 串联腿起立奖励设计复查（2026-09-22）

## 范围与结论

- 审查对象：当前工作区中 `chuanliantui_standup` 的实际生效奖励、成功/终止判据和两阶段课程；基线为 `26_wheelleg` / `e0e6a09` 加未提交候选改动。
- 本轮是主 Agent 的静态与训练日志复查，不冒充独立复审；当前没有可用的本任务 manifest，未创建任务注册或 worktree。
- 结论：没有 P0 数值安全问题；有 **2 项继续续训前必须处理的 P1 课程问题**，以及 3 项 P2/P3 建议。未改奖励代码、未启动训练。

## 必须处理

### S1 / P1：课程状态没有进入 checkpoint；`--resume` 会丢失“永久解锁”状态

- 位置：`wheel_legged_gym/envs/chuanliantui_standup/chuanliantui_standup.py:21-25,48-77`；`wheel_legged_gym/rsl_rl/runners/on_policy_runner.py:329-363`。
- 事实：环境内的 `standup_curriculum_unlocked`、窗口回合数和恢复数只存在 Python 进程内；checkpoint 仅保存网络、两个 optimizer、迭代号和 infos，load 后不会恢复任何环境课程状态。
- 实际影响：一旦在解锁后中断并 `--resume`，环境会重新锁到 `height=0.30 m, orientation=-1`，而不是继续已解锁的 `height=0.20 m, orientation=-10`。这与“永久解锁”的训练语义矛盾，也会让同一策略在续训时突然面对不同目标。
- 最小处理：把这四个课程字段写入/读出 checkpoint（或将解锁状态显式作为 resume 配置），并做一次“解锁后保存→新进程 resume”断言。若明确只允许单进程不间断训练，则必须删除“永久”的表述并在训练命令中禁止 resume。

### S2 / P1：当前课程的第一阶段已被训练日志证实卡住，目标 `0.20 m` 永远不会被采样

- 位置：`wheel_legged_gym/envs/chuanliantui_standup/chuanliantui_standup_config.py:16-25`；解锁调用见 `chuanliantui_standup.py:48-77,169-194`。
- 事实：解锁前要求策略在 **0.30 m** 命令、`base_height >= 0.28 m`、无 base_link 接触、姿态满足条件并连续 0.5 s 的完整成功率达到 30%；达成后才切到用户实际要的 **0.20 m**。
- 训练证据：最新 `logs/chuanliantui_standup/Sep21_23-37-17_standup_curriculum_explore_resume` 的最后记录为 iteration 3651：`recovered_rate=0`、`curriculum_unlocked=0`、`target_height=0.30`、最近窗口恢复率 0。高度和腿倾角项已有正反馈（分别约 0.167、0.163），但没有一次完整恢复。因此当前 run 尚未进入最终目标阶段。
- 实际影响：训练在更高、更严格的预解锁姿态上停滞；继续原样续训不能验证或改善 0.20 m 起立策略。
- 最小处理：将课程顺序改为“先易后难”。推荐先固定 `0.20 m`，以较弱姿态惩罚训练到 30% 恢复率，再仅收紧 orientation 到 -10；如确需训练 0.30 m，则应把它放在成功后的第二阶段，且用独立的渐进高度课程而非作为到 0.20 m 的前置条件。

## 建议处理

### S3 / P2：同一个 0.1 N 二值接触判据会让奖励与成功计时抖动

- 位置：`chuanliantui_standup.py:115-141,223-259`。
- 影响：`base_link` 力在 0.1 N 附近一帧切换，就会同时改变高度门控（1 ↔ 0.2）、接地/离地奖励（-0.003 ↔ +0.002 每策略步）及成功计时是否清零。高度项已经是软门控，但接触状态仍是硬二值；接触求解噪声会产生非物理的奖励跃迁。
- 建议：保留成功判据的严格性，但为奖励使用连续力门控（例如 0–5 N 线性/指数衰减），或引入 `on=1 N / off=0.1 N` 的迟滞和 2–5 个策略步去抖。成功计时可继续使用更严格阈值。

### S4 / P2：解锁瞬间同时改变高度和把姿态惩罚放大 10 倍，价值函数会遇到目标跳变

- 位置：`chuanliantui_standup_config.py:20-25`；`chuanliantui_standup.py:70-76`。
- 影响：解锁的同一策略步把命令高度从 0.30 m 改到 0.20 m，且 orientation 从 -1 变为 -10。奖励项逐项裁剪到 `±0.01`，因此后者会很快饱和为最大惩罚；已有 value target 会突变。
- 建议：分两个窗口切换，或在 0.5–1 s 内线性插值姿态权重和命令高度。若 S2 改为始终 0.20 m，则只剩姿态权重切换，风险会小很多。

### S5 / P3：终端标签“左右腿对称项”不完整，容易误判腿角奖励来源

- 位置：`chuanliantui_standup_config.py:69-71,115-116,131`；`chuanliantui_standup.py:269-272`。
- 事实：`nominal_state` 才是左右 `theta0` 差的对称惩罚；`leg_angle` 是两侧各自趋近 0 rad 的绝对腿倾角奖励。
- 建议：终端标签把 `rew_nominal_state` 改为“左右腿角差惩罚”，并保留现有“腿倾角奖励”。无需改变数值。

## 已确认合理的部分

- 高度奖励是 `exp(-error² / 0.01)`，在 10 cm 误差处仍有 `e^-1`；并非用户此前担心的“超过目标立即跳成负值”。
- `base_link_contact` 与 `base_link_airborne` 在首次轮触地后互斥；初次自由落体样本由 policy mask 排除，未被送入 PPO。
- `base_height=2.0`、`leg_angle=1.5`、`recovered=1.0` 等尺度均乘策略 dt=0.01，且每项再裁剪到 ±0.01；这避免单一奖励项压制其它项。注意提高超过该上限时只会扩大饱和区，不会继续提高单步最大回报。
- 双轮离地项与 0.2 s 终止仍互补：前者提供提前的每步负反馈，后者防止长期无支撑拖长回合。

## 复查证据与边界

- 执行：`/home/zn/miniforge3/envs/wheellegged_py38/bin/python scripts/agent/check_chuanliantui_base_link_airborne_reward.py`，结果 PASS；该检查覆盖软门控、互斥 base_link 项、双轮项和课程阈值的静态逻辑。
- 执行：`git diff --check`，结果通过。
- 使用 TensorBoard event 文件读取最新标量，未绘图、未启动训练或 TensorBoard。
- 未执行训练→Isaac 回放→MuJoCo 行为验收；上述 P1 是代码路径与现有训练指标均支持的课程/恢复语义问题，不将其误称为物理仿真故障。

## 审查后修复状态（2026-09-22）

- S1 已修复：runner checkpoint 现在可选保存环境状态；起立环境保存并恢复解锁标志、窗口计数和最近恢复率。恢复已解锁课程时同步 `orientation=-10` 与全部高度命令 `0.20 m`；旧 checkpoint 缺少环境状态时明确提示并使用配置初始课程。
- S3 已修复：奖励侧将 `base_link` 最大接触力在 0–5 N 归一化；高度门控从 1 连续降至 0.2，接地惩罚与离地奖励分别使用连续接触比例及其补值。成功判据仍使用独立、严格的 0.1 N 阈值。
- 修复验证：CPU 奖励检查通过；mock 的 runner `save/load` 往返验证 `env_state` 被写入并恢复；`py_compile` 与 `git diff --check` 通过。尚未进行训练→Isaac 回放→MuJoCo 行为验收。

---

# 串联腿起立、sim2sim 与 ONNX 候选复查（2026-09-22）

## 范围、版本与方法

- **自审**：当前环境没有为本候选创建 manifest；`~/.agent-projects/wheel-legged-gym/tasks/` 中仅有历史任务，未复用其 `tests.md` 或身份。未获得独立 reviewer，本报告不宣称独立复查。
- review commit：`e0e6a094e93e8175f641c8b9f225cc51a78bc849`；分支：`26_wheelleg`；候选为该 commit 上的全部未提交工作区改动。
- 本轮重点：25 维观测/500 Hz 时序、起立课程与 checkpoint、连续 base_link 奖励、腿倾角关闭、串联 MuJoCo 回放与 `model_6000_full.onnx` 导出。根 `logs/` 中已有训练权重仅只读使用，未删除、移动或覆盖。
- 结论：**没有发现 P0/P1 阻断问题；发现 1 项 P2、1 项 P3。** 当前 ONNX 文件结构与数值都通过验证，不能据此单独宣称已可部署到实机。

## 发现

### O1 / P2：ONNX 导出工具会静默覆盖已有产物

- 位置：`scripts/agent/export_chuanliantui_onnx.py:47-83,101`。
- 触发：对一个已经存在的 `--output` 路径再次运行导出命令。
- 实际影响：`torch.onnx.export` 与随后的 `onnx.save` 会直接覆盖旧 `.onnx`，没有 `--force`、哈希比对或备份。导出文件是人工验收后的部署候选，误用相同路径会丢失可比较版本。
- 最小修复：默认在 `output.exists()` 时失败；新增显式 `--force` 才允许覆盖，并在输出中打印 checkpoint SHA-256。

### O2 / P3：腿倾角奖励关闭后仍保留不可达的计算与配置参数

- 位置：`wheel_legged_gym/envs/chuanliantui_standup/chuanliantui_standup_config.py:59-60,72-73`；`wheel_legged_gym/envs/chuanliantui_standup/chuanliantui_standup.py:307-310`。
- 触发：当前 `leg_angle = 0.0` 配置下训练。
- 实际影响：奖励装配会删除零 scale，所以 `_reward_leg_angle()` 和 `leg_angle_reward_sigma` 不会被调用；行为不受影响，但留下的“奖励”注释与参数容易让后续调参者误以为其在生效。
- 最小修复：删除该函数及 sigma，或把它们移入明确的“可选恢复项”配置并注明默认禁用。

## 已检查且通过

| 项目 | 证据 |
|---|---|
| 观测/网络契约 | `check_chuanliantui_observation.py` 通过：actor 25、历史 125、critic 65，MuJoCo 构造一致。 |
| 500 Hz 串联训练代理 | `check_chuanliantui_train_proxy.py` 通过：`timestep=0.002`、6 DOF、6 motor、`neq=0`。 |
| 课程 checkpoint | 代码检查 `OnPolicyRunner.save/load` 调用环境可选 hook；CPU 检查覆盖锁定/解锁窗口、保存/恢复高度命令与 orientation 权重。旧 checkpoint 缺状态时有显式提示。 |
| 连续 base_link 奖励 | CPU 检查覆盖 0、2.5、5 N 的连续比例、高度门控、严格 0.1 N 成功阈值及 dt/单项裁剪路径。 |
| 腿倾角关闭 | CPU 检查确认 `leg_angle=0` 后该奖励名不进入装配列表，终端键也不再包含它。 |
| ONNX | `model_6000_full.onnx` 经 ONNX checker 通过；输入 `observations=[batch,25]`、`observation_history=[batch,125]`，输出 `actions=[batch,6]`、`latent=[batch,3]`。导出脚本用 ONNX ReferenceEvaluator 与 PyTorch 随机输入逐元素比对通过。 |
| 闭链候选结构 | `check_chuanliantui_closed.py` 通过：4 个 connect、8 个 actuator、0/150 N 气弹簧和 ±360° 闭合；5 秒无控制最大销轴误差 `4.591e-4 m`。该检查不等同于新策略的闭链行为验收。 |
| 静态质量 | 指定 Python 3.8 环境下 `py_compile` 通过，`git diff --check` 通过。 |

## 训练与运行证据、限制

- `Sep22_16-26-11_standup_no_legangle_resume/model_6000.pt` 的最后训练指标：`recovered_rate=1`、课程已解锁、目标高度 0.20 m、平均回合长度约 2015 策略步；这些说明 Isaac 训练内部成功判据稳定达成，不替代肉眼行为验收。
- 已执行该 checkpoint 的 `mj_sim2sim_ct.py --selfcheck`，25/125/6 网络形状通过；也曾在串联训练代理中以 `--standup --render --cmd_vx 0 --cmd_height 0.20 --friction 0.75` 观察到首次接地后高度回到约 0.202 m、姿态稳定的前 6 秒。该窗口未构成完整长时回放记录。
- **未执行当前 model_6000 的 Isaac `play.py` 人工回放，也未执行新策略的真实闭链 `--closed_chain` 长时行为验收、行走验收或实机验证。** 因此不应把本报告或 ONNX 导出视为跨引擎/硬件部署通过。
- 没有附带 Git 提交信息或 PR 正文，故无提交/PR 文案可审；本轮没有提交、推送、创建 PR 或修改 `logs/` 历史内容。
