# RMUC2027 完整场地与轮腿策略交互

2026-11-15 的原版基线取消了地形简化，恢复上游原始完整场地，共57187几何体，地形、视觉网格与碰撞资产均不简化；启动不再自动生成简化场地。保留仅同步显示状态、按真实时间刷新及关闭阴影/反射的画面优化，方向键仍为按下最大、松开归零。完整场地的物理计算较慢，既有有限诊断约0.43倍实时。

恢复后本机窗口有限检查完成100策略步/1秒仿真，原版路径及57187几何体已核对，无数值错误、进程退出0；报告`artifacts/rmuc2027/original-restored-20261115-01/native-check/report.json`。随后重开原版交互窗口。该检查只确认加载、短时运行和退出，未重新遍历整场或自动操作方向键。

历史：用户曾因卡顿允许降低精度，采用13573几何体的全场5cm体素版，纯物理约1.46倍实时。该版把连续缓坡变成了隐形5cm台阶，已撤回；不能再作为默认场地。有限性能数据不代表完整行为验收。

用户于2026-11-15选择完整场地、原生MuJoCo运行与操控。源码为[CosmosMount/rmuc2027_mjarena](https://github.com/CosmosMount/rmuc2027_mjarena)，固定提交`ffd1eb7d1d1a2d1654125d8f316fdb0fdac6d001`，按原MIT许可保留来源与许可证。下载位于被Git忽略的`artifacts/rmuc2027/upstream-<提交>/`，没有复制95MB资产进入版本控制。

当前默认赛场预载四个模型，场中央的红蓝能量机关、两侧固定支架、双方机库平台和下方科技核心已从画面和碰撞中移除。两个支架安装孔沿原孔口补平，可视面与周围台面同高（0.599 m），碰撞面与周围承重面同高（0.6015 m）。见[当前按键和出生点](#rmuc比赛场地)。

## 启动与操作

### 分区地形交互

2026-11-16用户要求在MuJoCo sim2sim中加入当前训练地形。显式选用`regions`时，直接复用训练的`RegionalTerrainCfg.function`，读取checkpoint的地形参数和种子，将生成器坐标平移到训练世界中心；不加载RMUC比赛场地。32×32m左半平地、右上5/10/15/20°四座金字塔坡、右下±1cm崎岖；全部滑动摩擦固定0.8。几何形状、接缝和粗糙高度不作视觉简化。颜色、光照及隐藏机器人碰撞壳只改变显示。

```bash
scripts/rmuc2027.sh --terrain regions
# 默认前后3m/s、普通转向2rad/s；G键切换原地10rad/s小陀螺。
# 从平地直接开启小陀螺：
scripts/rmuc2027.sh --terrain regions --spawn -4 4 --spin
```

当前该模式默认checkpoint、SHA和行为状态见[项目实验记录](../ai/project-log.md)顶部。前后指令上限±3m/s、yaw上限±10rad/s，`|yaw|>2`时vx清零；默认零速/.26m，在10°坡脚`(8.25,4)`面向上坡。按住↑/↓前后、←/→转向，松开归零；E/Q高度±.01m（.22～.30）、X停车、P暂停、R回当前出生点；数字1平地、2/3/4/5分别到5/10/15/20°坡脚、6崎岖，V切换全图和跟随机器人，鼠标可缩放/旋转。按数字复位后yaw归零，沿世界+X朝坡顶。

2026-11-16用户确认“速度”指运动命令，不改播放时间倍率：区域模式方向键前后默认从±1改为±3m/s，普通←/→仍±2rad/s。G切换持续小陀螺模式，固定目标vx=0/yaw=+10rad/s，方向键不能覆盖；再按G退出并清零，X停车、R复位或数字切区也退出。P只暂停物理，不改变所选模式。窗口左上显示DRIVE/SPIN、目标vx/yaw和实际机身yaw角速度。`--spin`只接受区域模式，不与脚本命令序列混用；浏览器界面提供相同小陀螺复选框。物理和控制周期保持原值，播放仍按正常时间运行，目标10不表示所有地形实际都能跟到10。

此体验采用标称质量/PD、无观测噪声，关闭持续外力和短推扰，分别保留训练记录与体验条件，不改写PT；原双10～20ms延迟保留，seed1此次实际动作18ms/传感13ms。Actor仍25维当前帧/125维历史→6动作，由原encoder估速；77点地形只用于训练Critic，推理不读取真值速度或高度扫描。物理/PD/策略1/5/10ms、共享XML/机械轮/气弹簧和限矩保持。没有奖励或自动失败复位逻辑，真实碰地仍发生；窗口不是正式行为验收。

检查入口：

```bash
MUJOCO_GL=egl PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  -m scripts.agent.fudan_ppo_walk.check_regional_native --output artifacts/<任务>/<新实验>/regions
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  -m scripts.agent.fudan_ppo_walk.check_arena_native_interface --terrain regions \
  --checkpoint <区域模型PT> --output artifacts/<任务>/<新实验>/interface
```

`check_regional_native`核对16坡面角、世界坐标、实际崎岖节点、接触摩擦、训练归一化和六区域各3秒停/走/停有限性；接口检查对100组同状态比较实际GPU实现的观测、策略输出、延迟、PD保持和气弹簧。均不执行优化器、不代替停车/通行验收。原生窗口短时运行及正常退出另测；方向键/区域切换由交互入口提供，未以自动按键测试声称通过。

### 独立多高度台阶场

`steps`模式预载行走和台阶两个模型，直接复用训练的`PlatformCfg`盒体生成器，在同一场景摆放九个独立单级平台；它们不是连续楼梯。按用户要求，默认以最新完整行走6838模型运行，按T进入台阶3200模型，再按T回行走。模型和行为状态见[项目实验记录](../ai/project-log.md)；`--checkpoint <台阶PT>`和`--walk-checkpoint <行走PT>`分别选择两侧模型。完整赛场的默认组合见下方 RMUC 比赛场地。

```bash
scripts/rmuc2027.sh --terrain steps
# 浏览器界面含台阶高度下拉框：
scripts/rmuc2027.sh --terrain steps --viewer viser --port 8792
# 有限命令回放直接使用台阶模型时，显式加--step-mode：
scripts/rmuc2027.sh --terrain steps --step-mode --viewer none --seconds 3 --command 0 0 .35
```

| 数字键 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 台阶毫米 | 平地 | 20 | 50 | 80 | 120 | 160 | 180 | 200 | 210 | 220 |

默认出生在200mm平台前1.2m，朝世界+X，先用行走模型/机身指令0.26m。行走恢复原方向键指令：前后±3m/s、普通转向±2rad/s，G切换持续vx=0/yaw=+10rad/s小陀螺；再按G退出，X/复位/数字键也退出。高转速`|yaw|>2`时vx清零，与原坡道交互一致。

按T切台阶模型时设0.35m，退出小陀螺，方向键改为前后±1m/s/转向±0.5rad/s；再按T回行走，恢复原高度及3m/s/2rad/s方向键幅度。X只停车、不切模。长按T只触发一次，松开再按才再次切换。浏览器提供切模和小陀螺复选框；小陀螺只在行走模式可选。切模不复位物理、PD、上一动作、延迟队列或五帧历史，按正常策略拍切换Actor；只接受控制/时序/物理与历史接口相同的两份PT。

数字键传送到对应平台前并清速度/动作历史，保持当前模型模式；0回平地。平台宽1.148m、深2.09m、垂直沿；200/210/220mm保留训练侧障碍，碰撞摩擦0.8。方向键松开归零，E/Q调机身指令高度（行走0.22～0.30m，台阶0.22～0.35m），R回当前平台出生点，P暂停，V全图/机器人视角。平台高度与机身指令高度是两个不同量。`--keyboard-speed`/`--keyboard-yaw`可覆盖方向键幅度，切模后按当前模型范围限幅；浏览器滑条范围分别为行走vx±3/yaw±10、台阶vx±2.5/yaw±2。速度指令不改变播放时间倍率，目标值不保证实际跟踪达标。

用 `--checkpoint logs/mjlab/fudan_ppo_step/20261118_fixed_speed2_L40_resume4847_to5200_01/model_final.pt` 加载固定 2 m/s 的 5200。窗口仍从 WALK 6838 开始，按 T 切 STEP 后上键发 2 m/s，切回 WALK 后上键恢复 3 m/s。该存档只含 200 毫米平台，按 1 回台阶前，按 0 回平地。

6200 起步与 5200 停车组合使用 `--checkpoint <6200 PT> --parking-checkpoint <固定速度5200 PT>`。开窗仍是 WALK 6838。按 T 后，↑ 发正向 2 m/s、零转向，使用 6200。已开始行驶后，松开 ↑ 或按 X 发零速，由 5200 停车。再次按 ↑ 回到 6200。切换共用观测历史和延迟队列。R 或数字键复位后清除上次停车状态，再次进入台阶模式也从 6200 待命。屏幕的 Actor 一栏显示当前实际使用的模型。[本次启动与验证](../../artifacts/fudan_ppo_step/20261119-native-6200-5200-parking-01/README.md)。

物理/PD/策略周期仍读取PT的1/5/10ms，延迟队列和Actor输入保持；行走模型交互条件与坡道场一样关闭训练外力和推扰，原训练条件另行保留，双延迟不变。没有新训练或自动失败复位。原生场景运行只用于交互体验，不能替代MuJoCo Warp GPU行为验收。[首次场景及控制检查](../../artifacts/fudan_ppo_step/20261117-native-steps-01/README.md)、[双模型切换检查](../../artifacts/fudan_ppo_step/20261117-native-policy-switch-01/README.md)、[最新6838切换检查](../../artifacts/fudan_ppo_step/20261117-native-latest-walk-01/README.md)。

### RMUC比赛场地

从仓库根目录启动。首次缺少场地资产时先运行 `scripts/setup_rmuc2027.sh`。

```bash
# 默认出生点 (-5, -3)，行走 6838
scripts/rmuc2027.sh

# 红方实测台阶前约 1.2 米，车头朝向台阶
scripts/rmuc2027.sh --spawn -5.76 -4.20 --spawn-yaw-deg -90

# 浏览器用同一套切换和落地逻辑
scripts/rmuc2027.sh --viewer viser --port 8790
```

| 操作 | 模型与指令 |
|---|---|
| 方向键、G | WALK 6838。前后 ±3 m/s，普通转向 ±2 rad/s；G 开关 10 rad/s 原地小陀螺。 |
| T、↑ | T 切换 WALK / STEP。台阶模式的机身指令为 0.35 m；按 ↑ 由 STEP 6200 发正向 2 m/s、零转向。 |
| 松开 ↑、X | 已开始台阶行驶时由 PARKING 5200 停车，再按 ↑ 回到 6200。行走模式的 X 停车并退出小陀螺。 |
| 空格 | 在行走模式跳一次，退出小陀螺。JUMP 5200 的指令限于 ±2.1 m/s、±2 rad/s。起跳确认后首次轮载超过 1 N，再缓冲 80 ms 回 WALK；未确认落地时最多运行 1.4 s。事件记录区分落地、超时和手动复位。 |
| R、P、E/Q | R 回当前出生点并恢复 WALK、零速度和 0.26 m 高度指令，P 暂停/继续，E/Q 调高度。 |

桌面交互遇到 MuJoCo 数值警告时会停住画面，按 R 回当前出生点。终端日志记录停住时的模型、指令和上一拍状态，便于回查。

桌面和浏览器均显示当前出动作的模型。台阶和行走使用同一套带延迟的传感和动作，Jump 读即时传感并直通动作。两套五帧历史跟随同一台机器人，前进指令分别乘 2 和 3，物理、底层电机控制和策略周期分别为 1、5、10 ms。

指定 `--checkpoint <PT>` 可在当前赛场只运行一个兼容模型。历史行走模型为 `logs/mjlab/fudan_ppo_walk/20261114_pd200_p20d1_half_random_delay_wide_resume1000_to3000_01/model_final.pt`。要复现含能量机关的上游原图，可显式指定 XML：

```bash
scripts/rmuc2027.sh --arena artifacts/rmuc2027/upstream-ffd1eb7d1d1a2d1654125d8f316fdb0fdac6d001/rmuc2027_mjarena/arena_mjcf/model.xml
```

赛场外沿设有四面半透明碰撞围栏，内侧位于 x = ±13.98 m 和 y = ±7.98 m，墙顶世界高度为 1.70 m。四角相接，车轮和机身均可与围栏接触。

当前赛场首次启动会在 `artifacts/rmuc2027/fields/` 生成去机关和合并平面后的几何文件，再编译 MuJoCo 缓存；以后复用。红方登台、接近及停车区域使用原碰撞件。水平碰撞块只在厚度和接触参数相同且共边时合并，合并边界误差限于 0.05 mm。连续斜坡保留原网格。完整碰撞面用于出生点地面探测。其他场景仍使用各自的训练地形。

有限时长运行并保存轨迹：

```bash
scripts/rmuc2027.sh --viewer none --seconds 5 --command 0 0 .26 \
  --output artifacts/rmuc2027/<新实验>/stand
```

四策略无窗口回放的分段命令可带 `"key":"SPACE"` 等单次按键；报告写入每段开始的策略拍，`policy-roles.npy` 逐拍记录模型，`contact-substeps.npy` 逐毫秒记录左右轮竖向力、最大非轮接触力和机身竖向速度。

`--seconds`是仿真时间。`--output`须为新目录，保存`trace.npz`和`report.json`；交互退出可关闭本机窗口或在终端Ctrl+C。每次输出独立保存，不覆盖历史模型或日志。

## 接入与边界

入口为`scripts/rmuc2027.sh → sim2sim/rmuc2027_arena.py`。机器人仍加载共享XML及机械轮JSON；场地以`arena/`前缀附加，默认移除上述中央装置，其余场地使用上游原精度碰撞与可视几何。机器人的求解器及积分参数沿用mjlab机器人配置，控制参数及周期从PT读取，场地不接管控制时序。默认PT为1ms积分、5ms PD、10ms策略；动作与传感延迟按其保存范围抽样，并写入运行报告。PD两次更新之间保持限幅后力矩，气弹簧和延迟队列仍逐物理步更新。

策略复用`load_policy`，Actor25维、五帧历史125维、六动作和v2/v3复位接口保持。交互采用名义物理、零初速度、无观测噪声；保留PT保存的延迟，不随机化质量/惯量/电机参数。不是训练随机化分布的复现。不支持旧速度曲线、额外轮端响应或持续外载模型，避免静默错误套用。

当前安装的MuJoCo为3.11；上游Python包声明3.15及其浏览器依赖。本次只复用场地MJCF资产和现有查看器，不安装该包或升级训练环境。当前MuJoCo Warp加载器先生成全部几何配对，完整约5.7万个碰撞块不适合直接送入本机GPU；因此按用户选择使用原生CPU物理作交互场地体验。

本入口不注册训练任务，不修改奖励、训练默认配置、机器人XML、H7或现有GPU验收。平地策略能加载不代表能走完坡道、台阶或整场；正式行为验收仍按统一GPU标准，不能把本入口的有限运行写成验收通过或真机通过。

## 正式接口检查

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  -m scripts.agent.fudan_ppo_walk.check_arena_native_interface \
  --output artifacts/rmuc2027/<新实验>/interface-check
```

以100组同位置、速度、姿态、饱和动作、指令及延迟，对照实际GPU环境的25维观测、策略输出、PD保持节奏、六路力矩与气弹簧输出。物理积分0、学习更新0；不检查两引擎轨迹等价或全场通行。2026-11-15检查通过，最大观测误差1.79e-7、策略误差3.82e-6、力矩误差5.37e-7Nm、气弹簧误差8.05e-6N；报告`artifacts/rmuc2027/integration-20261115-01/interface-check-03/report.json`。第一次检查因两端延迟队列初态未对齐而失败，已保留并修正检查初态，没有放宽误差阈值。

完整场地实际运行：编译后共57187几何体，含场地原56898个碰撞块、原视觉资产与机器人。默认出生点`(-5,-3)`处碰撞地面z约-0.077057m。零指令5秒和`vx=.3/yaw=.2/h=.26`5秒均完成500策略步、状态有限、无MuJoCo数值警告；前者水平净移0.025m、roll/pitch峰1.62°/2.58°，后者净移0.892m、峰2.16°/3.28°。零指令物理运行约0.43倍实时，不承诺实时速度或全场通行。证据分别在同目录`stand-03/`、`drive-01/`；首次射线接口缺参和透明碰撞壳导致的失败保留，已修正为MuJoCo3.11签名并仅在射线查询期间临时显露场地碰撞壳，恢复原显示属性。

浏览器场地/机器人已目视检查，前进滑条、停车、暂停及出生点复位已实际操作。随后用户要求打开可操控窗口，已关闭本次浏览器仿真进程并以`DISPLAY=:0 scripts/rmuc2027.sh --viewer native`打开独立MuJoCo键盘窗口，避免同时运行两份场地物理。窗口启动后控制提示与场地参数已打印，进程保持运行；未通过自动化向本机窗口发送按键，键盘行为由用户现场操作确认。桌面环境的DISPLAY取本机实际配置，此值不写入公共启动脚本。

## 连续路面修复与验证

本节为历史诊断记录；用户已取消全部场地简化，`continuous50-v8`也不再作为默认或自动生成。下述数据只对应当时的候选场地。

旧体素版在`x=-5,y=-4→-3.5`把原版约13mm连续上升变成50mm跳变。修复版先按原版几何和实际双轮轮距约0.4265m识别宽支撑面，并检查相邻小三角片是否共同提供轮距范围的连续支撑及0.45m净空；保留700个支撑网格。路面附近方块清除后，根据每个细节网格实际贡献的占据格，恢复16435个相关原始网格，避免误删真实障碍物或让方块继续顶住路面。其余区域保留2472个合并体素盒；原2929个盒体、封口、波纹高度场及外观保持。

必须原样复制原始碰撞资产和几何参数。曾将原网格世界坐标重新生成凸包，虽然高度检查通过，但接触响应变化、穿越轨迹退化，已拒绝。最终直接保留原XML资产、原始顶点顺序、局部坐标、位姿和接触属性；不修改机器人、策略、PD或时序。

正式复核命令（输出使用新目录）：

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  -m scripts.agent.fudan_ppo_walk.check_arena_floor --output <新路面检查目录>
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  -m scripts.agent.fudan_ppo_walk.check_arena_floor_crossing --output <新穿越检查目录>
```

`artifacts/rmuc2027/continuous-floor-20261115-01/floor-check-08`：两条路线按5mm间隔检查1142点，原版和修复版高度逐点相同；原5cm人工跳变消失。原版独立筛出的1259处可行驶轮距区域、6295个支撑采样点无缺失，与原版高度相同。`geometry-check`还核对机器人质量/惯量/质心、关节、闭链与执行器参数相同；1953条全场射线无整条原命中丢失，远处简化细节仍有少量擦边差异，不作为全场误差上限。

`crossing-08`：原版、旧方块版、修复版各7秒同初态/同延迟诊断，前2秒零指令、后5秒vx=0.5。轮地法向的水平分量比例>0.2且法向力>1N记为侧向冲击，旧版1320物理步/峰702.57N，原版和修复版均3步/峰29.52N；修复版700帧qpos与原版逐字相同。急加速仍有原策略同样的约23.8°瞬态俯仰，不声称该诊断满足运动门槛。`default-drive`复测默认出生点vx=.3/yaw=.2的5秒轨迹，与原版存档逐字相同，roll/pitch峰2.156/3.278°。

`performance`纯物理2秒耗2.377秒；`native-check`窗口5秒耗7.611秒、251次状态更新、无数值警告、进程退出0。原生键盘窗口已重开，按住方向键最大指令、松开归零保持；训练仍停止，学习更新0。均为有限场地诊断，尚未动态遍历整场，不替代GPU行为或真机验收。

## 历史：全场体素版的性能与限制（已撤回）

`prepare_arena_voxel`按原凸碰撞面的长边/高度采样（间隔约16.7mm），将命中的5cm格子占据，再沿X/Y/Z合并连续格子。53833个复杂网格碰撞替换为10219个合并盒体，保留原2929个盒体及其余封口/高度场。相邻体素盒半尺寸各外扩0.2mm，避免浮点接缝漏检。原视觉网格不简化；这会把斜面和小曲面变成阶梯状碰撞近似，不能用于真实地形精度或正式训练验收。

显示改用`handle.sync(state_only=True)`，不反复同步完整模型；依据[MuJoCo Python被动查看器接口](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)。实测20次同步，中位47.17→3.39ms（只代表同步调用）。显示按真实时间刷新，目标上限60Hz，并关闭阴影/反射；策略/PD/积分节奏不改。原场地四线程诊断未见明显收益，没有启用线程池。最初同步基准在退出窗口时出现GLX错误，保留失败日志，改为关闭后等待渲染线程退出，第二次完成。

正式检查入口，从仓库根目录运行，所有输出须为新目录：

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  -m scripts.agent.fudan_ppo_walk.prepare_arena_voxel --output <新场地目录>
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  -m scripts.agent.fudan_ppo_walk.check_arena_coarse \
  --arena artifacts/rmuc2027/continuous50-v8/arena.xml --output <新检查目录>
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python \
  -m scripts.agent.fudan_ppo_walk.check_arena_performance --output <新性能目录>
```

1953条垂直/水平射线中，无原有命中变成整条无命中，新增命中20；垂直1705条中位/95%首命中位置差约29.6/56.7mm。少量薄边、擦边射线命中不同表面，垂直最大差2.70m，全部方向最大24.26m；不能把5cm网格解释成全场5cm误差上限，也不据这些射线宣称全场可通行。原版与简化版机器人nq/nv/nu/闭链数、关节限位、惯量/阻尼/摩擦、执行器增益/限幅、机器人质量/惯量/质心逐项相等。

新场地200步/2秒名义物理诊断耗时1.373秒，约1.46倍实时；500步/5秒前进转弯耗时3.729秒，无警告、状态有限，净移0.843m、roll/pitch峰0.235°/3.457°。这些是有限性能/运行检查，不是完整行为通过。结果在`artifacts/rmuc2027/performance-20261115-01/`，关键目录为`coarse-check-02`、`voxel50-final-performance`、`coarse-drive`、`coarse-native-02`。第一次有限原生窗口在完成500步并写报告后，退出时渲染线程释放资源发生段错误；增加关闭后的250ms线程退出等待，第二次完整500步、报告和进程退出0均完成。状态同步约每秒44次（不是GPU实际绘制帧率），接近实时；最终交互窗口重新启动。平面合并的两次候选只减少约13%/19%，25mm体素候选仍有约4万碰撞体，未采用；只保留其准备摘要和必要日志。

## 用户要求停止训练

接入期间用户要求先停止后台训练，已向本次全幅随机化续训进程发送SIGINT，训练与协调进程退出，不自动恢复。保存到完整更新3018轮、双优化器各60360步；中断时另有184320条未完成采样，不能把其`partial_update=false`误认为完整采样边界。当时的场地入口使用原3000模型；完整模型和日志均保留。停止记录见`artifacts/fudan_ppo_walk/20261115-pd200-full-random-force3-push-resume3000-4000-01/user-stop-arena-request.json`。
