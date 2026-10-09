# 串联腿 mjlab 迁移

## 入口与范围

`mjlab_training/` 是独立 Python 3.12 包；`uv.lock` 锁定 mjlab 1.6.0、MuJoCo / MuJoCo Warp、Torch 和 RSL-RL。不要在 Isaac Gym Python 3.8 环境安装这些依赖。本机复用已有 `/home/zn/文档/mjlab/wheelleg_RL/.venv` 的已安装依赖，通过本仓 `.venv` 的路径引用加载，不修改那个仓库和环境；ONNX Runtime 1.26.0仅安装到本仓 `.venv`。`reused-runtime.json`记录实际版本。该外部仓库的已有任务／资产／弹簧曲线未纳入本次更改。

新任务：`ct-mjlab ... --task chuanliantui`。环境是基于 **mjlab Scene / Simulation / Entity / ContactSensor** 的 direct VecEnv，使用 MuJoCo Warp 批量物理与官方 RSL-RL 普通 PPO。direct 实现保留参考任务的动作→物理子步→终止→奖励→重置→观测顺序。使用本包入口，不通过 mjlab 的 ManagerBasedRlEnv 通用任务入口创建环境。

参考：SCUTRobotLab `wheeled-legged_RL`，提交 `868467c3427fc00d34ebb1176683652d9597ba1c`，`Robotics-Wheelbipe-V14-Flat-v0` / `WheelbipeV14FlatEnvCfg` / `WheelbipeV14FlatPPORunnerCfg`。源文件 SHA256、继承后的常量与完整奖励表在 `scut_v14.json`；MIT 许可保存在 `LICENSE.SCUT`。未采用 v1/v2、DreamWaq、HIM、NP3O、云台自转、课程或旧起立分阶段奖励。

用户指定的任务覆盖：后摆落地初姿、根高度 0.15 m、静站高度命令 0.22 m、三个速度命令恒零，关闭行走／跳跃命令覆盖。初始根速度为零。独立工作区仍在 `26_wheelleg`，本次未开始训练，未改板端固件，未更改历史 logs / exports。

## 机器人与命名

+x 前进，+y 左侧，+z 向上。原 CAD 的 `rf*` 位于 +y，`lf*` 位于 −y。本次交换 URDF/MJCF 的实体名称及父子、销轴、执行器引用，使 **lf=实际左，rf=实际右**。坐标、轴、惯量、限位、网格文件内容保持原物理含义。

网格文件名保留原 CAD 名称。因此新 `lf0` 引用旧 `rf0.STL`；不能从 body 名字拼 mesh 文件名。生成器从 URDF 的 visual mesh 引用读取。两个 URDF、闭链 XML、串联代理 XML 和旧环境／回放引用已同步。

旧 Isaac 与 H7 虚拟接口仍按原物理通道排序，其当前名称变为 `[rf0, rf1, rfwheel, lf0, lf1, lfwheel]`，旧闭链电机顺序为 `[rf0, rf00, rfwheel, lf0, lf00, lfwheel]`。历史交付 JSON 中的原名称需要按此更名映射解释，不能当作新接口字段使用。

新策略完全使用实体电机：

`[lf0, lf00, rf0, rf00, lfwheel, rfwheel]`

四个腿电机位置 PD：`target = nominal + 0.5 * action`，目标限幅 ±π，Kp=10、Kd=1，名义限矩 40 Nm。轮电机速度控制：`target = clip(10 * action, ±20 rad/s)`，Kv=0.1，名义限矩 3.9 Nm。动作本身无 ±1 全局限幅。普通 PPO 请求动作参与观测和动作差分奖励；控制目标在缩放后限幅并延迟。

`poses.json` 保存全部实体主动／被动关节的闭合初值和名义站姿。训练只读缓存，不运行虚拟关节反算，也不独立随机化被动关节。资产改变必须重新生成缓存。

气弹簧每侧恒定伸张力 150 N。两端从 URDF CAD 标记及局部旋转变换导出为 MJCF sites；原生 spatial tendon 根据当前世界端点计算轴向施力与姿态相关广义力。无腿电机前馈补偿，弹簧不计入电机力矩／功率奖励，也不占六维动作。旧 Isaac 的任意轴向几何补偿仍保留，单独由数值检查覆盖。

## 华南虎训练逻辑映射

- 物理 0.005 s，decimation=4，策略 **50 Hz**；20 s episode，timeout 为 episode step ≥999；timeout 与失败分开，RSL-RL 保留超时 bootstrap。
- actor / critic：256→128→64、ELU；普通高斯策略，初始每通道 std=1；关闭 empirical normalization。
- PPO：24 steps、20000 iterations、500 轮保存；5 epochs、4 minibatches、lr=1e-4、adaptive KL=.01、clip=.2、entropy=.005、value coefficient=4、gamma=.99、lambda=.95、max grad norm=1。
- 使用源 V14 **全部非零奖励**；每项仅乘一次策略 dt，无自定义奖励分档、总奖励截断或额外站稳奖励。`check_chuanliantui_mjlab.py` 对原始 `_get_rewards` AST 做数值对照。根／轮速度按源接口使用 COM 速度，位置使用 link 原点；轮相对速度在机体系投影，y 槽归零。关节加速度使用最后物理子步的速度差分，参照 [IsaacLab ArticulationData](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/assets/articulation/articulation_data.html)。
- 观测与动作延迟均按 **5 ms 物理子步** 更新。源配置采用 `torch.randint(low, high)`，上界排除：观测采样 `{1,2,3}`，动作采样 `{1,2}`，每次 reset 重采样；启动历史不足时读取最早可用帧。
- actor 的角速度／重力／位置／腿速度／轮速度噪声分别为 ±.25、±.05、±.025、±.5、±1。critic 使用干净状态。轮位置先置零再加源位置噪声，因此训练 actor 轮位置槽可能有 ±.025 噪声；无噪声回放为零。
- source ctrl mode 槽是 `[normal, stair, slope, recover, jump, height_target, state_time]`；本任务状态机关闭，值为 `[1,0,0,0,0,0,0]`。不加入预热完成标志。
- 普通失败：base 接触力 >1 N 或 roll/pitch >40°，连续20个策略周期才终止；NaN/Inf、实体关节速度 >500、根角速度 >200、根线速度 >100 立即终止。10步后 raw 观测 >120 或非有限值也立即终止；坏状态奖励归零。

### 域随机化与后端差别

源事件覆盖：base mass ×[.9,1.3]、其余实体腿和轮 mass ×[.9,1.1]；质量随机化同时同比缩放惯量；base COM x±.04 / y,z±.02；base 与轮材质64桶；主动腿／轮／被动腿关节摩擦与粘滞；reset 时 PD ×[.75,1.25]、腿输出力矩 ×[.8,1.1]、轮 ×[.9,1.1]（先缩放再按固定40 / 3.9 Nm限矩裁剪），采样间隔至少720策略步；每5–10 s 重新设置根 xy 速度 ±.25，并独立采样 base 三轴力 ±10 N、三轴力矩 ±1 Nm，外力在机体系保持到下一次采样或 reset，每个物理子步按当前姿态转换到世界系后提交。

质量和惯量的处理参照 [IsaacLab mass event](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.2/source/isaaclab/isaaclab/envs/mdp/events.py) 的 `recompute_inertia=True` 默认行为。

机器人适配及明确差别：

1. 缺少云台、滑动弹簧 joint、guide body，对应事件／实体省略；原机器人的400–600 N弹簧参数改为本机器人150 N原生拉索。
2. PhysX 静／动摩擦和恢复系数不能逐项等同于 MuJoCo。保留同样64桶参数及 critic材质字段，滑动接触使用动态摩擦，地面摩擦=1，机器人接触 priority=1以选择机器人摩擦（避免默认max吞掉随机化）；源地面恢复系数为0、使用multiply合成，因此保留采样e的critic字段，而地面接触采用临界阻尼近似零恢复；关节摩擦对应 frictionloss / damping。这是后端对应方式，不能声称 PhysX/MuJoCo 行为完全等价。
3. 原源文件 reset 重采样 delay lag，却未清自定义 delay历史。本迁移按 episode 清理所选环境的延迟历史，避免把上一回合姿态喂入新回合；动作差分历史、失败计数同样清理。此项修复单独有检查，不宣称与源残留行为一致。
4. 用户指定的后摆起立姿覆盖源随机根位置／姿态／速度和混合预设姿态；不随机化闭链被动关节。默认32环境与源一致，可通过CLI增加。
5. 使用 mjlab 固定依赖中的 RSL-RL 普通 PPO；旧 IsaacLab actor-critic类与新 RSL-RL actor/critic模型类的 checkpoint格式不同。参数一致不意味着随机训练轨迹相同。

## 观测契约

actor float32 `[N,35]`：

| 范围 | 字段 | 处理 |
|---|---|---|
| 0:3 | vx、vy、wz命令 | 恒零 |
| 3:4 | 高度命令 | .22×5 |
| 4:7 | 根角速度（机体系） | 限幅±100，×.5 |
| 7:10 | 投影重力 | 限幅±100 |
| 10:16 | 实体关节相对名义角，含轮槽 | 轮位置先置零；源噪声见上文 |
| 16:22 | 同顺序关节速度 | 限幅±200，×.1 |
| 22:28 | 当前策略请求动作 | 限幅±100；下一次决策看到上一拍动作 |
| 28:35 | 源 ctrl mode | `[1,0,0,0,0,0,0]` |

critic float32 `[N,78]`：干净core28 + 根机体系线速度3 + 绝对根高度1（×5）+ ctrl mode7 + extra39。extra39依次为 Kp6、Kd6、电机力矩6（×.05）、四路观测delay4、两路动作delay2、两轮相对机体系速度6、两轮历史接触状态2（>5 N）、base mass scale1、两轮材质6（static/dynamic/restitution）。没有补零槽。

接口 ID：`chuanliantui-mjlab-huananhu-v14-flat-standup-r1`。保存机器人资产、源配置、实体姿态缓存的 SHA256、真实控制参数和接口维度；加载校验不通过即拒绝。旧25/29维模型、旧虚拟关节 ONNX 和 H7 直接输入均不兼容。

## 可执行命令（本次未启动训练）

以下命令均在仓库根执行。本机已设置复用环境，直接使用 `scripts/mjlab.sh`；全新机器可通过 `uv sync`安装完整锁定环境。复用路径写在被忽略的 `.venv` 内，不进入Git。

```bash
# 本机现有mjlab环境的轻量接入（已执行）：只修改本仓目录。
scripts/setup_mjlab.sh /home/zn/文档/mjlab/wheelleg_RL/.venv/bin/python
# 全新机器才需要安装全套依赖：
# uv sync --project mjlab_training --locked --python 3.12

# 初始根高度改变后的可视预检：20环境，无策略学习；观察窗口中的初始机构和落地姿态。
scripts/mjlab.sh play --num-envs 20 --render --seconds 10

# 用户批准开始训练后执行；这条命令使用GPU批量仿真，无渲染。
scripts/mjlab.sh train --num-envs 1024 --iterations 20000
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/chuanliantui --port 8080

# 本机4060 8GB：1024环境是否合适还需正式运行时确认，可先使用32/128/256。
# 源默认32环境。仅新增独立run目录，不覆盖旧训练日志。

# mjlab原生回放
scripts/mjlab.sh play --checkpoint logs/mjlab/chuanliantui/<run>/model_<n>.pt --render

# 独立CPU MuJoCo回放 / 30秒验收（不自动reset隐藏失败）
scripts/mjlab.sh play --backend cpu --checkpoint logs/mjlab/chuanliantui/<run>/model_<n>.pt --render
scripts/mjlab.sh eval --checkpoint logs/mjlab/chuanliantui/<run>/model_<n>.pt --seconds 30 --report /tmp/chuanliantui-eval.json

# 新接口普通actor导出：float32 [1,35] -> [1,6]，opset13，预处理在图外。
scripts/mjlab.sh export --checkpoint logs/mjlab/chuanliantui/<run>/model_<n>.pt --output exports/mjlab/chuanliantui/<run>/policy.onnx

# 有限接口/物理检查，无learn()
scripts/mjlab.sh check --num-envs 4 --steps 64
mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_mjlab.py --gpu --out /tmp/chuanliantui-mjlab-check.json
```

ONNX 导出拒绝覆盖已有输出，附校验 JSON 和 PyTorch / ONNX Runtime误差。普通actor没有旧encoder结构；禁止使用旧`policy_1.pt`的要求继续适用于旧Isaac环境。

正式行为验收仍待训练策略：持续30 s，后半段高度误差≤.02 m、pitch≤5°、水平速度≤.05 m/s、整段漂移≤.20 m，且无失败；起立过程及是否满足真机要求还须可视回放判读。没有开始训练，因此有限数值检查、随机actor导出和零动作物理响应不能视为策略验收通过。
