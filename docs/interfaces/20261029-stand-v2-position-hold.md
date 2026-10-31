# 稳站v2：yaw配方上的停车位置外环

| 项目 | 内容 |
|---|---|
| 日期／作者／复查 | 2026-10-29；Codex实现与自查；H7负责人核对待确认 |
| 任务／父模型 | chuanliantui_stand_v2；B5999，SHA256 `a41b77caaf02389e2a1160a1781eda0d573ed33836e3cd6fe32ed02d090e825a` |
| 旧接口 | [yaw转向／松杆航向保持v7](20261029-stand-v2-yaw-heading-hold.md) |
| 新profile | `chuanliantui-stand-v2-isaac32-yaw-position-hold-v8` |
| 新command contract | `stand-v2-post-stand-vx-yaw-position-hold-v4` |
| 奖励recipe | 保留B的`stand-v2-upright-first-tracking-active-weight1-tight-v6`；奖励公式和权重未改，静止角色停车锁定后的激活范围见下文 |
| 训练仓／分支／基点 | Wheel-Legged-Gym；26_wheelleg；952e4260aedf04e28fdbf366d5cb1c5562c75263及已有工作区改动；远程URL本轮未重新核对 |
| 部署侧 | H7及独立CPU入口未改；新增外环、位移来源与H7已有LQR速度积分量是否可复用均待负责人确认 |
| 权威实现 | `stand_v2/mdp.py`的`_advance_position_command`、`_advance_speed_command`；runner/export及play／实验入口 |

实现和冻结检查阶段未训练；后续经用户授权从5999续训300轮至6299，随机化原成功保留未通过，未接条件预算3000轮，见[完成证据](../../artifacts/stand-v2-yaw-poshold-trial300-20261029/README.md)。允许约0.05m/s短时慢漂与限制长期位移是独立要求；外环在训练命令项内生成小幅反向速度，复用vx槽0，不新增位移观测或位移奖励。既有首次起立nominal_state缩放未撤销，因此未来从5999续训并非仅停车外环的单变量实验。

## 锁定、清除和输出

显式`--position-hold`，依赖`--yaw-control --post-stand-tracking --speed-curriculum`。默认参数均存入checkpoint顶层`position_hold_config`和课程内`position_hold`，加载权重前核对两份参数及新契约。

| 参数 | 默认 | 含义 |
|---|---|---|
| position_gain | 1 s⁻¹ | 位移误差到速度的比例kp |
| position_velocity_limit | 0.1 m/s | 外环速度请求上限 |
| position_tolerance | 0.02 m | 前向位移硬死区，边界以内输出0 |
| position_lock_speed | 0.05 m/s | 锁定前实测前向速度绝对值须严格低于此值 |
| position_lock_duration | 0.5 s | 原始vx为0且yaw_ready期间，连续低速时长；向上取整为策略拍数 |

原始vx请求为`tracking_velocity`，与外环输出及限斜率后的实际命令分别保存。已站起、NORMAL、未fallen并额外稳定50策略拍才yaw_ready；之后原始vx=0且连续低速50拍才锁定**当时**的xy，滑行阶段不追旧位置。低速计时中任一拍超阈值即清计数；锁定后不因纠偏导致速度上升而丢失基准。原地转kind=2也执行位置保持。

原始vx非零、失稳／RECOVER、yaw不ready、物理复位均清基准、低速计时、误差和外环请求。恢复后重新等待低速锁定；模型恢复也不携带旧物理回合停车点。重复读取同拍命令不推进计时、斜率或基准，局部复位仅影响选中环境。

令`d=current_xy-anchor_xy`，当前航向为ψ：

```text
e_forward = d.x*cos(ψ) + d.y*sin(ψ)
e_lateral = -d.x*sin(ψ) + d.y*cos(ψ)
v_position = 0                                      if abs(e_forward) <= tolerance
v_position = clamp(-kp*e_forward, -v_limit, v_limit) otherwise
```

只用前向分量控制；横向量单独记录。用当前航向的sin/cos投影，跨±π连续。实际vx继续走原每回合0.5～2m/s²限斜率；P输出限幅不等于移动转停车时上一拍实际命令会瞬间降到该幅值，失稳安全清零仍优先。

**本次仅P加死区，无位置误差积分项。** 若内层速度跟踪有恒定偏置b，未饱和、死区外的稳态位置误差近似`b/kp`；例如b=0.05m/s、kp=1/s时约5cm，不能承诺回到零误差。增大kp可缩小偏置造成的误差，但对测量噪声更敏感、回拉更急；较小v_limit更柔和，但限制可抵消的速度偏置和回位速度，偏置超过上限时仍可能持续走偏。死区减少小幅来回纠正，不消除偏置或里程计累计误差。

## 小速度训练风险与门控改法

原`_sample_yaw_speed_target`的非零**目标**幅值为0.1～当前档位上限，缺少持续的`0<|vx|<0.1`随机目标；已有加减速斜坡会经过此区间，不能说从未见过小速度。奖励实际读取槽0和真值vx，窄核`exp(-(command_vx-vx)²/0.02)`对小命令仍有效，没有0.1屏蔽阈值。

原静止角色`tracking_active=False`：宽核有效权重0.2，窄核为0。仅把纠偏写入vx会让这半数回合缺少窄核训练信号。因此已在实施前向用户说明并采用：

- 原`tracking_episode && ready50`仅用于随机目标采样和速度晋级，静止角色不会被随机采成移动。
- 实际命令／奖励的`tracking_active = 原运动激活 || 已锁定停车点`。静止角色锁定后，包括死区内输出0时，沿用现有激活期宽核有效权重1.0及窄核0.5；奖励文件、公式、配置权重均未改。
- 停车小命令不新增速度晋级样本；原接近当前档上限的非零直行／弯行段仍按旧条件计数。停车质量在独立评估中报告，不用它解锁高速档。

外环会在训练采样中自然生成0～0.1m/s的小目标，不能据此声称5999能准确执行。后续300轮试训的停车误差有局部改善，但原成功保留未通过；偏置、回拉及振荡结果见完成证据。

## 观测、ONNX和部署

32维单帧、160维历史、89维Critic、6动作及命令缩放`[2,0.25,5]`保持；PD、XML、物理参数、500Hz物理／100Hz策略保持。网络只看到实际vx/yaw命令。ONNX仍为float32 `observation_history[1,160] → actions[1,6], velocity[1,3]`、opset13。

停车基准、位移来源、当前航向投影、死区、P增益、限幅、vx限斜率、恢复／复位清除均由调用方维护，不进入ONNX。维度相同也必须成套同步新命令语义；旧checkpoint普通加载误开新开关、新checkpoint漏开开关均拒载。显式实验入口先完整恢复5999及两个优化器，再启用yaw和停车外环，保留速度档、清旧分布统计窗口；旧配置关闭路径包括RNG保持逐位等价。

训练默认使用仿真位置与前向速度真值。评估额外提供`--position-source wheel-odometry`：按`model.py`的电机顺序轮槽2/5，通过其`MODEL_XML`编译碰撞网格读取实际半径、关节轴读取符号（左+Y、右−Y），以`mean(r_i*sign_i*dq_i)`沿当前理想航向500Hz积分，再以100Hz供外环；该模式的锁定速度也使用轮速。半径、符号和来源写入结果。

两种来源均保存真值停车基准和真值误差，同时记录轮速里程与真值差，以暴露滑移等偏差。此简化里程计没有打滑、腿摆动或机身俯仰补偿；理想航向也不代表已验证IMU误差。H7是否能复用LQR现有速度积分量、轮速符号/估计器/坐标与复位语义是否一致，**待确认**；未修改H7代码。

评估可额外加`--truth-lock-speed`，仅在`--position-source wheel-odometry`时使用：位置仍取轮速积分，锁定低速条件改用仿真前向速度真值，单独观察位置积分误差。这是诊断消融，不是可部署的真值输入；默认轮速来源同时用轮速判锁。在本次旧5999对照中，轮速未能连续0.5秒低于0.05m/s，因此全轮速模式0/15环境锁定；结果保留空误差窗口，未放宽阈值强行锁点。

## 检查与测量

统一使用仓库虚拟环境及`PYTHONPATH=mjlab_training/src`。`scripts/agent/check_stand_v2_position_hold.py`检查关闭时yaw开／关两条旧路径的命令/RNG、低速门槛与连续计时、滑行后锁定、清除时机、P死区/饱和/斜率、当前航向及±π、静止角色和原地转、小命令奖励、课程隔离、部分复位、参数与契约拒载、两优化器、ONNX及真实play16步；另做40秒冻结推理。原正式检查仍保留。

扩展后的`assess_stand_v2_yaw_control.py`原yaw命令不变；新模型自动识别停车契约，旧5999须同时显式`--probe-yaw --probe-position-hold`。在原15组40秒/500Hz调度中，原始vx为0的段均报告全段锁定期间与末1秒的前向/横向误差、未锁定时长、回拉请求/实际命令/真实速度峰值、超过死区后的误差过零次数，以及里程计偏差；末窗口未全程锁定或提前失败时不填零冒充成功。旋转改变投影也可能使误差过零，振荡候选只是诊断信号。

本次实际退出码、失败修复记录、源码哈希和结果见[新artifact](../../artifacts/stand-v2-position-hold-20261029-214533/README.md)。冻结5999只能作为探测，不能作为新配方行为通过、随机化／多种子或真机通过。

## 从5999同时续训yaw和停车

以下保留从5999启用两项的通用命令；实际300轮运行及未接3000轮的决定见上述完成证据。将轮数和新运行名占位符替换后使用；输出和logs目录必须新建。TensorBoard命令指向该入口实际添加完整后缀的目录。

```bash
HOLD_ITERATIONS=<本次训练轮数>
HOLD_RUN=<新的运行名>
PYTHONPATH=mjlab_training/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
mjlab_training/.venv/bin/python scripts/agent/experiment_stand_v2_phase_mix.py \
  --checkpoint logs/mjlab/chuanliantui_stand_v2/20261029-155722_tight_fromA4999_2048_activew1_tight/model_5999.pt \
  --output "artifacts/$HOLD_RUN" --run-name "$HOLD_RUN" \
  --num-envs 2048 --seed 42 --iterations "$HOLD_ITERATIONS" --min-std 0.05 \
  --post-stand-tracking --speed-curriculum --tracking-weight-while-active \
  --tracking-tight-term --yaw-control --position-hold

mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir "logs/mjlab/chuanliantui_stand_v2/${HOLD_RUN}_activew1_tight_yawhold_poshold" --port 8080
```

新模型回放在原yaw命令后增加`--position-hold`。新模型冻结评估用`--checkpoint <新模型> --output <新目录>`；旧5999探测额外加`--probe-yaw --probe-position-hold`，里程计对照另开新输出目录并加`--position-source wheel-odometry`。
