# 稳站v2：转向角速度与松杆航向保持

后续显式`--position-hold`配方见[停车位置外环v8](20261029-stand-v2-position-hold.md)；本页保留仅yaw的v7接口与原检查记录。

| 项目 | 内容 |
|---|---|
| 日期／作者／复查 | 2026-10-29；Codex实现；独立子代理检查集成与验证脚本，不代替H7负责人核对 |
| 任务／父模型 | chuanliantui_stand_v2；B5999，SHA256 `a41b77caaf02389e2a1160a1781eda0d573ed33836e3cd6fe32ed02d090e825a` |
| 旧接口 | [速度限斜率v6](20261029-stand-v2-speed-slew.md)、[B窄核奖励](20261029-stand-v2-tight-tracking.md) |
| 新profile | `chuanliantui-stand-v2-isaac32-yaw-heading-hold-v7` |
| 新command contract | `stand-v2-post-stand-vx-yaw-heading-hold-v3` |
| 奖励recipe | 保留父模型B的`stand-v2-upright-first-tracking-active-weight1-tight-v6`，本次不调奖励权重 |
| 训练仓／基点 | Wheel-Legged-Gym；26_wheelleg；952e4260aedf04e28fdbf366d5cb1c5562c75263及已有未提交改动；远程未重新核对 |
| 部署侧 | H7_RL_ros、CPU sim2sim未修改；新命令处理与行为均待适配／负责人确认 |
| 权威实现 | stand_v2/mdp.py、runner.py、export.py；scripts/play.py；experiment_stand_v2_phase_mix.py |

用户确认允许0.05m/s慢漂，新增原地转、边走边转，以及松开转向指令后保持当时朝向。位移积分奖励仅讨论，未加入本配方。此处实现控制输入与训练分布，尚无训练后的yaw模型；不能把5999的接口迁移产物当作已学会转向。

## 命令与保持

显式`--yaw-control`，依赖`--post-stand-tracking --speed-curriculum`。默认`yaw_rate_limit=0.5 rad/s`、`yaw_acceleration=1 rad/s²`、`heading_gain=1.5 s⁻¹`，数值存入checkpoint并严格核对。

- 原始转向请求非零时按角速度转动，同时更新当前朝向参考；请求归零当拍锁定当前朝向。
- 保持时计算`wrap_to_pi(heading_ref-heading)`，乘heading_gain后限幅±0.5rad/s；再按每10ms最多改变0.01rad/s的斜率下发。跨±π使用最短角差，参考不跟随漂移移动。松杆后可能先制动超调、再回到锁定方向，评估单列超调。
- 已站起且NORMAL、未fallen，额外连续50策略拍才允许yaw；失稳立即清零实际yaw和原始请求、作废参考，恢复后重新锁当前方向，不追旧参考。
- 前向激活仍是`tracking_episode && ready`。50%静止角色保持vx=0，也允许yaw纠偏；因此新配方的静止组不再承诺`command[:2]`全零。vx窄核仍只跟随原`tracking_active`，不会因静止yaw纠偏误激活。
- 另50%运动回合，每2～4秒等概率抽取停止、直行、原地转、弯行。非零vx幅值0.1～当前速度档上限，非零yaw幅值0.1～0.5，正负等概率。vx原斜率0.5～2m/s²保持。原地转不参与vx晋级统计；直行与弯行共同计入非零vx课程，尚未新增yaw晋级课程。

## 兼容性与接口

32维Actor、160维五帧历史、89维Critic、6动作保持。命令槽6:9仍是实际vx、实际yaw角速度、目标高度，缩放`[2,0.25,5]`；不是直接给网络航向角。IMU角速度、坐标、电机顺序、PD、限幅、XML、100Hz策略／500Hz物理、复位与初始姿态均保持。

ONNX仍为`observation_history[1,160] → actions[1,6], velocity[1,3]`，float32、opset13。航向参考、角度回绕、限幅、限斜率及恢复清零由调用方维护，不在ONNX图内；未来部署必须同步这些逻辑，维度相同不代表可直接替换旧模型。

从5999先按原契约完整恢复Actor/Critic/encoder及两个优化器，再显式启用yaw。保留vx档位，清原统计窗口（迁移前窗口另记），避免旧直行成绩混入新分布。新checkpoint保存两处相互一致的yaw参数；加载权重前校验。回合内heading参考不跨物理复位保存。旧模型普通加载与新模型漏开yaw均拒绝，旧`yaw_control=False`路径保留。

本次不撤销用户已有的首次起立nominal_state缩放；当前奖励源码与5999训练快照存在该已有差异。因此未来续训不是“只改yaw”的单变量对照。

## 检查与测量

使用本仓mjlab解释器，命令前置`PYTHONPATH=mjlab_training/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。

`check_stand_v2_yaw_control.py --checkpoint <原B5999> --output <新目录>`检查旧源码关闭路径/RNG等价、49/50拍门控、转向/松杆锁向、±π、失稳与局部复位、实际命令观测/奖励、两优化器续载、配置/版本拒载、40秒冻结推理、ONNX及真实play有限步进。其interface-only产物不是训练模型。

`assess_stand_v2_yaw_control.py --checkpoint <yaw模型> --output <新目录> --conditions nominal --seed 43 --num-envs 60`按vx=0/±0.3与yaw=0/±0.3/±0.5共15组测量40秒。旧5999必须显式加`--probe-yaw`，仅探测未训练能力。前8秒起立；随后转动、松杆保持、反向转动、再保持、双轴停止。500Hz保存实际命令、原始请求、真实速度、yaw/reference/门控和物理轨迹；提前结束后不统计复位结果。

分别报告yaw误差≤0.1rad/s、vx误差≤0.1m/s、保持末1秒最大角差≤5°、零平移时末1秒平面速度≤0.05m/s等指标，以及接触/倾角/失败；这些是新任务测量指标，不重标旧v3.1、不把单个指标当整机行为通过。`--selfcheck`仅核对统计边界。实际命令、结果与未验证范围见[本次检查记录](../../artifacts/stand-v2-yaw-hold-20261029-q3YOzI/README.md)。

## 未来续训与TensorBoard

当前未获训练预算，只交付命令。将`YAW_ITERATIONS`填为本次明确选择的正整数，`YAW_RUN`用新的目录名；不覆盖旧日志。例中从已确认的5999续训2048环境：

```bash
YAW_ITERATIONS=<本次训练轮数>
YAW_RUN=<新的运行名>
PYTHONPATH=mjlab_training/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
mjlab_training/.venv/bin/python scripts/agent/experiment_stand_v2_phase_mix.py \
  --checkpoint logs/mjlab/chuanliantui_stand_v2/20261029-155722_tight_fromA4999_2048_activew1_tight/model_5999.pt \
  --output "artifacts/$YAW_RUN" --run-name "$YAW_RUN" \
  --num-envs 2048 --seed 42 --iterations "$YAW_ITERATIONS" --min-std 0.05 \
  --post-stand-tracking --speed-curriculum --tracking-weight-while-active \
  --tracking-tight-term --yaw-control

mjlab_training/.venv/bin/python -m tensorboard.main \
  --logdir "logs/mjlab/chuanliantui_stand_v2/${YAW_RUN}_activew1_tight_yawhold" --port 8080
```

训练后回放使用新训练模型：

```bash
scripts/mjlab.sh play --task chuanliantui_stand_v2 --checkpoint <新yaw模型路径> \
  --num-envs 20 --seed 43 --viewer viser --viser-port 8093 \
  --post-stand-tracking --speed-curriculum --tracking-weight-while-active \
  --tracking-tight-term --yaw-control
```

回放自动采样转向／保持场景，未新增GUI遥控按钮；命令处理复用原yaw角速度观测槽。单独开关不能让旧5999获得转向能力。没有启动训练、TensorBoard、浏览器服务或H7写入。
