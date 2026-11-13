# 模型交付：机械轮参数从零训练3000轮候选

## 任务与接口

2026-11-13，AI按用户授权完成训练与GPU自查并准备保存，接收人为用户，独立复查/H7负责人核对待确认。机器人chuanliantui_new_1闭链串联轮腿，任务fudan_ppo_walk。本模型用于机械轮参数试验，摆腿减小但停车/慢移退化，保持实验候选，不作为实机主模型替换或完整行为通过。

接口为`fudan-ppo-walk-physical25-history125-critic144-v1`，Actor25/历史125/Critic141+3/动作6。**history v3**：首帧同一含噪、限幅观测重复五帧，后续保存限幅历史；与17679的v2预处理语义不同，即使维度相同也不能直接套原v2部署历史处理。[v3接口](../interfaces/20261102-fudan-history-rollback-v3.md)、[机械轮接口](../interfaces/20261112-fudan-mechanical-wheels.md)。本次无ONNX/板端网络/固件导出，无H7修改或负责人确认；无PR。

## 代码与训练来源

训练仓`https://github.com/zn1135/wheelleg_RL_2026.git`，分支`26_wheelleg`。训练实际149份运行源码/资产/文档哈希逐一与基准提交`9a7e43c77be71db08c75e20a12ab8a6ba573c032`的Git字节相同；模型提交另外纳入4个训练/评估/回放辅助文件，其SHA及来源见[清单](data/20261113-model3000-mechanical-manifest.json)。工作区另有未提交部署改动，HEAD不能代表整工作区，但本次149项运行来源已单独核对。

这7文件包包含PT、4个辅助文件、本文和清单；运行时基准已在Git中，无需额外源码补丁。训练启动器与实际训练日志快照逐字节一致；有效配置和实际控制记录随清单保存。原始训练日志、GPU轨迹、视频与曲线保留本机，不随包提交；接收人可用清单复核逐例指标和来源。依赖环境仍需自行配置，清单版本是打包时读取，不冒充训练完整环境封存。

| 训练信息 | 实际配置 |
|---|---|
| 初始化/预算 | 从零初始化，未加载17679或其他checkpoint；0→3000即停，无追加学习 |
| 实际入口 | `PYTHONPATH=mjlab_training/src MUJOCO_GL=egl mjlab_training/.venv/bin/python -u -m scripts.agent.fudan_ppo_walk.train_mechanical_3000 run --output artifacts/fudan_ppo_walk/20261113-mechanical-scratch3000-01`；经统一train/task_registry/runner；复现须另建日志和产物目录 |
| seed/规模/更新 | seed1，4096环境×48步，589824000条转移，两Adam各60000更新，25928.21秒（约7h12） |
| 实际PD | 腿P60/D2，轮P0.1/D0；共享默认腿P30由启动器仅在进程内覆盖为60，不能只看默认配置认定训练用P30 |
| 轮端物理 | 单位增益，无原转速出力曲线/额外0.45；wheel_mechanical.json SHA `e0dff9534577e83869ca3eba9a899d8b932530bbf1cdfdeb2a08ef42449011ae`；具体左右armature/damping/frictionloss见清单 |
| 腿/XML | 四腿armature0.03159；XML SHA `ed817cf8b0d41bc3625021c97deef33549f0c8aef7ec4e98a49d6d6a9ac2bb11`；原闭链/弹簧/限位保持 |
| 奖励/随机化 | 原17奖励，10%站立/10%停车采样；动作6～14ms、传感2ms、PD/强度/质量/COM/摩擦随机化、±10N/±10Nm持续外载及7秒推扰保持 |
| 时序 | 2ms物理/PD，10ms策略；GPU诊断使用10ms动作/4ms传感延迟 |
| 环境 | RTX4060 Laptop、MuJoCo Warp CUDA；Python/PyTorch/CUDA/MuJoCo等版本在清单versions_at_packaging |
| 日志/TensorBoard | `logs/mjlab/fudan_ppo_walk/20261113_p60d2_wheelP01_mechanical_scratch3000_01/`；TensorBoard `mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/fudan_ppo_walk/20261113_p60d2_wheelP01_mechanical_scratch3000_01 --host 127.0.0.1 --port 6015` |

## 产物

[20261113_model3000_p60d2_armature2_wheelP01_mechanical.pt](../../artifacts/fudan_ppo_walk/models/20261113_model3000_p60d2_armature2_wheelP01_mechanical.pt)，1442057字节，完整3000轮checkpoint，含encoder/actor/critic、两优化器及RNG。SHA256：`9d4d3123d42451e8942581b7cc943bc35fa8e35f7947b5c5747abeb89de215ee`。

清单SHA256：`69380fa09f3f4f2780cc170f114a1dd10bcf23b577a28f810dcb30ae7388a87e`。没有ONNX、板端生成网络或固件，因此opset/导出数值容差/固件工具链不适用。加载PT不会自动选取PD与物理；MuJoCo回放必须显式腿60/2、轮0.1、机械轮gain1，并按PT自己的history v3处理。

## 分阶段验证

| 阶段 | 结果和边界 |
|---|---|
| 训练预检 | 20环境GPU真实训练复位、3510物理拍通过 |
| 训练审计 | 零初始化/空优化器、恰3000轮、两Adam各60000、有限权重、全部奖励步数和源码校验通过 |
| GPU模型对照 | 旧17679/新3000各21条，共42条完整有限轨迹；21组初态夹具逐项一致，两版同机械轮/轮P0.1，各保留原v2/v3历史；不是只改变轮数的消融 |
| 零指令腿摆 | 0.22/0.26/0.30m末1秒四腿最大往返范围1.35→0.13、0.73→0.49、0.77→0.05° |
| 慢移/自转 | 同三档30秒净位移0.08→0.67、0.18→0.98、0.23→0.75m；0.30m偏航+67.5→−113.0°，摆幅小不代表停稳 |
| 静止/停车 | 无外扰7/9→3/9，低/中站高4个原成功停车丢失、无新增；六个停车6/6→2/6；30秒零指令3/3→1/3，正式同初态评分判退化 |
| 外扰 | 两版均0/9，新3000存在后退倾倒 |
| 视频 | 7段正常速度左右对照，7500帧全量解码/FPS/时长/SHA通过，AI已抽查截图；用户最终视频验收未确认 |
| 奖励 | 前100轮均值−0.01691，末100轮0.01944，约1500～2000轮后平台；奖励不替代停稳验收 |
| 起立/完整987/转向/变高 | 未执行，定向子集不代表完整能力 |
| Isaac | 不适用，本轮为独立mjlab训练 |
| ONNX/板端数值/时序/真机 | 未执行，H7负责人待确认；不替换原17679部署 |

[本机完整结果及所有视频](../../artifacts/fudan_ppo_walk/20261113-mechanical-scratch3000-01/RESULTS.md)；各阶段实际命令、退出码、逐例指标/配对、时间及代码/模型SHA在清单records中。旧17679本次基准分数是在机械轮P0.1下新测，不能与其历史曲线/P0.045成绩混用。

允许范围为归档、代码核对与离线复现；不声明稳定或真机合格。下一步先记录零指令左右轮目标/实际轮速和估速偏差，慢移/自转减小且原通过停车不丢失才接受后续调整。回退保留[17679原模型包](20261112-fudan-model17679-output045.md)及其物理/PD/v2说明，本包不自动切换部署模型。
