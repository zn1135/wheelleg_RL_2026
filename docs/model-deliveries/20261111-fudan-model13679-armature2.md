# 复旦13679轮PT：P60/D2与两倍腿惯量

2026-11-11按用户“提交这个模型到git”整理13679完整检查点。交付人Codex，接收人／H7独立核对人待确认。本包为PT与来源资料；提交状态以Git实际记录为准，本次未授权推送，无新ONNX、固件或默认选用模型替换。

## 模型与训练

| 项目 | 内容 |
|---|---|
| 机器人／任务 | 本仓闭链轮腿、fudan_ppo_walk，原线性气弹簧、标定限位 |
| 完整PT | `artifacts/fudan_ppo_walk/models/20261111_model13679_p60d2_armature2.pt`，1442121字节 |
| SHA256 | `5055badffffc65e6ca2d5df9da96f4d92542a4a76c3a4b04edfb0db9c3257678` |
| 网络合同 | `fudan-ppo-walk-physical25-history125-critic144-v1` |
| 历史合同 | `fudan-physical-pd-centred-warmup-clean4-noisy1-unclipped-history-v2` |
| 输入输出 | Actor当前25＋历史125→6动作；Critic环境141＋encoder3；PT包含两个Adam和RNG |
| 本轮 | 从13179新增500完整轮到13679；4096环境×48步、seed1 |
| 优化器 | 两Adam各新增10000，累计273580；初始学习状态与CPU/CUDA RNG逐值恢复 |
| 控制 | **实际P60/D2**，原±15%增益／力矩倍率；物理／PD2ms、策略10ms |
| 四腿惯量 | `lf0/lf00/rf0/rf00`的armature各0.03159kg·m²，为原值两倍 |
| 奖励 | 17项roll/pitch配方；前倾0～+5°弱罚1/20，无固定2°目标或图外补偿 |
| 延迟与外载 | 原训练动作6～14ms、传感2ms；各轴±10N／±10Nm持续外载及7秒／0.3m/s小推扰 |
| 未添加 | 没有加入后续诊断中的PD速度反馈噪声或6ms PD速度反馈滞后 |

**冻结源码中的磁盘默认配置是P30/D2，但本次启动器在训练进程内覆盖为P60/D2。加载或部署时必须采用本页的实际P60/D2，不能根据磁盘默认值猜测。** 两倍惯量是仿真物理候选的采用，不代表真实转子惯量测量，也不是要求修改真机惯量参数。

## 来源、配置与重建

训练仓`https://github.com/zn1135/wheelleg_RL_2026.git`，分支`26_wheelleg`，训练时工作区有未提交改动。训练时HEAD未单独封存；本包选择`b70edf7aa2ee5132bb3519c9af44b29f4586863b`作为已经验证的**重建基准**，不将其冒充训练时全部代码。

- [配置与校验清单](data/20261111-model13679-armature2-manifest.json)，SHA256 `1abad905059c490074f24e14e2b27e11b6ba0ee0edd69ef736b3de3e5f11b35c`，包含实际配置、源模型、训练审计和逐例停车比较摘要。
- [源码／XML补丁](data/20261111-model13679-armature2-source.patch)，SHA256 `5f5fcbae8a802bfabce5949c23e6f3c476716dafa7bd5205466351f488165f2e`。在重建基准上使用`git apply --unidiff-zero`；已在独立临时索引中实际应用并逐字节核对151份文件，未修改用户工作树或原暂存区。
- 其中143份运行Python和2份资源资产按训练封存SHA核对；训练启动器使用日志内冻结副本。启动器依赖与此前封存副本相同，其他辅助检查脚本提供当前版本，并在清单中区分来源。该补丁用于模型追溯，不表示生产代码已在本次提交中直接合入。
- 新XML SHA256 `ed817cf8b0d41bc3625021c97deef33549f0c8aef7ec4e98a49d6d6a9ac2bb11`，四个主动腿惯量0.03159；原线性气弹簧、附加拟合关闭。其余资产SHA在清单中。
- 输入模型为`artifacts/fudan_ppo_walk/models/20261111_model13179_p60d2_roll_pitch.pt`，SHA256 `8ef6f785dab2bbb47bb0145fd0a59844d7051b323e73498cc4ca6450a559b77c`；恢复学习状态，物理／回合／历史现场重新初始化。
- 原日志`logs/mjlab/fudan_ppo_walk/20261111_p60d2_armature2_resume13179_to13679_01/`，原审计`artifacts/fudan_ppo_walk/20261111-p60d2-armature2-resume13179-500-01/`；只引用，不修改历史记录。

训练为MuJoCo Warp GPU／RTX4060 Laptop；Isaac不适用。交付读取工具版本见清单，完整外部依赖未独立锁定。历史完整日志、父运行快照、评估轨迹和视频未随本包提交，启动器中的历史路径需要相应本机资料，不能声称这是独立的一键训练包。重跑须依据有效配置建立新目录并重新预检，保留原P60/D2覆盖。

原启动与TensorBoard命令仅供追溯，训练已经完成，不应重复写入原目录：

```bash
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python -m scripts.agent.fudan_ppo_walk.resume_armature2_13179 run --output artifacts/fudan_ppo_walk/20261111-p60d2-armature2-resume13179-500-01
mjlab_training/.venv/bin/python -m tensorboard.main --logdir logs/mjlab/fudan_ppo_walk/20261111_p60d2_armature2_resume13179_to13679_01 --port 6014
```

## 验证与允许使用范围

| 阶段 | 结果 |
|---|---|
| PT复制／参数／优化器 | SHA与最终训练文件一致；权重有限、std正、两Adam各273580更新；相同字节已在GPU评估严格加载 |
| 训练完成 | 500完整轮、恢复／RNG、17项奖励日志及运行源码审计通过，目标13679，没有追加训练 |
| 同条件停车诊断 | 两模型同P60D2／两倍惯量，各18例完整有限；初态／扰动／惯量逐数组与cohort相同 |
| 无外力摆腿 | 六次行驶后停车末1秒最大关节峰峰值中位数3.022→2.351°，下降约22.2% |
| 停车判据 | 无外力仍3/9，保留原成功但无新增；0.22/0.26m四次停车仍超过3秒；有外扰仍0/9且部分摆幅增大 |
| 完整987／转向／起立 | 未执行，停车子集不能替代完整运动验收 |
| 新ONNX／板端数值时序／真机 | 未执行；H7版本与独立核对待确认，无本次固件交付或关联PR |

逐例摘要随清单提供；本机[完整对照与视频](../../artifacts/fudan_ppo_walk/20261111-armature2-13679-vs13179-01/README.md)未随本包上传。旧模型保留，可用于离线评估或另行授权续训；不能据模型已提交认定实机可部署或已通过验收。
