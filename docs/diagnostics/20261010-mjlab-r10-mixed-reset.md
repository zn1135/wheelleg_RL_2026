# r10 混合初态检查

2026-10-10，`26_wheelleg`，HEAD `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交工作区。本轮按用户选择将普通／地面初态采样概率恢复到各 0.5，关闭地面 root 姿态和高度扰动。接口及完整范围见 [r10记录](../interfaces/chuanliantui-mjlab-wheelleg-recover-r10.md)。随机网络初始化仍为默认，没有启动训练或 TensorBoard。

## 执行与结果

独立 Python3.12/mjlab 环境，RTX4060、NVIDIA595.99.02。执行现有有限检查入口，退出码0：

```bash
mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_reference_training.py \
  --num-envs 32 --steps 128 --seed 42 \
  --out /tmp/chuanliantui-r10-mixed-reset-check.json
```

- 2048 个实际复位样本：1000 个地面（48.83%）、1048 个普通；按概率独立采样，不是每批固定各一半。
- 地面 root 高度相对环境原点为0.25 m，最大误差0；单位四元数最大误差0。普通 root 最低约0.38 m。
- 模式复位：地面为recover、普通为normal；动态切换和部分复位保持检查通过。
- 32 环境×128控制步，1280物理子步：观测、奖励有限；数值异常0；自动复位77次。这些复位来自零请求动作检查，不能当作训练失败率。
- 对每个实际积分后的 GPU qpos 独立重新计算 CPU 运动学，最大闭链pin偏差 **2.3234 mm**，低于5 mm检查门槛；终止前状态也计入。
- PPO配置与参考一致；除初态、机器人、模式、动作顺序及膝权重这些已记录差异外，环境配置一致。奖励公式、高度门控、动作顺序及腿／轮控制映射检查通过。
- checkpoint保存／加载、临时随机actor的ONNX数值检查通过，最大差约1.19e-7。临时文件在临时目录清理，不交付任何已训练新模型。

完整manifest、配置、文件SHA256和检查结果保存在 [检查JSON](20261010-mjlab-r10-mixed-reset-check.json)。

## 20环境可视预检

```bash
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  scripts/mjlab.sh play --num-envs 20 --random-reset --render --seconds 10
```

退出码0，原生窗口正常运行并结束；零请求动作，无学习。桌面授权路径仅记录本次实际执行情况，其他登录会话需读取各自当前路径。此预检没有人工逐轨迹判读或新策略站稳验收。

## 历史模型回放

r9历史manifest在改变默认配置前保存，原生和CPU入口从历史checkpoint恢复原初态参数；r6～r8也读取各自记录的初态参数。r9真实模型加载通过，完整resume到r10被拒绝。

执行正式CPU入口，退出码0：

```bash
scripts/mjlab.sh eval \
  --checkpoint logs/mjlab/chuanliantui/20261010_101723_976809_wheelleg_recover_r9_no_knee_4096_1000/model_999.pt \
  --seconds 2 --report /tmp/chuanliantui-r10-historical-r9-eval.json
```

历史r9在0.92秒机身接地失败，报告metrics与此前r9 CPU评估完全一致。这里只证明回放配置兼容，旧策略行为仍未通过；没有把旧模型自动改成混合初态评估。

## 尚未执行

r10策略训练、训练后连续站稳评估、模型交付和H7真机验收均未执行。没有改动机器人资产、另一位协作者的部署代码或两侧历史logs；没有Git提交。
