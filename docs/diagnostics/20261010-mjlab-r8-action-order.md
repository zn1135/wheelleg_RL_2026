# r8动作顺序恢复检查

2026-10-10，用户要求动作输出恢复`[lf0,lf00,rf0,rf00,lfwheel,rfwheel]`。分支26_wheelleg，HEAD58947e8fabc4683b2025d42bda742b06c84dc566加未提交改动。接口见 [r8](../interfaces/chuanliantui-mjlab-wheelleg-recover-r8.md)。未训练或提交。

执行：

```bash
mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_reference_training.py --out /tmp/chuanliantui-r8-action-order-20261010.json
scripts/mjlab.sh play --checkpoint logs/mjlab/chuanliantui/20261010_003055_713431_wheelleg_recover_r6_4096_1000/model_999.pt --num-envs 1 --seconds 1
scripts/mjlab.sh eval --checkpoint logs/mjlab/chuanliantui/20261010_003055_713431_wheelleg_recover_r6_4096_1000/model_999.pt --seconds 1 --report /tmp/chuanliantui-r8-r6-cpu-compat-20261010.json
```

最终三项均退出0。[完整检查JSON](20261010-mjlab-r8-action-order-check.json)保存本次manifest。

- 六个单槽动作逐项检查实际XML具名电机ctrl：前四位置PD、后两轮速度控制正确；观测22:28请求动作与策略顺序一致。
- 三个不同的动作历史验证腿／轮平滑奖励正确分组；27项纯奖励与参考数值误差均0。配置除机器人Entity、动态模式和policy_joint_order外与参考一致，PPO完全一致。
- 动态mode连续25步站稳、5步失稳、当前高度命令、部分reset和重复查询检查继续通过。
- GPU32环境×128策略步、1280物理子步；独立CPU运动学逐个检查积分后GPU qpos，最大闭链误差2.578mm<5mm，数值异常0。零请求动作普通reset74次，不表示起立成功。
- 临时随机actor runner构造／checkpoint保存加载／ONNX检查通过，无PPO学习。ONNX最大绝对误差1.19e-07。
- r5/r6真实历史actor可加载，拒绝完整resume；r6最终策略用旧顺序无渲染回放1秒退出0，仅证明入口兼容。
- r6旧顺序CPU入口1秒评估退出0，无终止失败；因时长不足且未严格恢复，passed=false，不能作为行为通过。与原生入口共同检查新动作顺序参数的历史路由。
- r7没有正式训练模型，以保存的r7历史manifest生成推理夹具，检查旧顺序路由及加载、完整resume拒绝；不声称验证了r7训练策略行为。
- 首次新检查在仿真首次step前读取尚未创建的reset_terminated属性失败；修复检查器的初始无终止状态并重跑通过，没有改训练环境的终止规则。
- git diff --check通过；H7适配及新策略行为验收未执行。
