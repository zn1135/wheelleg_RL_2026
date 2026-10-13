# r9关闭膝限位惩罚预检

2026-10-10，26_wheelleg，HEAD58947e8fabc4683b2025d42bda742b06c84dc566加未提交改动。用户要求关闭knee_limit并再训练，接口见 [r9](../interfaces/chuanliantui-mjlab-wheelleg-recover-r9.md)。

```bash
mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_reference_training.py --out /tmp/chuanliantui-r9-no-knee-limit-20261010.json
```

退出0，[完整检查JSON](20261010-mjlab-r9-no-knee-limit-check.json)记录当前manifest。

- 对照参考，仅knee_limit权重-0.5→0；其余26项权重相同，全部27项原始公式数值对照误差0。环境配置另保留用户此前指定的动态模式和策略动作顺序、机器人Entity例外，PPO完全相同。
- 每步knee_limit加权贡献精确为0。冻结同一个状态，对照权重0与-0.5的总奖励差，误差1.43e-07<1e-6，证明只移除此项贡献。
- 四腿两轮具名电机映射、三拍动作历史、动态模式、当前高度命令和部分reset检查通过。
- GPU32环境×128策略步／1280物理子步，最大闭链误差2.578mm<5mm，数值异常0，零请求动作普通reset72次；不是策略起立成功率。
- 临时随机actor构造／checkpoint保存加载／ONNX检查通过，无PPO学习，ONNX误差1.19e-07。
- r5/r6真实actor推理加载保留；r7/r8历史manifest推理夹具加载和原knee权重恢复通过，拒绝完整resume到r9。r7/r8没有正式训练策略，本项不证明其行为通过。
- git diff --check通过；检查本身未训练。随后启动用户授权训练，状态见 [训练记录](20261010-mjlab-r9-formal-training.md)。
