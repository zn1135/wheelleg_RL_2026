# 轮角39 PPO：Isaac起立奖励200轮候选

2026-10-26，本地仿真实验记录；未作为停车改进模型采纳，未交付硬件，独立人工复查人及H7负责人核对待确认。

当前串联腿闭链共享XML，正常起立及零速平衡任务；actor39／critic82／动作6，轮角首次恢复清零v2合同。奖励配置见[Isaac起立奖励移植](../interfaces/20261026-wheel39-isaac-standup-rewards.md)。无PR、提交、推送、分支或工作区变更。

训练仓为当前detached HEAD `2893b88ae99b61c3f9d06f06cb54b8d3ea94c4c3`，脏工作区；完整源码及差异见[实验目录](../../artifacts/ppo-isaac-standup200-20261026-v1/README.md)，源码清单SHA `1f1708b6fe455fee3bd8dfd66fbbee999cc33f9168d4778b2dac3b55b8bf3a73`，tracked diff SHA `7f51ace50a62c4fac21ec76b4fe5f28d390b614a7f0036dcbca1bc079f726119`。部署仓未修改，未构建固件，部署版本不用于本轮仿真结论。

Python3.12.3／Torch2.14.0+cu130／CUDA13.0／MuJoCo3.11.0／RTX4060 Laptop GPU。实际命令、env／agent快照与哈希见实验目录README及provenance.json。当前XML SHA `cac8061de2850602bfd1421fc856bf217c4e5040a11c8f5d33cf1abe61d5f0cf`；BC和SMP输入哈希见plan.json，均未改变。

本地北京时间15:22:29～15:31:29，从稳定BC的actor开始，新critic／Adam；不是旧PPO完整续训。200轮含20轮预热，实际actor894次、critic4000次、BC／SMP训练更新0。最终`model_199.pt`路径见实验报告，SHA `e7dd411c42846940edb649c524b75cdab3912aab31c0a308aab69864289cf98e`。完整checkpoint包含actor／critic／Adam／冻结BC参考／recipe及SMP状态，无encoder、无ONNX或板端产物。

正式预算审计和原40组20秒回放复算通过：起立／最终双轮平衡／平滑40/40，停车0/40。末段速度0.02211702m/s、漂移0.10984481m、腿速RMS0.00842176rad/s；相对此前v1累计400轮漂移更大，未超过LQR。不同奖励／约束及预算，不能作唯一因果归因。固定case0完整视频和轨迹哈希见实验目录。

Isaac运行时不适用于本轮mjlab实验；CPU sim2sim、运动、10ms、完整随机化、ONNX、板端与真机未验证。原稳定BC及旧模型均保留，未替换默认模型。预算用完后停止，后续训练须另行确定。
