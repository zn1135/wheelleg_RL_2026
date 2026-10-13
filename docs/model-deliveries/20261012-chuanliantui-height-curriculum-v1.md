# 串联腿高度课程两组模型交付记录

授权日期2026-10-12；训练与评估由Codex执行，交付后由Claude独立复跑评估。接口为 `chuanliantui-mjlab-height-curriculum-v1:curriculum/curriculum_knee`，用途是A2500高度响应的500轮对照，非真机部署授权。

## 代码与模型来源

训练工作副本：`D:\IMCA\RL\Wheel-Legged-Gym`，WSL `/mnt/d/IMCA/RL/Wheel-Legged-Gym`。本次无Git操作；当前提交SHA未核实，不能用迁移历史提交代表新增实验。完整实际源码和资产用远端冻结目录及127文件SHA256记录标识。

冻结目录：`/root/gpufree-data/wheelleg-remote/releases/20261012-height-curriculum-v1`；冻结包release-v4.zip SHA256 `bd0485b76e560687e6b572cfbdaf8ff598a21f309af33be75d3d07e6a5340090`。逐文件哈希在campaign `frozen-release.json`，本地复制于 `artifacts/height-curriculum-training-20261012/campaign-launch/`。默认V14、历史训练/评估代码与资产未修改。

源模型是A2500 baseline完整checkpoint，SHA256 `706c8387cba4815b491be3f5914a5951881c2966edabbd5fbb90d0da9c9d3dbd`，原接口校验通过；迁移actor、critic、Adam逐项相等，累计2500、学习率1e-5、自适应PPO，L=0.21。训练两组分别4096环境、seed42、追加500更新，预算至3000停止；远端RTX4090，mjlab1.6.0、MuJoCo/mujoco-warp3.11.0、torch2.14.0+cu130、Python3.12.3。

实际run根：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261012_231810_449023_height_curriculum`，两case独立子目录。启动北京时间2026-10-12 23:18:10。实际环境/PPO配置、完整源码快照与模型初始迁移证据保存在各新run，历史logs保留。

## 产物与验收状态

训练与评估正在运行，最终模型和SHA256、完成时间、课程级别及逐样本判定将在结束后补入 [训练与评估报告](../diagnostics/20261012-mjlab-height-curriculum-training.md)，此处不提前宣称成功。

| 验证阶段 | 状态 |
|---|---|
| 两组16/4096环境各64步有限 | 通过，数值失败0 |
| 奖励隔离与actor/critic/Adam完整迁移 | 通过 |
| A源行为同初态复现／v2成功集合保留 | 通过，普通64／地面60 |
| 2601/3000轮六高度、普通／地面回放 | 待完成 |
| 相对A原成功样本退化否决 | 待完成；新增不抵消丢失 |
| 20秒样本0视频 | 待完成 |
| Isaac旧入口回放 | 不适用，本实验原生mjlab |
| ONNX／H7板端／真机 | 未执行，未授权；H7代码未改 |

H7部署仓提交未核实且本次未涉及，现有部署接口不能直接加载该35维实体关节策略。形状检查、有限检查、静力可达及保留部分初态不等于真机验收。单seed、64初态只说明本批趋势。
