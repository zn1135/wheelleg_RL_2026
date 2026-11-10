# 模型交付记录：2026-11-04共享闭链约束

本次交付物是仿真物理XML，未产生新的学习权重、ONNX或固件。

## 任务与接口

| 项目 | 内容 |
|---|---|
| 日期、交付人、接收人、独立复查人 | 2026-11-04；Codex；用户；独立复查/H7负责人待确认 |
| 机器人/任务 | chuanliantui_new_1当前六连接闭链；mjlab共享机器人及正式MuJoCo仿真 |
| 用途与验收目标 | 接入闭链连接及限位响应参数，保持标定范围；检查GPU动态连接间隙和旧策略回归 |
| 接口标识 | [shared-closed-chain-stability-20261104](../interfaces/20261104-shared-closed-chain-stability.md)，网络接口不变 |
| 关联PR | 无，用户未授权提交/推送 |

## 代码与来源

训练/部署同仓github.com/zn1135/wheelleg_RL_2026.git、26_wheelleg、HEAD c3c8fdffe0739aed314fb71c69f2a2d3135b3fb9，工作区存在既有修改。H7控制未变；1366个ROS固件来源哈希核对保持。HEAD不代表运行源码；本次增量、完整相关已跟踪差异、未跟踪源码和指纹见[产物目录](../../artifacts/shared_validation/closed-chain-training-integration-20261104-01/README.md)中的this-task.patch、workspace-source.diff、source/、source-sha256.json。

未执行训练，seed/run/优化器恢复/TensorBoard不适用。GPU验证使用2ms物理/PD、10ms策略，实际配置及CLI参数保存在源快照/日志；MuJoCo3.11.0、MuJoCoWarp3.11.0、PyTorch2.14.0，其他运行信息见summary.json。共享XML与全部运行模型mjb、初态随机化数组均保存。没有向Isaac旧环境安装或导入依赖。

## 模型与导出产物

| 产物 | 地址 | SHA256 | 说明 |
|---|---|---|---|
| 新物理XML | resources/robots/chuanliantui_new_1/mjlab/chuanliantui.xml | 505c9b9bb2b7a20e15cf9d2cd4ca664332db402687e34185d8a57de173dedbe8 | 六connect及限位约束；几何/质量/惯量/接触/气弹簧不变 |
| 旧物理XML | artifacts/shared_validation/closed-chain-training-integration-20261104-01/before-source.xml | 14fdae8d3046b7cccf6ae9feea27294c97719a98e3185e76616aa456ec79014a | 保留原字节 |
| 回归所用旧checkpoint | logs/mjlab/fudan_ppo_walk/20261102_parking_resume7000_to8000/model_final.pt | 39fa104d176f82a33db8317a75b4db28749362daf6bc4d2e06cb83d2c5ae32c9 | 8000-v2，原encoder/actor权重不变，停车+2°图外参考 |
| 新ONNX/板端网络/固件 | 不适用 | 不适用 | 无新网络或硬件交付，未重导出 |

## 分阶段验证

执行于当前工作区，2026-11-04完成；实际完成时间、源/资产hash、版本、退出码证据与日志见[验证目录](../../artifacts/shared_validation/closed-chain-training-integration-20261104-01/README.md)。

| 阶段 | 状态 | 依据和限制 |
|---|---|---|
| 新训练/优化器更新 | 未执行 | 用户要求同步设置，本次0训练更新 |
| Isaac回放 | 不适用 | 当前mjlab闭链，不改旧串联资产 |
| 接口/静态/标定范围 | 通过 | static-check：20,214姿态0越界；训练注册线与原生加载器实际编译到新值 |
| GPU接触闭链 | 通过 | 各20环境3s逐2ms，间隙2.777662→0.050962mm，无翻支路/限位超出；脚本动作不是行走验收 |
| GPU旧策略回归 | 诊断通过 | 20例8000+2°同初态/同4+4ms，统一技术通过19→20，无旧成功退化 |
| 起立/完整987/其他算法线行为 | 未执行 | 诊断不替代完整任务能力或所有训练线验收 |
| ONNX图/数值 | 不适用 | 网络接口和导出文件未改 |
| 板端时序/真机 | 未执行 | 无本次部署，H7负责人待确认 |

允许当前训练/仿真加载新XML，不将20例诊断写成完整任务或真机通过。原架空数值排序内核不移植；2ms积分保留refsafe，有效最小时间常数4ms。下一步完整GPU验收，随后由用户决定训练预算；回退/历史复现使用对应冻结XML和源，不改历史模型或hash。

补充范围：实际策略10ms采样六连接峰值4.055692→0.106436mm，略超0.1mm诊断目标；限位0超出、无翻支路。更硬第二候选峰值0.120856mm且旧成功diagnostic_9新增22ms连续起步触地，判退化并拒绝，最终恢复本记录505c9b9b…模型。不能把轻载接触通过和20例动作通过写成全部策略闭链精度或完整987验收通过。
