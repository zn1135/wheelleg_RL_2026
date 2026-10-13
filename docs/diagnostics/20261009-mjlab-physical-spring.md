# mjlab 实体气弹簧修正与检查

日期：2026-10-09；分支 `26_wheelleg`；基础提交 `58947e8fabc4683b2025d42bda742b06c84dc566` 加本次未提交改动。独立复查未执行。

## 问题与修正

用户通过图形预检指出此前气弹簧错误，要求对照本机 `mjlab/wheelleg_RL`。此前r1仅使用端点tendon与恒定150 N，保留了CAD定位辅助杆，没有实体筒体／活塞杆、其质量、伸缩坐标和压缩／阻尼力律。初始销轴残差小只能说明腿闭合，不能证明弹簧建模正确。

本次移植参考工程的两个实体弹簧、0～150 N力律、10 mm预压及两个清理网格。新增资产与旧sim2sim XML分开，旧模型和历史日志保留；训练侧CPU和GPU共用新资产与力律。参考工程仅被读取，其未提交改动保留。来源和原始／清理网格哈希见 `resources/robots/chuanliantui_new_1/mjlab/spring-source.json`。

六电机顺序、35／78维观测、50 Hz策略、Kp10／Kd1和V14 flat v0 PPO保持现有定义。接口升级r2，拒绝旧恒力版本checkpoint。部署源码未修改。

## 已执行检查

运行目录为训练仓根；使用本仓Python 3.12 / mjlab环境。离线姿态生成退出0，CPU/GPU有限检查退出0：

```bash
mjlab_training/.venv/bin/python scripts/agent/generate_chuanliantui_mjlab_pose.py
PYTHONPATH=mjlab_training/src mjlab_training/.venv/bin/python scripts/agent/check_chuanliantui_mjlab.py --gpu --out /tmp/chuanliantui-physical-spring-check-20261009.json
```

详细结果保存于 [JSON记录](20261009-mjlab-physical-spring-check.json)。

| 检查 | 结果与范围 |
|---|---|
| 原生模型结构 | 19关节、6组connect、0个tendon、8执行器 |
| reset／nominal弹簧闭合 | 两侧弹簧末端距离均小于1e−8 m |
| 气弹簧名义曲线与限幅 | 压缩0／45／90 mm对应0／75／150 N；伸缩阻尼及弹性力先限幅顺序通过 |
| 与指定参考力律数值对照 | 2048组位移／速度，NumPy和Torch最大误差均0 N，零偏置 |
| CPU有限物理 | 60策略拍；六处销轴／弹簧末端最大误差0.001610356 m；弹簧输出1.614195～41.195530 N，滑动关节实际广义力与执行器力一致 |
| GPU有限物理 | 4环境、64策略拍；六处最大误差0.001751901 m；状态与奖励有限，无数值异常重置掩盖；弹簧输出在0～150 N |
| 源奖励对照 | 最大绝对误差9.313226e−10 |
| 随机未训练actor ONNX | PyTorch／ONNX Runtime最大误差1.043081e−7，临时模型清理 |
| PPO runner | 仅构造，未执行learn或优化器更新 |
| 图形可视化 | 实际mjlab 20环境、显示环境0，初始暂停；筒体与活塞杆可见，左右分别高亮；保存初态左右／斜视截图 |
| 初态原生图形状态 | 闭合约束最大分量残差1.589839e−12 m；轴向长度约219.204300 mm；两侧零速度推力约16.666668 N |

图形窗口支持空格运行／暂停、N单步、R复位、T透明、1／2／3切换视角、Q退出。无策略模型时使用零请求动作和现有PD；运行中终止则自动复位并暂停。该响应不证明学会起立；用户人工检查结论待反馈。

截图由实际mjlab模型渲染，保存在 `/tmp/chuanliantui-mjlab-physical-spring-20261009/`，未写入历史logs。临时查看器不会改变资产参数：只调整筒体颜色、标记大小和相机。

## 尚未验证

未训练策略，未执行新策略30秒起立／静站验收，未编译固件或进行真机验证。10 mm预压、0～150 N曲线和参考质量不是本次实物测量；主腿合计惯量可能包含CAD定位辅助件贡献，需要后续CAD／实测资料校准。有限物理检查不证明真实气弹簧模型已标定或策略行为通过。
