# 高频小幅独立验证采集接口

本轮只更新H7_RL_ros采集轨迹，模型侧验证原低频等效负载/摩擦/阻尼参数，不更新训练、气弹簧或实体映射。H7负责人核对待确认。

原策略7变为仅0.05rad、2→5Hz、四轴各30秒共120秒。每轴5秒保持/20秒包络扫频/5秒保持，结束保持中心；左下退出，左中右下原10mm准备保留，左中右中启动。接管/2ms PD60-2/40Nm/零轮矩和既有保护保留，无新增实际速度门控。

H7OUT001/88字节记录不变，类型5 flags bit2新增small-high-v1身份，活动stage2，完成stage3；其余位保持。新prepare_pace_capture必须核该标志及完整配对/时序，拒绝旧数据；新replay_pace_capture_gpu --segment small-high按elapsed从0开始的stage2四段回放，固定原参数，不再拟合。页面按bit2显示120秒及幅频。

C数值/任务/RC/新旧打包/传输/新导出检查、大小机编译和合成GPU连续120秒预检通过，已取得本次动力断开确认并烧录/校验/复位，USB新会话冷态请求0；已完成独立四轴120秒采集并封存，60000阶段2样本质量通过；固定参数GPU回放显示频响振幅偏大，未做参数再拟合。[结果](../../artifacts/pace-joint-capture/20261109-small-high-validation-01/README.md)。[候选清单和证据](../../artifacts/pace-joint-capture/20261109-small-high-release-01/README.md)。
