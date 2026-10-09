# 气弹簧资产几何与力矩验证

后续 mjlab 迁移与正式资产左右更名已完成，见 [2026-10-09 迁移检查](20261009-mjlab-migration.md)。本文保留 10 月 8 日阶段的源码标识与验证范围。

## 版本与范围

- 日期：2026-10-08，Asia/Shanghai，约 20:14–20:20。
- 工作区：Wheel-Legged-Gym，分支 `26_wheelleg`。
- 基础提交：`b4aea50f2b75a730eff0d6a7d993454e98c3be51`，加本次未提交改动。
- 解释器：完整激活 `.env.local` 指定的 Python 3.8 Isaac Gym conda 环境；Isaac Gym 先于 Torch 导入。
- 目标：用户选择优化气弹簧物理建模，从资产读取端点和轴向；不增加电机前馈。
- 默认仍为每侧 150 N 恒定伸张力。真实气弹簧力—长度曲线、阻尼及行程未标定。

本次替换固定端点和按 `lf/rf` 名字推断轴向的计算，支持 joint origin RPY、任意
非零旋转轴、显式 joint/marker 名称映射及两侧不同力值。常量变换在初始化时缓存；
共轴机构不在每个子步重复计算髋角三角函数。没有运行性能基准，不声称吞吐提升。

旧 XML 下安装点与当前 URDF 存在约 `1.068e-9 m` 差异，最初的严格 float64
对照因此失败，最大力矩差约 `8.95e-8 N·m`。已按现有 URDF 生成器更新两份 XML
的四个气弹簧 site，保留原校验容差；此数值级差异不能解释起立或漂移问题。

## 已执行检查

所有命令在仓库根运行，使用上述已激活环境的 `python`，下表为最终代码的结果。

| 命令 | 退出码 | 结果与范围 |
|---|---|---|
| `python scripts/agent/check_gas_spring_asset_geometry.py` | 0 | 原始/训练 URDF、移动安装点、三维 RPY、非平行斜轴与归一化、左右改名、joint/link 不同名、两侧推力、退化拒绝；MuJoCo 最大双精度力矩差 `8.88e-15 N·m`；长度差分通过，CUDA 批量数学对照通过 |
| `python scripts/agent/check_chuanliantui_gas_spring.py` | 0 | 27 姿态 × 3 力值，float32/64 对照；最大差 `9.82e-7 N·m`；世界方向、推力合成、关闭清零、真实五子步 API 路径和电机限矩分离通过 |
| `python scripts/agent/check_chuanliantui_train_proxy.py` | 0 | 两气弹簧端点与闭链一致；六电机、tendon、0/150 N、长度差分和 16 步状态有限检查通过 |
| `python scripts/agent/check_chuanliantui_closed.py` | 0 | 结构、限位、两气弹簧、±360° 闭合及 5 秒无控制检查通过；最大销轴误差 `0.000813068208 m` |
| `git diff --check` | 0 | 无空白错误 |

任意轴/RPY 检查使用独立构建的实体髋/膝 MuJoCo 模型和 SciPy 安装旋转，
没有将被检查模块的力臂或端点结果作为 oracle。最初安装点变更检查使用了过大的
响应门槛，已调整为检测可辨认的非零力矩变化，MuJoCo 对照精度门槛未放宽。

## 数值源码标识

以下 SHA-256 对应通过检查的未提交源码及资产；本记录及说明文档不参与标识。

| 文件 | SHA-256 |
|---|---|
| `wheel_legged_gym/utils/chuanliantui_gas_spring.py` | `47403faa30b8b6a4f5ef419cd8338e21a71b6e8750fc65086bc14973218a4e71` |
| `wheel_legged_gym/envs/chuanliantui/chuanliantui.py` | `c8864c06b779502296029e4631650de403cb82c9c14b86a7f78b8348867a073d` |
| `sim2sim/chuanliantui.xml` | `dc784026bb2c09d9e8cd6d96963bd726a71a2a005daaba8371f330e5071267e8` |
| `sim2sim/chuanliantui_train_proxy.xml` | `ec06cee3b66ce35daecee4e8aa8b5becf6617f0695d39570f40cf09390431346` |
| `scripts/agent/check_gas_spring_asset_geometry.py` | `b0f2b3e43647a82e7fc80ad495204b9c1fe55f7cef5cf3b47a0e003011deae8b` |

## 未执行与边界

未启动训练、TensorBoard、Isaac 策略回放、MuJoCo 策略行为验收或真机测试。
5 秒无控制仿真只检查模型约束与有限性，不证明起立或静站成功。
本次仍使用历史串联训练环境；mjlab 迁移及实际 URDF 左右改名尚未实施。
数学模块的左右改名检查只使用临时资产，不修改正式机器人命名。
H7 固件与历史日志未修改，未执行 Git 提交。
