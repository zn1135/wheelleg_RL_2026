# r10 RTX4090 远端训练记录

2026-10-10，用户指定算力自由实例 vajnykxo-tp49pgnv 开始训练；按此前4096环境／1000轮预算执行。当前本机训练源码来自 `/home/zn/文档/Wheel-Legged-Gym`，26_wheelleg、HEAD `58947e8fabc4683b2025d42bda742b06c84dc566` 加未提交r10改动；不是实例说明所在旧worktree的训练版本。

## 配置与启动证据

- GPU：RTX4090 24GB；驱动580.126.09；独立Python3.12环境，Torch2.14.0+cu130。
- 新快照：`20261010T071246Z-58947e8-RkAOH0RvWJ`；119个训练源码／资产文件上传后SHA256与本机全部一致。
- 正式运行目录：`/root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_151622_292362_wheelleg_recover_r10_cloud_4096_1000`。
- 后台控制PID：1403；SSH断开后继续运行。
- Actor／Critic及优化器从新实例随机初始化，无resume或init_source。
- r10混合初态：每次复位0.5概率普通／地面；地面root零姿态、0.25m，无额外机体扰动。保留动态模式、四腿两轮顺序、零速度命令、0.27～0.34m目标高度和关闭膝惩罚。
- 截至2026-10-10T15:18:04.880177+08:00，已观察到迭代索引73/1000的PPO更新；`model_0.pt`已保存，累计训练进度为1，SHA256 `fe1fdc042661f8999b49dbb9cb7708c463fafe52b428535ccefb2153e970ffea`。
- 当时每轮约1.27秒，日志ETA约20分钟，仅为当时估计。没有观察到训练完成，也没有开展新模型行为评估。

实际生效训练命令（已启动，不要再次执行以创建重复训练）：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m ct_mjlab.cli train --num-envs 4096 --iterations 1000 --seed 42 --device cuda:0 --run-name wheelleg_recover_r10_cloud_4096_1000
```

TensorBoard命令，在远端另一终端执行；本次没有启动TensorBoard：

```bash
/root/gpufree-data/wheelleg-remote/runtime/mjlab/bin/python -m tensorboard.main --logdir /root/gpufree-data/wheelleg-remote/logs/mjlab/chuanliantui/20261010_151622_292362_wheelleg_recover_r10_cloud_4096_1000 --host 127.0.0.1 --port 8080
```

本机端口转发：

```bash
ssh -p 30702 -i /home/zn/.ssh/id_ed25519 -N -L 8080:127.0.0.1:8080 root@183.147.142.40
```

## 启动前检查和修正

本机r10的20环境可视预检见 [初态诊断](20261010-mjlab-r10-mixed-reset.md)。上传后在远端使用相同CLI检查32环境、128控制步，退出码0；明确只检查有限数值与接口，不等于策略行为通过。

远端首次预检因缺少EGL图形库失败，设置 `MUJOCO_GL=disable` 后通过。正式后台脚本同样设置此变量，并清除初始化checkpoint环境变量，保证随机权重起步；没有安装系统图形库或更改机器人物理参数。

修正本仓CLI的源码快照保存：本机仍从Git保存HEAD／分支／dirty源码；远端没有`.git`时读取上传脚本的`source-version.txt`，验证完整提交SHA并保存全部实际训练源码及文件SHA256。真实运行已成功保存source-state、interface、environment和runner记录，没有伪造Git元数据。

复用了用户指定worktree的SSH配置及上传脚本；只在临时启动器中把源码根目录指向当前工作区，未切换分支或复制旧训练配置。默认1024环境／20000轮由命令尾部4096／1000参数覆盖，实际运行快照已核实。

## 日志与产物

控制台：`/root/gpufree-data/wheelleg-remote/releases/20261010T071246Z-58947e8-RkAOH0RvWJ/.remote-job/output.log`。
退出码文件：同目录`exit-code`，运行时尚不存在，结束后由后台脚本生成。
完整版本、初始checkpoint校验值、当时进度及控制台摘要见 [记录JSON](20261010-mjlab-r10-cloud-training.json)。

本次未执行Git提交；未改动H7代码；未覆盖或删除本机及远端历史logs。按实例原设置，训练结束不会自动关机。

## 完成与效果更新

2026-10-10 15:37:20（Asia/Shanghai）正常结束，退出码0，1000轮、98,304,000采样步，训练耗时20分44秒。最终model_999 SHA256为`5514f88c186fb3d6ff8855fe0ba00708f08a276f4afe67fe355817c4c71a5240`，已下载到本机新目录并校验一致。

CPU及原生GPU回放均未通过严格起立／站高验收，主要维持约0.20 m低姿态。完整结果见 [效果评估](20261010-mjlab-r10-cloud-model999-evaluation.md)，训练JSON已附完成和评估记录。上文73轮及尚未结束的状态属于启动时历史快照。
