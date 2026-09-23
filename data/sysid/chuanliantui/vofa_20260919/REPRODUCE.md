# 系统辨识数据包

本目录是自包含的数据包：`raw/` 为不可改写的 VOFA 原始导出副本，`docs/` 为采集/辨识约定，`tools/` 为分析脚本；目录根部是脚本生成的派生结果。

从仓库根目录重新生成派生结果：

```bash
/home/zn/miniforge3/envs/wheellegged_py38/bin/python \
  data/sysid/chuanliantui/vofa_20260919/tools/analyze_chuanliantui_vofa_sysid.py \
  --hip-csv data/sysid/chuanliantui/vofa_20260919/raw/hip_vofa.csv \
  --wheel-csv data/sysid/chuanliantui/vofa_20260919/raw/wheel_vofa.csv \
  --output-dir data/sysid/chuanliantui/vofa_20260919
```

`hip_trajectory_replay.csv` 是闭链腿 real2sim 的参考、实测和下发力矩时间序列；MuJoCo 回放还需复刻固件控制律。该控制律说明文件未随本批资料提供，故不在本数据包内。
