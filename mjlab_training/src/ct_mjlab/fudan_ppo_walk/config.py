"""复旦行走任务配置：修改这里的开关和参数后再运行。"""


class FudanWalkCfg:
    class domain_rand:
        # 额外动作延迟；范围来自复旦参考，并非本机实测。
        randomize_action_delay = False
        delay_ms_range = [0., 10.]

        # 额外传感数据延迟；开启前填写已确定的毫秒范围。
        randomize_sensor_delay = False
        sensor_delay_ms_range = None

        # 与复旦配置同名：主动推扰开关、间隔和冲量速度尺度。
        push_robots = False
        push_interval_s = 7.
        max_push_vel_xy = 2.
