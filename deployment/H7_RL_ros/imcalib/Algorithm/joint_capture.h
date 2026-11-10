#ifndef JOINT_CAPTURE_H
#define JOINT_CAPTURE_H

#include <stdint.h>

typedef struct {
    float center[4];
    float target[4]; /* 已有模型坐标，遥测时转换回固件坐标。 */
    float integral[4]; /* 本次右腿PID积分贡献Nm，限幅8Nm；退出统一清零。 */
    float takeover[4]; /* 首拍模型Nm差值，5秒五次曲线退去；独立于积分，不扩大积分限幅。 */
    uint32_t start_ms;
    uint32_t elapsed_ms;
    uint8_t running, blocked, stage, axis;
    uint16_t saturated;
} joint_capture_t;

/* 本次只校验出力侧：沿用已标定右膝范围，340mm以内初态允许缓慢收回。 */
uint8_t Joint_Capture_Geometry(const float firmware_q[4], float lu, float lg);
/* 只向右腿返回固件坐标Nm，左腿恒零；原Kp/Kd加Ki20与冻结负载表。
 * valid失效/几何不合法锁止直到退出清状态，接口不发送电机命令。 */
uint8_t Joint_Capture_Update(joint_capture_t *state, uint32_t now_ms, uint8_t valid,
                            const float q[4], const float v[4], const float previous_torque[4],
                            const int8_t sign[6], const float zero[4], const float kp[4],
                            const float kd[4], float lu, float lg, float limit, float torque[4]);

#endif
