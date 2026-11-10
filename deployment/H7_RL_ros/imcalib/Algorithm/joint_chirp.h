/* 本次单右腿慢速定位目标/负载表查询；不访问遥控、驱动或电机输出。 */
#ifndef JOINT_CHIRP_H
#define JOINT_CHIRP_H

#include <stdint.h>

#define JOINT_CHIRP_TOTAL_MS 25000u

/* stage:1接管保持/2平滑定位/3终点保持；须先准备至240..280mm，目标190mm、腿角锁定接管实测值。
 * sign/zero只用于已有固件rad与模型rad映射；velocity模型rad/s、feedforward模型Nm。 */
uint8_t Joint_Pose_Target(uint32_t elapsed_ms, const float center[4], const int8_t sign[6],
                          const float zero[4], float lu, float lg, float target[4],
                          float velocity[4], float feedforward[4], uint8_t *stage);

#endif
