#include "joint_chirp.h"
#include "joint_pose_load.h"
#include <math.h>

/* 本次右腿只改腿长：保持5秒，20秒五次曲线到190mm，腿角锁定接管实测值。
 * center/target/velocity为原模型坐标rad/rad/s；只消费现有sign/zero，不改实体标定。
 * 左腿目标仅作记录，不参与输出；本函数不访问遥控或驱动。 */
uint8_t Joint_Pose_Target(uint32_t elapsed_ms, const float center[4], const int8_t sign[6],
                          const float zero[4], float lu, float lg, float target[4],
                          float velocity[4], float feedforward[4], uint8_t *stage)
{
    float front = sign[2] * center[2] + zero[2];
    float rear = sign[3] * center[3] + zero[3];
    float half = 0.5f * (front - rear + 3.141592653589793f);
    float start_angle = -0.5f * (front + rear);
    float start_length = lu * cosf(half) + sqrtf(lg * lg - lu * lu * sinf(half) * sinf(half));
    float t = fminf(fmaxf(((float)elapsed_ms - 5000.0f) / 20000.0f, 0.0f), 1.0f);
    float blend = t * t * t * (10.0f - 15.0f * t + 6.0f * t * t);
    float rate = 30.0f * t * t * (1.0f - t) * (1.0f - t) / 20.0f;
    float length = start_length + blend * (0.190f - start_length);
    float angle = start_angle;
    float cosine, spread_rate, x, y, fx, fy;
    uint8_t ix, iy;

    /* 长腿零矩接管预检未通过；先用原10mm模式准备，禁止从>280mm直接定位。 */
    if (!isfinite(start_length) || !isfinite(start_angle) || start_length < 0.240f || start_length > 0.280f || start_angle < -0.174532925199433f || start_angle > 0.698131700797732f) { return 0u; }
    cosine = (length * length + lu * lu - lg * lg) / (2.0f * lu * length);
    if (!isfinite(cosine) || fabsf(cosine) >= 1.0f) { return 0u; }
    spread_rate = -(1.0f - (lu * lu - lg * lg) / (length * length)) * rate * (0.190f - start_length) / (2.0f * lu * sqrtf(1.0f - cosine * cosine));
    for (uint8_t i = 0u; i < 4u; i++) { target[i] = center[i]; velocity[i] = 0.0f; feedforward[i] = 0.0f; }
    target[2] = sign[2] * (-angle + acosf(cosine) - 1.570796326794897f - zero[2]);
    target[3] = sign[3] * (-angle - acosf(cosine) + 1.570796326794897f - zero[3]);
    velocity[2] = sign[2] * spread_rate;
    velocity[3] = -sign[3] * spread_rate;
    /* 冻结MuJoCo静态负载表双线性插值，输出模型Nm；不再叠加其他补偿。 */
    x = fminf(fmaxf((length - 0.180f) * 100.0f, 0.0f), 16.0f);
    y = fminf(fmaxf((angle + 0.174532925199433f) / 0.0872664625997165f, 0.0f), 10.0f);
    ix = (uint8_t)fminf(floorf(x), 15.0f); iy = (uint8_t)fminf(floorf(y), 9.0f);
    fx = x - ix; fy = y - iy;
    for (uint8_t i = 0u; i < 2u; i++)
    {
        feedforward[i + 2u] = (1.0f - fx) * ((1.0f - fy) * joint_pose_load[ix][iy][i] + fy * joint_pose_load[ix][iy + 1u][i]) + fx * ((1.0f - fy) * joint_pose_load[ix + 1u][iy][i] + fy * joint_pose_load[ix + 1u][iy + 1u][i]);
    }
    *stage = elapsed_ms < 5000u ? 1u : (elapsed_ms < JOINT_CHIRP_TOTAL_MS ? 2u : 3u);
    return 1u;
}
