#include "joint_capture.h"
#include "joint_chirp.h"
#include <math.h>
#include <string.h>

uint8_t Joint_Capture_Geometry(const float q[4], float lu, float lg)
{
    /* 本次仅右腿出力，左腿零矩，不以未参与定位的左腿工作域锁止右腿。 */
    for (uint8_t side = 1u; side < 2u; side++)
    {
        float a = q[2u * side] + 3.141592653589793f;
        float b = q[2u * side + 1u];
        float ax = lu * cosf(a), ay = lu * sinf(a);
        float bx = lu * cosf(b), by = lu * sinf(b);
        float dx = bx - ax, dy = by - ay;
        float distance = hypotf(dx, dy);
        float h2 = lg * lg - 0.25f * distance * distance;
        float px, py, length, knee;
        if (!isfinite(distance) || distance < 1e-5f || h2 <= 0.0f) { return 0u; }
        px = 0.5f * (ax + bx) - sqrtf(h2) * dy / distance;
        py = 0.5f * (ay + by) + sqrtf(h2) * dx / distance;
        length = hypotf(px, py);
        /* 共享XML的lf1/rf1局部铰角表达式；不改变编码器零点或驱动极性。 */
        knee = remainderf(atan2f(py - ay, px - ax) - a, 6.283185307179586f) + 1.6260029368f;
        if (side != 0u) { knee = -knee; }
        if (!isfinite(knee) || !isfinite(length) || length < 0.14f || length > 0.34f) { return 0u; }
        if (side == 0u && (knee < -0.97f || knee > 0.14f)) { return 0u; }
        if (side != 0u && (knee < -0.16f || knee > 1.01f)) { return 0u; }
    }
    return 1u;
}

uint8_t Joint_Capture_Update(joint_capture_t *state, uint32_t now_ms, uint8_t valid,
                            const float q[4], const float v[4], const float previous[4],
                            const int8_t sign[6], const float zero[4], const float kp[4],
                            const float kd[4], float lu, float lg, float limit, float torque[4])
{
    float velocity[4], feedforward[4];
    float takeover_t, takeover_weight;
    uint8_t starting = (uint8_t)!state->running;
    memset(torque, 0, 4u * sizeof(float));
    if (!valid || !isfinite(limit) || limit <= 0.0f) { state->blocked = 1u; }
    for (uint8_t i = 0u; i < 4u; i++)
    {
        if (!isfinite(q[i]) || !isfinite(v[i]) || !isfinite(kp[i]) || !isfinite(kd[i]) ||
            kp[i] <= 0.0f || kd[i] < 0.0f || !isfinite(zero[i]) || (sign[i] != 1 && sign[i] != -1))
        { state->blocked = 1u; }
    }
    if (state->blocked) { return 0u; }
    if (!state->running)
    {
        for (uint8_t i = 0u; i < 4u; i++)
        {
            if (!isfinite(previous[i])) { state->blocked = 1u; return 0u; }
            state->center[i] = sign[i] * (q[i] - zero[i]);
        }
        state->start_ms = now_ms;
        state->running = 1u;
    }
    state->elapsed_ms = now_ms - state->start_ms;
    state->axis = 2u; /* 右腿两个电机同时定位，不是单轴扫频。 */
    if (!Joint_Pose_Target(state->elapsed_ms, state->center, sign, zero, lu, lg, state->target, velocity, feedforward, &state->stage))
    { state->blocked = 1u; return 0u; }
    takeover_t = fminf((float)state->elapsed_ms / 5000.0f, 1.0f);
    takeover_weight = 1.0f - takeover_t * takeover_t * takeover_t * (10.0f - 15.0f * takeover_t + 6.0f * takeover_t * takeover_t);
    state->saturated = 0u;
    for (uint8_t i = 2u; i < 4u; i++)
    {
        float error = state->target[i] - sign[i] * (q[i] - zero[i]);
        float value;
        /* 首拍延续原请求矩；差值5秒平滑退去，积分从零开始且仍限8Nm。 */
        if (starting)
        {
            state->takeover[i] = sign[i] * (previous[i] + kd[i] * v[i]) - feedforward[i];
            state->integral[i] = 0.0f;
        }
        value = feedforward[i] + kp[i] * error + kd[i] * (velocity[i] - sign[i] * v[i]) + state->integral[i] + takeover_weight * state->takeover[i];
        if (!isfinite(value)) { state->blocked = 1u; memset(torque, 0, 4u * sizeof(float)); return 0u; }
        if (fabsf(value) > limit) { state->saturated |= (uint16_t)(1u << i); }
        torque[i] = sign[i] * fminf(fmaxf(value, -limit), limit);
        /* 2ms原执行节拍，积分增益20；饱和且误差继续推向饱和方向时暂停积分。 */
        if (fabsf(value) < limit || value * error < 0.0f) { state->integral[i] = fminf(fmaxf(state->integral[i] + 20.0f * error * 0.002f, -8.0f), 8.0f); }
    }
    return 1u;
}
