/*
 * model5999稳站v2输入与恢复模式：原生电机反馈、实际速度命令、五帧历史。
 * 位置/速度使用rad、rad/s；先缩放、拼接再整体裁剪。机器配置直接取固定表，不扫描配置合法性。
 * 执行任务保存最新反馈，策略任务复制后构建；不添加训练噪声或延迟，不调用网络或发送电机。
 */
#include "rl_observation.h"
#include "math_utils.h"
#include <math.h>
#include <string.h>
/* 阈值来自5999训练接口；W取同XML名义值，不宣称为实机实测重量。 */
#define RL_MODEL_WEIGHT_N 131.4008700f
#define RL_RECOVERY_HOLD_MS 200u
#define RL_RECOVERY_STABLE_MS 500u
#define RL_RECOVERY_TICK_MS ((uint16_t)(MACHINE_POLICY_DT * 1000.0f + 0.5f))
void RL_Recovery_Reset(rl_recovery_state_t *state)
{
    memset(state, 0, sizeof(*state));
}
/* 5999已解锁目标0.22m；拨轮不再改变本模型的训练高度命令。 */
float RL_Command_Height(float leg_length_command)
{
    (void)leg_length_command;
    return 0.22f;
}
/*
 * 每10ms推进一次模式/起立计时，失败须退出投入才复位。
 * 复用反馈估力，不读取LQR目标F或AIR飞行状态；高度/触地采用明确的实机代理。
 * 首次起立前允许长期恢复，不能沿旧multimode的2秒倒地超时终止起立。
 */
void RL_Recovery_Update(rl_recovery_state_t *state, uint8_t request, uint8_t feedback_valid, uint8_t fallen, float height, const float wheel_force[2], float gravity_z)
{
    uint8_t supported, down, stable, was_stood;
    const float minimum = fmaxf(0.1f * RL_MODEL_WEIGHT_N, 5.0f);
    if (!request) { RL_Recovery_Reset(state); return; }
    if (state->phase == RL_RECOVERY_FAILED) { return; }
    if (!feedback_valid) { RL_Recovery_Reset(state); return; }
    state->wheel_force[0] = wheel_force[0];
    state->wheel_force[1] = wheel_force[1];
    state->support_valid = 1u;
    supported = (uint8_t)(wheel_force[0] >= minimum && wheel_force[1] >= minimum && wheel_force[0] + wheel_force[1] >= 0.8f * RL_MODEL_WEIGHT_N);
    down = (uint8_t)(fallen || height <= 0.20f || !supported);
    if (state->phase == RL_RECOVERY_IDLE)
    {
        state->ground_start = 1u;
        state->phase = RL_RECOVERY_HOLD;
        return; /* 新投入t=0，和训练一样统一保持200ms零力矩。 */
    }
    if (state->hold_ms < RL_RECOVERY_HOLD_MS) { state->hold_ms += RL_RECOVERY_TICK_MS; }
    state->stable_ms = down ? 0u : (state->stable_ms < RL_RECOVERY_STABLE_MS ? state->stable_ms + RL_RECOVERY_TICK_MS : state->stable_ms);
    if (state->hold_ms >= RL_RECOVERY_HOLD_MS)
    {
        state->phase = state->stable_ms >= RL_RECOVERY_STABLE_MS ? RL_RECOVERY_NORMAL : RL_RECOVERY_RECOVER;
    }
    /* 无机身接触传感器：此处是姿态/几何高度/估算承重代理，须台架核对。 */
    was_stood = state->has_stood;
    stable = (uint8_t)(height >= 0.20f && gravity_z <= -0.9659258263f && supported);
    state->standing_ms = stable ? (state->standing_ms < 500u ? state->standing_ms + RL_RECOVERY_TICK_MS : state->standing_ms) : 0u;
    if (state->standing_ms >= 500u) { state->has_stood = 1u; }
    state->tracking_ms = state->has_stood && state->phase == RL_RECOVERY_NORMAL && !fallen ? (state->tracking_ms < 500u ? state->tracking_ms + RL_RECOVERY_TICK_MS : state->tracking_ms) : 0u;
    state->airborne_ms = was_stood && wheel_force[0] <= 1.0f && wheel_force[1] <= 1.0f ? state->airborne_ms + RL_RECOVERY_TICK_MS : 0u;
    state->fallen_ms = was_stood && (fallen || height <= 0.20f) ? state->fallen_ms + RL_RECOVERY_TICK_MS : 0u;
    if (was_stood && (gravity_z > 0.0f || state->airborne_ms >= 200u || state->fallen_ms > 1000u))
    {
        state->phase = RL_RECOVERY_FAILED;
        state->tracking_ms = 0u;
    }
}
/*
 * 固件→模型：首次有效样本按原零偏/sign转换，锁存使位置靠近q_reference的整圈偏移。
 * position_period与驱动计圈周期同源，锁存量只选初始表示，不重标零点。
 * 运行中只减同一偏移，不逐拍取模；跨编码器边界/参考分支切口都保留连续运动。
 * 反馈失效不改偏移，退出RL后由执行任务清initialized；观测和PD共用out->pos。
 * 不对PD角差另做环绕，速度/力矩映射保持原值。
 */
uint8_t RL_Motor_Map(const machine_rl_cfg_t *config, float position_period, const float position[4], const float velocity[4], const float wheel_velocity[2], uint8_t source_valid, rl_motor_map_state_t *state, rl_motor_state_t *out)
{
    out->valid = 0u;
    if (!source_valid || !config->configured) { return 0u; }
    for (uint8_t i = 0u; i < RL_LEG_MOTOR_NUM; i++)
    {
        out->pos[i] = (float)config->motor_sign[i] * (position[i] - config->motor_zero[i]);
        if (!state->initialized)
        {
            state->position_offset[i] = position_period * floorf((out->pos[i] - config->reference_pos[i]) / position_period + 0.5f);
        }
        out->pos[i] -= state->position_offset[i];
        out->vel[i] = (float)config->motor_sign[i] * velocity[i];
    }
    for (uint8_t i = 0u; i < 2u; i++)
    {
        out->vel[RL_MOTOR_L_WHEEL + i] = (float)config->motor_sign[RL_MOTOR_L_WHEEL + i] * wheel_velocity[i];
    }
    out->valid = (uint8_t)(Float_Array_Finite(out->pos, RL_LEG_MOTOR_NUM) && Float_Array_Finite(out->vel, RL_ACTION_SIZE));
    if (out->valid) { state->initialized = 1u; }
    return out->valid;
}
/* 四元数顺序w/x/y/z；归一化后将世界单位重力[0,0,-1]逆旋转到机体系。 */
uint8_t RL_Observation_Project_Gravity(const float quat[4], float gravity[3])
{
    float norm, w, x, y, z;
    norm = sqrtf(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
    w = quat[0] / norm;
    x = quat[1] / norm;
    y = quat[2] / norm;
    z = quat[3] / norm;
    gravity[0] = 2.0f * (w * y - x * z);
    gravity[1] = -2.0f * (w * x + y * z);
    gravity[2] = 2.0f * (x * x + y * y) - 1.0f;
    return Float_Array_Finite(gravity, 3u);
}
/* 退出RL或初始化时清观测、原始动作历史及最新样本。 */
void RL_Observation_Reset(rl_observation_state_t *state)
{
    memset(state, 0, sizeof(*state));
}
/* 每个RL控制子步保存未缩放反馈；源头online/valid保证原始量，新投影在这里检查。 */
void RL_Observation_Sample(rl_observation_state_t *state, const float gyro[3], const float quat[4], const rl_motor_state_t *motor, uint8_t source_valid)
{
    rl_observation_sample_t sample;
    if (!source_valid || !motor->valid || !RL_Observation_Project_Gravity(quat, sample.gravity))
    {
        state->sample_valid = 0u; /* 坏反馈不能继续借用上一帧。 */
        return;
    }
    memcpy(sample.gyro, gyro, sizeof(sample.gyro));
    memcpy(sample.pos, motor->pos, sizeof(sample.pos));
    memcpy(sample.vel, motor->vel, sizeof(sample.vel));
    state->sample = sample;
    state->sample_valid = 1u;
}
/* 调用方在短临界区复制最新有效样本，采样与推理之间仅保留实际任务相位。 */
uint8_t RL_Observation_Read_Sample(const rl_observation_state_t *state, rl_observation_sample_t *sample)
{
    if (!state->sample_valid) { return 0u; }
    *sample = state->sample;
    return 1u;
}
/* 100Hz构造一帧；5999为已解锁0.22m、yaw=0、速度第0档±0.5m/s。 */
uint8_t RL_Observation_Build(rl_observation_state_t *state, const machine_rl_cfg_t *config, const rl_observation_sample_t *sample, const rc_command_t *command, const rl_recovery_state_t *recovery, uint8_t source_valid)
{
    float target;
    memset(state->obs, 0, sizeof(state->obs));
    state->valid = 0u;
    if (!source_valid || recovery->phase == RL_RECOVERY_FAILED)
    {
        state->history_initialized = 0u;
        state->command_velocity = 0.0f;
        return 0u;
    }
    /* 只把实际限速命令送入历史；恢复/未起立时当拍归零，不走减速斜坡。 */
    if (recovery->phase == RL_RECOVERY_NORMAL && recovery->has_stood && recovery->tracking_ms >= 500u)
    {
        target = clip(command->vel, 0.5f);
        state->command_velocity += clip(target - state->command_velocity, 1.0f * MACHINE_POLICY_DT);
    }
    else
    {
        state->command_velocity = 0.0f;
    }
    state->obs[RL_OBS_CMD_VX] = state->command_velocity * 2.0f;
    state->obs[RL_OBS_CMD_YAW_RATE] = 0.0f;
    state->obs[RL_OBS_CMD_HEIGHT] = 0.22f * config->observation.height_scale;
    for (uint8_t i = 0u; i < 3u; i++)
    {
        state->obs[RL_OBS_GYRO_X + i] = sample->gyro[i] * config->observation.gyro_scale;
        state->obs[RL_OBS_GRAV_X + i] = sample->gravity[i];
    }
    for (uint8_t i = 0u; i < RL_LEG_MOTOR_NUM; i++)
    {
        state->obs[RL_OBS_MOTOR_POS + i] = sample->pos[i] - config->nominal_pos[i];
    }
    for (uint8_t i = 0u; i < RL_ACTION_SIZE; i++)
    {
        state->obs[RL_OBS_MOTOR_VEL + RL_Action_Index(i)] = sample->vel[i] * config->observation.joint_vel_scale;
        state->obs[RL_OBS_LAST_ACTION + i] = state->last_action[i];
    }
    state->obs[RL_OBS_CONTROL_MODE + (recovery->phase == RL_RECOVERY_NORMAL ? 0u : 3u)] = 1.0f;
    /* 训练先缩放、拼接再整体裁剪；不能套旧39维的缩放前速度裁剪。 */
    for (uint8_t i = 0u; i < RL_OBS_SIZE; i++)
    {
        state->obs[i] = clip(state->obs[i], config->observation.obs_clip);
    }
    if (!Float_Array_Finite(state->obs, RL_OBS_SIZE))
    {
        state->history_initialized = 0u;
        state->command_velocity = 0.0f;
        return 0u;
    }
    if (!state->history_initialized)
    {
        for (uint8_t i = 0u; i < RL_HISTORY_LENGTH; i++)
        {
            memcpy(&state->history[i * RL_OBS_SIZE], state->obs, sizeof(state->obs));
        }
        state->history_initialized = 1u;
    }
    else
    {
        memmove(state->history, &state->history[RL_OBS_SIZE], (RL_HISTORY_SIZE - RL_OBS_SIZE) * sizeof(float));
        memcpy(&state->history[RL_HISTORY_SIZE - RL_OBS_SIZE], state->obs, sizeof(state->obs));
    }
    state->valid = 1u;
    return 1u;
}
/* 保存网络请求本身；不能用PD目标、延迟后的目标或最终力矩替代。 */
void RL_Observation_Set_Last_Action(rl_observation_state_t *state, const float action[RL_ACTION_SIZE])
{
    memcpy(state->last_action, action, sizeof(state->last_action));
}
