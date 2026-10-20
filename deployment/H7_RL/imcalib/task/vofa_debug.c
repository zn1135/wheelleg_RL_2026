/*
 * 文件职责：固定48通道赋值与VOFA发送（vofa_debug.c）。
 * 输入/来源：共享反馈/控制/离地/补偿诊断及驱动记录。
 * 输出/去向：后台每次调用固定赋值后提交一帧；改通道时手动注释旧赋值、启用新赋值。
 * 关键边界：只读观测，不推进检测或修改目标；异步来源同帧不等于同一物理采样时刻，通道表见md/file-guide.md#vofa。
 * 阅读入口：md/file-guide.md；参数当前值以源码为准。
 */

#include "vofa_debug.h"
#include "robot_state.h"
#include "dm.h"
#include "Vofa_send.h"
#include "dr16.h"
#include "jetson_uart.h"
#include "FreeRTOS.h"
#include "task.h"

#include <math.h>
#include <string.h>

/* CRC16/CCITT-FALSE，校验线上小端float原始字节；结果用float数值传输。 */
static uint16_t Mirror_Crc16(const uint8_t *data, uint16_t size)
{
    uint16_t crc = 0xFFFFu;
    for (uint16_t i = 0u; i < size; i++)
    {
        crc ^= (uint16_t)data[i] << 8;
        for (uint8_t bit = 0u; bit < 8u; bit++)
        {
            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u) : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

/* uint32拆低/高16位，不能将完整序号或毫秒直接转换成float。 */
static void Mirror_U32(float *out, uint32_t value)
{
    out[0] = (float)(value & 0xFFFFu);
    out[1] = (float)(value >> 16);
}

/* 软件发布年龄不是传感器采样年龄；-1表示从未发布，超过60秒饱和。 */
static float Mirror_Publication_Age(const feedback_meta_t *meta, uint64_t captured_ns)
{
    uint64_t age_ms;
    if (!meta->seen)
    {
        return -1.0f;
    }
    age_ms = (captured_ns - meta->published_ns) / 1000000u;
    return (age_ms > 60000u) ? 60000.0f : (float)age_ms;
}

/*
 * 只读镜像：未投入RL也发送公共反馈；约100Hz尝试、DMA忙不等待。
 * valid位0=IMU，1~4=四髋，5~6=两轮；无效来源报文填0，不修改原反馈或控制许可。
 * seq只计算DMA提交成功的帧；软件快照仍包含异步来源，不能称为同步采样。
 */
void Vofa_Debug_Send(void)
{
    static robot_feedback_t feedback;
    static float dbg[VOFA_MAX_CH];
    static uint32_t last_attempt_ms;
    static uint32_t sequence;
    uint32_t now_ms = HAL_GetTick();
    uint32_t candidate = sequence + 1u;
    uint16_t state_bits;
    uint8_t online_mask;
    uint8_t valid_mask = 0u;

    if ((uint32_t)(now_ms - last_attempt_ms) < 10u)
    {
        return;
    }
    last_attempt_ms = now_ms;
    (void)Robot_Feedback_Read(&feedback);
    memset(dbg, 0, sizeof(dbg));
    online_mask = feedback.imu.online ? 0x01u : 0x00u;
    online_mask |= feedback.rc.online ? 0x02u : 0x00u;
    for (uint8_t i = 0u; i < DM_MOTOR_NUM; i++)
    {
        online_mask |= feedback.motor.dm.online[i] ? (uint8_t)(1u << (i + 2u)) : 0u;
        /* 打包边界只保证报文字段有限，不裁剪或回写实机角度。 */
        if (feedback.motor.dm.online[i] && isfinite(feedback.motor.dm.pos_continuous_rad[i]) && isfinite(feedback.motor.dm.vel_rad_s[i]))
        {
            dbg[10u + i] = feedback.motor.dm.pos_continuous_rad[i];
            dbg[14u + i] = feedback.motor.dm.vel_rad_s[i];
            valid_mask |= (uint8_t)(1u << (i + 1u));
        }
    }
    for (uint8_t i = 0u; i < DJI_MOTOR_NUM; i++)
    {
        online_mask |= feedback.motor.dji.online[i] ? (uint8_t)(1u << (i + 6u)) : 0u;
        if (feedback.motor.dji.online[i] && isfinite(feedback.motor.dji.angle_total_rad[i]) && isfinite(feedback.motor.dji.vel_rad_s[i]))
        {
            dbg[18u + i] = feedback.motor.dji.angle_total_rad[i];
            dbg[20u + i] = feedback.motor.dji.vel_rad_s[i];
            valid_mask |= (uint8_t)(1u << (i + 5u));
        }
    }
    if (feedback.imu.online)
    {
        valid_mask |= 0x01u;
        for (uint8_t i = 0u; i < 4u; i++)
        {
            dbg[22u + i] = feedback.imu.quat[i];
        }
        for (uint8_t i = 0u; i < 3u; i++)
        {
            dbg[26u + i] = feedback.imu.gyro_rad_s[i];
        }
    }
    taskENTER_CRITICAL();
    state_bits = robot_state.motor_enabled ? 0x01u : 0x00u;
    state_bits |= robot_state.fallen ? 0x02u : 0x00u;
    state_bits |= feedback.left.output.valid ? 0x04u : 0x00u;
    state_bits |= feedback.right.output.valid ? 0x08u : 0x00u;
    state_bits |= Dm_Is_Enabled(DM_MOTOR_LEG_F_LFT) ? 0x10u : 0x00u;
    state_bits |= Dm_Is_Enabled(DM_MOTOR_LEG_B_LFT) ? 0x20u : 0x00u;
    state_bits |= Dm_Is_Enabled(DM_MOTOR_LEG_F_RGT) ? 0x40u : 0x00u;
    state_bits |= Dm_Is_Enabled(DM_MOTOR_LEG_B_RGT) ? 0x80u : 0x00u;
    state_bits |= output_task_lqr_engaged() ? 0x100u : 0x00u;
    state_bits |= output_task_rl_engaged() ? 0x200u : 0x00u;
    dbg[2] = (float)ctrl_fault;
    /* 通信探测回报：只复制序号/随机数/计数，不影响动作或控制许可。 */
    dbg[32] = (float)JETSON_UART_PROTOCOL;
    Mirror_U32(&dbg[33], jetson_uart_status.sequence);
    Mirror_U32(&dbg[35], jetson_uart_status.token);
    Mirror_U32(&dbg[37], jetson_uart_status.received);
    taskEXIT_CRITICAL();
    dbg[0] = (float)online_mask;
    dbg[1] = (float)state_bits;
    dbg[3] = 7101.0f;
    dbg[4] = 1.0f;
    dbg[5] = (float)MACHINE_DEFAULT;
    Mirror_U32(&dbg[6], candidate);
    Mirror_U32(&dbg[8], (uint32_t)(feedback.captured_ns / 1000000u));
    dbg[29] = (float)valid_mask;
    dbg[30] = Mirror_Publication_Age(&feedback.imu_meta, feedback.captured_ns);
    dbg[31] = Mirror_Publication_Age(&feedback.motor_meta, feedback.captured_ns);
    dbg[47] = (float)Mirror_Crc16((const uint8_t *)dbg, 47u * sizeof(float));
    if (Vofa_Send(dbg, VOFA_MAX_CH))
    {
        sequence = candidate;
    }
}

/* 原LQR/RL布局完整保留；手动切回时注释上面的镜像函数，并取消下方旧函数的行注释。 */
// void Vofa_Debug_Send(void)
// {
//     static float dbg[VOFA_MAX_CH];
//     uint8_t online_mask;
//     uint16_t state_bits;
//
//     memset(dbg, 0, sizeof(dbg));
//     /* ch0 在线掩码 */
//     online_mask  = imu_state.online ? 0x01u : 0x00u;
//     online_mask |= DR16_Online() ? 0x02u : 0x00u;
//     online_mask |= motor_state.dm.online[0] ? 0x04u : 0x00u;
//     online_mask |= motor_state.dm.online[1] ? 0x08u : 0x00u;
//     online_mask |= motor_state.dm.online[2] ? 0x10u : 0x00u;
//     online_mask |= motor_state.dm.online[3] ? 0x20u : 0x00u;
//     online_mask |= motor_state.dji.online[0] ? 0x40u : 0x00u;
//     online_mask |= motor_state.dji.online[1] ? 0x80u : 0x00u;
//     dbg[0] = (float)online_mask;
//
//     /* ch1 状态位: 使能/跌倒/左腿有效/右腿有效/四髋使能/已投入 */
//     state_bits  = robot_state.motor_enabled ? 0x01u : 0x00u;
//     state_bits |= robot_state.fallen ? 0x02u : 0x00u;
//     state_bits |= leg_l.output.valid ? 0x04u : 0x00u;
//     state_bits |= leg_r.output.valid ? 0x08u : 0x00u;
//     state_bits |= Dm_Is_Enabled(DM_MOTOR_LEG_F_LFT) ? 0x10u : 0x00u;
//     state_bits |= Dm_Is_Enabled(DM_MOTOR_LEG_B_LFT) ? 0x20u : 0x00u;
//     state_bits |= Dm_Is_Enabled(DM_MOTOR_LEG_F_RGT) ? 0x40u : 0x00u;
//     state_bits |= Dm_Is_Enabled(DM_MOTOR_LEG_B_RGT) ? 0x80u : 0x00u;
//     state_bits |= output_task_lqr_engaged() ? 0x100u : 0x00u;
//     state_bits |= output_task_rl_engaged() ? 0x200u : 0x00u;
//     dbg[1] = (float)state_bits;
//     dbg[2] = (float)ctrl_fault;
//
//     /* 旧腿部通道，保留为手动改布局时的参考。 */
//     // dbg[3] = (float)leg_l.input.hip_f;
//     // dbg[4] = (float)leg_l.input.hip_b;
//     // dbg[5] = (float)leg_r.input.hip_f;
//     // dbg[6] = (float)leg_r.input.hip_b;
//     // dbg[7] = (float)leg_l.output.virtual_leg_length;
//     // dbg[8] = (float)leg_l.output.virtual_leg_angle;
//     // dbg[9] = (float)leg_l.output.virtual_shank_angle;
//     // dbg[10] = (float)leg_r.output.virtual_leg_length;
//     // dbg[11] = (float)leg_r.output.virtual_leg_angle;
//     // dbg[12] = (float)leg_r.output.virtual_shank_angle;
//     // dbg[13] = (float)leg_l.output.d_virtual_leg_length;
//     // dbg[14] = (float)leg_l.output.d_virtual_leg_angle;
//     // dbg[15] = (float)leg_l.output.d_virtual_shank_angle;
//     // dbg[16] = (float)leg_r.output.d_virtual_leg_length;
//     // dbg[17] = (float)leg_r.output.d_virtual_leg_angle;
//     // dbg[18] = (float)leg_r.output.d_virtual_shank_angle;
//
//     /* 一次复制LQR状态、总力矩和分项，避免控制任务在复制中途推进一拍。 */
//     taskENTER_CRITICAL();
//     for(int i = 0;i<10;i++)
//     {
//         dbg[i + 3] = lqr_state.x[i];
//     }
//     for(int i = 0;i<10;i++)
//     {
//         dbg[i + 13] = lqr_state.target[i];
//     }
//     dbg[23] = lqr_state.u[0];
//     dbg[24] = lqr_state.u[1];
//     dbg[25] = lqr_state.u[2];
//     dbg[26] = lqr_state.u[3];
//     for (uint8_t i = 0u; i < LQR_X_NUM; i++)
//     {
//         dbg[27u + i] = lqr_state.u_col[i];
//         dbg[37u + i] = lqr_state.u_hip_col[i];
//     }
//     dbg[47] = (float)lqr_state.air_control;
//     taskEXIT_CRITICAL();
//     // dbg[23] = lqr_state.a_fwd;
//
//     /* RL布局参考：obs39及恢复诊断，切换观测时手动启用。 */
//     // taskENTER_CRITICAL();
//     // for (uint8_t i = 0u; i < RL_OBS_SIZE; i++)
//     // {
//     //     dbg[3u + i] = rl_control.observation.obs[i];
//     // }
//     // // dbg[35] = (float)motor_state.dm.pos_continuous_rad[0];
//     // // dbg[36] = (float)motor_state.dm.pos_continuous_rad[1];
//     // // dbg[37] = (float)motor_state.dm.pos_continuous_rad[2];
//     // // dbg[38] = (float)motor_state.dm.pos_continuous_rad[3];
//     // dbg[42] = (float)rl_control.policy.run_us;
//     // dbg[43] = (float)rl_control.recovery.phase; /* 0未投入/1启动保持/2恢复/3正常/4失败。 */
//     // dbg[44] = robot_state.height;
//     // dbg[45] = (float)robot_state.height_valid;
//     // dbg[46] = (float)rl_control.recovery.stable_ms;
//     // dbg[47] = (float)rl_control.recovery.fallen_ms;
//     // taskEXIT_CRITICAL();
//
//     Vofa_Send(dbg, VOFA_MAX_CH);
// }
