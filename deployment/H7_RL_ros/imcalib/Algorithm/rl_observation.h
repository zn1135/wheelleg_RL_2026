/* 稳站v2输入：500Hz原生反馈；100Hz构造32维当前帧和五帧历史。 */
#ifndef RL_OBSERVATION_H
#define RL_OBSERVATION_H
#include "rl_types.h"
#include "machine_config.h"
#include "rc_types.h"
/* RL实际投入阶段；数值供Watch查看，不直接作为网络模式编号。 */
typedef enum {
    RL_RECOVERY_IDLE = 0, RL_RECOVERY_HOLD, RL_RECOVERY_RECOVER, RL_RECOVERY_NORMAL, RL_RECOVERY_FAILED
} rl_recovery_phase_t;
typedef struct {
    rl_recovery_phase_t phase;
    uint8_t has_stood;            /* 已连续满足首次起立条件，投入段内锁存。 */
    uint16_t tracking_ms;          /* 首次起立后NORMAL额外连续500ms才接受速度。 */
    uint8_t ground_start;          /* 实际投入时锁存，站起/中途倒地均不改写。 */
    uint16_t hold_ms;              /* 趴地启动零力矩保持200ms。 */
    uint16_t stable_ms;            /* 连续非倒地500ms后完成恢复。 */
    uint16_t fallen_ms;            /* 首次站起后连续倒地超过1000ms锁定失败。 */
    uint16_t standing_ms;          /* 15度/高度/双轮承重连续500ms锁存has_stood。 */
    uint16_t airborne_ms;          /* 站起后双轮均无支持连续200ms锁定失败。 */
    float wheel_force[2];          /* 独立AIR反馈估力N；不是LQR目标力。 */
    uint8_t support_valid;         /* 本次反馈估力有效，不表示已承重。 */
} rl_recovery_state_t;
/* 32维帧的顺序与训练一致；速度和上一动作按网络左右两侧排列。 */
typedef enum {
    RL_OBS_GYRO_X = 0, RL_OBS_GYRO_Y, RL_OBS_GYRO_Z,
    RL_OBS_GRAV_X, RL_OBS_GRAV_Y, RL_OBS_GRAV_Z,
    RL_OBS_CMD_VX, RL_OBS_CMD_YAW_RATE, RL_OBS_CMD_HEIGHT,
    RL_OBS_MOTOR_POS = 9, RL_OBS_MOTOR_VEL = 13, RL_OBS_LAST_ACTION = 19,
    RL_OBS_CONTROL_MODE = 25, RL_OBS_JUMP_HEIGHT = 30, RL_OBS_PHASE_TIME
} rl_obs_index_t;
/* 一份模型坐标反馈，未进行观测缩放或减名义角。 */
typedef struct {
    float gyro[3];                  /* 机体XYZ角速度，rad/s */
    float gravity[3];               /* 机体系单位重力 */
    float pos[RL_LEG_MOTOR_NUM];     /* 四电机模型角分支，rad；观测与PD共用 */
    float vel[RL_ACTION_SIZE];      /* 四电机与左右轮速度，rad/s */
} rl_observation_sample_t;
typedef struct {
    float obs[RL_OBS_SIZE];         /* 当前帧，整体按训练裁剪到±100 */
    float history[RL_HISTORY_SIZE]; /* 唯一网络输入；旧→新，末32维等于obs */
    float command_velocity;        /* 限斜率后的实际前向命令，m/s */
    uint8_t history_initialized;    /* 首帧填满五格；退出/失败重新播种 */
    float last_action[RL_ACTION_SIZE]; /* 上一次网络原始请求，保持左右两侧顺序 */
    rl_observation_sample_t sample; /* 最新500Hz反馈，不主动添加观测延迟 */
    uint8_t sample_valid;           /* 最新反馈可用，坏反馈立即清除 */
    uint8_t valid;                  /* 本次32维帧和160维历史构建成功 */
} rl_observation_state_t;
/* 执行任务独占：首次有效RL反馈锁存整周期偏移，退出RL才重新选分支。 */
typedef struct {
    float position_offset[RL_LEG_MOTOR_NUM]; /* 模型坐标整周期偏移，rad；不是零点标定 */
    uint8_t initialized;
} rl_motor_map_state_t;
/* 输入标定连续角/rad；只在初始化时选q_reference分支，运行中不逐拍取模。 */
uint8_t RL_Motor_Map(const machine_rl_cfg_t *config, float position_period, const float position[4], const float velocity[4], const float wheel_velocity[2], uint8_t source_valid, rl_motor_map_state_t *state, rl_motor_state_t *out);
uint8_t RL_Observation_Project_Gravity(const float quat[4], float gravity[3]);
void RL_Observation_Reset(rl_observation_state_t *state);
void RL_Observation_Sample(rl_observation_state_t *state, const float gyro[3], const float quat[4], const rl_motor_state_t *motor, uint8_t source_valid);
uint8_t RL_Observation_Read_Sample(const rl_observation_state_t *state, rl_observation_sample_t *sample);
void RL_Recovery_Reset(rl_recovery_state_t *state);
/* 平地直立目标的腿长→训练机体参考点高度；不估计当前实测机高。 */
float RL_Command_Height(float leg_length_command);
/* 100Hz推进；height为实机几何换算代理值，轮力来自独立AIR估计；非训练接触传感器。 */
void RL_Recovery_Update(rl_recovery_state_t *state, uint8_t request, uint8_t feedback_valid, uint8_t fallen, float height, const float wheel_force[2], float gravity_z);
uint8_t RL_Observation_Build(rl_observation_state_t *state, const machine_rl_cfg_t *config, const rl_observation_sample_t *sample, const rc_command_t *command, const rl_recovery_state_t *recovery, uint8_t source_valid);
void RL_Observation_Set_Last_Action(rl_observation_state_t *state, const float action[RL_ACTION_SIZE]);
#endif
