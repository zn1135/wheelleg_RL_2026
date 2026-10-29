/*
 * 文件职责：跨任务共享状态的声明集合（robot_state.h）。
 * 本头文件定义共享数据类型、常量或内联辅助，由引用方读取/调用，不单独创建任务或维护外设。
 * 输入/来源：robot_control.c中定义的IMU/电机/腿/RL/LQR和许可状态。
 * 输出/去向：给各任务提供统一extern声明、原始动作/机器人/RL运行状态类型。
 * 关键边界：该头包含共享对象，算法层应接收参数而不是包含这个任务头；同帧读写使用快照接口。
 * 阅读入口：md/file-guide.md；参数当前值以源码为准。
 */

#ifndef ROBOT_STATE_H
#define ROBOT_STATE_H
#include "latency_capture.h"

#include "robot_control.h"
#include "imu_state.h"
#include "feedback_snapshot.h"
#include "leg_solver.h"
#include "rl_torque.h"
#include "lqr_balance.h"
#include "leg_balance.h"







typedef struct {
    float a[RL_ACTION_SIZE];    /* 原始网络顺序[LF前,LF后,L轮,RF前,RF后,R轮]，未缩放 */
    uint32_t reset_count;       /* 动作清零次数：拒绝清零前已在途、清零后才返回的推理 */
    uint32_t last_ok_tick;      /* 最近一次有效动作发布时间，ms */
    uint8_t updated;            /* 动作已发布，看门狗可开始检查更新时间 */
    uint8_t rl_ready;               /* 推理动作可用 */
} action_state_t;

typedef struct {
    int8_t dm_front;
    int8_t dm_rear;
} leg_map_t;

typedef struct {
    uint8_t rc_enable;
    uint8_t motor_enabled;
    uint8_t fallen; /* 公共姿态倒地：IMU俯仰或横滚角任一超过40度，不包含高度。 */
    float height; /* 公共髋轮投影间距m；通信发布，LQR当前不消费。 */
    uint8_t height_valid;
} robot_state_t;

/* 最新采样/PD归执行任务，观测/原始action历史归策略任务；参数直接读机器表。 */
typedef struct {
    rl_observation_state_t observation;
    rl_policy_t policy;
    rl_torque_state_t torque_state;
    rl_recovery_state_t recovery; /* 执行任务推进，策略只复制；阶段供Watch观察。 */
} rl_control_state_t;

/* 只读遥测：成功发布的同一次obs/action配对，不参与控制或推进算法。 */
typedef struct {
    float obs[RL_OBS_SIZE];
    float action[RL_ACTION_SIZE]; /* 网络左右两侧顺序，未缩放。 */
    float command[3]; /* 本次帧的实际vx/yaw/机高，单位m/s、rad/s、m。 */
    uint32_t sequence;
    uint32_t sample_tick_ms; /* Mono_Ns_Get毫秒：500Hz缓存时刻，不是传感器采样时刻。 */
    uint32_t published_tick_ms; /* 与镜像captured_ns同原点，不使用HAL tick。 */
    uint32_t run_us;
    uint8_t valid;
} mirror_policy_snapshot_t;

/* 一个输出拍的PD输入/目标及最终门控命令；HAL成功只表示发送入队。 */
typedef struct {
    float target[RL_ACTION_SIZE]; /* 四腿角度rad、双轮速度rad/s。 */
    float action[RL_ACTION_SIZE]; /* 本拍实际使用的网络顺序action。 */
    float pos[RL_LEG_MOTOR_NUM]; /* 本拍PD模型坐标rad。 */
    float vel[RL_ACTION_SIZE]; /* 本拍PD模型坐标rad/s。 */
    float torque[RL_ACTION_SIZE]; /* 门控后驱动入口Nm，固件四腿两轮顺序。 */
    uint32_t sequence;
    uint32_t tick_ms; /* Mono_Ns_Get毫秒，驱动下发前时刻，不是电机生效时刻。 */
    uint32_t policy_sequence;
    uint8_t flags; /* bit0=PD有效，1=零输出，2/3=DM/DJI入队，4=总开关，5=drive。 */
    uint8_t valid;
} mirror_output_snapshot_t;

/* 写入归对应任务 */
extern imu_state_t imu_state;
extern motor_state_t motor_state;
extern leg_state_t leg_l;
extern leg_state_t leg_r;
extern action_state_t action_state;
extern rc_command_t rc_command;
extern leg_map_t leg_map_l;
extern leg_map_t leg_map_r;
extern robot_state_t robot_state;
extern rl_control_state_t rl_control;
extern mirror_policy_snapshot_t mirror_policy_snapshot;
extern mirror_output_snapshot_t mirror_output_snapshot;
extern uint32_t rl_observation_sample_tick_ms;
/* 仅用于时序配对；与对应样本/动作在原临界区一同发布。 */
extern latency_sample_t latency_sample_snapshot;
extern uint64_t latency_policy_published_ns;
extern lqr_state_t lqr_state;
extern leg_balance_t leg_balance;
extern volatile ctrl_strategy_t ctrl_strategy;
extern volatile uint32_t ctrl_fault;
extern volatile uint8_t output_debug_dm_sent;
extern volatile uint8_t output_debug_dji_sent;
extern volatile control_time_debug_t control_time_debug;
extern volatile float rl_output_dm_cmd_nm[DM_MOTOR_NUM];
extern volatile float rl_output_wheel_cmd_nm[DJI_MOTOR_NUM];
extern uint8_t torque_output_enabled;
extern volatile compensation_debug_t compensation_debug;

#endif
