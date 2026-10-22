#ifndef WHEEL_BENCH_H
#define WHEEL_BENCH_H

#include "rc_types.h"
#include "torque_output.h"

/* 轮端命令刻度，非实测轴矩；每次手动触发锁存300ms，运行中不提前撤销。 */
#define WHEEL_BENCH_TORQUE_NM 3.0f
#define WHEEL_BENCH_PULSE_MS 300u
#define WHEEL_BENCH_ARM_MS 300u
#define WHEEL_BENCH_SPEED_MAX 30.0f /* 仅触发前检查；运行中无轮速撤销。 */
#define WHEEL_BENCH_SPEED_IDLE 0.2f
#define WHEEL_BENCH_MAX_GAP_MS 5u

typedef enum {
    WHEEL_BENCH_IDLE = 0,
    WHEEL_BENCH_WAIT,
    WHEEL_BENCH_READY,
    WHEEL_BENCH_RUN,
    WHEEL_BENCH_DONE,
    WHEEL_BENCH_ABORTED
} wheel_bench_phase_t;

typedef enum {
    WHEEL_BENCH_OK = 0,
    WHEEL_BENCH_GATED,
    WHEEL_BENCH_SPEED,
    WHEEL_BENCH_SWITCH,
    WHEEL_BENCH_TIMING,
    WHEEL_BENCH_SEND
} wheel_bench_reason_t;

typedef struct {
    wheel_bench_phase_t phase;
    wheel_bench_reason_t reason;
    uint32_t start_ms;
    uint32_t last_ms;
    uint32_t dwell_ms;
    uint32_t count;
    uint8_t reserved; /* 准备后独占输出，跨中间档也不进入腿补偿或RL。 */
    uint8_t dwell;
    uint8_t wheel;
    float command_nm;
} wheel_bench_state_t;

/* 状态仅由执行任务写；通信任务只读reserved以保持四髋失能。 */
extern volatile wheel_bench_state_t wheel_bench;
void Wheel_Bench_Update(const rc_command_t *rc, uint8_t permit, const float speed[2], uint32_t now_ms, torque_output_t *torque);
void Wheel_Bench_Abort(wheel_bench_reason_t reason);
uint8_t Wheel_Bench_Output_Valid(uint32_t now_ms);

#endif
