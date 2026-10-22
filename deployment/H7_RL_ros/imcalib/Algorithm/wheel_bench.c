/* 单轮短脉冲测试：只计算请求，不发送CAN，不改变驱动刻度、极性或零点。 */
#include "wheel_bench.h"
#include <math.h>

volatile wheel_bench_state_t wheel_bench;

void Wheel_Bench_Abort(wheel_bench_reason_t reason)
{
    wheel_bench.phase = WHEEL_BENCH_ABORTED;
    wheel_bench.reason = reason;
    wheel_bench.command_nm = 0.0f;
    wheel_bench.dwell = 0u;
}

uint8_t Wheel_Bench_Output_Valid(uint32_t now_ms)
{
    return (uint8_t)(wheel_bench.reserved && wheel_bench.phase == WHEEL_BENCH_RUN && now_ms - wheel_bench.start_ms < WHEEL_BENCH_PULSE_MS);
}

void Wheel_Bench_Update(const rc_command_t *rc, uint8_t permit, const float speed[2], uint32_t now_ms, torque_output_t *torque)
{
    uint8_t prepare;
    uint8_t release;
    uint8_t selected;
    uint8_t stationary;
    uint32_t gap;

    Torque_Output_Clear(torque);
    prepare = (uint8_t)(rc->online && rc->s1 == DR16_SW_DOWN && rc->s2 == DR16_SW_UP);
    release = (uint8_t)(rc->online && rc->s1 == DR16_SW_DOWN && rc->s2 == DR16_SW_DOWN);
    selected = (uint8_t)(rc->online && rc->mode.strategy == CTRL_STRATEGY_WHEEL_BENCH && rc->mode.engage);
    stationary = (uint8_t)(isfinite(speed[0]) && isfinite(speed[1]) && fabsf(speed[0]) < WHEEL_BENCH_SPEED_IDLE && fabsf(speed[1]) < WHEEL_BENCH_SPEED_IDLE);
    gap = now_ms - wheel_bench.last_ms;
    wheel_bench.last_ms = now_ms;

    if (!wheel_bench.reserved)
    {
        if (!prepare && !selected)
        {
            return;
        }
        wheel_bench.reserved = 1u;
        wheel_bench.phase = WHEEL_BENCH_WAIT;
    }
    /* 作者要求：触发后锁存本次轮子/指令，只由300ms绝对期限结束；档位/反馈变化不撤销。 */
    if (wheel_bench.phase == WHEEL_BENCH_RUN)
    {
        if (now_ms - wheel_bench.start_ms >= WHEEL_BENCH_PULSE_MS)
        {
            wheel_bench.phase = WHEEL_BENCH_DONE;
            wheel_bench.command_nm = 0.0f;
        }
        else
        {
            torque->dji[wheel_bench.wheel] = wheel_bench.command_nm;
            torque->valid = 1u;
        }
        return;
    }
    /* 下/上持续准备；下/下持续退出。短暂经过中间档不释放独占许可。 */
    if (release)
    {
        wheel_bench.phase = WHEEL_BENCH_WAIT;
        wheel_bench.command_nm = 0.0f;
    }
    if ((prepare && permit && stationary) || release)
    {
        if (!wheel_bench.dwell || gap > WHEEL_BENCH_MAX_GAP_MS || wheel_bench.dwell != (release ? 2u : 1u))
        {
            wheel_bench.dwell = release ? 2u : 1u;
            wheel_bench.dwell_ms = now_ms;
        }
        if (now_ms - wheel_bench.dwell_ms >= WHEEL_BENCH_ARM_MS)
        {
            wheel_bench.reason = WHEEL_BENCH_OK;
            wheel_bench.phase = release ? WHEEL_BENCH_IDLE : WHEEL_BENCH_READY;
            if (release)
            {
                wheel_bench.reserved = 0u;
                wheel_bench.dwell = 0u;
            }
        }
    }
    else
    {
        wheel_bench.dwell = 0u;
        if (wheel_bench.phase == WHEEL_BENCH_READY)
        {
            if (gap > WHEEL_BENCH_MAX_GAP_MS)
            {
                Wheel_Bench_Abort(WHEEL_BENCH_TIMING);
            }
            else if (!permit || !rc->online)
            {
                Wheel_Bench_Abort(WHEEL_BENCH_GATED);
            }
            else if (!stationary)
            {
                Wheel_Bench_Abort(WHEEL_BENCH_SPEED);
            }
            else if (selected)
            {
                /* L+、L-、R+、R-；每次重新准备只执行下一项，不自动连发。 */
                wheel_bench.wheel = (uint8_t)((wheel_bench.count / 2u) % 2u);
                wheel_bench.command_nm = (wheel_bench.count % 2u) ? -WHEEL_BENCH_TORQUE_NM : WHEEL_BENCH_TORQUE_NM;
                wheel_bench.count++;
                wheel_bench.start_ms = now_ms;
                wheel_bench.phase = WHEEL_BENCH_RUN;
            }
        }
    }
    if (Wheel_Bench_Output_Valid(now_ms))
    {
        torque->dji[wheel_bench.wheel] = wheel_bench.command_nm;
        torque->valid = 1u;
    }
}
