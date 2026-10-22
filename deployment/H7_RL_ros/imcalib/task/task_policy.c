/*
 * 50Hz策略：复制500Hz最新反馈及原始动作历史→组装obs39→推理→发布原始action6。
 * 同帧复制公共遥控与恢复标志；趴地启动保持由执行任务控制，不以SDK就绪替代模型激活。
 * 共享样本/动作只在短临界区读写，观测构建和网络调用均放在临界区之外。
 */
#include "robot_control.h"
#include "robot_state.h"
#include "rl_observation.h"
#include "rl_policy.h"
#include "mono_ns.h"
#include <string.h>
/* 策略任务启动时创建本机网络；configured是机型启用开关，不扫描固定配置。 */
void ctrl_task_init(void)
{
    if (machine->rl.configured) { (void)RL_Policy_Init(&rl_control.policy); }
}
/* 所有退出路径都保留尝试结果；失败/跨模式丢弃不是成功动作发布。 */
static void Policy_Timing_End(latency_record_t *record, uint8_t result)
{
    record->flags = result;
    if (record->end_ns == 0u) { record->end_ns = Mono_Ns_Get(); }
    Latency_Capture_Push(record);
}

/* 本地构造后完整发布；清零期间仍在途的推理不能覆盖新控制段的动作。 */
void ctrl_task_body(void)
{
    latency_record_t timing = {0};
    latency_tick_t trigger;
    uint64_t build_started_ns, read_offset_ns;
    rl_observation_state_t observation;
    rl_observation_sample_t sample;
    rl_recovery_state_t recovery;
    rc_command_t command;
    float action[RL_ACTION_SIZE] = {0.0f};
    uint32_t reset_count;
    uint32_t sample_tick_ms;
    uint8_t selected;
    uint8_t sampled;
    uint8_t built;
    timing.kind = LATENCY_POLICY;
    timing.value[3] = UINT32_MAX; /* 未执行/失败SDK不使用陈旧run_us。 */
    timing.begin_ns = Mono_Ns_Get();
    trigger = Latency_Tick_Read(1u);
    timing.value[7] = trigger.sequence;
    if (!machine->rl.configured) { Policy_Timing_End(&timing, 0u); return; }
    /* 样本缓存由高优先级执行任务推进，先在临界区复制，避免读取半更新帧。 */
    taskENTER_CRITICAL();
    selected = (uint8_t)(rc_command.online && rc_command.mode.strategy == CTRL_STRATEGY_RL);
    reset_count = action_state.reset_count;
    timing.reset_count = reset_count;
    timing.value[4] = mirror_policy_snapshot.sequence;
    timing.value[5] = (uint32_t)rc_command.mode.strategy;
    timing.value[6] = (uint32_t)rl_control.recovery.phase;
    if (selected)
    {
        recovery = rl_control.recovery;
        command = rc_command;
        sampled = RL_Observation_Read_Sample(&rl_control.observation, &sample);
        sample_tick_ms = rl_observation_sample_tick_ms;
        timing.value[0] = latency_sample_snapshot.sequence;
        read_offset_ns = Mono_Ns_Get() - timing.begin_ns;
        timing.value[1] = read_offset_ns >= UINT32_MAX ? UINT32_MAX : (uint32_t)read_offset_ns;
        memcpy(observation.last_action, rl_control.observation.last_action, sizeof(observation.last_action));
    }
    else
    {
        sampled = 0u;
        RL_Observation_Reset(&rl_control.observation);
    }
    taskEXIT_CRITICAL();
    if (!selected)
    {
        Action_State_Clear();
        Policy_Timing_End(&timing, 0u);
        return;
    }
    build_started_ns = Mono_Ns_Get();
    built = RL_Observation_Build(&observation, &machine->rl, &sample, &command, &recovery, sampled);
    timing.value[2] = (uint32_t)((Mono_Ns_Get() - build_started_ns) / 1000u);
    taskENTER_CRITICAL();
    if (reset_count != action_state.reset_count)
    {
        taskEXIT_CRITICAL();
        Policy_Timing_End(&timing, 3u);
        return;
    }
    memcpy(rl_control.observation.obs, observation.obs, sizeof(observation.obs));
    rl_control.observation.valid = built;
    taskEXIT_CRITICAL();
    if (!built || !RL_Policy_Run(&rl_control.policy, &observation, action))
    {
        taskENTER_CRITICAL();
        if (reset_count == action_state.reset_count)
        {
            memset(rl_control.observation.last_action, 0, sizeof(rl_control.observation.last_action));
            Action_State_Clear();
        }
        taskEXIT_CRITICAL();
        Policy_Timing_End(&timing, built ? 2u : 1u);
        return;
    }
    timing.value[3] = rl_control.policy.run_us;
    /* 请求动作不反号、不全局裁剪；观测裁剪与目标裁剪由各自生成端完成。
     * 推理期间可能退出RL，发布前重查模式，防止新动作覆盖退出时的清零。 */
    taskENTER_CRITICAL();
    if (reset_count == action_state.reset_count && rc_command.online && rc_command.mode.strategy == CTRL_STRATEGY_RL)
    {
        RL_Observation_Set_Last_Action(&rl_control.observation, action);
        memcpy(action_state.a, action, sizeof(action_state.a));
        action_state.last_ok_tick = HAL_GetTick();
        action_state.updated = 1u;
        action_state.rl_ready = 1u;
        /* 本地obs与本次实际发布action一同保存，不能混入下一次推理的共享obs。 */
        memcpy(mirror_policy_snapshot.obs, observation.obs, sizeof(mirror_policy_snapshot.obs));
        memcpy(mirror_policy_snapshot.action, action, sizeof(mirror_policy_snapshot.action));
        mirror_policy_snapshot.command[0] = command.vel;
        mirror_policy_snapshot.command[1] = command.yaw;
        mirror_policy_snapshot.command[2] = command.len;
        mirror_policy_snapshot.sequence++;
        mirror_policy_snapshot.sample_tick_ms = sample_tick_ms;
        mirror_policy_snapshot.published_tick_ms = (uint32_t)(Mono_Ns_Get() / 1000000u);
        mirror_policy_snapshot.run_us = rl_control.policy.run_us;
        mirror_policy_snapshot.valid = 1u;
        latency_policy_published_ns = Mono_Ns_Get();
        timing.correlation = mirror_policy_snapshot.sequence;
        timing.flags = 4u;
        timing.end_ns = latency_policy_published_ns;
    }
    /* 不匹配时直接丢弃；不能用旧推理的失败路径清掉新控制段。 */
    taskEXIT_CRITICAL();
    Policy_Timing_End(&timing, timing.flags == 4u ? 4u : 3u);
}
