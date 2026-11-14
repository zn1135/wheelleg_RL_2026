/* UART10的字节收发和动作发布。USB采集、驱动、物理标定与PD留在原链路。 */
#include "jetson_uart.h"
#include "jetson_session.h"
#include "robot_control.h"
#include "robot_state.h"
#include "usart.h"
#include "dma_cache.h"
#include "mono_ns.h"
#include <string.h>
#if MACHINE_DEFAULT == MACHINE_ID_BIG_WHEELLEG
#define RX_BYTES 1024u
static uint8_t rx_dma[RX_BYTES] __attribute__((aligned(DMA_CACHE_LINE_SIZE)));
static uint8_t tx_dma[JRL_MAX_FRAME] __attribute__((aligned(DMA_CACHE_LINE_SIZE)));
static jrl_parser_t parser;
static jrl_session_t link;
static uint16_t read_pos;
static uint8_t was_selected;
static uint8_t requested(void) { return rc_command.online && rc_command.mode.strategy==CTRL_STRATEGY_RL && rc_command.mode.engage; }
static uint8_t permitted(void) {
    return machine->rl.configured && requested() && ctrl_fault==FAULT_NONE &&
        rl_control.recovery.phase==RL_RECOVERY_NORMAL && rl_control.recovery.has_stood &&
        rl_control.observation.sample_valid && !link.blocked;
}
void Jetson_Uart_Action_Clear(uint32_t reset) { jrl_clear_action(&link,reset); }
uint8_t Jetson_Uart_Blocked(void) { return link.blocked; }
void Jetson_Uart_Pd(uint32_t sequence,uint32_t reset) {
    taskENTER_CRITICAL();
    if(sequence==link.ack && reset==link.reset) jrl_pd(&link,Mono_Ns_Get()/1000u);
    taskEXIT_CRITICAL();
}
void Jetson_Uart_Init(void) {
    read_pos=0;parser.used=0;
    Dma_Cache_Invalidate_Rx(rx_dma,sizeof(rx_dma));
    if(HAL_UART_Receive_DMA(&huart10,rx_dma,sizeof(rx_dma))==HAL_OK) __HAL_DMA_DISABLE_IT(huart10.hdmarx,DMA_IT_HT);
}
static void receive_packet(const uint8_t *p) {
    taskENTER_CRITICAL();
    int result=jrl_receive(&link,p,Mono_Ns_Get()/1000u,requested(),permitted());
    if(result==1) rl_control.policy.ready=1;
    if(result==3) { rl_control.policy.ready=0;Action_State_Clear(); }
    if(result==2) {
        for(unsigned i=0;i<6;i++) action_state.a[i]=jrl_f32(p+32+4*i);
        memcpy(rl_control.observation.last_action,action_state.a,sizeof(action_state.a));
        action_state.last_ok_tick=HAL_GetTick();action_state.updated=1;action_state.rl_ready=1;
        rl_control.policy.run_ok++;rl_control.policy.run_us=jrl_u32(p+60);
        /* USB保留动作/执行诊断；obs槽没有本地构造，valid=0，不能伪装成有效观测。 */
        memcpy(mirror_policy_snapshot.action,action_state.a,sizeof(action_state.a));
        mirror_policy_snapshot.sequence=link.ack;mirror_policy_snapshot.valid=0;
        mirror_policy_snapshot.sample_tick_ms=(uint32_t)(link.source_us/1000u);
        mirror_policy_snapshot.published_tick_ms=(uint32_t)(Mono_Ns_Get()/1000000u);
        mirror_policy_snapshot.run_us=rl_control.policy.run_us;
        latency_policy_published_ns=Mono_Ns_Get();
    }
    taskEXIT_CRITICAL();
}
void Jetson_Uart_Process(void) {
    taskENTER_CRITICAL();
    if(requested() && (ctrl_fault & FAULT_ACTION) && link.session) { jrl_disconnect(&link);rl_control.policy.ready=0;Action_State_Clear(); }
    if(jrl_expire(&link,Mono_Ns_Get()/1000u,requested())) { rl_control.policy.ready=0;Action_State_Clear(); }
    taskEXIT_CRITICAL();
    if(huart10.RxState!=HAL_UART_STATE_BUSY_RX) {
        taskENTER_CRITICAL();jrl_disconnect(&link);rl_control.policy.ready=0;Action_State_Clear();taskEXIT_CRITICAL();
        Jetson_Uart_Init();return;
    }
    uint16_t write_pos=(uint16_t)((RX_BYTES-__HAL_DMA_GET_COUNTER(huart10.hdmarx))%RX_BYTES);
    Dma_Cache_Invalidate_Rx(rx_dma,sizeof(rx_dma));
    while(read_pos!=write_pos) {
        unsigned n=jrl_feed(&parser,rx_dma[read_pos]);read_pos=(uint16_t)((read_pos+1u)%RX_BYTES);
        if(n) receive_packet(parser.data);
    }
}
void Jetson_Uart_Publish(void) {
    uint8_t frame[JRL_HEADER+JRL_STATE_BYTES+4]={0};
    uint8_t *p=frame+JRL_HEADER;
    uint64_t now=Mono_Ns_Get()/1000u;
    taskENTER_CRITICAL();
    uint8_t selected=permitted();
    if(!selected && was_selected) Action_State_Clear();
    was_selected=selected;
    /* TX忙时不发布新的可响应序号；DMA持有的缓冲绝不覆写。 */
    if(huart10.gState!=HAL_UART_STATE_READY) { link.tx_drops++;taskEXIT_CRITICAL();return; }
    link.reset=action_state.reset_count;link.sequence++;link.published_us=now;
    link.source_us=selected ? latency_sample_snapshot.cached_ns/1000u : now;
    uint64_t cached_us=latency_sample_snapshot.cached_ns/1000u;
    uint8_t fresh=(uint8_t)(rl_control.observation.sample_valid && now>=cached_us && now-cached_us<10000u);
    uint32_t flags=(fresh ? JRL_VALID:0u)|(selected ? JRL_SELECTED:0u)|
        (link.session ? JRL_CONNECTED:0u)|(link.blocked ? JRL_BLOCKED:0u)|
        (rl_control.recovery.tracking_ms>=500u ? JRL_TRACKING:0u);
    jrl_header(frame,JRL_STATE,JRL_STATE_BYTES,link.session,link.reset,link.sequence,link.source_us);
    jrl_p32(p,flags);jrl_p32(p+4,ctrl_fault);jrl_p32(p+8,link.ack);jrl_p32(p+12,link.warmup);
    jrl_p32(p+16,link.applied);jrl_p32(p+20,link.latency_us);jrl_p32(p+24,JRL_CONTRACT);jrl_p32(p+28,link.rejected+parser.errors);
    const rl_observation_sample_t *s=&rl_control.observation.sample;
    for(unsigned i=0;i<3;i++) { jrl_pf32(p+32+4*i,s->gyro[i]);jrl_pf32(p+44+4*i,s->gravity[i]); }
    for(unsigned i=0;i<4;i++) { jrl_pf32(p+56+4*i,s->pos[i]);jrl_pf32(p+180+4*i,imu_state.quat[i]); }
    for(unsigned i=0;i<6;i++) {
        jrl_pf32(p+72+4*i,s->vel[i]);jrl_pf32(p+108+4*i,action_state.a[i]);
        jrl_pf32(p+132+4*i,mirror_output_snapshot.target[i]);jrl_pf32(p+156+4*i,mirror_output_snapshot.torque[i]);
    }
    jrl_pf32(p+96,rc_command.vel);jrl_pf32(p+100,rc_command.yaw);jrl_pf32(p+104,rc_command.len);
    jrl_p64(p+196,link.applied_us);jrl_p32(p+204,link.tx_drops);
    jrl_p32(p+208,mirror_output_snapshot.sequence);jrl_p32(p+212,mirror_output_snapshot.policy_sequence);
    jrl_p32(p+216,mirror_output_snapshot.flags);jrl_p32(p+220,mirror_output_snapshot.valid);
    taskEXIT_CRITICAL();
    unsigned bytes=jrl_finish(frame);memcpy(tx_dma,frame,bytes);Dma_Cache_Clean_Tx(tx_dma,bytes);
    if(HAL_UART_Transmit_DMA(&huart10,tx_dma,(uint16_t)bytes)!=HAL_OK) { taskENTER_CRITICAL();link.tx_drops++;taskEXIT_CRITICAL(); }
}

#else
/* 小机未配置RL，不能通过外部链路启用；UART10仅存在于大机。 */
void Jetson_Uart_Init(void) {}
void Jetson_Uart_Process(void) {}
void Jetson_Uart_Publish(void) {}
void Jetson_Uart_Action_Clear(uint32_t reset) { (void)reset; }
void Jetson_Uart_Pd(uint32_t sequence,uint32_t reset) { (void)sequence;(void)reset; }
uint8_t Jetson_Uart_Blocked(void) { return 0u; }
#endif
