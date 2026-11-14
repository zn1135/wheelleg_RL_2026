"""Run actual jetson_uart.c with UART DMA/clock/RTOS substitutes; no board access."""
import argparse
import json
from pathlib import Path
import subprocess

STUB=r'''
#pragma once
#include <stdint.h>
#include <string.h>
#define MACHINE_DEFAULT 0
#define MACHINE_ID_BIG_WHEELLEG 0
#define DMA_CACHE_LINE_SIZE 32
#define HAL_OK 0
#define HAL_UART_STATE_READY 0
#define HAL_UART_STATE_BUSY_RX 1
#define DMA_IT_HT 1
#define CTRL_STRATEGY_RL 2
#define RL_RECOVERY_NORMAL 3
#define FAULT_NONE 0
#define FAULT_ACTION 32
#define taskENTER_CRITICAL() ((void)0)
#define taskEXIT_CRITICAL() ((void)0)
#define __HAL_DMA_DISABLE_IT(a,b) ((void)0)
#define __HAL_DMA_GET_COUNTER(a) ((a)->left)
typedef struct { unsigned left; } DMA;
typedef struct { unsigned gState,RxState;DMA *hdmarx; } UART;
extern UART huart10;
extern uint64_t clock_us;
static inline uint64_t Mono_Ns_Get(void) { return clock_us*1000; }
static inline uint32_t HAL_GetTick(void) { return (uint32_t)(clock_us/1000); }
static inline void Dma_Cache_Invalidate_Rx(const void *p,unsigned n) { (void)p;(void)n; }
static inline void Dma_Cache_Clean_Tx(const void *p,unsigned n) { (void)p;(void)n; }
int HAL_UART_Receive_DMA(UART*,uint8_t*,unsigned);
int HAL_UART_Transmit_DMA(UART*,uint8_t*,unsigned);
void Action_State_Clear(void);
typedef struct { float gyro[3],gravity[3],pos[4],vel[6]; } rl_observation_sample_t;
extern struct control_t { struct { unsigned sample_valid;float last_action[6];rl_observation_sample_t sample; } observation; struct { unsigned ready,run_ok,run_us; } policy;struct { unsigned phase,has_stood,tracking_ms; } recovery; } rl_control;
extern struct action_t { float a[6];unsigned reset_count,last_ok_tick,updated,rl_ready; } action_state;
extern struct rc_t { unsigned online;struct { unsigned strategy,engage; } mode;float vel,yaw,len; } rc_command;
extern struct machine_t { struct { unsigned configured; } rl; } machine_value;
#define machine (&machine_value)
extern unsigned ctrl_fault;
extern struct policy_t { float action[6];unsigned sequence,valid,sample_tick_ms,published_tick_ms,run_us; } mirror_policy_snapshot;
extern struct output_t { float target[6],torque[6];unsigned sequence,policy_sequence,flags,valid; } mirror_output_snapshot;
extern struct imu_t { float quat[4]; } imu_state;
extern struct sample_t { uint64_t cached_ns; } latency_sample_snapshot;
extern uint64_t latency_policy_published_ns;
'''
HARNESS=r'''
#include <assert.h>
#include <stdio.h>
#include "stub.h"
#include "jetson_uart.c"
uint64_t clock_us=1000,latency_policy_published_ns;
unsigned ctrl_fault;
struct control_t rl_control;
struct action_t action_state;
struct rc_t rc_command;
struct machine_t machine_value={.rl={.configured=1}};
struct policy_t mirror_policy_snapshot;
struct output_t mirror_output_snapshot;
struct imu_t imu_state;
struct sample_t latency_sample_snapshot;
static DMA dma;
UART huart10={.hdmarx=&dma};
static uint8_t sent[384];static unsigned sent_size,writer;
int HAL_UART_Receive_DMA(UART *u,uint8_t *p,unsigned n) { assert(p==rx_dma&&n==RX_BYTES);u->RxState=HAL_UART_STATE_BUSY_RX;dma.left=n;writer=0;return HAL_OK; }
int HAL_UART_Transmit_DMA(UART *u,uint8_t *p,unsigned n) { assert(u->gState==HAL_UART_STATE_READY);memcpy(sent,p,n);sent_size=n;u->gState=2;return HAL_OK; }
void Action_State_Clear(void) { memset(action_state.a,0,sizeof(action_state.a));action_state.updated=action_state.rl_ready=0;action_state.reset_count++;Jetson_Uart_Action_Clear(action_state.reset_count); }
static void complete(void) { huart10.gState=HAL_UART_STATE_READY; }
static void inject(uint8_t *p,unsigned n) { for(unsigned i=0;i<n;i++) { rx_dma[writer]=p[i];writer=(writer+1)%RX_BYTES; }dma.left=RX_BYTES-writer;Jetson_Uart_Process(); }
static void response(uint8_t *p,uint8_t type,int warm) {
 memset(p,0,68);jrl_header(p,type,type==JRL_ACTION?32:4,123,jrl_u32(sent+16),jrl_u32(sent+20),jrl_u64(sent+24));
 if(type==JRL_HELLO) jrl_p32(p+32,JRL_CONTRACT);else jrl_p32(p+56,(unsigned)warm);
}
int main(void) {
 Jetson_Uart_Init();Action_State_Clear();uint8_t p[68];
 Jetson_Uart_Publish();assert(sent_size==260&&!rl_control.policy.ready);
 response(p,JRL_HELLO,0);unsigned n=jrl_finish(p);inject(p,7);assert(!rl_control.policy.ready);inject(p+7,n-7);assert(rl_control.policy.ready);
 rc_command.online=1;rc_command.mode.strategy=CTRL_STRATEGY_RL;rc_command.mode.engage=1;
 rl_control.recovery.phase=RL_RECOVERY_NORMAL;rl_control.recovery.has_stood=1;rl_control.recovery.tracking_ms=500;rl_control.observation.sample_valid=1;
 complete();clock_us=11000;latency_sample_snapshot.cached_ns=10000000;Jetson_Uart_Publish();assert(jrl_u32(sent+32)&JRL_SELECTED);
 response(p,JRL_ACTION,1);n=jrl_finish(p);clock_us=12000;inject(p,n);assert(action_state.rl_ready&&action_state.updated);
 uint32_t ack=mirror_policy_snapshot.sequence,reset=action_state.reset_count;
 for(unsigned i=0;i<5;i++) { clock_us=13000+2000*i;Jetson_Uart_Pd(ack,reset); }
 assert(link.warmup==5&&link.latency_us==3000);
 uint8_t held[384];memcpy(held,tx_dma,sizeof(held));Jetson_Uart_Publish();assert(memcmp(held,tx_dma,sizeof(held))==0&&link.tx_drops==1);
 complete();clock_us=22000;latency_sample_snapshot.cached_ns=21000000;Jetson_Uart_Publish();response(p,JRL_ACTION,0);jrl_pf32(p+32,.25f);n=jrl_finish(p);clock_us=23000;inject(p,n);assert(action_state.a[0]==.25f);
 unsigned last=action_state.last_ok_tick;clock_us=24000;inject(p,n);assert(action_state.last_ok_tick==last);
 Action_State_Clear();clock_us=25000;inject(p,n);assert(!action_state.rl_ready);
 complete();Jetson_Uart_Publish();response(p,JRL_ACTION,1);n=jrl_finish(p);clock_us=40000;inject(p,n);assert(!action_state.rl_ready);
 clock_us=123000;Jetson_Uart_Process();assert(Jetson_Uart_Blocked()&&!rl_control.policy.ready&&!action_state.rl_ready);
 complete();Jetson_Uart_Publish();response(p,JRL_HELLO,0);inject(p,jrl_finish(p));assert(!rl_control.policy.ready);
 rc_command.mode.engage=0;Jetson_Uart_Process();complete();Jetson_Uart_Publish();response(p,JRL_HELLO,0);inject(p,jrl_finish(p));assert(rl_control.policy.ready);
 huart10.RxState=0;Jetson_Uart_Process();assert(!rl_control.policy.ready&&Jetson_Uart_Blocked()&&!action_state.rl_ready);
 puts("actual firmware UART DMA/handshake/acceptance/reset/PD/timeout/rearm checks passed");return 0;
}
'''
def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--cc',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 (a.out/'stub.h').write_text(STUB)
 for name in ['robot_control.h','robot_state.h','usart.h','dma_cache.h','mono_ns.h']:(a.out/name).write_text('#include "stub.h"\n')
 (a.out/'harness.c').write_text(HARNESS)
 cmd=[a.cc,'-std=c11','-O1','-Wall','-Wextra','-I'+str(a.out),'-Ideployment/H7_RL_ros/imcalib/drivers','-Ideployment/H7_RL_ros/imcalib/common',str(a.out/'harness.c'),'-lm','-o',str(a.out/'check')]
 r=subprocess.run(cmd,capture_output=True,text=True);(a.out/'build.log').write_text(r.stdout+r.stderr);r.check_returncode()
 r=subprocess.run([str(a.out/'check')],capture_output=True,text=True);(a.out/'run.log').write_text(r.stdout+r.stderr);r.check_returncode()
 report=dict(ok=True,scope='actual jetson_uart.c and session/protocol, HAL/clock/RTOS substitutes; no board timing validation')
 (a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(r.stdout)
if __name__=='__main__': main()
